import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import cvxpy as cp

class DSBLOAlgorithm:
    """
    Doubly Stochastic Bilevel Optimization Algorithm (DS-BLO)
    Based on the paper: "A Doubly Stochastically Perturbed Algorithm for Linearly Constrained Bilevel Optimization"
    
    Implements the stochastic perturbation approach with exact KKT-based gradient computation
    """
    
    def __init__(self, dim: int = 5, num_constraints: int = 3, noise_std: float = 0.01, 
                 device: str = 'cpu', seed: int = 42):
        self.dim = dim
        self.num_constraints = num_constraints
        self.noise_std = noise_std
        self.device = device
        self.dtype = torch.float64
        torch.set_default_dtype(self.dtype)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize problem parameters (strongly convex)
        self._setup_problem_parameters()
        
        # Algorithm parameters
        self.epsilon = 1e-3  # Convergence tolerance
        self.delta_bar = 1e-2  # Goldstein stationarity parameter
        self.beta = 0.9  # Momentum parameter
        self.max_iterations = 1000
        
        # Perturbation distribution parameters
        self.perturbation_scale = 0.01
        
        print(f"DS-BLO Algorithm initialized (dim={dim}, constraints={num_constraints})")
        print(f"Target: (ε={self.epsilon}, δ={self.delta_bar})-Goldstein stationarity")
    
    def _setup_problem_parameters(self):
        """Setup strongly convex bilevel problem parameters"""
        # Upper level parameters with strong convexity
        noise_scale = 0.01 / np.sqrt(self.dim)
        self.Q_upper = (torch.randn(self.dim, self.dim, device=self.device, dtype=self.dtype) * noise_scale)
        self.Q_upper = self.Q_upper + self.Q_upper.T  # Symmetric
        self.Q_upper += torch.eye(self.dim, device=self.device, dtype=self.dtype) * 2.0  # Strong convexity
        
        # Lower level parameters with strong convexity
        self.Q_lower = (torch.randn(self.dim, self.dim, device=self.device, dtype=self.dtype) * noise_scale)
        self.Q_lower = self.Q_lower + self.Q_lower.T  # Symmetric
        self.Q_lower += torch.eye(self.dim, device=self.device, dtype=self.dtype) * 2.0  # Strong convexity
        
        # Other parameters
        param_scale = 0.1 / np.sqrt(self.dim)
        self.c_upper = torch.randn(self.dim, device=self.device, dtype=self.dtype) * param_scale
        self.c_lower = torch.randn(self.dim, device=self.device, dtype=self.dtype) * param_scale
        self.P = torch.randn(self.dim, self.dim, device=self.device, dtype=self.dtype) * param_scale
        self.x_target = torch.randn(self.dim, device=self.device, dtype=self.dtype) * 0.1
        
        # Constraint matrices
        constraint_scale = 0.05 / np.sqrt(self.dim)
        self.A = torch.randn(self.num_constraints, self.dim, device=self.device, dtype=self.dtype) * constraint_scale
        self.B = torch.randn(self.num_constraints, self.dim, device=self.device, dtype=self.dtype) * constraint_scale
        self.b = torch.abs(torch.randn(self.num_constraints, device=self.device, dtype=self.dtype)) * 0.2 + 0.1
        
        # Verify strong convexity
        upper_eigenvals = torch.linalg.eigvals(self.Q_upper).real
        lower_eigenvals = torch.linalg.eigvals(self.Q_lower).real
        
        print(f"Upper level strong convexity: λ_min={upper_eigenvals.min():.3f}, λ_max={upper_eigenvals.max():.3f}")
        print(f"Lower level strong convexity: λ_min={lower_eigenvals.min():.3f}, λ_max={lower_eigenvals.max():.3f}")
    
    def upper_objective(self, x: torch.Tensor, y: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """Upper level objective with optional noise"""
        term1 = 0.5 * (x - self.x_target) @ self.Q_upper @ (x - self.x_target)
        term2 = self.c_upper @ y
        
        if add_noise:
            noise = torch.randn_like(x) * self.noise_std
            return term1 + term2 + noise.sum()
        return term1 + term2
    
    def lower_objective(self, x: torch.Tensor, y: torch.Tensor, q: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """Lower level objective with perturbation q"""
        term1 = 0.5 * y @ self.Q_lower @ y
        term2 = (self.c_lower + self.P.T @ x + q) @ y
        
        if add_noise:
            noise = torch.randn_like(y) * self.noise_std
            return term1 + term2 + noise.sum()
        return term1 + term2
    
    def solve_lower_level_exact(self, x: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Solve the perturbed lower-level problem exactly using CVXPy
        min_y g_q(x,y) = g(x,y) + q^T y
        s.t. Ay + Bx ≤ b
        """
        x_det = x.detach()
        q_det = q.detach()
        
        # Prepare problem data
        Q = (self.Q_lower.detach().cpu().numpy() + self.Q_lower.detach().cpu().numpy().T) / 2.0
        d = (self.c_lower + self.P.T @ x_det + q_det).detach().cpu().numpy()
        A_np = self.A.detach().cpu().numpy()
        B_np = self.B.detach().cpu().numpy()
        b_np = self.b.detach().cpu().numpy()
        c_vec = (self.A @ x_det - self.b).detach().cpu().numpy()
        
        # Define and solve QP
        y_var = cp.Variable(self.dim)
        objective = cp.Minimize(0.5 * cp.quad_form(y_var, Q) + d.T @ y_var)
        constraints = [A_np @ y_var >= c_vec]
        problem = cp.Problem(objective, constraints)
        
        problem.solve(solver=cp.SCS, verbose=False)
        
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"CVXPy LL solve failed: status={problem.status}")
        
        y_star = torch.tensor(y_var.value, device=self.device, dtype=self.dtype)
        lambda_dual = torch.tensor(constraints[0].dual_value, device=self.device, dtype=self.dtype)
        
        # Active set detection
        constraint_values = A_np @ y_star.cpu().numpy() - c_vec
        active_mask = np.abs(constraint_values) < 1e-6
        
        info = {
            'status': problem.status,
            'converged': problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE],
            'lambda': lambda_dual.detach(),
            'active_mask': torch.tensor(active_mask, device=self.device),
            'constraint_violations': torch.tensor(np.maximum(-constraint_values, 0), device=self.device)
        }
        
        return y_star, info
    
    def compute_implicit_gradient(self, x: torch.Tensor, y_star: torch.Tensor, q: torch.Tensor, 
                                 active_mask: torch.Tensor, lambda_dual: torch.Tensor) -> torch.Tensor:
        """
        Compute exact implicit gradient using DS-BLO formulas:
        ∇F_q(x) = ∇_x f(x,y_q*(x)) + [∇y_q*(x)]^T ∇_y f(x,y_q*(x))
        
        where:
        ∇y_q*(x) = [∇²_{yy}g(x,y_q*(x))]^{-1}[-∇²_{xy}g(x,y_q*(x)) - Ā^T∇λ̄_q*(x)]
        ∇λ̄_q*(x) = -[Ā[∇²_{yy}g(x,y_q*(x))]^{-1}Ā^T]^{-1}[Ā[∇²_{yy}g(x,y_q*(x))]^{-1}∇²_{xy}g(x,y_q*(x)) - B̄]
        """
        # Get active constraints
        active_indices = torch.where(active_mask)[0]
        if len(active_indices) == 0:
            # No active constraints - unconstrained case
            A_bar = torch.empty(0, self.dim, device=self.device, dtype=self.dtype)
            B_bar = torch.empty(0, self.dim, device=self.device, dtype=self.dtype)
            lambda_bar = torch.empty(0, device=self.device, dtype=self.dtype)
        else:
            A_bar = self.A[active_indices]
            B_bar = self.B[active_indices]
            lambda_bar = lambda_dual[active_indices]
        
        # Compute Hessians
        H_yy = self.Q_lower  # ∇²_{yy}g(x,y) = Q_lower (constant)
        H_xy = self.P.T      # ∇²_{xy}g(x,y) = P^T (constant)
        
        # Compute gradients of upper objective
        grad_x_f = self.Q_upper @ (x - self.x_target)
        grad_y_f = self.c_upper
        
        if len(active_indices) == 0:
            # Unconstrained case
            dy_dx = -torch.linalg.solve(H_yy, H_xy)
        else:
            # Constrained case - use DS-BLO formulas
            # ∇λ̄_q*(x) = -[Ā[∇²_{yy}g]^{-1}Ā^T]^{-1}[Ā[∇²_{yy}g]^{-1}∇²_{xy}g - B̄]
            H_yy_inv = torch.linalg.solve(H_yy, torch.eye(self.dim, device=self.device, dtype=self.dtype))
            A_bar_H_inv = A_bar @ H_yy_inv
            
            # Compute ∇λ̄_q*(x)
            term1 = A_bar_H_inv @ A_bar.T
            term2 = A_bar_H_inv @ H_xy - B_bar
            dlambda_dx = -torch.linalg.solve(term1, term2)
            
            # ∇y_q*(x) = [∇²_{yy}g]^{-1}[-∇²_{xy}g - Ā^T∇λ̄_q*(x)]
            dy_dx = torch.linalg.solve(H_yy, -H_xy - A_bar.T @ dlambda_dx)
        
        # Total implicit gradient
        implicit_grad = grad_x_f + dy_dx.T @ grad_y_f
        
        return implicit_grad
    
    def sample_perturbation(self) -> torch.Tensor:
        """Sample perturbation vector q from continuous distribution"""
        return torch.randn(self.dim, device=self.device, dtype=self.dtype) * self.perturbation_scale
    
    def compute_goldstein_stationarity(self, x: torch.Tensor, gradients: List[torch.Tensor]) -> float:
        """Compute Goldstein stationarity measure"""
        if len(gradients) == 0:
            return float('inf')
        
        # Average gradient in the neighborhood
        avg_grad = torch.stack(gradients).mean(dim=0)
        return float(torch.norm(avg_grad))
    
    def run(self, x_init: Optional[torch.Tensor] = None) -> Dict:
        """
        Run DS-BLO algorithm until convergence
        """
        # Initialize
        if x_init is None:
            x = torch.randn(self.dim, device=self.device, dtype=self.dtype) * 0.1
        else:
            x = x_init.clone()
        
        # Algorithm parameters
        K = max(10, int(1 / np.log(1/self.beta) * np.log(32 * (4 + 2 * 10) / self.epsilon)))
        gamma_1 = K / self.delta_bar
        gamma_2 = 4 * gamma_1 * (4 + 2 * 10)
        
        # Initialize momentum
        q = self.sample_perturbation()
        y_star, ll_info = self.solve_lower_level_exact(x, q)
        m = self.compute_implicit_gradient(x, y_star, q, ll_info['active_mask'], ll_info['lambda'])
        
        # Storage for convergence tracking
        x_history = [x.clone()]
        gap_history = []
        gradient_history = []
        
        print(f"\n🚀 Starting DS-BLO Algorithm")
        print(f"Initial x: {x}")
        print(f"Initial gap: {float('inf')}")
        
        for t in range(self.max_iterations):
            # Update upper-level variable
            eta_t = 1 / (gamma_1 * torch.norm(m) + gamma_2)
            x_new = x - eta_t * m
            
            # Sample intermediate point (second perturbation)
            alpha = torch.rand(1, device=self.device, dtype=self.dtype)
            x_bar = x + alpha * (x_new - x)
            
            # Sample new perturbation
            q_new = self.sample_perturbation()
            
            # Solve lower-level problem at perturbed point
            y_star_new, ll_info_new = self.solve_lower_level_exact(x_bar, q_new)
            
            # Compute gradient at perturbed point
            g_new = self.compute_implicit_gradient(x_bar, y_star_new, q_new, 
                                                 ll_info_new['active_mask'], ll_info_new['lambda'])
            
            # Update momentum
            m = self.beta * m + (1 - self.beta) * g_new
            
            # Update x
            x = x_new
            
            # Store history
            x_history.append(x.clone())
            gradient_history.append(g_new.clone())
            
            # Compute gap (Goldstein stationarity)
            if len(gradient_history) >= K:
                recent_gradients = gradient_history[-K:]
                gap = self.compute_goldstein_stationarity(x, recent_gradients)
                gap_history.append(gap)
                
                # Check convergence
                if gap < self.epsilon:
                    print(f"\n✅ CONVERGED at iteration {t+1}")
                    print(f"Final gap: {gap:.6f} < ε={self.epsilon}")
                    break
                
                if (t + 1) % 50 == 0:
                    print(f"Iteration {t+1}: gap={gap:.6f}, ||x||={torch.norm(x):.6f}")
        
        # Final solution
        q_final = self.sample_perturbation()
        y_final, ll_info_final = self.solve_lower_level_exact(x, q_final)
        final_objective = self.upper_objective(x, y_final, add_noise=False)
        
        results = {
            'x_optimal': x,
            'y_optimal': y_final,
            'final_objective': final_objective,
            'gap_history': gap_history,
            'x_history': x_history,
            'converged': len(gap_history) > 0 and gap_history[-1] < self.epsilon,
            'iterations': len(gap_history),
            'final_gap': gap_history[-1] if gap_history else float('inf')
        }
        
        print(f"\n📊 Final Results:")
        print(f"Final objective: {final_objective:.6f}")
        print(f"Final gap: {results['final_gap']:.6f}")
        print(f"Converged: {results['converged']}")
        print(f"Iterations: {results['iterations']}")
        
        return results

def main():
    """Main function to run DS-BLO algorithm"""
    print("=" * 60)
    print("DS-BLO Algorithm Implementation")
    print("=" * 60)
    
    # Create algorithm instance
    dsblo = DSBLOAlgorithm(dim=5, num_constraints=3, noise_std=0.01)
    
    # Run algorithm
    start_time = time.time()
    results = dsblo.run()
    end_time = time.time()
    
    print(f"\n⏱️  Total runtime: {end_time - start_time:.2f} seconds")
    
    # Plot results
    if results['gap_history']:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.semilogy(results['gap_history'])
        plt.xlabel('Iteration')
        plt.ylabel('Goldstein Stationarity Gap')
        plt.title('DS-BLO Convergence')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        x_history = torch.stack(results['x_history'])
        for i in range(x_history.shape[1]):
            plt.plot(x_history[:, i].cpu().numpy(), label=f'x[{i}]')
        plt.xlabel('Iteration')
        plt.ylabel('Variable Value')
        plt.title('Upper-Level Variables')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('dsblo_convergence.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n📈 Convergence plot saved as 'dsblo_convergence.png'")
    
    return results

if __name__ == "__main__":
    results = main()
