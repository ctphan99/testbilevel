import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DSBLOAlgorithm:
    """
    Doubly Stochastic Bilevel Optimization Algorithm (DS-BLO)
    Based on the paper: "A Doubly Stochastically Perturbed Algorithm for Linearly Constrained Bilevel Optimization"
    
    Implements the stochastic perturbation approach with exact KKT-based gradient computation
    """
    
    def __init__(self, dim: int = 5, num_constraints: int = 3, noise_std: float = 0.01, seed: int = 42):
        self.dim = dim
        self.num_constraints = num_constraints
        self.noise_std = noise_std
        self.dtype = np.float64
        
        np.random.seed(seed)
        
        # Initialize problem parameters (strongly convex)
        self._setup_problem_parameters()
        
        # Algorithm parameters
        self.epsilon = 1e-2  # Convergence tolerance (relaxed)
        self.delta_bar = 1e-2  # Goldstein stationarity parameter
        self.beta = 0.95  # Momentum parameter (increased)
        self.max_iterations = 2000
        
        # Perturbation distribution parameters
        self.perturbation_scale = 0.01
        
        print(f"DS-BLO Algorithm initialized (dim={dim}, constraints={num_constraints})")
        print(f"Target: (ε={self.epsilon}, δ={self.delta_bar})-Goldstein stationarity")
    
    def _setup_problem_parameters(self):
        """Setup strongly convex bilevel problem parameters"""
        # Upper level parameters with strong convexity
        noise_scale = 0.01 / np.sqrt(self.dim)
        Q_upper = np.random.randn(self.dim, self.dim) * noise_scale
        self.Q_upper = Q_upper + Q_upper.T  # Symmetric
        self.Q_upper += np.eye(self.dim) * 2.0  # Strong convexity
        
        # Lower level parameters with strong convexity
        Q_lower = np.random.randn(self.dim, self.dim) * noise_scale
        self.Q_lower = Q_lower + Q_lower.T  # Symmetric
        self.Q_lower += np.eye(self.dim) * 2.0  # Strong convexity
        
        # Other parameters
        param_scale = 0.1 / np.sqrt(self.dim)
        self.c_upper = np.random.randn(self.dim) * param_scale
        self.c_lower = np.random.randn(self.dim) * param_scale
        self.P = np.random.randn(self.dim, self.dim) * param_scale
        self.x_target = np.random.randn(self.dim) * 0.1
        
        # Constraint matrices
        constraint_scale = 0.05 / np.sqrt(self.dim)
        self.A = np.random.randn(self.num_constraints, self.dim) * constraint_scale
        self.B = np.random.randn(self.num_constraints, self.dim) * constraint_scale
        self.b = np.abs(np.random.randn(self.num_constraints)) * 0.2 + 0.1
        
        # Verify strong convexity
        upper_eigenvals = np.linalg.eigvals(self.Q_upper).real
        lower_eigenvals = np.linalg.eigvals(self.Q_lower).real
        
        print(f"Upper level strong convexity: λ_min={upper_eigenvals.min():.3f}, λ_max={upper_eigenvals.max():.3f}")
        print(f"Lower level strong convexity: λ_min={lower_eigenvals.min():.3f}, λ_max={lower_eigenvals.max():.3f}")
    
    def upper_objective(self, x: np.ndarray, y: np.ndarray, add_noise: bool = True) -> float:
        """Upper level objective with optional noise"""
        term1 = 0.5 * (x - self.x_target).T @ self.Q_upper @ (x - self.x_target)
        term2 = self.c_upper @ y
        
        if add_noise:
            noise = np.random.randn(self.dim) * self.noise_std
            return term1 + term2 + noise.sum()
        return term1 + term2
    
    def lower_objective(self, x: np.ndarray, y: np.ndarray, q: np.ndarray, add_noise: bool = True) -> float:
        """Lower level objective with perturbation q"""
        term1 = 0.5 * y.T @ self.Q_lower @ y
        term2 = (self.c_lower + self.P.T @ x + q) @ y
        
        if add_noise:
            noise = np.random.randn(self.dim) * self.noise_std
            return term1 + term2 + noise.sum()
        return term1 + term2
    
    def solve_lower_level_exact(self, x: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Solve the perturbed lower-level problem exactly using analytical solution
        min_y g_q(x,y) = g(x,y) + q^T y
        s.t. Ay + Bx ≤ b
        
        For strongly convex QP with linear constraints, we can solve analytically
        """
        # Prepare problem data
        Q = self.Q_lower
        d = self.c_lower + self.P.T @ x + q
        c_vec = self.A @ x - self.b
        
        # Try unconstrained solution first
        y_unconstrained = -np.linalg.solve(Q, d)
        
        # Check constraint satisfaction
        constraint_values = self.A @ y_unconstrained - c_vec
        violations = np.maximum(-constraint_values, 0)
        
        if np.sum(violations) < 1e-10:
            # Unconstrained solution is feasible
            y_star = y_unconstrained
            lambda_dual = np.zeros(self.num_constraints)
            active_mask = np.zeros(self.num_constraints, dtype=bool)
        else:
            # Need to solve constrained problem
            # Use active set method for small problems
            y_star, lambda_dual, active_mask = self._solve_constrained_qp(Q, d, c_vec)
        
        info = {
            'converged': True,
            'lambda': lambda_dual,
            'active_mask': active_mask,
            'constraint_violations': np.maximum(-constraint_values, 0)
        }
        
        return y_star, info
    
    def _solve_constrained_qp(self, Q: np.ndarray, d: np.ndarray, c_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve constrained QP using active set method"""
        n_vars = Q.shape[0]
        n_constraints = self.A.shape[0]
        
        # Try all possible active sets (2^m possibilities)
        best_y = None
        best_lambda = None
        best_objective = float('inf')
        best_active_mask = None
        
        for active_mask_int in range(2**n_constraints):
            try:
                # Determine which constraints are active
                active_constraints = []
                for i in range(n_constraints):
                    if (active_mask_int >> i) & 1:
                        active_constraints.append(i)
                
                if len(active_constraints) == 0:
                    # No active constraints - unconstrained solution
                    y = -np.linalg.solve(Q, d)
                    lambda_full = np.zeros(n_constraints)
                    active_mask = np.zeros(n_constraints, dtype=bool)
                else:
                    # Active constraints: A_active y = c_active
                    A_active = self.A[active_constraints, :]
                    c_active = c_vec[active_constraints]
                    
                    # Solve KKT system
                    kkt_matrix = np.zeros((n_vars + len(active_constraints), n_vars + len(active_constraints)))
                    kkt_matrix[:n_vars, :n_vars] = Q
                    kkt_matrix[:n_vars, n_vars:] = -A_active.T
                    kkt_matrix[n_vars:, :n_vars] = A_active
                    
                    kkt_rhs = np.zeros(n_vars + len(active_constraints))
                    kkt_rhs[:n_vars] = -d
                    kkt_rhs[n_vars:] = c_active
                    
                    # Solve the system
                    kkt_solution = np.linalg.solve(kkt_matrix, kkt_rhs)
                    y = kkt_solution[:n_vars]
                    lambda_active = kkt_solution[n_vars:]
                    
                    # Reconstruct full lambda vector
                    lambda_full = np.zeros(n_constraints)
                    lambda_full[active_constraints] = lambda_active
                    
                    # Create active mask
                    active_mask = np.zeros(n_constraints, dtype=bool)
                    active_mask[active_constraints] = True
                
                # Check KKT conditions
                # 1. Stationarity: Q y - A^T λ + d = 0
                stationarity_residual = Q @ y - self.A.T @ lambda_full + d
                stationarity_error = np.linalg.norm(stationarity_residual)
                
                # 2. Primal feasibility: A y ≥ c
                constraint_values = self.A @ y - c_vec
                primal_violation = np.linalg.norm(np.maximum(-constraint_values, 0))
                
                # 3. Dual feasibility: λ ≥ 0
                dual_violation = np.linalg.norm(np.maximum(-lambda_full, 0))
                
                # 4. Complementary slackness: λ^T(A y - c) = 0
                complementarity_error = np.abs(lambda_full @ constraint_values)
                
                # Check if this is a valid KKT point
                kkt_error = stationarity_error + primal_violation + dual_violation + complementarity_error
                
                if kkt_error < 1e-8:
                    # Valid KKT point - compute objective
                    objective = 0.5 * y @ Q @ y + d @ y
                    
                    if objective < best_objective:
                        best_objective = objective
                        best_y = y.copy()
                        best_lambda = lambda_full.copy()
                        best_active_mask = active_mask.copy()
            
            except np.linalg.LinAlgError:
                # This active set didn't work, try next one
                continue
        
        if best_y is not None:
            return best_y, best_lambda, best_active_mask
        else:
            # Fallback to unconstrained solution
            y_unconstrained = -np.linalg.solve(Q, d)
            return y_unconstrained, np.zeros(n_constraints), np.zeros(n_constraints, dtype=bool)
    
    def compute_implicit_gradient(self, x: np.ndarray, y_star: np.ndarray, q: np.ndarray, 
                                 active_mask: np.ndarray, lambda_dual: np.ndarray) -> np.ndarray:
        """
        Compute exact implicit gradient using DS-BLO formulas:
        ∇F_q(x) = ∇_x f(x,y_q*(x)) + [∇y_q*(x)]^T ∇_y f(x,y_q*(x))
        """
        # Get active constraints
        active_indices = np.where(active_mask)[0]
        
        # Compute Hessians
        H_yy = self.Q_lower  # ∇²_{yy}g(x,y) = Q_lower (constant)
        H_xy = self.P.T      # ∇²_{xy}g(x,y) = P^T (constant)
        
        # Compute gradients of upper objective
        grad_x_f = self.Q_upper @ (x - self.x_target)
        grad_y_f = self.c_upper
        
        if len(active_indices) == 0:
            # Unconstrained case
            dy_dx = -np.linalg.solve(H_yy, H_xy)
        else:
            # Constrained case - use DS-BLO formulas
            A_bar = self.A[active_indices]
            B_bar = self.B[active_indices]
            lambda_bar = lambda_dual[active_indices]
            
            # ∇λ̄_q*(x) = -[Ā[∇²_{yy}g]^{-1}Ā^T]^{-1}[Ā[∇²_{yy}g]^{-1}∇²_{xy}g - B̄]
            H_yy_inv = np.linalg.inv(H_yy)
            A_bar_H_inv = A_bar @ H_yy_inv
            
            # Compute ∇λ̄_q*(x)
            term1 = A_bar_H_inv @ A_bar.T
            term2 = A_bar_H_inv @ H_xy - B_bar
            dlambda_dx = -np.linalg.solve(term1, term2)
            
            # ∇y_q*(x) = [∇²_{yy}g]^{-1}[-∇²_{xy}g - Ā^T∇λ̄_q*(x)]
            dy_dx = np.linalg.solve(H_yy, -H_xy - A_bar.T @ dlambda_dx)
        
        # Total implicit gradient
        implicit_grad = grad_x_f + dy_dx.T @ grad_y_f
        
        return implicit_grad
    
    def sample_perturbation(self) -> np.ndarray:
        """Sample perturbation vector q from continuous distribution"""
        return np.random.randn(self.dim) * self.perturbation_scale
    
    def compute_goldstein_stationarity(self, x: np.ndarray, gradients: List[np.ndarray]) -> float:
        """Compute Goldstein stationarity measure"""
        if len(gradients) == 0:
            return float('inf')
        
        # Average gradient in the neighborhood
        avg_grad = np.mean(gradients, axis=0)
        return float(np.linalg.norm(avg_grad))
    
    def run(self, x_init: Optional[np.ndarray] = None) -> Dict:
        """
        Run DS-BLO algorithm until convergence
        """
        # Initialize
        if x_init is None:
            x = np.random.randn(self.dim) * 0.1
        else:
            x = x_init.copy()
        
        # Algorithm parameters (adjusted for faster convergence)
        K = max(5, int(1 / np.log(1/self.beta) * np.log(32 * (4 + 2 * 10) / self.epsilon)))
        gamma_1 = K / self.delta_bar * 0.1  # Reduced for larger steps
        gamma_2 = 4 * gamma_1 * (4 + 2 * 10) * 0.1  # Reduced
        
        # Initialize momentum
        q = self.sample_perturbation()
        y_star, ll_info = self.solve_lower_level_exact(x, q)
        m = self.compute_implicit_gradient(x, y_star, q, ll_info['active_mask'], ll_info['lambda'])
        
        # Storage for convergence tracking
        x_history = [x.copy()]
        gap_history = []
        gradient_history = []
        upper_loss_history = []
        grad_norm_history = []
        
        print(f"\n🚀 Starting DS-BLO Algorithm")
        print(f"Initial x: {x}")
        print(f"Initial gap: {float('inf')}")
        
        for t in range(self.max_iterations):
            # Update upper-level variable
            eta_t = 1 / (gamma_1 * np.linalg.norm(m) + gamma_2)
            x_new = x - eta_t * m
            
            # Sample intermediate point (second perturbation)
            alpha = np.random.rand()
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
            x_history.append(x.copy())
            gradient_history.append(g_new.copy())
            
            # Compute and store upper level loss and gradient norm
            current_upper_loss = self.upper_objective(x, y_star_new, add_noise=False)
            current_grad_norm = np.linalg.norm(g_new)
            upper_loss_history.append(current_upper_loss)
            grad_norm_history.append(current_grad_norm)
            
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
                    print(f"Iteration {t+1}: gap={gap:.6f}, ||x||={np.linalg.norm(x):.6f}, upper_loss={current_upper_loss:.6f}, grad_norm={current_grad_norm:.6f}")
        
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
            'upper_loss_history': upper_loss_history,
            'grad_norm_history': grad_norm_history,
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
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Goldstein Stationarity Gap
        plt.subplot(2, 3, 1)
        plt.semilogy(results['gap_history'])
        plt.xlabel('Iteration')
        plt.ylabel('Goldstein Stationarity Gap')
        plt.title('DS-BLO Convergence (Gap)')
        plt.grid(True)
        
        # Plot 2: Upper Level Loss
        plt.subplot(2, 3, 2)
        plt.plot(results['upper_loss_history'])
        plt.xlabel('Iteration')
        plt.ylabel('Upper Level Loss')
        plt.title('Upper Level Objective')
        plt.grid(True)
        
        # Plot 3: Gradient Norm
        plt.subplot(2, 3, 3)
        plt.semilogy(results['grad_norm_history'])
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Norm (log scale)')
        plt.title('Implicit Gradient Norm')
        plt.grid(True)
        
        # Plot 4: Upper Level Variables
        plt.subplot(2, 3, 4)
        x_history = np.array(results['x_history'])
        for i in range(x_history.shape[1]):
            plt.plot(x_history[:, i], label=f'x[{i}]')
        plt.xlabel('Iteration')
        plt.ylabel('Variable Value')
        plt.title('Upper-Level Variables')
        plt.legend()
        plt.grid(True)
        
        # Plot 5: Combined Loss and Gap
        plt.subplot(2, 3, 5)
        ax1 = plt.gca()
        color1 = 'tab:red'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Upper Level Loss', color=color1)
        line1 = ax1.plot(results['upper_loss_history'], color=color1, label='Upper Loss')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel('Gap (log scale)', color=color2)
        line2 = ax2.semilogy(results['gap_history'], color=color2, label='Gap')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title('Loss vs Gap Evolution')
        
        # Plot 6: Gradient Norm vs Loss
        plt.subplot(2, 3, 6)
        plt.scatter(results['grad_norm_history'], results['upper_loss_history'], 
                   c=range(len(results['grad_norm_history'])), cmap='viridis', alpha=0.6)
        plt.xlabel('Gradient Norm')
        plt.ylabel('Upper Level Loss')
        plt.title('Gradient Norm vs Loss')
        plt.colorbar(label='Iteration')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('dsblo_convergence.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n📈 Convergence plot saved as 'dsblo_convergence.png'")
    
    return results

if __name__ == "__main__":
    results = main()
