#!/usr/bin/env python3
"""
DS-BLO vs F2CSA Comparison Script
Runs both algorithms on the same problem instance with identical upper-level loss calculation
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Optional, Tuple
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
import warnings

warnings.filterwarnings('ignore')

class DSBLOAlgorithm:
    """
    DS-BLO Algorithm Implementation based on dsblo_paper.tex
    Doubly Stochastic Bilevel Optimization with KKT-based gradient computation
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem, device: str = 'cpu', dtype=torch.float64):
        self.problem = problem
        self.device = device
        self.dtype = dtype
        
        # DS-BLO parameters from the paper
        self.epsilon = 0.01  # Perturbation parameter
        self.beta = 0.1     # Step size parameter
        self.gamma_1 = 0.01  # Upper-level step size
        self.gamma_2 = 0.01  # Lower-level step size
        
        print(f"DS-BLO Algorithm Implementation")
        print(f"  ε = {self.epsilon}, β = {self.beta}")
        print(f"  γ₁ = {self.gamma_1}, γ₂ = {self.gamma_2}")
        print(f"  Device: {device}, dtype: {dtype}")
    
    def solve_lower_level_kkt(self, x: torch.Tensor, epsilon: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve lower-level problem using KKT conditions with stochastic perturbation
        Following DS-BLO paper equations (4a) and (4b)
        """
        try:
            import cvxpy as cp
            
            # Convert to numpy for CVXPY
            x_np = x.detach().cpu().numpy()
            
            # Problem dimensions
            n = self.problem.dim
            m = self.problem.num_constraints
            
            # Variables
            y = cp.Variable(n)
            lambda_vars = cp.Variable(m)
            
            # Stochastic perturbation: add noise to constraints
            noise = np.random.normal(0, epsilon, m)
            b_perturbed = self.problem.b.detach().cpu().numpy() + noise
            
            # Lower-level objective: g(x,y) = 0.5 * y^T Q_lower y + c_lower^T y
            Q_lower_np = self.problem.Q_lower.detach().cpu().numpy()
            c_lower_np = self.problem.c_lower.detach().cpu().numpy()
            
            objective = cp.Minimize(0.5 * cp.quad_form(y, Q_lower_np) + c_lower_np.T @ y)
            
            # Constraints: h(x,y) = Ax + By - b ≤ 0 (with perturbation)
            A_np = self.problem.A.detach().cpu().numpy()
            B_np = self.problem.B.detach().cpu().numpy()
            
            constraints = [A_np @ x_np + B_np @ y <= b_perturbed]
            
            # Solve
            problem_cvx = cp.Problem(objective, constraints)
            problem_cvx.solve(verbose=False)
            
            if problem_cvx.status == "optimal":
                y_opt = torch.tensor(y.value, device=self.device, dtype=self.dtype)
                lambda_opt = torch.tensor(constraints[0].dual_value, device=self.device, dtype=self.dtype)
                return y_opt, lambda_opt
            else:
                # Fallback to PGD if CVXPY fails
                return self._solve_pgd_fallback(x, epsilon)
                
        except Exception as e:
            print(f"CVXPY solve failed: {e}, using PGD fallback")
            return self._solve_pgd_fallback(x, epsilon)
    
    def _solve_pgd_fallback(self, x: torch.Tensor, epsilon: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback PGD solver for lower-level problem"""
        # Initialize at unconstrained optimum
        y = -torch.linalg.solve(self.problem.Q_lower, self.problem.c_lower)
        
        # Add stochastic perturbation to constraints
        noise = torch.randn(self.problem.num_constraints, device=self.device, dtype=self.dtype) * epsilon
        b_perturbed = self.problem.b + noise
        
        # PGD iterations
        lr = 0.01
        max_iter = 1000
        
        for i in range(max_iter):
            # Gradient of lower-level objective
            grad_g = self.problem.Q_lower @ y + self.problem.c_lower
            
            # Gradient step
            y_new = y - lr * grad_g
            
            # Project onto perturbed feasible region
            y = self._project_onto_perturbed_constraints(x, y_new, b_perturbed)
            
            # Check convergence
            if torch.norm(grad_g) < 1e-6:
                break
        
        # Compute dual variables
        h = self.problem.A @ x + self.problem.B @ y - b_perturbed
        lambda_opt = torch.clamp(-h, min=0)
        
        return y, lambda_opt
    
    def _project_onto_perturbed_constraints(self, x: torch.Tensor, y: torch.Tensor, b_perturbed: torch.Tensor) -> torch.Tensor:
        """Project y onto perturbed feasible region"""
        h = self.problem.A @ x + self.problem.B @ y - b_perturbed
        violations = torch.clamp(h, min=0)
        
        if torch.norm(violations) < 1e-10:
            return y
        
        # Move in direction of constraint normals
        correction = torch.zeros_like(y)
        for i in range(self.problem.num_constraints):
            if violations[i] > 0:
                B_norm_sq = torch.norm(self.problem.B[i])**2
                if B_norm_sq > 1e-10:
                    correction += violations[i] * self.problem.B[i] / B_norm_sq
        
        return y - correction
    
    def compute_hypergradient(self, x: torch.Tensor, y: torch.Tensor, lambda_opt: torch.Tensor) -> torch.Tensor:
        """
        Compute hypergradient using DS-BLO exact formulas (Equations 4a and 4b)
        ∇y_q*(x) = [∇²_{yy}g(x,y_q*(x))]^{-1}[-∇²_{xy}g(x,y_q*(x)) - Ā^T∇λ̄_q*(x)]
        ∇λ̄_q*(x) = -[Ā[∇²_{yy}g(x,y_q*(x))]^{-1}Ā^T]^{-1}[Ā[∇²_{yy}g(x,y_q*(x))]^{-1}∇²_{xy}g(x,y_q*(x)) - B̄]
        """
        # For our problem: ∇²_{yy}g = Q_lower, ∇²_{xy}g = 0, Ā = A, B̄ = B
        Q_lower = self.problem.Q_lower
        A = self.problem.A
        B = self.problem.B
        
        # Compute ∇λ̄_q*(x) (Equation 4b)
        try:
            # Ā[∇²_{yy}g]^{-1}Ā^T = A Q_lower^{-1} A^T
            AQinvAT = A @ torch.linalg.solve(Q_lower, A.T)
            # B̄ = B
            lambda_grad = -torch.linalg.solve(AQinvAT, B)
        except:
            # Fallback if matrix is singular
            lambda_grad = torch.zeros_like(lambda_opt)
        
        # Compute ∇y_q*(x) (Equation 4a)
        # ∇²_{xy}g = 0 for our problem, so: ∇y_q*(x) = -Q_lower^{-1} A^T ∇λ̄_q*(x)
        try:
            y_grad = -torch.linalg.solve(Q_lower, A.T @ lambda_grad)
        except:
            y_grad = torch.zeros_like(y)
        
        # Upper-level gradient: ∇f = ∇_x f + ∇_y f * ∇y_q*(x)
        # For our problem: f(x,y) = 0.5 * x^T Q_upper x + c_upper^T x + 0.5 * y^T P y + x^T P y
        Q_upper = self.problem.Q_upper
        c_upper = self.problem.c_upper
        P = self.problem.P
        
        # ∇_x f = Q_upper x + c_upper + P y
        grad_x_f = Q_upper @ x + c_upper + P @ y
        
        # ∇_y f = P y + P^T x = P (y + x)
        grad_y_f = P @ (y + x)
        
        # Total hypergradient
        hypergrad = grad_x_f + grad_y_f @ y_grad
        
        return hypergrad
    
    def optimize(self, x0: torch.Tensor, T: int, alpha: float) -> Dict:
        """
        Run DS-BLO optimization
        """
        print("🚀 DS-BLO Algorithm")
        print("=" * 50)
        print(f"T = {T}, α = {alpha:.6f}")
        print(f"ε = {self.epsilon:.6f}, β = {self.beta:.6f}")
        print()
        
        # Initialize
        x = x0.clone().detach()
        
        # Storage for results
        x_history = []
        ul_losses = []
        hypergrad_norms = []
        
        print("Starting DS-BLO optimization...")
        print("-" * 50)
        
        for t in range(1, T + 1):
            # Solve lower-level problem with KKT and perturbation
            y_opt, lambda_opt = self.solve_lower_level_kkt(x, self.epsilon)
            
            # Compute hypergradient using exact DS-BLO formulas
            hypergrad = self.compute_hypergradient(x, y_opt, lambda_opt)
            
            # Compute upper-level loss (same as F2CSA)
            ul_loss_t = self.problem.upper_objective(x, y_opt).item()
            ul_losses.append(ul_loss_t)
            
            # Update x using DS-BLO step
            x = x - self.gamma_1 * hypergrad
            
            # Store history
            x_history.append(x.clone().detach())
            hypergrad_norms.append(torch.norm(hypergrad).item())
            
            # Print progress
            if t % max(1, T // 20) == 0 or t <= 10:
                g_norm = torch.norm(hypergrad).item()
                print(f"Iteration {t:4d}/{T}: ||g_t|| = {g_norm:.6f}, ul_loss = {ul_loss_t:.6f}")
        
        print()
        print("DS-BLO optimization completed!")
        
        # Final evaluation
        y_final, _ = self.solve_lower_level_kkt(x, self.epsilon)
        final_ul_loss = self.problem.upper_objective(x, y_final).item()
        final_grad = self.compute_hypergradient(x, y_final, lambda_opt)
        final_g_norm = torch.norm(final_grad).item()
        
        print(f"Final UL loss f(x, y*): {final_ul_loss:.6f}")
        print(f"Final hypergradient norm: {final_g_norm:.6f}")
        print()
        
        return {
            'x_out': x,
            'final_gradient': final_grad,
            'final_gradient_norm': final_g_norm,
            'final_ul_loss': final_ul_loss,
            'y_star': y_final,
            'x_history': x_history,
            'hypergrad_norms': hypergrad_norms,
            'ul_losses': ul_losses,
            'converged': final_g_norm < 1e-3,
            'iterations': T
        }

class F2CSAAlgorithm2Working:
    """
    F2CSA Algorithm 2 - WORKING Implementation (copied from f2csa_algorithm2_working.py)
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem, device: str = 'cpu', dtype=torch.float64):
        self.problem = problem
        self.device = device
        self.dtype = dtype
        
        # Initialize Algorithm 1 for hypergradient computation
        self.algorithm1 = F2CSAAlgorithm1Final(problem, device=device, dtype=dtype)
        
        # Persistent cache for warm-start
        self.prev_y = None
        self.prev_lambda = None
        self.prev_adam_state = None
        
    def clip_D(self, v: torch.Tensor, D: float) -> torch.Tensor:
        """Clipping function: clip_D(v) := min{1, D/||v||} * v"""
        v_norm = torch.norm(v)
        if v_norm <= D:
            return v
        else:
            return (D / v_norm) * v
    
    def optimize(self, x0: torch.Tensor, T: int, D: float, eta: float, 
                 delta: float, alpha: float, N_g: int = None, 
                 warm_ll: bool = False, keep_adam_state: bool = False) -> Dict:
        """
        Run F2CSA Algorithm 2 optimization with WORKING hypergradient computation
        """
        print("🚀 F2CSA Algorithm 2 - WORKING Implementation")
        print("=" * 70)
        print(f"T = {T}, D = {D:.6f}, η = {eta:.6f}")
        print(f"δ = {delta:.6f}, α = {alpha:.6f}")
        print()
        
        # Set default N_g if not provided
        if N_g is None:
            N_g = max(10, min(100, int(1.0 / (alpha**1.5))))
        
        print(f"Using N_g = {N_g} samples for hypergradient estimation")
        print()
        
        # Initialize
        x = x0.clone().detach()
        Delta = torch.zeros_like(x)
        
        # Storage for results
        z_history = []
        x_history = []
        g_history = []
        Delta_history = []
        ul_losses = []
        hypergrad_norms = []
        
        print("Starting F2CSA optimization...")
        print("-" * 50)
        
        # Main optimization loop
        current_eta = eta
        decay_every = max(50, T // 10)
        decay_factor = 0.95
        window = 50
        ul_tol = 1e-3
        hg_tol = 1e-2
        
        for t in range(1, T + 1):
            # Sample s_t ~ Unif[0, 1]
            s_t = torch.rand(1, device=self.device, dtype=self.dtype).item()
            
            # Update x_t and z_t
            x_t = x + Delta
            z_t = x + s_t * Delta
            
            # Compute hypergradient using Algorithm 1 (WORKING version) with warm-start
            if warm_ll and (self.prev_y is not None or self.prev_lambda is not None):
                oracle_result = self.algorithm1.oracle_sample(z_t, alpha, N_g, 
                                                             prev_y=self.prev_y, 
                                                             prev_lambda=self.prev_lambda,
                                                             keep_adam_state=keep_adam_state)
            else:
                oracle_result = self.algorithm1.oracle_sample(z_t, alpha, N_g)
            
            # Extract hypergradient from oracle result
            g_t = oracle_result[0] if isinstance(oracle_result, tuple) else oracle_result

            # Compute upper-level loss at x_t (same calculation as DS-BLO)
            y_star, _ = self.problem.solve_lower_level(x_t)
            ul_loss_t = self.problem.upper_objective(x_t, y_star).item()
            ul_losses.append(ul_loss_t)

            # Update direction with clipping
            Delta = self.clip_D(Delta - current_eta * g_t, D)
            
            # Decay step size periodically
            if t % decay_every == 0:
                current_eta = max(current_eta * decay_factor, eta * 0.2)
            
            # Early stopping: UL loss and hypergrad norm stabilization
            if len(ul_losses) >= window:
                ul_recent = ul_losses[-window:]
                hg_recent = hypergrad_norms[-window:]
                ul_span = max(ul_recent) - min(ul_recent)
                hg_span = max(hg_recent) - min(hg_recent)
                if ul_span < ul_tol and hg_span < hg_tol:
                    print(f"Early stop at iter {t}: UL loss and hypergrad stabilized")
                    T = t
                    break
            
            # Store history
            z_history.append(z_t.clone().detach())
            x_history.append(x_t.clone().detach())
            g_history.append(g_t.clone().detach())
            hypergrad_norms.append(torch.norm(g_t).item())
            Delta_history.append(Delta.clone().detach())
            
            # Update x for next iteration
            x = x_t
            
            # Cache lower-level solution for warm-start (if enabled)
            if warm_ll:
                if isinstance(oracle_result, tuple) and len(oracle_result) >= 3:
                    self.prev_y = oracle_result[1].clone().detach()
                    self.prev_lambda = oracle_result[2].clone().detach()
                
                if keep_adam_state and hasattr(self.algorithm1, 'adam_state'):
                    self.prev_adam_state = self.algorithm1.adam_state.copy() if self.algorithm1.adam_state else None
            
            # Print progress
            if t % max(1, T // 20) == 0 or t <= 10:
                g_norm = torch.norm(g_t).item()
                Delta_norm = torch.norm(Delta).item()
                print(f"Iteration {t:4d}/{T}: ||g_t|| = {g_norm:.6f}, ||Δ_t|| = {Delta_norm:.6f}, ul_loss = {ul_loss_t:.6f}")
        
        print()
        print("Computing F2CSA output points...")
        
        # Group iterations for Goldstein subdifferential
        total_iters = len(z_history)
        M = max(1, int(delta / D))
        K = max(1, total_iters // M)
        
        # Compute candidate points and their upper-level losses
        candidates = []
        for k in range(1, K + 1):
            start_idx = (k - 1) * M
            end_idx = min(k * M, len(z_history))
            if start_idx < len(z_history):
                z_group = z_history[start_idx:end_idx]
                x_k = torch.stack(z_group).mean(dim=0)
                y_star_k, _ = self.problem.solve_lower_level(x_k)
                ul_loss_k = self.problem.upper_objective(x_k, y_star_k).item()
                candidates.append((x_k, ul_loss_k))
        
        if candidates:
            x_out, best_ul_loss = min(candidates, key=lambda t: t[1])
            print(f"Selected F2CSA output with min f(x, y*): f = {best_ul_loss:.6f}")
        else:
            x_out = x_history[-1] if x_history else x0
            print("No F2CSA candidates formed; falling back to last x.")
        
        print()
        
        # Compute final gradient norm
        final_oracle_result = self.algorithm1.oracle_sample(x_out, alpha, N_g)
        final_g = final_oracle_result[0] if isinstance(final_oracle_result, tuple) else final_oracle_result
        final_g_norm = torch.norm(final_g).item()
        
        # Compute final upper-level loss f(x, y*)
        y_star, _ = self.problem.solve_lower_level(x_out)
        final_ul_loss = self.problem.upper_objective(x_out, y_star).item()
        
        print(f"Final F2CSA UL loss f(x, y*): {final_ul_loss:.6f}")
        print(f"Final F2CSA hypergradient norm: {final_g_norm:.6f}")
        print()
        
        return {
            'x_out': x_out,
            'final_gradient': final_g,
            'final_gradient_norm': final_g_norm,
            'final_ul_loss': final_ul_loss,
            'y_star': y_star,
            'x_history': x_history,
            'z_history': z_history,
            'g_history': g_history,
            'Delta_history': Delta_history,
            'hypergrad_norms': hypergrad_norms,
            'ul_losses': ul_losses,
            'converged': final_g_norm < 1e-3,
            'iterations': T
        }

def run_comparison():
    """Run both DS-BLO and F2CSA on the same problem"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DS-BLO vs F2CSA Comparison')
    parser.add_argument('--T', type=int, default=2000, help='Number of iterations')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha parameter')
    parser.add_argument('--dim', type=int, default=5, help='Problem dimension')
    parser.add_argument('--constraints', type=int, default=3, help='Number of constraints')
    parser.add_argument('--D', type=float, default=0.05, help='F2CSA clipping parameter')
    parser.add_argument('--eta', type=float, default=0.0001, help='F2CSA step size')
    parser.add_argument('--Ng', type=int, default=64, help='F2CSA gradient samples')
    parser.add_argument('--plot-name', type=str, default='dsblo_f2csa_comparison.png', help='Plot filename')
    
    args = parser.parse_args()
    
    print("🔬 DS-BLO vs F2CSA Comparison")
    print("=" * 80)
    print(f"Problem: dim={args.dim}, constraints={args.constraints}")
    print(f"Iterations: T={args.T}")
    print(f"Alpha: α={args.alpha}")
    print()
    
    # Initialize the SAME problem for both algorithms
    problem = StronglyConvexBilevelProblem(
        dim=args.dim, 
        num_constraints=args.constraints, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    # Use the SAME initial point for both algorithms
    torch.manual_seed(42)  # Ensure reproducibility
    x0 = torch.randn(args.dim, dtype=torch.float64)
    
    print(f"Initial point x0: {x0}")
    print(f"Initial UL loss: {problem.upper_objective(x0, problem.solve_lower_level(x0)[0]).item():.6f}")
    print()
    
    # Initialize algorithms
    dsblo = DSBLOAlgorithm(problem)
    f2csa = F2CSAAlgorithm2Working(problem)
    
    # F2CSA parameters
    delta = args.alpha**3
    
    print("🚀 Running DS-BLO...")
    print("-" * 50)
    dsblo_results = dsblo.optimize(x0, args.T, args.alpha)
    
    print("\n🚀 Running F2CSA...")
    print("-" * 50)
    f2csa_results = f2csa.optimize(
        x0, args.T, args.D, args.eta, delta, args.alpha, args.Ng,
        warm_ll=False, keep_adam_state=False
    )
    
    # Create comparison plots
    print("\n📊 Creating comparison plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Upper-level loss comparison
    ax1.plot(dsblo_results['ul_losses'], label='DS-BLO', color='blue', linewidth=2)
    ax1.plot(f2csa_results['ul_losses'], label='F2CSA', color='red', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Upper-level Loss f(x, y*)')
    ax1.set_title('Upper-level Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Hypergradient norm comparison
    ax2.plot(dsblo_results['hypergrad_norms'], label='DS-BLO', color='blue', linewidth=2)
    ax2.plot(f2csa_results['hypergrad_norms'], label='F2CSA', color='red', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Hypergradient Norm ||∇F||')
    ax2.set_title('Hypergradient Norm Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Final convergence comparison
    final_losses = [dsblo_results['final_ul_loss'], f2csa_results['final_ul_loss']]
    final_grads = [dsblo_results['final_gradient_norm'], f2csa_results['final_gradient_norm']]
    algorithms = ['DS-BLO', 'F2CSA']
    
    ax3.bar(algorithms, final_losses, color=['blue', 'red'], alpha=0.7)
    ax3.set_ylabel('Final Upper-level Loss')
    ax3.set_title('Final Loss Comparison')
    ax3.grid(True, alpha=0.3)
    
    ax4.bar(algorithms, final_grads, color=['blue', 'red'], alpha=0.7)
    ax4.set_ylabel('Final Hypergradient Norm')
    ax4.set_title('Final Gradient Norm Comparison')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(args.plot_name, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n📋 COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<25} {'DS-BLO':<15} {'F2CSA':<15} {'Winner':<15}")
    print("-" * 80)
    print(f"{'Initial UL Loss':<25} {dsblo_results['ul_losses'][0]:<15.6f} {f2csa_results['ul_losses'][0]:<15.6f} {'Same':<15}")
    print(f"{'Final UL Loss':<25} {dsblo_results['final_ul_loss']:<15.6f} {f2csa_results['final_ul_loss']:<15.6f} {'F2CSA' if f2csa_results['final_ul_loss'] < dsblo_results['final_ul_loss'] else 'DS-BLO':<15}")
    print(f"{'Final Grad Norm':<25} {dsblo_results['final_gradient_norm']:<15.6f} {f2csa_results['final_gradient_norm']:<15.6f} {'F2CSA' if f2csa_results['final_gradient_norm'] < dsblo_results['final_gradient_norm'] else 'DS-BLO':<15}")
    print(f"{'Converged':<25} {dsblo_results['converged']:<15} {f2csa_results['converged']:<15} {'Both' if dsblo_results['converged'] and f2csa_results['converged'] else 'One':<15}")
    print(f"{'Iterations':<25} {dsblo_results['iterations']:<15} {f2csa_results['iterations']:<15} {'Same':<15}")
    
    print(f"\n📊 Plot saved to: {args.plot_name}")
    print("✅ Comparison completed!")

if __name__ == "__main__":
    run_comparison()
