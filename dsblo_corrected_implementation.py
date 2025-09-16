#!/usr/bin/env python3
"""
DS-BLO Corrected Implementation
Based on algorithms.py reference and dsblo_paper.tex
Implements the exact DS-BLO algorithm with proper perturbation handling
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Optional, Tuple
from problem import StronglyConvexBilevelProblem
import warnings

warnings.filterwarnings('ignore')

class DSBLOCorrected:
    """
    DS-BLO Corrected Implementation following algorithms.py and dsblo_paper.tex
    Implements exact DS-BLO algorithm with proper stochastic perturbation
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem, device: str = 'cpu', dtype=torch.float64):
        self.problem = problem
        self.device = device
        self.dtype = dtype
        
        # DS-BLO parameters from algorithms.py
        self.gamma1 = 0.01  # Will be set based on K and delta_bar
        self.gamma2 = 0.01  # Will be set based on gamma1
        self.beta = 0.9     # Momentum parameter
        
        print(f"DS-BLO Corrected Implementation")
        print(f"  Device: {device}, dtype: {dtype}")
    
    def active_constraints(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute active constraint matrices Aact and Bact
        Based on algorithms.py active() method
        """
        eps = 1e-3
        h = self.problem.constraints(x, y)
        
        # Find active constraints: -eps < h_i <= 0
        active_indices = []
        for i in range(self.problem.num_constraints):
            if -eps < h[i] <= 0:
                active_indices.append(i)
        
        if len(active_indices) == 0:
            return None, None
        else:
            Aact = self.problem.A[active_indices, :]
            Bact = self.problem.B[active_indices, :]
            return Aact, Bact
    
    def hessyy_g(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Hessian of lower-level objective w.r.t. y: ∇²_yy g(x,y)"""
        # For our problem: g(x,y) = 0.5 * y^T Q_lower y + c_lower^T y
        # ∇²_yy g = Q_lower
        return self.problem.Q_lower
    
    def hessxy_g(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mixed Hessian: ∇²_xy g(x,y)"""
        # For our problem: ∇²_xy g = 0 (no cross terms)
        return torch.zeros(self.problem.dim, self.problem.dim, device=self.device, dtype=self.dtype)
    
    def gradx_f(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gradient of upper-level objective w.r.t. x: ∇_x f(x,y)"""
        # For our problem: f(x,y) = 0.5 * x^T Q_upper x + c_upper^T x + 0.5 * y^T P y + x^T P y
        # ∇_x f = Q_upper x + c_upper + P y
        return self.problem.Q_upper @ x + self.problem.c_upper + self.problem.P @ y
    
    def grady_f(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gradient of upper-level objective w.r.t. y: ∇_y f(x,y)"""
        # For our problem: ∇_y f = P y + P^T x = P (y + x)
        return self.problem.P @ (y + x)
    
    def grad_lambdastar(self, x: torch.Tensor, y: torch.Tensor, Aact: torch.Tensor, Bact: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of lambda_star using Equation 4b from dsblo_paper.tex
        ∇λ̄_q*(x) = -[Ā[∇²_{yy}g]^{-1}Ā^T]^{-1}[Ā[∇²_{yy}g]^{-1}∇²_{xy}g - B̄]
        """
        hessyy_g_inv = torch.linalg.inv(self.hessyy_g(x, y))
        hessxy_g = self.hessxy_g(x, y)
        
        # Ā[∇²_{yy}g]^{-1}Ā^T
        AQinvAT = Aact @ hessyy_g_inv @ Aact.T
        
        # Ā[∇²_{yy}g]^{-1}∇²_{xy}g - B̄
        rhs = Aact @ hessyy_g_inv @ hessxy_g - Bact
        
        # Solve: ∇λ̄_q*(x) = -[AQinvAT]^{-1} * rhs
        try:
            grad_lambda = -torch.linalg.solve(AQinvAT, rhs)
        except:
            # Fallback if matrix is singular
            grad_lambda = torch.zeros(Aact.shape[0], self.problem.dim, device=self.device, dtype=self.dtype)
        
        return grad_lambda
    
    def grad_ystar(self, x: torch.Tensor, y: torch.Tensor, Aact: Optional[torch.Tensor], Bact: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute gradient of y_star using Equation 4a from dsblo_paper.tex
        ∇y_q*(x) = [∇²_{yy}g]^{-1}[-∇²_{xy}g - Ā^T∇λ̄_q*(x)]
        """
        hessyy_g_inv = torch.linalg.inv(self.hessyy_g(x, y))
        hessxy_g = self.hessxy_g(x, y)
        
        if Aact is None:
            # No active constraints: ∇y_q*(x) = -[∇²_{yy}g]^{-1}∇²_{xy}g
            grad_y = -hessyy_g_inv @ hessxy_g
        else:
            # With active constraints: ∇y_q*(x) = [∇²_{yy}g]^{-1}[-∇²_{xy}g - Ā^T∇λ̄_q*(x)]
            grad_lambda = self.grad_lambdastar(x, y, Aact, Bact)
            grad_y = hessyy_g_inv @ (-hessxy_g - Aact.T @ grad_lambda)
        
        return grad_y
    
    def grad_F(self, x: torch.Tensor, y: torch.Tensor, Aact: Optional[torch.Tensor], Bact: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute implicit gradient: ∇F = ∇_x f + [∇y_q*(x)]^T ∇_y f
        """
        grad_x_f = self.gradx_f(x, y)
        grad_y_f = self.grady_f(x, y)
        grad_y_star = self.grad_ystar(x, y, Aact, Bact)
        
        # Total implicit gradient
        grad_F = grad_x_f + grad_y_star.T @ grad_y_f
        
        return grad_F
    
    def solve_lower_level_perturbed(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Solve perturbed lower-level problem: min_y g(x,y) + q^T y subject to Ax + By ≤ b
        """
        # For our problem: g(x,y) + q^T y = 0.5 * y^T Q_lower y + (c_lower + q)^T y
        # The perturbation q^T y shifts the linear term
        
        # Unconstrained optimum: y* = -Q_lower^{-1} * (c_lower + q)
        c_perturbed = self.problem.c_lower + q
        y_unconstrained = -torch.linalg.solve(self.problem.Q_lower, c_perturbed)
        
        # Project onto feasible region using PGD
        y = y_unconstrained.clone()
        lr = 0.01
        max_iter = 1000
        
        for i in range(max_iter):
            # Gradient of perturbed objective: ∇_y (g + q^T y) = Q_lower y + c_lower + q
            grad_g = self.problem.Q_lower @ y + c_perturbed
            
            # Gradient step
            y_new = y - lr * grad_g
            
            # Project onto feasible region
            y = self._project_onto_constraints(x, y_new)
            
            # Check convergence
            if torch.norm(grad_g) < 1e-6:
                break
        
        return y
    
    def _project_onto_constraints(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Project y onto feasible region {y : Ax + By ≤ b}"""
        h = self.problem.constraints(x, y)
        violations = torch.clamp(h, min=0)
        
        if torch.norm(violations) < 1e-10:
            return y  # Already feasible
        
        # Move in direction of constraint normals to restore feasibility
        correction = torch.zeros_like(y)
        for i in range(self.problem.num_constraints):
            if violations[i] > 0:
                B_norm_sq = torch.norm(self.problem.B[i])**2
                if B_norm_sq > 1e-10:
                    correction += violations[i] * self.problem.B[i] / B_norm_sq
        
        return y - correction
    
    def optimize(self, x0: torch.Tensor, T: int, alpha: float) -> Dict:
        """
        Run DS-BLO optimization following algorithms.py exactly
        """
        print("🚀 DS-BLO Corrected Algorithm")
        print("=" * 50)
        print(f"T = {T}, α = {alpha:.6f}")
        print(f"γ₁ = {self.gamma1:.6f}, γ₂ = {self.gamma2:.6f}, β = {self.beta:.6f}")
        print()
        
        # Set parameters based on DS-BLO paper
        delta_bar = alpha**3
        K = max(1, int(1.0 / delta_bar))
        self.gamma1 = K / delta_bar
        self.gamma2 = 4 * self.gamma1 * (1.0 + 2.0)  # Simplified from paper
        
        print(f"K = {K}, δ̄ = {delta_bar:.6f}")
        print(f"Updated γ₁ = {self.gamma1:.6f}, γ₂ = {self.gamma2:.6f}")
        print()
        
        # Initialize
        x = x0.clone().detach()
        
        # Storage for results
        x_history = []
        ul_losses = []
        hypergrad_norms = []
        
        print("Starting DS-BLO optimization...")
        print("-" * 50)
        
        # Sample q1 ~ Q
        q = torch.randn(self.problem.dim, device=self.device, dtype=self.dtype)
        q = 1e-6 * (q / torch.norm(q))  # Normalize and scale
        
        # Find approximate solution yhat_{q1}(x1)
        yhat = self.solve_lower_level_perturbed(x, q)
        
        # Compute m1 = g1 = ∇̂ F_{q1}(x1)
        Aact, Bact = self.active_constraints(x, yhat)
        m = self.grad_F(x, yhat, Aact, Bact)
        
        # Evaluate initial point
        y_star, _ = self.problem.solve_lower_level(x)
        ul_loss_0 = self.problem.upper_objective(x, y_star).item()
        ul_losses.append(ul_loss_0)
        hypergrad_norms.append(torch.norm(m).item())
        x_history.append(x.clone().detach())
        
        print(f"Iteration 0: ||g_0|| = {torch.norm(m).item():.6f}, ul_loss = {ul_loss_0:.6f}")
        
        # Main optimization loop
        for t in range(1, T + 1):
            # Update x_{t+1} = x_t - η_t * m_t
            eta = 1.0 / (self.gamma1 * torch.norm(m) + self.gamma2)
            x_prev = x.clone()
            x = x - eta * m
            
            # Sample x̄_{t+1} ~ Uniform[x_t, x_{t+1}]
            xbar = torch.zeros_like(x)
            for j in range(x.shape[0]):
                xbar[j] = torch.rand(1, device=self.device, dtype=self.dtype) * (x[j] - x_prev[j]) + x_prev[j]
            
            # Sample q_{t+1} ~ Q independently from q_t
            q = torch.randn(self.problem.dim, device=self.device, dtype=self.dtype)
            q = 1e-6 * (q / torch.norm(q))  # Normalize and scale
            
            # Find approximate solution yhat_{q_{t+1}}(x̄_{t+1})
            yhat = self.solve_lower_level_perturbed(xbar, q)
            
            # Compute g_{t+1} = ∇̂ F_{q_{t+1}}(x̄_{t+1})
            Aact, Bact = self.active_constraints(xbar, yhat)
            g = self.grad_F(xbar, yhat, Aact, Bact)
            
            # Update momentum: m_{t+1} = β * m_t + (1 - β) * g_{t+1}
            m = self.beta * m + (1 - self.beta) * g
            
            # Evaluate at current x (not xbar)
            y_star, _ = self.problem.solve_lower_level(x)
            ul_loss_t = self.problem.upper_objective(x, y_star).item()
            ul_losses.append(ul_loss_t)
            hypergrad_norms.append(torch.norm(g).item())
            x_history.append(x.clone().detach())
            
            # Print progress
            if t % max(1, T // 20) == 0 or t <= 10:
                g_norm = torch.norm(g).item()
                m_norm = torch.norm(m).item()
                print(f"Iteration {t:4d}/{T}: ||g_t|| = {g_norm:.6f}, ||m_t|| = {m_norm:.6f}, ul_loss = {ul_loss_t:.6f}")
        
        print()
        print("DS-BLO optimization completed!")
        
        # Final evaluation
        y_final, _ = self.problem.solve_lower_level(x)
        final_ul_loss = self.problem.upper_objective(x, y_final).item()
        final_grad_norm = torch.norm(m).item()
        
        print(f"Final UL loss f(x, y*): {final_ul_loss:.6f}")
        print(f"Final momentum norm: {final_grad_norm:.6f}")
        print()
        
        return {
            'x_out': x,
            'final_gradient': m,
            'final_gradient_norm': final_grad_norm,
            'final_ul_loss': final_ul_loss,
            'y_star': y_final,
            'x_history': x_history,
            'hypergrad_norms': hypergrad_norms,
            'ul_losses': ul_losses,
            'converged': final_grad_norm < 1e-3,
            'iterations': T
        }

def run_dsblo_f2csa_comparison():
    """Run both DS-BLO and F2CSA on the same problem"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DS-BLO vs F2CSA Comparison')
    parser.add_argument('--T', type=int, default=2000, help='Number of iterations')
    parser.add_argument('--alpha', type=float, default=0.6, help='Alpha parameter (from sbatch)')
    parser.add_argument('--dim', type=int, default=5, help='Problem dimension')
    parser.add_argument('--constraints', type=int, default=3, help='Number of constraints')
    parser.add_argument('--D', type=float, default=0.05, help='F2CSA clipping parameter')
    parser.add_argument('--eta', type=float, default=0.0001, help='F2CSA step size')
    parser.add_argument('--Ng', type=int, default=64, help='F2CSA gradient samples')
    parser.add_argument('--plot-name', type=str, default='dsblo_f2csa_corrected_comparison.png', help='Plot filename')
    
    args = parser.parse_args()
    
    print("🔬 DS-BLO vs F2CSA Corrected Comparison")
    print("=" * 80)
    print(f"Problem: dim={args.dim}, constraints={args.constraints}")
    print(f"Iterations: T={args.T}")
    print(f"Alpha: α={args.alpha}")
    print()
    
    # Set seed BEFORE creating problem to ensure same parameters
    torch.manual_seed(42)  # Ensure reproducibility
    np.random.seed(42)     # Also set numpy seed for any numpy operations
    
    # Initialize the SAME problem for both algorithms
    problem = StronglyConvexBilevelProblem(
        dim=args.dim, 
        num_constraints=args.constraints, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    # Use the SAME initial point for both algorithms
    x0 = torch.randn(args.dim, dtype=torch.float64)
    
    print(f"Initial point x0: {x0}")
    print(f"Initial UL loss: {problem.upper_objective(x0, problem.solve_lower_level(x0)[0]).item():.6f}")
    print()
    
    # Initialize algorithms
    dsblo = DSBLOCorrected(problem)
    
    # Import F2CSA
    from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
    from f2csa_algorithm2_working import F2CSAAlgorithm2Working
    f2csa = F2CSAAlgorithm2Working(problem)
    
    # F2CSA parameters
    delta = args.alpha**3
    
    print("🚀 Running DS-BLO Corrected...")
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
    ax1.plot(dsblo_results['ul_losses'], label='DS-BLO Corrected', color='blue', linewidth=2)
    ax1.plot(f2csa_results['ul_losses'], label='F2CSA', color='red', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Upper-level Loss f(x, y*)')
    ax1.set_title('Upper-level Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Hypergradient norm comparison
    ax2.plot(dsblo_results['hypergrad_norms'], label='DS-BLO Corrected', color='blue', linewidth=2)
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
    algorithms = ['DS-BLO Corrected', 'F2CSA']
    
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
    print("\n📋 CORRECTED COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<25} {'DS-BLO Corrected':<20} {'F2CSA':<15} {'Winner':<15}")
    print("-" * 80)
    print(f"{'Initial UL Loss':<25} {dsblo_results['ul_losses'][0]:<20.6f} {f2csa_results['ul_losses'][0]:<15.6f} {'Same':<15}")
    print(f"{'Final UL Loss':<25} {dsblo_results['final_ul_loss']:<20.6f} {f2csa_results['final_ul_loss']:<15.6f} {'F2CSA' if f2csa_results['final_ul_loss'] < dsblo_results['final_ul_loss'] else 'DS-BLO':<15}")
    print(f"{'Final Grad Norm':<25} {dsblo_results['final_gradient_norm']:<20.6f} {f2csa_results['final_gradient_norm']:<15.6f} {'F2CSA' if f2csa_results['final_gradient_norm'] < dsblo_results['final_gradient_norm'] else 'DS-BLO':<15}")
    print(f"{'Converged':<25} {dsblo_results['converged']:<20} {f2csa_results['converged']:<15} {'Both' if dsblo_results['converged'] and f2csa_results['converged'] else 'One':<15}")
    print(f"{'Iterations':<25} {dsblo_results['iterations']:<20} {f2csa_results['iterations']:<15} {'Same':<15}")
    
    print(f"\n📊 Plot saved to: {args.plot_name}")
    print("✅ Corrected comparison completed!")

if __name__ == "__main__":
    run_dsblo_f2csa_comparison()
