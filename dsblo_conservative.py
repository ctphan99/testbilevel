#!/usr/bin/env python3
"""
DS-BLO Conservative Implementation
Conservative version that matches algorithms.py more closely
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

class DSBLOConservative:
    """
    DS-BLO Conservative Implementation with better parameter handling
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem, device: str = 'cpu', dtype=torch.float64):
        self.problem = problem
        self.device = device
        self.dtype = dtype
        
        print(f"DS-BLO Conservative Implementation")
        print(f"  Device: {device}, dtype: {dtype}")
    
    def active(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute active constraint matrices"""
        eps = 1e-3
        h = self.problem.constraints(x, y)
        
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
    
    def grad_lambdastar(self, x: torch.Tensor, y: torch.Tensor, Aact: torch.Tensor, Bact: torch.Tensor) -> torch.Tensor:
        """Compute gradient of lambda_star"""
        hessyy_g_inv = torch.linalg.inv(self.hessyy_g(x, y))
        g = -torch.linalg.inv(Aact @ hessyy_g_inv @ Aact.T) @ (Aact @ hessyy_g_inv @ self.hessxy_g(x, y) - Bact)
        return g
    
    def grad_ystar(self, x: torch.Tensor, y: torch.Tensor, Aact: Optional[torch.Tensor], Bact: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute gradient of y_star"""
        if Aact is None:
            g = -torch.linalg.inv(self.hessyy_g(x, y)) @ self.hessxy_g(x, y)
        else:
            g = torch.linalg.inv(self.hessyy_g(x, y)) @ (-self.hessxy_g(x, y) - Aact.T @ self.grad_lambdastar(x, y, Aact, Bact))
        return g
    
    def grad_F(self, x: torch.Tensor, y: torch.Tensor, Aact: Optional[torch.Tensor], Bact: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute implicit gradient"""
        g = self.gradx_f(x, y) + self.grad_ystar(x, y, Aact, Bact).T @ self.grady_f(x, y)
        return g
    
    def hessyy_g(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Hessian of g w.r.t. y"""
        return self.problem.Q_lower
    
    def hessxy_g(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Cross Hessian of g w.r.t. x and y"""
        return self.problem.P
    
    def gradx_f(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gradient of f w.r.t. x"""
        return self.problem.Q_upper @ x + self.problem.c_upper + self.problem.P @ y
    
    def grady_f(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gradient of f w.r.t. y"""
        return self.problem.P.T @ x + self.problem.Q_lower @ y + self.problem.c_lower
    
    def solve_ll(self, x: torch.Tensor) -> torch.Tensor:
        """Solve lower-level problem exactly"""
        y_star, _ = self.problem.solve_lower_level(x)
        return y_star
    
    def optimize(self, x0: torch.Tensor, T: int, alpha: float) -> Dict:
        """
        Run DS-BLO optimization with conservative parameters
        """
        print("🚀 DS-BLO Conservative Algorithm")
        print("=" * 50)
        print(f"T = {T}, α = {alpha:.6f}")
        print()
        
        # Conservative parameters - much smaller step sizes
        gamma1 = 0.1  # Much smaller than before
        gamma2 = 0.1  # Much smaller than before
        beta = 0.9
        
        print(f"γ₁ = {gamma1:.6f}, γ₂ = {gamma2:.6f}, β = {beta:.6f}")
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
        yhat = self.solve_ll(x)
        
        # Compute m1 = g1 = ∇̂ F_{q1}(x1)
        Aact, Bact = self.active(x, yhat)
        m = self.grad_F(x, yhat, Aact, Bact)
        
        # Evaluate initial point
        y_star, _ = self.problem.solve_lower_level(x)
        ul_loss_0 = self.problem.upper_objective(x, y_star).item()
        ul_losses.append(ul_loss_0)
        hypergrad_norms.append(torch.norm(m).item())
        x_history.append(x.clone().detach())
        
        print(f"Iteration 0: ||g_0|| = {torch.norm(m).item():.6f}, ul_loss = {ul_loss_0:.6f}")
        
        # Main optimization loop with very conservative step sizes
        for t in range(1, T + 1):
            # Update x_{t+1} = x_t - η_t * m_t with very conservative step size
            grad_norm = torch.norm(m)
            eta = 1.0 / (gamma1 * grad_norm + gamma2)
            
            # Very conservative step size clipping
            eta = min(eta, 0.001)  # Very small cap
            
            x_prev = x.clone()
            x = x - eta * m
            
            # Sample xbar_{t+1} ~ U[x_t, x_{t+1}]
            xbar = torch.zeros_like(x)
            for j in range(x.shape[0]):
                xbar[j] = torch.rand(1, device=self.device, dtype=self.dtype) * (x[j] - x_prev[j]) + x_prev[j]
            
            # Sample q_{t+1} ~ Q independently from q_t
            q = torch.randn(self.problem.dim, device=self.device, dtype=self.dtype)
            q = 1e-6 * (q / torch.norm(q))  # Normalize and scale
            
            # Find approximate solution yhat_{q_{t+1}}(xbar_{t+1})
            yhat = self.solve_ll(xbar)
            
            # Compute g
            Aact, Bact = self.active(x, yhat)
            g = self.grad_F(xbar, yhat, Aact, Bact)
            
            # Update momentum with very conservative gradient clipping
            g_norm = torch.norm(g)
            if g_norm > 10:  # Very conservative clipping
                g = g / g_norm * 10
            
            m = beta * m + (1 - beta) * g
            
            # Evaluate current point
            y_star, _ = self.problem.solve_lower_level(x)
            ul_loss = self.problem.upper_objective(x, y_star).item()
            ul_losses.append(ul_loss)
            hypergrad_norms.append(torch.norm(m).item())
            x_history.append(x.clone().detach())
            
            if t % 100 == 0:
                print(f"Iteration {t}/{T}: ||g|| = {torch.norm(g).item():.6f}, ul_loss = {ul_loss:.6f}")
        
        # Final evaluation
        y_star, _ = self.problem.solve_lower_level(x)
        final_ul_loss = self.problem.upper_objective(x, y_star).item()
        final_grad_norm = torch.norm(m).item()
        
        print()
        print("DS-BLO Results:")
        print(f"  Final UL loss: {final_ul_loss:.6f}")
        print(f"  Final gradient norm: {final_grad_norm:.6f}")
        print(f"  Converged: {final_grad_norm < 1e-3}")
        print(f"  Iterations: {T}")
        
        return {
            'x_out': x,
            'final_gradient': m,
            'final_gradient_norm': final_grad_norm,
            'final_ul_loss': final_ul_loss,
            'y_star': y_star,
            'x_history': x_history,
            'hypergrad_norms': hypergrad_norms,
            'ul_losses': ul_losses,
            'converged': final_grad_norm < 1e-3,
            'iterations': T
        }

def main():
    """Test DS-BLO conservative implementation"""
    parser = argparse.ArgumentParser(description='DS-BLO Conservative Implementation')
    parser.add_argument('--T', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--alpha', type=float, default=0.6, help='Accuracy parameter')
    parser.add_argument('--dim', type=int, default=5, help='Problem dimension')
    parser.add_argument('--constraints', type=int, default=3, help='Number of constraints')
    parser.add_argument('--plot-name', type=str, default='dsblo_conservative.png', help='Plot filename')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DS-BLO CONSERVATIVE IMPLEMENTATION TEST")
    print("=" * 80)
    print(f"Problem: dim={args.dim}, constraints={args.constraints}")
    print(f"DS-BLO Config: T={args.T}, alpha={args.alpha}")
    print()
    
    # Create problem instance
    problem = StronglyConvexBilevelProblem(dim=args.dim, num_constraints=args.constraints)
    
    # Initialize starting point
    x0 = torch.randn(args.dim, dtype=torch.float64)
    print(f"Starting point: x0 = {x0}")
    print()
    
    # Run DS-BLO
    dsblo = DSBLOConservative(problem)
    results = dsblo.optimize(x0, args.T, args.alpha)
    
    # Create plot
    print("Creating plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Upper-level loss
    ax1.plot(results['ul_losses'], linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Upper-level Loss')
    ax1.set_title('DS-BLO Conservative Upper-level Loss')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Gradient norm
    ax2.plot(results['hypergrad_norms'], linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Hypergradient Norm')
    ax2.set_title('DS-BLO Conservative Hypergradient Norm')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(args.plot_name, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {args.plot_name}")

if __name__ == "__main__":
    main()
