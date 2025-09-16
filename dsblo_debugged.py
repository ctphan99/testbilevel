#!/usr/bin/env python3
"""
DS-BLO Debugged Implementation
Exact match to algorithms.py reference implementation
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

class DSBLODebugged:
    """
    DS-BLO Debugged Implementation following algorithms.py exactly
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem, device: str = 'cpu', dtype=torch.float64):
        self.problem = problem
        self.device = device
        self.dtype = dtype
        
        # DS-BLO parameters from algorithms.py
        self.gamma1 = 0.01
        self.gamma2 = 0.01
        self.beta = 0.9
        
        print(f"DS-BLO Debugged Implementation")
        print(f"  Device: {device}, dtype: {dtype}")
    
    def active(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute active constraint matrices Aact and Bact
        Exact match to algorithms.py active() method
        """
        eps = 1e-3
        h = self.problem.constraints(x, y)
        
        # Find active constraints: -eps < h_i <= 0
        active_indices = []
        for i in range(self.problem.num_constraints):
            if -eps < h[i] <= 0:
                active_indices.append(i)
        
        if len(active_indices) == 0:
            Aact = None
            Bact = None
        else:
            Aact = self.problem.A[active_indices, :]
            Bact = self.problem.B[active_indices, :]
        
        return Aact, Bact
    
    def grad_lambdastar(self, x: torch.Tensor, y: torch.Tensor, Aact: torch.Tensor, Bact: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of lambda_star
        Exact match to algorithms.py grad_lambdastar() method
        """
        hessyy_g_inv = torch.linalg.inv(self.hessyy_g(x, y))
        g = -torch.linalg.inv(Aact @ hessyy_g_inv @ Aact.T) @ (Aact @ hessyy_g_inv @ self.hessxy_g(x, y) - Bact)
        return g
    
    def grad_ystar(self, x: torch.Tensor, y: torch.Tensor, Aact: Optional[torch.Tensor], Bact: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute gradient of y_star
        Exact match to algorithms.py grad_ystar() method
        """
        if Aact is None:
            g = -torch.linalg.inv(self.hessyy_g(x, y)) @ self.hessxy_g(x, y)
        else:
            g = torch.linalg.inv(self.hessyy_g(x, y)) @ (-self.hessxy_g(x, y) - Aact.T @ self.grad_lambdastar(x, y, Aact, Bact))
        return g
    
    def grad_F(self, x: torch.Tensor, y: torch.Tensor, Aact: Optional[torch.Tensor], Bact: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute implicit gradient
        Exact match to algorithms.py grad_F() method
        """
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
    
    def projy(self, y: torch.Tensor) -> torch.Tensor:
        """Project y onto feasible region"""
        # For our problem, this is just solving the lower-level problem
        return self.solve_ll(torch.zeros_like(y))  # Dummy x for projection
    
    def run(self, x: torch.Tensor, y: torch.Tensor, out_iter: int) -> Dict:
        """
        Run DS-BLO optimization following algorithms.py exactly
        """
        print("🚀 DS-BLO Debugged Algorithm (algorithms.py exact match)")
        print("=" * 60)
        print(f"out_iter = {out_iter}")
        print(f"γ₁ = {self.gamma1:.6f}, γ₂ = {self.gamma2:.6f}, β = {self.beta:.6f}")
        print()
        
        # Storage for results
        x_iter = []
        y_iter = []
        gradF = []
        loss = []
        iter_time = []
        start_time = 0
        
        # 0th iteration
        self.eval(x, y, x_iter, y_iter, gradF, loss, iter_time, start_time)
        start_time = torch.cuda.Event(enable_timing=True) if self.device == 'cuda' else None
        if start_time is None:
            import time
            start_time = time.time()
        
        # Sample q1 ~ Q
        q = torch.randn(self.problem.dim, device=self.device, dtype=self.dtype)
        q = 1e-6 * (q / torch.norm(q))  # Normalize and scale
        
        # Find an approximate solution yhat_{q1}(x1) s.t. Assumption 4 is satisfied
        yhat = self.projy(torch.randn(self.problem.dim, device=self.device, dtype=self.dtype))
        yhat = self.solve_ll(x)
        
        # Compute m1 = g1 = ∇̂ F_{q1}(x1)
        Aact, Bact = self.active(x, yhat)
        m = self.grad_F(x, yhat, Aact, Bact)
        
        print("Starting DS-BLO optimization...")
        print("-" * 50)
        
        for i in range(out_iter):
            # Update x_{t+1} = x_t - η_t * m_t
            eta = 1.0 / (self.gamma1 * torch.norm(m) + self.gamma2)
            x_prev = x.clone()
            x = x - eta * m
            
            # Sample xbar_{t+1} ~ U[x_t, x_{t+1}]
            xbar = torch.zeros_like(x)
            for j in range(x.shape[0]):
                xbar[j] = torch.rand(1, device=self.device, dtype=self.dtype) * (x[j] - x_prev[j]) + x_prev[j]
            
            # Sample q_{t+1} ~ Q independently from q_t
            q = torch.randn(self.problem.dim, device=self.device, dtype=self.dtype)
            q = 1e-6 * (q / torch.norm(q))  # Normalize and scale
            
            # Find an approximate solution yhat_{q_{t+1}}(xbar_{t+1}) s.t. Assumption 4 is satisfied
            yhat = self.projy(torch.randn(self.problem.dim, device=self.device, dtype=self.dtype))
            yhat = self.solve_ll(xbar)
            
            # Compute g
            Aact, Bact = self.active(x, yhat)
            g = self.grad_F(xbar, yhat, Aact, Bact)
            
            # Update momentum: m = β*m + (1-β)*g
            m = self.beta * m + (1 - self.beta) * g
            
            # Evaluate current point
            self.eval(x, y, x_iter, y_iter, gradF, loss, iter_time, start_time)
            
            if i % 100 == 0:
                print(f"Iteration {i+1}/{out_iter}: ||g|| = {torch.norm(g).item():.6f}, loss = {loss[-1]:.6f}")
        
        # Final evaluation
        y_star, _ = self.problem.solve_lower_level(x)
        final_ul_loss = self.problem.upper_objective(x, y_star).item()
        final_grad_norm = torch.norm(m).item()
        
        print()
        print("DS-BLO Results:")
        print(f"  Final UL loss: {final_ul_loss:.6f}")
        print(f"  Final gradient norm: {final_grad_norm:.6f}")
        print(f"  Converged: {final_grad_norm < 1e-3}")
        print(f"  Iterations: {out_iter}")
        
        return {
            'x_out': x,
            'final_gradient': m,
            'final_gradient_norm': final_grad_norm,
            'final_ul_loss': final_ul_loss,
            'y_star': y_star,
            'x_history': x_iter,
            'y_history': y_iter,
            'hypergrad_norms': gradF,
            'ul_losses': loss,
            'converged': final_grad_norm < 1e-3,
            'iterations': out_iter
        }
    
    def eval(self, x: torch.Tensor, y: torch.Tensor, x_iter: list, y_iter: list, 
             gradF: list, loss: list, iter_time: list, start_time):
        """Evaluation function matching algorithms.py"""
        x_iter.append(x.clone().detach())
        y_iter.append(y.clone().detach())
        
        ystar = self.solve_ll(x)
        Aact, Bact = self.active(x, ystar)
        grad_norm = torch.norm(self.grad_F(x, ystar, Aact, Bact)).item()
        gradF.append(grad_norm)
        
        ul_loss = self.problem.upper_objective(x, ystar).item()
        loss.append(ul_loss)
        
        if isinstance(start_time, (int, float)):
            import time
            iter_time.append(time.time() - start_time)
        else:
            iter_time.append(0)  # Placeholder for timing

def main():
    """Test DS-BLO debugged implementation"""
    parser = argparse.ArgumentParser(description='DS-BLO Debugged Implementation')
    parser.add_argument('--T', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--alpha', type=float, default=0.6, help='Accuracy parameter')
    parser.add_argument('--dim', type=int, default=5, help='Problem dimension')
    parser.add_argument('--constraints', type=int, default=3, help='Number of constraints')
    parser.add_argument('--plot-name', type=str, default='dsblo_debugged.png', help='Plot filename')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DS-BLO DEBUGGED IMPLEMENTATION TEST")
    print("=" * 80)
    print(f"Problem: dim={args.dim}, constraints={args.constraints}")
    print(f"DS-BLO Config: T={args.T}, alpha={args.alpha}")
    print()
    
    # Create problem instance
    problem = StronglyConvexBilevelProblem(dim=args.dim, num_constraints=args.constraints)
    
    # Initialize starting point
    x0 = torch.randn(args.dim, dtype=torch.float64)
    y0 = torch.randn(args.dim, dtype=torch.float64)
    print(f"Starting point: x0 = {x0}")
    print(f"Starting point: y0 = {y0}")
    print()
    
    # Run DS-BLO
    dsblo = DSBLODebugged(problem)
    results = dsblo.run(x0, y0, args.T)
    
    # Create plot
    print("Creating plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Upper-level loss
    ax1.plot(results['ul_losses'], linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Upper-level Loss')
    ax1.set_title('DS-BLO Upper-level Loss')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Gradient norm
    ax2.plot(results['hypergrad_norms'], linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Hypergradient Norm')
    ax2.set_title('DS-BLO Hypergradient Norm')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(args.plot_name, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {args.plot_name}")

if __name__ == "__main__":
    main()
