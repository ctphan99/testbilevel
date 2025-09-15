#!/usr/bin/env python3
"""
F2CSA Algorithm 2 with PRACTICAL penalty parameters
Test convergence with the practical implementation
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm1_practical import F2CSAAlgorithm1Practical
import warnings

warnings.filterwarnings('ignore')

class F2CSAAlgorithm2Practical:
    """
    F2CSA Algorithm 2: Nonsmooth Nonconvex Algorithm with Inexact Stochastic Hypergradient Oracle
    Using PRACTICAL penalty parameters for stable implementation
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem, device: str = 'cpu', dtype=torch.float64):
        self.problem = problem
        self.device = device
        self.dtype = dtype
        
        # Initialize Algorithm 1 for hypergradient computation
        self.algorithm1 = F2CSAAlgorithm1Practical(problem, device=device, dtype=dtype)
        
    def clip_D(self, v: torch.Tensor, D: float) -> torch.Tensor:
        """
        Clipping function: clip_D(v) := min{1, D/||v||} * v
        """
        v_norm = torch.norm(v)
        if v_norm <= D:
            return v
        else:
            return (D / v_norm) * v
    
    def optimize(self, x0: torch.Tensor, T: int, D: float, eta: float, 
                 delta: float, alpha: float, N_g: int = None) -> Dict:
        """
        Run F2CSA Algorithm 2 optimization with PRACTICAL parameters
        """
        print("🚀 F2CSA Algorithm 2 - PRACTICAL Implementation")
        print("=" * 70)
        print(f"T = {T}, D = {D:.6f}, η = {eta:.6f}")
        print(f"δ = {delta:.6f}, α = {alpha:.6f}")
        print()
        
        # Set default N_g if not provided
        if N_g is None:
            # Balanced N_g for practical implementation
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
        
        print("Starting optimization...")
        print("-" * 50)
        
        # Main optimization loop
        for t in range(1, T + 1):
            # Sample s_t ~ Unif[0, 1]
            s_t = torch.rand(1, device=self.device, dtype=self.dtype).item()
            
            # Update x_t and z_t
            x_t = x + Delta
            z_t = x + s_t * Delta
            
            # Compute hypergradient using Algorithm 1
            g_t = self.algorithm1.oracle_sample(z_t, alpha, N_g)
            
            # Update direction with clipping
            Delta = self.clip_D(Delta - eta * g_t, D)
            
            # Store history
            z_history.append(z_t.clone().detach())
            x_history.append(x_t.clone().detach())
            g_history.append(g_t.clone().detach())
            Delta_history.append(Delta.clone().detach())
            
            # Update x for next iteration
            x = x_t
            
            # Print progress
            if t % max(1, T // 20) == 0 or t <= 10:
                g_norm = torch.norm(g_t).item()
                Delta_norm = torch.norm(Delta).item()
                print(f"Iteration {t:4d}/{T}: ||g_t|| = {g_norm:.6f}, ||Δ_t|| = {Delta_norm:.6f}")
        
        print()
        print("Computing output points...")
        
        # Group iterations for Goldstein subdifferential
        M = max(1, int(delta / D))  # M = ⌊δ/D⌋
        K = max(1, int(T / M))      # K = ⌊T/M⌋
        
        print(f"M = {M}, K = {K}")
        
        # Compute averaged points
        x_outputs = []
        for k in range(1, K + 1):
            start_idx = (k - 1) * M
            end_idx = min(k * M, len(z_history))
            
            if start_idx < len(z_history):
                # Average z_t over the k-th group
                z_group = z_history[start_idx:end_idx]
                x_k = torch.stack(z_group).mean(dim=0)
                x_outputs.append(x_k)
        
        # Select output uniformly at random
        if x_outputs:
            output_idx = np.random.randint(0, len(x_outputs))
            x_out = x_outputs[output_idx]
        else:
            x_out = x_history[-1] if x_history else x0
        
        print(f"Selected output point from {len(x_outputs)} candidates")
        print()
        
        # Compute final gradient norm
        final_g = self.algorithm1.oracle_sample(x_out, alpha, N_g)
        final_g_norm = torch.norm(final_g).item()
        
        print(f"Final gradient norm: {final_g_norm:.6f}")
        print(f"Final point: {x_out}")
        print()
        
        return {
            'x_out': x_out,
            'final_gradient': final_g,
            'final_gradient_norm': final_g_norm,
            'x_history': x_history,
            'z_history': z_history,
            'g_history': g_history,
            'Delta_history': Delta_history,
            'grad_norms': [torch.norm(g).item() for g in g_history],
            'converged': final_g_norm < 1e-3,
            'iterations': T
        }

if __name__ == "__main__":
    # Test the practical Algorithm 2 implementation
    problem = StronglyConvexBilevelProblem(
        dim=5, 
        num_constraints=3, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    algorithm2 = F2CSAAlgorithm2Practical(problem)
    
    # Test parameters
    alpha = 0.1
    x0 = torch.randn(5, dtype=torch.float64)
    T = 100
    D = 1.0
    eta = 0.001
    delta = alpha**3
    N_g = 10
    
    print(f"Test point x0: {x0}")
    print(f"α = {alpha}")
    print(f"T = {T}, D = {D}, η = {eta}")
    print(f"δ = {delta:.6f}, N_g = {N_g}")
    print()
    
    # Test Algorithm 2 optimization
    results = algorithm2.optimize(x0, T, D, eta, delta, alpha, N_g)
    
    # Check convergence
    grad_norms = results['grad_norms']
    print(f"Gradient norms history (first 10): {[f'{g:.1f}' for g in grad_norms[:10]]}")
    print(f"Gradient norms history (last 10): {[f'{g:.1f}' for g in grad_norms[-10:]]}")
    
    # Check for stabilization
    if len(grad_norms) >= 10:
        last_10_norms = np.array(grad_norms[-10:])
        std_dev = np.std(last_10_norms)
        mean_norm = np.mean(last_10_norms)
        print(f"Last 10 gradient norms std dev: {std_dev:.4f}")
        print(f"Last 10 gradient norms range: {np.max(last_10_norms) - np.min(last_10_norms):.4f}")
        if std_dev < 5.0 and (np.max(last_10_norms) - np.min(last_10_norms)) < 20.0:
            print("Status: ✅ CONVERGED (based on heuristic)")
        else:
            print("Status: ❌ NOT CONVERGED (based on heuristic)")
    else:
        print("Not enough iterations to check stability.")
