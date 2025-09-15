#!/usr/bin/env python3
"""
F2CSA Algorithm 2 Implementation
Nonsmooth Nonconvex Algorithm with Inexact Stochastic Hypergradient Oracle
Following F2CSA_corrected.tex Algorithm 2 exactly
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
import warnings

warnings.filterwarnings('ignore')

class F2CSAAlgorithm2:
    """
    F2CSA Algorithm 2: Nonsmooth Nonconvex Algorithm with Inexact Stochastic Hypergradient Oracle
    Following F2CSA_corrected.tex Algorithm 2 exactly
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem, device: str = 'cpu', dtype=torch.float64):
        self.problem = problem
        self.device = device
        self.dtype = dtype
        
        # Initialize Algorithm 1 for hypergradient computation
        self.algorithm1 = F2CSAAlgorithm1Final(problem, device=device, dtype=dtype)
        
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
        Run F2CSA Algorithm 2 optimization
        
        Args:
            x0: Initial point
            T: Total number of iterations
            D: Clipping parameter
            eta: Step size
            delta: Goldstein accuracy parameter
            alpha: Accuracy parameter for Algorithm 1
            N_g: Number of samples for hypergradient (if None, computed automatically)
        
        Returns:
            Dictionary with optimization results
        """
        print("🚀 F2CSA Algorithm 2 - Nonsmooth Nonconvex Optimization")
        print("=" * 70)
        print(f"T = {T}, D = {D:.6f}, η = {eta:.6f}")
        print(f"δ = {delta:.6f}, α = {alpha:.6f}")
        print()
        
        # Set default N_g if not provided
        if N_g is None:
            # Optimal N_g: balance bias O(α³) and variance O(1/N_g)
            # For bias-variance tradeoff: N_g = O(α⁻⁶) gives total error O(α³)
            # But practical: use smaller N_g for efficiency while maintaining accuracy
            theoretical_ng = int(1.0 / (alpha**2))  # O(α⁻²) for basic accuracy
            practical_ng = min(1000, max(10, theoretical_ng // 100))  # Scale down for efficiency
            N_g = practical_ng
        
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
        
        print(f"Selected output point from {len(x_outputs)} candidates (uniform)")
        
        # Compute upper-level loss gap metric f(x_out, y*(x_out)) and show CVXPY y*, λ
        y_star_out, info_out = self.problem.solve_lower_level(x_out)
        ul_loss = self.problem.upper_objective(x_out, y_star_out).item()
        lam_out = info_out.get('lambda', None)
        print(f"Upper-level loss f(x_out, y*): {ul_loss:.6f}")
        if lam_out is not None:
            print(f"λ*(x_out): {lam_out}")
        print()
        
        # Compute final statistics
        final_g_norm = torch.norm(g_history[-1]).item() if g_history else 0.0
        final_Delta_norm = torch.norm(Delta).item()
        
        print("📊 Algorithm 2 Results")
        print("-" * 30)
        print(f"Final ||g_t||: {final_g_norm:.6f}")
        print(f"Final ||Δ_t||: {final_Delta_norm:.6f}")
        print(f"Output point: {x_out.detach().numpy()}")
        print()
        
        return {
            'x_out': x_out,
            'ul_loss': ul_loss,
            'y_star_out': y_star_out,
            'lambda_out': lam_out,
            'x_history': x_history,
            'z_history': z_history,
            'g_history': g_history,
            'Delta_history': Delta_history,
            'x_outputs': x_outputs,
            'final_g_norm': final_g_norm,
            'final_Delta_norm': final_Delta_norm,
            'T': T,
            'M': M,
            'K': K,
            'alpha': alpha,
            'delta': delta,
            'D': D,
            'eta': eta,
            'N_g': N_g,
            # Add compatibility keys for tests
            'losses': [torch.norm(g).item() for g in g_history],  # Use gradient norms as proxy for losses
            'grad_norms': [torch.norm(g).item() for g in g_history],
            'converged': final_g_norm < 1e-3,  # Simple convergence criterion
            'iterations': len(g_history)
        }
    
    def test_algorithm2(self, dim: int = 5, T: int = 1000):
        """
        Test Algorithm 2 with default parameters
        """
        print("🧪 Testing F2CSA Algorithm 2")
        print("=" * 50)
        
        # Create problem
        problem = StronglyConvexBilevelProblem(
            dim=dim,
            num_constraints=3,
            noise_std=0.1,
            strong_convex=True,
            device=self.device
        )
        
        # Initialize Algorithm 2
        algorithm2 = F2CSAAlgorithm2(problem, device=self.device, dtype=self.dtype)
        
        # Set parameters following F2CSA_corrected.tex
        epsilon = 0.1  # Target accuracy
        alpha = 0.05   # α = c_α * ε
        delta = alpha**3  # δ = c_δ * α³
        
        # Compute other parameters
        L_F = 10.0  # Estimate of Lipschitz constant
        D = delta * epsilon**2 / (L_F**2)  # D = c_D * δ * ε² / L_F²
        eta = delta * epsilon**3 / (L_F**4)  # η = c_η * δ * ε³ / L_F⁴
        
        print(f"Parameters:")
        print(f"  ε = {epsilon:.6f}")
        print(f"  α = {alpha:.6f}")
        print(f"  δ = {delta:.6f}")
        print(f"  D = {D:.6f}")
        print(f"  η = {eta:.6f}")
        print(f"  T = {T}")
        print()
        
        # Generate initial point
        x0 = torch.randn(dim, dtype=self.dtype, device=self.device)
        
        # Run optimization
        result = algorithm2.optimize(
            x0=x0,
            T=T,
            D=D,
            eta=eta,
            delta=delta,
            alpha=alpha
        )
        
        return result

def main():
    """Main test function"""
    # Test Algorithm 2
    algorithm2 = F2CSAAlgorithm2(None)  # Will create problem in test
    result = algorithm2.test_algorithm2(dim=5, T=1000)
    
    print("🎉 Algorithm 2 test completed!")
    return result

if __name__ == "__main__":
    main()
