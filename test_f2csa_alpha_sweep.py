#!/usr/bin/env python3
"""
Test F2CSA Algorithm 1 with different alpha values to find optimal parameters
"""

import torch
import numpy as np
from f2csa_algorithm_accurate import F2CSAAlgorithm1
from problem import StronglyConvexBilevelProblem

def test_alpha_values():
    """Test different alpha values to find optimal parameters"""
    print("Testing F2CSA Algorithm 1 with different alpha values")
    print("=" * 60)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.01, strong_convex=True)
    
    # Initialize algorithm
    algorithm = F2CSAAlgorithm1(problem)
    
    # Test point
    x0 = torch.randn(5, device=problem.device, dtype=problem.dtype) * 0.1
    
    # Test different alpha values
    alpha_values = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01]
    N_g = 5
    
    print(f"Test point: x0 = {x0}")
    print(f"Batch size: N_g = {N_g}")
    print()
    
    for alpha in alpha_values:
        print(f"--- Testing α = {alpha} ---")
        print(f"  α₁ = α⁻² = {alpha**(-2):.1f}")
        print(f"  α₂ = α⁻⁴ = {alpha**(-4):.1f}")
        print(f"  δ = α³ = {alpha**3:.2e}")
        
        try:
            # Test single oracle call
            hypergradient = algorithm.oracle_sample(x0, alpha, N_g)
            grad_norm = torch.norm(hypergradient).item()
            
            print(f"  Hypergradient norm: ||∇F̃|| = {grad_norm:.6f}")
            
            # Test short optimization
            result = algorithm.optimize(x0, max_iterations=5, alpha=alpha, N_g=N_g, lr=1e-3)
            
            print(f"  Final loss: {result['losses'][-1]:.6f}")
            print(f"  Final grad norm: {result['grad_norms'][-1]:.6f}")
            print(f"  Loss reduction: {result['losses'][0] - result['losses'][-1]:.6f}")
            
            if grad_norm < 50:  # Reasonable gradient norm
                print(f"  ✅ Good gradient norm")
            else:
                print(f"  ⚠️ Large gradient norm")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print()

if __name__ == "__main__":
    test_alpha_values()
