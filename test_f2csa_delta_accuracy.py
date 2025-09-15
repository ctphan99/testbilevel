#!/usr/bin/env python3
"""
Test F2CSA Algorithm 1 to achieve δ-accuracy < 0.1
"""

import torch
import numpy as np
from f2csa_algorithm_accurate import F2CSAAlgorithm1
from problem import StronglyConvexBilevelProblem

def test_delta_accuracy():
    """Test F2CSA Algorithm 1 to achieve δ-accuracy < 0.1"""
    print("Testing F2CSA Algorithm 1 for δ-accuracy < 0.1")
    print("=" * 60)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.01, strong_convex=True)
    
    # Initialize algorithm
    algorithm = F2CSAAlgorithm1(problem)
    
    # Test point
    x0 = torch.randn(5, device=problem.device, dtype=problem.dtype) * 0.1
    
    # Test different alpha values to achieve δ < 0.1
    # We need α³ < 0.1, so α < 0.1^(1/3) ≈ 0.464
    alpha_values = [0.4, 0.35, 0.3, 0.25, 0.2]
    N_g = 10
    lr = 1e-3
    max_iterations = 15
    
    print(f"Test point: x0 = {x0}")
    print(f"Target: δ = α³ < 0.1")
    print()
    
    for alpha in alpha_values:
        delta = alpha**3
        print(f"--- Testing α = {alpha} (δ = {delta:.6f}) ---")
        
        if delta >= 0.1:
            print(f"  ❌ δ >= 0.1: {delta:.6f} >= 0.1")
            continue
        
        print(f"  ✅ δ < 0.1: {delta:.6f} < 0.1")
        print(f"  α₁ = α⁻² = {alpha**(-2):.1f}")
        print(f"  α₂ = α⁻⁴ = {alpha**(-4):.1f}")
        
        try:
            # Test single oracle call
            hypergradient = algorithm.oracle_sample(x0, alpha, N_g)
            grad_norm = torch.norm(hypergradient).item()
            print(f"  Single oracle gradient norm: {grad_norm:.6f}")
            
            if grad_norm > 100:  # Too large
                print(f"  ⚠️ Gradient norm too large: {grad_norm:.6f}")
                continue
            
            # Test short optimization
            result = algorithm.optimize(x0, max_iterations=max_iterations, alpha=alpha, N_g=N_g, lr=lr)
            
            print(f"  Final loss: {result['losses'][-1]:.6f}")
            print(f"  Final gradient norm: {result['grad_norms'][-1]:.6f}")
            print(f"  Loss reduction: {result['losses'][0] - result['losses'][-1]:.6f}")
            print(f"  Relative improvement: {((result['losses'][0] - result['losses'][-1]) / abs(result['losses'][0])) * 100:.2f}%")
            
            if result['grad_norms'][-1] < 10:  # Good gradient norm
                print(f"  ✅ Good performance with δ-accuracy < 0.1")
                return alpha, result
            else:
                print(f"  ⚠️ Moderate performance")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print()
    
    print("No suitable α found for δ-accuracy < 0.1 with good performance")
    return None, None

if __name__ == "__main__":
    test_delta_accuracy()
