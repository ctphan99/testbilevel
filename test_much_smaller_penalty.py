#!/usr/bin/env python3
"""
Test with MUCH smaller penalty parameters to achieve convergence
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm1_practical import F2CSAAlgorithm1Practical

def test_much_smaller_penalty_parameters():
    """Test with much smaller penalty parameters"""
    print("🔧 TESTING MUCH SMALLER PENALTY PARAMETERS")
    print("=" * 60)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5, 
        num_constraints=3, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    # Create algorithm
    algorithm = F2CSAAlgorithm1Practical(problem)
    
    # Test with much smaller alpha
    alpha_values = [0.1, 0.05, 0.01, 0.005, 0.001]
    x = torch.randn(5, dtype=torch.float64)
    N_g = 10
    
    print(f"Test point x: {x}")
    print()
    
    for alpha in alpha_values:
        print(f"--- Testing α = {alpha} ---")
        
        # Test hypergradient computation
        hypergradient = algorithm.oracle_sample(x, alpha, N_g)
        hypergradient_norm = torch.norm(hypergradient).item()
        
        print(f"  Hypergradient norm: {hypergradient_norm:.6f}")
        
        if hypergradient_norm < 10:
            print(f"  ✅ SUCCESS: Gradient norm is very small!")
            print(f"  This α = {alpha} should work for Algorithm 2 convergence")
            break
        else:
            print(f"  ❌ Still too large")
        print()
    
    print("=" * 60)

if __name__ == "__main__":
    test_much_smaller_penalty_parameters()
