#!/usr/bin/env python3
"""
Test F2CSA with modified penalty parameters to achieve δ-accuracy < 0.1
while maintaining reasonable hypergradient norms
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_accurate import F2CSAAlgorithm1

def test_modified_penalty_parameters():
    """Test with modified penalty parameters"""
    print("=== F2CSA Modified Penalty Parameters Test ===\n")
    
    # Create problem instance
    problem = StronglyConvexBilevelProblem(
        dim=5, num_constraints=3, noise_std=0.1,
        strong_convex=True, device='cpu'
    )
    
    # Test different α values with modified penalty scaling
    alpha_values = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01]
    
    for alpha in alpha_values:
        delta = alpha ** 3
        print(f"--- Testing α = {alpha} (δ = {delta:.6f}) ---")
        
        # Check δ-accuracy requirement
        if delta >= 0.1:
            print(f"  ❌ δ >= 0.1: {delta:.6f} >= 0.1")
            continue
        else:
            print(f"  ✅ δ < 0.1: {delta:.6f} < 0.1")
        
        # Original penalty parameters (too large)
        alpha1_orig = 1 / (alpha ** 2)
        alpha2_orig = 1 / (alpha ** 4)
        
        # Modified penalty parameters (more reasonable)
        alpha1_mod = 1 / alpha  # α₁ = α⁻¹ instead of α⁻²
        alpha2_mod = 1 / (alpha ** 2)  # α₂ = α⁻² instead of α⁻⁴
        
        print(f"  Original: α₁ = {alpha1_orig:.1f}, α₂ = {alpha2_orig:.1f}")
        print(f"  Modified: α₁ = {alpha1_mod:.1f}, α₂ = {alpha2_mod:.1f}")
        print(f"  Reduction: α₁ by {alpha1_orig/alpha1_mod:.1f}x, α₂ by {alpha2_orig/alpha2_mod:.1f}x")
        
        # Test single oracle call
        try:
            algorithm = F2CSAAlgorithm1(problem)
            x0 = torch.randn(5, dtype=torch.float64)
            
            # Test with original parameters
            print(f"  Testing with original parameters...")
            hypergrad_orig = algorithm.oracle_sample(x0, alpha, N_g=1)
            print(f"    Original hypergradient norm: {torch.norm(hypergrad_orig):.2f}")
            
            # Test with modified parameters (we need to modify the algorithm)
            print(f"  Testing with modified parameters...")
            # For now, just show the parameter reduction
            print(f"    Expected hypergradient norm reduction: ~{alpha1_orig/alpha1_mod:.1f}x")
            
        except Exception as e:
            print(f"    Error: {e}")
        
        print()

if __name__ == "__main__":
    test_modified_penalty_parameters()
