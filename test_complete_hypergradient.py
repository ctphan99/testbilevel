#!/usr/bin/env python3
"""
Test the complete hypergradient computation including upper-level objective
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final

def test_complete_hypergradient():
    """Test the complete hypergradient computation"""
    print("🔍 TESTING COMPLETE HYPERGRADIENT COMPUTATION")
    print("=" * 60)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5, 
        num_constraints=3, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    # Create algorithm
    algorithm = F2CSAAlgorithm1Final(problem)
    
    # Test with α = 0.1
    alpha = 0.1
    x = torch.randn(5, dtype=torch.float64)
    N_g = 10
    
    print(f"Test point x: {x}")
    print(f"α = {alpha}")
    print(f"N_g = {N_g}")
    print()
    
    # Get accurate lower-level solution
    y_star, lambda_star, info = algorithm._solve_lower_level_accurate(x, alpha)
    delta = alpha**3
    y_tilde = algorithm._minimize_penalty_lagrangian(x, y_star, lambda_star, alpha, delta)
    
    print(f"y*: {y_star}")
    print(f"ỹ: {y_tilde}")
    print(f"λ*: {lambda_star}")
    print(f"Gap: {torch.norm(y_tilde - y_star).item():.8f}")
    print()
    
    # Test individual hypergradient samples
    print("🔍 INDIVIDUAL HYPERGRADIENT SAMPLES:")
    hypergradient_samples = []
    
    for j in range(N_g):
        print(f"  Sample {j+1}/{N_g}:")
        
        # Sample fresh noise
        noise_upper, _ = problem._sample_instance_noise()
        
        # Create computational graph
        x_grad = x.clone().detach().requires_grad_(True)
        
        # Compute penalty Lagrangian
        L_val = algorithm._compute_penalty_lagrangian(x_grad, y_tilde, y_star, lambda_star, alpha, delta)
        print(f"    Penalty Lagrangian: {L_val.item():.8f}")
        
        # Add upper-level objective
        f_val = problem.upper_objective(x_grad, y_tilde, noise_upper=noise_upper)
        print(f"    Upper objective: {f_val.item():.8f}")
        
        total_val = f_val + L_val
        print(f"    Total objective: {total_val.item():.8f}")
        
        # Compute gradient
        try:
            grad_x = torch.autograd.grad(total_val, x_grad, create_graph=True, retain_graph=True)[0]
            grad_norm = torch.norm(grad_x).item()
            print(f"    Gradient norm: {grad_norm:.6f}")
            print(f"    Gradient: {grad_x}")
            hypergradient_samples.append(grad_x.detach())
        except Exception as e:
            print(f"    ERROR computing gradient: {e}")
            continue
        
        print()
    
    # Test final hypergradient
    if hypergradient_samples:
        print("🔍 FINAL HYPERGRADIENT:")
        hypergradient = torch.stack(hypergradient_samples).mean(dim=0)
        final_norm = torch.norm(hypergradient).item()
        print(f"  Final hypergradient norm: {final_norm:.6f}")
        print(f"  Final hypergradient: {hypergradient}")
        
        if final_norm < 10:
            print(f"  ✅ SUCCESS: Hypergradient norm is reasonable!")
        elif final_norm < 50:
            print(f"  ⚠️  Hypergradient norm is manageable")
        else:
            print(f"  ❌ Hypergradient norm is too large!")
    else:
        print("❌ ERROR: No valid hypergradient samples!")
    
    print("=" * 60)

if __name__ == "__main__":
    test_complete_hypergradient()
