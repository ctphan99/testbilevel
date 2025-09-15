#!/usr/bin/env python3
"""
Debug hypergradient computation in Algorithm 1 to find why Algorithm 2 diverges
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final

def debug_hypergradient_computation():
    """Debug hypergradient computation step by step"""
    print("🔍 DEBUGGING HYPERGRADIENT COMPUTATION")
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
    
    # Test parameters
    alpha = 0.1
    x = torch.randn(5, dtype=torch.float64)
    N_g = 10
    
    print(f"Test point x: {x}")
    print(f"α = {alpha}")
    print(f"N_g = {N_g}")
    print()
    
    # Step 1: Test accurate lower-level solution
    print("1️⃣ TESTING ACCURATE LOWER-LEVEL SOLUTION")
    y_star, lambda_star, info = algorithm._solve_lower_level_accurate(x, alpha)
    print(f"  y*: {y_star}")
    print(f"  λ*: {lambda_star}")
    print(f"  Info: {info}")
    print()
    
    # Step 2: Test penalty minimizer
    print("2️⃣ TESTING PENALTY MINIMIZER")
    delta = alpha**3
    y_tilde = algorithm._minimize_penalty_lagrangian(x, y_star, lambda_star, alpha, delta)
    print(f"  ỹ: {y_tilde}")
    print(f"  Gap ||ỹ - y*||: {torch.norm(y_tilde - y_star).item():.6f}")
    print()
    
    # Step 3: Test individual hypergradient samples
    print("3️⃣ TESTING INDIVIDUAL HYPERGRADIENT SAMPLES")
    hypergradient_samples = []
    
    for j in range(N_g):
        print(f"  Sample {j+1}/{N_g}:")
        
        # Sample fresh noise
        noise_upper, _ = problem._sample_instance_noise()
        print(f"    Noise upper: {noise_upper}")
        
        # Create computational graph
        x_grad = x.clone().detach().requires_grad_(True)
        
        # Compute penalty Lagrangian
        L_val = algorithm._compute_penalty_lagrangian(x_grad, y_tilde, y_star, lambda_star, alpha, delta)
        print(f"    Penalty Lagrangian: {L_val.item():.6f}")
        
        # Add upper-level objective
        f_val = problem.upper_objective(x_grad, y_tilde, noise_upper=noise_upper)
        print(f"    Upper objective: {f_val.item():.6f}")
        
        total_val = f_val + L_val
        print(f"    Total objective: {total_val.item():.6f}")
        
        # Compute gradient
        try:
            grad_x = torch.autograd.grad(total_val, x_grad, create_graph=True, retain_graph=True)[0]
            grad_norm = torch.norm(grad_x).item()
            print(f"    Gradient norm: {grad_norm:.6f}")
            hypergradient_samples.append(grad_x.detach())
        except Exception as e:
            print(f"    ERROR computing gradient: {e}")
            continue
    
    # Step 4: Test final hypergradient
    if hypergradient_samples:
        print("4️⃣ TESTING FINAL HYPERGRADIENT")
        hypergradient = torch.stack(hypergradient_samples).mean(dim=0)
        final_norm = torch.norm(hypergradient).item()
        print(f"  Final hypergradient norm: {final_norm:.6f}")
        print(f"  Hypergradient: {hypergradient}")
    else:
        print("4️⃣ ERROR: No valid hypergradient samples!")
    
    print("=" * 60)

if __name__ == "__main__":
    debug_hypergradient_computation()
