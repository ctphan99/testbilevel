#!/usr/bin/env python3
"""
Debug upper-level objective computation - the issue is likely there
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final

def debug_upper_objective():
    """Debug upper-level objective computation"""
    print("🔍 DEBUGGING UPPER-LEVEL OBJECTIVE COMPUTATION")
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
    
    print(f"Test point x: {x}")
    print(f"α = {alpha}")
    print()
    
    # Get accurate lower-level solution
    y_star, lambda_star, info = algorithm._solve_lower_level_accurate(x, alpha)
    delta = alpha**3
    y_tilde = algorithm._minimize_penalty_lagrangian(x, y_star, lambda_star, alpha, delta)
    
    print(f"y*: {y_star}")
    print(f"ỹ: {y_tilde}")
    print()
    
    # Debug upper-level objective
    print("🔍 DEBUGGING UPPER-LEVEL OBJECTIVE")
    
    # Sample noise
    noise_upper, _ = problem._sample_instance_noise()
    print(f"Noise upper shape: {noise_upper.shape}")
    print(f"Noise upper norm: {torch.norm(noise_upper).item():.6f}")
    
    # Create computational graph
    x_grad = x.clone().detach().requires_grad_(True)
    
    # Compute upper-level objective
    f_val = problem.upper_objective(x_grad, y_tilde, noise_upper=noise_upper)
    print(f"Upper objective: {f_val.item():.6f}")
    
    # Compute gradient of upper objective only
    try:
        grad_f = torch.autograd.grad(f_val, x_grad, create_graph=True, retain_graph=True)[0]
        grad_f_norm = torch.norm(grad_f).item()
        print(f"Upper objective gradient norm: {grad_f_norm:.6f}")
        print(f"Upper objective gradient: {grad_f}")
    except Exception as e:
        print(f"❌ ERROR computing upper objective gradient: {e}")
    
    # Compute penalty Lagrangian
    L_val = algorithm._compute_penalty_lagrangian(x_grad, y_tilde, y_star, lambda_star, alpha, delta)
    print(f"Penalty Lagrangian: {L_val.item():.6f}")
    
    # Compute gradient of penalty Lagrangian only
    try:
        grad_L = torch.autograd.grad(L_val, x_grad, create_graph=True, retain_graph=True)[0]
        grad_L_norm = torch.norm(grad_L).item()
        print(f"Penalty Lagrangian gradient norm: {grad_L_norm:.6f}")
        print(f"Penalty Lagrangian gradient: {grad_L}")
    except Exception as e:
        print(f"❌ ERROR computing penalty Lagrangian gradient: {e}")
    
    # Compute total gradient
    total_val = f_val + L_val
    print(f"Total objective: {total_val.item():.6f}")
    
    try:
        grad_total = torch.autograd.grad(total_val, x_grad, create_graph=True, retain_graph=True)[0]
        grad_total_norm = torch.norm(grad_total).item()
        print(f"Total gradient norm: {grad_total_norm:.6f}")
        print(f"Total gradient: {grad_total}")
        
        # Check which component dominates
        if grad_f_norm > grad_L_norm:
            print(f"🔍 Upper objective gradient dominates: {grad_f_norm:.2f} vs {grad_L_norm:.2f}")
        else:
            print(f"🔍 Penalty Lagrangian gradient dominates: {grad_L_norm:.2f} vs {grad_f_norm:.2f}")
            
    except Exception as e:
        print(f"❌ ERROR computing total gradient: {e}")
    
    print("=" * 60)

if __name__ == "__main__":
    debug_upper_objective()
