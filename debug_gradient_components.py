#!/usr/bin/env python3
"""
Debug the actual gradient components to understand why they're so large
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final

def debug_gradient_components():
    """Debug the actual gradient components"""
    print("🔍 DEBUGGING GRADIENT COMPONENTS")
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
    
    print(f"Test point x: {x}")
    print(f"α = {alpha}")
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
    
    # Debug gradient components
    print("🔍 GRADIENT COMPONENTS ANALYSIS:")
    
    # Create computational graph
    x_grad = x.clone().detach().requires_grad_(True)
    
    # Compute constraint violations
    h_val = problem.constraints(x_grad, y_tilde)
    print(f"Constraint violations h(x,ỹ): {h_val}")
    
    # Compute smooth activation
    rho = algorithm._compute_smooth_activation(h_val, lambda_star, delta)
    print(f"Smooth activation ρ: {rho}")
    
    # Debug g(x,ỹ) gradient
    print("\n📊 g(x,ỹ) GRADIENT ANALYSIS:")
    g_xy = problem.lower_objective(x_grad, y_tilde)
    try:
        grad_g_xy = torch.autograd.grad(g_xy, x_grad, create_graph=True, retain_graph=True)[0]
        grad_g_xy_norm = torch.norm(grad_g_xy).item()
        print(f"  g(x,ỹ) gradient norm: {grad_g_xy_norm:.6f}")
        print(f"  g(x,ỹ) gradient: {grad_g_xy}")
    except Exception as e:
        print(f"  ERROR computing g(x,ỹ) gradient: {e}")
    
    # Debug g(x,y*) gradient
    print("\n📊 g(x,y*) GRADIENT ANALYSIS:")
    g_ystar = problem.lower_objective(x_grad, y_star)
    try:
        grad_g_ystar = torch.autograd.grad(g_ystar, x_grad, create_graph=True, retain_graph=True)[0]
        grad_g_ystar_norm = torch.norm(grad_g_ystar).item()
        print(f"  g(x,y*) gradient norm: {grad_g_ystar_norm:.6f}")
        print(f"  g(x,y*) gradient: {grad_g_ystar}")
    except Exception as e:
        print(f"  ERROR computing g(x,y*) gradient: {e}")
    
    # Debug λ̃^T h(x,ỹ) gradient
    print("\n📊 λ̃^T h(x,ỹ) GRADIENT ANALYSIS:")
    lambda_h = torch.sum(lambda_star * h_val)
    try:
        grad_lambda_h = torch.autograd.grad(lambda_h, x_grad, create_graph=True, retain_graph=True)[0]
        grad_lambda_h_norm = torch.norm(grad_lambda_h).item()
        print(f"  λ̃^T h(x,ỹ) gradient norm: {grad_lambda_h_norm:.6f}")
        print(f"  λ̃^T h(x,ỹ) gradient: {grad_lambda_h}")
    except Exception as e:
        print(f"  ERROR computing λ̃^T h(x,ỹ) gradient: {e}")
    
    # Debug the difference gradient
    print("\n📊 DIFFERENCE GRADIENT ANALYSIS:")
    diff = g_xy + lambda_h - g_ystar
    try:
        grad_diff = torch.autograd.grad(diff, x_grad, create_graph=True, retain_graph=True)[0]
        grad_diff_norm = torch.norm(grad_diff).item()
        print(f"  (g(x,ỹ) + λ̃^T h(x,ỹ) - g(x,y*)) gradient norm: {grad_diff_norm:.6f}")
        print(f"  (g(x,ỹ) + λ̃^T h(x,ỹ) - g(x,y*)) gradient: {grad_diff}")
        
        # This is the key insight - this gradient norm should be small!
        if grad_diff_norm < 1.0:
            print(f"  ✅ Difference gradient is small - this is good!")
        else:
            print(f"  ❌ Difference gradient is large - this is the problem!")
    except Exception as e:
        print(f"  ERROR computing difference gradient: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    debug_gradient_components()
