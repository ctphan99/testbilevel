#!/usr/bin/env python3
"""
Debug penalty Lagrangian computation step by step to find the root cause
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm1_original import F2CSAAlgorithm1Original

def debug_penalty_lagrangian_step_by_step():
    """Debug penalty Lagrangian computation step by step"""
    print("🔍 DEBUGGING PENALTY LAGRANGIAN STEP BY STEP")
    print("=" * 60)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5, 
        num_constraints=3, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    # Create algorithm
    algorithm = F2CSAAlgorithm1Original(problem)
    
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
    print(f"λ*: {lambda_star}")
    print(f"Gap: {torch.norm(y_tilde - y_star).item():.6f}")
    print()
    
    # Debug penalty Lagrangian computation step by step
    print("🔍 DEBUGGING PENALTY LAGRANGIAN COMPUTATION")
    
    # Create computational graph
    x_grad = x.clone().detach().requires_grad_(True)
    
    # Check penalty parameters
    alpha1 = 1.0 / (alpha**2)  # α₁ = α⁻²
    alpha2 = 1.0 / (alpha**4)  # α₂ = α⁻⁴
    print(f"α₁ = {alpha1:.1f}")
    print(f"α₂ = {alpha2:.1f}")
    print()
    
    # Compute constraint violations
    h_val = problem.constraints(x_grad, y_tilde)
    print(f"Constraint violations h(x,ỹ): {h_val}")
    print(f"Max constraint violation: {torch.max(h_val).item():.6f}")
    print()
    
    # Compute smooth activation
    rho = algorithm._compute_smooth_activation(x_grad, alpha)
    print(f"Smooth activation ρ: {rho}")
    print()
    
    # Term 1: α₁ * (g(x,y) + λ̃^T h(x,y) - g(x,ỹ*(x)))
    g_xy = problem.lower_objective(x_grad, y_tilde)
    g_ystar = problem.lower_objective(x_grad, y_star)
    lambda_h = torch.sum(lambda_star * h_val)
    
    print(f"g(x,ỹ): {g_xy.item():.6f}")
    print(f"g(x,y*): {g_ystar.item():.6f}")
    print(f"λ̃^T h(x,ỹ): {lambda_h.item():.6f}")
    print(f"g(x,ỹ) + λ̃^T h(x,ỹ) - g(x,y*): {(g_xy + lambda_h - g_ystar).item():.6f}")
    
    term1 = alpha1 * (g_xy + lambda_h - g_ystar)
    print(f"Term 1: {term1.item():.6f}")
    print()
    
    # Term 2: α₂/2 * Σ_i ρ_i(x) * h_i(x,y)²
    h_squared = h_val ** 2
    rho_h_squared = rho * h_squared
    sum_rho_h_squared = torch.sum(rho_h_squared)
    
    print(f"h(x,ỹ)²: {h_squared}")
    print(f"ρ * h(x,ỹ)²: {rho_h_squared}")
    print(f"Σ ρ * h(x,ỹ)²: {sum_rho_h_squared.item():.6f}")
    
    term2 = (alpha2 / 2.0) * sum_rho_h_squared
    print(f"Term 2: {term2.item():.6f}")
    print()
    
    # Total penalty Lagrangian
    L_penalty = term1 + term2
    print(f"Total penalty Lagrangian: {L_penalty.item():.6f}")
    print()
    
    # Compute gradient
    try:
        grad_L = torch.autograd.grad(L_penalty, x_grad, create_graph=True, retain_graph=True)[0]
        grad_L_norm = torch.norm(grad_L).item()
        print(f"Penalty Lagrangian gradient norm: {grad_L_norm:.6f}")
        print(f"Penalty Lagrangian gradient: {grad_L}")
        
        # Check which term dominates the gradient
        grad_term1 = torch.autograd.grad(term1, x_grad, create_graph=True, retain_graph=True)[0]
        grad_term2 = torch.autograd.grad(term2, x_grad, create_graph=True, retain_graph=True)[0]
        
        grad_term1_norm = torch.norm(grad_term1).item()
        grad_term2_norm = torch.norm(grad_term2).item()
        
        print(f"Term 1 gradient norm: {grad_term1_norm:.6f}")
        print(f"Term 2 gradient norm: {grad_term2_norm:.6f}")
        
        if grad_term1_norm > grad_term2_norm:
            print(f"🔍 Term 1 gradient dominates: {grad_term1_norm:.2f} vs {grad_term2_norm:.2f}")
        else:
            print(f"🔍 Term 2 gradient dominates: {grad_term2_norm:.2f} vs {grad_term1_norm:.2f}")
            
    except Exception as e:
        print(f"❌ ERROR computing gradient: {e}")
    
    print("=" * 60)

if __name__ == "__main__":
    debug_penalty_lagrangian_step_by_step()
