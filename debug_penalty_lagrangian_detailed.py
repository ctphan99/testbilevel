#!/usr/bin/env python3
"""
Comprehensive debugging of penalty Lagrangian computation
Trace every step to find what causes gradient norm instability
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final

def debug_penalty_lagrangian_detailed():
    """Debug penalty Lagrangian computation step by step with detailed logging"""
    print("🔍 COMPREHENSIVE PENALTY LAGRANGIAN DEBUGGING")
    print("=" * 80)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5, 
        num_constraints=3, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    # Create algorithm
    algorithm = F2CSAAlgorithm1Final(problem)
    
    # Test with different alpha values
    alpha_values = [0.1, 0.05, 0.01, 0.005, 0.001]
    x = torch.randn(5, dtype=torch.float64)
    
    print(f"Test point x: {x}")
    print()
    
    for alpha in alpha_values:
        print(f"🔍 TESTING α = {alpha}")
        print("-" * 60)
        
        # Get accurate lower-level solution
        y_star, lambda_star, info = algorithm._solve_lower_level_accurate(x, alpha)
        delta = alpha**3
        y_tilde = algorithm._minimize_penalty_lagrangian(x, y_star, lambda_star, alpha, delta)
        
        print(f"y*: {y_star}")
        print(f"ỹ: {y_tilde}")
        print(f"λ*: {lambda_star}")
        print(f"Gap ||ỹ - y*||: {torch.norm(y_tilde - y_star).item():.8f}")
        print()
        
        # Debug penalty Lagrangian computation with detailed logging
        print("📊 DETAILED PENALTY LAGRANGIAN COMPUTATION:")
        
        # Create computational graph
        x_grad = x.clone().detach().requires_grad_(True)
        
        # Check penalty parameters
        alpha1 = 1.0 / alpha  # α₁ = α⁻¹ (modified)
        alpha2 = 1.0 / (alpha**2)  # α₂ = α⁻² (modified)
        print(f"  α₁ = {alpha1:.1f}")
        print(f"  α₂ = {alpha2:.1f}")
        print()
        
        # Compute constraint violations
        h_val = problem.constraints(x_grad, y_tilde)
        print(f"  Constraint violations h(x,ỹ): {h_val}")
        print(f"  Max constraint violation: {torch.max(h_val).item():.8f}")
        print(f"  Min constraint violation: {torch.min(h_val).item():.8f}")
        print()
        
        # Compute smooth activation
        rho = algorithm._compute_smooth_activation(h_val, lambda_star, delta)
        print(f"  Smooth activation ρ: {rho}")
        print()
        
        # Term 1: α₁ * (g(x,y) + λ̃^T h(x,y) - g(x,ỹ*(x)))
        print("  📈 TERM 1 COMPUTATION:")
        g_xy = problem.lower_objective(x_grad, y_tilde)
        g_ystar = problem.lower_objective(x_grad, y_star)
        lambda_h = torch.sum(lambda_star * h_val)
        
        print(f"    g(x,ỹ): {g_xy.item():.8f}")
        print(f"    g(x,y*): {g_ystar.item():.8f}")
        print(f"    λ̃^T h(x,ỹ): {lambda_h.item():.8f}")
        print(f"    g(x,ỹ) + λ̃^T h(x,ỹ) - g(x,y*): {(g_xy + lambda_h - g_ystar).item():.8f}")
        
        term1 = alpha1 * (g_xy + lambda_h - g_ystar)
        print(f"    α₁ * (g(x,ỹ) + λ̃^T h(x,ỹ) - g(x,y*)): {term1.item():.8f}")
        print()
        
        # Term 2: α₂/2 * Σ_i ρ_i(x) * h_i(x,y)²
        print("  📈 TERM 2 COMPUTATION:")
        h_squared = h_val ** 2
        rho_h_squared = rho * h_squared
        sum_rho_h_squared = torch.sum(rho_h_squared)
        
        print(f"    h(x,ỹ)²: {h_squared}")
        print(f"    ρ * h(x,ỹ)²: {rho_h_squared}")
        print(f"    Σ ρ * h(x,ỹ)²: {sum_rho_h_squared.item():.8f}")
        
        term2 = (alpha2 / 2.0) * sum_rho_h_squared
        print(f"    α₂/2 * Σ ρ * h(x,ỹ)²: {term2.item():.8f}")
        print()
        
        # Total penalty Lagrangian
        L_penalty = term1 + term2
        print(f"  📊 TOTAL PENALTY LAGRANGIAN: {L_penalty.item():.8f}")
        print()
        
        # Compute gradients for each term separately
        print("  🔍 GRADIENT ANALYSIS:")
        try:
            # Gradient of Term 1
            grad_term1 = torch.autograd.grad(term1, x_grad, create_graph=True, retain_graph=True)[0]
            grad_term1_norm = torch.norm(grad_term1).item()
            print(f"    Term 1 gradient norm: {grad_term1_norm:.8f}")
            print(f"    Term 1 gradient: {grad_term1}")
            
            # Gradient of Term 2
            grad_term2 = torch.autograd.grad(term2, x_grad, create_graph=True, retain_graph=True)[0]
            grad_term2_norm = torch.norm(grad_term2).item()
            print(f"    Term 2 gradient norm: {grad_term2_norm:.8f}")
            print(f"    Term 2 gradient: {grad_term2}")
            
            # Total gradient
            grad_total = torch.autograd.grad(L_penalty, x_grad, create_graph=True, retain_graph=True)[0]
            grad_total_norm = torch.norm(grad_total).item()
            print(f"    Total gradient norm: {grad_total_norm:.8f}")
            print(f"    Total gradient: {grad_total}")
            
            # Check which term dominates
            if grad_term1_norm > grad_term2_norm:
                ratio = grad_term1_norm / grad_term2_norm
                print(f"    🔍 Term 1 dominates by factor {ratio:.2f}")
            else:
                ratio = grad_term2_norm / grad_term1_norm
                print(f"    🔍 Term 2 dominates by factor {ratio:.2f}")
            
            # Check if gradient is reasonable
            if grad_total_norm < 10:
                print(f"    ✅ Gradient norm is reasonable!")
            elif grad_total_norm < 100:
                print(f"    ⚠️  Gradient norm is large but manageable")
            else:
                print(f"    ❌ Gradient norm is too large!")
                
        except Exception as e:
            print(f"    ❌ ERROR computing gradients: {e}")
        
        print()
        print("=" * 80)
        print()

if __name__ == "__main__":
    debug_penalty_lagrangian_detailed()
