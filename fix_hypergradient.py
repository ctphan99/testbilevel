#!/usr/bin/env python3
"""
Fix hypergradient computation - the gradient norms are too large (~164)
Need to debug the penalty Lagrangian computation
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final

def fix_hypergradient_computation():
    """Fix the hypergradient computation to reduce gradient norms"""
    print("🔧 FIXING HYPERGRADIENT COMPUTATION")
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
    print(f"Gap: {torch.norm(y_tilde - y_star).item():.6f}")
    print()
    
    # Debug penalty Lagrangian computation
    print("🔍 DEBUGGING PENALTY LAGRANGIAN COMPUTATION")
    
    # Create computational graph
    x_grad = x.clone().detach().requires_grad_(True)
    
    # Compute penalty Lagrangian components
    L_val = algorithm._compute_penalty_lagrangian(x_grad, y_tilde, y_star, lambda_star, alpha, delta)
    print(f"Penalty Lagrangian: {L_val.item():.6f}")
    
    # Check if L_val is reasonable
    if abs(L_val.item()) > 1000:
        print("❌ PROBLEM: Penalty Lagrangian is too large!")
        print("This suggests the penalty parameters are too large")
        
        # Check penalty parameters
        alpha1 = 1.0 / alpha  # α₁ = α⁻¹
        alpha2 = 1.0 / (alpha**2)  # α₂ = α⁻²
        print(f"α₁ = {alpha1:.1f}")
        print(f"α₂ = {alpha2:.1f}")
        print("These are VERY large penalty parameters!")
        
        # Try with smaller penalty parameters
        print("\n🔧 TRYING WITH SMALLER PENALTY PARAMETERS")
        
        # Use original F2CSA parameters
        alpha1_orig = 1.0 / (alpha**2)  # α₁ = α⁻²
        alpha2_orig = 1.0 / (alpha**4)  # α₂ = α⁻⁴
        print(f"Original α₁ = {alpha1_orig:.1f}")
        print(f"Original α₂ = {alpha2_orig:.1f}")
        
        # Manually compute penalty Lagrangian with original parameters
        h_val = problem.constraints(x_grad, y_tilde)
        rho = algorithm._compute_smooth_activation(x_grad, alpha)
        
        # Term 1: α₁ * (g(x,y) + λ̃^T h(x,y) - g(x,ỹ*(x)))
        g_xy = problem.lower_objective(x_grad, y_tilde)
        g_ystar = problem.lower_objective(x_grad, y_star)
        term1 = alpha1_orig * (g_xy + torch.sum(lambda_star * h_val) - g_ystar)
        
        # Term 2: α₂/2 * Σ_i ρ_i(x) * h_i(x,y)²
        term2 = (alpha2_orig / 2.0) * torch.sum(rho * (h_val ** 2))
        
        L_val_orig = term1 + term2
        print(f"Penalty Lagrangian (original): {L_val_orig.item():.6f}")
        
        # Compute gradient with original parameters
        try:
            grad_x_orig = torch.autograd.grad(L_val_orig, x_grad, create_graph=True, retain_graph=True)[0]
            grad_norm_orig = torch.norm(grad_x_orig).item()
            print(f"Gradient norm (original): {grad_norm_orig:.6f}")
            
            if grad_norm_orig < 50:  # Much more reasonable
                print("✅ SUCCESS: Original parameters give much smaller gradients!")
                return True
            else:
                print("❌ Still too large with original parameters")
                return False
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return False
    else:
        print("✅ Penalty Lagrangian is reasonable")
        return True
    
    print("=" * 60)

if __name__ == "__main__":
    fix_hypergradient_computation()
