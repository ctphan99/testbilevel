#!/usr/bin/env python3
"""
Test different penalty parameters to see if we can get better convergence
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
import warnings

warnings.filterwarnings('ignore')

def test_penalty_parameters():
    """Test different penalty parameters"""
    print("🔧 PENALTY PARAMETERS TEST")
    print("=" * 50)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5, num_constraints=3, noise_std=0.1, 
        strong_convex=True, device='cpu'
    )
    
    algorithm = F2CSAAlgorithm1Final(problem)
    
    # Test with a specific point
    x = torch.randn(5, dtype=torch.float64)
    alpha = 0.05
    
    print(f"Testing with x = {x}")
    print(f"α = {alpha}")
    print()
    
    # Get accurate lower-level solution
    y_star, info = problem.solve_lower_level(x, 'accurate', 1000, 1e-8, alpha)
    lambda_star = info.get('lambda_star', torch.zeros(3, dtype=torch.float64))
    h_star = problem.constraints(x, y_star)
    
    print(f"y* = {y_star}")
    print(f"λ* = {lambda_star}")
    print(f"h(x,y*) = {h_star}")
    print(f"||h(x,y*)|| = {torch.norm(h_star).item():.8f}")
    print()
    
    # Test different penalty parameter combinations
    penalty_configs = [
        ("Original F2CSA", alpha**(-2), alpha**(-4)),
        ("Modified F2CSA", alpha**(-1), alpha**(-2)),
        ("Smaller penalties", alpha**(-0.5), alpha**(-1)),
        ("Much smaller", alpha**(0), alpha**(0)),
        ("Very small", alpha**(0.5), alpha**(1)),
    ]
    
    print("Testing different penalty parameter combinations:")
    print("=" * 60)
    
    for name, alpha1, alpha2 in penalty_configs:
        print(f"\n{name}: α₁ = {alpha1:.6f}, α₂ = {alpha2:.6f}")
        
        # Check gradient norm at y*
        y_test = y_star.clone().detach().requires_grad_(True)
        L_test = algorithm._compute_penalty_lagrangian(x, y_test, y_star, lambda_star, alpha, alpha**3)
        L_test.backward()
        grad_norm = torch.norm(y_test.grad).item()
        print(f"  Gradient norm at y*: {grad_norm:.6f}")
        
        # Try to minimize penalty Lagrangian
        y_penalty = y_star.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([y_penalty], lr=0.001)
        
        for iteration in range(1000):
            optimizer.zero_grad()
            L_val = algorithm._compute_penalty_lagrangian(x, y_penalty, y_star, lambda_star, alpha, alpha**3)
            L_val.backward()
            optimizer.step()
            
            if iteration % 200 == 0:
                gap = torch.norm(y_penalty - y_star).item()
                h_val = problem.constraints(x, y_penalty)
                print(f"    Iter {iteration}: gap={gap:.6f}, ||h||={torch.norm(h_val).item():.6f}")
        
        final_gap = torch.norm(y_penalty.detach() - y_star).item()
        final_h = problem.constraints(x, y_penalty.detach())
        final_h_norm = torch.norm(final_h).item()
        
        print(f"  Final gap: {final_gap:.6f}")
        print(f"  Final ||h||: {final_h_norm:.6f}")
        print(f"  Status: {'✅' if final_gap < 0.1 else '❌'}")
    
    # Test with the original penalty parameters but different alpha values
    print(f"\nTesting different alpha values with modified F2CSA parameters:")
    print("=" * 60)
    
    alpha_values = [0.1, 0.05, 0.01, 0.005, 0.001]
    
    for alpha_test in alpha_values:
        print(f"\nα = {alpha_test}")
        alpha1 = alpha_test**(-1)
        alpha2 = alpha_test**(-2)
        delta = alpha_test**3
        
        print(f"  α₁ = {alpha1:.6f}, α₂ = {alpha2:.6f}, δ = {delta:.6f}")
        
        # Try to minimize penalty Lagrangian
        y_penalty = y_star.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([y_penalty], lr=0.001)
        
        for iteration in range(1000):
            optimizer.zero_grad()
            L_val = algorithm._compute_penalty_lagrangian(x, y_penalty, y_star, lambda_star, alpha_test, delta)
            L_val.backward()
            optimizer.step()
            
            if iteration % 200 == 0:
                gap = torch.norm(y_penalty - y_star).item()
                h_val = problem.constraints(x, y_penalty)
                print(f"    Iter {iteration}: gap={gap:.6f}, ||h||={torch.norm(h_val).item():.6f}")
        
        final_gap = torch.norm(y_penalty.detach() - y_star).item()
        final_h = problem.constraints(x, y_penalty.detach())
        final_h_norm = torch.norm(final_h).item()
        
        print(f"  Final gap: {final_gap:.6f}")
        print(f"  Final ||h||: {final_h_norm:.6f}")
        print(f"  Status: {'✅' if final_gap < 0.1 else '❌'}")

if __name__ == "__main__":
    test_penalty_parameters()
