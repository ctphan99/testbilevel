#!/usr/bin/env python3
"""
Fix the penalty Lagrangian minimization to ensure it converges to the correct solution
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
import warnings

warnings.filterwarnings('ignore')

def test_penalty_fix():
    """Test if we can fix the penalty Lagrangian minimization"""
    print("🔧 PENALTY LAGRANGIAN FIX TEST")
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
    delta = alpha**3
    
    print(f"Testing with x = {x}")
    print(f"α = {alpha}, δ = {delta}")
    print()
    
    # 1. Get accurate lower-level solution
    print("1️⃣ Getting accurate solution y*")
    y_star, info = problem.solve_lower_level(x, 'accurate', 1000, 1e-8, alpha)
    lambda_star = info.get('lambda_star', torch.zeros(3, dtype=torch.float64))
    
    print(f"   y* = {y_star}")
    print(f"   λ* = {lambda_star}")
    h_star = problem.constraints(x, y_star)
    print(f"   h(x,y*) = {h_star}")
    print(f"   ||h(x,y*)|| = {torch.norm(h_star).item():.8f}")
    print()
    
    # 2. Test different penalty solver approaches
    print("2️⃣ Testing different penalty solver approaches")
    
    # Approach 1: Start from y* and use very small learning rate
    print("   Approach 1: Start from y*, small LR")
    y1 = y_star.clone().detach().requires_grad_(True)
    optimizer1 = torch.optim.Adam([y1], lr=0.0001)
    
    for iteration in range(2000):
        optimizer1.zero_grad()
        L_val = algorithm._compute_penalty_lagrangian(x, y1, y_star, lambda_star, alpha, delta)
        L_val.backward()
        optimizer1.step()
        
        if iteration % 400 == 0:
            h_val = problem.constraints(x, y1)
            gap = torch.norm(y1 - y_star).item()
            print(f"     Iter {iteration}: L={L_val.item():.8f}, gap={gap:.8f}, ||h||={torch.norm(h_val).item():.8f}")
    
    h1 = problem.constraints(x, y1.detach())
    gap1 = torch.norm(y1.detach() - y_star).item()
    print(f"   Final gap: {gap1:.8f}, ||h||: {torch.norm(h1).item():.8f}")
    print()
    
    # Approach 2: Use LBFGS optimizer
    print("   Approach 2: LBFGS optimizer")
    y2 = y_star.clone().detach().requires_grad_(True)
    optimizer2 = torch.optim.LBFGS([y2], lr=0.1, max_iter=20)
    
    def closure():
        optimizer2.zero_grad()
        L_val = algorithm._compute_penalty_lagrangian(x, y2, y_star, lambda_star, alpha, delta)
        L_val.backward()
        return L_val
    
    for iteration in range(100):
        optimizer2.step(closure)
        if iteration % 20 == 0:
            h_val = problem.constraints(x, y2)
            gap = torch.norm(y2 - y_star).item()
            print(f"     Iter {iteration}: gap={gap:.8f}, ||h||={torch.norm(h_val).item():.8f}")
    
    h2 = problem.constraints(x, y2.detach())
    gap2 = torch.norm(y2.detach() - y_star).item()
    print(f"   Final gap: {gap2:.8f}, ||h||: {torch.norm(h2).item():.8f}")
    print()
    
    # Approach 3: Use the original method but with better convergence criteria
    print("   Approach 3: Original method with better convergence")
    y3 = algorithm._minimize_penalty_lagrangian(x, y_star, lambda_star, alpha, delta)
    h3 = problem.constraints(x, y3)
    gap3 = torch.norm(y3 - y_star).item()
    print(f"   Final gap: {gap3:.8f}, ||h||: {torch.norm(h3).item():.8f}")
    print()
    
    # 3. Check if the issue is in the penalty Lagrangian computation
    print("3️⃣ Checking penalty Lagrangian computation")
    
    # Test penalty Lagrangian at y* and y~
    L_star = algorithm._compute_penalty_lagrangian(x, y_star, y_star, lambda_star, alpha, delta)
    L_tilde = algorithm._compute_penalty_lagrangian(x, y3, y_star, lambda_star, alpha, delta)
    
    print(f"   L(x, y*, y*, λ*) = {L_star:.8f}")
    print(f"   L(x, y~, y*, λ*) = {L_tilde:.8f}")
    print(f"   Difference: {abs(L_tilde - L_star).item():.8f}")
    
    # Check if y* is actually a minimum of the penalty Lagrangian
    print("   Checking if y* is a minimum...")
    y_test = y_star.clone().detach().requires_grad_(True)
    L_test = algorithm._compute_penalty_lagrangian(x, y_test, y_star, lambda_star, alpha, delta)
    L_test.backward()
    grad_norm = torch.norm(y_test.grad).item()
    print(f"   Gradient norm at y*: {grad_norm:.8f}")
    
    # 4. Summary
    print("4️⃣ SUMMARY")
    print("=" * 20)
    print(f"Target gap: < 0.1")
    print(f"Approach 1 gap: {gap1:.6f} {'✅' if gap1 < 0.1 else '❌'}")
    print(f"Approach 2 gap: {gap2:.6f} {'✅' if gap2 < 0.1 else '❌'}")
    print(f"Approach 3 gap: {gap3:.6f} {'✅' if gap3 < 0.1 else '❌'}")
    
    best_gap = min(gap1, gap2, gap3)
    best_approach = ["Approach 1", "Approach 2", "Approach 3"][np.argmin([gap1, gap2, gap3])]
    
    print(f"Best approach: {best_approach}")
    print(f"Best gap: {best_gap:.6f}")
    print(f"Status: {'✅ SUCCESS' if best_gap < 0.1 else '❌ FAILED'}")

if __name__ == "__main__":
    test_penalty_fix()
