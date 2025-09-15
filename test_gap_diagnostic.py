#!/usr/bin/env python3
"""
Focused diagnostic to understand why the gap between y~ and y* is large
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
import warnings

warnings.filterwarnings('ignore')

def diagnose_gap_issue():
    """Diagnose why gap between y~ and y* is large"""
    print("🔍 GAP DIAGNOSTIC: Why is ||y~ - y*|| so large?")
    print("=" * 60)
    
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
    print("1️⃣ Computing accurate lower-level solution y*")
    y_star, info = problem.solve_lower_level(x, 'accurate', 1000, 1e-6, alpha)
    lambda_star = info.get('lambda_star', torch.zeros(3, dtype=torch.float64))
    
    print(f"   y* = {y_star}")
    print(f"   λ* = {lambda_star}")
    print(f"   Constraint violations: {info.get('constraint_violations', 'N/A')}")
    print()
    
    # 2. Get penalty Lagrangian solution
    print("2️⃣ Computing penalty Lagrangian solution y~")
    y_tilde = algorithm._minimize_penalty_lagrangian(x, y_star, lambda_star, alpha, delta)
    
    print(f"   y~ = {y_tilde}")
    print()
    
    # 3. Compute gap
    gap = torch.norm(y_tilde - y_star).item()
    print(f"3️⃣ Gap Analysis")
    print(f"   ||y~ - y*|| = {gap:.6f}")
    print(f"   Target: < 0.1")
    print(f"   Status: {'✅ PASS' if gap < 0.1 else '❌ FAIL'}")
    print()
    
    # 4. Analyze penalty Lagrangian values
    print("4️⃣ Penalty Lagrangian Analysis")
    L_star = algorithm._compute_penalty_lagrangian(x, y_star, y_star, lambda_star, alpha, delta)
    L_tilde = algorithm._compute_penalty_lagrangian(x, y_tilde, y_star, lambda_star, alpha, delta)
    
    print(f"   L(x, y*, y*, λ*) = {L_star:.6f}")
    print(f"   L(x, y~, y*, λ*) = {L_tilde:.6f}")
    print(f"   Difference: {abs(L_tilde - L_star):.6f}")
    print()
    
    # 5. Check constraint satisfaction
    print("5️⃣ Constraint Satisfaction Analysis")
    h_star = problem.constraints(x, y_star)
    h_tilde = problem.constraints(x, y_tilde)
    
    print(f"   h(x, y*) = {h_star}")
    print(f"   h(x, y~) = {h_tilde}")
    print(f"   ||h(x, y*)|| = {torch.norm(h_star).item():.6f}")
    print(f"   ||h(x, y~)|| = {torch.norm(h_tilde).item():.6f}")
    print()
    
    # 6. Check if penalty terms are the issue
    print("6️⃣ Penalty Term Analysis")
    alpha1 = alpha**(-1)  # α₁ = α⁻¹
    alpha2 = alpha**(-2)  # α₂ = α⁻²
    
    # Compute penalty terms
    rho_h_star = algorithm._compute_smooth_activation(h_star, lambda_star, alpha1)
    rho_lambda_star = algorithm._compute_smooth_activation(lambda_star, lambda_star, alpha2)
    
    rho_h_tilde = algorithm._compute_smooth_activation(h_tilde, lambda_star, alpha1)
    rho_lambda_tilde = algorithm._compute_smooth_activation(lambda_star, lambda_star, alpha2)  # Same λ*
    
    penalty_star = torch.sum(rho_h_star * lambda_star) + 0.5 * alpha1 * torch.sum(rho_h_star**2)
    penalty_tilde = torch.sum(rho_h_tilde * lambda_star) + 0.5 * alpha1 * torch.sum(rho_h_tilde**2)
    
    print(f"   α₁ = {alpha1:.6f}")
    print(f"   α₂ = {alpha2:.6f}")
    print(f"   ρ_h(y*) = {rho_h_star}")
    print(f"   ρ_h(y~) = {rho_h_tilde}")
    print(f"   Penalty(y*) = {penalty_star:.6f}")
    print(f"   Penalty(y~) = {penalty_tilde:.6f}")
    print()
    
    # 7. Check if the penalty solver is actually minimizing
    print("7️⃣ Penalty Solver Convergence Check")
    print("   Running penalty solver with more iterations...")
    
    # Run penalty solver with more iterations and better convergence criteria
    y_tilde_detailed = algorithm._minimize_penalty_lagrangian_detailed(x, y_star, lambda_star, alpha, delta)
    
    gap_detailed = torch.norm(y_tilde_detailed - y_star).item()
    print(f"   Detailed solver gap: {gap_detailed:.6f}")
    print(f"   Improvement: {gap - gap_detailed:.6f}")
    
    return gap, gap_detailed

if __name__ == "__main__":
    gap, gap_detailed = diagnose_gap_issue()
    
    print("\n🎯 SUMMARY")
    print("=" * 30)
    print(f"Original gap: {gap:.6f}")
    print(f"Detailed gap: {gap_detailed:.6f}")
    print(f"Target: < 0.1")
    print(f"Status: {'✅ IMPROVED' if gap_detailed < gap else '❌ NO IMPROVEMENT'}")
