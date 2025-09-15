#!/usr/bin/env python3
"""
Deep analysis of the gap calculation and lower-level solution accuracy
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final

def analyze_gap_components():
    """Deep dive into gap calculation components"""
    print("🔍 DEEP GAP ANALYSIS")
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
    
    # Test point
    x = torch.randn(5, dtype=torch.float64)
    alpha = 0.001
    
    print(f"Test point x: {x}")
    print(f"Alpha: {alpha}")
    print()
    
    # Step 1: Get accurate lower-level solution y*
    print("1️⃣ ACCURATE LOWER-LEVEL SOLUTION (y*)")
    print("-" * 40)
    y_star, info = problem.solve_lower_level(x, 'accurate', 1000, 1e-6, alpha)
    lambda_star = info.get('lambda_star', torch.zeros(problem.num_constraints, dtype=torch.float64))
    
    print(f"y* = {y_star}")
    print(f"λ* = {lambda_star}")
    print(f"Constraint violations: {info.get('constraint_violations', 'N/A')}")
    print(f"Max violation: {info.get('max_violation', 'N/A')}")
    print()
    
    # Step 2: Check if y* actually satisfies constraints
    print("2️⃣ CONSTRAINT SATISFACTION CHECK")
    print("-" * 40)
    h_val = problem.constraints(x, y_star)
    print(f"h(x, y*) = {h_val}")
    print(f"All constraints satisfied: {torch.all(h_val <= 1e-6)}")
    print(f"Max constraint violation: {torch.max(h_val).item():.2e}")
    print()
    
    # Step 3: Get penalty minimizer y~
    print("3️⃣ PENALTY MINIMIZER (y~)")
    print("-" * 40)
    delta = alpha**3
    y_penalty = algorithm._minimize_penalty_lagrangian(x, y_star, lambda_star, alpha, delta)
    
    print(f"y~ = {y_penalty}")
    print()
    
    # Step 4: Check if y~ satisfies constraints
    print("4️⃣ PENALTY MINIMIZER CONSTRAINT CHECK")
    print("-" * 40)
    h_penalty = problem.constraints(x, y_penalty)
    print(f"h(x, y~) = {h_penalty}")
    print(f"All constraints satisfied: {torch.all(h_penalty <= 1e-6)}")
    print(f"Max constraint violation: {torch.max(h_penalty).item():.2e}")
    print()
    
    # Step 5: Calculate the actual gap
    print("5️⃣ GAP CALCULATION")
    print("-" * 40)
    gap = torch.norm(y_penalty - y_star).item()
    print(f"||y~ - y*|| = {gap:.2e}")
    print()
    
    # Step 6: Check objective values
    print("6️⃣ OBJECTIVE VALUES")
    print("-" * 40)
    f_star = problem.upper_objective(x, y_star)
    f_penalty = problem.upper_objective(x, y_penalty)
    g_star = problem.lower_objective(x, y_star)
    g_penalty = problem.lower_objective(x, y_penalty)
    
    print(f"f(x, y*) = {f_star:.6f}")
    print(f"f(x, y~) = {f_penalty:.6f}")
    print(f"g(x, y*) = {g_star:.6f}")
    print(f"g(x, y~) = {g_penalty:.6f}")
    print()
    
    # Step 7: Check penalty Lagrangian values
    print("7️⃣ PENALTY LAGRANGIAN VALUES")
    print("-" * 40)
    L_star = algorithm._compute_penalty_lagrangian(x, y_star, y_star, lambda_star, alpha, delta)
    L_penalty = algorithm._compute_penalty_lagrangian(x, y_penalty, y_star, lambda_star, alpha, delta)
    
    print(f"L(x, y*, y*, λ*) = {L_star:.6f}")
    print(f"L(x, y~, y*, λ*) = {L_penalty:.6f}")
    print(f"L_penalty < L_star: {L_penalty < L_star}")
    print()
    
    # Step 8: Check if penalty minimizer is actually minimizing
    print("8️⃣ PENALTY MINIMIZATION CHECK")
    print("-" * 40)
    if L_penalty < L_star:
        print("✅ y~ is actually minimizing the penalty Lagrangian")
    else:
        print("❌ y~ is NOT minimizing the penalty Lagrangian!")
        print("This suggests the penalty solver is not working correctly")
    print()
    
    # Step 9: Check hypergradient components
    print("9️⃣ HYPERGRADIENT COMPONENTS")
    print("-" * 40)
    hypergradient = algorithm.oracle_sample(x, alpha, 10)
    print(f"Hypergradient norm: {torch.norm(hypergradient).item():.6f}")
    print(f"Hypergradient: {hypergradient}")
    print()
    
    return {
        'gap': gap,
        'y_star': y_star,
        'y_penalty': y_penalty,
        'constraints_satisfied': torch.all(h_val <= 1e-6),
        'penalty_minimizing': L_penalty < L_star,
        'hypergradient_norm': torch.norm(hypergradient).item()
    }

def test_multiple_points():
    """Test gap calculation on multiple random points"""
    print("\n🔄 TESTING MULTIPLE POINTS")
    print("=" * 60)
    
    problem = StronglyConvexBilevelProblem(
        dim=5, 
        num_constraints=3, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    algorithm = F2CSAAlgorithm1Final(problem)
    alpha = 0.001
    
    gaps = []
    hypergradient_norms = []
    
    for i in range(5):
        print(f"\n--- Test Point {i+1} ---")
        x = torch.randn(5, dtype=torch.float64)
        
        # Get solutions
        y_star, info = problem.solve_lower_level(x, 'accurate', 1000, 1e-6, alpha)
        lambda_star = info.get('lambda_star', torch.zeros(problem.num_constraints, dtype=torch.float64))
        delta = alpha**3
        y_penalty = algorithm._minimize_penalty_lagrangian(x, y_star, lambda_star, alpha, delta)
        
        # Calculate gap
        gap = torch.norm(y_penalty - y_star).item()
        gaps.append(gap)
        
        # Get hypergradient
        hypergradient = algorithm.oracle_sample(x, alpha, 10)
        hypergradient_norm = torch.norm(hypergradient).item()
        hypergradient_norms.append(hypergradient_norm)
        
        print(f"Gap: {gap:.2e}")
        print(f"Hypergradient norm: {hypergradient_norm:.2f}")
    
    print(f"\n📊 SUMMARY:")
    print(f"Average gap: {np.mean(gaps):.2e}")
    print(f"Max gap: {np.max(gaps):.2e}")
    print(f"Average hypergradient norm: {np.mean(hypergradient_norms):.2f}")
    print(f"Max hypergradient norm: {np.max(hypergradient_norms):.2f}")

if __name__ == "__main__":
    # Run deep analysis
    results = analyze_gap_components()
    
    # Test multiple points
    test_multiple_points()
    
    print("\n🎯 CONCLUSION:")
    if results['gap'] < 0.1:
        print("✅ Gap is small, but let's check if it's meaningful")
    else:
        print("❌ Gap is too large")
    
    if results['penalty_minimizing']:
        print("✅ Penalty minimizer is working")
    else:
        print("❌ Penalty minimizer is NOT working correctly")
    
    if results['constraints_satisfied']:
        print("✅ Constraints are satisfied")
    else:
        print("❌ Constraints are NOT satisfied")
