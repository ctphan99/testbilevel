#!/usr/bin/env python3
"""
Focused diagnostic for penalty computation in F2CSA Algorithm 1
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_accurate import F2CSAAlgorithm1

def analyze_penalty_computation():
    """Analyze the penalty computation in detail"""
    print("=== F2CSA Penalty Computation Diagnostic ===\n")
    
    # Create problem instance
    problem = StronglyConvexBilevelProblem(
        dim=5, num_constraints=3, noise_std=0.1,
        strong_convex=True, device='cpu'
    )
    
    # Test with α = 0.1 (δ = 0.001)
    alpha = 0.1
    delta = alpha ** 3
    print(f"Testing with α = {alpha}, δ = {delta:.6f}")
    print(f"α₁ = α⁻² = {1/alpha**2:.1f}")
    print(f"α₂ = α⁻⁴ = {1/alpha**4:.1f}")
    print()
    
    # Initialize algorithm
    algorithm = F2CSAAlgorithm1(problem)
    
    # Test point
    x = torch.randn(5, dtype=torch.float64)
    print(f"Test point x: {x}")
    print()
    
    # Get accurate lower-level solution
    print("1. Computing accurate lower-level solution...")
    y_star, info = problem.solve_lower_level(x, solver='accurate', alpha=alpha)
    lambda_star = info.get('lambda', torch.zeros(3, dtype=torch.float64))
    print(f"   y* = {y_star}")
    print(f"   λ* = {lambda_star}")
    print(f"   Constraint violations: {info['constraint_violations']}")
    print(f"   Max violation: {info['max_violation']}")
    print()
    
    # Compute constraint values
    h_val = problem.constraints(x, y_star)
    print(f"2. Constraint values h(x,y*): {h_val}")
    print(f"   All constraints satisfied: {torch.all(h_val <= 0)}")
    print()
    
    # Test penalty minimizer using oracle_sample
    print("3. Testing penalty minimizer via oracle_sample...")
    try:
        hypergradient = algorithm.oracle_sample(x, alpha, N_g=1)
        print(f"   Hypergradient norm: {torch.norm(hypergradient):.6f}")
    except Exception as e:
        print(f"   Error in oracle_sample: {e}")
    
    print()
    
    # Test penalty Lagrangian computation
    print("4. Testing penalty Lagrangian computation...")
    try:
        # Create a test y for penalty computation
        y_test = torch.randn(5, dtype=torch.float64, requires_grad=True)
        
        # Compute penalty Lagrangian
        L_val = algorithm.penalty_lagrangian(x, y_test, y_star, lambda_star, alpha)
        print(f"   Penalty Lagrangian value: {L_val:.6f}")
        
        # Compute gradient
        L_val.backward()
        if y_test.grad is not None:
            print(f"   Penalty gradient norm: {torch.norm(y_test.grad):.6f}")
        
    except Exception as e:
        print(f"   Error in penalty Lagrangian computation: {e}")
    
    print()
    
    # Test with different α values
    print("6. Testing different α values...")
    alpha_values = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01]
    
    for alpha_test in alpha_values:
        delta_test = alpha_test ** 3
        alpha1_test = 1 / (alpha_test ** 2)
        alpha2_test = 1 / (alpha_test ** 4)
        
        print(f"   α = {alpha_test:.2f}: δ = {delta_test:.6f}, α₁ = {alpha1_test:.1f}, α₂ = {alpha2_test:.1f}")

if __name__ == "__main__":
    analyze_penalty_computation()
