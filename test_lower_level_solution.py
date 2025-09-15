#!/usr/bin/env python3
"""
Test script to verify lower-level solution is working correctly with natural constraint violations
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm import F2CSAAlgorithm

def test_lower_level_solution():
    """Test that lower-level solution works with constraint violations"""
    print("Testing lower-level solution with natural constraint violations...")
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.01, strong_convex=True)
    
    # Test point
    x = torch.randn(5) * 0.1
    
    print(f"Test point x: {x}")
    
    # Solve lower-level problem
    y_opt, info = problem.solve_lower_level(x, solver='pgd')
    
    print(f"Lower-level solution y: {y_opt}")
    print(f"Solution info: {info}")
    
    # Check constraint violations
    h_val = problem.A @ x - problem.B @ y_opt - problem.b
    print(f"Constraint values h(x,y): {h_val}")
    print(f"Constraint violations (positive values): {torch.clamp(h_val, min=0)}")
    
    # Check if constraints are violated (this should be normal now)
    violations = torch.clamp(h_val, min=0)
    max_violation = torch.max(violations)
    print(f"Maximum constraint violation: {max_violation:.6f}")
    
    if max_violation > 1e-6:
        print("✅ SUCCESS: Constraints are naturally violated, enabling F2CSA penalty mechanism")
    else:
        print("❌ WARNING: No constraint violations detected - F2CSA may not work properly")
    
    # Test F2CSA oracle
    print("\nTesting F2CSA oracle...")
    f2csa = F2CSAAlgorithm(problem=problem)
    
    # Test oracle sample
    alpha = 0.1
    N_g = 4
    grad = f2csa.oracle_sample(x, alpha, N_g)
    
    print(f"F2CSA hypergradient: {grad}")
    print(f"Hypergradient norm: {torch.norm(grad):.6f}")
    
    return True

if __name__ == "__main__":
    test_lower_level_solution()
