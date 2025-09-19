#!/usr/bin/env python3
"""
Test script to verify that F2CSA's _solve_lower_level_accurate and 
problem.py's solve_lower_level('cvxpy') return the same results
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final

def test_solver_consistency():
    """Test that both solvers return identical results"""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create problem
    dim = 5
    problem = StronglyConvexBilevelProblem(dim=dim, noise_std=0.01, device='cpu')
    
    # Create F2CSA algorithm
    f2csa = F2CSAAlgorithm1Final(problem)
    
    # Test point
    x = torch.randn(dim, dtype=torch.float64)
    alpha = 0.05
    
    print(f"Testing solver consistency with x = {x}")
    print(f"Problem dimension: {dim}")
    print(f"Alpha: {alpha}")
    print("-" * 50)
    
    # Method 1: F2CSA using centralized solver
    print("Method 1: F2CSA using problem.solve_lower_level('cvxpy')")
    try:
        # Reset random state before F2CSA method
        torch.manual_seed(42)
        np.random.seed(42)
        y1, lambda1, info1 = f2csa.problem.solve_lower_level(x, solver='cvxpy', alpha=alpha)
        print(f"  y1 = {y1}")
        print(f"  lambda1 = {lambda1}")
        print(f"  info1 = {info1}")
        print(f"  Status: SUCCESS")
    except Exception as e:
        print(f"  Status: FAILED - {e}")
        y1, lambda1, info1 = None, None, None
    
    print()
    
    # Method 2: Problem's centralized solver
    print("Method 2: problem.solve_lower_level('cvxpy')")
    try:
        # Reset random state before problem method
        torch.manual_seed(42)
        np.random.seed(42)
        y2, lambda2, info2 = problem.solve_lower_level(x, solver='cvxpy', alpha=alpha)
        print(f"  y2 = {y2}")
        print(f"  lambda2 = {lambda2}")
        print(f"  info2 = {info2}")
        print(f"  Status: SUCCESS")
    except Exception as e:
        print(f"  Status: FAILED - {e}")
        y2, lambda2, info2 = None, None, None
    
    print()
    print("-" * 50)
    
    # Compare results
    if y1 is not None and y2 is not None:
        y_diff = torch.norm(y1 - y2).item()
        lambda_diff = torch.norm(lambda1 - lambda2).item()
        
        print("COMPARISON:")
        print(f"  y difference: {y_diff:.2e}")
        print(f"  lambda difference: {lambda_diff:.2e}")
        
        if y_diff < 1e-6 and lambda_diff < 1e-6:
            print("  ✅ RESULTS MATCH! Both solvers return identical solutions.")
        else:
            print("  ❌ RESULTS DIFFER! Solvers return different solutions.")
    else:
        print("  ❌ Cannot compare - one or both solvers failed.")
    
    print()
    
    # Test with different noise seeds
    print("Testing with different noise instances...")
    for i in range(3):
        print(f"\nNoise instance {i+1}:")
        
        # Reset random state
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)
        
        # F2CSA method
        y1, _, _ = f2csa.problem.solve_lower_level(x, solver='cvxpy', alpha=alpha)
        
        # Reset random state again for problem method
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)
        
        # Problem method
        y2, _, _ = problem.solve_lower_level(x, solver='cvxpy', alpha=alpha)
        
        y_diff = torch.norm(y1 - y2).item()
        print(f"  y difference: {y_diff:.2e}")
        
        if y_diff < 1e-6:
            print(f"  ✅ Instance {i+1}: Results match")
        else:
            print(f"  ❌ Instance {i+1}: Results differ")

if __name__ == "__main__":
    test_solver_consistency()
