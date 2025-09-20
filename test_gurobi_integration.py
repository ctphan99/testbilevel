#!/usr/bin/env python3
"""
Test Gurobi integration with the bilevel optimization problem
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem

def test_gurobi_integration():
    """Test that Gurobi integration works correctly"""
    
    print("Testing Gurobi integration...")
    print("=" * 50)
    
    # Create a small problem for testing
    problem = StronglyConvexBilevelProblem(
        dim=5, 
        num_constraints=3, 
        noise_std=0.01, 
        strong_convex=True,
        device='cpu'
    )
    
    # Test point
    x = torch.randn(5, dtype=torch.float64)
    print(f"Test point x: {x}")
    print()
    
    # Test Gurobi solver
    print("Testing Gurobi solver...")
    try:
        y_gurobi, lambda_gurobi, info_gurobi = problem.solve_lower_level(x, solver='gurobi')
        print(f"✅ Gurobi solver successful!")
        print(f"   Solution: {y_gurobi}")
        print(f"   Lambda: {lambda_gurobi}")
        print(f"   Status: {info_gurobi['status']}")
        print(f"   Iterations: {info_gurobi['iterations']}")
        print(f"   Max violation: {info_gurobi['max_violation']:.2e}")
        print()
    except Exception as e:
        print(f"❌ Gurobi solver failed: {e}")
        print()
    
    # Test CVXPY solver for comparison
    print("Testing CVXPY solver...")
    try:
        y_cvxpy, lambda_cvxpy, info_cvxpy = problem.solve_lower_level(x, solver='cvxpy')
        print(f"✅ CVXPY solver successful!")
        print(f"   Solution: {y_cvxpy}")
        print(f"   Lambda: {lambda_cvxpy}")
        print(f"   Status: {info_cvxpy['status']}")
        print(f"   Iterations: {info_cvxpy['iterations']}")
        print(f"   Max violation: {info_cvxpy['max_violation']:.2e}")
        print()
    except Exception as e:
        print(f"❌ CVXPY solver failed: {e}")
        print()
    
    # Compare solutions if both worked
    try:
        if 'y_gurobi' in locals() and 'y_cvxpy' in locals():
            diff = torch.norm(y_gurobi - y_cvxpy).item()
            print(f"Solution difference (Gurobi vs CVXPY): {diff:.2e}")
            if diff < 1e-6:
                print("✅ Solutions are very close!")
            else:
                print("⚠️  Solutions differ significantly")
    except:
        pass
    
    print("=" * 50)
    print("Gurobi integration test completed!")

if __name__ == "__main__":
    test_gurobi_integration()