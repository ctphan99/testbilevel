#!/usr/bin/env python3
"""
Test script to verify Gurobi integration with bilevel optimization problem
"""

import torch
import numpy as np
import time
from problem import StronglyConvexBilevelProblem

def test_gurobi_solver():
    """Test Gurobi solver on bilevel optimization problem"""
    print("Testing Gurobi integration with bilevel optimization...")
    
    # Create problem instance
    dim = 10
    problem = StronglyConvexBilevelProblem(dim=dim, device='cpu')
    
    # Test point
    x_test = torch.randn(dim, dtype=torch.float64)
    
    print(f"Problem dimension: {dim}")
    print(f"Test point x: {x_test.numpy()}")
    
    # Test different solvers
    solvers = ['gurobi', 'cvxpy', 'pgd']
    results = {}
    
    for solver in solvers:
        print(f"\n--- Testing {solver.upper()} solver ---")
        try:
            start_time = time.time()
            y_opt, lambda_opt, info = problem.solve_lower_level(x_test, solver=solver)
            solve_time = time.time() - start_time
            
            # Compute objective value
            obj_value = problem.lower_objective(x_test, y_opt).item()
            
            results[solver] = {
                'y_opt': y_opt,
                'lambda_opt': lambda_opt,
                'info': info,
                'solve_time': solve_time,
                'obj_value': obj_value
            }
            
            print(f"Status: {info.get('status', 'unknown')}")
            print(f"Converged: {info.get('converged', False)}")
            print(f"Solve time: {solve_time:.4f}s")
            print(f"Objective value: {obj_value:.6f}")
            print(f"Max constraint violation: {info.get('max_violation', 'N/A')}")
            if 'iterations' in info:
                print(f"Iterations: {info['iterations']}")
            if 'obj_value' in info:
                print(f"Solver objective: {info['obj_value']:.6f}")
                
        except Exception as e:
            print(f"Error with {solver}: {e}")
            results[solver] = None
    
    # Compare results
    print("\n--- Comparison ---")
    if results['gurobi'] and results['cvxpy']:
        gurobi_obj = results['gurobi']['obj_value']
        cvxpy_obj = results['cvxpy']['obj_value']
        diff = abs(gurobi_obj - cvxpy_obj)
        print(f"Gurobi vs CVXPY objective difference: {diff:.2e}")
        print(f"Gurobi vs CVXPY time ratio: {results['gurobi']['solve_time']/results['cvxpy']['solve_time']:.2f}x")
    
    return results

def test_gurobi_parameters():
    """Test Gurobi with different parameter settings"""
    print("\n--- Testing Gurobi Parameters ---")
    
    problem = StronglyConvexBilevelProblem(dim=5, device='cpu')
    x_test = torch.randn(5, dtype=torch.float64)
    
    # Test with different tolerances
    tolerances = [1e-6, 1e-8, 1e-10]
    
    for tol in tolerances:
        print(f"\nTesting with tolerance: {tol}")
        try:
            start_time = time.time()
            y_opt, lambda_opt, info = problem.solve_lower_level(x_test, solver='gurobi')
            solve_time = time.time() - start_time
            
            print(f"Solve time: {solve_time:.4f}s")
            print(f"Objective: {problem.lower_objective(x_test, y_opt).item():.8f}")
            print(f"Max violation: {info.get('max_violation', 'N/A')}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Test basic functionality
    results = test_gurobi_solver()
    
    # Test parameters
    test_gurobi_parameters()
    
    print("\nGurobi integration test completed!")
