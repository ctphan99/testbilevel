#!/usr/bin/env python3
"""
Test lower-level solver convergence to ensure gap < 0.1 between y~ and y*
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem

def test_lower_level_convergence():
    """Test current lower-level solver convergence"""
    print("Testing current lower-level solver convergence...")
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.01, strong_convex=True)
    
    # Test point
    x = torch.randn(5) * 0.1
    print(f"Test point x: {x}")
    
    # Test current PGD solver
    print("\n=== Current PGD Solver ===")
    y_opt, info = problem.solve_lower_level(x, solver='pgd', max_iter=1000, tol=1e-6)
    
    print(f"Solution y: {y_opt}")
    print(f"Iterations: {info['iterations']}")
    print(f"Converged: {info['converged']}")
    print(f"Constraint violations: {info['constraint_violations']}")
    print(f"Max violation: {torch.max(info['constraint_violations']):.6f}")
    
    # Check if solution is actually optimal
    h_val = problem.constraints(x, y_opt)
    print(f"Constraint values: {h_val}")
    
    # Test with different tolerances
    print("\n=== Testing Different Tolerances ===")
    for tol in [1e-3, 1e-6, 1e-8]:
        y_test, info_test = problem.solve_lower_level(x, solver='pgd', max_iter=2000, tol=tol)
        grad_norm = torch.norm(problem.Q_lower @ y_test + problem.c_lower)
        print(f"Tol {tol}: iterations={info_test['iterations']}, grad_norm={grad_norm:.2e}, converged={info_test['converged']}")
    
    # Test convergence to true solution
    print("\n=== Testing Convergence to True Solution ===")
    
    # Get true solution using CVXPY (if available)
    try:
        import cvxpy as cp
        print("CVXPY available - computing true solution...")
        
        # Set up CVXPY problem
        y_cvx = cp.Variable(problem.dim)
        objective = cp.Minimize(0.5 * cp.quad_form(y_cvx, problem.Q_lower.cpu().numpy()) + 
                               problem.c_lower.cpu().numpy().T @ y_cvx)
        
        # Constraints: h(x,y) = Ax + By - b <= 0
        constraints = [problem.A.cpu().numpy() @ x.cpu().numpy() + 
                      problem.B.cpu().numpy() @ y_cvx - problem.b.cpu().numpy() <= 0]
        
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=False)
        
        if prob.status == cp.OPTIMAL:
            y_true = torch.tensor(y_cvx.value, dtype=problem.dtype, device=problem.device)
            print(f"True solution y*: {y_true}")
            
            # Test our solver with high precision
            y_approx, info_approx = problem.solve_lower_level(x, solver='pgd', max_iter=5000, tol=1e-10)
            
            # Compute gap
            gap = torch.norm(y_approx - y_true)
            print(f"Gap ||y~ - y*||: {gap:.6f}")
            
            if gap < 0.1:
                print("✓ SUCCESS: Gap < 0.1 achieved!")
            else:
                print("✗ FAILURE: Gap >= 0.1, need better solver")
                
        else:
            print(f"CVXPY failed with status: {prob.status}")
            
    except ImportError:
        print("CVXPY not available, testing with high precision PGD...")
        
        # Test with very high precision
        y_high_prec, info_high = problem.solve_lower_level(x, solver='pgd', max_iter=10000, tol=1e-12)
        grad_norm_high = torch.norm(problem.Q_lower @ y_high_prec + problem.c_lower)
        print(f"High precision: iterations={info_high['iterations']}, grad_norm={grad_norm_high:.2e}")
        
        # Use this as reference for gap testing
        y_ref = y_high_prec
        
        # Test different solver settings
        print("\n=== Testing Different Solver Settings ===")
        for max_iter in [1000, 2000, 5000]:
            for tol in [1e-6, 1e-8, 1e-10]:
                y_test, info_test = problem.solve_lower_level(x, solver='pgd', max_iter=max_iter, tol=tol)
                gap = torch.norm(y_test - y_ref)
                print(f"max_iter={max_iter}, tol={tol}: gap={gap:.6f}, converged={info_test['converged']}")

if __name__ == "__main__":
    test_lower_level_convergence()
