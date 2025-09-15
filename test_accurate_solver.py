#!/usr/bin/env python3
"""
Test accurate lower-level solver to ensure gap < 0.1 between y~ and y*
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from accurate_lower_level_solver import AccurateLowerLevelSolver

def test_accurate_solver():
    """Test accurate solver convergence"""
    print("Testing accurate lower-level solver...")
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.01, strong_convex=True)
    
    # Test point
    x = torch.randn(5) * 0.1
    print(f"Test point x: {x}")
    
    # Get ground truth using CVXPY directly
    print("\n=== Getting Ground Truth (CVXPY) ===")
    try:
        import cvxpy as cp
        x_np = x.detach().cpu().numpy()
        y_cp = cp.Variable(problem.dim)
        
        # Lower-level objective: 0.5 * y^T Q_lower y + c_lower^T y
        Q_lower_np = problem.Q_lower.detach().cpu().numpy()
        c_lower_np = problem.c_lower.detach().cpu().numpy()
        objective = cp.Minimize(0.5 * cp.quad_form(y_cp, Q_lower_np) + c_lower_np @ y_cp)
        
        # Constraints: Ax + By - b <= 0
        A_np = problem.A.detach().cpu().numpy()
        B_np = problem.B.detach().cpu().numpy()
        b_np = problem.b.detach().cpu().numpy()
        constraints = [A_np @ x_np + B_np @ y_cp - b_np <= 0]
        
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=False)
        
        if prob.status in ["optimal", "optimal_near"]:
            y_true_np = y_cp.value
            lambda_true_np = constraints[0].dual_value if constraints[0].dual_value is not None else np.zeros(problem.num_constraints)
            
            y_true = torch.tensor(y_true_np, device=problem.device, dtype=problem.dtype)
            lambda_true = torch.tensor(lambda_true_np, device=problem.device, dtype=problem.dtype)
            
            print(f"True solution y*: {y_true}")
            print(f"True dual variables λ*: {lambda_true}")
            
            # Check constraint satisfaction
            h_true = problem.constraints(x, y_true)
            violations_true = torch.clamp(h_true, min=0)
            print(f"True constraint violations: {violations_true}")
            print(f"Max violation: {torch.max(violations_true):.6f}")
        else:
            print(f"CVXPY failed with status: {prob.status}")
            return
            
    except Exception as e:
        print(f"CVXPY failed: {e}")
        return
    
    # Test accurate solver with different alpha values
    print("\n=== Testing Accurate Solver ===")
    alpha_values = [0.1, 0.05, 0.01, 0.005]
    
    # Create solver instance
    solver = AccurateLowerLevelSolver(problem, device=problem.device, dtype=problem.dtype)
    
    for alpha in alpha_values:
        print(f"\n--- Testing with α = {alpha} ---")
        
        # Test accurate solver (tries CVXPY first, then iterative)
        print("Accurate solver:")
        y_opt, lambda_opt, info = solver.solve_lower_level_accurate(x, alpha, max_iter=5000, tol=1e-8)
        
        # Compute gap
        gap = torch.norm(y_opt - y_true)
        print(f"Gap ||y~ - y*||: {gap:.6f}")
        print(f"Iterations: {info['iterations']}")
        print(f"Converged: {info['converged']}")
        if 'final_grad_norm' in info:
            print(f"Final grad norm: {info['final_grad_norm']:.2e}")
        if 'final_violation' in info:
            print(f"Final violation: {info['final_violation']:.2e}")
        if 'delta' in info:
            print(f"δ = α³: {info['delta']:.2e}")
        if 'solver' in info:
            print(f"Solver used: {info['solver']}")
        
        if gap < 0.1:
            print("✓ SUCCESS: Gap < 0.1 achieved!")
        else:
            print("✗ FAILURE: Gap >= 0.1")
        
        # Test SGD-based solver
        print("\nSGD-based solver:")
        y_sgd, lambda_sgd, info_sgd = solver.solve_with_sgd(x, alpha, max_iter=5000)
        
        gap_sgd = torch.norm(y_sgd - y_true)
        print(f"Gap ||y~ - y*||: {gap_sgd:.6f}")
        print(f"Iterations: {info_sgd['iterations']}")
        print(f"Converged: {info_sgd['converged']}")
        print(f"Final grad norm: {info_sgd['final_grad_norm']:.2e}")
        print(f"Final violation: {info_sgd['final_violation']:.2e}")
        
        if gap_sgd < 0.1:
            print("✓ SUCCESS: Gap < 0.1 achieved!")
        else:
            print("✗ FAILURE: Gap >= 0.1")
    
    # Test with problem's solve_lower_level method
    print("\n=== Testing Problem's solve_lower_level Method ===")
    for alpha in [0.1, 0.05, 0.01]:
        print(f"\n--- Testing with α = {alpha} ---")
        
        y_opt, info = problem.solve_lower_level(x, solver='accurate', alpha=alpha, max_iter=5000, tol=1e-8)
        
        gap = torch.norm(y_opt - y_true)
        print(f"Gap ||y~ - y*||: {gap:.6f}")
        print(f"Iterations: {info['iterations']}")
        print(f"Converged: {info['converged']}")
        if 'final_grad_norm' in info:
            print(f"Final grad norm: {info['final_grad_norm']:.2e}")
        if 'final_violation' in info:
            print(f"Final violation: {info['final_violation']:.2e}")
        if 'solver' in info:
            print(f"Solver used: {info['solver']}")
        
        if gap < 0.1:
            print("✓ SUCCESS: Gap < 0.1 achieved!")
        else:
            print("✗ FAILURE: Gap >= 0.1")
    
    # Test convergence with different problem instances
    print("\n=== Testing Multiple Problem Instances ===")
    success_count = 0
    total_tests = 5
    
    for i in range(total_tests):
        print(f"\n--- Problem Instance {i+1} ---")
        
        # Create new problem instance
        problem_test = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.01, strong_convex=True)
        x_test = torch.randn(5) * 0.1
        
        try:
            # Get ground truth
            solver_test = AccurateLowerLevelSolver(problem_test, device=problem_test.device, dtype=problem_test.dtype)
            y_true_test, _ = solver_test.solve_with_cvxpy(x_test)
            
            # Test accurate solver
            y_opt_test, info_test = problem_test.solve_lower_level(x_test, solver='accurate', alpha=0.05, max_iter=3000, tol=1e-8)
            
            gap_test = torch.norm(y_opt_test - y_true_test)
            print(f"Gap: {gap_test:.6f}, Iterations: {info_test['iterations']}, Converged: {info_test['converged']}")
            
            if gap_test < 0.1:
                success_count += 1
                print("✓ SUCCESS")
            else:
                print("✗ FAILURE")
                
        except Exception as e:
            print(f"Test failed: {e}")
    
    print(f"\n=== Summary ===")
    print(f"Successful tests: {success_count}/{total_tests}")
    print(f"Success rate: {success_count/total_tests*100:.1f}%")
    
    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED! Accurate solver achieves gap < 0.1 consistently!")
    else:
        print("⚠️  Some tests failed. Need to improve solver accuracy.")

if __name__ == "__main__":
    test_accurate_solver()
