#!/usr/bin/env python3
"""
Focused debug script for lower-level solver with 100 iterations
Test constraint handling and ensure stable optimization
"""

import torch
import numpy as np
import cvxpy as cp
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm import F2CSAAlgorithm
import matplotlib.pyplot as plt

def test_constraint_activity_detailed(problem, x_test):
    """Test constraint activity in detail for a given x"""
    print(f"\n--- Testing constraint activity for x = {x_test.detach().numpy()} ---")
    
    # Solve lower level
    y_opt, info = problem.solve_lower_level(x_test)
    dual_vars = info.get('lambda', None)
    
    # Compute constraint values
    constraint_values = problem.A @ x_test - problem.B @ y_opt - problem.b
    
    print(f"Constraint values: {constraint_values.detach().numpy()}")
    print(f"Dual variables: {dual_vars.detach().numpy() if dual_vars is not None else 'None'}")
    print(f"Constraint activity: {(constraint_values >= -1e-6).sum().item()}/{len(constraint_values)}")
    
    # Check KKT conditions
    if dual_vars is not None:
        # Stationarity: Q y - B^T λ + c = 0
        stationarity = problem.Q_lower @ y_opt - problem.B.T @ dual_vars + problem.c_lower
        print(f"Stationarity residual: {torch.norm(stationarity).item():.6f}")
        
        # Primal feasibility: A x - B y - b <= 0
        primal_violation = torch.norm(torch.clamp(constraint_values, min=0)).item()
        print(f"Primal violation: {primal_violation:.6f}")
        
        # Dual feasibility: λ >= 0
        dual_violation = torch.norm(torch.clamp(-dual_vars, min=0)).item()
        print(f"Dual violation: {dual_violation:.6f}")
        
        # Complementary slackness: λ^T (A x - B y - b) = 0
        complementarity = torch.abs(dual_vars @ constraint_values).item()
        print(f"Complementary slackness: {complementarity:.6f}")
    
    return y_opt, dual_vars, constraint_values

def test_f2csa_with_detailed_logging(problem, max_iters=100):
    """Test F2CSA with detailed logging for debugging"""
    print("=" * 80)
    print("F2CSA DETAILED DEBUGGING - 100 ITERATIONS")
    print("=" * 80)
    
    # Create F2CSA algorithm
    f2csa = F2CSAAlgorithm(
        problem=problem,
        alpha_override=0.08,
        eta_override=0.001,
        D_override=0.01,
        Ng_override=64,
        grad_ema_beta_override=0.9,
        prox_weight_override=0.1,
        grad_clip_override=1.0
    )
    
    # Run optimization with detailed logging
    results = f2csa.optimize(max_iterations=max_iters, verbose=True)
    
    print(f"\nFinal results:")
    print(f"  Final gap: {results['final_gap']:.6f}")
    print(f"  Final EMA gap: {results['final_ema_gap']:.6f}")
    print(f"  Total iterations: {results['total_iterations']}")
    
    # Analyze gap components
    if 'gap_history' in results:
        gaps = results['gap_history']
        direct_gaps = results.get('direct_gap_history', [])
        implicit_gaps = results.get('implicit_gap_history', [])
        
        print(f"\nGap analysis:")
        print(f"  Initial gap: {gaps[0]:.6f}")
        print(f"  Final gap: {gaps[-1]:.6f}")
        print(f"  Gap change: {gaps[-1] - gaps[0]:.6f}")
        
        if direct_gaps and implicit_gaps:
            print(f"  Final direct: {direct_gaps[-1]:.6f}")
            print(f"  Final implicit: {implicit_gaps[-1]:.6f}")
            print(f"  Implicit change: {implicit_gaps[-1] - implicit_gaps[0]:.6f}")
    
    return results

def test_problem_setup(problem):
    """Test the problem setup to ensure it's correct"""
    print("=" * 80)
    print("PROBLEM SETUP TESTING")
    print("=" * 80)
    
    print(f"Problem dimensions:")
    print(f"  x dimension: {problem.dim}")
    print(f"  y dimension: {problem.dim}")
    print(f"  constraint dimension: {problem.num_constraints}")
    
    print(f"\nConstraint matrices:")
    print(f"  A shape: {problem.A.shape}")
    print(f"  B shape: {problem.B.shape}")
    print(f"  b shape: {problem.b.shape}")
    
    print(f"\nConstraint values:")
    print(f"  b: {problem.b.detach().numpy()}")
    print(f"  A norm: {torch.norm(problem.A).item():.6f}")
    print(f"  B norm: {torch.norm(problem.B).item():.6f}")
    
    # Test constraint feasibility at origin
    x_origin = torch.zeros(problem.dim, dtype=torch.float64)
    y_origin, info = problem.solve_lower_level(x_origin)
    constraint_origin = problem.A @ x_origin - problem.B @ y_origin - problem.b
    print(f"\nConstraint feasibility at origin:")
    print(f"  Max violation: {constraint_origin.max().item():.6f}")
    print(f"  Should be <= 0: {constraint_origin.max().item() <= 1e-6}")
    
    return problem

def main():
    """Main debugging function"""
    print("FOCUSED LOWER-LEVEL SOLVER DEBUGGING")
    print("Testing for 100 iterations to debug until fix")
    print("=" * 80)
    
    # Create problem with balanced constraint tightening
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, seed=42)
    
    # Apply balanced constraint tightening
    problem.b = problem.b - 0.2  # Moderate tightening
    problem.B = problem.B * 2.5  # Moderate scaling
    problem.Q_lower = problem.Q_lower * 1.8  # Moderate steepening
    
    # Test problem setup
    test_problem_setup(problem)
    
    # Test constraint activity with different x values
    print("\n" + "=" * 80)
    print("CONSTRAINT ACTIVITY TESTING")
    print("=" * 80)
    
    for i in range(3):
        x_test = torch.randn(problem.dim, dtype=torch.float64)
        test_constraint_activity_detailed(problem, x_test)
    
    # Test F2CSA with detailed logging
    print("\n" + "=" * 80)
    print("F2CSA OPTIMIZATION TESTING")
    print("=" * 80)
    
    results = test_f2csa_with_detailed_logging(problem, max_iters=100)
    
    # Check if we achieved the target
    final_gap = results['final_ema_gap']
    target_achieved = final_gap < 0.1
    
    print(f"\n" + "=" * 80)
    print("DEBUGGING SUMMARY")
    print("=" * 80)
    print(f"Final EMA gap: {final_gap:.6f}")
    print(f"Target achieved (< 0.1): {target_achieved}")
    
    if not target_achieved:
        print("\nIssues identified:")
        if final_gap > 1.0:
            print("  - Gap too high, likely constraint tightening too aggressive")
        elif final_gap > 0.5:
            print("  - Gap moderate, may need parameter tuning")
        else:
            print("  - Gap reasonable, may need more iterations")
    
    return results

if __name__ == "__main__":
    main()
