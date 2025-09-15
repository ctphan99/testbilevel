#!/usr/bin/env python3
"""
Deep diagnostic script for lower-level solver constraint handling
Based on web search insights about how solvers handle constraints
"""

import torch
import numpy as np
import cvxpy as cp
from problem import StronglyConvexBilevelProblem
import matplotlib.pyplot as plt

def analyze_constraint_structure(problem):
    """Analyze the constraint structure in detail"""
    print("=" * 80)
    print("CONSTRAINT STRUCTURE ANALYSIS")
    print("=" * 80)
    
    # Problem dimensions
    print(f"Problem dimensions:")
    print(f"  x dimension: {problem.dim}")
    print(f"  y dimension: {problem.dim}")
    print(f"  constraint dimension: {problem.num_constraints}")
    
    # Constraint matrices
    print(f"\nConstraint matrices:")
    print(f"  A shape: {problem.A.shape}")
    print(f"  B shape: {problem.B.shape}")
    print(f"  b shape: {problem.b.shape}")
    
    # Matrix properties
    print(f"\nMatrix properties:")
    print(f"  A rank: {torch.linalg.matrix_rank(problem.A)}")
    print(f"  B rank: {torch.linalg.matrix_rank(problem.B)}")
    print(f"  B condition number: {torch.linalg.cond(problem.B)}")
    
    # Constraint feasibility
    print(f"\nConstraint feasibility:")
    print(f"  b values: {problem.b}")
    print(f"  b norm: {torch.norm(problem.b)}")
    
    return problem

def test_constraint_activity_with_perturbations(problem, num_tests=10):
    """Test constraint activity with various x perturbations"""
    print("\n" + "=" * 80)
    print("CONSTRAINT ACTIVITY TESTING WITH PERTURBATIONS")
    print("=" * 80)
    
    results = []
    
    for i in range(num_tests):
        # Generate random x
        x_test = torch.randn(problem.dim, dtype=torch.float64)
        
        # Solve lower level
        y_opt, info = problem.solve_lower_level(x_test)
        dual_vars = info.get('lambda', None)
        
        # Compute constraint values
        constraint_values = problem.A @ x_test - problem.B @ y_opt - problem.b
        
        # Check constraint activity
        active_constraints = constraint_values >= -1e-6  # Small tolerance
        num_active = active_constraints.sum().item()
        
        # Check dual variables
        dual_norm = torch.norm(dual_vars) if dual_vars is not None else 0.0
        
        results.append({
            'x': x_test,
            'y_opt': y_opt,
            'constraint_values': constraint_values,
            'active_constraints': active_constraints,
            'num_active': num_active,
            'dual_vars': dual_vars,
            'dual_norm': dual_norm
        })
        
        print(f"Test {i+1:2d}: Active={num_active:2d}, Dual norm={dual_norm:.6f}, Max constraint={constraint_values.max():.6f}")
    
    return results

def analyze_cvxpy_solver_behavior(problem, x_test):
    """Analyze how CVXPy solver handles the constraint problem"""
    print("\n" + "=" * 80)
    print("CVXPY SOLVER BEHAVIOR ANALYSIS")
    print("=" * 80)
    
    # Convert to numpy for CVXPy
    A_np = problem.A.detach().numpy()
    B_np = problem.B.detach().numpy()
    b_np = problem.b.detach().numpy()
    c_np = problem.c_lower.detach().numpy()
    Q_np = problem.Q_lower.detach().numpy()
    x_np = x_test.detach().numpy()
    
    # Create CVXPy problem
    y = cp.Variable(problem.dim)
    
    # Objective: 0.5 * y^T Q y + c^T y
    objective = cp.Minimize(0.5 * cp.quad_form(y, Q_np) + c_np.T @ y)
    
    # Constraints: A @ x - B @ y <= b
    constraints = [A_np @ x_np - B_np @ y <= b_np]
    
    problem_cvx = cp.Problem(objective, constraints)
    
    print(f"CVXPy problem setup:")
    print(f"  Objective: 0.5 * y^T Q y + c^T y")
    print(f"  Constraints: A @ x - B @ y <= b")
    print(f"  x = {x_np}")
    print(f"  b = {b_np}")
    
    # Test different solvers
    solvers_to_test = ['SCS', 'ECOS', 'OSQP', 'CLARABEL']
    
    for solver_name in solvers_to_test:
        try:
            print(f"\n--- Testing {solver_name} ---")
            
            # Solve with specific solver
            if solver_name == 'SCS':
                problem_cvx.solve(solver=cp.SCS, verbose=True, eps=1e-8, max_iters=10000)
            elif solver_name == 'ECOS':
                problem_cvx.solve(solver=cp.ECOS, verbose=True, max_iters=10000)
            elif solver_name == 'OSQP':
                problem_cvx.solve(solver=cp.OSQP, verbose=True, eps_abs=1e-8, eps_rel=1e-8, max_iter=10000)
            elif solver_name == 'CLARABEL':
                problem_cvx.solve(solver=cp.CLARABEL, verbose=True, eps_abs=1e-8, eps_rel=1e-8, max_iter=10000)
            
            if problem_cvx.status == 'optimal':
                y_sol = y.value
                dual_vars = constraints[0].dual_value
                
                # Compute constraint values
                constraint_values = A_np @ x_np - B_np @ y_sol - b_np
                
                print(f"  Status: {problem_cvx.status}")
                print(f"  Objective value: {problem_cvx.value:.6f}")
                print(f"  Solution norm: {np.linalg.norm(y_sol):.6f}")
                print(f"  Constraint values: {constraint_values}")
                print(f"  Max constraint violation: {np.max(constraint_values):.6f}")
                print(f"  Dual variables: {dual_vars}")
                print(f"  Dual norm: {np.linalg.norm(dual_vars):.6f}")
                
                # Check if constraints are active
                active_tol = 1e-6
                active_constraints = constraint_values >= -active_tol
                num_active = np.sum(active_constraints)
                print(f"  Active constraints: {num_active}/{len(constraint_values)}")
                
            else:
                print(f"  Status: {problem_cvx.status}")
                print(f"  Problem not solved optimally")
                
        except Exception as e:
            print(f"  Error with {solver_name}: {e}")
    
    return problem_cvx

def test_constraint_tightening_strategies(problem):
    """Test different constraint tightening strategies"""
    print("\n" + "=" * 80)
    print("CONSTRAINT TIGHTENING STRATEGIES TESTING")
    print("=" * 80)
    
    strategies = [
        ("Original", lambda p: p),
        ("Tighten b", lambda p: setattr(p, 'b', p.b - 0.1) or p),
        ("Scale B", lambda p: setattr(p, 'B', p.B * 2.0) or p),
        ("Scale Q", lambda p: setattr(p, 'Q_lower', p.Q_lower * 1.5) or p),
        ("Combined 1", lambda p: (setattr(p, 'b', p.b - 0.1), setattr(p, 'B', p.B * 2.0), setattr(p, 'Q_lower', p.Q_lower * 1.5)) or p),
        ("Aggressive", lambda p: (setattr(p, 'b', p.b - 0.5), setattr(p, 'B', p.B * 5.0), setattr(p, 'Q_lower', p.Q_lower * 3.0)) or p),
    ]
    
    x_test = torch.randn(problem.dim, dtype=torch.float64)
    
    for strategy_name, strategy_func in strategies:
        print(f"\n--- {strategy_name} ---")
        
        # Apply strategy
        strategy_func(problem)
        
        # Solve lower level
        y_opt, info = problem.solve_lower_level(x_test)
        dual_vars = info.get('lambda', None)
        
        # Compute constraint values
        constraint_values = problem.A @ x_test - problem.B @ y_opt - problem.b
        
        # Check activity
        active_constraints = constraint_values >= -1e-6
        num_active = active_constraints.sum().item()
        dual_norm = torch.norm(dual_vars) if dual_vars is not None else 0.0
        
        print(f"  Active constraints: {num_active}/{len(constraint_values)}")
        print(f"  Dual norm: {dual_norm:.6f}")
        print(f"  Max constraint: {constraint_values.max():.6f}")
        print(f"  Min constraint: {constraint_values.min():.6f}")
        print(f"  Constraint values: {constraint_values}")

def main():
    """Main diagnostic function"""
    print("DEEP LOWER-LEVEL SOLVER CONSTRAINT HANDLING DIAGNOSTIC")
    print("Based on web search insights about solver constraint handling")
    print("=" * 80)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, seed=42)
    
    # Analyze constraint structure
    analyze_constraint_structure(problem)
    
    # Test constraint activity with perturbations
    results = test_constraint_activity_with_perturbations(problem, num_tests=5)
    
    # Analyze CVXPy solver behavior
    x_test = torch.randn(problem.dim, dtype=torch.float64)
    analyze_cvxpy_solver_behavior(problem, x_test)
    
    # Test constraint tightening strategies
    test_constraint_tightening_strategies(problem)
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
