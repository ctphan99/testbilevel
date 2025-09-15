#!/usr/bin/env python3
"""
Debug constraint feasibility to understand why constraints are violated
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem

def debug_constraint_feasibility():
    """Debug why constraints are being violated"""
    print("=" * 80)
    print("DEBUGGING CONSTRAINT FEASIBILITY")
    print("=" * 80)
    
    # Create problem with constraint tightening
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, seed=42)
    
    print("Original problem:")
    print(f"  b: {problem.b.detach().numpy()}")
    print(f"  B: {problem.B.detach().numpy()}")
    print(f"  Q_lower: {problem.Q_lower.detach().numpy()}")
    
    # Apply constraint tightening
    problem.b = problem.b - 0.2
    problem.B = problem.B * 2.5
    problem.Q_lower = problem.Q_lower * 1.8
    
    print("\nAfter constraint tightening:")
    print(f"  b: {problem.b.detach().numpy()}")
    print(f"  B: {problem.B.detach().numpy()}")
    print(f"  Q_lower: {problem.Q_lower.detach().numpy()}")
    
    # Test constraint feasibility at origin
    x_origin = torch.zeros(problem.dim, dtype=torch.float64)
    y_origin = torch.zeros(problem.dim, dtype=torch.float64)
    
    h_origin = problem.A @ x_origin - problem.B @ y_origin - problem.b
    print(f"\nConstraint values at origin (x=0, y=0):")
    print(f"  h = A@x - B@y - b = {h_origin.detach().numpy()}")
    print(f"  Max violation: {torch.max(h_origin).item():.6f}")
    
    # Test with a random x
    x_test = torch.randn(problem.dim, dtype=torch.float64)
    print(f"\nTesting with x = {x_test.detach().numpy()}")
    
    # Solve lower level
    y_opt, info = problem.solve_lower_level(x_test)
    dual_vars = info.get('lambda', None)
    
    print(f"Lower-level solution y* = {y_opt.detach().numpy()}")
    print(f"Dual variables λ* = {dual_vars.detach().numpy() if dual_vars is not None else 'None'}")
    
    # Check constraint values
    h_val = problem.A @ x_test - problem.B @ y_opt - problem.b
    print(f"Constraint values h = A@x - B@y - b = {h_val.detach().numpy()}")
    print(f"Max violation: {torch.max(h_val).item():.6f}")
    print(f"Min violation: {torch.min(h_val).item():.6f}")
    
    # Check if constraints are satisfied
    feasible = torch.all(h_val <= 1e-6)
    print(f"Constraints satisfied: {feasible}")
    
    # Try to find a feasible point
    print(f"\nTrying to find feasible point...")
    
    # Use CVXPY to solve the constrained problem
    import cvxpy as cp
    
    x_np = x_test.detach().numpy()
    A_np = problem.A.detach().numpy()
    B_np = problem.B.detach().numpy()
    b_np = problem.b.detach().numpy()
    Q_np = problem.Q_lower.detach().numpy()
    c_np = problem.c_lower.detach().numpy()
    P_np = problem.P.detach().numpy()
    
    # Create CVXPY problem
    y_cp = cp.Variable(problem.dim)
    objective = cp.Minimize(0.5 * cp.quad_form(y_cp, Q_np) + (c_np + P_np.T @ x_np) @ y_cp)
    constraints = [A_np @ x_np + B_np @ y_cp - b_np <= 0]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    print(f"CVXPY status: {prob.status}")
    if prob.status == "optimal":
        y_cvxpy = y_cp.value
        h_cvxpy = A_np @ x_np + B_np @ y_cvxpy - b_np
        print(f"CVXPY solution y* = {y_cvxpy}")
        print(f"CVXPY constraint values h = {h_cvxpy}")
        print(f"CVXPY max violation: {np.max(h_cvxpy):.6f}")
        print(f"CVXPY constraints satisfied: {np.all(h_cvxpy <= 1e-6)}")
    else:
        print(f"CVXPY failed with status: {prob.status}")
        
        # Check if the problem is infeasible
        print(f"\nChecking problem feasibility...")
        
        # Try to find any feasible y
        y_feas = cp.Variable(problem.dim)
        constraints_feas = [A_np @ x_np + B_np @ y_feas - b_np <= 0]
        prob_feas = cp.Problem(cp.Minimize(0), constraints_feas)
        prob_feas.solve()
        
        print(f"Feasibility check status: {prob_feas.status}")
        if prob_feas.status == "optimal":
            y_feas_val = y_feas.value
            h_feas = A_np @ x_np + B_np @ y_feas_val - b_np
            print(f"Feasible solution y = {y_feas_val}")
            print(f"Constraint values h = {h_feas}")
            print(f"Max violation: {np.max(h_feas):.6f}")
        else:
            print("Problem is infeasible!")

if __name__ == "__main__":
    debug_constraint_feasibility()
