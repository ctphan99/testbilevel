#!/usr/bin/env python3
"""
Deep debug of lower-level solver to identify accuracy issues
"""

import torch
import numpy as np
import cvxpy as cp
from problem import StronglyConvexBilevelProblem

def debug_lower_level_solver():
    """Debug the lower-level solver step by step"""
    
    # Create problem with same parameters as comprehensive experiment
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.001,
        device='cpu',
        seed=42,
        strong_convex=True
    )
    
    print("=== LOWER-LEVEL SOLVER DEEP DEBUG ===")
    print(f"Problem setup:")
    print(f"  dim: {problem.dim}")
    print(f"  num_constraints: {problem.num_constraints}")
    print(f"  Q_lower shape: {problem.Q_lower.shape}")
    print(f"  A shape: {problem.A.shape}")
    print(f"  B shape: {problem.B.shape}")
    print(f"  b: {problem.b}")
    
    # Test with a specific x value
    x_test = torch.randn(5, requires_grad=True)
    print(f"\n=== TESTING WITH x = {x_test.detach().numpy()} ===")
    
    # Solve lower level
    y_opt, info = problem.solve_lower_level(x_test)
    print(f"Lower-level solution: {y_opt.detach().numpy()}")
    print(f"Solver info: {info}")
    
    # Check constraint violations
    h_val = problem.A @ x_test - problem.B @ y_opt - problem.b
    print(f"Constraint values h(x,y): {h_val.detach().numpy()}")
    print(f"Max constraint violation: {torch.max(torch.relu(h_val)).item()}")
    print(f"Number of active constraints: {(h_val > -1e-6).sum().item()}")
    
    # Check dual variables
    if 'lambda' in info:
        lambda_val = info['lambda']
        print(f"Dual variables λ: {lambda_val}")
        print(f"Number of active duals: {(lambda_val > 1e-6).sum().item()}")
    
    # Check KKT conditions
    print(f"\n=== KKT CONDITIONS CHECK ===")
    
    # Stationarity: ∇_y g(x,y) + λ^T ∇_y h(x,y) = 0
    grad_g = problem.c_lower + problem.Q_lower @ y_opt
    if 'lambda' in info:
        grad_h = -problem.B.T  # ∇_y h = -B^T
        stationarity = grad_g + grad_h @ info['lambda']
        print(f"Stationarity residual: {torch.norm(stationarity).item()}")
    
    # Complementary slackness: λ_i * h_i = 0
    if 'lambda' in info:
        comp_slack = info['lambda'] * h_val
        print(f"Complementary slackness residual: {torch.norm(comp_slack).item()}")
    
    # Primal feasibility: h(x,y) ≤ 0
    primal_feas = torch.max(torch.relu(h_val))
    print(f"Primal feasibility violation: {primal_feas.item()}")
    
    # Dual feasibility: λ ≥ 0
    if 'lambda' in info:
        dual_feas = torch.min(info['lambda'])
        print(f"Dual feasibility violation: {min(0, dual_feas.item())}")
    
    # Test with different x values to see if solver is consistent
    print(f"\n=== TESTING CONSISTENCY WITH DIFFERENT x VALUES ===")
    for i in range(3):
        x_new = torch.randn(5, requires_grad=True)
        y_new, info_new = problem.solve_lower_level(x_new)
        h_new = problem.A @ x_new - problem.B @ y_new - problem.b
        print(f"x_{i+1}: {x_new.detach().numpy()}")
        print(f"  y_{i+1}: {y_new.detach().numpy()}")
        print(f"  h_{i+1}: {h_new.detach().numpy()}")
        print(f"  max_violation_{i+1}: {torch.max(torch.relu(h_new)).item()}")
        if 'lambda' in info_new:
            print(f"  λ_{i+1}: {info_new['lambda']}")
    
    # Check if constraints are ever active
    print(f"\n=== CONSTRAINT ACTIVITY ANALYSIS ===")
    active_count = 0
    for i in range(10):
        x_rand = torch.randn(5, requires_grad=True)
        y_rand, info_rand = problem.solve_lower_level(x_rand)
        h_rand = problem.A @ x_rand - problem.B @ y_rand - problem.b
        max_viol = torch.max(torch.relu(h_rand)).item()
        if max_viol > 1e-6:
            active_count += 1
            print(f"  Active constraints found at x_{i+1}: max_violation = {max_viol}")
    
    print(f"Constraints active in {active_count}/10 random samples")
    
    # Check unconstrained optimum vs constrained optimum
    print(f"\n=== UNCONSTRAINED vs CONSTRAINED COMPARISON ===")
    y_unconstrained = -torch.linalg.solve(problem.Q_lower, problem.c_lower)
    h_unconstrained = problem.A @ x_test - problem.B @ y_unconstrained - problem.b
    print(f"Unconstrained y*: {y_unconstrained.detach().numpy()}")
    print(f"Constrained y*: {y_opt.detach().numpy()}")
    print(f"Difference: {torch.norm(y_opt - y_unconstrained).item()}")
    print(f"Unconstrained h: {h_unconstrained.detach().numpy()}")
    print(f"Constrained h: {h_val.detach().numpy()}")
    print(f"Unconstrained max violation: {torch.max(torch.relu(h_unconstrained)).item()}")
    print(f"Constrained max violation: {torch.max(torch.relu(h_val)).item()}")

if __name__ == '__main__':
    debug_lower_level_solver()

Deep debug of lower-level solver to identify accuracy issues
"""

import torch
import numpy as np
import cvxpy as cp
from problem import StronglyConvexBilevelProblem

def debug_lower_level_solver():
    """Debug the lower-level solver step by step"""
    
    # Create problem with same parameters as comprehensive experiment
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.001,
        device='cpu',
        seed=42,
        strong_convex=True
    )
    
    print("=== LOWER-LEVEL SOLVER DEEP DEBUG ===")
    print(f"Problem setup:")
    print(f"  dim: {problem.dim}")
    print(f"  num_constraints: {problem.num_constraints}")
    print(f"  Q_lower shape: {problem.Q_lower.shape}")
    print(f"  A shape: {problem.A.shape}")
    print(f"  B shape: {problem.B.shape}")
    print(f"  b: {problem.b}")
    
    # Test with a specific x value
    x_test = torch.randn(5, requires_grad=True)
    print(f"\n=== TESTING WITH x = {x_test.detach().numpy()} ===")
    
    # Solve lower level
    y_opt, info = problem.solve_lower_level(x_test)
    print(f"Lower-level solution: {y_opt.detach().numpy()}")
    print(f"Solver info: {info}")
    
    # Check constraint violations
    h_val = problem.A @ x_test - problem.B @ y_opt - problem.b
    print(f"Constraint values h(x,y): {h_val.detach().numpy()}")
    print(f"Max constraint violation: {torch.max(torch.relu(h_val)).item()}")
    print(f"Number of active constraints: {(h_val > -1e-6).sum().item()}")
    
    # Check dual variables
    if 'lambda' in info:
        lambda_val = info['lambda']
        print(f"Dual variables λ: {lambda_val}")
        print(f"Number of active duals: {(lambda_val > 1e-6).sum().item()}")
    
    # Check KKT conditions
    print(f"\n=== KKT CONDITIONS CHECK ===")
    
    # Stationarity: ∇_y g(x,y) + λ^T ∇_y h(x,y) = 0
    grad_g = problem.c_lower + problem.Q_lower @ y_opt
    if 'lambda' in info:
        grad_h = -problem.B.T  # ∇_y h = -B^T
        stationarity = grad_g + grad_h @ info['lambda']
        print(f"Stationarity residual: {torch.norm(stationarity).item()}")
    
    # Complementary slackness: λ_i * h_i = 0
    if 'lambda' in info:
        comp_slack = info['lambda'] * h_val
        print(f"Complementary slackness residual: {torch.norm(comp_slack).item()}")
    
    # Primal feasibility: h(x,y) ≤ 0
    primal_feas = torch.max(torch.relu(h_val))
    print(f"Primal feasibility violation: {primal_feas.item()}")
    
    # Dual feasibility: λ ≥ 0
    if 'lambda' in info:
        dual_feas = torch.min(info['lambda'])
        print(f"Dual feasibility violation: {min(0, dual_feas.item())}")
    
    # Test with different x values to see if solver is consistent
    print(f"\n=== TESTING CONSISTENCY WITH DIFFERENT x VALUES ===")
    for i in range(3):
        x_new = torch.randn(5, requires_grad=True)
        y_new, info_new = problem.solve_lower_level(x_new)
        h_new = problem.A @ x_new - problem.B @ y_new - problem.b
        print(f"x_{i+1}: {x_new.detach().numpy()}")
        print(f"  y_{i+1}: {y_new.detach().numpy()}")
        print(f"  h_{i+1}: {h_new.detach().numpy()}")
        print(f"  max_violation_{i+1}: {torch.max(torch.relu(h_new)).item()}")
        if 'lambda' in info_new:
            print(f"  λ_{i+1}: {info_new['lambda']}")
    
    # Check if constraints are ever active
    print(f"\n=== CONSTRAINT ACTIVITY ANALYSIS ===")
    active_count = 0
    for i in range(10):
        x_rand = torch.randn(5, requires_grad=True)
        y_rand, info_rand = problem.solve_lower_level(x_rand)
        h_rand = problem.A @ x_rand - problem.B @ y_rand - problem.b
        max_viol = torch.max(torch.relu(h_rand)).item()
        if max_viol > 1e-6:
            active_count += 1
            print(f"  Active constraints found at x_{i+1}: max_violation = {max_viol}")
    
    print(f"Constraints active in {active_count}/10 random samples")
    
    # Check unconstrained optimum vs constrained optimum
    print(f"\n=== UNCONSTRAINED vs CONSTRAINED COMPARISON ===")
    y_unconstrained = -torch.linalg.solve(problem.Q_lower, problem.c_lower)
    h_unconstrained = problem.A @ x_test - problem.B @ y_unconstrained - problem.b
    print(f"Unconstrained y*: {y_unconstrained.detach().numpy()}")
    print(f"Constrained y*: {y_opt.detach().numpy()}")
    print(f"Difference: {torch.norm(y_opt - y_unconstrained).item()}")
    print(f"Unconstrained h: {h_unconstrained.detach().numpy()}")
    print(f"Constrained h: {h_val.detach().numpy()}")
    print(f"Unconstrained max violation: {torch.max(torch.relu(h_unconstrained)).item()}")
    print(f"Constrained max violation: {torch.max(torch.relu(h_val)).item()}")

if __name__ == '__main__':
    debug_lower_level_solver()
