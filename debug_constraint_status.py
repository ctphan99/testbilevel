#!/usr/bin/env python3
"""
Debug constraint status after tightening
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem

def debug_constraint_status():
    """Debug the constraint status after tightening"""
    
    # Create problem with same parameters as comprehensive experiment
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.001,
        device='cpu',
        seed=42,
        strong_convex=True
    )
    
    print("=== BEFORE CONSTRAINT TIGHTENING ===")
    print(f"Original b: {problem.b}")
    print(f"Original Q_lower eigenvalues: {torch.linalg.eigvals(problem.Q_lower).real}")
    
    # Check constraint feasibility at origin
    origin_constraint = -problem.b  # Since Ax - By = 0 at origin
    max_violation = torch.max(origin_constraint)
    print(f"Constraint feasibility at origin: max_violation = {max_violation}")
    
    # Check unconstrained optimum (solve without constraints)
    x_test = torch.zeros(5, requires_grad=True)
    # For unconstrained, we need to solve: min_y ||y||^2 + 2*y^T*Q_lower*y
    # This gives us y* = -Q_lower^{-1} * (x + noise)
    y_unconstrained = -torch.linalg.solve(problem.Q_lower, x_test + torch.randn_like(x_test) * problem.noise_std)
    constraint_at_unconstrained = problem.A @ x_test - problem.B @ y_unconstrained - problem.b
    max_violation_unconstrained = torch.max(constraint_at_unconstrained)
    print(f"Constraint at unconstrained optimum: max_violation = {max_violation_unconstrained}")
    
    print("\n=== AFTER CONSTRAINT TIGHTENING ===")
    # Apply the same tightening as in comprehensive experiment
    problem.b = problem.b * 0.1  # Tighten constraints
    problem.Q_lower = problem.Q_lower * 2.0  # Steepen LL objective
    
    print(f"Tightened b: {problem.b}")
    print(f"Tightened Q_lower eigenvalues: {torch.linalg.eigvals(problem.Q_lower).real}")
    
    # Check constraint feasibility at origin after tightening
    origin_constraint_tightened = -problem.b
    max_violation_tightened = torch.max(origin_constraint_tightened)
    print(f"Constraint feasibility at origin (tightened): max_violation = {max_violation_tightened}")
    
    # Check unconstrained optimum after tightening
    y_unconstrained_tightened = -torch.linalg.solve(problem.Q_lower, x_test + torch.randn_like(x_test) * problem.noise_std)
    constraint_at_unconstrained_tightened = problem.A @ x_test - problem.B @ y_unconstrained_tightened - problem.b
    max_violation_unconstrained_tightened = torch.max(constraint_at_unconstrained_tightened)
    print(f"Constraint at unconstrained optimum (tightened): max_violation = {max_violation_unconstrained_tightened}")
    
    # Test with a few random x points
    print("\n=== TESTING WITH RANDOM X POINTS ===")
    for i in range(3):
        x_random = torch.randn(5, requires_grad=True)
        y_opt, _ = problem.solve_lower_level(x_random)
        constraint_violation = problem.A @ x_random - problem.B @ y_opt - problem.b
        max_violation = torch.max(constraint_violation)
        print(f"Random x {i+1}: max_violation = {max_violation:.6f}")
        
        # Check if constraints are active
        active_constraints = constraint_violation > 1e-6
        num_active = torch.sum(active_constraints)
        print(f"  Active constraints: {num_active}/{len(constraint_violation)}")

if __name__ == "__main__":
    debug_constraint_status()
Debug constraint status after tightening
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem

def debug_constraint_status():
    """Debug the constraint status after tightening"""
    
    # Create problem with same parameters as comprehensive experiment
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.001,
        device='cpu',
        seed=42,
        strong_convex=True
    )
    
    print("=== BEFORE CONSTRAINT TIGHTENING ===")
    print(f"Original b: {problem.b}")
    print(f"Original Q_lower eigenvalues: {torch.linalg.eigvals(problem.Q_lower).real}")
    
    # Check constraint feasibility at origin
    origin_constraint = -problem.b  # Since Ax - By = 0 at origin
    max_violation = torch.max(origin_constraint)
    print(f"Constraint feasibility at origin: max_violation = {max_violation}")
    
    # Check unconstrained optimum (solve without constraints)
    x_test = torch.zeros(5, requires_grad=True)
    # For unconstrained, we need to solve: min_y ||y||^2 + 2*y^T*Q_lower*y
    # This gives us y* = -Q_lower^{-1} * (x + noise)
    y_unconstrained = -torch.linalg.solve(problem.Q_lower, x_test + torch.randn_like(x_test) * problem.noise_std)
    constraint_at_unconstrained = problem.A @ x_test - problem.B @ y_unconstrained - problem.b
    max_violation_unconstrained = torch.max(constraint_at_unconstrained)
    print(f"Constraint at unconstrained optimum: max_violation = {max_violation_unconstrained}")
    
    print("\n=== AFTER CONSTRAINT TIGHTENING ===")
    # Apply the same tightening as in comprehensive experiment
    problem.b = problem.b * 0.1  # Tighten constraints
    problem.Q_lower = problem.Q_lower * 2.0  # Steepen LL objective
    
    print(f"Tightened b: {problem.b}")
    print(f"Tightened Q_lower eigenvalues: {torch.linalg.eigvals(problem.Q_lower).real}")
    
    # Check constraint feasibility at origin after tightening
    origin_constraint_tightened = -problem.b
    max_violation_tightened = torch.max(origin_constraint_tightened)
    print(f"Constraint feasibility at origin (tightened): max_violation = {max_violation_tightened}")
    
    # Check unconstrained optimum after tightening
    y_unconstrained_tightened = -torch.linalg.solve(problem.Q_lower, x_test + torch.randn_like(x_test) * problem.noise_std)
    constraint_at_unconstrained_tightened = problem.A @ x_test - problem.B @ y_unconstrained_tightened - problem.b
    max_violation_unconstrained_tightened = torch.max(constraint_at_unconstrained_tightened)
    print(f"Constraint at unconstrained optimum (tightened): max_violation = {max_violation_unconstrained_tightened}")
    
    # Test with a few random x points
    print("\n=== TESTING WITH RANDOM X POINTS ===")
    for i in range(3):
        x_random = torch.randn(5, requires_grad=True)
        y_opt, _ = problem.solve_lower_level(x_random)
        constraint_violation = problem.A @ x_random - problem.B @ y_opt - problem.b
        max_violation = torch.max(constraint_violation)
        print(f"Random x {i+1}: max_violation = {max_violation:.6f}")
        
        # Check if constraints are active
        active_constraints = constraint_violation > 1e-6
        num_active = torch.sum(active_constraints)
        print(f"  Active constraints: {num_active}/{len(constraint_violation)}")

if __name__ == "__main__":
    debug_constraint_status()