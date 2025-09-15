#!/usr/bin/env python3
"""
Debug the constraint structure to understand why constraints are never active
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem

def debug_constraint_structure():
    """Debug the constraint structure"""
    
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.001,
        device='cpu',
        seed=42,
        strong_convex=True
    )
    
    print("=== CONSTRAINT STRUCTURE ANALYSIS ===")
    print(f"Constraint matrix A shape: {problem.A.shape}")
    print(f"Constraint matrix B shape: {problem.B.shape}")
    print(f"Constraint vector b: {problem.b}")
    
    print(f"\nA matrix:\n{problem.A}")
    print(f"\nB matrix:\n{problem.B}")
    
    # Check if B is invertible
    try:
        B_inv = torch.linalg.inv(problem.B)
        print(f"\nB is invertible: True")
        print(f"B condition number: {torch.linalg.cond(problem.B):.2e}")
    except:
        print(f"\nB is invertible: False")
    
    # Check constraint structure: Ax - By - b <= 0
    # This means: By >= Ax - b
    # For the unconstrained optimum, we need to understand what y* looks like
    
    print("\n=== UNCONSTRAINED OPTIMUM ANALYSIS ===")
    x_test = torch.randn(5, requires_grad=True)
    
    # The unconstrained lower-level problem is:
    # min_y 0.5 * y^T Q_lower y + (c_lower + P^T x + noise)^T y
    # Solution: y* = -Q_lower^{-1} * (c_lower + P^T x + noise)
    
    # Let's compute this manually
    c_lower = problem.c_lower
    P = problem.P
    noise = torch.randn_like(x_test) * problem.noise_std
    
    y_unconstrained = -torch.linalg.solve(problem.Q_lower, c_lower + P.T @ x_test + noise)
    
    print(f"x_test: {x_test}")
    print(f"y_unconstrained: {y_unconstrained}")
    
    # Check constraint violation
    constraint_violation = problem.A @ x_test - problem.B @ y_unconstrained - problem.b
    print(f"Constraint violation: {constraint_violation}")
    print(f"Max violation: {torch.max(constraint_violation)}")
    
    # The issue might be that B @ y_unconstrained is too small compared to A @ x_test
    print(f"\nA @ x_test: {problem.A @ x_test}")
    print(f"B @ y_unconstrained: {problem.B @ y_unconstrained}")
    print(f"b: {problem.b}")
    
    # Check if the problem is that B is too small
    print(f"\nB matrix norm: {torch.norm(problem.B)}")
    print(f"A matrix norm: {torch.norm(problem.A)}")
    print(f"b vector norm: {torch.norm(problem.b)}")
    
    # Try to force constraint violation by making b very negative
    print("\n=== FORCING CONSTRAINT VIOLATION ===")
    problem.b = problem.b - 10.0  # Make b much more negative
    constraint_violation_forced = problem.A @ x_test - problem.B @ y_unconstrained - problem.b
    print(f"Constraint violation (forced): {constraint_violation_forced}")
    print(f"Max violation (forced): {torch.max(constraint_violation_forced)}")

if __name__ == "__main__":
    debug_constraint_structure()
"""
Debug the constraint structure to understand why constraints are never active
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem

def debug_constraint_structure():
    """Debug the constraint structure"""
    
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.001,
        device='cpu',
        seed=42,
        strong_convex=True
    )
    
    print("=== CONSTRAINT STRUCTURE ANALYSIS ===")
    print(f"Constraint matrix A shape: {problem.A.shape}")
    print(f"Constraint matrix B shape: {problem.B.shape}")
    print(f"Constraint vector b: {problem.b}")
    
    print(f"\nA matrix:\n{problem.A}")
    print(f"\nB matrix:\n{problem.B}")
    
    # Check if B is invertible
    try:
        B_inv = torch.linalg.inv(problem.B)
        print(f"\nB is invertible: True")
        print(f"B condition number: {torch.linalg.cond(problem.B):.2e}")
    except:
        print(f"\nB is invertible: False")
    
    # Check constraint structure: Ax - By - b <= 0
    # This means: By >= Ax - b
    # For the unconstrained optimum, we need to understand what y* looks like
    
    print("\n=== UNCONSTRAINED OPTIMUM ANALYSIS ===")
    x_test = torch.randn(5, requires_grad=True)
    
    # The unconstrained lower-level problem is:
    # min_y 0.5 * y^T Q_lower y + (c_lower + P^T x + noise)^T y
    # Solution: y* = -Q_lower^{-1} * (c_lower + P^T x + noise)
    
    # Let's compute this manually
    c_lower = problem.c_lower
    P = problem.P
    noise = torch.randn_like(x_test) * problem.noise_std
    
    y_unconstrained = -torch.linalg.solve(problem.Q_lower, c_lower + P.T @ x_test + noise)
    
    print(f"x_test: {x_test}")
    print(f"y_unconstrained: {y_unconstrained}")
    
    # Check constraint violation
    constraint_violation = problem.A @ x_test - problem.B @ y_unconstrained - problem.b
    print(f"Constraint violation: {constraint_violation}")
    print(f"Max violation: {torch.max(constraint_violation)}")
    
    # The issue might be that B @ y_unconstrained is too small compared to A @ x_test
    print(f"\nA @ x_test: {problem.A @ x_test}")
    print(f"B @ y_unconstrained: {problem.B @ y_unconstrained}")
    print(f"b: {problem.b}")
    
    # Check if the problem is that B is too small
    print(f"\nB matrix norm: {torch.norm(problem.B)}")
    print(f"A matrix norm: {torch.norm(problem.A)}")
    print(f"b vector norm: {torch.norm(problem.b)}")
    
    # Try to force constraint violation by making b very negative
    print("\n=== FORCING CONSTRAINT VIOLATION ===")
    problem.b = problem.b - 10.0  # Make b much more negative
    constraint_violation_forced = problem.A @ x_test - problem.B @ y_unconstrained - problem.b
    print(f"Constraint violation (forced): {constraint_violation_forced}")
    print(f"Max violation (forced): {torch.max(constraint_violation_forced)}")

if __name__ == "__main__":
    debug_constraint_structure()
