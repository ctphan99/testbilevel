#!/usr/bin/env python3
"""
Test the constraint structure fix
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem

def test_constraint_fix():
    """Test the constraint structure fix"""
    
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.001,
        device='cpu',
        seed=42,
        strong_convex=True
    )
    
    print("=== BEFORE CONSTRAINT FIX ===")
    print(f"B matrix norm: {torch.norm(problem.B)}")
    print(f"A matrix norm: {torch.norm(problem.A)}")
    print(f"b vector: {problem.b}")
    
    # Test with original constraints
    x_test = torch.randn(5, requires_grad=True)
    y_opt, _ = problem.solve_lower_level(x_test)
    constraint_violation = problem.A @ x_test - problem.B @ y_opt - problem.b
    max_violation = torch.max(constraint_violation)
    print(f"Original max violation: {max_violation:.6f}")
    
    print("\n=== AFTER CONSTRAINT FIX ===")
    # Apply the fix
    problem.B = problem.B * 10.0  # Scale up B
    problem.b = problem.b - 0.1   # Make more restrictive
    problem.Q_lower = problem.Q_lower * 2.0  # Steepen objective
    
    print(f"B matrix norm (scaled): {torch.norm(problem.B)}")
    print(f"A matrix norm: {torch.norm(problem.A)}")
    print(f"b vector (adjusted): {problem.b}")
    
    # Test with fixed constraints
    y_opt_fixed, _ = problem.solve_lower_level(x_test)
    constraint_violation_fixed = problem.A @ x_test - problem.B @ y_opt_fixed - problem.b
    max_violation_fixed = torch.max(constraint_violation_fixed)
    print(f"Fixed max violation: {max_violation_fixed:.6f}")
    
    print(f"\nA @ x_test: {problem.A @ x_test}")
    print(f"B @ y_opt_fixed: {problem.B @ y_opt_fixed}")
    print(f"b: {problem.b}")
    
    # Test with multiple random points
    print("\n=== TESTING WITH RANDOM X POINTS ===")
    active_count = 0
    for i in range(10):
        x_random = torch.randn(5, requires_grad=True)
        y_opt, _ = problem.solve_lower_level(x_random)
        constraint_violation = problem.A @ x_random - problem.B @ y_opt - problem.b
        max_violation = torch.max(constraint_violation)
        
        # Check if constraints are active
        active_constraints = constraint_violation > 1e-6
        num_active = torch.sum(active_constraints)
        
        if num_active > 0:
            active_count += 1
            
        print(f"Random x {i+1}: max_violation = {max_violation:.6f}, active = {num_active}/{len(constraint_violation)}")
    
    print(f"\nSummary: {active_count}/10 random points had active constraints")
    return active_count > 0

if __name__ == "__main__":
    has_active = test_constraint_fix()
    print(f"\nConstraints are now active: {has_active}")
"""
Test the constraint structure fix
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem

def test_constraint_fix():
    """Test the constraint structure fix"""
    
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.001,
        device='cpu',
        seed=42,
        strong_convex=True
    )
    
    print("=== BEFORE CONSTRAINT FIX ===")
    print(f"B matrix norm: {torch.norm(problem.B)}")
    print(f"A matrix norm: {torch.norm(problem.A)}")
    print(f"b vector: {problem.b}")
    
    # Test with original constraints
    x_test = torch.randn(5, requires_grad=True)
    y_opt, _ = problem.solve_lower_level(x_test)
    constraint_violation = problem.A @ x_test - problem.B @ y_opt - problem.b
    max_violation = torch.max(constraint_violation)
    print(f"Original max violation: {max_violation:.6f}")
    
    print("\n=== AFTER CONSTRAINT FIX ===")
    # Apply the fix
    problem.B = problem.B * 10.0  # Scale up B
    problem.b = problem.b - 0.1   # Make more restrictive
    problem.Q_lower = problem.Q_lower * 2.0  # Steepen objective
    
    print(f"B matrix norm (scaled): {torch.norm(problem.B)}")
    print(f"A matrix norm: {torch.norm(problem.A)}")
    print(f"b vector (adjusted): {problem.b}")
    
    # Test with fixed constraints
    y_opt_fixed, _ = problem.solve_lower_level(x_test)
    constraint_violation_fixed = problem.A @ x_test - problem.B @ y_opt_fixed - problem.b
    max_violation_fixed = torch.max(constraint_violation_fixed)
    print(f"Fixed max violation: {max_violation_fixed:.6f}")
    
    print(f"\nA @ x_test: {problem.A @ x_test}")
    print(f"B @ y_opt_fixed: {problem.B @ y_opt_fixed}")
    print(f"b: {problem.b}")
    
    # Test with multiple random points
    print("\n=== TESTING WITH RANDOM X POINTS ===")
    active_count = 0
    for i in range(10):
        x_random = torch.randn(5, requires_grad=True)
        y_opt, _ = problem.solve_lower_level(x_random)
        constraint_violation = problem.A @ x_random - problem.B @ y_opt - problem.b
        max_violation = torch.max(constraint_violation)
        
        # Check if constraints are active
        active_constraints = constraint_violation > 1e-6
        num_active = torch.sum(active_constraints)
        
        if num_active > 0:
            active_count += 1
            
        print(f"Random x {i+1}: max_violation = {max_violation:.6f}, active = {num_active}/{len(constraint_violation)}")
    
    print(f"\nSummary: {active_count}/10 random points had active constraints")
    return active_count > 0

if __name__ == "__main__":
    has_active = test_constraint_fix()
    print(f"\nConstraints are now active: {has_active}")
