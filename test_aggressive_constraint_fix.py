#!/usr/bin/env python3
"""
Test the aggressive constraint structure fix
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem

def test_aggressive_constraint_fix():
    """Test the aggressive constraint structure fix"""
    
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.001,
        device='cpu',
        seed=42,
        strong_convex=True
    )
    
    print("=== BEFORE AGGRESSIVE CONSTRAINT FIX ===")
    print(f"B matrix norm: {torch.norm(problem.B)}")
    print(f"A matrix norm: {torch.norm(problem.A)}")
    print(f"b vector: {problem.b}")
    print(f"c_lower: {problem.c_lower}")
    
    # Test with original constraints
    x_test = torch.randn(5, requires_grad=True)
    y_opt, _ = problem.solve_lower_level(x_test)
    constraint_violation = problem.A @ x_test - problem.B @ y_opt - problem.b
    max_violation = torch.max(constraint_violation)
    print(f"Original max violation: {max_violation:.6f}")
    
    print("\n=== AFTER AGGRESSIVE CONSTRAINT FIX ===")
    # Apply the aggressive fix
    problem.b = problem.b - 1.0      # Make much more restrictive
    problem.B = problem.B * 5.0      # Scale up B
    problem.Q_lower = problem.Q_lower * 3.0  # Steepen objective
    problem.c_lower = problem.c_lower * 2.0  # Modify linear term
    
    print(f"B matrix norm (scaled): {torch.norm(problem.B)}")
    print(f"A matrix norm: {torch.norm(problem.A)}")
    print(f"b vector (adjusted): {problem.b}")
    print(f"c_lower (scaled): {problem.c_lower}")
    
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
    violation_sum = 0.0
    for i in range(10):
        x_random = torch.randn(5, requires_grad=True)
        y_opt, _ = problem.solve_lower_level(x_random)
        constraint_violation = problem.A @ x_random - problem.B @ y_opt - problem.b
        max_violation = torch.max(constraint_violation)
        violation_sum += max_violation.item()
        
        # Check if constraints are active
        active_constraints = constraint_violation > 1e-6
        num_active = torch.sum(active_constraints)
        
        if num_active > 0:
            active_count += 1
            
        print(f"Random x {i+1}: max_violation = {max_violation:.6f}, active = {num_active}/{len(constraint_violation)}")
    
    avg_violation = violation_sum / 10
    print(f"\nSummary: {active_count}/10 random points had active constraints")
    print(f"Average max violation: {avg_violation:.6f}")
    return active_count > 0, avg_violation

if __name__ == "__main__":
    has_active, avg_violation = test_aggressive_constraint_fix()
    print(f"\nConstraints are now active: {has_active}")
    print(f"Average violation: {avg_violation:.6f}")
"""
Test the aggressive constraint structure fix
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem

def test_aggressive_constraint_fix():
    """Test the aggressive constraint structure fix"""
    
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.001,
        device='cpu',
        seed=42,
        strong_convex=True
    )
    
    print("=== BEFORE AGGRESSIVE CONSTRAINT FIX ===")
    print(f"B matrix norm: {torch.norm(problem.B)}")
    print(f"A matrix norm: {torch.norm(problem.A)}")
    print(f"b vector: {problem.b}")
    print(f"c_lower: {problem.c_lower}")
    
    # Test with original constraints
    x_test = torch.randn(5, requires_grad=True)
    y_opt, _ = problem.solve_lower_level(x_test)
    constraint_violation = problem.A @ x_test - problem.B @ y_opt - problem.b
    max_violation = torch.max(constraint_violation)
    print(f"Original max violation: {max_violation:.6f}")
    
    print("\n=== AFTER AGGRESSIVE CONSTRAINT FIX ===")
    # Apply the aggressive fix
    problem.b = problem.b - 1.0      # Make much more restrictive
    problem.B = problem.B * 5.0      # Scale up B
    problem.Q_lower = problem.Q_lower * 3.0  # Steepen objective
    problem.c_lower = problem.c_lower * 2.0  # Modify linear term
    
    print(f"B matrix norm (scaled): {torch.norm(problem.B)}")
    print(f"A matrix norm: {torch.norm(problem.A)}")
    print(f"b vector (adjusted): {problem.b}")
    print(f"c_lower (scaled): {problem.c_lower}")
    
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
    violation_sum = 0.0
    for i in range(10):
        x_random = torch.randn(5, requires_grad=True)
        y_opt, _ = problem.solve_lower_level(x_random)
        constraint_violation = problem.A @ x_random - problem.B @ y_opt - problem.b
        max_violation = torch.max(constraint_violation)
        violation_sum += max_violation.item()
        
        # Check if constraints are active
        active_constraints = constraint_violation > 1e-6
        num_active = torch.sum(active_constraints)
        
        if num_active > 0:
            active_count += 1
            
        print(f"Random x {i+1}: max_violation = {max_violation:.6f}, active = {num_active}/{len(constraint_violation)}")
    
    avg_violation = violation_sum / 10
    print(f"\nSummary: {active_count}/10 random points had active constraints")
    print(f"Average max violation: {avg_violation:.6f}")
    return active_count > 0, avg_violation

if __name__ == "__main__":
    has_active, avg_violation = test_aggressive_constraint_fix()
    print(f"\nConstraints are now active: {has_active}")
    print(f"Average violation: {avg_violation:.6f}")
