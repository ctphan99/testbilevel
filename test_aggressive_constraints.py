#!/usr/bin/env python3
"""
Test aggressive constraint tightening
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem

def test_aggressive_constraints():
    """Test the more aggressive constraint tightening"""
    
    # Create problem with same parameters as comprehensive experiment
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.001,
        device='cpu',
        seed=42,
        strong_convex=True
    )
    
    print("=== BEFORE AGGRESSIVE CONSTRAINT TIGHTENING ===")
    print(f"Original b: {problem.b}")
    print(f"Original Q_lower eigenvalues: {torch.linalg.eigvals(problem.Q_lower).real}")
    
    # Check constraint feasibility at origin
    origin_constraint = -problem.b
    max_violation = torch.max(origin_constraint)
    print(f"Constraint feasibility at origin: max_violation = {max_violation}")
    
    print("\n=== AFTER AGGRESSIVE CONSTRAINT TIGHTENING ===")
    # Apply the more aggressive tightening
    problem.b = problem.b * 0.01  # Much more restrictive
    problem.Q_lower = problem.Q_lower * 5.0  # Much steeper
    
    print(f"Tightened b: {problem.b}")
    print(f"Tightened Q_lower eigenvalues: {torch.linalg.eigvals(problem.Q_lower).real}")
    
    # Check constraint feasibility at origin after tightening
    origin_constraint_tightened = -problem.b
    max_violation_tightened = torch.max(origin_constraint_tightened)
    print(f"Constraint feasibility at origin (tightened): max_violation = {max_violation_tightened}")
    
    # Test with random x points
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
    has_active = test_aggressive_constraints()
    print(f"\nConstraints are now active: {has_active}")
"""
Test aggressive constraint tightening
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem

def test_aggressive_constraints():
    """Test the more aggressive constraint tightening"""
    
    # Create problem with same parameters as comprehensive experiment
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.001,
        device='cpu',
        seed=42,
        strong_convex=True
    )
    
    print("=== BEFORE AGGRESSIVE CONSTRAINT TIGHTENING ===")
    print(f"Original b: {problem.b}")
    print(f"Original Q_lower eigenvalues: {torch.linalg.eigvals(problem.Q_lower).real}")
    
    # Check constraint feasibility at origin
    origin_constraint = -problem.b
    max_violation = torch.max(origin_constraint)
    print(f"Constraint feasibility at origin: max_violation = {max_violation}")
    
    print("\n=== AFTER AGGRESSIVE CONSTRAINT TIGHTENING ===")
    # Apply the more aggressive tightening
    problem.b = problem.b * 0.01  # Much more restrictive
    problem.Q_lower = problem.Q_lower * 5.0  # Much steeper
    
    print(f"Tightened b: {problem.b}")
    print(f"Tightened Q_lower eigenvalues: {torch.linalg.eigvals(problem.Q_lower).real}")
    
    # Check constraint feasibility at origin after tightening
    origin_constraint_tightened = -problem.b
    max_violation_tightened = torch.max(origin_constraint_tightened)
    print(f"Constraint feasibility at origin (tightened): max_violation = {max_violation_tightened}")
    
    # Test with random x points
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
    has_active = test_aggressive_constraints()
    print(f"\nConstraints are now active: {has_active}")
