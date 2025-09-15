#!/usr/bin/env python3
"""
DEBUG CORRECTED GAP COMPUTATION
==============================

Test the corrected gap computation method to see why implicit component is still constant
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_corrected_gap_computation import F2CSACorrectedGapComputation
import warnings
warnings.filterwarnings('ignore')


def debug_corrected_gap():
    """
    Debug the corrected gap computation method
    """
    print("="*80)
    print("DEBUGGING CORRECTED GAP COMPUTATION")
    print("="*80)
    
    # Create problem and solver
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.0005, device='cpu', seed=42)
    solver = F2CSACorrectedGapComputation(problem=problem, device='cpu', seed=42, verbose=True)
    
    # Test point
    x = torch.randn(5, device='cpu', dtype=torch.float64) * 0.1
    print(f"Test point x: {x}")
    
    # Test corrected gap computation
    print(f"\nTesting corrected gap computation:")
    gap, direct, implicit = solver.compute_gap_corrected(x)
    print(f"  Gap: {gap:.6f}")
    print(f"  Direct: {direct:.6f}")
    print(f"  Implicit: {implicit:.6f}")
    
    # Test multiple times to see if implicit changes
    print(f"\nTesting multiple calls to see if implicit changes:")
    for i in range(5):
        x_test = torch.randn(5, device='cpu', dtype=torch.float64) * 0.1
        gap_test, direct_test, implicit_test = solver.compute_gap_corrected(x_test)
        print(f"  Call {i+1}: Gap={gap_test:.6f}, Direct={direct_test:.6f}, Implicit={implicit_test:.6f}")
    
    # Test if constraints are inactive
    print(f"\nTesting constraint status:")
    y_star, _ = problem.solve_lower_level(x)
    h_val = problem.A @ x - problem.B @ y_star - problem.b
    slacks = problem.B @ y_star - (problem.A @ x - problem.b)
    min_slack = torch.min(slacks).item()
    print(f"  Constraint slacks: {slacks}")
    print(f"  Min slack: {min_slack:.6f}")
    print(f"  Constraints inactive: {min_slack > 1e-6}")
    
    # Test analytical Jacobian
    if hasattr(problem, 'jacobian_dy_dx_if_inactive'):
        print(f"\nTesting analytical Jacobian:")
        J = problem.jacobian_dy_dx_if_inactive()
        print(f"  Jacobian shape: {J.shape}")
        print(f"  Jacobian:\n{J}")
        
        # Test implicit vector
        if hasattr(problem, 'implicit_vector_if_inactive'):
            v = problem.implicit_vector_if_inactive()
            print(f"  Implicit vector: {v}")
            print(f"  ||v||: {torch.norm(v).item():.6f}")
    
    # Test direct gradient computation
    print(f"\nTesting direct gradient computation:")
    x_copy = x.clone().requires_grad_(True)
    y_copy = y_star.clone().requires_grad_(True)
    
    f_val = problem.upper_objective(x_copy, y_copy, add_noise=True)
    grad_x_direct = torch.autograd.grad(f_val, x_copy, create_graph=True, retain_graph=True)[0]
    grad_y = torch.autograd.grad(f_val, y_copy, create_graph=True, retain_graph=True)[0]
    
    print(f"  grad_x_direct: {grad_x_direct}")
    print(f"  grad_y: {grad_y}")
    print(f"  ||grad_x_direct||: {torch.norm(grad_x_direct).item():.6f}")
    print(f"  ||grad_y||: {torch.norm(grad_y).item():.6f}")
    
    # Test analytical implicit gradient
    if min_slack > 1e-6 and hasattr(problem, 'jacobian_dy_dx_if_inactive'):
        print(f"\nTesting analytical implicit gradient:")
        J = problem.jacobian_dy_dx_if_inactive()
        implicit_analytical = J.T @ grad_y
        print(f"  Implicit analytical: {implicit_analytical}")
        print(f"  ||implicit_analytical||: {torch.norm(implicit_analytical).item():.6f}")
        
        # Compare with numerical
        print(f"\nTesting numerical implicit gradient:")
        implicit_numerical = solver.compute_implicit_gradient_numerical(x, y_star, grad_y)
        print(f"  Implicit numerical: {implicit_numerical}")
        print(f"  ||implicit_numerical||: {torch.norm(implicit_numerical).item():.6f}")
        
        # Check difference
        diff = torch.norm(implicit_analytical - implicit_numerical).item()
        print(f"  Difference: {diff:.6f}")
    
    return gap, direct, implicit


if __name__ == "__main__":
    debug_corrected_gap()
"""
DEBUG CORRECTED GAP COMPUTATION
==============================

Test the corrected gap computation method to see why implicit component is still constant
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_corrected_gap_computation import F2CSACorrectedGapComputation
import warnings
warnings.filterwarnings('ignore')


def debug_corrected_gap():
    """
    Debug the corrected gap computation method
    """
    print("="*80)
    print("DEBUGGING CORRECTED GAP COMPUTATION")
    print("="*80)
    
    # Create problem and solver
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.0005, device='cpu', seed=42)
    solver = F2CSACorrectedGapComputation(problem=problem, device='cpu', seed=42, verbose=True)
    
    # Test point
    x = torch.randn(5, device='cpu', dtype=torch.float64) * 0.1
    print(f"Test point x: {x}")
    
    # Test corrected gap computation
    print(f"\nTesting corrected gap computation:")
    gap, direct, implicit = solver.compute_gap_corrected(x)
    print(f"  Gap: {gap:.6f}")
    print(f"  Direct: {direct:.6f}")
    print(f"  Implicit: {implicit:.6f}")
    
    # Test multiple times to see if implicit changes
    print(f"\nTesting multiple calls to see if implicit changes:")
    for i in range(5):
        x_test = torch.randn(5, device='cpu', dtype=torch.float64) * 0.1
        gap_test, direct_test, implicit_test = solver.compute_gap_corrected(x_test)
        print(f"  Call {i+1}: Gap={gap_test:.6f}, Direct={direct_test:.6f}, Implicit={implicit_test:.6f}")
    
    # Test if constraints are inactive
    print(f"\nTesting constraint status:")
    y_star, _ = problem.solve_lower_level(x)
    h_val = problem.A @ x - problem.B @ y_star - problem.b
    slacks = problem.B @ y_star - (problem.A @ x - problem.b)
    min_slack = torch.min(slacks).item()
    print(f"  Constraint slacks: {slacks}")
    print(f"  Min slack: {min_slack:.6f}")
    print(f"  Constraints inactive: {min_slack > 1e-6}")
    
    # Test analytical Jacobian
    if hasattr(problem, 'jacobian_dy_dx_if_inactive'):
        print(f"\nTesting analytical Jacobian:")
        J = problem.jacobian_dy_dx_if_inactive()
        print(f"  Jacobian shape: {J.shape}")
        print(f"  Jacobian:\n{J}")
        
        # Test implicit vector
        if hasattr(problem, 'implicit_vector_if_inactive'):
            v = problem.implicit_vector_if_inactive()
            print(f"  Implicit vector: {v}")
            print(f"  ||v||: {torch.norm(v).item():.6f}")
    
    # Test direct gradient computation
    print(f"\nTesting direct gradient computation:")
    x_copy = x.clone().requires_grad_(True)
    y_copy = y_star.clone().requires_grad_(True)
    
    f_val = problem.upper_objective(x_copy, y_copy, add_noise=True)
    grad_x_direct = torch.autograd.grad(f_val, x_copy, create_graph=True, retain_graph=True)[0]
    grad_y = torch.autograd.grad(f_val, y_copy, create_graph=True, retain_graph=True)[0]
    
    print(f"  grad_x_direct: {grad_x_direct}")
    print(f"  grad_y: {grad_y}")
    print(f"  ||grad_x_direct||: {torch.norm(grad_x_direct).item():.6f}")
    print(f"  ||grad_y||: {torch.norm(grad_y).item():.6f}")
    
    # Test analytical implicit gradient
    if min_slack > 1e-6 and hasattr(problem, 'jacobian_dy_dx_if_inactive'):
        print(f"\nTesting analytical implicit gradient:")
        J = problem.jacobian_dy_dx_if_inactive()
        implicit_analytical = J.T @ grad_y
        print(f"  Implicit analytical: {implicit_analytical}")
        print(f"  ||implicit_analytical||: {torch.norm(implicit_analytical).item():.6f}")
        
        # Compare with numerical
        print(f"\nTesting numerical implicit gradient:")
        implicit_numerical = solver.compute_implicit_gradient_numerical(x, y_star, grad_y)
        print(f"  Implicit numerical: {implicit_numerical}")
        print(f"  ||implicit_numerical||: {torch.norm(implicit_numerical).item():.6f}")
        
        # Check difference
        diff = torch.norm(implicit_analytical - implicit_numerical).item()
        print(f"  Difference: {diff:.6f}")
    
    return gap, direct, implicit


if __name__ == "__main__":
    debug_corrected_gap()
