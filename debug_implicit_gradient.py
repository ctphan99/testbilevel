#!/usr/bin/env python3
"""
DEBUG IMPLICIT GRADIENT - DIAGNOSTIC SCRIPT
==========================================

CRITICAL ISSUE: Implicit component stuck at exactly 0.041634
ROOT CAUSE: Lower-level solver not responding to x perturbations

This script will:
1. Test if lower-level solver responds to x perturbations
2. Check if the problem setup is correct
3. Identify why implicit gradient computation fails
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
import warnings
warnings.filterwarnings('ignore')


def debug_implicit_gradient():
    """
    Debug why implicit gradient computation is failing
    """
    print("="*80)
    print("DEBUGGING IMPLICIT GRADIENT COMPUTATION")
    print("="*80)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.0005, device='cpu', seed=42)
    
    # Test point
    x = torch.randn(5, device='cpu', dtype=torch.float64) * 0.1
    print(f"Test point x: {x}")
    
    # Solve lower level at x
    y_star, ll_info = problem.solve_lower_level(x)
    print(f"Lower-level solution y*: {y_star}")
    print(f"Lower-level info: {ll_info}")
    
    # Test perturbations
    eps = 1e-4
    print(f"\nTesting perturbations with eps = {eps}")
    
    for i in range(x.shape[0]):
        print(f"\nPerturbing x[{i}] by {eps}:")
        
        # Forward perturbation
        x_pert_fwd = x.clone()
        x_pert_fwd[i] += eps
        print(f"  x_pert_fwd: {x_pert_fwd}")
        
        y_pert_fwd, ll_info_fwd = problem.solve_lower_level(x_pert_fwd)
        print(f"  y_pert_fwd: {y_pert_fwd}")
        print(f"  ll_info_fwd: {ll_info_fwd}")
        
        # Backward perturbation
        x_pert_bwd = x.clone()
        x_pert_bwd[i] -= eps
        print(f"  x_pert_bwd: {x_pert_bwd}")
        
        y_pert_bwd, ll_info_bwd = problem.solve_lower_level(x_pert_bwd)
        print(f"  y_pert_bwd: {y_pert_bwd}")
        print(f"  ll_info_bwd: {ll_info_bwd}")
        
        # Check if solutions are different
        y_diff = y_pert_fwd - y_pert_bwd
        y_diff_norm = torch.norm(y_diff).item()
        print(f"  y_diff_norm: {y_diff_norm}")
        
        if y_diff_norm < 1e-10:
            print(f"  ⚠️  WARNING: Lower-level solver not responding to x[{i}] perturbation!")
        else:
            print(f"  ✅ Lower-level solver responding to x[{i}] perturbation")
    
    # Test analytical Jacobian if available
    print(f"\nTesting analytical Jacobian:")
    if hasattr(problem, 'jacobian_dy_dx_if_inactive'):
        try:
            J = problem.jacobian_dy_dx_if_inactive()
            print(f"  Analytical Jacobian J shape: {J.shape}")
            print(f"  J = {J}")
            
            # Test implicit vector
            if hasattr(problem, 'implicit_vector_if_inactive'):
                v = problem.implicit_vector_if_inactive()
                print(f"  Implicit vector v: {v}")
                print(f"  ||v||: {torch.norm(v).item()}")
        except Exception as e:
            print(f"  Error computing analytical Jacobian: {e}")
    
    # Test problem constraints
    print(f"\nTesting problem constraints:")
    h_val = problem.A @ x - problem.B @ y_star - problem.b
    print(f"  Constraint values h(x, y*): {h_val}")
    print(f"  Max violation: {torch.max(torch.relu(h_val)).item()}")
    
    # Test if constraints are active
    slacks = problem.B @ y_star - (problem.A @ x - problem.b)
    print(f"  Constraint slacks: {slacks}")
    print(f"  Min slack: {torch.min(slacks).item()}")
    
    if torch.min(slacks).item() > 1e-6:
        print("  ✅ Constraints are inactive - analytical Jacobian should work")
    else:
        print("  ⚠️  Constraints are active - need numerical Jacobian")
    
    return problem, x, y_star


def test_implicit_gradient_computation():
    """
    Test the actual implicit gradient computation
    """
    print("\n" + "="*80)
    print("TESTING IMPLICIT GRADIENT COMPUTATION")
    print("="*80)
    
    problem, x, y_star = debug_implicit_gradient()
    
    # Compute direct gradient
    x_copy = x.clone().requires_grad_(True)
    y_copy = y_star.clone().requires_grad_(True)
    
    f_val = problem.upper_objective(x_copy, y_copy, add_noise=True)
    grad_x_direct = torch.autograd.grad(f_val, x_copy, create_graph=True, retain_graph=True)[0]
    grad_y = torch.autograd.grad(f_val, y_copy, create_graph=True, retain_graph=True)[0]
    
    print(f"\nDirect gradient components:")
    print(f"  grad_x_direct: {grad_x_direct}")
    print(f"  grad_y: {grad_y}")
    print(f"  ||grad_x_direct||: {torch.norm(grad_x_direct).item()}")
    print(f"  ||grad_y||: {torch.norm(grad_y).item()}")
    
    # Test implicit gradient computation
    print(f"\nTesting implicit gradient computation:")
    eps = 1e-4
    implicit_component = torch.zeros_like(x)
    
    for i in range(x.shape[0]):
        # Forward perturbation
        x_pert_fwd = x.clone()
        x_pert_fwd[i] += eps
        y_pert_fwd, _ = problem.solve_lower_level(x_pert_fwd)
        
        # Backward perturbation
        x_pert_bwd = x.clone()
        x_pert_bwd[i] -= eps
        y_pert_bwd, _ = problem.solve_lower_level(x_pert_bwd)
        
        # Central difference
        dy_dxi = (y_pert_fwd - y_pert_bwd) / (2 * eps)
        
        # Add contribution to implicit gradient
        implicit_component[i] = torch.dot(dy_dxi, grad_y)
        
        print(f"  i={i}: dy_dxi = {dy_dxi}, contribution = {implicit_component[i].item()}")
    
    print(f"\nImplicit component: {implicit_component}")
    print(f"||implicit_component||: {torch.norm(implicit_component).item()}")
    
    # Total gradient
    total_grad = grad_x_direct + implicit_component
    gap_value = float(torch.norm(total_grad))
    direct_norm = float(torch.norm(grad_x_direct))
    implicit_norm = float(torch.norm(implicit_component))
    
    print(f"\nGap computation:")
    print(f"  Direct norm: {direct_norm:.6f}")
    print(f"  Implicit norm: {implicit_norm:.6f}")
    print(f"  Total gap: {gap_value:.6f}")
    
    return gap_value, direct_norm, implicit_norm


if __name__ == "__main__":
    test_implicit_gradient_computation()
"""
DEBUG IMPLICIT GRADIENT - DIAGNOSTIC SCRIPT
==========================================

CRITICAL ISSUE: Implicit component stuck at exactly 0.041634
ROOT CAUSE: Lower-level solver not responding to x perturbations

This script will:
1. Test if lower-level solver responds to x perturbations
2. Check if the problem setup is correct
3. Identify why implicit gradient computation fails
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
import warnings
warnings.filterwarnings('ignore')


def debug_implicit_gradient():
    """
    Debug why implicit gradient computation is failing
    """
    print("="*80)
    print("DEBUGGING IMPLICIT GRADIENT COMPUTATION")
    print("="*80)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.0005, device='cpu', seed=42)
    
    # Test point
    x = torch.randn(5, device='cpu', dtype=torch.float64) * 0.1
    print(f"Test point x: {x}")
    
    # Solve lower level at x
    y_star, ll_info = problem.solve_lower_level(x)
    print(f"Lower-level solution y*: {y_star}")
    print(f"Lower-level info: {ll_info}")
    
    # Test perturbations
    eps = 1e-4
    print(f"\nTesting perturbations with eps = {eps}")
    
    for i in range(x.shape[0]):
        print(f"\nPerturbing x[{i}] by {eps}:")
        
        # Forward perturbation
        x_pert_fwd = x.clone()
        x_pert_fwd[i] += eps
        print(f"  x_pert_fwd: {x_pert_fwd}")
        
        y_pert_fwd, ll_info_fwd = problem.solve_lower_level(x_pert_fwd)
        print(f"  y_pert_fwd: {y_pert_fwd}")
        print(f"  ll_info_fwd: {ll_info_fwd}")
        
        # Backward perturbation
        x_pert_bwd = x.clone()
        x_pert_bwd[i] -= eps
        print(f"  x_pert_bwd: {x_pert_bwd}")
        
        y_pert_bwd, ll_info_bwd = problem.solve_lower_level(x_pert_bwd)
        print(f"  y_pert_bwd: {y_pert_bwd}")
        print(f"  ll_info_bwd: {ll_info_bwd}")
        
        # Check if solutions are different
        y_diff = y_pert_fwd - y_pert_bwd
        y_diff_norm = torch.norm(y_diff).item()
        print(f"  y_diff_norm: {y_diff_norm}")
        
        if y_diff_norm < 1e-10:
            print(f"  ⚠️  WARNING: Lower-level solver not responding to x[{i}] perturbation!")
        else:
            print(f"  ✅ Lower-level solver responding to x[{i}] perturbation")
    
    # Test analytical Jacobian if available
    print(f"\nTesting analytical Jacobian:")
    if hasattr(problem, 'jacobian_dy_dx_if_inactive'):
        try:
            J = problem.jacobian_dy_dx_if_inactive()
            print(f"  Analytical Jacobian J shape: {J.shape}")
            print(f"  J = {J}")
            
            # Test implicit vector
            if hasattr(problem, 'implicit_vector_if_inactive'):
                v = problem.implicit_vector_if_inactive()
                print(f"  Implicit vector v: {v}")
                print(f"  ||v||: {torch.norm(v).item()}")
        except Exception as e:
            print(f"  Error computing analytical Jacobian: {e}")
    
    # Test problem constraints
    print(f"\nTesting problem constraints:")
    h_val = problem.A @ x - problem.B @ y_star - problem.b
    print(f"  Constraint values h(x, y*): {h_val}")
    print(f"  Max violation: {torch.max(torch.relu(h_val)).item()}")
    
    # Test if constraints are active
    slacks = problem.B @ y_star - (problem.A @ x - problem.b)
    print(f"  Constraint slacks: {slacks}")
    print(f"  Min slack: {torch.min(slacks).item()}")
    
    if torch.min(slacks).item() > 1e-6:
        print("  ✅ Constraints are inactive - analytical Jacobian should work")
    else:
        print("  ⚠️  Constraints are active - need numerical Jacobian")
    
    return problem, x, y_star


def test_implicit_gradient_computation():
    """
    Test the actual implicit gradient computation
    """
    print("\n" + "="*80)
    print("TESTING IMPLICIT GRADIENT COMPUTATION")
    print("="*80)
    
    problem, x, y_star = debug_implicit_gradient()
    
    # Compute direct gradient
    x_copy = x.clone().requires_grad_(True)
    y_copy = y_star.clone().requires_grad_(True)
    
    f_val = problem.upper_objective(x_copy, y_copy, add_noise=True)
    grad_x_direct = torch.autograd.grad(f_val, x_copy, create_graph=True, retain_graph=True)[0]
    grad_y = torch.autograd.grad(f_val, y_copy, create_graph=True, retain_graph=True)[0]
    
    print(f"\nDirect gradient components:")
    print(f"  grad_x_direct: {grad_x_direct}")
    print(f"  grad_y: {grad_y}")
    print(f"  ||grad_x_direct||: {torch.norm(grad_x_direct).item()}")
    print(f"  ||grad_y||: {torch.norm(grad_y).item()}")
    
    # Test implicit gradient computation
    print(f"\nTesting implicit gradient computation:")
    eps = 1e-4
    implicit_component = torch.zeros_like(x)
    
    for i in range(x.shape[0]):
        # Forward perturbation
        x_pert_fwd = x.clone()
        x_pert_fwd[i] += eps
        y_pert_fwd, _ = problem.solve_lower_level(x_pert_fwd)
        
        # Backward perturbation
        x_pert_bwd = x.clone()
        x_pert_bwd[i] -= eps
        y_pert_bwd, _ = problem.solve_lower_level(x_pert_bwd)
        
        # Central difference
        dy_dxi = (y_pert_fwd - y_pert_bwd) / (2 * eps)
        
        # Add contribution to implicit gradient
        implicit_component[i] = torch.dot(dy_dxi, grad_y)
        
        print(f"  i={i}: dy_dxi = {dy_dxi}, contribution = {implicit_component[i].item()}")
    
    print(f"\nImplicit component: {implicit_component}")
    print(f"||implicit_component||: {torch.norm(implicit_component).item()}")
    
    # Total gradient
    total_grad = grad_x_direct + implicit_component
    gap_value = float(torch.norm(total_grad))
    direct_norm = float(torch.norm(grad_x_direct))
    implicit_norm = float(torch.norm(implicit_component))
    
    print(f"\nGap computation:")
    print(f"  Direct norm: {direct_norm:.6f}")
    print(f"  Implicit norm: {implicit_norm:.6f}")
    print(f"  Total gap: {gap_value:.6f}")
    
    return gap_value, direct_norm, implicit_norm


if __name__ == "__main__":
    test_implicit_gradient_computation()
