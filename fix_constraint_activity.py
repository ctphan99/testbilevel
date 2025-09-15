#!/usr/bin/env python3
"""
Fix constraint activity to make F2CSA penalty mechanism work
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem

def create_active_constraint_problem():
    """Create a problem where constraints are actually active"""
    
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.001,
        device='cpu',
        seed=42,
        strong_convex=True
    )
    
    print("=== ORIGINAL PROBLEM ===")
    print(f"b: {problem.b}")
    print(f"Q_lower eigenvalues: {torch.linalg.eigvals(problem.Q_lower).real}")
    
    # Test constraint activity
    x_test = torch.randn(5, requires_grad=True)
    y_opt, info = problem.solve_lower_level(x_test)
    h_val = problem.A @ x_test - problem.B @ y_opt - problem.b
    print(f"Original max violation: {torch.max(torch.relu(h_val)).item()}")
    print(f"Original λ: {info.get('lambda', torch.zeros(3))}")
    
    # FIX: Make constraints much more restrictive
    print(f"\n=== APPLYING CONSTRAINT FIX ===")
    
    # Method 1: Make b much more negative (tighter constraints)
    problem.b = problem.b - 0.5  # Make constraints much more restrictive
    
    # Method 2: Scale up B matrix to make By larger
    problem.B = problem.B * 5.0  # Scale up B to make By more influential
    
    # Method 3: Make Q_lower steeper to push unconstrained optimum away
    problem.Q_lower = problem.Q_lower * 3.0
    
    print(f"Modified b: {problem.b}")
    print(f"Modified Q_lower eigenvalues: {torch.linalg.eigvals(problem.Q_lower).real}")
    
    # Test constraint activity after fix
    y_opt_fixed, info_fixed = problem.solve_lower_level(x_test)
    h_val_fixed = problem.A @ x_test - problem.B @ y_opt_fixed - problem.b
    print(f"Fixed max violation: {torch.max(torch.relu(h_val_fixed)).item()}")
    print(f"Fixed λ: {info_fixed.get('lambda', torch.zeros(3))}")
    print(f"Active constraints: {(h_val_fixed > -1e-6).sum().item()}")
    
    # Test with multiple random points
    print(f"\n=== TESTING CONSTRAINT ACTIVITY ===")
    active_count = 0
    for i in range(10):
        x_rand = torch.randn(5, requires_grad=True)
        y_rand, info_rand = problem.solve_lower_level(x_rand)
        h_rand = problem.A @ x_rand - problem.B @ y_rand - problem.b
        max_viol = torch.max(torch.relu(h_rand)).item()
        lambda_rand = info_rand.get('lambda', torch.zeros(3))
        active_constraints = (h_rand > -1e-6).sum().item()
        
        if max_viol > 1e-6 or active_constraints > 0:
            active_count += 1
            print(f"  Sample {i+1}: max_viol={max_viol:.6f}, active={active_constraints}, λ={lambda_rand.numpy()}")
    
    print(f"Constraints active in {active_count}/10 samples")
    
    return problem

def test_f2csa_with_active_constraints():
    """Test F2CSA with active constraints"""
    
    problem = create_active_constraint_problem()
    
    print(f"\n=== TESTING F2CSA WITH ACTIVE CONSTRAINTS ===")
    
    # Test F2CSA penalty mechanism
    x = torch.randn(5, requires_grad=True)
    alpha = 0.1
    alpha_1 = alpha ** (-2)
    alpha_2 = alpha ** (-4)
    
    # Solve lower level
    y_tilde, info = problem.solve_lower_level(x)
    lambda_tilde = info.get('lambda', torch.zeros(problem.num_constraints))
    
    print(f"x: {x.detach().numpy()}")
    print(f"ỹ*(x): {y_tilde.detach().numpy()}")
    print(f"λ̃(x): {lambda_tilde.numpy()}")
    
    # Check constraint violations
    h_val = problem.A @ x - problem.B @ y_tilde - problem.b
    print(f"h(x,ỹ*): {h_val.detach().numpy()}")
    print(f"Max violation: {torch.max(torch.relu(h_val)).item()}")
    print(f"Active constraints: {(h_val > -1e-6).sum().item()}")
    
    # Check smooth activation
    delta = alpha ** 3
    tau = delta
    epsilon_lambda = 1e-6
    
    def sigma_h(z):
        if z < -tau * delta:
            return 0.0
        elif z <= 0:
            return (tau * delta + z) / (tau * delta)
        else:
            return 1.0
    
    def sigma_lambda(z):
        if z <= 0:
            return 0.0
        elif z < epsilon_lambda:
            return z / epsilon_lambda
        else:
            return 1.0
    
    rho_values = []
    for i in range(problem.num_constraints):
        h_i = h_val[i].item()
        lambda_i = lambda_tilde[i].item()
        rho_i = sigma_h(h_i) * sigma_lambda(lambda_i)
        rho_values.append(rho_i)
        print(f"  Constraint {i}: h_i={h_i:.6f}, λ_i={lambda_i:.6f}, ρ_i={rho_i:.6f}")
    
    rho = torch.tensor(rho_values)
    print(f"ρ(x): {rho.numpy()}")
    
    # Check penalty terms
    g_val = 0.5 * y_tilde.T @ problem.Q_lower @ y_tilde + problem.c_lower.T @ y_tilde
    term1 = alpha_1 * (g_val + lambda_tilde.T @ h_val - g_val)
    term2 = 0.5 * alpha_2 * torch.sum(rho * h_val**2)
    
    print(f"Penalty terms:")
    print(f"  Term 1: {term1.item()}")
    print(f"  Term 2: {term2.item()}")
    print(f"  Total: {term1.item() + term2.item()}")
    
    return problem

if __name__ == '__main__':
    test_f2csa_with_active_constraints()
"""
Fix constraint activity to make F2CSA penalty mechanism work
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem

def create_active_constraint_problem():
    """Create a problem where constraints are actually active"""
    
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.001,
        device='cpu',
        seed=42,
        strong_convex=True
    )
    
    print("=== ORIGINAL PROBLEM ===")
    print(f"b: {problem.b}")
    print(f"Q_lower eigenvalues: {torch.linalg.eigvals(problem.Q_lower).real}")
    
    # Test constraint activity
    x_test = torch.randn(5, requires_grad=True)
    y_opt, info = problem.solve_lower_level(x_test)
    h_val = problem.A @ x_test - problem.B @ y_opt - problem.b
    print(f"Original max violation: {torch.max(torch.relu(h_val)).item()}")
    print(f"Original λ: {info.get('lambda', torch.zeros(3))}")
    
    # FIX: Make constraints much more restrictive
    print(f"\n=== APPLYING CONSTRAINT FIX ===")
    
    # Method 1: Make b much more negative (tighter constraints)
    problem.b = problem.b - 0.5  # Make constraints much more restrictive
    
    # Method 2: Scale up B matrix to make By larger
    problem.B = problem.B * 5.0  # Scale up B to make By more influential
    
    # Method 3: Make Q_lower steeper to push unconstrained optimum away
    problem.Q_lower = problem.Q_lower * 3.0
    
    print(f"Modified b: {problem.b}")
    print(f"Modified Q_lower eigenvalues: {torch.linalg.eigvals(problem.Q_lower).real}")
    
    # Test constraint activity after fix
    y_opt_fixed, info_fixed = problem.solve_lower_level(x_test)
    h_val_fixed = problem.A @ x_test - problem.B @ y_opt_fixed - problem.b
    print(f"Fixed max violation: {torch.max(torch.relu(h_val_fixed)).item()}")
    print(f"Fixed λ: {info_fixed.get('lambda', torch.zeros(3))}")
    print(f"Active constraints: {(h_val_fixed > -1e-6).sum().item()}")
    
    # Test with multiple random points
    print(f"\n=== TESTING CONSTRAINT ACTIVITY ===")
    active_count = 0
    for i in range(10):
        x_rand = torch.randn(5, requires_grad=True)
        y_rand, info_rand = problem.solve_lower_level(x_rand)
        h_rand = problem.A @ x_rand - problem.B @ y_rand - problem.b
        max_viol = torch.max(torch.relu(h_rand)).item()
        lambda_rand = info_rand.get('lambda', torch.zeros(3))
        active_constraints = (h_rand > -1e-6).sum().item()
        
        if max_viol > 1e-6 or active_constraints > 0:
            active_count += 1
            print(f"  Sample {i+1}: max_viol={max_viol:.6f}, active={active_constraints}, λ={lambda_rand.numpy()}")
    
    print(f"Constraints active in {active_count}/10 samples")
    
    return problem

def test_f2csa_with_active_constraints():
    """Test F2CSA with active constraints"""
    
    problem = create_active_constraint_problem()
    
    print(f"\n=== TESTING F2CSA WITH ACTIVE CONSTRAINTS ===")
    
    # Test F2CSA penalty mechanism
    x = torch.randn(5, requires_grad=True)
    alpha = 0.1
    alpha_1 = alpha ** (-2)
    alpha_2 = alpha ** (-4)
    
    # Solve lower level
    y_tilde, info = problem.solve_lower_level(x)
    lambda_tilde = info.get('lambda', torch.zeros(problem.num_constraints))
    
    print(f"x: {x.detach().numpy()}")
    print(f"ỹ*(x): {y_tilde.detach().numpy()}")
    print(f"λ̃(x): {lambda_tilde.numpy()}")
    
    # Check constraint violations
    h_val = problem.A @ x - problem.B @ y_tilde - problem.b
    print(f"h(x,ỹ*): {h_val.detach().numpy()}")
    print(f"Max violation: {torch.max(torch.relu(h_val)).item()}")
    print(f"Active constraints: {(h_val > -1e-6).sum().item()}")
    
    # Check smooth activation
    delta = alpha ** 3
    tau = delta
    epsilon_lambda = 1e-6
    
    def sigma_h(z):
        if z < -tau * delta:
            return 0.0
        elif z <= 0:
            return (tau * delta + z) / (tau * delta)
        else:
            return 1.0
    
    def sigma_lambda(z):
        if z <= 0:
            return 0.0
        elif z < epsilon_lambda:
            return z / epsilon_lambda
        else:
            return 1.0
    
    rho_values = []
    for i in range(problem.num_constraints):
        h_i = h_val[i].item()
        lambda_i = lambda_tilde[i].item()
        rho_i = sigma_h(h_i) * sigma_lambda(lambda_i)
        rho_values.append(rho_i)
        print(f"  Constraint {i}: h_i={h_i:.6f}, λ_i={lambda_i:.6f}, ρ_i={rho_i:.6f}")
    
    rho = torch.tensor(rho_values)
    print(f"ρ(x): {rho.numpy()}")
    
    # Check penalty terms
    g_val = 0.5 * y_tilde.T @ problem.Q_lower @ y_tilde + problem.c_lower.T @ y_tilde
    term1 = alpha_1 * (g_val + lambda_tilde.T @ h_val - g_val)
    term2 = 0.5 * alpha_2 * torch.sum(rho * h_val**2)
    
    print(f"Penalty terms:")
    print(f"  Term 1: {term1.item()}")
    print(f"  Term 2: {term2.item()}")
    print(f"  Total: {term1.item() + term2.item()}")
    
    return problem

if __name__ == '__main__':
    test_f2csa_with_active_constraints()
