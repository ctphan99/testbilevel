#!/usr/bin/env python3
"""
Test F2CSA constraint activity with the fixed problem
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm import F2CSAAlgorithm

def test_f2csa_constraint_activity():
    """Test F2CSA with active constraints"""
    
    # Create problem with constraint fix
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.001,
        device='cpu',
        seed=42,
        strong_convex=True
    )
    
    # Apply constraint fix
    problem.b = problem.b - 0.5  # Make constraints much more restrictive
    problem.B = problem.B * 5.0  # Scale up B to make By more influential
    problem.Q_lower = problem.Q_lower * 3.0  # Steepen LL objective
    
    print("=== TESTING F2CSA WITH ACTIVE CONSTRAINTS ===")
    print(f"Modified b: {problem.b}")
    print(f"Modified B norm: {torch.norm(problem.B)}")
    print(f"Modified Q_lower eigenvalues: {torch.linalg.eigvals(problem.Q_lower).real}")
    
    # Test constraint activity
    x_test = torch.randn(5, requires_grad=True)
    y_opt, info = problem.solve_lower_level(x_test)
    h_val = problem.A @ x_test - problem.B @ y_opt - problem.b
    lambda_opt = info.get('lambda', torch.zeros(3))
    
    print(f"\nConstraint activity test:")
    print(f"x: {x_test.detach().numpy()}")
    print(f"y_opt: {y_opt.detach().numpy()}")
    print(f"h_val: {h_val.detach().numpy()}")
    print(f"lambda_opt: {lambda_opt.numpy()}")
    print(f"Max violation: {torch.max(torch.relu(h_val)).item()}")
    print(f"Active constraints: {(h_val > -1e-6).sum().item()}")
    
    # Test F2CSA algorithm
    print(f"\n=== TESTING F2CSA ALGORITHM ===")
    
    f2csa = F2CSAAlgorithm(
        problem=problem,
        device='cpu',
        seed=42,
        alpha_override=0.1,
        eta_override=0.001,
        D_override=0.01,
        Ng_override=32
    )
    
    # Test a single iteration
    x_init = torch.randn(5, requires_grad=True)
    
    # Test the penalty mechanism
    alpha = 0.1
    alpha_1 = alpha ** (-2)
    alpha_2 = alpha ** (-4)
    
    # Solve lower level
    y_opt, info = problem.solve_lower_level(x_init)
    lambda_opt = info.get('lambda', torch.zeros(3))
    h_val = problem.A @ x_init - problem.B @ y_opt - problem.b
    
    print(f"F2CSA penalty test:")
    print(f"  x: {x_init.detach().numpy()}")
    print(f"  y_opt: {y_opt.detach().numpy()}")
    print(f"  lambda_opt: {lambda_opt.numpy()}")
    print(f"  h_val: {h_val.detach().numpy()}")
    print(f"  Active constraints: {(h_val > -1e-6).sum().item()}")
    
    # Test smooth activation
    tau_delta = 0.10
    epsilon_lambda = 0.10
    rho_i = f2csa.smooth_activation(h_val, lambda_opt, tau_delta, epsilon_lambda)
    print(f"  rho_i: {rho_i.detach().numpy()}")
    
    # Test penalty terms
    f_val = problem.upper_objective(x_init, y_opt, add_noise=True)
    g_val = problem.lower_objective(x_init, y_opt, add_noise=False)
    g_val_at_y_star = problem.lower_objective(x_init, y_opt, add_noise=False)
    
    term1 = f_val
    term2 = alpha_1 * (g_val + torch.dot(lambda_opt, h_val) - g_val_at_y_star)
    term3 = 0.5 * alpha_2 * torch.sum(rho_i * (h_val**2))
    
    print(f"  Penalty terms:")
    print(f"    term1: {term1.item()}")
    print(f"    term2: {term2.item()}")
    print(f"    term3: {term3.item()}")
    print(f"    total: {term1.item() + term2.item() + term3.item()}")
    
    # Test gradient computation
    try:
        penalty_term = term2 + term3
        grad_penalty = torch.autograd.grad(penalty_term, x_init, create_graph=False)[0]
        print(f"  grad_penalty: {grad_penalty.detach().numpy()}")
        print(f"  ||grad_penalty||: {torch.norm(grad_penalty).item()}")
    except Exception as e:
        print(f"  Gradient computation failed: {e}")
    
    return problem, f2csa

if __name__ == '__main__':
    test_f2csa_constraint_activity()

Test F2CSA constraint activity with the fixed problem
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm import F2CSAAlgorithm

def test_f2csa_constraint_activity():
    """Test F2CSA with active constraints"""
    
    # Create problem with constraint fix
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.001,
        device='cpu',
        seed=42,
        strong_convex=True
    )
    
    # Apply constraint fix
    problem.b = problem.b - 0.5  # Make constraints much more restrictive
    problem.B = problem.B * 5.0  # Scale up B to make By more influential
    problem.Q_lower = problem.Q_lower * 3.0  # Steepen LL objective
    
    print("=== TESTING F2CSA WITH ACTIVE CONSTRAINTS ===")
    print(f"Modified b: {problem.b}")
    print(f"Modified B norm: {torch.norm(problem.B)}")
    print(f"Modified Q_lower eigenvalues: {torch.linalg.eigvals(problem.Q_lower).real}")
    
    # Test constraint activity
    x_test = torch.randn(5, requires_grad=True)
    y_opt, info = problem.solve_lower_level(x_test)
    h_val = problem.A @ x_test - problem.B @ y_opt - problem.b
    lambda_opt = info.get('lambda', torch.zeros(3))
    
    print(f"\nConstraint activity test:")
    print(f"x: {x_test.detach().numpy()}")
    print(f"y_opt: {y_opt.detach().numpy()}")
    print(f"h_val: {h_val.detach().numpy()}")
    print(f"lambda_opt: {lambda_opt.numpy()}")
    print(f"Max violation: {torch.max(torch.relu(h_val)).item()}")
    print(f"Active constraints: {(h_val > -1e-6).sum().item()}")
    
    # Test F2CSA algorithm
    print(f"\n=== TESTING F2CSA ALGORITHM ===")
    
    f2csa = F2CSAAlgorithm(
        problem=problem,
        device='cpu',
        seed=42,
        alpha_override=0.1,
        eta_override=0.001,
        D_override=0.01,
        Ng_override=32
    )
    
    # Test a single iteration
    x_init = torch.randn(5, requires_grad=True)
    
    # Test the penalty mechanism
    alpha = 0.1
    alpha_1 = alpha ** (-2)
    alpha_2 = alpha ** (-4)
    
    # Solve lower level
    y_opt, info = problem.solve_lower_level(x_init)
    lambda_opt = info.get('lambda', torch.zeros(3))
    h_val = problem.A @ x_init - problem.B @ y_opt - problem.b
    
    print(f"F2CSA penalty test:")
    print(f"  x: {x_init.detach().numpy()}")
    print(f"  y_opt: {y_opt.detach().numpy()}")
    print(f"  lambda_opt: {lambda_opt.numpy()}")
    print(f"  h_val: {h_val.detach().numpy()}")
    print(f"  Active constraints: {(h_val > -1e-6).sum().item()}")
    
    # Test smooth activation
    tau_delta = 0.10
    epsilon_lambda = 0.10
    rho_i = f2csa.smooth_activation(h_val, lambda_opt, tau_delta, epsilon_lambda)
    print(f"  rho_i: {rho_i.detach().numpy()}")
    
    # Test penalty terms
    f_val = problem.upper_objective(x_init, y_opt, add_noise=True)
    g_val = problem.lower_objective(x_init, y_opt, add_noise=False)
    g_val_at_y_star = problem.lower_objective(x_init, y_opt, add_noise=False)
    
    term1 = f_val
    term2 = alpha_1 * (g_val + torch.dot(lambda_opt, h_val) - g_val_at_y_star)
    term3 = 0.5 * alpha_2 * torch.sum(rho_i * (h_val**2))
    
    print(f"  Penalty terms:")
    print(f"    term1: {term1.item()}")
    print(f"    term2: {term2.item()}")
    print(f"    term3: {term3.item()}")
    print(f"    total: {term1.item() + term2.item() + term3.item()}")
    
    # Test gradient computation
    try:
        penalty_term = term2 + term3
        grad_penalty = torch.autograd.grad(penalty_term, x_init, create_graph=False)[0]
        print(f"  grad_penalty: {grad_penalty.detach().numpy()}")
        print(f"  ||grad_penalty||: {torch.norm(grad_penalty).item()}")
    except Exception as e:
        print(f"  Gradient computation failed: {e}")
    
    return problem, f2csa

if __name__ == '__main__':
    test_f2csa_constraint_activity()
