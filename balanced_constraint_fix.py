#!/usr/bin/env python3
"""
Create a balanced constraint fix that makes constraints active without destabilizing optimization
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem

def create_balanced_constraint_problem():
    """Create a problem with balanced constraint activity"""
    
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.001,
        device='cpu',
        seed=42,
        strong_convex=True
    )
    
    print("=== CREATING BALANCED CONSTRAINT PROBLEM ===")
    print(f"Original b: {problem.b}")
    print(f"Original B norm: {torch.norm(problem.B)}")
    print(f"Original Q_lower eigenvalues: {torch.linalg.eigvals(problem.Q_lower).real}")
    
    # More balanced constraint modifications
    # Method 1: Make constraints moderately more restrictive
    problem.b = problem.b - 0.1  # Moderate tightening
    
    # Method 2: Scale up B matrix moderately
    problem.B = problem.B * 2.0  # Moderate scaling
    
    # Method 3: Make Q_lower moderately steeper
    problem.Q_lower = problem.Q_lower * 1.5  # Moderate steepening
    
    print(f"Modified b: {problem.b}")
    print(f"Modified B norm: {torch.norm(problem.B)}")
    print(f"Modified Q_lower eigenvalues: {torch.linalg.eigvals(problem.Q_lower).real}")
    
    # Test constraint activity
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

def test_f2csa_penalty_mechanism(problem):
    """Test F2CSA penalty mechanism with balanced constraints"""
    
    print(f"\n=== TESTING F2CSA PENALTY MECHANISM ===")
    
    # Test parameters
    alpha = 0.1
    alpha_1 = alpha ** (-2)
    alpha_2 = alpha ** (-4)
    
    # Test with multiple x values
    for i in range(3):
        x = torch.randn(5, requires_grad=True)
        
        # Solve lower level
        y_opt, info = problem.solve_lower_level(x)
        lambda_opt = info.get('lambda', torch.zeros(3))
        h_val = problem.A @ x - problem.B @ y_opt - problem.b
        
        print(f"\nTest {i+1}:")
        print(f"  x: {x.detach().numpy()}")
        print(f"  y_opt: {y_opt.detach().numpy()}")
        print(f"  λ_opt: {lambda_opt.numpy()}")
        print(f"  h_val: {h_val.detach().numpy()}")
        print(f"  Active constraints: {(h_val > -1e-6).sum().item()}")
        
        # Test smooth activation
        tau_delta = 0.10
        epsilon_lambda = 0.10
        
        def sigma_h(z):
            if z < -tau_delta:
                return 0.0
            elif z <= 0:
                return (tau_delta + z) / tau_delta
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
        for j in range(problem.num_constraints):
            h_j = h_val[j].item()
            lambda_j = lambda_opt[j].item()
            rho_j = sigma_h(h_j) * sigma_lambda(lambda_j)
            rho_values.append(rho_j)
        
        rho = torch.tensor(rho_values)
        print(f"  ρ: {rho.numpy()}")
        
        # Test penalty terms
        f_val = problem.upper_objective(x, y_opt, add_noise=True)
        g_val = problem.lower_objective(x, y_opt, add_noise=False)
        g_val_at_y_star = problem.lower_objective(x, y_opt, add_noise=False)
        
        term1 = f_val
        term2 = alpha_1 * (g_val + torch.dot(lambda_opt, h_val) - g_val_at_y_star)
        term3 = 0.5 * alpha_2 * torch.sum(rho * (h_val**2))
        
        print(f"  Penalty terms:")
        print(f"    term1: {term1.item()}")
        print(f"    term2: {term2.item()}")
        print(f"    term3: {term3.item()}")
        print(f"    total: {term1.item() + term2.item() + term3.item()}")
        
        # Test gradient
        try:
            penalty_term = term2 + term3
            grad_penalty = torch.autograd.grad(penalty_term, x, create_graph=False)[0]
            print(f"  ||grad_penalty||: {torch.norm(grad_penalty).item()}")
        except Exception as e:
            print(f"  Gradient computation failed: {e}")

if __name__ == '__main__':
    problem = create_balanced_constraint_problem()
    test_f2csa_penalty_mechanism(problem)
"""
Create a balanced constraint fix that makes constraints active without destabilizing optimization
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem

def create_balanced_constraint_problem():
    """Create a problem with balanced constraint activity"""
    
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.001,
        device='cpu',
        seed=42,
        strong_convex=True
    )
    
    print("=== CREATING BALANCED CONSTRAINT PROBLEM ===")
    print(f"Original b: {problem.b}")
    print(f"Original B norm: {torch.norm(problem.B)}")
    print(f"Original Q_lower eigenvalues: {torch.linalg.eigvals(problem.Q_lower).real}")
    
    # More balanced constraint modifications
    # Method 1: Make constraints moderately more restrictive
    problem.b = problem.b - 0.1  # Moderate tightening
    
    # Method 2: Scale up B matrix moderately
    problem.B = problem.B * 2.0  # Moderate scaling
    
    # Method 3: Make Q_lower moderately steeper
    problem.Q_lower = problem.Q_lower * 1.5  # Moderate steepening
    
    print(f"Modified b: {problem.b}")
    print(f"Modified B norm: {torch.norm(problem.B)}")
    print(f"Modified Q_lower eigenvalues: {torch.linalg.eigvals(problem.Q_lower).real}")
    
    # Test constraint activity
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

def test_f2csa_penalty_mechanism(problem):
    """Test F2CSA penalty mechanism with balanced constraints"""
    
    print(f"\n=== TESTING F2CSA PENALTY MECHANISM ===")
    
    # Test parameters
    alpha = 0.1
    alpha_1 = alpha ** (-2)
    alpha_2 = alpha ** (-4)
    
    # Test with multiple x values
    for i in range(3):
        x = torch.randn(5, requires_grad=True)
        
        # Solve lower level
        y_opt, info = problem.solve_lower_level(x)
        lambda_opt = info.get('lambda', torch.zeros(3))
        h_val = problem.A @ x - problem.B @ y_opt - problem.b
        
        print(f"\nTest {i+1}:")
        print(f"  x: {x.detach().numpy()}")
        print(f"  y_opt: {y_opt.detach().numpy()}")
        print(f"  λ_opt: {lambda_opt.numpy()}")
        print(f"  h_val: {h_val.detach().numpy()}")
        print(f"  Active constraints: {(h_val > -1e-6).sum().item()}")
        
        # Test smooth activation
        tau_delta = 0.10
        epsilon_lambda = 0.10
        
        def sigma_h(z):
            if z < -tau_delta:
                return 0.0
            elif z <= 0:
                return (tau_delta + z) / tau_delta
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
        for j in range(problem.num_constraints):
            h_j = h_val[j].item()
            lambda_j = lambda_opt[j].item()
            rho_j = sigma_h(h_j) * sigma_lambda(lambda_j)
            rho_values.append(rho_j)
        
        rho = torch.tensor(rho_values)
        print(f"  ρ: {rho.numpy()}")
        
        # Test penalty terms
        f_val = problem.upper_objective(x, y_opt, add_noise=True)
        g_val = problem.lower_objective(x, y_opt, add_noise=False)
        g_val_at_y_star = problem.lower_objective(x, y_opt, add_noise=False)
        
        term1 = f_val
        term2 = alpha_1 * (g_val + torch.dot(lambda_opt, h_val) - g_val_at_y_star)
        term3 = 0.5 * alpha_2 * torch.sum(rho * (h_val**2))
        
        print(f"  Penalty terms:")
        print(f"    term1: {term1.item()}")
        print(f"    term2: {term2.item()}")
        print(f"    term3: {term3.item()}")
        print(f"    total: {term1.item() + term2.item() + term3.item()}")
        
        # Test gradient
        try:
            penalty_term = term2 + term3
            grad_penalty = torch.autograd.grad(penalty_term, x, create_graph=False)[0]
            print(f"  ||grad_penalty||: {torch.norm(grad_penalty).item()}")
        except Exception as e:
            print(f"  Gradient computation failed: {e}")

if __name__ == '__main__':
    problem = create_balanced_constraint_problem()
    test_f2csa_penalty_mechanism(problem)
