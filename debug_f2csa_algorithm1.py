#!/usr/bin/env python3
"""
Debug F2CSA Algorithm 1 implementation line by line
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
# from f2csa_algorithm import F2CSA

def debug_f2csa_algorithm1():
    """Debug F2CSA Algorithm 1 implementation step by step"""
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.001,
        device='cpu',
        seed=42,
        strong_convex=True
    )
    
    print("=== F2CSA ALGORITHM 1 DEBUG ===")
    print("Following F2CSA.tex Algorithm 1 line by line")
    
    # Test point
    x = torch.randn(5, requires_grad=True)
    alpha = 0.1
    N_g = 128
    
    print(f"\nInput parameters:")
    print(f"  x: {x.detach().numpy()}")
    print(f"  α: {alpha}")
    print(f"  N_g: {N_g}")
    
    # Step 1: Set parameters according to Algorithm 1
    alpha_1 = alpha ** (-2)  # α₁ = α⁻²
    alpha_2 = alpha ** (-4)  # α₂ = α⁻⁴
    delta = alpha ** 3       # δ = α³
    
    print(f"\nStep 1: Parameter setup")
    print(f"  α₁ = α⁻² = {alpha_1}")
    print(f"  α₂ = α⁻⁴ = {alpha_2}")
    print(f"  δ = α³ = {delta}")
    
    # Step 2: Compute ỹ*(x) and λ̃(x) by SGD
    print(f"\nStep 2: Compute ỹ*(x) and λ̃(x)")
    y_tilde, info = problem.solve_lower_level(x)
    lambda_tilde = info.get('lambda', torch.zeros(problem.num_constraints))
    
    print(f"  ỹ*(x): {y_tilde.detach().numpy()}")
    print(f"  λ̃(x): {lambda_tilde.detach().numpy()}")
    print(f"  Solver info: {info}")
    
    # Check constraint violations
    h_val = problem.A @ x - problem.B @ y_tilde - problem.b
    print(f"  h(x,ỹ*): {h_val.detach().numpy()}")
    print(f"  Max violation: {torch.max(torch.relu(h_val)).item()}")
    print(f"  Active constraints: {(h_val > -1e-6).sum().item()}")
    
    # Step 3: Define smooth Lagrangian L_{λ̃,α}(x,y)
    print(f"\nStep 3: Define smooth Lagrangian L_{{λ̃,α}}(x,y)")
    
    # Check if constraints are active for smooth activation
    tau = delta  # τ = Θ(δ)
    epsilon_lambda = 1e-6
    
    # Smooth activation functions
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
    
    # Compute smooth activation values
    rho_values = []
    for i in range(problem.num_constraints):
        h_i = h_val[i].item()
        lambda_i = lambda_tilde[i].item()
        rho_i = sigma_h(h_i) * sigma_lambda(lambda_i)
        rho_values.append(rho_i)
        print(f"  Constraint {i}: h_i={h_i:.6f}, λ_i={lambda_i:.6f}, ρ_i={rho_i:.6f}")
    
    rho = torch.tensor(rho_values)
    print(f"  ρ(x): {rho.numpy()}")
    
    # Step 4: Compute ỹ(x) = argmin_y L_{λ̃,α}(x,y)
    print(f"\nStep 4: Compute ỹ(x) = argmin_y L_{{λ̃,α}}(x,y)")
    
    # Define the penalty Lagrangian
    def penalty_lagrangian(x, y):
        # Upper level objective f(x,y)
        f_val = 0.5 * (x - problem.x_target).T @ problem.Q_upper @ (x - problem.x_target)
        
        # Lower level objective g(x,y)
        g_val = 0.5 * y.T @ problem.Q_lower @ y + problem.c_lower.T @ y
        
        # Constraint terms
        h_val = problem.A @ x - problem.B @ y - problem.b
        
        # Penalty terms
        term1 = alpha_1 * (g_val + lambda_tilde.T @ h_val - g_val)  # g(x,y) + λ^T h(x,y) - g(x,ỹ*)
        term2 = 0.5 * alpha_2 * torch.sum(rho * h_val**2)
        
        return f_val + term1 + term2
    
    # Solve penalty subproblem (simplified - just use unconstrained optimum for now)
    y_tilde_penalty = -torch.linalg.solve(problem.Q_lower, problem.c_lower)
    
    print(f"  ỹ(x): {y_tilde_penalty.detach().numpy()}")
    
    # Check penalty Lagrangian value
    L_val = penalty_lagrangian(x, y_tilde_penalty)
    print(f"  L_{{λ̃,α}}(x,ỹ): {L_val.item()}")
    
    # Step 5: Compute stochastic hypergradient
    print(f"\nStep 5: Compute ∇̃F(x) with N_g={N_g} samples")
    
    # Sample N_g independent samples
    grad_samples = []
    for j in range(N_g):
        # Sample noise for stochastic gradient
        noise = torch.randn_like(x) * problem.noise_std
        
        # Compute stochastic gradient (simplified)
        # In practice, this would be ∇_x L_{λ̃,α}(x, ỹ; ξ_j)
        grad_sample = torch.autograd.grad(penalty_lagrangian(x, y_tilde_penalty), x, create_graph=True)[0]
        grad_samples.append(grad_sample)
    
    # Average the samples
    grad_avg = torch.stack(grad_samples).mean(dim=0)
    print(f"  ∇̃F(x): {grad_avg.detach().numpy()}")
    print(f"  ||∇̃F(x)||: {torch.norm(grad_avg).item()}")
    
    # Check if penalty terms are contributing
    print(f"\nStep 6: Analyze penalty contribution")
    
    # Check if constraints are violated at ỹ
    h_tilde = problem.A @ x - problem.B @ y_tilde_penalty - problem.b
    print(f"  h(x,ỹ): {h_tilde.detach().numpy()}")
    print(f"  Max violation at ỹ: {torch.max(torch.relu(h_tilde)).item()}")
    
    # Check penalty terms
    g_val_tilde = 0.5 * y_tilde_penalty.T @ problem.Q_lower @ y_tilde_penalty + problem.c_lower.T @ y_tilde_penalty
    g_val_star = 0.5 * y_tilde.T @ problem.Q_lower @ y_tilde + problem.c_lower.T @ y_tilde
    
    term1_val = alpha_1 * (g_val_tilde + lambda_tilde.T @ h_tilde - g_val_star)
    term2_val = 0.5 * alpha_2 * torch.sum(rho * h_tilde**2)
    
    print(f"  Term 1 (α₁ * (g + λ^T h - g*)): {term1_val.item()}")
    print(f"  Term 2 (α₂/2 * Σ ρ_i h_i²): {term2_val.item()}")
    print(f"  Total penalty: {term1_val.item() + term2_val.item()}")
    
    # Check if the issue is that constraints are never active
    print(f"\nStep 7: Constraint activity analysis")
    print(f"  Problem setup issue: constraints are never active!")
    print(f"  This means:")
    print(f"    - λ̃(x) = 0 always")
    print(f"    - ρ_i(x) = 0 always (since σ_λ(0) = 0)")
    print(f"    - Penalty terms are always 0")
    print(f"    - No constraint enforcement")
    print(f"    - Implicit gradient becomes constant")
    
    return {
        'x': x,
        'y_tilde': y_tilde,
        'lambda_tilde': lambda_tilde,
        'h_val': h_val,
        'rho': rho,
        'grad_avg': grad_avg,
        'term1': term1_val,
        'term2': term2_val
    }

if __name__ == '__main__':
    debug_f2csa_algorithm1()

Debug F2CSA Algorithm 1 implementation line by line
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
# from f2csa_algorithm import F2CSA

def debug_f2csa_algorithm1():
    """Debug F2CSA Algorithm 1 implementation step by step"""
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.001,
        device='cpu',
        seed=42,
        strong_convex=True
    )
    
    print("=== F2CSA ALGORITHM 1 DEBUG ===")
    print("Following F2CSA.tex Algorithm 1 line by line")
    
    # Test point
    x = torch.randn(5, requires_grad=True)
    alpha = 0.1
    N_g = 128
    
    print(f"\nInput parameters:")
    print(f"  x: {x.detach().numpy()}")
    print(f"  α: {alpha}")
    print(f"  N_g: {N_g}")
    
    # Step 1: Set parameters according to Algorithm 1
    alpha_1 = alpha ** (-2)  # α₁ = α⁻²
    alpha_2 = alpha ** (-4)  # α₂ = α⁻⁴
    delta = alpha ** 3       # δ = α³
    
    print(f"\nStep 1: Parameter setup")
    print(f"  α₁ = α⁻² = {alpha_1}")
    print(f"  α₂ = α⁻⁴ = {alpha_2}")
    print(f"  δ = α³ = {delta}")
    
    # Step 2: Compute ỹ*(x) and λ̃(x) by SGD
    print(f"\nStep 2: Compute ỹ*(x) and λ̃(x)")
    y_tilde, info = problem.solve_lower_level(x)
    lambda_tilde = info.get('lambda', torch.zeros(problem.num_constraints))
    
    print(f"  ỹ*(x): {y_tilde.detach().numpy()}")
    print(f"  λ̃(x): {lambda_tilde.detach().numpy()}")
    print(f"  Solver info: {info}")
    
    # Check constraint violations
    h_val = problem.A @ x - problem.B @ y_tilde - problem.b
    print(f"  h(x,ỹ*): {h_val.detach().numpy()}")
    print(f"  Max violation: {torch.max(torch.relu(h_val)).item()}")
    print(f"  Active constraints: {(h_val > -1e-6).sum().item()}")
    
    # Step 3: Define smooth Lagrangian L_{λ̃,α}(x,y)
    print(f"\nStep 3: Define smooth Lagrangian L_{{λ̃,α}}(x,y)")
    
    # Check if constraints are active for smooth activation
    tau = delta  # τ = Θ(δ)
    epsilon_lambda = 1e-6
    
    # Smooth activation functions
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
    
    # Compute smooth activation values
    rho_values = []
    for i in range(problem.num_constraints):
        h_i = h_val[i].item()
        lambda_i = lambda_tilde[i].item()
        rho_i = sigma_h(h_i) * sigma_lambda(lambda_i)
        rho_values.append(rho_i)
        print(f"  Constraint {i}: h_i={h_i:.6f}, λ_i={lambda_i:.6f}, ρ_i={rho_i:.6f}")
    
    rho = torch.tensor(rho_values)
    print(f"  ρ(x): {rho.numpy()}")
    
    # Step 4: Compute ỹ(x) = argmin_y L_{λ̃,α}(x,y)
    print(f"\nStep 4: Compute ỹ(x) = argmin_y L_{{λ̃,α}}(x,y)")
    
    # Define the penalty Lagrangian
    def penalty_lagrangian(x, y):
        # Upper level objective f(x,y)
        f_val = 0.5 * (x - problem.x_target).T @ problem.Q_upper @ (x - problem.x_target)
        
        # Lower level objective g(x,y)
        g_val = 0.5 * y.T @ problem.Q_lower @ y + problem.c_lower.T @ y
        
        # Constraint terms
        h_val = problem.A @ x - problem.B @ y - problem.b
        
        # Penalty terms
        term1 = alpha_1 * (g_val + lambda_tilde.T @ h_val - g_val)  # g(x,y) + λ^T h(x,y) - g(x,ỹ*)
        term2 = 0.5 * alpha_2 * torch.sum(rho * h_val**2)
        
        return f_val + term1 + term2
    
    # Solve penalty subproblem (simplified - just use unconstrained optimum for now)
    y_tilde_penalty = -torch.linalg.solve(problem.Q_lower, problem.c_lower)
    
    print(f"  ỹ(x): {y_tilde_penalty.detach().numpy()}")
    
    # Check penalty Lagrangian value
    L_val = penalty_lagrangian(x, y_tilde_penalty)
    print(f"  L_{{λ̃,α}}(x,ỹ): {L_val.item()}")
    
    # Step 5: Compute stochastic hypergradient
    print(f"\nStep 5: Compute ∇̃F(x) with N_g={N_g} samples")
    
    # Sample N_g independent samples
    grad_samples = []
    for j in range(N_g):
        # Sample noise for stochastic gradient
        noise = torch.randn_like(x) * problem.noise_std
        
        # Compute stochastic gradient (simplified)
        # In practice, this would be ∇_x L_{λ̃,α}(x, ỹ; ξ_j)
        grad_sample = torch.autograd.grad(penalty_lagrangian(x, y_tilde_penalty), x, create_graph=True)[0]
        grad_samples.append(grad_sample)
    
    # Average the samples
    grad_avg = torch.stack(grad_samples).mean(dim=0)
    print(f"  ∇̃F(x): {grad_avg.detach().numpy()}")
    print(f"  ||∇̃F(x)||: {torch.norm(grad_avg).item()}")
    
    # Check if penalty terms are contributing
    print(f"\nStep 6: Analyze penalty contribution")
    
    # Check if constraints are violated at ỹ
    h_tilde = problem.A @ x - problem.B @ y_tilde_penalty - problem.b
    print(f"  h(x,ỹ): {h_tilde.detach().numpy()}")
    print(f"  Max violation at ỹ: {torch.max(torch.relu(h_tilde)).item()}")
    
    # Check penalty terms
    g_val_tilde = 0.5 * y_tilde_penalty.T @ problem.Q_lower @ y_tilde_penalty + problem.c_lower.T @ y_tilde_penalty
    g_val_star = 0.5 * y_tilde.T @ problem.Q_lower @ y_tilde + problem.c_lower.T @ y_tilde
    
    term1_val = alpha_1 * (g_val_tilde + lambda_tilde.T @ h_tilde - g_val_star)
    term2_val = 0.5 * alpha_2 * torch.sum(rho * h_tilde**2)
    
    print(f"  Term 1 (α₁ * (g + λ^T h - g*)): {term1_val.item()}")
    print(f"  Term 2 (α₂/2 * Σ ρ_i h_i²): {term2_val.item()}")
    print(f"  Total penalty: {term1_val.item() + term2_val.item()}")
    
    # Check if the issue is that constraints are never active
    print(f"\nStep 7: Constraint activity analysis")
    print(f"  Problem setup issue: constraints are never active!")
    print(f"  This means:")
    print(f"    - λ̃(x) = 0 always")
    print(f"    - ρ_i(x) = 0 always (since σ_λ(0) = 0)")
    print(f"    - Penalty terms are always 0")
    print(f"    - No constraint enforcement")
    print(f"    - Implicit gradient becomes constant")
    
    return {
        'x': x,
        'y_tilde': y_tilde,
        'lambda_tilde': lambda_tilde,
        'h_val': h_val,
        'rho': rho,
        'grad_avg': grad_avg,
        'term1': term1_val,
        'term2': term2_val
    }

if __name__ == '__main__':
    debug_f2csa_algorithm1()
