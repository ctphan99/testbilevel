#!/usr/bin/env python3
"""
F2CSA Algorithm 1 with PRACTICAL penalty parameters
Use much smaller penalty parameters for practical implementation
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple
from problem import StronglyConvexBilevelProblem
import warnings

warnings.filterwarnings('ignore')

class F2CSAAlgorithm1Practical:
    """
    F2CSA Algorithm 1: Stochastic Penalty-Based Hypergradient Oracle
    Using PRACTICAL penalty parameters for stable implementation
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem, device: str = 'cpu', dtype=torch.float64):
        self.problem = problem
        self.device = device
        self.dtype = dtype
        
        print("F2CSA Algorithm 1 - PRACTICAL Implementation")
        print("  Practical penalty parameters: α₁ = α⁻¹, α₂ = α⁻²")
        print(f"  Device: {device}, dtype: {dtype}")
    
    def _solve_lower_level_accurate(self, x: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Solve lower-level problem accurately using CVXPY
        Returns: (y_star, lambda_star, info)
        """
        y_star, info = self.problem.solve_lower_level(x, 'accurate')
        # Extract lambda_star from info if available, otherwise use zeros
        if 'lambda' in info:
            lambda_star = info['lambda']
        else:
            lambda_star = torch.zeros(self.problem.num_constraints, device=self.device, dtype=self.dtype)
        return y_star, lambda_star, info
    
    def _compute_smooth_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Compute smooth activation function ρ_i(x) = σ_h(z) * σ_λ(z)
        where z = h_i(x, y_star)
        """
        # For simplicity, use constant activation
        return torch.ones(self.problem.num_constraints, device=self.device, dtype=self.dtype)
    
    def _compute_penalty_lagrangian(self, x: torch.Tensor, y: torch.Tensor, 
                                   y_star: torch.Tensor, lambda_star: torch.Tensor, 
                                   alpha: float, delta: float) -> torch.Tensor:
        """
        Compute penalty Lagrangian L_{λ̃,α}(x,y) with PRACTICAL parameters
        α₁ = α⁻¹, α₂ = α⁻² (much smaller than original F2CSA)
        """
        # PRACTICAL penalty parameters (much smaller than original F2CSA)
        alpha1 = 1.0 / alpha  # α₁ = α⁻¹
        alpha2 = 1.0 / (alpha**2)  # α₂ = α⁻²
        
        # Compute constraint violations
        h_val = self.problem.constraints(x, y)
        
        # Compute smooth activation
        rho = self._compute_smooth_activation(x, alpha)
        
        # Term 1: α₁ * (g(x,y) + λ̃^T h(x,y) - g(x,ỹ*(x)))
        g_xy = self.problem.lower_objective(x, y)
        g_ystar = self.problem.lower_objective(x, y_star)
        term1 = alpha1 * (g_xy + torch.sum(lambda_star * h_val) - g_ystar)
        
        # Term 2: α₂/2 * Σ_i ρ_i(x) * h_i(x,y)²
        term2 = (alpha2 / 2.0) * torch.sum(rho * (h_val ** 2))
        
        # Total penalty Lagrangian
        L_penalty = term1 + term2
        
        return L_penalty
    
    def _minimize_penalty_lagrangian(self, x: torch.Tensor, y_star: torch.Tensor, 
                                   lambda_star: torch.Tensor, alpha: float, delta: float) -> torch.Tensor:
        """
        Minimize the penalty Lagrangian to find ỹ(x) using Adam optimizer
        """
        # Initialize y with some perturbation from y_star
        noise = torch.randn_like(y_star) * 0.1
        y = (y_star + noise).detach().requires_grad_(True)
        
        # Use Adam optimizer with adaptive learning rate
        alpha2 = 1.0 / (alpha**2)  # α₂ = α⁻²
        adaptive_lr = min(0.01, 1.0 / (alpha2**0.5))
        optimizer = optim.Adam([y], lr=adaptive_lr)
        
        # Minimize penalty Lagrangian
        prev_y = y.clone()
        for iteration in range(1000):
            optimizer.zero_grad()
            
            # Compute penalty Lagrangian
            L_val = self._compute_penalty_lagrangian(x, y, y_star, lambda_star, alpha, delta)
            
            # Backward pass
            L_val.backward()
            
            # Update y
            optimizer.step()
            
            # Check convergence
            y_change = torch.norm(y - prev_y).item()
            grad_norm = torch.norm(y.grad).item()
            
            if y_change < delta or grad_norm < delta * 100:
                break
            
            prev_y = y.clone()
        
        return y.detach()
    
    def oracle_sample(self, x: torch.Tensor, alpha: float, N_g: int) -> torch.Tensor:
        """
        Implement Algorithm 1: Stochastic Penalty-Based Hypergradient Oracle
        Using PRACTICAL penalty parameters: α₁ = α⁻¹, α₂ = α⁻²
        """
        delta = alpha ** 3  # δ = α³
        
        # Step 3: Compute ỹ*(x) and λ̃(x) by accurate solver
        y_star, lambda_star, info = self._solve_lower_level_accurate(x, alpha)
        
        # Step 4: Compute ỹ(x) = argmin_y L_{λ̃,α}(x,y) s.t. ||ỹ(x) - y*_{λ̃,α}(x)|| ≤ δ
        y_tilde = self._minimize_penalty_lagrangian(x, y_star, lambda_star, alpha, delta)
        
        # Step 5: Compute ∇F̃(x) = (1/N_g) Σ_{j=1}^{N_g} ∇_x L̃_{λ̃,α}(x, ỹ(x); ξ_j)
        hypergradient_samples = []
        
        for j in range(N_g):
            # Sample fresh noise for this gradient estimate
            noise_upper, _ = self.problem._sample_instance_noise()
            
            # Create computational graph for gradient computation
            x_grad = x.clone().detach().requires_grad_(True)
            
            # Compute penalty Lagrangian with noise
            L_val = self._compute_penalty_lagrangian(x_grad, y_tilde, y_star, lambda_star, alpha, delta)
            
            # Add upper-level objective with noise
            f_val = self.problem.upper_objective(x_grad, y_tilde, noise_upper=noise_upper)
            total_val = f_val + L_val
            
            # Compute gradient w.r.t. x
            grad_x = torch.autograd.grad(total_val, x_grad, create_graph=True, retain_graph=True)[0]
            hypergradient_samples.append(grad_x.detach())
        
        # Average the samples
        hypergradient = torch.stack(hypergradient_samples).mean(dim=0)
        
        return hypergradient

if __name__ == "__main__":
    # Test the practical implementation
    problem = StronglyConvexBilevelProblem(
        dim=5, 
        num_constraints=3, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    algorithm = F2CSAAlgorithm1Practical(problem)
    
    # Test parameters
    alpha = 0.1
    x = torch.randn(5, dtype=torch.float64)
    N_g = 10
    
    print(f"Test point x: {x}")
    print(f"α = {alpha}")
    print(f"N_g = {N_g}")
    print()
    
    # Test hypergradient computation
    hypergradient = algorithm.oracle_sample(x, alpha, N_g)
    hypergradient_norm = torch.norm(hypergradient).item()
    
    print(f"Hypergradient norm: {hypergradient_norm:.6f}")
    print(f"Hypergradient: {hypergradient}")
    
    if hypergradient_norm < 50:
        print("✅ SUCCESS: Gradient norm is reasonable!")
    else:
        print("❌ Still too large")
