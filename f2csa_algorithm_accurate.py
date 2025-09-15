#!/usr/bin/env python3
"""
F2CSA Algorithm 1 Implementation - Stochastic Penalty-Based Hypergradient Oracle
Following F2CSA.tex Algorithm 1 exactly with accurate lower-level solver
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
from problem import StronglyConvexBilevelProblem
from accurate_lower_level_solver import AccurateLowerLevelSolver

class F2CSAAlgorithm1:
    """
    F2CSA Algorithm 1: Stochastic Penalty-Based Hypergradient Oracle
    Implements F2CSA.tex Algorithm 1 exactly
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem, device: str = 'cpu', dtype=torch.float64):
        self.problem = problem
        self.device = device
        self.dtype = dtype
        
        # Initialize accurate lower-level solver
        self.lower_solver = AccurateLowerLevelSolver(problem, device=device, dtype=dtype)
        
    def smooth_activation_h(self, z: torch.Tensor, tau: float, delta: float) -> torch.Tensor:
        """
        Smooth activation function σ_h(z) as defined in F2CSA.tex
        """
        # σ_h(z) = 0 if z < -τδ, (τδ + z)/(τδ) if -τδ ≤ z < 0, 1 if z ≥ 0
        result = torch.zeros_like(z)
        
        # Case 1: z < -τδ → 0
        mask1 = z < -tau * delta
        result[mask1] = 0.0
        
        # Case 2: -τδ ≤ z < 0 → (τδ + z)/(τδ)
        mask2 = (z >= -tau * delta) & (z < 0)
        result[mask2] = (tau * delta + z[mask2]) / (tau * delta)
        
        # Case 3: z ≥ 0 → 1
        mask3 = z >= 0
        result[mask3] = 1.0
        
        return result
    
    def smooth_activation_lambda(self, z: torch.Tensor, epsilon_lambda: float) -> torch.Tensor:
        """
        Smooth activation function σ_λ(z) as defined in F2CSA.tex
        """
        # σ_λ(z) = 0 if z ≤ 0, z/ε_λ if 0 < z < ε_λ, 1 if z ≥ ε_λ
        result = torch.zeros_like(z)
        
        # Case 1: z ≤ 0 → 0
        mask1 = z <= 0
        result[mask1] = 0.0
        
        # Case 2: 0 < z < ε_λ → z/ε_λ
        mask2 = (z > 0) & (z < epsilon_lambda)
        result[mask2] = z[mask2] / epsilon_lambda
        
        # Case 3: z ≥ ε_λ → 1
        mask3 = z >= epsilon_lambda
        result[mask3] = 1.0
        
        return result
    
    def rho_activation(self, h_val: torch.Tensor, lambda_val: torch.Tensor, 
                      tau: float, delta: float, epsilon_lambda: float) -> torch.Tensor:
        """
        Combined smooth activation function ρ_i(x) = σ_h(h_i) · σ_λ(λ_i)
        """
        sigma_h = self.smooth_activation_h(h_val, tau, delta)
        sigma_lambda = self.smooth_activation_lambda(lambda_val, epsilon_lambda)
        return sigma_h * sigma_lambda
    
    def penalty_lagrangian(self, x: torch.Tensor, y: torch.Tensor, 
                          y_tilde: torch.Tensor, lambda_tilde: torch.Tensor,
                          alpha: float, noise_upper: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute penalty Lagrangian L_{λ̃,α}(x,y) as defined in F2CSA.tex Equation (415-416)
        
        L_{λ̃,α}(x,y) = f(x,y) + α₁(g(x,y) + λ̃^T h(x,y) - g(x,ỹ*)) + (α₂/2)∑ᵢ ρᵢ(x) hᵢ(x,y)²
        where α₁ = α⁻², α₂ = α⁻⁴
        """
        # Ensure tensors are on correct device and dtype
        x = x.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)
        y_tilde = y_tilde.to(device=self.device, dtype=self.dtype)
        lambda_tilde = lambda_tilde.to(device=self.device, dtype=self.dtype)
        
        # Parameters as per Algorithm 1
        alpha_1 = alpha**(-2)  # α₁ = α⁻²
        alpha_2 = alpha**(-4)  # α₂ = α⁻⁴
        delta = alpha**3       # δ = α³
        
        # Smooth activation parameters
        tau = delta  # τ = Θ(δ)
        epsilon_lambda = 1e-6  # Small positive parameter
        
        # Upper-level objective f(x,y)
        f_val = self.problem.upper_objective(x, y, noise_upper=noise_upper)
        
        # Lower-level objective g(x,y)
        g_val = self.problem.lower_objective(x, y)
        
        # Lower-level objective at y_tilde: g(x, ỹ*)
        g_tilde_val = self.problem.lower_objective(x, y_tilde)
        
        # Constraints h(x,y) = Ax + By - b
        h_val = self.problem.constraints(x, y)
        
        # Linear penalty term: α₁(g(x,y) + λ̃^T h(x,y) - g(x,ỹ*))
        linear_penalty = alpha_1 * (g_val + torch.sum(lambda_tilde * h_val) - g_tilde_val)
        
        # Quadratic penalty term: (α₂/2)∑ᵢ ρᵢ(x) hᵢ(x,y)²
        rho_vals = self.rho_activation(h_val, lambda_tilde, tau, delta, epsilon_lambda)
        quadratic_penalty = (alpha_2 / 2.0) * torch.sum(rho_vals * h_val**2)
        
        # Total penalty Lagrangian
        L_val = f_val + linear_penalty + quadratic_penalty
        
        return L_val
    
    def oracle_sample(self, x: torch.Tensor, alpha: float, N_g: int, 
                     noise_upper: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        F2CSA Algorithm 1: Stochastic Penalty-Based Hypergradient Oracle
        
        Input: Point x ∈ ℝⁿ, accuracy parameter α > 0, batch size N_g
        Output: ∇F̃(x) - stochastic hypergradient estimate
        """
        # Ensure x is on correct device and dtype
        x = x.to(device=self.device, dtype=self.dtype)
        
        # Step 1: Set parameters as per Algorithm 1
        alpha_1 = alpha**(-2)  # α₁ = α⁻²
        alpha_2 = alpha**(-4)  # α₂ = α⁻⁴
        delta = alpha**3       # δ = α³
        
        # Step 2: Compute ỹ*(x) and λ̃(x) by accurate solver s.t. ||ỹ*(x) - y*(x)|| ≤ O(δ)
        print(f"  Computing accurate lower-level solution with δ = {delta:.2e}")
        y_tilde, lambda_tilde, info = self.lower_solver.solve_lower_level_accurate(
            x, alpha, max_iter=10000, tol=1e-8
        )
        
        print(f"  Lower-level solution: ỹ = {y_tilde}")
        print(f"  Lower-level multipliers: λ̃ = {lambda_tilde}")
        print(f"  Lower-level info: {info}")
        
        # Step 3: Define smooth Lagrangian L_{λ̃,α}(x,y) and compute ỹ(x) = argmin_y L_{λ̃,α}(x,y)
        print(f"  Computing penalty minimizer ỹ(x) with δ = {delta:.2e}")
        
        # Initialize y for penalty minimization
        y_penalty = torch.randn(self.problem.dim, device=self.device, dtype=self.dtype, requires_grad=True)
        optimizer = optim.Adam([y_penalty], lr=1e-3)
        
        # Minimize penalty Lagrangian
        for i in range(1000):  # Max iterations for penalty minimization
            optimizer.zero_grad()
            
            # Compute penalty Lagrangian
            L_val = self.penalty_lagrangian(x, y_penalty, y_tilde, lambda_tilde, alpha, noise_upper)
            
            L_val.backward()
            optimizer.step()
            
            # Check convergence
            if i > 0 and i % 100 == 0:
                grad_norm = torch.norm(y_penalty.grad).item() if y_penalty.grad is not None else float('nan')
                if grad_norm < delta:
                    print(f"  Penalty minimization converged at iteration {i}")
                    break
        
        y_penalty_final = y_penalty.detach()
        print(f"  Penalty minimizer: ỹ = {y_penalty_final}")
        
        # Step 4: Compute ∇F̃(x) = (1/N_g)∑_{j=1}^{N_g} ∇_x L̃_{λ̃,α}(x, ỹ(x); ξ_j)
        print(f"  Computing stochastic hypergradient with N_g = {N_g}")
        
        hypergradient_samples = []
        
        for j in range(N_g):
            # Sample noise for this iteration
            if noise_upper is None:
                noise_upper_j, noise_lower_j = self.problem._sample_instance_noise()
                noise_j = noise_upper_j
            else:
                noise_j = noise_upper
            
            # Create tensor for gradient computation
            x_grad = x.clone().detach().requires_grad_(True)
            
            # Compute penalty Lagrangian with noise
            L_val = self.penalty_lagrangian(x_grad, y_penalty_final, y_tilde, lambda_tilde, alpha, noise_j)
            
            # Compute gradient w.r.t. x
            grad_x = torch.autograd.grad(L_val, x_grad, create_graph=True, retain_graph=True)[0]
            hypergradient_samples.append(grad_x.detach())
        
        # Average the samples
        hypergradient = torch.stack(hypergradient_samples).mean(dim=0)
        
        print(f"  Final hypergradient: ∇F̃ = {hypergradient}")
        print(f"  Hypergradient norm: ||∇F̃|| = {torch.norm(hypergradient).item():.6f}")
        
        return hypergradient
    
    def optimize(self, x0: torch.Tensor, max_iterations: int = 1000, 
                alpha: float = 0.1, N_g: int = 10, lr: float = 1e-3) -> Dict:
        """
        Optimize using F2CSA Algorithm 1
        
        Args:
            x0: Initial point
            max_iterations: Maximum number of iterations
            alpha: Accuracy parameter (α)
            N_g: Batch size for stochastic gradient estimation
            lr: Learning rate for upper-level optimization
            
        Returns:
            Dictionary with optimization results
        """
        print(f"Starting F2CSA Algorithm 1 optimization")
        print(f"  Initial point: x0 = {x0}")
        print(f"  Parameters: α = {alpha}, N_g = {N_g}, lr = {lr}")
        print(f"  Max iterations: {max_iterations}")
        
        # Initialize
        x = x0.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([x], lr=lr)
        
        # Track progress
        losses = []
        grad_norms = []
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            # Sample noise for this iteration
            noise_upper, noise_lower = self.problem._sample_instance_noise()
            
            # Compute stochastic hypergradient using Algorithm 1
            hypergradient = self.oracle_sample(x, alpha, N_g, noise_upper)
            
            # Update x
            optimizer.zero_grad()
            x.grad = hypergradient
            optimizer.step()
            
            # Compute current loss
            with torch.no_grad():
                current_loss = self.problem.upper_objective(x, 
                    self.problem.solve_lower_level(x, solver='accurate', alpha=alpha)[0], 
                    noise_upper=noise_upper).item()
                losses.append(current_loss)
                grad_norms.append(torch.norm(hypergradient).item())
            
            print(f"  Loss: {current_loss:.6f}")
            print(f"  Gradient norm: {grad_norms[-1]:.6f}")
            
            # Check convergence
            if iteration > 5 and grad_norms[-1] < 1e-3:
                print(f"  Converged at iteration {iteration + 1}")
                break
            
            # Check for early stopping if loss is not improving
            if iteration > 10:
                recent_losses = losses[-5:]
                if len(recent_losses) >= 5 and max(recent_losses) - min(recent_losses) < 1e-6:
                    print(f"  Early stopping at iteration {iteration + 1} (loss plateau)")
                    break
        
        return {
            'x_final': x.detach(),
            'losses': losses,
            'loss_history': losses,  # Add alias for compatibility
            'grad_norms': grad_norms,
            'grad_norm_history': grad_norms,  # Add alias for compatibility
            'converged': grad_norms[-1] < 1e-4 if grad_norms else False,
            'iterations': len(losses)
        }

def test_f2csa_algorithm1():
    """Test F2CSA Algorithm 1 implementation"""
    print("Testing F2CSA Algorithm 1 Implementation")
    print("=" * 50)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.01, strong_convex=True)
    
    # Initialize algorithm
    algorithm = F2CSAAlgorithm1(problem)
    
    # Test point
    x0 = torch.randn(5, device=problem.device, dtype=problem.dtype) * 0.1
    
    print(f"Test point: x0 = {x0}")
    
    # Test single oracle call
    print("\n--- Testing Single Oracle Call ---")
    alpha = 0.1
    N_g = 5
    
    hypergradient = algorithm.oracle_sample(x0, alpha, N_g)
    
    print(f"Hypergradient: {hypergradient}")
    print(f"Hypergradient norm: {torch.norm(hypergradient).item():.6f}")
    
    # Test full optimization
    print("\n--- Testing Full Optimization ---")
    result = algorithm.optimize(x0, max_iterations=10, alpha=alpha, N_g=N_g, lr=1e-3)
    
    print(f"Final result: {result}")

if __name__ == "__main__":
    test_f2csa_algorithm1()
