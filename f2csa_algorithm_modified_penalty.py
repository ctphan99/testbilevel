#!/usr/bin/env python3
"""
Modified F2CSA Algorithm 1 with reasonable penalty parameters
Preserves core Algorithm 1 structure but uses different penalty scaling
"""

import torch
import torch.optim as optim
from typing import Dict, Optional, Tuple
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_accurate import AccurateLowerLevelSolver

class F2CSAAlgorithm1ModifiedPenalty:
    """
    Modified F2CSA Algorithm 1 with reasonable penalty parameters
    Core Algorithm 1 structure preserved, but penalty scaling adjusted
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
        return torch.clamp((tau * delta + z) / (tau * delta), 0, 1)
    
    def smooth_activation_lambda(self, z: torch.Tensor, epsilon_lambda: float) -> torch.Tensor:
        """
        Smooth activation function σ_λ(z) as defined in F2CSA.tex
        """
        return torch.clamp(z / epsilon_lambda, 0, 1)
    
    def rho_activation(self, h_val: torch.Tensor, lambda_val: torch.Tensor, 
                      tau: float, delta: float, epsilon_lambda: float) -> torch.Tensor:
        """
        Compute ρ_i(x) = σ_h(h_i(x, ỹ*(x))) · σ_λ(λ̃_i(x))
        """
        sigma_h = self.smooth_activation_h(h_val, tau, delta)
        sigma_lambda = self.smooth_activation_lambda(lambda_val, epsilon_lambda)
        return sigma_h * sigma_lambda
    
    def penalty_lagrangian(self, x: torch.Tensor, y: torch.Tensor, 
                          y_tilde: torch.Tensor, lambda_tilde: torch.Tensor,
                          alpha: float, penalty_scale: str = 'moderate', 
                          noise_upper: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute penalty Lagrangian L_{λ̃,α}(x,y) with modified penalty scaling
        
        Args:
            penalty_scale: 'moderate', 'conservative', or 'aggressive'
                - 'moderate': α₁ = α⁻¹, α₂ = α⁻² (reduced from original)
                - 'conservative': α₁ = 1, α₂ = α⁻¹ (very reduced)
                - 'aggressive': α₁ = α⁻¹.⁵, α₂ = α⁻³ (slightly reduced)
        """
        # Compute objective values
        f_val = self.problem.upper_objective(x, y, noise_upper=noise_upper)
        g_val = self.problem.lower_objective(x, y)
        g_tilde_val = self.problem.lower_objective(x, y_tilde)
        
        # Compute constraint values
        h_val = self.problem.constraints(x, y)
        
        # Set penalty parameters based on scaling strategy
        if penalty_scale == 'moderate':
            alpha_1 = 1 / alpha  # α₁ = α⁻¹ instead of α⁻²
            alpha_2 = 1 / (alpha ** 2)  # α₂ = α⁻² instead of α⁻⁴
        elif penalty_scale == 'conservative':
            alpha_1 = 1.0  # α₁ = 1 (constant)
            alpha_2 = 1 / alpha  # α₂ = α⁻¹
        elif penalty_scale == 'aggressive':
            alpha_1 = 1 / (alpha ** 1.5)  # α₁ = α⁻¹.⁵
            alpha_2 = 1 / (alpha ** 3)  # α₂ = α⁻³
        else:
            raise ValueError(f"Unknown penalty_scale: {penalty_scale}")
        
        delta = alpha ** 3
        tau = delta
        epsilon_lambda = delta
        
        # Compute smooth activation
        rho_vals = self.rho_activation(h_val, lambda_tilde, tau, delta, epsilon_lambda)
        
        # Linear penalty term: α₁(g(x,y) + λ̃^T h(x,y) - g(x,ỹ*))
        linear_penalty = alpha_1 * (g_val + torch.sum(lambda_tilde * h_val) - g_tilde_val)
        
        # Quadratic penalty term: (α₂/2)∑ᵢ ρᵢ(x) hᵢ(x,y)²
        quadratic_penalty = (alpha_2 / 2.0) * torch.sum(rho_vals * h_val**2)
        
        # Total penalty Lagrangian
        L_val = f_val + linear_penalty + quadratic_penalty
        
        return L_val
    
    def oracle_sample(self, x: torch.Tensor, alpha: float, N_g: int, 
                     penalty_scale: str = 'moderate',
                     noise_upper: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Modified F2CSA Algorithm 1: Stochastic Penalty-Based Hypergradient Oracle
        with reasonable penalty parameters
        """
        # Ensure x is on correct device and dtype
        x = x.to(device=self.device, dtype=self.dtype)
        
        # Step 1: Set parameters as per Algorithm 1
        delta = alpha ** 3  # δ = α³ (preserved from original)
        
        # Step 2: Compute ỹ*(x) and λ̃(x) by accurate solver
        print(f"  Computing accurate lower-level solution with δ = {delta:.2e}")
        y_tilde, lambda_tilde, info = self.lower_solver.solve_lower_level_accurate(
            x, alpha, max_iter=10000, tol=1e-8
        )
        
        print(f"  Lower-level solution: ỹ = {y_tilde}")
        print(f"  Lower-level multipliers: λ̃ = {lambda_tilde}")
        print(f"  Lower-level info: {info}")
        
        # Step 3: Define smooth Lagrangian and compute ỹ(x) = argmin_y L_{λ̃,α}(x,y)
        print(f"  Computing penalty minimizer ỹ(x) with δ = {delta:.2e}")
        
        # Initialize y for penalty minimization
        y_penalty = torch.randn(self.problem.dim, device=self.device, dtype=self.dtype, requires_grad=True)
        optimizer = optim.Adam([y_penalty], lr=1e-3)
        
        # Minimize penalty Lagrangian
        for i in range(1000):  # Max iterations for penalty minimization
            optimizer.zero_grad()
            
            # Compute penalty Lagrangian with modified scaling
            L_val = self.penalty_lagrangian(x, y_penalty, y_tilde, lambda_tilde, alpha, penalty_scale, noise_upper)
            
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
        
        # Step 4: Compute stochastic hypergradient
        print(f"  Computing stochastic hypergradient with N_g = {N_g}")
        
        gradients = []
        for j in range(N_g):
            # Sample noise for this gradient computation
            noise_j = self.problem._sample_instance_noise()
            noise_upper_j = noise_j[0] if isinstance(noise_j, tuple) else noise_j
            
            # Create a new variable for gradient computation
            x_grad = x.clone().detach().requires_grad_(True)
            
            # Compute penalty Lagrangian with noise
            L_val = self.penalty_lagrangian(x_grad, y_penalty_final, y_tilde, lambda_tilde, alpha, penalty_scale, noise_upper_j)
            
            # Compute gradient w.r.t. x
            grad_x = torch.autograd.grad(L_val, x_grad, create_graph=False)[0]
            gradients.append(grad_x)
        
        # Average gradients
        final_gradient = torch.stack(gradients).mean(dim=0)
        print(f"  Final hypergradient: ∇F̃ = {final_gradient}")
        print(f"  Hypergradient norm: ||∇F̃|| = {torch.norm(final_gradient):.6f}")
        
        return final_gradient
    
    def optimize(self, x0: torch.Tensor, max_iterations: int = 1000, 
                alpha: float = 0.1, N_g: int = 10, lr: float = 1e-3,
                penalty_scale: str = 'moderate') -> Dict:
        """
        Optimize using modified F2CSA Algorithm 1
        """
        print(f"Starting Modified F2CSA Algorithm 1 optimization")
        print(f"  Initial point: x0 = {x0}")
        print(f"  Parameters: α = {alpha}, N_g = {N_g}, lr = {lr}")
        print(f"  Penalty scale: {penalty_scale}")
        print(f"  Max iterations: {max_iterations}")
        print()
        
        x = x0.clone()
        losses = []
        gradient_norms = []
        
        for iteration in range(max_iterations):
            print(f"--- Iteration {iteration+1}/{max_iterations} ---")
            
            # Sample noise for this iteration
            noise_upper, noise_lower = self.problem._sample_instance_noise()
            
            # Compute stochastic hypergradient using modified Algorithm 1
            hypergradient = self.oracle_sample(x, alpha, N_g, penalty_scale, noise_upper)
            
            # Update x
            with torch.no_grad():
                x = x - lr * hypergradient
            
            # Compute current loss
            with torch.no_grad():
                current_loss = self.problem.upper_objective(x, 
                    self.problem.solve_lower_level(x, solver='accurate', alpha=alpha)[0], 
                    noise_upper=noise_upper).item()
                losses.append(current_loss)
                gradient_norms.append(torch.norm(hypergradient).item())
                
                print(f"  Loss: {current_loss:.6f}")
                print(f"  Gradient norm: {gradient_norms[-1]:.6f}")
                print()
        
        final_loss = losses[-1]
        final_gradient_norm = gradient_norms[-1]
        loss_reduction = losses[0] - final_loss
        relative_improvement = loss_reduction / abs(losses[0]) * 100
        
        print(f"Final loss: {final_loss:.6f}")
        print(f"Final gradient norm: {final_gradient_norm:.6f}")
        print(f"Loss reduction: {loss_reduction:.6f}")
        print(f"Relative improvement: {relative_improvement:.2f}%")
        
        # Performance assessment
        if relative_improvement > 5:
            print("✅ Excellent performance")
        elif relative_improvement > 2:
            print("✅ Good performance")
        elif relative_improvement > 0.5:
            print("⚠️ Moderate performance")
        else:
            print("❌ Poor performance")
        
        return {
            'final_x': x,
            'final_loss': final_loss,
            'final_gradient_norm': final_gradient_norm,
            'losses': losses,
            'gradient_norms': gradient_norms,
            'loss_reduction': loss_reduction,
            'relative_improvement': relative_improvement
        }


if __name__ == "__main__":
    # Test the modified algorithm
    problem = StronglyConvexBilevelProblem(
        dim=5, num_constraints=3, noise_std=0.1,
        strong_convex=True, device='cpu'
    )
    
    algorithm = F2CSAAlgorithm1ModifiedPenalty(problem)
    x0 = torch.randn(5, dtype=torch.float64)
    
    # Test with different penalty scales
    penalty_scales = ['moderate', 'conservative', 'aggressive']
    alpha_values = [0.3, 0.2, 0.1, 0.05]
    
    for penalty_scale in penalty_scales:
        print(f"\n{'='*60}")
        print(f"Testing penalty scale: {penalty_scale}")
        print(f"{'='*60}")
        
        for alpha in alpha_values:
            delta = alpha ** 3
            if delta >= 0.1:
                print(f"\nSkipping α = {alpha} (δ = {delta:.6f} >= 0.1)")
                continue
                
            print(f"\n--- Testing α = {alpha} (δ = {delta:.6f}) ---")
            
            try:
                result = algorithm.optimize(
                    x0, max_iterations=10, alpha=alpha, N_g=5, lr=1e-3,
                    penalty_scale=penalty_scale
                )
                
                print(f"Result: Loss reduction = {result['loss_reduction']:.6f}, "
                      f"Relative improvement = {result['relative_improvement']:.2f}%")
                
            except Exception as e:
                print(f"Error with α = {alpha}: {e}")
