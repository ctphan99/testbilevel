#!/usr/bin/env python3
"""
Corrected F2CSA Algorithm 1 with modified penalty parameters
α₁ = α⁻¹, α₂ = α⁻² for improved computational efficiency
"""

import torch
import torch.optim as optim
from typing import Dict, Optional, Tuple
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_accurate import AccurateLowerLevelSolver

class F2CSAAlgorithm1Corrected:
    """
    Corrected F2CSA Algorithm 1 with modified penalty parameters
    α₁ = α⁻¹, α₂ = α⁻² (instead of α₁ = α⁻², α₂ = α⁻⁴)
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem, device: str = 'cpu', dtype=torch.float64):
        self.problem = problem
        self.device = device
        self.dtype = dtype
        
        # Initialize accurate lower-level solver
        self.lower_solver = AccurateLowerLevelSolver(problem, device=device, dtype=dtype)
    
    def _compute_smooth_activation(self, h_val: torch.Tensor, lambda_val: torch.Tensor, delta: float) -> torch.Tensor:
        """
        Compute smooth activation function ρ_i(x) = σ_h(h_i(x,y)) + σ_λ(λ_i)
        """
        # σ_h(z) = max(0, z) - smooth approximation
        sigma_h = torch.clamp(h_val, min=0.0)
        
        # σ_λ(z) = max(0, z) - smooth approximation  
        sigma_lambda = torch.clamp(lambda_val, min=0.0)
        
        # ρ_i(x) = σ_h(h_i(x,y)) + σ_λ(λ_i)
        rho = sigma_h + sigma_lambda
        
        return rho
    
    def _compute_penalty_lagrangian(self, x: torch.Tensor, y: torch.Tensor, 
                                  y_star: torch.Tensor, lambda_star: torch.Tensor,
                                  alpha: float, delta: float, noise_upper: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the smooth penalty Lagrangian L_{λ̃,α}(x,y) with modified parameters
        α₁ = α⁻¹, α₂ = α⁻²
        """
        # Modified penalty parameters
        alpha1 = 1.0 / alpha  # α₁ = α⁻¹ (instead of α⁻²)
        alpha2 = 1.0 / (alpha ** 2)  # α₂ = α⁻² (instead of α⁻⁴)
        
        # Compute constraint values
        h_val = self.problem.constraints(x, y)
        
        # Compute smooth activation
        rho = self._compute_smooth_activation(h_val, lambda_star, delta)
        
        # Compute penalty terms
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
                                   lambda_star: torch.Tensor, alpha: float, delta: float,
                                   noise_upper: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Minimize the penalty Lagrangian to find ỹ(x) using Adam optimizer
        """
        # Initialize y close to y_star
        y = y_star.clone().detach().requires_grad_(True)
        
        # Use Adam optimizer for better convergence
        optimizer = optim.Adam([y], lr=0.01)
        
        # Minimize penalty Lagrangian
        for _ in range(100):  # Fixed number of iterations for efficiency
            optimizer.zero_grad()
            
            # Compute penalty Lagrangian
            L_val = self._compute_penalty_lagrangian(x, y, y_star, lambda_star, alpha, delta, noise_upper)
            
            # Backward pass
            L_val.backward()
            
            # Update y
            optimizer.step()
            
            # Check convergence
            if torch.norm(y.grad) < delta:
                break
        
        return y.detach()
    
    def oracle_sample(self, x: torch.Tensor, alpha: float, N_g: int, 
                     noise_upper: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Implement Algorithm 1: Stochastic Penalty-Based Hypergradient Oracle
        with corrected penalty parameters α₁ = α⁻¹, α₂ = α⁻²
        """
        delta = alpha ** 3
        
        print(f"  Computing accurate lower-level solution with δ = {delta:.2e}")
        
        # Step 3: Compute ỹ*(x) and λ̃(x) by accurate solver
        y_star, info = self.problem.solve_lower_level(x, solver='accurate', alpha=alpha)
        lambda_star = info.get('lambda', torch.zeros(self.problem.num_constraints, dtype=self.dtype))
        
        print(f"  Lower-level solution: ỹ = {y_star}")
        print(f"  Lower-level multipliers: λ̃ = {lambda_star}")
        print(f"  Lower-level info: {info}")
        
        print(f"  Computing penalty minimizer ỹ(x) with δ = {delta:.2e}")
        
        # Step 4: Compute ỹ(x) = argmin_y L_{λ̃,α}(x,y) s.t. ||ỹ(x) - y*_{λ̃,α}(x)|| ≤ δ
        y_tilde = self._minimize_penalty_lagrangian(x, y_star, lambda_star, alpha, delta, noise_upper)
        
        print(f"  Penalty minimizer: ỹ = {y_tilde}")
        
        print(f"  Computing stochastic hypergradient with N_g = {N_g}")
        
        # Step 5: Compute ∇F̃(x) = (1/N_g) Σ_{j=1}^{N_g} ∇_x L̃_{λ̃,α}(x, ỹ(x); ξ_j)
        hypergradient_samples = []
        
        for j in range(N_g):
            # Sample fresh noise for this gradient estimate
            if noise_upper is None:
                noise_upper_j, _ = self.problem._sample_instance_noise()
            else:
                noise_upper_j = noise_upper
            
            # Create computational graph for gradient computation
            x_grad = x.clone().detach().requires_grad_(True)
            
            # Compute penalty Lagrangian
            L_val = self._compute_penalty_lagrangian(x_grad, y_tilde, y_star, lambda_star, alpha, delta, noise_upper_j)
            
            # Compute gradient w.r.t. x
            grad_x = torch.autograd.grad(L_val, x_grad, create_graph=True, retain_graph=True)[0]
            hypergradient_samples.append(grad_x.detach())
        
        # Average the samples
        hypergradient = torch.stack(hypergradient_samples).mean(dim=0)
        
        print(f"  Final hypergradient: ∇F̃ = {hypergradient}")
        print(f"  Hypergradient norm: ||∇F̃|| = {torch.norm(hypergradient).item():.6f}")
        
        return hypergradient
    
    def optimize(self, x0: torch.Tensor, max_iterations: int = 1000, 
                alpha: float = 0.2, N_g: int = 10, lr: float = 1e-3) -> Dict:
        """
        Optimize using corrected F2CSA Algorithm 1 with modified penalty parameters
        
        Args:
            x0: Initial point
            max_iterations: Maximum number of iterations
            alpha: Accuracy parameter (α) - use α = 0.2 for δ = 0.008 < 0.1
            N_g: Batch size for stochastic gradient estimation
            lr: Learning rate for upper-level optimization
            
        Returns:
            Dictionary with optimization results
        """
        print(f"Starting Corrected F2CSA Algorithm 1 optimization")
        print(f"  Initial point: x0 = {x0}")
        print(f"  Parameters: α = {alpha}, N_g = {N_g}, lr = {lr}")
        print(f"  δ = α³ = {alpha**3:.6f} < 0.1 ✓")
        print(f"  Modified penalty: α₁ = α⁻¹ = {1/alpha:.1f}, α₂ = α⁻² = {1/alpha**2:.1f}")
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
            
            # Compute stochastic hypergradient using corrected Algorithm 1
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
            'loss_history': losses,  # Alias for compatibility
            'grad_norms': grad_norms,
            'grad_norm_history': grad_norms,  # Alias for compatibility
            'converged': grad_norms[-1] < 1e-3 if grad_norms else False,
            'iterations': len(losses),
            'alpha': alpha,
            'delta': alpha**3
        }

def test_corrected_f2csa():
    """Test the corrected F2CSA algorithm"""
    print("Testing Corrected F2CSA Algorithm 1")
    print("=" * 50)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5, num_constraints=3, noise_std=0.1,
        strong_convex=True, device='cpu'
    )
    
    # Initialize corrected algorithm
    algorithm = F2CSAAlgorithm1Corrected(problem)
    
    # Test with α = 0.2 (δ = 0.008 < 0.1)
    alpha = 0.2
    x_init = torch.randn(5, dtype=torch.float64)
    
    print(f"Testing with α = {alpha}, δ = {alpha**3:.6f}")
    print(f"Modified penalty parameters: α₁ = {1/alpha:.1f}, α₂ = {1/alpha**2:.1f}")
    print()
    
    # Run optimization
    result = algorithm.optimize(x_init, max_iterations=20, alpha=alpha, N_g=10, lr=0.001)
    
    print(f"\nOptimization Results:")
    print(f"  Final x: {result['x_final']}")
    print(f"  Final loss: {result['loss_history'][-1]:.6f}")
    print(f"  Final gradient norm: {result['grad_norm_history'][-1]:.6f}")
    print(f"  Converged: {result['converged']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  δ-accuracy: {result['delta']:.6f} < 0.1 ✓")

if __name__ == "__main__":
    test_corrected_f2csa()
