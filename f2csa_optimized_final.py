#!/usr/bin/env python3
"""
F2CSA Algorithm 1 - Optimized Final Version
Based on comprehensive debugging analysis
"""

import torch
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
import warnings

warnings.filterwarnings('ignore')

class F2CSAAlgorithm1Optimized(F2CSAAlgorithm1Final):
    """Optimized F2CSA Algorithm 1 with stability fixes (no gradient normalization/clipping)"""
    
    def __init__(self, problem, device='cpu', dtype=torch.float64):
        super().__init__(problem, device, dtype)
    
    def _compute_penalty_lagrangian(self, x: jnp.ndarray, y: jnp.ndarray, 
                                  y_star: jnp.ndarray, lambda_star: jnp.ndarray, 
                                  alpha: float, delta: float, 
                                  noise_upper: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Use base penalty Lagrangian (α₁ = α⁻², α₂ = α⁻⁴)."""
        return super()._compute_penalty_lagrangian(x, y, y_star, lambda_star, alpha, delta, noise_upper)
    
    def _solve_lower_level_cvxpy(self, x: jnp.ndarray, alpha: float) -> tuple:
        """Lower-level solver: use noiseless Q for DCP safety; also compute noisy variant for debug comparison."""
        # Noiseless solution (used for algorithm to avoid DCP issues)
        y_clean, lambda_clean, info_clean = self.problem.solve_ll_with_duals(x, noise_lower=None)
        
        # Debug: attempt noisy solution and compare (do not print noise)
        y_noisy = None
        try:
            _, noise_lower = self.problem._sample_instance_noise()
            noise_lower_jax = jnp.array(noise_lower.detach().cpu().numpy())
            y_noisy, _, _ = self.problem.solve_ll_with_duals(x, noise_lower_jax)
        except Exception:
            y_noisy = None
        
        # Optional: compare solutions; suppress noisy debug prints
        # diff_norm can be computed here if needed for diagnostics
        
        return y_clean, lambda_clean, info_clean
    
    def _minimize_penalty_lagrangian(self, x: jnp.ndarray, y_star: jnp.ndarray, 
                                   lambda_star: jnp.ndarray, alpha: float, delta: float,
                                   init_y: Optional[jnp.ndarray] = None,
                                   trust_region_radius: float = 0.2) -> jnp.ndarray:
        """Penalty Lagrangian minimization with trust-region projection around y* and lighter LBFGS."""
        from jaxopt import LBFGS
        
        # Initialize y
        y_init = init_y if init_y is not None else y_star
        
        # Define penalty objective
        penalty_objective = lambda y: self._compute_penalty_lagrangian(x, y, y_star, lambda_star, alpha, delta)
        
        # Lighter LBFGS settings for stability
        lbfgs_solver = LBFGS(
            fun=penalty_objective,
            maxiter=50,
            tol=1e-6,
            linesearch='zoom',
            maxls=10,
            history_size=10,
            use_gamma=False,
            verbose=False
        )
        
        try:
            result = lbfgs_solver.run(y_init)
            y_out = result.params
        except Exception:
            # Fallback: short gradient descent
            y_out = y_init
            for _ in range(30):
                grad = jax.grad(penalty_objective)(y_out)
                y_out = y_out - 0.001 * grad
                if jnp.linalg.norm(grad) < 1e-6:
                    break
        
        # Trust-region projection around y_star
        diff = y_out - y_star
        norm_diff = jnp.linalg.norm(diff)
        scale = jnp.minimum(1.0, trust_region_radius / (norm_diff + 1e-12))
        y_proj = y_star + diff * scale
        # Box constraints projection as well
        y_proj = jnp.clip(y_proj, -1.0, 1.0)
        return y_proj
    
    def oracle_sample(self, x: jnp.ndarray, alpha: float, N_g: int,
                      prev_y: Optional[jnp.ndarray] = None,
                      warm_ll: bool = False,
                      deterministic_grad: bool = True) -> tuple:
        """Optimized oracle sample with adaptive learning and gradient normalization"""
        delta = alpha ** 3
        
        # Step 3: Compute ỹ*(x) and λ̃(x)
        y_star, lambda_star, info = self._solve_lower_level_cvxpy(x, alpha)
        
        # Step 4: Compute ỹ(x)
        init_y = prev_y if warm_ll and prev_y is not None else y_star
        y_tilde = self._minimize_penalty_lagrangian(x, y_star, lambda_star, alpha, delta, init_y,
                                                    trust_region_radius=0.05)
        
        # Step 5: Compute ∇F̃(x)
        hypergradient_samples = []
        
        for j in range(N_g):
            # Deterministic or stochastic hypergradient
            if deterministic_grad:
                noise_upper_jax = jnp.zeros_like(self.problem.Q_upper)
            else:
                noise_upper, _ = self.problem._sample_instance_noise()
                noise_upper_jax = jnp.array(noise_upper.detach().cpu().numpy())
            
            # Compute penalty Lagrangian with selected noise
            penalty_with_noise = lambda x: self._compute_penalty_lagrangian(x, y_tilde, y_star, lambda_star, alpha, delta, noise_upper_jax)
            grad_x = jax.grad(penalty_with_noise)(x)
            hypergradient_samples.append(grad_x)
        
        hypergradient = jnp.mean(jnp.stack(hypergradient_samples), axis=0)
        
        return hypergradient, y_tilde, lambda_star

def test_f2csa_optimized():
    """Test the optimized F2CSA algorithm"""
    print("F2CSA Algorithm 1 - Optimized Final Version")
    print("=" * 60)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=10, num_constraints=3, device='cpu')
    
    # Test different alpha values
    alpha_values = [0.1]
    N_g = 64
    
    for alpha in alpha_values:
        print(f"\nTesting with alpha = {alpha}")
        print(f"   Penalty parameters: alpha1=alpha^-2={alpha**(-2):.2f}, alpha2=alpha^-4={alpha**(-4):.2f}")
        
        # Initial point
        x0 = jnp.zeros(problem.dim)
        
        # Initialize optimized algorithm
        algorithm1 = F2CSAAlgorithm1Optimized(problem, device='cpu', dtype=torch.float64)
        
        # Track progress
        x = x0
        Delta = jnp.zeros_like(x)
        losses = []
        grad_norms = []
        
        max_iterations = 2000
        convergence_tol = 1e-2
        lr = 5e-5  # tighter step
        D = 0.015  # smaller clipping radius
        N_g = 32   # Prior success: Ng=32

        def clip_D(v: jnp.ndarray, Dval: float) -> jnp.ndarray:
            v_norm = jnp.linalg.norm(v)
            scale = jnp.minimum(1.0, Dval / (v_norm + 1e-12))
            return v * scale
        
        print(f"   Iter | F_Loss   | F_Grad   | Status")
        print(f"   " + "-" * 35)
        
        prev_y = None
        for iteration in range(max_iterations):
            try:
                # Preserve previous y_tilde for drift logging
                prev_y_old = prev_y
                # Compute stochastic hypergradient
                hypergradient, y_tilde, lambda_final = algorithm1.oracle_sample(
                    x, alpha, N_g, prev_y=prev_y_old, warm_ll=True, deterministic_grad=True
                )
                # Update prev_y after using it for drift computation
                prev_y = y_tilde
                
                # Fixed-step update with Algorithm 2-style clipping and guards
                if jnp.any(jnp.isnan(hypergradient)) or jnp.any(jnp.isinf(hypergradient)):
                    print("   Hypergradient became NaN/Inf — stopping this alpha run")
                    break
                grad_norm = float(jnp.linalg.norm(hypergradient))
                Delta_candidate = clip_D(Delta - lr * hypergradient, D)
                if jnp.any(jnp.isnan(Delta_candidate)) or jnp.any(jnp.isinf(Delta_candidate)):
                    print("   Update Delta produced NaN/Inf — resetting Delta to zero and continuing")
                    Delta = jnp.zeros_like(Delta)
                else:
                    Delta = Delta_candidate
                x = x + Delta
                
                # Compute current loss
                y_current, _, _ = algorithm1._solve_lower_level_cvxpy(x, alpha)
                current_loss = problem.upper_objective(x, y_current)
                
                losses.append(float(current_loss))
                grad_norms.append(grad_norm)
                
                # Print progress every 5 iterations
                if (iteration + 1) % 5 == 0 or iteration < 3:
                    status = "Converged" if grad_norm < convergence_tol else "Optimizing"
                    print(f"   {iteration+1:4d} | {current_loss:8.6f} | {grad_norm:8.6f} | {status}")
                
                # Check convergence
                if grad_norm < convergence_tol:
                    print(f"   CONVERGED at iteration {iteration + 1}!")
                    break
                
                # Debug logging: component-wise gradient stats and y_tilde drift
                if (iteration + 1) % 25 == 0:
                    grad_min = float(jnp.min(hypergradient))
                    grad_max = float(jnp.max(hypergradient))
                    grad_mean = float(jnp.mean(hypergradient))
                    drift = float(jnp.linalg.norm(y_tilde - prev_y_old)) if prev_y_old is not None else 0.0
                    dist_to_y_star = float(jnp.linalg.norm(y_tilde - y_current))  # proxy since y_current ~ y*(x)
                    print(f"   Grad stats: min={grad_min:.3e}, max={grad_max:.3e}, mean={grad_mean:.3e}; y_tilde drift={drift:.3e}, dist_to_y*~{dist_to_y_star:.3e}")
                    
            except Exception as e:
                print(f"   ❌ Error at iteration {iteration + 1}: {e}")
                break
        
        else:
            print(f"   Max iterations reached")
        
        # Final summary for this alpha
        if len(losses) > 0:
            print(f"   Final: Loss={losses[-1]:.6f}, Grad={grad_norms[-1]:.6f}")
            print(f"   Convergence: {'YES' if grad_norms[-1] < convergence_tol else 'NO'}")
            
            # Plot results for this alpha
            if len(losses) > 1:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Loss plot
                ax1.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=2)
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Loss')
                ax1.set_title(f'F2CSA Optimized - Loss (α={alpha})')
                ax1.grid(True, alpha=0.3)
                
                # Gradient norm plot
                ax2.plot(range(1, len(grad_norms) + 1), grad_norms, 'r-', linewidth=2)
                ax2.axhline(y=convergence_tol, color='g', linestyle='--', label='Convergence threshold')
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Gradient Norm')
                ax2.set_title(f'F2CSA Optimized - Gradient (α={alpha})')
                ax2.set_yscale('log')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                plt.tight_layout()
                plt.savefig(f'f2csa_optimized_alpha{alpha}.png', dpi=150, bbox_inches='tight')
                print(f"   Plot saved as 'f2csa_optimized_alpha{alpha}.png'")
    
    print(f"\nOPTIMIZATION COMPLETE!")
    print(f"   Tested α values: {alpha_values}")
    print(f"   Check individual plots for convergence behavior")

if __name__ == "__main__":
    test_f2csa_optimized()
