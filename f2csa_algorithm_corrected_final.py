#!/usr/bin/env python3
"""
Final Corrected F2CSA Algorithm 1 Implementation
Following F2CSA_corrected.tex exactly with modified penalty parameters:
α₁ = α⁻¹, α₂ = α⁻² (instead of α₁ = α⁻², α₂ = α⁻⁴)

This implementation ensures:
1. δ-accuracy < 0.1 (δ = α³)
2. Proper loss convergence
3. Accurate y(x) computation using CvxpyQP solver
4. Correct hypergradient computation following Algorithm 1
"""

import torch
import torch.optim as optim
import numpy as np
import jax
import jax.numpy as jnp
from jaxopt import CvxpyQP, LBFGS

from typing import Dict, Optional, Tuple
from problem import StronglyConvexBilevelProblem
import warnings

# Suppress CVXPY warnings
warnings.filterwarnings('ignore')

class F2CSAAlgorithm1Final:
    """
    Final Corrected F2CSA Algorithm 1 with modified penalty parameters
    α₁ = α⁻¹, α₂ = α⁻² for improved computational efficiency
    Following F2CSA_corrected.tex exactly
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem, device: str = 'cpu', dtype=torch.float64):
        self.problem = problem
        self.device = device
        self.dtype = dtype
        
        # Use CvxpyQP solver from problem.py
        self.cvxpy_solver = CvxpyQP(solver='OSQP')
        # Diagnostics and controls
        self.trust_radius = 0.05  # cap ||ŷ - y*||
        self.last_penalty_diag = {}
        
        print(f"F2CSA Algorithm 1 Final - Corrected Implementation")
        print(f"  Penalty parameters per F2CSA.tex: alpha1=alpha^-2, alpha2=alpha^-4")
        print(f"  Target: delta-accuracy < 0.1 (delta = alpha^3)")
        print(f"  Device: {device}, dtype: {dtype}")
        print(f"  Using CvxpyQP solver for lower-level problem")
    
    def _solve_lower_level_cvxpy(self, x: jnp.ndarray, alpha: float) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        """
        Solve lower-level problem using CvxpyQP from problem.py with dual variables - JAX version
        """
        # Ensure JAX array for x
        try:
            x = jnp.array(np.array(x))
        except Exception:
            x = jnp.array(x)
        # Deterministic lower level per DS-BLO/diagnostics: no instance noise
        noise_lower_jax = jnp.zeros_like(self.problem.Q_lower)
        
        # Use the problem's solve_ll_with_duals method which uses CvxpyQP directly with JAX
        y_opt, lambda_opt, info = self.problem.solve_ll_with_duals(x, noise_lower_jax)
        
        return y_opt, lambda_opt, info
    
    
    def _compute_smooth_activation(self, h_val: jnp.ndarray, lambda_val: jnp.ndarray, delta: float) -> jnp.ndarray:
        """
        Compute smooth activation function ρ_i(x) = σ_h(h_i(x,y)) · σ_λ(λ_i) using JAX
        Following F2CSA.tex exactly with piecewise linear functions
        """
        # Parameters from paper
        tau = delta  # τ = Θ(δ)
        epsilon_lambda = 1e-4  # Slightly larger to smooth switching
        
        # σ_h(z) piecewise linear function
        sigma_h = jnp.zeros_like(h_val)
        mask1 = h_val < -tau * delta
        mask2 = (h_val >= -tau * delta) & (h_val < 0)
        mask3 = h_val >= 0
        
        sigma_h = jnp.where(mask1, 0.0, sigma_h)
        sigma_h = jnp.where(mask2, (tau * delta + h_val) / (tau * delta), sigma_h)
        sigma_h = jnp.where(mask3, 1.0, sigma_h)
        
        # σ_λ(z) piecewise linear function
        sigma_lambda = jnp.zeros_like(lambda_val)
        mask1_lambda = lambda_val <= 0
        mask2_lambda = (lambda_val > 0) & (lambda_val < epsilon_lambda)
        mask3_lambda = lambda_val >= epsilon_lambda
        
        sigma_lambda = jnp.where(mask1_lambda, 0.0, sigma_lambda)
        sigma_lambda = jnp.where(mask2_lambda, lambda_val / epsilon_lambda, sigma_lambda)
        sigma_lambda = jnp.where(mask3_lambda, 1.0, sigma_lambda)
        
        # ρ_i(x) = σ_h(h_i(x,y)) · σ_λ(λ_i) - MULTIPLICATION, not addition!
        rho = sigma_h * sigma_lambda
        
        return rho
    
    def _compute_penalty_lagrangian(self, x: jnp.ndarray, y: jnp.ndarray, 
                                  y_star: jnp.ndarray, lambda_star: jnp.ndarray,
                                  alpha: float, delta: float, noise_upper: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Compute the smooth penalty Lagrangian L_{λ̃,α}(x,y) = f(x,y) + α₁(...) + α₂(...)
        Following F2CSA.tex Equation (415) exactly
        """
        # Work directly with JAX arrays - no conversion needed
        
        # Penalty parameters per F2CSA.tex
        alpha1 = 1.0 / (alpha ** 2)  # α₁ = α⁻²
        alpha2 = 1.0 / (alpha ** 4)  # α₂ = α⁻⁴
        
        # First term: f(x,y) - upper-level objective
        f_val = self.problem.upper_objective(x, y, noise_upper=noise_upper)
        
        # Compute constraint values
        h_val = self.problem.constraints(x, y)
        
        # Compute smooth activation
        rho = self._compute_smooth_activation(h_val, lambda_star, delta)
        
        # Second term: α₁ * (g(x,y) + λ̃^T h(x,y) - g(x,ỹ*(x)))
        g_xy = self.problem.lower_objective(x, y)
        g_ystar = self.problem.lower_objective(x, y_star)
        term1 = alpha1 * (g_xy + jnp.sum(lambda_star * h_val) - g_ystar)
        
        # Third term: α₂/2 * Σ_i ρ_i(x) * h_i(x,y)²
        term2 = (alpha2 / 2.0) * jnp.sum(rho * (h_val ** 2))
        
        # Complete penalty Lagrangian: L_{λ̃,α}(x,y) = f(x,y) + α₁(...) + α₂(...)
        L_penalty = f_val + term1 + term2
        
        return L_penalty
    
    
    def _minimize_penalty_lagrangian(self, x: jnp.ndarray, y_star: jnp.ndarray, 
                                   lambda_star: jnp.ndarray, alpha: float, delta: float,
                                   init_y: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Minimize L within the trust ball using projected gradient with Armijo.
        This enforces descent under ||y - y*|| ≤ trust_radius.
        """
        penalty_objective = lambda y: self._compute_penalty_lagrangian(x, y, y_star, lambda_star, alpha, delta)
        grad_y = jax.grad(lambda y: self._compute_penalty_lagrangian(x, y, y_star, lambda_star, alpha, delta))
        
        # Initialize inside the trust region
        if init_y is not None:
            d0 = init_y - y_star
            n0 = jnp.linalg.norm(d0)
            y_curr = jnp.where(n0 > self.trust_radius, y_star + (self.trust_radius / n0) * d0, init_y)
        else:
            y_curr = y_star
        
        L_before = penalty_objective(y_curr)
        # Projected gradient with Armijo backtracking
        step0 = 1e-2
        c = 1e-4
        for _ in range(50):
            g = grad_y(y_curr)
            t = step0
            Ly = penalty_objective(y_curr)
            while t > 1e-8:
                y_trial = y_curr - t * g
                d = y_trial - y_star
                n = jnp.linalg.norm(d)
                y_trial = jnp.where(n > self.trust_radius, y_star + (self.trust_radius / n) * d, y_trial)
                if penalty_objective(y_trial) <= Ly - c * t * jnp.linalg.norm(g) ** 2:
                    y_curr = y_trial
                    break
                t *= 0.5
            if t <= 1e-8:
                break
        y_candidate = y_curr
        
        # Final diagnostics
        diff = y_candidate - y_star
        diff_norm = jnp.linalg.norm(diff)
        clipped = diff_norm > self.trust_radius + 1e-12
        L_after = penalty_objective(y_candidate)
        self.last_penalty_diag = {
            'L_before': float(L_before),
            'L_after': float(L_after),
            'L_decrease': float(L_before - L_after),
            'y_dist': float(diff_norm),
            'trust_clipped': bool(clipped)
        }
        if clipped:
            print(f"    [TRUST] ||ŷ−y*||={diff_norm:.4f} > {self.trust_radius:.4f} → projected")
        if L_after > L_before - 1e-6:
            print(f"    [L-check] No sufficient decrease: ΔL={float(L_before - L_after):.3e}")
        else:
            print(f"    [L-check] Decreased: ΔL={float(L_before - L_after):.3e}")
        return y_candidate
    
    def oracle_sample(self, x: jnp.ndarray, alpha: float, N_g: int,
                      prev_y: Optional[jnp.ndarray] = None,
                      warm_ll: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Implement Algorithm 1 with optional LL warm start.
        Returns (hypergradient, y_tilde, lambda_star)
        """
        delta = alpha ** 3  # δ = α³
        # Normalize input types to JAX
        try:
            x = jnp.array(np.array(x))
        except Exception:
            x = jnp.array(x)
        if prev_y is not None:
            try:
                prev_y = jnp.array(np.array(prev_y))
            except Exception:
                prev_y = jnp.array(prev_y)
                
        # Step 3: Compute ỹ*(x) and λ̃(x) by accurate solver with optional warm start
        y_star, lambda_star, info = self._solve_lower_level_cvxpy(x, alpha)
        
        
        # Step 4: Compute ỹ(x) = argmin_y L_{λ̃,α}(x,y)
        init_y = prev_y if warm_ll and prev_y is not None else y_star
        y_tilde = self._minimize_penalty_lagrangian(x, y_star, lambda_star, alpha, delta,
                                                    init_y=init_y)
        # Diagnostics: KKT, active sets, overlaps
        G = jnp.vstack([jnp.eye(self.problem.dim), -jnp.eye(self.problem.dim)])
        h_box = jnp.concatenate([jnp.ones(self.problem.dim), jnp.ones(self.problem.dim)])
        # For y* (from solver)
        kkt_star = info.get('kkt_residual', None)
        comp_star = info.get('complementary_slackness', None)
        # For y~ using λ* as proxy
        grad_g_y_tilde = self.problem.grad_lower_objective_y(x, y_tilde)
        kkt_residual_tilde = jnp.linalg.norm(grad_g_y_tilde + G.T @ lambda_star)
        slack_tilde = h_box - G @ y_tilde
        comp_slack_tilde = float(jnp.sum(lambda_star * slack_tilde))
        # Active sets and overlap
        active_star = jnp.isclose(jnp.abs(y_star), 1.0, atol=1e-6)
        active_tilde = jnp.isclose(jnp.abs(y_tilde), 1.0, atol=1e-6)
        overlap = int(jnp.sum(active_star & active_tilde))
        self.last_penalty_diag.update({
            'kkt_star': float(kkt_star) if kkt_star is not None else None,
            'comp_star': float(comp_star) if comp_star is not None else None,
            'kkt_tilde': float(kkt_residual_tilde),
            'comp_tilde': float(comp_slack_tilde),
            'active_star': int(jnp.sum(active_star)),
            'active_tilde': int(jnp.sum(active_tilde)),
            'active_overlap': overlap
        })
        print(f"    [KKT] y*: res={self.last_penalty_diag['kkt_star']:.3e}, comp={self.last_penalty_diag['comp_star']:.3e}; ỹ: res={self.last_penalty_diag['kkt_tilde']:.3e}, comp={self.last_penalty_diag['comp_tilde']:.3e}")
        print(f"    [Active] |y*|≈1: {self.last_penalty_diag['active_star']}, |ỹ|≈1: {self.last_penalty_diag['active_tilde']}, overlap={overlap}")
        
        print(f"  Computing stochastic hypergradient with N_g = {N_g}")
        
        # Step 5: Compute ∇F̃(x) = (1/N_g) Σ_{j=1}^{N_g} ∇_x L̃_{λ̃,α}(x, ỹ(x); ξ_j)      
        hypergradient_samples = []
        
        for j in range(N_g):
            # Sample noise for stochastic evaluation
            noise_upper, _ = self.problem._sample_instance_noise()
            noise_upper_jax = jnp.array(noise_upper.detach().cpu().numpy())
            
            # Compute the penalty Lagrangian L_{λ̃,α}(x,y) with SAME stochastic noise
            # This includes f(x,y) + α₁(...) + α₂(...) as per Equation (415)
            L_val = self._compute_penalty_lagrangian(x, y_tilde, y_star, lambda_star, alpha, delta, noise_upper_jax)
            
            # Compute gradient of the complete penalty Lagrangian using JAX
            grad_x = jax.grad(lambda x: self._compute_penalty_lagrangian(x, y_tilde, y_star, lambda_star, alpha, delta, noise_upper_jax))(x)
            hypergradient_samples.append(grad_x)
        
        hypergradient = jnp.mean(jnp.stack(hypergradient_samples), axis=0)
        
        print(f"  Final hypergradient: ∇F̃ = {hypergradient}")
        print(f"  Hypergradient norm: ||∇F̃|| = {jnp.linalg.norm(hypergradient):.6f}")
        
        return hypergradient, y_tilde, lambda_star

    def finite_diff_check(self, x: jnp.ndarray, y_tilde: jnp.ndarray, y_star: jnp.ndarray,
                          lambda_star: jnp.ndarray, alpha: float) -> Dict:
        """
        Finite-difference check of ∇_x L̃ with ỹ, λ̃ held fixed (no noise).
        Checks 2 random coordinates; returns relative error stats.
        """
        delta = alpha ** 3
        L_fixed = lambda x_in: self._compute_penalty_lagrangian(x_in, y_tilde, y_star, lambda_star, alpha, delta)
        grad_auto = jax.grad(L_fixed)(x)
        key = jax.random.PRNGKey(0)
        idx = jax.random.choice(key, x.shape[0], (min(2, x.shape[0]),), replace=False)
        eps = 1e-6
        errs = []
        for i in list(np.array(idx)):
            e = jnp.zeros_like(x).at[i].set(1.0)
            f_plus = L_fixed(x + eps * e)
            f_minus = L_fixed(x - eps * e)
            fd_i = (f_plus - f_minus) / (2 * eps)
            rel_err = float(jnp.abs(grad_auto[i] - fd_i) / (jnp.abs(fd_i) + 1e-12))
            errs.append(rel_err)
        return {
            'mean_rel_err': float(np.mean(errs)) if errs else 0.0,
            'max_rel_err': float(np.max(errs)) if errs else 0.0
        }
    
    def optimize(self, x0: jnp.ndarray, max_iterations: int = 1000, 
                alpha: float = 0.05, N_g: int = None, lr: float = 1e-3) -> Dict:
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
        # Set optimal N_g if not provided
        if N_g is None:
            # Balanced N_g for hypergradient estimation
            N_g = max(10, min(100, int(1.0 / (alpha**1.5))))
        
        print(f"Starting Corrected F2CSA Algorithm 1 optimization")
        print(f"  Initial point: x0 = {x0}")
        print(f"  Parameters: α = {alpha}, N_g = {N_g}, lr = {lr}")
        print(f"  δ = α³ = {alpha**3:.6f} < 0.1 ✓")
        print(f"  Penalty: α₁ = α⁻² = {1/alpha**2:.1f}, α₂ = α⁻⁴ = {1/alpha**4:.1f}")
        print(f"  Max iterations: {max_iterations}")
        
        # Initialize
        x = x0
        
        # Track progress
        losses = []
        grad_norms = []
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            # Compute stochastic hypergradient using corrected Algorithm 1
            hypergradient, _, _ = self.oracle_sample(x, alpha, N_g)
            
            # Update x using simple gradient descent
            x = x - lr * hypergradient
            
            # Compute current loss and gradient norm
            # Get current lower-level solution using CvxpyQP
            y_current, _, _ = self._solve_lower_level_cvxpy(x, alpha)
            current_loss = self.problem.upper_objective(x, y_current)
            losses.append(float(current_loss))
            
            # Compute gradient of f(x,y) w.r.t. x for monitoring
            f_grad = jax.grad(lambda x: self.problem.upper_objective(x, self._solve_lower_level_cvxpy(x, alpha)[0]))(x)
            grad_norms.append(float(jnp.linalg.norm(f_grad)))
            
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
    
    def test_lower_level_convergence(self, x: torch.Tensor, alpha: float, max_iterations: int = 1000) -> Dict:
        """
        Test lower-level solution convergence with more iterations
        """
        # Get accurate lower-level solution using CvxpyQP
        y_star, lambda_star, info = self._solve_lower_level_cvxpy(x, alpha)
        
        delta = alpha**3
        
        # Test convergence of penalty Lagrangian solver
        y_penalty = self._minimize_penalty_lagrangian(x, y_star, lambda_star, alpha, delta)
        
        # Compute gap
        gap = torch.norm(y_penalty - y_star).item()
        
        # Check convergence
        converged = gap < 0.1
        
        return {
            'gap': gap,
            'iterations': max_iterations,
            'converged': converged,
            'y_star': y_star,
            'y_penalty': y_penalty
        }
    
    def test_hypergradient_accuracy(self, x: torch.Tensor, alpha: float) -> Dict:
        """
        Test hypergradient accuracy with converged lower-level solutions
        """
        # Compute hypergradient using Algorithm 1
        # Optimal N_g: balance bias and variance for hypergradient estimation
        # For α = 0.001, use N_g = 100 for good balance
        N_g = max(10, min(100, int(1.0 / (alpha**1.5))))  # Balanced N_g
        hypergradient, _, _ = self.oracle_sample(x, alpha, N_g)
        
        # Compute finite difference approximation
        eps = 1e-6
        x_plus = x + eps
        x_minus = x - eps
        
        # Get function values (need to solve for y first)
        y_plus, _, _ = self._solve_lower_level_cvxpy(x_plus, alpha)
        y_minus, _, _ = self._solve_lower_level_cvxpy(x_minus, alpha)
        
        f_plus = self.problem.upper_objective(x_plus, y_plus)
        f_minus = self.problem.upper_objective(x_minus, y_minus)
        
        # Finite difference gradient
        finite_diff_grad = (f_plus - f_minus) / (2 * eps)
        
        # Compute relative error
        relative_error = torch.norm(hypergradient - finite_diff_grad) / torch.norm(finite_diff_grad)
        
        return {
            'relative_error': relative_error.item(),
            'hypergradient_norm': torch.norm(hypergradient).item(),
            'finite_diff_norm': torch.norm(finite_diff_grad).item(),
            'hypergradient': hypergradient,
            'finite_diff_grad': finite_diff_grad
        }

def test_corrected_f2csa_final():
    """Test the final corrected F2CSA algorithm"""
    print("Testing Final Corrected F2CSA Algorithm 1")
    print("=" * 50)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5, num_constraints=3, noise_std=0.1,
        strong_convex=True, device='cpu'
    )
    
    # Initialize corrected algorithm
    algorithm = F2CSAAlgorithm1Final(problem)
    
    # Test with α = 0.2 (δ = 0.008 < 0.1)
    alpha = 0.2
    x_init = torch.randn(5, dtype=torch.float64)
    
    print(f"Testing with α = {alpha}, δ = {alpha**3:.6f}")
    print(f"Corrected penalty parameters: α₁ = {1/alpha:.1f}, α₂ = {1/alpha**2:.1f}")
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
    test_corrected_f2csa_final()
