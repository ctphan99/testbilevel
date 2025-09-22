#!/usr/bin/env python3
"""
Natural Bilevel Optimization Problem with JAX Ecosystem
Uses constraint form: h(x,y) = Ax + By - b Γëñ 0
"""

import torch
import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Optional
import warnings

class StronglyConvexBilevelProblem:
    """
    Natural bilevel optimization problem with constraints h(x,y) = Ax + By - b Γëñ 0
    
    Upper level: min_x f(x, y*(x)) where y*(x) solves:
    Lower level: min_y g(x,y) subject to h(x,y) = Ax + By - b Γëñ 0
    """
    
    def __init__(self, dim: int = 10, num_constraints: int = 5, noise_std: float = 0.01, 
                 strong_convex: bool = True, device: str = 'cpu'):
        """
        Initialize natural bilevel problem
        
        Args:
            dim: Problem dimension
            num_constraints: Number of inequality constraints
            noise_std: Standard deviation for instance noise
            strong_convex: Whether to ensure strong convexity
            device: Device for tensors
        """
        self.dim = dim
        self.num_constraints = num_constraints
        self.noise_std = noise_std
        self.strong_convex = strong_convex
        self.device = device
        self.dtype = torch.float64
        
        # Generate problem parameters directly as JAX arrays
        self._setup_problem_parameters()
        
    def _setup_problem_parameters(self):
        """Generate natural problem parameters directly as JAX arrays"""
        
        # Upper-level objective: f(x,y) = 0.5 * x^T Q_upper x + c_upper^T x + 0.5 * y^T P y + x^T P y
        param_scale = 1.0
        
        # Generate JAX arrays directly
        key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)
        
        # Upper-level objective: f(x,y) = 0.5 * x^T Q_upper x + c_upper^T x + 0.5 * y^T P y + x^T P y
        Q_upper_raw = jax.random.normal(key1, (self.dim, self.dim)) * param_scale
        self.Q_upper = Q_upper_raw @ Q_upper_raw.T  # Make positive definite
        
        if self.strong_convex:
            # Ensure strong convexity using JAX
            eigenvals = jnp.linalg.eigvals(self.Q_upper).real
            min_eigenval = jnp.min(eigenvals)
            if min_eigenval <= 0:
                self.Q_upper += (1.0 - min_eigenval) * jnp.eye(self.dim)
        
        self.c_upper = jax.random.normal(key2, (self.dim,)) * param_scale
        P_raw = jax.random.normal(key3, (self.dim, self.dim)) * param_scale
        # Make P matrix symmetric
        self.P = (P_raw + P_raw.T) / 2
        self.x_target = jax.random.normal(key4, (self.dim,)) * 0.1
        
        # Lower-level objective: g(x,y) = 0.5 * y^T Q_lower y + c_lower^T y + (c_lower * x)^T y
        Q_lower_raw = jax.random.normal(key5, (self.dim, self.dim)) * param_scale
        self.Q_lower = Q_lower_raw @ Q_lower_raw.T  # Make positive definite
        
        if self.strong_convex:
            # Ensure strong convexity using JAX
            eigenvals = jnp.linalg.eigvals(self.Q_lower).real
            min_eigenval = jnp.min(eigenvals)
            if min_eigenval <= 0:
                self.Q_lower += (1.0 - min_eigenval) * jnp.eye(self.dim)
        
        self.c_lower = jax.random.normal(key6, (self.dim,)) * param_scale
        
        # Initialize qΓéÇ perturbation for SSIGD (fixed throughout optimization)
        self.q_perturbation = jax.random.normal(jax.random.PRNGKey(123), (self.dim,)) * 1e-6
        
        # Box constraints: |y_i| Γëñ 1 for all i (y Γëñ 1 and -y Γëñ 1)
        self.num_constraints = 2 * self.dim
        
        # Define all JAX functions
        self._define_jax_functions()
        
        # Create JAX-compiled versions for optimal performance
        self._setup_jax_compiled_functions()
        
        # Print problem information
        print(f"Natural Bilevel Problem (dim={self.dim}, constraints={self.num_constraints})")
        
        # Check box constraint feasibility at origin
        x0 = jnp.zeros(self.dim)
        y0 = jnp.zeros(self.dim)
        h0 = self.constraints(x0, y0)
        max_violation = jnp.max(jnp.clip(h0, 0, None))
        print(f"Constraint violations at origin: {max_violation:.6f}")
        print("Origin is feasible - constraints may not be active" if max_violation <= 1e-6 else "Natural constraint violations present - F2CSA penalty mechanism will engage")

        # Verify strong convexity using JAX arrays
        upper_eigenvals = jnp.linalg.eigvals(self.Q_upper).real
        lower_eigenvals = jnp.linalg.eigvals(self.Q_lower).real

        print(f"Upper level strong convexity: lambda_min={jnp.min(upper_eigenvals):.3f}, lambda_max={jnp.max(upper_eigenvals):.3f}")
        print(f"Lower level strong convexity: lambda_min={jnp.min(lower_eigenvals):.3f}, lambda_max={jnp.max(lower_eigenvals):.3f}")
        print(f"[OK] JAX parameters and functions initialized (dim={self.dim})")
    
    def solve_ll(self, x: jnp.ndarray, noise_lower: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Solve clean lower-level problem using CVXPY QP - returns only primal solution"""
        y_opt, _, _ = self.solve_ll_with_duals(x, noise_lower)
        return y_opt
    
    def solve_ll_with_duals(self, x: jnp.ndarray, noise_lower: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        """Solve lower-level problem using CVXPY QP and return primal, dual variables, and info"""
        # Prepare QP parameters
        if noise_lower is None:
            noise_lower_jax = jnp.zeros_like(self.Q_lower)
        else:
            noise_lower_jax = noise_lower
        
        # QP objective: 0.5 * y^T Q_lower y + c_lower^T y + (c_lower * x)^T y + qΓéÇ^T y
        Q_lower_noisy = self.Q_lower + noise_lower_jax
        
        # Ensure Q_lower_noisy is positive definite for CVXPY
        eigenvals = jnp.linalg.eigvals(Q_lower_noisy).real
        min_eigenval = jnp.min(eigenvals)
        if min_eigenval <= 1e-6:
            Q_lower_noisy += (1e-6 - min_eigenval) * jnp.eye(self.dim)
        
        c_lower_total = self.c_lower + self.c_lower * x + self.q_perturbation
        
        # Box constraints: -1 Γëñ y_i Γëñ 1 for all i
        # Convert to standard form: G y Γëñ h
        # y Γëñ 1  ΓåÆ  I y Γëñ 1
        # -y Γëñ 1  ΓåÆ  -I y Γëñ 1
        G = jnp.vstack([jnp.eye(self.dim), -jnp.eye(self.dim)])
        h = jnp.concatenate([jnp.ones(self.dim), jnp.ones(self.dim)])
        
        # Use CVXPY QP solver
        from jaxopt import CvxpyQP
        
        qp = CvxpyQP()
        # Convert all parameters to NumPy arrays for CVXPY and enforce symmetry
        Q_np = np.array(Q_lower_noisy)
        Q_np = (Q_np + Q_np.T) / 2.0
        # Ensure positive semidefinite: shift by smallest eigenvalue if needed
        try:
            eigvals_np = np.linalg.eigvalsh(Q_np)
            min_eig_np = float(eigvals_np.min())
        except Exception:
            min_eig_np = 0.0
        if min_eig_np < 1e-12:
            Q_np = Q_np + (1e-12 - min_eig_np) * np.eye(self.dim)
        c_np = np.array(c_lower_total)
        G_np = np.array(G)
        h_np = np.array(h)
        # Initial guess for y
        y_init = jnp.zeros(self.dim)
        sol = qp.run(params_obj=(Q_np, c_np), 
                    params_eq=None, 
                    params_ineq=(G_np, h_np),
                    init_params=y_init).params
        
        y_opt = sol.primal
        lambda_opt = sol.dual_ineq
        
        # Check constraint satisfaction
        constraint_violations = G @ y_opt - h
        max_violation = jnp.max(jnp.clip(constraint_violations, 0, None))
        
        # Check KKT optimality conditions
        grad_obj = Q_lower_noisy @ y_opt + c_lower_total
        kkt_residual = grad_obj + G.T @ lambda_opt
        kkt_norm = jnp.linalg.norm(kkt_residual)
        
        # Check complementary slackness
        slack = h - G @ y_opt
        comp_slack = jnp.sum(lambda_opt * slack)
        
        # Create info dictionary
        info = {
            'solver': 'cvxpy',
            'status': 'optimal',
            'objective_value': float(0.5 * jnp.dot(y_opt, Q_lower_noisy @ y_opt) + jnp.dot(c_lower_total, y_opt)),
            'constraint_violations': float(max_violation),
            'kkt_residual': float(kkt_norm),
            'complementary_slackness': float(comp_slack),
            'active_constraints': int(jnp.sum(constraint_violations > -1e-6))
        }
        
        return y_opt, lambda_opt, info
    
    
    
    def _define_jax_functions(self):
        """Define all JAX functions as methods"""
        
        def upper_objective(x: jnp.ndarray, y: jnp.ndarray, 
                           noise_upper: Optional[jnp.ndarray] = None) -> jnp.ndarray:
            """Upper-level objective: f(x,y) = 0.5 * x^T Q_upper x + c_upper^T x + 0.5 * y^T P y + x^T P y"""
            if noise_upper is None:
                noise_upper = jnp.zeros_like(self.Q_upper)
            
            Q_upper_noisy = self.Q_upper + noise_upper
            return (0.5 * jnp.dot(x, Q_upper_noisy @ x) + 
                   jnp.dot(self.c_upper, x) + 
                   0.5 * jnp.dot(y, self.P @ y) + 
                   jnp.dot(x, self.P @ y))
        
        def lower_objective(x: jnp.ndarray, y: jnp.ndarray, 
                           noise_lower: Optional[jnp.ndarray] = None) -> jnp.ndarray:
            """Lower-level objective: g(x,y) = 0.5 * y^T Q_lower y + c_lower^T y + (c_lower * x)^T y + qΓéÇ^T y"""
            if noise_lower is None:
                noise_lower = jnp.zeros_like(self.Q_lower)
            
            Q_lower_noisy = self.Q_lower + noise_lower
            return (0.5 * jnp.dot(y, Q_lower_noisy @ y) + 
                   jnp.dot(self.c_lower, y) + 
                   jnp.dot(self.c_lower * x, y) +
                   jnp.dot(self.q_perturbation, y))  # Add qΓéÇ perturbation
        
        def constraints(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            """Box constraints: y Γëñ 1 and -y Γëñ 1"""
            return jnp.concatenate([y - 1.0, -y - 1.0])
        
        def grad_upper_objective_x(x: jnp.ndarray, y: jnp.ndarray, 
                                  noise_upper: Optional[jnp.ndarray] = None) -> jnp.ndarray:
            """Gradient of upper objective w.r.t. x: Γêçx f(x,y)"""
            if noise_upper is None:
                noise_upper = jnp.zeros_like(self.Q_upper)
            
            Q_upper_noisy = self.Q_upper + noise_upper
            return Q_upper_noisy @ x + self.c_upper + self.P @ y
        
        def grad_upper_objective_y(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            """Gradient of upper objective w.r.t. y: Γêçy f(x,y)"""
            return self.P @ y + self.P.T @ x
        
        def grad_lower_objective_x(x: jnp.ndarray, y: jnp.ndarray, 
                                  noise_lower: Optional[jnp.ndarray] = None) -> jnp.ndarray:
            """Gradient of lower objective w.r.t. x: Γêçx g(x,y)"""
            if noise_lower is None:
                noise_lower = jnp.zeros_like(self.Q_lower)
            
            return self.c_lower * y
        
        def grad_lower_objective_y(x: jnp.ndarray, y: jnp.ndarray, 
                                  noise_lower: Optional[jnp.ndarray] = None) -> jnp.ndarray:
            """Gradient of lower objective w.r.t. y: Γêçy g(x,y)"""
            if noise_lower is None:
                noise_lower = jnp.zeros_like(self.Q_lower)
            
            Q_lower_noisy = self.Q_lower + noise_lower
            return Q_lower_noisy @ y + self.c_lower + self.c_lower * x + self.q_perturbation
        
        def hess_lower_objective_yy(x: jnp.ndarray, y: jnp.ndarray, 
                                   noise_lower: Optional[jnp.ndarray] = None) -> jnp.ndarray:
            """Hessian of lower objective w.r.t. y: Γêçyy g(x,y)"""
            if noise_lower is None:
                noise_lower = jnp.zeros_like(self.Q_lower)
            
            Q_lower_noisy = self.Q_lower + noise_lower
            return Q_lower_noisy
        
        def jac_lower_objective_yx(x: jnp.ndarray, y: jnp.ndarray, 
                                  noise_lower: Optional[jnp.ndarray] = None) -> jnp.ndarray:
            """Jacobian of Γêçy g w.r.t. x: Γêçyx g(x,y)"""
            if noise_lower is None:
                noise_lower = jnp.zeros_like(self.Q_lower)
            
            return jnp.diag(self.c_lower)
        
        def compute_implicit_gradient(x: jnp.ndarray, y: jnp.ndarray, 
                                     noise_lower: Optional[jnp.ndarray] = None) -> jnp.ndarray:
            """Compute implicit gradient using the research paper formula:
            Γêç╠éF(x) = Γêçx f(x, ┼╖(x)) + [Γêç╠éy*(x)]^T Γêçy f(x, ┼╖(x))
            """
            # Step 1: Compute Γêçx f(x, ┼╖(x))
            grad_x_f = grad_upper_objective_x(x, y)
            
            # Step 2: Compute Γêçy f(x, ┼╖(x))
            grad_y_f = grad_upper_objective_y(x, y)
            
            # Step 3: Compute Γêç╠éy*(x) using implicit function theorem
            hess_yy_g = hess_lower_objective_yy(x, y, noise_lower)
            grad_yx_g = jac_lower_objective_yx(x, y, noise_lower)
            
            # Apply implicit function theorem: Γêçy* = -[Γêçyy g]^{-1} Γêçyx g
            try:
                grad_y_star = -jnp.linalg.solve(hess_yy_g, grad_yx_g)
            except jnp.linalg.LinAlgError:
                grad_y_star = -jnp.linalg.pinv(hess_yy_g) @ grad_yx_g
            
            # Step 4: Apply the exact formula
            return grad_x_f + grad_y_star.T @ grad_y_f
        
        def compute_all_gradients(x: jnp.ndarray, y: jnp.ndarray, 
                                 noise_upper: Optional[jnp.ndarray] = None,
                                 noise_lower: Optional[jnp.ndarray] = None) -> Dict[str, jnp.ndarray]:
            """Compute all gradients in one unified call for comprehensive analysis"""
            # Upper-level gradients
            grad_x_f = grad_upper_objective_x(x, y, noise_upper)
            grad_y_f = grad_upper_objective_y(x, y)
            
            # Lower-level gradients
            grad_x_g = grad_lower_objective_x(x, y, noise_lower)
            grad_y_g = grad_lower_objective_y(x, y, noise_lower)
            
            # Hessian and Jacobian for implicit differentiation
            hess_yy_g = hess_lower_objective_yy(x, y, noise_lower)
            jac_yx_g = jac_lower_objective_yx(x, y, noise_lower)
            
            # Implicit function theorem: Γêçy* = -[Γêçyy g]^{-1} Γêçyx g
            try:
                grad_y_star = -jnp.linalg.solve(hess_yy_g, jac_yx_g)
            except jnp.linalg.LinAlgError:
                grad_y_star = -jnp.linalg.pinv(hess_yy_g) @ jac_yx_g
            
            # Implicit gradient: Γêç╠éF(x) = Γêçx f(x, ┼╖(x)) + [Γêç╠éy*(x)]^T Γêçy f(x, ┼╖(x))
            grad_F = grad_x_f + grad_y_star.T @ grad_y_f
            
            return {
                'grad_x_f': grad_x_f,
                'grad_y_f': grad_y_f,
                'grad_x_g': grad_x_g,
                'grad_y_g': grad_y_g,
                'hess_yy_g': hess_yy_g,
                'jac_yx_g': jac_yx_g,
                'grad_y_star': grad_y_star,
                'grad_F': grad_F
            }
        
        # Assign functions as methods
        self.upper_objective = upper_objective
        self.lower_objective = lower_objective
        self.constraints = constraints
        self.grad_upper_objective_x = grad_upper_objective_x
        self.grad_upper_objective_y = grad_upper_objective_y
        self.grad_lower_objective_x = grad_lower_objective_x
        self.grad_lower_objective_y = grad_lower_objective_y
        self.hess_lower_objective_yy = hess_lower_objective_yy
        self.jac_lower_objective_yx = jac_lower_objective_yx
        self.compute_implicit_gradient = compute_implicit_gradient
        self.compute_all_gradients = compute_all_gradients

        def compute_constrained_implicit_gradient(x: jnp.ndarray,
                                                  y: jnp.ndarray,
                                                  lambda_ineq: Optional[jnp.ndarray] = None,
                                                  tol: float = 1e-6,
                                                  noise_lower: Optional[jnp.ndarray] = None) -> jnp.ndarray:
            """JIT-safe constrained implicit gradient using diagonal penalization.
            Approximate dy_A/dx Γëê 0 by adding a large penalty to active coordinates
            on the Hessian diagonal, then solve full-system dy/dx = -H_tilde^{-1} J.
            """
            grad_x_f = grad_upper_objective_x(x, y)
            grad_y_f = grad_upper_objective_y(x, y)

            H = hess_lower_objective_yy(x, y, noise_lower)
            J = jac_lower_objective_yx(x, y, noise_lower)

            # Active set detection (box) as boolean mask
            if lambda_ineq is None:
                active_mask = (jnp.abs(y) >= 1.0 - tol)
            else:
                box_active = (jnp.abs(y) >= 1.0 - tol)
                dual_active = (lambda_ineq[: self.dim] > tol) | (lambda_ineq[self.dim :] > tol)
                active_mask = box_active | dual_active

            # Build penalized Hessian to freeze active coordinates
            mask_f = active_mask.astype(H.dtype)
            gamma = jnp.asarray(1e6, dtype=H.dtype)
            H_tilde = H + jnp.diag(gamma * mask_f)

            # Solve for dy/dx
            try:
                dy_dx = -jnp.linalg.solve(H_tilde, J)
            except jnp.linalg.LinAlgError:
                dy_dx = -jnp.linalg.pinv(H_tilde) @ J

            grad_F = grad_x_f + dy_dx.T @ grad_y_f
            return grad_F

        self.compute_constrained_implicit_gradient = compute_constrained_implicit_gradient

        # --- Custom derivative path for y*(x) using JAX custom_vjp ---
        # Non-JAX forward that calls the QP solver
        def _y_star_py(x_in: jnp.ndarray, noise_lower: Optional[jnp.ndarray] = None) -> jnp.ndarray:
            y_opt, _, _ = self.solve_ll_with_duals(x_in, noise_lower)
            return y_opt

        @jax.custom_vjp
        def y_star(x_in: jnp.ndarray, noise_lower: Optional[jnp.ndarray] = None) -> jnp.ndarray:
            return _y_star_py(x_in, noise_lower)

        def y_star_fwd(x_in: jnp.ndarray, noise_lower: Optional[jnp.ndarray]):
            y = _y_star_py(x_in, noise_lower)
            return y, (x_in, y, noise_lower)

        def y_star_bwd(res, y_bar: jnp.ndarray):
            x_in, y, noise_lower = res
            # Implicit differentiation: dy/dx = -[Γêçyy g]^{-1} Γêçyx g
            H = self.hess_lower_objective_yy(x_in, y, noise_lower)
            J = self.jac_lower_objective_yx(x_in, y, noise_lower)
            try:
                V = jnp.linalg.solve(H, J)
            except jnp.linalg.LinAlgError:
                V = jnp.linalg.pinv(H) @ J
            dy_dx = -V
            x_bar = dy_dx.T @ y_bar
            return (x_bar, None)

        y_star.defvjp(y_star_fwd, y_star_bwd)

        # Convenience: upper-level objective composed with y*(x)
        def f_value_custom(x_in: jnp.ndarray, noise_lower: Optional[jnp.ndarray] = None) -> jnp.ndarray:
            y = y_star(x_in, noise_lower)
            return self.upper_objective(x_in, y)

        # Expose methods (do not jit these to avoid tracing Python QP)
        self.y_star = y_star
        self.f_value_custom = f_value_custom
    
    def _setup_jax_compiled_functions(self):
        """Setup JAX-compiled functions for optimal performance"""
        # JAX-compiled versions for optimal performance
        self.upper_objective_compiled = jax.jit(self.upper_objective)
        self.lower_objective_compiled = jax.jit(self.lower_objective)
        self.grad_upper_objective_x_compiled = jax.jit(self.grad_upper_objective_x)
        self.grad_upper_objective_y_compiled = jax.jit(self.grad_upper_objective_y)
        self.grad_lower_objective_x_compiled = jax.jit(self.grad_lower_objective_x)
        self.grad_lower_objective_y_compiled = jax.jit(self.grad_lower_objective_y)
        self.hess_lower_objective_yy_compiled = jax.jit(self.hess_lower_objective_yy)
        self.jac_lower_objective_yx_compiled = jax.jit(self.jac_lower_objective_yx)
        self.compute_implicit_gradient_compiled = jax.jit(self.compute_implicit_gradient)
        self.compute_constrained_implicit_gradient_compiled = jax.jit(self.compute_constrained_implicit_gradient)
        self.compute_all_gradients_compiled = jax.jit(self.compute_all_gradients)
        
        print(f"[OK] JAX-compiled functions ready for optimal performance")
    
    def _sample_instance_noise(self):
        """Sample instance noise for upper and lower level problems"""
        # Sample noise for upper level (Q_upper)
        noise_upper_raw = torch.randn(self.dim, self.dim, device='cpu', dtype=torch.float64) * self.noise_std
        # Make noise symmetric
        noise_upper = (noise_upper_raw + noise_upper_raw.T) / 2
        
        # Sample noise for lower level (Q_lower) 
        noise_lower_raw = torch.randn(self.dim, self.dim, device='cpu', dtype=torch.float64) * self.noise_std
        # Make noise symmetric
        noise_lower = (noise_lower_raw + noise_lower_raw.T) / 2
        
        return noise_upper, noise_lower
    
    def compute_upper_objective_torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute upper-level objective using PyTorch tensors"""
        # Convert to JAX arrays
        x_jax = jnp.array(x.detach().cpu().numpy())
        y_jax = jnp.array(y.detach().cpu().numpy())
        
        # Use JAX-compiled function
        F_jax = self.upper_objective_compiled(x_jax, y_jax)
        
        # Convert back to PyTorch tensor
        return torch.tensor(np.array(F_jax), dtype=torch.float64, device='cpu')
    
