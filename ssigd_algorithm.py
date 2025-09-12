#!/usr/bin/env python3
"""
[S]SIGD ([Stochastic] Smoothed Implicit Gradient Descent) Algorithm
Implementation based on ssigd-paper.tex

Key Features:
1. Single perturbation for almost-sure differentiability
2. Moreau envelope-based convergence analysis
3. Weakly convex implicit function assumption
4. Stochastic bilevel optimization with linear constraints
5. Convergence to ε-stationary point in O(ε⁻⁴) iterations
"""

import torch
import numpy as np
import cvxpy as cp
from typing import Dict, Tuple, Optional
import time
from torch.optim import Adam, SGD
import warnings

class SSIGDAlgorithm:
    """
    [S]SIGD: [Stochastic] Smoothed Implicit Gradient Descent
    
    Core mechanisms:
    1. Single perturbation: Linear perturbation q^T y for differentiability
    2. Moreau envelope: For weakly convex functions
    3. Stochastic implicit gradient: Handles stochastic upper-level objective
    4. Projection-based updates: Ensures feasibility
    """
    
    def __init__(self, 
                 problem,
                 # [S]SIGD specific parameters
                 q_distribution: str = 'normal',
                 q_std: float = 0.01,
                 fixed_perturbation: bool = True,
                 rho: float = 0.1,
                 projection_radius: float = 10.0,
                 device: str = 'cpu',
                 seed: int = 42):
        """
        Initialize [S]SIGD algorithm
        
        Args:
            problem: StronglyConvexBilevelProblem instance
            q_distribution: Distribution for perturbation q ('normal' or 'uniform')
            q_std: Standard deviation for perturbation q
            fixed_perturbation: Whether to use fixed perturbation throughout training
            rho: Weak convexity parameter
            projection_radius: Radius for projection onto feasible set
            device: Device to use
            seed: Random seed
        """
        # Validate inputs
        if problem is None:
            raise ValueError("Problem instance is required")
        if not hasattr(problem, 'dim') or not hasattr(problem, 'upper_objective') or not hasattr(problem, 'compute_gap'):
            raise ValueError("Problem must be a StronglyConvexBilevelProblem instance")
        if q_std <= 0 or rho <= 0 or projection_radius <= 0:
            raise ValueError("q_std, rho, and projection_radius must be positive")
        
        # Validate device
        if device != 'cpu' and not torch.cuda.is_available():
            print(f"Warning: CUDA not available, falling back to CPU")
            device = 'cpu'
        
        self.problem = problem
        self.device = device
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Perturbation parameters ([S]SIGD specific)
        self.q_distribution = q_distribution
        self.q_std = q_std
        self.fixed_perturbation = fixed_perturbation
        
        # Moreau envelope parameters
        self.rho = rho
        self.lambda_moreau = 1.0 / self.rho
        
        # Projection parameters
        self.projection_radius = projection_radius
        
        # Storage for analysis
        self.history = {
            'iterations': [],
            'objectives': [],
            'gradients': [],
            'gaps': [],
            'times': []
        }
        
        # Fixed perturbation (single draw as per [S]SIGD)
        if self.fixed_perturbation:
            self.fixed_q = self.sample_perturbation()
        
        
    def sample_perturbation(self) -> torch.Tensor:
        """
        Sample perturbation q for the lower-level objective
        This is the single perturbation in [S]SIGD for differentiability
        """
        if self.q_distribution == 'normal':
            q = torch.randn(self.problem.dim, device=self.device, dtype=self.problem.dtype) * self.q_std
        elif self.q_distribution == 'uniform':
            q = (torch.rand(self.problem.dim, device=self.device, dtype=self.problem.dtype) - 0.5) * 2 * self.q_std
        else:
            raise ValueError(f"Unknown perturbation distribution: {self.q_distribution}")
        
        return q
        
    def solve_smoothed_lower_level(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Solve the smoothed lower-level problem: min_y h(x,y) + q^T y s.t. Ay ≤ b
        
        This implements the smoothed LL problem from [S]SIGD paper
        """
        # Use fixed perturbation if enabled
        q = self.fixed_q if self.fixed_perturbation else self.sample_perturbation()
        
        # Convert to numpy for CVXPy
        x_np = x.detach().cpu().numpy()
        q_np = q.detach().cpu().numpy()
        
        # Problem parameters
        Q_lower_np = self.problem.Q_lower.detach().cpu().numpy()
        c_lower_np = self.problem.c_lower.detach().cpu().numpy()
        P_np = self.problem.P.detach().cpu().numpy()
        A_np = self.problem.A.detach().cpu().numpy()
        B_np = self.problem.B.detach().cpu().numpy()
        b_np = self.problem.b.detach().cpu().numpy()
        
        # Create CVXPy problem with perturbation
        y_var = cp.Variable(self.problem.dim)
        
        # Smoothed objective: h(x,y) + q^T y (simplified as quadratic)
        objective = cp.Minimize(0.5 * cp.quad_form(y_var, Q_lower_np) + 
                              (c_lower_np + P_np.T @ x_np + q_np) @ y_var)
        
        # Constraints: Ay + Bx ≤ b (simplified as Ay ≤ b for [S]SIGD)
        constraints = [A_np @ y_var <= b_np]
        
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, eps_abs=self.epsilon**2)
        
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"CVXPy solve failed: status={problem.status}")
        
        y_opt = torch.tensor(y_var.value, device=self.device, dtype=self.problem.dtype)
        
        info = {
            'status': problem.status,
            'value': problem.value,
            'perturbation_norm': torch.norm(q).item()
        }
        
        return y_opt, info
        
    def compute_stochastic_implicit_gradient(self, x: torch.Tensor, y_star: torch.Tensor, 
                                           xi: torch.Tensor) -> torch.Tensor:
        """
        Compute the stochastic implicit gradient: ∇̃F(x;ξ) = ∇x f̃(x,y*(x);ξ) + [∇y*(x)]^T ∇y f̃(x,y*(x);ξ)
        
        This is the core gradient computation in [S]SIGD
        """
        # Make x differentiable
        x_grad = x.clone().requires_grad_(True)
        
        # Compute upper-level objective with stochastic sample
        f_val = self.problem.upper_objective(x_grad, y_star, add_noise=True)
        
        # Compute gradient w.r.t. x
        grad_x = torch.autograd.grad(f_val, x_grad, create_graph=False)[0]
        
        # For the second term [∇y*(x)]^T ∇y f̃(x,y*(x);ξ), we need to compute ∇y*(x)
        # Approximate ∇y*(x) using finite differences
        eps_fd = 1e-6
        y_plus = torch.zeros_like(y_star)
        y_minus = torch.zeros_like(y_star)
        
        for i in range(self.problem.dim):
            # Perturb x in the i-th direction
            x_plus = x.clone()
            x_plus[i] += eps_fd
            x_minus = x.clone()
            x_minus[i] -= eps_fd
            
            # Solve smoothed LL problems
            y_plus_i, _ = self.solve_smoothed_lower_level(x_plus)
            y_minus_i, _ = self.solve_smoothed_lower_level(x_minus)
            
            y_plus[i] = y_plus_i[i]
            y_minus[i] = y_minus_i[i]
        
        # Approximate ∇y*(x) using finite differences
        dy_dx = (y_plus - y_minus) / (2 * eps_fd)
        
        # Compute ∇y f̃(x,y*(x);ξ)
        y_grad = y_star.clone().requires_grad_(True)
        f_val_y = self.problem.upper_objective(x, y_grad, add_noise=True)
        grad_y = torch.autograd.grad(f_val_y, y_grad, create_graph=False)[0]
        
        # Combine the gradients: ∇x f̃ + [∇y*(x)]^T ∇y f̃
        implicit_gradient = grad_x + dy_dx * grad_y
        
        return implicit_gradient
        
    def stochastic_implicit_gradient_oracle(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stochastic implicit gradient oracle ([S]SIGD core mechanism)
        
        This implements the stochastic implicit gradient approach:
        1. Single perturbation: q for differentiability (fixed throughout training)
        2. Stochastic sampling: ξ for stochasticity
        """
        # Use fixed perturbation
        q = self.fixed_q if self.fixed_perturbation else self.sample_perturbation()
        
        # Sample stochastic sample ξ
        xi = torch.randn(1, device=self.device, dtype=self.problem.dtype)
        
        # Solve smoothed lower-level problem
        y_star, info = self.solve_smoothed_lower_level(x)
        
        # Compute stochastic implicit gradient
        gradient = self.compute_stochastic_implicit_gradient(x, y_star, xi)
        
        
        return gradient
        
    def compute_moreau_envelope_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the Moreau envelope: ∇H_λ(x)
        
        This is the convergence measure used in [S]SIGD
        """
        # Moreau envelope gradient: ∇H_λ(x) = (x - prox_λH(x)) / λ
        # For weakly convex functions, this provides a measure of stationarity
        
        # Compute proximal point (simplified)
        lambda_val = self.lambda_moreau
        prox_point = x - lambda_val * self.stochastic_implicit_gradient_oracle(x)
        
        # Project onto feasible set
        prox_point = self.project_to_feasible_set(prox_point)
        
        # Moreau envelope gradient
        moreau_gradient = (x - prox_point) / lambda_val
        
        return moreau_gradient
        
    def project_to_feasible_set(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project x onto the feasible set X
        
        This implements the projection step in [S]SIGD
        """
        # Simple projection onto ball (can be customized for specific constraints)
        norm_x = torch.norm(x)
        if norm_x > self.projection_radius:
            x = x / norm_x * self.projection_radius
        
        return x
        
    def optimize(self, max_iterations: int = 1000, target_gap: float = 1e-3, run_until_convergence: bool = False) -> Dict:
        """
        Main [S]SIGD optimization loop
        
        Implements the stochastic implicit gradient descent with Moreau envelope convergence
        """
        
        # Initialize with consistent seed-based random value
        torch.manual_seed(42)  # Use same seed for consistent initialization
        x = torch.randn(self.problem.dim, device=self.device, dtype=self.problem.dtype) * 0.1
        x.requires_grad_(True)
        

        start_time = time.time()
        
        # Main optimization loop
        for iteration in range(max_iterations):
            iter_start_time = time.time()
            
            # Stochastic implicit gradient oracle
            g_t = self.stochastic_implicit_gradient_oracle(x)
            
            # Update x using gradient descent with projection
            x_new = x - self.eta * g_t
            x = self.project_to_feasible_set(x_new)
            
            # Compute current objective and gap using unified gap metric
            current_obj = self.problem.upper_objective(x, torch.zeros_like(x))
            current_gap = self.problem.compute_gap(x)  # Use same gap as F2CSA
            
            # Store history for analysis
            self.history['iterations'].append(iteration)
            self.history['objectives'].append(current_obj.item())
            self.history['gradients'].append(torch.norm(g_t).item())
            self.history['gaps'].append(current_gap)
            
            iter_time = time.time() - iter_start_time
            self.history['times'].append(iter_time)
            
            # Progress monitoring
            if iteration % 50 == 0:
                pass  # Debug output removed
            

        
        total_time = time.time() - start_time
        
        # Final analysis (use unified gap metric)
        final_gap = float(self.problem.compute_gap(x)) if len(self.history['gaps']) == 0 else float(self.history['gaps'][-1])

        results = {
            'final_x': x.detach().cpu().numpy(),
            'final_gap': final_gap,
            'total_iterations': len(self.history['iterations']),
            'total_time': total_time,
            'target_achieved': final_gap < target_gap,
            'history': self.history
        }
        
        return results
        
    def analyze_weak_convexity(self, x: torch.Tensor) -> Dict:
        """
        Analyze weak convexity assumption
        
        This helps understand:
        1. Whether the implicit function satisfies weak convexity
        2. Impact of weak convexity parameter ρ
        3. Moreau envelope behavior
        """
        
        # Test weak convexity assumption: F(z) ≥ F(x) + ⟨∇F(x), z-x⟩ - (ρ/2)||z-x||²
        n_samples = 20
        weak_convexity_violations = []
        moreau_norms = []
        
        for i in range(n_samples):
            # Sample random direction
            direction = torch.randn_like(x) * 0.1
            z = x + direction
            
            # Compute F(x) and F(z)
            y_star_x, _ = self.solve_smoothed_lower_level(x)
            y_star_z, _ = self.solve_smoothed_lower_level(z)
            
            F_x = self.problem.upper_objective(x, y_star_x)
            F_z = self.problem.upper_objective(z, y_star_z)
            
            # Compute ∇F(x)
            grad_F = self.stochastic_implicit_gradient_oracle(x)
            
            # Check weak convexity: F(z) ≥ F(x) + ⟨∇F(x), z-x⟩ - (ρ/2)||z-x||²
            linear_term = torch.dot(grad_F, z - x)
            quadratic_term = (self.rho / 2) * torch.norm(z - x) ** 2
            
            weak_convexity_check = F_z - (F_x + linear_term - quadratic_term)
            weak_convexity_violations.append(weak_convexity_check.item())
            
            # Compute Moreau envelope norm
            moreau_grad = self.compute_moreau_envelope_gradient(x)
            moreau_norms.append(torch.norm(moreau_grad).item())
        
        analysis = {
            'weak_convexity_violations': weak_convexity_violations,
            'moreau_norms': moreau_norms,
            'avg_violation': np.mean(weak_convexity_violations),
            'max_violation': np.max(weak_convexity_violations),
            'avg_moreau_norm': np.mean(moreau_norms)
        }
        
        
        return analysis
        
    def analyze_perturbation_impact(self, q_std_values: list) -> Dict:
        """
        Analyze impact of perturbation on differentiability
        
        This helps understand:
        1. How perturbation affects differentiability
        2. Trade-off between perturbation size and approximation error
        3. Optimal perturbation parameters
        """
        
        results = {}
        x_test = torch.randn(self.problem.dim, device=self.device, dtype=self.problem.dtype) * 0.1
        
        for q_std in q_std_values:
            
            # Update perturbation parameter
            self.q_std = q_std
            if self.fixed_perturbation:
                self.fixed_q = self.sample_perturbation()
            
            # Test stochastic implicit gradient oracle
            try:
                grad_estimate = self.stochastic_implicit_gradient_oracle(x_test)
                true_gap = self.problem.compute_gap(x_test)
                moreau_grad = self.compute_moreau_envelope_gradient(x_test)
                
                results[q_std] = {
                    'gradient_norm': torch.norm(grad_estimate).item(),
                    'true_gap': true_gap,
                    'moreau_norm': torch.norm(moreau_grad).item(),
                    'perturbation_scale': q_std
                }
                
                
            except Exception as e:
                results[q_std] = {'error': str(e)}
        
        return results
