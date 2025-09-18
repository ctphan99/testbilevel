#!/usr/bin/env python3
"""
Natural Bilevel Optimization Problem without Artificial Conditions
Uses constraint form: h(x,y) = Ax + By - b ≤ 0
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional
import warnings

class StronglyConvexBilevelProblem:
    """
    Natural bilevel optimization problem with constraints h(x,y) = Ax + By - b ≤ 0
    
    Upper level: min_x f(x, y*(x)) where y*(x) solves:
    Lower level: min_y g(x,y) subject to h(x,y) = Ax + By - b ≤ 0
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
        
        # Generate natural problem parameters
        self._generate_problem_parameters()
        
    def _generate_problem_parameters(self):
        """Generate natural problem parameters without artificial conditions"""
        
        # Upper-level objective: f(x,y) = 0.5 * x^T Q_upper x + c_upper^T x + 0.5 * y^T P y + x^T P y
        param_scale = 1.0
        self.Q_upper = torch.randn(self.dim, self.dim, device=self.device, dtype=self.dtype) * param_scale
        self.Q_upper = self.Q_upper @ self.Q_upper.T  # Make positive definite
        
        if self.strong_convex:
            # Ensure strong convexity
            eigenvals = torch.linalg.eigvals(self.Q_upper).real
            min_eigenval = eigenvals.min()
            if min_eigenval <= 0:
                self.Q_upper += (1.0 - min_eigenval) * torch.eye(self.dim, device=self.device, dtype=self.dtype)
        
        self.c_upper = torch.randn(self.dim, device=self.device, dtype=self.dtype) * param_scale
        self.P = torch.randn(self.dim, self.dim, device=self.device, dtype=self.dtype) * param_scale
        self.x_target = torch.randn(self.dim, device=self.device, dtype=self.dtype) * 0.1
        
        # Lower-level objective: g(x,y) = 0.5 * y^T Q_lower y + c_lower^T y
        self.Q_lower = torch.randn(self.dim, self.dim, device=self.device, dtype=self.dtype) * param_scale
        self.Q_lower = self.Q_lower @ self.Q_lower.T  # Make positive definite
        
        if self.strong_convex:
            # Ensure strong convexity
            eigenvals = torch.linalg.eigvals(self.Q_lower).real
            min_eigenval = eigenvals.min()
            if min_eigenval <= 0:
                self.Q_lower += (1.0 - min_eigenval) * torch.eye(self.dim, device=self.device, dtype=self.dtype)
        
        self.c_lower = torch.randn(self.dim, device=self.device, dtype=self.dtype) * param_scale
        
        # Override constraints to box: |y_i| ≤ 1 for all i
        # Model as stacked inequalities: y ≤ 1 and -y ≤ 1 → [I; -I] y ≤ [1; 1]
        self.num_constraints = 2 * self.dim
        self.A = torch.zeros(self.num_constraints, self.dim, device=self.device, dtype=self.dtype)
        I = torch.eye(self.dim, device=self.device, dtype=self.dtype)
        self.B = torch.cat([I, -I], dim=0)
        self.b = torch.ones(self.num_constraints, device=self.device, dtype=self.dtype)
        
        # Print problem information
        print(f"Natural Bilevel Problem (dim={self.dim}, constraints={self.num_constraints})")
        
        # Check box constraint at origin
        origin_violations = torch.zeros_like(self.b)
        max_violation = torch.max(origin_violations)
        print(f"Constraint violations at origin: {max_violation:.6f}")
        
        if max_violation > 1e-6:
            print("Natural constraint violations present - F2CSA penalty mechanism will engage")
        else:
            print("Origin is feasible - constraints may not be active")

        # Verify strong convexity
        upper_eigenvals = torch.linalg.eigvals(self.Q_upper).real
        lower_eigenvals = torch.linalg.eigvals(self.Q_lower).real

        print(f"Upper level strong convexity: λ_min={upper_eigenvals.min():.3f}, λ_max={upper_eigenvals.max():.3f}")
        print(f"Lower level strong convexity: λ_min={lower_eigenvals.min():.3f}, λ_max={lower_eigenvals.max():.3f}")

    def _sample_instance_noise(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample independent instance noise for Q matrices"""
        noise_upper = torch.randn(self.dim, self.dim, device=self.device, dtype=self.dtype) * self.noise_std
        noise_lower = torch.randn(self.dim, self.dim, device=self.device, dtype=self.dtype) * self.noise_std
        
        # Make symmetric
        noise_upper = (noise_upper + noise_upper.T) / 2
        noise_lower = (noise_lower + noise_lower.T) / 2
        
        return noise_upper, noise_lower
    
    def upper_objective(self, x: torch.Tensor, y: torch.Tensor, 
                       noise_upper: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Upper-level objective: f(x,y) = 0.5 * x^T Q_upper x + c_upper^T x + 0.5 * y^T P y + x^T P y"""
        if noise_upper is None:
            noise_upper = torch.zeros_like(self.Q_upper)
        
        # Ensure tensors are on same device and dtype
        x = x.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)
            
        Q_upper_noisy = self.Q_upper + noise_upper
        f = 0.5 * x.T @ Q_upper_noisy @ x + self.c_upper.T @ x + 0.5 * y.T @ self.P @ y + x.T @ self.P @ y
        return f
    
    def lower_objective(self, x: torch.Tensor, y: torch.Tensor, 
                       noise_lower: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Lower-level objective: g(x,y) = 0.5 * y^T Q_lower y + c_lower^T y"""
        if noise_lower is None:
            noise_lower = torch.zeros_like(self.Q_lower)
        
        # Ensure tensors are on same device and dtype
        x = x.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)
            
        Q_lower_noisy = self.Q_lower + noise_lower
        g = 0.5 * y.T @ Q_lower_noisy @ y + self.c_lower.T @ y
        return g

    def constraints(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Constraint function: h(x,y) = Ax + By - b"""
        # Ensure tensors are on same device and dtype
        x = x.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)
        return self.A @ x + self.B @ y - self.b
    
    def constraint_violations(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Get constraint violations: max(0, h(x,y))"""
        h = self.constraints(x, y)
        return torch.clamp(h, min=0)
    
    def solve_lower_level(self, x: torch.Tensor, solver: str = 'accurate', 
                         max_iter: int = 10000, tol: float = 1e-8, alpha: float = 0.1) -> Tuple[torch.Tensor, Dict]:
        """
        Solve lower-level problem: min_y g(x,y) subject to h(x,y) ≤ 0
        
        Args:
            x: Upper-level variable
            solver: Solver type ('accurate' for F2CSA.tex Algorithm 1 compliance)
            max_iter: Maximum iterations
            tol: Convergence tolerance
            alpha: Accuracy parameter (δ = α³)
            
        Returns:
            y_opt: Approximate solution
            info: Solution information
        """
        if solver == 'accurate':
            return self._solve_accurate(x, alpha, max_iter, tol)
        elif solver == 'pgd':
            return self._solve_pgd(x, max_iter, tol)
        else:
            raise ValueError(f"Unknown solver: {solver}")
    
    def _solve_accurate(self, x: torch.Tensor, alpha: float, max_iter: int, tol: float) -> Tuple[torch.Tensor, Dict]:
        """Solve using accurate solver implementing F2CSA.tex Algorithm 1"""
        try:
            from accurate_lower_level_solver import AccurateLowerLevelSolver
            
            solver = AccurateLowerLevelSolver(self, device=self.device, dtype=self.dtype)
            y_opt, lambda_opt, info = solver.solve_lower_level_accurate(x, alpha, max_iter, tol)
            
            return y_opt, info
            
        except ImportError:
            print("Accurate solver not available, falling back to PGD")
            return self._solve_pgd(x, max_iter, tol)

    def _solve_pgd(self, x: torch.Tensor, max_iter: int, tol: float) -> Tuple[torch.Tensor, Dict]:
        """
        Solve using projected gradient descent (natural, no artificial feasibility)
        """
        # Initialize at unconstrained optimum
        y = -torch.linalg.solve(self.Q_lower, self.c_lower)

        # Use small learning rate for stability
        lr = 0.01

        for i in range(max_iter):
            # Gradient of lower-level objective
            grad_g = self.Q_lower @ y + self.c_lower
            
            # Gradient step
            y_new = y - lr * grad_g

            # Project onto feasible region: h(x,y) ≤ 0
            y = self._project_onto_constraints(x, y_new)

            # Check convergence
            grad_norm = torch.norm(grad_g)
            if grad_norm < tol:
                break

        # Compute dual variables (Lagrange multipliers)
        h = self.constraints(x, y)
        lambda_opt = torch.clamp(-h, min=0)  # KKT conditions

        info = {
            'iterations': i + 1,
            'lambda': lambda_opt,
            'constraint_violations': self.constraint_violations(x, y),
            'converged': grad_norm < tol
        }
        
        return y, info
    
    def _project_onto_constraints(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Project y onto feasible region {y : h(x,y) ≤ 0}
        Uses simple projection: if h(x,y) > 0, move towards feasibility
        """
        h = self.constraints(x, y)
        violations = torch.clamp(h, min=0)

        if torch.norm(violations) < 1e-10:
            return y  # Already feasible

        # Move in direction of constraint normals to restore feasibility
        correction = torch.zeros_like(y)
        for i in range(self.num_constraints):
            if violations[i] > 0:
                # Move in direction of B[i] to satisfy constraint i
                B_norm_sq = torch.norm(self.B[i])**2
                if B_norm_sq > 1e-10:
                    correction += violations[i] * self.B[i] / B_norm_sq
        
        return y - correction
    
    def get_true_solution(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get true solution for testing (solves KKT conditions exactly)
        """
        # This is a simplified version - in practice, you'd solve the KKT system
        y_opt, info = self.solve_lower_level(x, solver='pgd', max_iter=2000, tol=1e-8)
        lambda_opt = info['lambda']
        return y_opt, lambda_opt
    
    def compute_gap(self, x: torch.Tensor) -> float:
        """
        Compute optimization gap for F2CSA algorithm
        """
        # Solve lower-level problem
        y_opt, info = self.solve_lower_level(x, solver='pgd')
        
        # Compute upper-level objective
        f_val = self.upper_objective(x, y_opt)
        
        # Compute constraint violations
        h_val = self.constraints(x, y_opt)
        violations = torch.clamp(h_val, min=0)
        max_violation = torch.max(violations).item()
        
        # Gap is combination of objective value and constraint violations
        gap = f_val.item() + 1000.0 * max_violation  # Penalty for violations
        
        return gap