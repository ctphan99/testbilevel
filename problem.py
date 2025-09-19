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
        
        # Box constraints: |y_i| ≤ 1 for all i (y ≤ 1 and -y ≤ 1)
        self.num_constraints = 2 * self.dim
        
        # Print problem information
        print(f"Natural Bilevel Problem (dim={self.dim}, constraints={self.num_constraints})")
        
        # Check box constraint feasibility at origin
        x0 = torch.zeros(self.dim, device=self.device, dtype=self.dtype)
        y0 = torch.zeros(self.dim, device=self.device, dtype=self.dtype)
        h0 = self.constraints(x0, y0)
        max_violation = torch.max(torch.clamp(h0, min=0)).item()
        print(f"Constraint violations at origin: {max_violation:.6f}")
        print("Origin is feasible - constraints may not be active" if max_violation <= 1e-6 else "Natural constraint violations present - F2CSA penalty mechanism will engage")

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
        """Box constraints: y ≤ 1 and -y ≤ 1"""
        # Ensure tensors are on same device and dtype
        x = x.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)
        # Explicit box constraints: y ≤ 1 and -y ≤ 1 → h(y) = [y - 1; -y - 1] ≤ 0
        return torch.cat([y - 1.0, -y - 1.0], dim=0)
    
    def constraint_violations(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Get constraint violations: max(0, h(x,y))"""
        h = self.constraints(x, y)
        return torch.clamp(h, min=0)
    
    def project_X(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projection of UL variable onto feasible set X. For current setup, X is unconstrained,
        so this is identity. Override if UL constraints are added.
        """
        return x.to(device=self.device, dtype=self.dtype)
    
    def solve_lower_level(self, x: torch.Tensor, solver: str = 'cvxpy', 
                         max_iter: int = 10000, tol: float = 1e-8, alpha: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Solve lower-level problem: min_y g(x,y) subject to h(x,y) ≤ 0
        
        Args:
            x: Upper-level variable
            solver: Solver type ('cvxpy' for accurate, 'pgd' for gradient descent)
            max_iter: Maximum iterations
            tol: Convergence tolerance
            alpha: Accuracy parameter (δ = α³)
            
        Returns:
            y_opt: Optimal solution
            lambda_opt: Lagrange multipliers
            info: Solution information
        """
        if solver == 'cvxpy':
            return self._solve_cvxpy(x, alpha)
        elif solver == 'pgd':
            return self._solve_pgd(x, max_iter, tol)
        elif solver == 'accurate':
            return self._solve_accurate(x, alpha, max_iter, tol)
        else:
            raise ValueError(f"Unknown solver: {solver}")
    
    def _solve_cvxpy(self, x: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Solve using CVXPY with SCS solver - centralized for all algorithms
        Uses NOISY Q_lower for stochastic setting
        """
        try:
            import cvxpy as cp
            
            # CRITICAL: Use noisy Q_lower for stochastic setting
            _, noise_lower = self._sample_instance_noise()
            Q_lower_noisy = self.Q_lower + noise_lower
            
            # Ensure Q_lower_noisy remains positive definite
            min_eigenval = torch.linalg.eigvals(Q_lower_noisy).real.min()
            if min_eigenval <= 1e-8:
                Q_lower_noisy = Q_lower_noisy + (1e-6 - min_eigenval) * torch.eye(Q_lower_noisy.shape[0], device=Q_lower_noisy.device, dtype=Q_lower_noisy.dtype)
            
            Q_lower_np = Q_lower_noisy.detach().cpu().numpy()
            c_lower_np = self.c_lower.detach().cpu().numpy()
            
            # Create variables
            y = cp.Variable(self.dim)
            
            # Objective: min_y 0.5 * y^T Q_lower_noisy y + c_lower^T y
            objective = cp.Minimize(0.5 * cp.quad_form(y, Q_lower_np) + c_lower_np.T @ y)
            
            # Explicit box constraints
            constraints = [y <= 1, -y <= 1]
            
            # Solve using SCS solver
            problem_cvx = cp.Problem(objective, constraints)
            problem_cvx.solve(verbose=False, solver=cp.SCS, warm_start=True)
            
            if problem_cvx.status == cp.OPTIMAL:
                y_opt = torch.tensor(y.value, dtype=self.dtype, device=self.device)
                
                # Extract dual variables (stack both box sides)
                import numpy as _np
                lambda_np = _np.concatenate([constraints[0].dual_value, constraints[1].dual_value])
                lambda_opt = torch.tensor(lambda_np, dtype=self.dtype, device=self.device)
                
                # Compute constraint violations
                h_val = self.constraints(x, y_opt)
                violations = torch.clamp(h_val, min=0)
                max_violation = torch.max(violations).item()
                
                info = {
                    'status': 'optimal',
                    'iterations': 0,  # CVXPY doesn't report iterations
                    'lambda': lambda_opt,
                    'constraint_violations': violations,
                    'converged': True,
                    'max_violation': max_violation,
                    'solver': 'CVXPY-SCS'
                }
                
                return y_opt, lambda_opt, info
            else:
                raise ValueError(f"CVXPY solve failed with status: {problem_cvx.status}")
                
        except ImportError:
            print("CVXPY not available, falling back to PGD")
            return self._solve_pgd(x, 1000, 1e-6)
    
    def _solve_accurate(self, x: torch.Tensor, alpha: float, max_iter: int, tol: float) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Solve using accurate solver implementing F2CSA.tex Algorithm 1"""
        try:
            from accurate_lower_level_solver import AccurateLowerLevelSolver
            
            solver = AccurateLowerLevelSolver(self, device=self.device, dtype=self.dtype)
            y_opt, lambda_opt, info = solver.solve_lower_level_accurate(x, alpha, max_iter, tol)
            
            return y_opt, lambda_opt, info
            
        except ImportError:
            print("Accurate solver not available, falling back to PGD")
            return self._solve_pgd(x, max_iter, tol)

    def _solve_pgd(self, x: torch.Tensor, max_iter: int, tol: float) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Solve using projected gradient descent (natural, no artificial feasibility)
        Uses NOISY Q_lower for stochastic setting
        """
        # CRITICAL: Use noisy Q_lower for stochastic setting
        _, noise_lower = self._sample_instance_noise()
        Q_lower_noisy = self.Q_lower + noise_lower
        
        # Ensure Q_lower_noisy remains positive definite
        min_eigenval = torch.linalg.eigvals(Q_lower_noisy).real.min()
        if min_eigenval <= 1e-8:
            Q_lower_noisy = Q_lower_noisy + (1e-6 - min_eigenval) * torch.eye(Q_lower_noisy.shape[0], device=Q_lower_noisy.device, dtype=Q_lower_noisy.dtype)
        
        # Initialize at unconstrained optimum
        y = -torch.linalg.solve(Q_lower_noisy, self.c_lower)

        # Use small learning rate for stability
        lr = 0.01

        for i in range(max_iter):
            # Gradient of lower-level objective with noisy Q
            grad_g = Q_lower_noisy @ y + self.c_lower
            
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
            'converged': grad_norm < tol,
            'solver': 'PGD'
        }
        
        return y, lambda_opt, info
    
 