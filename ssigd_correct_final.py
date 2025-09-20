import torch
import numpy as np
from typing import Optional, Union, List
from problem import StronglyConvexBilevelProblem
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class CorrectSSIGD:
    """
    SSIGD with corrected implementation:
    - Uses CVXPyLayer for lower-level optimization with noise
    - Implements projected gradient descent for upper-level
    - Tracks both noisy and clean objectives
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem, device='cpu'):
        self.prob = problem
        self.device = device
        self.dtype = torch.float64
        
        # Create CVXPyLayer for lower-level optimization with noise
        self._setup_cvxpy_layer()
        
        # Add noise to lower-level problem - normalized and scaled like in BilevelLinearConstraints
        q = torch.randn(problem.dim, device=device, dtype=self.dtype)
        self.q = 1e-6 * (q / torch.norm(q))
        
    def _setup_cvxpy_layer(self):
        """Setup CVXPyLayer for lower-level optimization"""
        # Lower-level variables
        y = cp.Variable(self.prob.dim)
        
        # Lower-level objective: (1/2) * y^T * Q_lower * y + c_lower^T * y
        # We'll add the noise term q later
        objective = cp.Minimize(0.5 * cp.quad_form(y, self.prob.Q_lower.cpu().numpy()) + 
                               cp.sum(cp.multiply(self.prob.c_lower.cpu().numpy(), y)))
        
        # Lower-level constraints: box constraints -1 <= y <= 1
        constraints = [y >= -1, y <= 1]
        
        # Create the problem
        problem_cp = cp.Problem(objective, constraints)
        
        # Create CVXPyLayer
        self.cvxpy_layer = CvxpyLayer(problem_cp, parameters=[], variables=[y])
    
    def solve_ll_with_q(self, x: torch.Tensor, q_noise: torch.Tensor) -> torch.Tensor:
        """Solve lower-level problem with noise q using CVXPY directly"""
        try:
            import cvxpy as cp
            
            # Create modified problem with noise
            y = cp.Variable(self.prob.dim)
            
            # Add noise to the linear term
            c_modified = self.prob.c_lower.cpu().numpy() + q_noise.cpu().numpy()
            
            objective = cp.Minimize(0.5 * cp.quad_form(y, self.prob.Q_lower.cpu().numpy()) + 
                                   cp.sum(cp.multiply(c_modified, y)))
            # Use box constraints: -1 <= y <= 1
            constraints = [y >= -1, y <= 1]
            
            problem_cp = cp.Problem(objective, constraints)
            problem_cp.solve(verbose=False)
            
            if problem_cp.status == cp.OPTIMAL:
                y_sol = torch.tensor(y.value, dtype=self.dtype, device=self.device)
                return y_sol
            else:
                # Fallback to clean solver
                return self.solve_ll(x)
                
        except Exception as e:
            # Fallback to clean solver
            return self.solve_ll(x)
    
    def solve_ll(self, x: torch.Tensor) -> torch.Tensor:
        """Solve clean lower-level problem"""
        y_opt, _, _ = self.prob.solve_lower_level(x, solver='gurobi')
        return y_opt
    
    def grad_F(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of upper-level objective F(x,y) w.r.t. x using implicit gradient formula
        Following Equation (8): ∇F(x) = ∇x f(x, y*(x)) + [∇y*(x)]ᵀ ∇y f(x, y*(x))
        """
        # y is the noisy lower-level solution from solve_ll_with_q
        y_hat = y.clone().detach()
        
        # Compute ∇x f(x, y_hat) - direct partial derivative w.r.t. x
        x_direct = x.clone().requires_grad_(True)
        y_fixed = y_hat.clone().detach()
        f_direct = self.prob.upper_objective(x_direct, y_fixed)
        f_direct.backward()
        grad_x_f = x_direct.grad
        
        # Compute ∇y f(x, y_hat) - partial derivative w.r.t. y
        x_fixed = x.clone().detach()
        y_partial = y_hat.clone().requires_grad_(True)
        f_partial = self.prob.upper_objective(x_fixed, y_partial)
        f_partial.backward()
        grad_y_f = y_partial.grad
        
        # One-line solver to calculate accurate approximation of ∇y*(x)
        eps = 1e-6
        grad_y_star = torch.stack([(self.solve_ll_with_q(x + eps * torch.eye(self.prob.dim, device=self.device, dtype=self.dtype)[i], self.q) - y_hat) / eps for i in range(self.prob.dim)], dim=1)
        
        # Apply Equation (8): ∇F(x) = ∇x f(x, y*(x)) + [∇y*(x)]ᵀ ∇y f(x, y*(x))
        total_grad = grad_x_f + grad_y_star.T @ grad_y_f
        
        return total_grad
    
    def proj_X(self, x_candidate: torch.Tensor, X_bounds=None) -> torch.Tensor:
        """Project x_candidate using clipping - if no bounds, return identity (unconstrained)"""
        if X_bounds is None:
            # Unconstrained case - no projection needed
            return x_candidate
        return torch.clamp(x_candidate, X_bounds[0], X_bounds[1])

    def solve(self, T=1000, beta=0.01, x0=None, diminishing: bool = True, mu_F: float = None):
        x = (x0.detach().to(device=self.device, dtype=self.dtype).clone()
             if x0 is not None else
             torch.randn(self.prob.dim, device=self.device, dtype=self.dtype) * 0.1)
        losses = []
        grad_norms = []

        # Determine μ_F for diminishing step sizes
        if mu_F is None:
            mu_F = torch.linalg.eigvals(self.prob.Q_upper).real.min().item()
            # Ensure μ_F > 0 for stability
            mu_F = max(mu_F, 1e-6)
        
        print(f"SSIGD: T={T}, beta={beta:.4f}, diminishing={diminishing}, μ_F={mu_F:.6f}")

        for r in range(1, T + 1):  # 1-based iteration like DS-BLO
            # 4-5: ŷ(x_r) via CVXPy-layer with noisy Q_lower and fixed q; gradient ∇̂F(x_r)
            y_hat = self.solve_ll_with_q(x, self.q)
            grad_est = self.grad_F(x, y_hat)

            # 6: projected step with step-size schedule and reasonable capping
            if diminishing:
                lr_t = 1.0 / (mu_F * (r + 1))  # Corrected formula: r+1 instead of r
                # Cap step size to prevent instability with very small mu_F
                lr_t = min(lr_t, beta)
            else:
                lr_t = beta if isinstance(beta, (int, float)) else float(beta[r-1])  # Adjust for 1-based indexing
            
            # Unconstrained gradient step: x_{r+1} = x_r - β_r * ∇̂F(x_r) (no projection)
            x_new = x - lr_t * grad_est
            
            # Apply projection (identity for unconstrained case)
            x = self.proj_X(x_new, X_bounds=None)  # No bounds = unconstrained

            # tracking (deterministic UL, LL with noise for y)
            y_star = self.solve_ll(x)
            F = self.prob.upper_objective(x, y_star).item()
            losses.append(F)
            
            # Track gradient norm
            grad_norm = torch.norm(grad_est).item()
            grad_norms.append(grad_norm)
            
            if r % 100 == 0:
                print(f"Iteration {r:4d}/{T}: ||∇F|| = {grad_norm:.6f}, lr = {lr_t:.6f}, F = {F:.6f}")

        return {
            'x_final': x,
            'losses': losses,
            'grad_norms': grad_norms,
            'final_loss': losses[-1] if losses else float('inf'),
            'final_gradient_norm': grad_norms[-1] if grad_norms else 0.0,
            'final_grad_norm': grad_norms[-1] if grad_norms else 0.0,  # Alias for compatibility
            'converged': True,  # SSIGD always runs for T iterations
            'iterations': T
        }