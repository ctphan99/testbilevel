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
        
        # Add noise to lower-level problem
        self.q = torch.randn(problem.dim, device=device, dtype=self.dtype) * 0.1
        
    def _setup_cvxpy_layer(self):
        """Setup CVXPyLayer for lower-level optimization"""
        # Lower-level variables
        y = cp.Variable(self.prob.dim)
        
        # Lower-level objective: (1/2) * y^T * Q_lower * y + c_lower^T * y
        # We'll add the noise term q later
        objective = cp.Minimize(0.5 * cp.quad_form(y, self.prob.Q_lower.cpu().numpy()) + 
                               cp.sum(cp.multiply(self.prob.c_lower.cpu().numpy(), y)))
        
        # Lower-level constraints: y >= 0
        constraints = [y >= 0]
        
        # Create the problem
        problem_cp = cp.Problem(objective, constraints)
        
        # Create CVXPyLayer
        self.cvxpy_layer = CvxpyLayer(problem_cp, parameters=[], variables=[y])
    
    def solve_ll_with_q(self, x: torch.Tensor, q_noise: torch.Tensor) -> torch.Tensor:
        """Solve lower-level problem with noise q using CVXPyLayer"""
        # Create modified problem with noise
        y = cp.Variable(self.prob.dim)
        
        # Add noise to the linear term
        c_modified = self.prob.c_lower.cpu().numpy() + q_noise.cpu().numpy()
        
        objective = cp.Minimize(0.5 * cp.quad_form(y, self.prob.Q_lower.cpu().numpy()) + 
                               cp.sum(cp.multiply(c_modified, y)))
        constraints = [y >= 0]
        
        problem_cp = cp.Problem(objective, constraints)
        cvxpy_layer = CvxpyLayer(problem_cp, parameters=[], variables=[y])
        
        # Solve
        y_sol, = cvxpy_layer()
        return y_sol.to(device=self.device, dtype=self.dtype)
    
    def solve_ll(self, x: torch.Tensor) -> torch.Tensor:
        """Solve clean lower-level problem"""
        return self.prob.solve_lower_level(x, solver='gurobi')
    
    def grad_F(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute gradient of upper-level objective F(x,y) w.r.t. x"""
        # F(x,y) = (1/2) * x^T * Q_upper * x + (1/2) * y^T * Q_upper * y
        # ∇_x F(x,y) = Q_upper * x
        return torch.mv(self.prob.Q_upper, x)
    
    def proj_X(self, x: torch.Tensor) -> torch.Tensor:
        # If problem defines a projection, use it; otherwise identity
        if hasattr(self.prob, 'project_X') and callable(getattr(self.prob, 'project_X')):
            return self.prob.project_X(x)
        return x
    
    def proj_X_gurobi(self, x_candidate: torch.Tensor, X_bounds=(-1, 1)) -> torch.Tensor:
        """Project x_candidate onto box constraints using Gurobi for accurate projection"""
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            model = gp.Model("projection")
            model.setParam('OutputFlag', 0)
            
            # Variables
            x_proj = model.addVars(len(x_candidate), lb=X_bounds[0], ub=X_bounds[1], name="x")
            
            # Objective: minimize ||x - x_candidate||²
            obj = gp.QuadExpr()
            for i in range(len(x_candidate)):
                obj += (x_proj[i] - x_candidate[i].item())**2
            
            model.setObjective(obj, GRB.MINIMIZE)
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                return torch.tensor([x_proj[i].x for i in range(len(x_candidate))], 
                                  dtype=x_candidate.dtype, device=x_candidate.device)
            else:
                # Fallback to simple clipping
                return torch.clamp(x_candidate, X_bounds[0], X_bounds[1])
                
        except ImportError:
            # Fallback to simple clipping
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
        
        print(f"SSIGD (projected): T={T}, beta={beta:.4f}, diminishing={diminishing}, μ_F={mu_F:.6f}")

        for r in range(1, T + 1):  # 1-based iteration like DS-BLO
            # 4-5: ŷ(x_r) via CVXPy-layer with noisy Q_lower and fixed q; gradient ∇̂F(x_r)
            y_hat = self.solve_ll_with_q(x, self.q)
            grad_est = self.grad_F(x, y_hat)

            # 6: projected step with step-size schedule and capping
            if diminishing:
                lr_t = 1.0 / (mu_F * r)  # r is now 1-based, so use r directly
                # Cap step size to beta (max step size)
                lr_t = min(lr_t, beta)
            else:
                lr_t = beta if isinstance(beta, (int, float)) else float(beta[r-1])  # Adjust for 1-based indexing
            
            # Projected gradient step: x_{r+1} = proj_X(x_r - β_r * ∇̂F(x_r))
            x = self.proj_X_gurobi(x - lr_t * grad_est)

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
            'final_gradient_norm': grad_norms[-1] if grad_norms else 0.0
        }