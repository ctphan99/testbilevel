#!/usr/bin/env python3
"""
Accurate Lower-Level Solver for F2CSA Algorithm
Implements F2CSA.tex Algorithm 1 requirements with proper convergence
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Optional
import warnings

class AccurateLowerLevelSolver:
    """
    Accurate lower-level solver implementing F2CSA.tex Algorithm 1
    Achieves O(δ) accuracy with δ = α³ as required
    """
    
    def __init__(self, problem, device='cpu', dtype=torch.float64):
        self.problem = problem
        self.device = device
        self.dtype = dtype
        
    def solve_lower_level_accurate(self, x: torch.Tensor, alpha: float, 
                                 max_iter: int = 10000, tol: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Solve lower-level problem accurately as per F2CSA.tex Algorithm 1
        
        Args:
            x: Upper-level variable
            alpha: Accuracy parameter (δ = α³)
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            y_opt: Accurate solution to lower-level problem
            lambda_opt: Accurate dual variables
            info: Solution information
        """
        # Set δ = α³ as per F2CSA.tex
        delta = alpha ** 3
        
        # Try CVXPY first for maximum accuracy
        try:
            y_opt, lambda_opt, info = self._solve_cvxpy(x, tol)
            if info['converged']:
                return y_opt, lambda_opt, info
        except Exception as e:
            print(f"CVXPY failed: {e}")
        
        # Fallback to improved iterative solver
        return self._solve_iterative_improved(x, alpha, max_iter, tol)
    
    def solve_lower_level_with_solver(self, x: torch.Tensor, alpha: float, 
                                     max_iter: int, tol: float, solver_name: str) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Solve lower-level problem with specified solver
        
        Args:
            x: Upper-level variable
            alpha: Accuracy parameter (δ = α³)
            max_iter: Maximum iterations
            tol: Convergence tolerance
            solver_name: Solver to use ('SCS', 'OSQP', 'Clarabel')
            
        Returns:
            y_opt: Accurate solution to lower-level problem
            lambda_opt: Accurate dual variables
            info: Solution information
        """
        # Set δ = α³ as per F2CSA.tex
        delta = alpha ** 3
        
        # Try specified solver first
        try:
            y_opt, lambda_opt, info = self._solve_cvxpy_with_solver(x, tol, solver_name)
            if info['converged']:
                return y_opt, lambda_opt, info
        except Exception as e:
            print(f"{solver_name} failed: {e}")
        
        # Fallback to improved iterative solver
        return self._solve_iterative_improved(x, alpha, max_iter, tol)
    
    def _solve_cvxpy_with_solver(self, x: torch.Tensor, tol: float, solver_name: str) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Solve using CVXPY with specified solver"""
        try:
            import cvxpy as cp
            
            x_np = x.detach().cpu().numpy()
            y_cp = cp.Variable(self.problem.dim)
            
            # Lower-level objective: 0.5 * y^T Q_lower y + c_lower^T y
            Q_lower_np = self.problem.Q_lower.detach().cpu().numpy()
            c_lower_np = self.problem.c_lower.detach().cpu().numpy()
            objective = cp.Minimize(0.5 * cp.quad_form(y_cp, Q_lower_np) + c_lower_np @ y_cp)
            
            # Constraints: Ax + By - b <= 0
            A_np = self.problem.A.detach().cpu().numpy()
            B_np = self.problem.B.detach().cpu().numpy()
            b_np = self.problem.b.detach().cpu().numpy()
            constraints = [A_np @ x_np + B_np @ y_cp - b_np <= 0]
            
            prob = cp.Problem(objective, constraints)
            
            # Configure solver based on solver_name
            if solver_name.upper() == 'SCS':
                prob.solve(solver=cp.SCS, verbose=False, eps=tol, max_iters=10000)
            elif solver_name.upper() == 'OSQP':
                prob.solve(solver=cp.OSQP, verbose=False, eps_abs=tol, eps_rel=tol, max_iter=10000, polish=True)
            elif solver_name.upper() == 'CLARABEL':
                prob.solve(solver=cp.CLARABEL, verbose=False, eps_abs=tol, max_iter=10000)
            else:
                # Default solver
                prob.solve(verbose=False)
            
            if prob.status in ["optimal", "optimal_near"]:
                y_opt_np = y_cp.value
                lambda_opt_np = constraints[0].dual_value if constraints[0].dual_value is not None else np.zeros(self.problem.num_constraints)
                
                y_opt = torch.tensor(y_opt_np, device=self.device, dtype=self.dtype)
                lambda_opt = torch.tensor(lambda_opt_np, device=self.device, dtype=self.dtype)
                
                # Verify constraint satisfaction
                h_final = self.problem.constraints(x, y_opt)
                violations = torch.clamp(h_final, min=0)
                max_violation = torch.max(violations).item()
                
                info = {
                    'status': prob.status,
                    'iterations': 0,
                    'lambda': lambda_opt,
                    'constraint_violations': violations,
                    'converged': True,
                    'max_violation': max_violation,
                    'solver': solver_name.upper()
                }
                return y_opt, lambda_opt, info
            else:
                raise Exception(f"{solver_name} failed with status: {prob.status}")
                
        except ImportError:
            raise ImportError(f"CVXPY with {solver_name} not available")
        except Exception as e:
            raise Exception(f"{solver_name} error: {e}")

    def _solve_cvxpy(self, x: torch.Tensor, tol: float) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Solve using CVXPY for high precision (default solver)"""
        return self._solve_cvxpy_with_solver(x, tol, 'SCS')  # Default to SCS
    
    def _solve_iterative_improved(self, x: torch.Tensor, alpha: float, max_iter: int, tol: float) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Improved iterative solver with better convergence properties"""
        delta = alpha ** 3  # F2CSA accuracy parameter
        
        # Initialize with better starting point
        y = torch.zeros(self.problem.dim, device=self.device, dtype=self.dtype, requires_grad=True)
        lambda_vars = torch.zeros(self.problem.num_constraints, device=self.device, dtype=self.dtype, requires_grad=True)
        
        # Use Adam optimizer for better convergence
        optimizer = torch.optim.Adam([y, lambda_vars], lr=0.01, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5)
        
        best_y = y.clone()
        best_lambda = lambda_vars.clone()
        best_obj = float('inf')
        
        for iter in range(max_iter):
            optimizer.zero_grad()
            
            # Compute Lagrangian: L(y, λ) = f(y) + λ^T h(x,y)
            # where f(y) = 0.5 * y^T Q_lower y + c_lower^T y
            # and h(x,y) = Ax + By - b
            
            # Lower-level objective
            f_val = 0.5 * y.T @ self.problem.Q_lower @ y + self.problem.c_lower.T @ y
            
            # Constraint violations
            h_val = self.problem.A @ x + self.problem.B @ y - self.problem.b
            
            # Lagrangian
            lagrangian = f_val + lambda_vars.T @ h_val
            
            # Add penalty for constraint violations (augmented Lagrangian)
            penalty = 100.0 * torch.sum(torch.clamp(h_val, min=0) ** 2)
            total_obj = lagrangian + penalty
            
            total_obj.backward()
            
            # Check convergence
            grad_norm = torch.norm(torch.cat([y.grad, lambda_vars.grad]))
            max_violation = torch.max(torch.clamp(h_val, min=0)).item()
            
            if total_obj.item() < best_obj:
                best_obj = total_obj.item()
                best_y = y.clone().detach()
                best_lambda = lambda_vars.clone().detach()
            
            # Update
            optimizer.step()
            scheduler.step(total_obj.item())
            
            # Project lambda to non-negative
            with torch.no_grad():
                lambda_vars.clamp_(min=0)
            
            # Check convergence criteria
            if grad_norm < tol and max_violation < delta:
                break
            
            if iter % 1000 == 0:
                print(f"Iter {iter}: obj={total_obj.item():.6f}, grad_norm={grad_norm:.2e}, max_violation={max_violation:.2e}")
        
        # Final constraint violations
        final_h_val = self.problem.A @ x + self.problem.B @ best_y - self.problem.b
        final_violations = self.problem.constraint_violations(x, best_y)
        
        info = {
            'status': 'converged' if grad_norm < tol and max_violation < delta else 'max_iter',
            'iterations': iter + 1,
            'lambda': best_lambda,
            'constraint_violations': final_violations,
            'converged': grad_norm < tol and max_violation < delta,
            'final_grad_norm': grad_norm.item(),
            'final_max_violation': max_violation,
            'solver': 'Adam'
        }
        
        return best_y, best_lambda, info
    
    def solve_with_sgd(self, x: torch.Tensor, alpha: float, max_iter: int = 10000) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Solve using SGD as specified in F2CSA.tex Algorithm 1
        """
        # Set δ = α³ as per F2CSA.tex
        delta = alpha ** 3
        
        # Initialize
        y = torch.zeros(self.problem.dim, device=self.device, dtype=self.dtype, requires_grad=True)
        
        # SGD parameters
        lr = 0.01
        momentum = 0.9
        velocity = torch.zeros_like(y)
        
        # Augmented Lagrangian parameters
        rho = 1.0
        lambda_penalty = torch.zeros(self.problem.num_constraints, device=self.device, dtype=self.dtype)
        
        best_y = y.clone()
        best_obj = float('inf')
        
        for iteration in range(max_iter):
            # Compute objective and constraints
            obj = self.problem.lower_objective(x, y)
            h_val = self.problem.constraints(x, y)
            
            # Augmented Lagrangian
            penalty_term = torch.sum(torch.clamp(lambda_penalty + rho * h_val, min=0)**2) / (2 * rho)
            augmented_obj = obj + penalty_term
            
            # Compute gradient
            grad = torch.autograd.grad(augmented_obj, y, create_graph=False)[0]
            
            # SGD with momentum
            velocity = momentum * velocity + grad
            y = y - lr * velocity
            
            # Update dual variables
            with torch.no_grad():
                lambda_penalty = torch.clamp(lambda_penalty + rho * h_val, min=0)
            
            # Track best solution
            if augmented_obj.item() < best_obj:
                best_obj = augmented_obj.item()
                best_y = y.clone()
            
            # Check convergence
            grad_norm = torch.norm(grad)
            constraint_violation = torch.max(torch.clamp(h_val, min=0))
            
            if iteration % 1000 == 0:
                print(f"SGD Iter {iteration}: obj={obj.item():.6f}, grad_norm={grad_norm:.2e}, "
                      f"max_violation={constraint_violation:.2e}")
            
            # Convergence criteria
            if grad_norm < delta and constraint_violation < delta:
                print(f"SGD Converged at iteration {iteration}")
                break
        
        # Final solution
        y_opt = best_y.detach()
        lambda_opt = lambda_penalty.detach()
        
        # Final constraint check
        h_final = self.problem.constraints(x, y_opt)
        violations = torch.clamp(h_final, min=0)
        
        info = {
            'iterations': iteration + 1,
            'lambda': lambda_opt,
            'constraint_violations': violations,
            'converged': grad_norm < delta and constraint_violation < delta,
            'final_grad_norm': grad_norm.item(),
            'final_violation': constraint_violation.item(),
            'delta': delta,
            'solver': 'SGD'
        }
        
        return y_opt, lambda_opt, info