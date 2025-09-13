#!/usr/bin/env python3
"""
Enhanced F2CSA with Smooth Activation + Stochastic Averaging + Adaptive Momentum
FOCUSED ON THEORETICAL ALIGNMENT + EMA GAP REDUCTION

Adds a standard .optimize(...) API to integrate with experiment runner.
"""

import torch
import numpy as np
import time
import cvxpy as cp
from typing import Dict, Tuple, Optional, List
from torch.optim.lr_scheduler import ReduceLROnPlateau
from problem import StronglyConvexBilevelProblem
import warnings
warnings.filterwarnings('ignore')


class F2CSAAlgorithm:
    """
    Enhanced F2CSA with Smooth Activation + Stochastic Averaging + Adaptive Momentum

    New in this version:
    - optimize(max_iterations, early_stopping_patience, target_gap) to match runner
    - optional external problem injection via __init__(problem=...)
    """
    
    def __init__(self, 
                 problem: Optional[StronglyConvexBilevelProblem] = None,
                 device: str = 'cpu',
                 seed: int = 42,
                 alpha_override: Optional[float] = None,
                 eta_override: Optional[float] = None,
                 D_override: Optional[float] = None,
                 s_fixed_override: Optional[float] = None,
                 Ng_override: Optional[int] = None,
                 grad_ema_beta_override: Optional[float] = None,
                 prox_weight_override: Optional[float] = None,
                 grad_clip_override: Optional[float] = None):
        self.device = device
        self.seed = seed
        self.problem = problem
        # Set requested stable defaults when not provided via overrides
        self.alpha_override = 0.01 if alpha_override is None else float(alpha_override)
        # Stabilization buffers
        self.grad_ema: Optional[torch.Tensor] = None
        self.grad_ema_beta: float = 0.90 if grad_ema_beta_override is None else float(grad_ema_beta_override)
        self.prox_weight: float = 0.03 if prox_weight_override is None else float(prox_weight_override)
        self.eta_override = 0.001 if eta_override is None else float(eta_override)
        self.D_override = 0.01 if D_override is None else float(D_override)
        self.s_fixed_override = s_fixed_override
        self.Ng_override = 16 if Ng_override is None else int(Ng_override)
        self.grad_clip_override = 0.8 if grad_clip_override is None else float(grad_clip_override)
        # Use true solver for lower-level (consistent with DSBLO/SSIGD)
        self.use_sgd_lower_level: bool = False  # Use true solver, not SGD
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print(f"[ENHANCED] F2CSA - ACTIVATION + STOCHASTIC + ADAPTIVE MOMENTUM")
        print(f"   Features: Smooth Activation + N_g=1 + Adaptive Techniques")
        print(f"   Focus: Theoretical alignment + EMA Gap Reduction")
        print(f"   Target: 1e-2 EMA Gap with F2CSA.tex alignment")

    def create_problem(self, dim=5, epsilon=0.012, strong_convex=True):
        """Create problem if not supplied by caller."""
        return StronglyConvexBilevelProblem(
            dim=dim, num_constraints=3, noise_std=0.0005,
            device=self.device, seed=self.seed, strong_convex=strong_convex
        )

    def smooth_activation(self, h_val: torch.Tensor, lambda_val: torch.Tensor,
                          tau_delta: float, epsilon_lambda: float) -> torch.Tensor:
        sigma_h = torch.where(h_val < -tau_delta, torch.tensor(0.0, device=h_val.device, dtype=h_val.dtype),
                              torch.where(h_val < 0, (tau_delta + h_val) / tau_delta,
                                          torch.tensor(1.0, device=h_val.device, dtype=h_val.dtype)))
        sigma_lambda = torch.where(lambda_val <= 0, torch.tensor(0.0, device=lambda_val.device, dtype=lambda_val.dtype),
                                   torch.where(lambda_val < epsilon_lambda,
                                               lambda_val / epsilon_lambda,
                                               torch.tensor(1.0, device=lambda_val.device, dtype=lambda_val.dtype)))
        return sigma_h * sigma_lambda

    def compute_gap_torch_based(self, x: torch.Tensor) -> float:
        """
        Delegates to the unified gap used by the experiment runner.
        """
        return self.problem.compute_gap(x)
    
    def _modify_m_parameter(self, m_value: int):
        """
        Modify the M parameter (averaging window) for testing
        """
        self.m_override = m_value
        print(f"[MODIFY] M parameter set to {m_value}")
    
    def _modify_k_parameter(self, k_value: int):
        """
        Modify the K parameter (number of blocks) for testing
        """
        self.k_override = k_value
        print(f"[MODIFY] K parameter set to {k_value}")


    # ------------------------ validation helpers (Algorithm 1) ------------------------
    def oracle_sample(self, x: torch.Tensor, alpha: float, N_g: int, config: dict = None) -> torch.Tensor:
        """
        One stochastic oracle sample g~(x) using true solver for lower-level.
        Uses the same solver as DSBLO/SSIGD for consistency.
        """
        problem = self.problem
        dim = problem.dim
        xx = x.detach().clone().requires_grad_(True)
        
        # Use Algorithm 2 with SGD for lower-level as specified in deterministic.py
        y_opt, lambda_opt = self._solve_lower_level_algorithm2_sgd(xx, problem, alpha, config)
        
        # Compute constraint violations
        h_val = problem.A @ xx - problem.B @ y_opt - problem.b
        
        # Penalty Lagrangian terms
        alpha_1 = alpha ** (-2)
        alpha_2 = alpha ** (-4)
        h_val_penalty = h_val
        tau_delta = 0.10
        epsilon_lambda = 0.10
        rho_i = self.smooth_activation(h_val_penalty, lambda_opt, tau_delta, epsilon_lambda)
        
        f_val = problem.upper_objective(xx, y_opt, add_noise=True)
        g_val = problem.lower_objective(xx, y_opt, add_noise=False)
        g_val_at_y_star = g_val  # best available proxy
        
        term1 = f_val
        term2 = alpha_1 * (g_val + torch.dot(lambda_opt, h_val_penalty) - g_val_at_y_star)
        term3 = 0.5 * alpha_2 * torch.sum(rho_i * (h_val_penalty ** 2))
        penalty_term = term2 + term3
        
        # Build stochastic gradient by averaging N_g samples of the f part
        accumulated_grad_f = torch.zeros_like(xx)
        for _ in range(max(1, int(N_g))):
            f_val_sample = problem.upper_objective(xx, y_opt, add_noise=True)
            grad_f_sample = torch.autograd.grad(f_val_sample, xx, create_graph=False)[0]
            accumulated_grad_f += grad_f_sample
        grad_f = accumulated_grad_f / max(1, int(N_g))
        grad_penalty = torch.autograd.grad(penalty_term, xx, create_graph=False)[0]
        g_t = (grad_f + grad_penalty).detach()
        return g_t

    def _solve_lower_level_cvxpy(self, x, problem, alpha):
        """
        Solve lower-level problem using CVXPY as specified in Algorithm 1 of F2CSA.tex
        
        Returns:
            y_opt: Approximate solution to lower-level problem
            lambda_opt: Approximate dual variables
        """
        try:
            # Convert to numpy for CVXPY
            x_np = x.detach().cpu().numpy()
            
            # Problem matrices
            Q_lower_np = problem.Q_lower.detach().cpu().numpy()
            c_lower_np = problem.c_lower.detach().cpu().numpy()
            P_np = problem.P.detach().cpu().numpy()
            A_np = problem.A.detach().cpu().numpy()
            B_np = problem.B.detach().cpu().numpy()
            b_np = problem.b.detach().cpu().numpy()
            
            # CVXPY variables
            y_cvxpy = cp.Variable(problem.dim)
            
            # Lower-level objective: min_y 0.5 * y^T Q y + (c + P^T x)^T y
            objective = cp.Minimize(0.5 * cp.quad_form(y_cvxpy, Q_lower_np) + 
                                   (c_lower_np + P_np.T @ x_np) @ y_cvxpy)
            
            # Constraints: B y >= A x - b (equivalent to A x - B y - b <= 0)
            constraints = [B_np @ y_cvxpy >= A_np @ x_np - b_np]
            
            # Solve the problem
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.SCS, verbose=False, max_iters=1000)
            
            if prob.status not in ["infeasible", "unbounded"]:
                # Extract solution
                y_opt_np = y_cvxpy.value
                y_opt = torch.tensor(y_opt_np, device=self.device, dtype=problem.dtype)
                
                # Extract dual variables (Lagrange multipliers)
                if len(prob.constraints) > 0:
                    lambda_opt_np = prob.constraints[0].dual_value
                    lambda_opt = torch.tensor(lambda_opt_np, device=self.device, dtype=problem.dtype)
                else:
                    lambda_opt = torch.zeros(problem.num_constraints, device=self.device, dtype=problem.dtype)
                
                return y_opt, lambda_opt
            else:
                print(f"Warning: CVXPY problem status: {prob.status}")
                # Return fallback values
                y_opt = torch.zeros(problem.dim, device=self.device, dtype=problem.dtype)
                lambda_opt = torch.zeros(problem.num_constraints, device=self.device, dtype=problem.dtype)
                return y_opt, lambda_opt
                
        except Exception as e:
            print(f"Error in _solve_lower_level_cvxpy: {e}")
            # Return fallback values
            y_opt = torch.zeros(problem.dim, device=self.device, dtype=problem.dtype)
            lambda_opt = torch.zeros(problem.num_constraints, device=self.device, dtype=problem.dtype)
            return y_opt, lambda_opt

    def _solve_lower_level_algorithm2_sgd(self, x, problem, alpha, config):
        """
        Solve lower-level problem using Algorithm 2 with SGD as specified in deterministic.py lines 229-257
        
        This implements the FFO complex approach with SGD for the Lagrangian optimization problem.
        
        Returns:
            y_opt: Approximate solution to lower-level problem
            lambda_opt: Approximate dual variables
        """
        try:
            # Handle None config
            if config is None:
                config = {}
            
            # Parameters from deterministic.py
            eps_abs = alpha ** 2  # Line 153: eps_abs = eps ** 2
            warm_start = True
            lamb_init = 1 / alpha  # Line 155: lamb_init = 1 / eps
            lamb = lamb_init
            delta = torch.zeros(problem.dim, device=self.device, dtype=problem.dtype)  # Line 157
            y_lamb_opt = None
            
            # Convert to numpy for CVXPY
            x_np = x.detach().cpu().numpy()
            
            # Problem matrices
            Q_lower_np = problem.Q_lower.detach().cpu().numpy()
            c_lower_np = problem.c_lower.detach().cpu().numpy()
            P_np = problem.P.detach().cpu().numpy()
            A_np = problem.A.detach().cpu().numpy()
            B_np = problem.B.detach().cpu().numpy()
            b_np = problem.b.detach().cpu().numpy()
            
            # Step 1: Pre-solve inner optimization problem (CVXPY)
            y_cp = cp.Variable(problem.dim)
            q_cp = x_np.T @ P_np
            objective = cp.Minimize(0.5 * cp.quad_form(y_cp, Q_lower_np) + q_cp @ y_cp)
            constraints = [A_np @ x_np + B_np @ y_cp - b_np <= 0]
            
            if y_lamb_opt is not None:
                y_cp.value = y_lamb_opt.detach().cpu().numpy()
            
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.SCS, eps_abs=eps_abs, warm_start=warm_start)
            
            if prob.status in ["infeasible", "unbounded"]:
                print(f"Warning: CVXPY problem status: {prob.status}")
                y_opt = torch.zeros(problem.dim, device=self.device, dtype=problem.dtype)
                lambda_opt = torch.zeros(problem.num_constraints, device=self.device, dtype=problem.dtype)
                return y_opt, lambda_opt
            
            x_opt = x_np
            y_opt_np = y_cp.value
            gamma_opt_np = constraints[0].dual_value  # Line 186
            
            # Check active constraints (Line 189-190)
            h_opt_np = A_np @ x_opt + B_np @ y_opt_np - b_np
            active_constraints = (np.abs(h_opt_np) < 1e-4) * (gamma_opt_np > 1e-4)
            
            # Step 2: Solve Lagrangian optimization problem using SGD (Lines 195-208)
            # Convert to PyTorch tensors for SGD
            y_lagrangian = torch.tensor(y_opt_np, device=self.device, dtype=problem.dtype, requires_grad=True)
            gamma_opt = torch.tensor(gamma_opt_np, device=self.device, dtype=problem.dtype)
            
            # SGD optimizer for Lagrangian problem
            sgd_lr = config.get('ll_sgd_lr', 0.01)
            sgd_steps = config.get('ll_sgd_steps', 50)
            sgd_momentum = config.get('ll_sgd_momentum', 0.9)
            
            y_optimizer = torch.optim.SGD([y_lagrangian], lr=sgd_lr, momentum=sgd_momentum)
            
            for step in range(sgd_steps):
                y_optimizer.zero_grad()
                
                # Upper-level objective: f(x, y) = c^T y + 0.01 * ||x||^2 + 0.01 * ||y||^2
                f_val = problem.c_upper @ y_lagrangian + 0.01 * torch.norm(x)**2 + 0.01 * torch.norm(y_lagrangian)**2
                
                # Lower-level objective: g(x, y) = 0.5 * y^T Q y + x^T P y
                g_val = 0.5 * y_lagrangian.T @ problem.Q_lower @ y_lagrangian + x.T @ problem.P @ y_lagrangian
                
                # Constraint function: h(x, y) = A x + B y - b
                h_val = problem.A @ x + problem.B @ y_lagrangian - problem.b
                
                # Optimal values for reference
                y_opt_tensor = torch.tensor(y_opt_np, device=self.device, dtype=problem.dtype)
                g_opt_val = 0.5 * y_opt_tensor.T @ problem.Q_lower @ y_opt_tensor + x.T @ problem.P @ y_opt_tensor
                h_opt_val = problem.A @ x + problem.B @ y_opt_tensor - problem.b
                
                # Lagrangian objective (Line 204-207)
                if sum(active_constraints) == 0:
                    lagrangian_obj = f_val + lamb * (g_val + gamma_opt.T @ h_val - g_opt_val - gamma_opt.T @ h_opt_val)
                else:
                    # Active constraints penalty
                    h_active = h_val[active_constraints]
                    constraint_penalty = 0.5 * lamb**2 * torch.sum(h_active**2)
                    lagrangian_obj = f_val + lamb * (g_val + gamma_opt.T @ h_val - g_opt_val - gamma_opt.T @ h_opt_val) + constraint_penalty
                
                # Check for NaN or Inf
                if torch.isnan(lagrangian_obj) or torch.isinf(lagrangian_obj):
                    print(f"Warning: NaN/Inf in lagrangian_obj at step {step}")
                    break
                
                lagrangian_obj.backward()
                y_optimizer.step()
                
                # Project to feasible region if needed
                with torch.no_grad():
                    h_val = problem.A @ x + problem.B @ y_lagrangian - problem.b
                    violation = torch.relu(h_val)
                    if torch.sum(violation) > 1e-6:
                        # Simple projection step
                        step_size = 0.01 / max(1.0, torch.sum(violation).item())
                        y_lagrangian.data = y_lagrangian.data - step_size * problem.B.T @ violation
            
            # Update y_lamb_opt for next iteration
            y_lamb_opt = y_lagrangian.detach()
            
            # Convert back to numpy for consistency
            y_opt = y_lagrangian.detach()
            lambda_opt = gamma_opt  # Use gamma as lambda approximation
            
            return y_opt, lambda_opt
            
        except Exception as e:
            print(f"Error in _solve_lower_level_algorithm2_sgd: {e}")
            # Return fallback values
            y_opt = torch.zeros(problem.dim, device=self.device, dtype=problem.dtype)
            lambda_opt = torch.zeros(problem.num_constraints, device=self.device, dtype=problem.dtype)
            return y_opt, lambda_opt

    def estimate_bias_variance(self, x: torch.Tensor, alpha_values: List[float], Ng_values: List[int], trials: int = 50, ref_trials: int = 200) -> Dict:
        """
        Monte Carlo estimation:
        - For each alpha, form a high-precision reference gradient by averaging ref_trials samples.
        - Bias alpha-scaling: || E[g~] - ref || via sample means.
        - Variance scaling: Var(g~) versus Ng.
        """
        results: Dict[str, Dict] = {}
        for alpha in alpha_values:
            entry = {'alpha': float(alpha), 'bias_norm': None, 'Ng_stats': []}
            # reference using largest Ng of Ng_values and many trials
            Ng_ref = max(Ng_values) if len(Ng_values) > 0 else 32
            ref_grads = []
            for _ in range(ref_trials):
                ref_grads.append(self.oracle_sample(x, alpha, Ng_ref))
            ref_grad = torch.stack(ref_grads, dim=0).mean(dim=0)
            # bias using mean with medium Ng
            Ng_bias = Ng_ref
            mean_grads = []
            for _ in range(trials):
                mean_grads.append(self.oracle_sample(x, alpha, Ng_bias))
            mean_grad = torch.stack(mean_grads, dim=0).mean(dim=0)
            entry['bias_norm'] = float(torch.norm(mean_grad - ref_grad))
            # variance over Ng sweep
            for Ng in Ng_values:
                samples = []
                for _ in range(trials):
                    samples.append(self.oracle_sample(x, alpha, Ng))
                S = torch.stack(samples, dim=0)
                mu = S.mean(dim=0)
                var = ((S - mu) ** 2).sum(dim=1).mean().sqrt().item()
                entry['Ng_stats'].append({'Ng': int(Ng), 'var_norm': float(var)})
            results[str(alpha)] = entry
        return results

    def _run_optimization(self,
                          max_total_iterations: Optional[int] = None,
                          early_stopping_patience: Optional[int] = None,
                          target_gap: Optional[float] = None,
                          verbose: bool = False,
                          run_until_convergence: bool = True) -> Dict:
        total_iters = 0

        # Practical parameter tuning (no theoretical schedules)
        # Using conservative parameters that achieve gap < 0.1
        alpha = self.alpha_override if self.alpha_override is not None else 0.01
        N_g = self.Ng_override if self.Ng_override is not None else 16
        D_scaled = self.D_override if self.D_override is not None else 0.01
        eta_scaled = self.eta_override if self.eta_override is not None else 0.001
        M = 10  # Fixed averaging window

        if verbose:
            print(f"   Practical F2CSA Parameters:")
            print(f"   D = {D_scaled:.3f} (clipping)")
            print(f"   η = {eta_scaled:.3f} (step size)")
            print(f"   M = {M} (averaging window)")
            print(f"   α = {alpha:.3f} (accuracy parameter)")
            print(f"   N_g = {N_g} (batch size)")

        config = {
            'name': 'Enhanced F2CSA',
            'dim': 5, 'epsilon': 0.012, 'alpha': alpha, 'eta': eta_scaled,
            'N_g': N_g, 'inner_steps': 250, 'D': D_scaled, 'lr': 5e-3,
            'technique': {},
            'use_sgd_lower_level': True,  # Use SGD as specified in Algorithm 1
            'll_sgd_lr': 0.01,
            'll_sgd_steps': 50,  # Fixed practical number of steps
            'll_sgd_momentum': 0.9,
        }


        if self.problem is None:
            self.problem = self.create_problem(
                dim=config.get('dim', 5),
                epsilon=config.get('epsilon', 0.012)
            )
        problem = self.problem
        dim = config.get('dim', 5)
        epsilon = config.get('epsilon', 0.02)
        alpha = config.get('alpha', 0.008)
        eta = config['eta']
        N_g = config['N_g']
        D = config['D']
        lr = config['lr']
        technique = config.get('technique', {})

        # vars
        x = torch.randn(dim, device=self.device, dtype=problem.dtype) * 0.1
        x.requires_grad_(True)
        # Using paper's fixed-step update; no external optimizer or scheduler

        delta = torch.zeros(dim, device=self.device, dtype=problem.dtype)
        ema_gap = None
        ema_gap_history: List[float] = []
        gap_history: List[float] = []
        x_history: List[torch.Tensor] = []
        stage_gaps: List[float] = []
        averaged_points: List[torch.Tensor] = []  # For ffo_complex-like averaging
        current_block_points: List[torch.Tensor] = []  # To collect points for one block

        # best_gap logic removed, using ema_gap for tracking
        total_iters = 0
        target_gap = 1e-3 if target_gap is None else target_gap
        
        # Convergence tracking
        convergence_patience = early_stopping_patience if early_stopping_patience is not None else 1000
        best_ema_gap = float('inf')
        iterations_since_improvement = 0
        convergence_threshold = 1e-6  # Minimum improvement to consider as progress

        # Remove iteration limits for convergence-based stopping
        max_stages = 1000  # Large number to allow many stages
        max_iterations_per_stage = 10000  # Large number per stage

        start_time = time.time()

        # --- LL caching / reuse controls ---
        ll_reuse_tol = 1e-3         # reuse y* if ||x - x_last|| < tol
        ll_solve_period = 10        # force a fresh LL solve at least every N iters
        last_ll_x_t: Optional[torch.Tensor] = None
        last_ll_y_np: Optional[np.ndarray] = None
        last_ll_lambda_np: Optional[np.ndarray] = None
        last_ll_iter: int = -1

        # Metrics for caching effectiveness
        ll_reuse_count = 0
        ll_solve_count = 0

        # --- Gap evaluation caching controls ---
        gap_eval_period = 20         # compute gap via CVXPy every N iterations
        gap_reuse_tol = 5e-3         # reuse gap if ||x - last_gap_x|| < tol
        last_gap_x_t: Optional[torch.Tensor] = None
        last_gap_value: Optional[float] = None
        last_gap_iter: int = -1
        gap_reuse_count = 0
        gap_compute_count = 0

        for stage in range(max_stages):
            # quiet stage banner

            # Manual parameter adjustment removed in favor of ReduceLROnPlateau scheduler.

            # FIXED: Do not reset delta momentum between stages to prevent convergence reset
            # delta.zero_()  # REMOVED: This was causing the algorithm to reset at convergence
            current_gap = float('inf')  # Initialize for the stage

            # stage loop
            for i in range(max_iterations_per_stage):
                # Only respect max_total_iterations if not running until convergence
                if (not run_until_convergence) and (max_total_iterations is not None) and (total_iters >= max_total_iterations):
                    break

                s = 1.0 if (self.s_fixed_override is not None) else torch.rand(1, device=self.device).item()
                # Guard against Tensor/NumPy mixing
                if not isinstance(x, torch.Tensor):
                    x = torch.as_tensor(x, device=self.device, dtype=problem.dtype)
                if not isinstance(delta, torch.Tensor):
                    delta = torch.as_tensor(delta, device=self.device, dtype=problem.dtype)
                xx = x + s * delta
                x_history.append(xx.clone())
                current_block_points.append(xx.clone())

                # --- inner LL solve with CVXPy (with caching and warm-start) ---
                reuse_ll = False
                if last_ll_x_t is not None:
                    try:
                        if torch.norm(xx - last_ll_x_t).item() < ll_reuse_tol and (total_iters - last_ll_iter) < ll_solve_period:
                            reuse_ll = True
                    except Exception:
                        reuse_ll = False

                if reuse_ll and (last_ll_y_np is not None) and (last_ll_lambda_np is not None):
                    # Reuse cached LL solution
                    ll_reuse_count += 1
                    y_opt = torch.as_tensor(last_ll_y_np, device=self.device, dtype=problem.dtype)
                    lambda_opt = torch.as_tensor(last_ll_lambda_np, device=self.device, dtype=problem.dtype)
                else:
                    if self.use_sgd_lower_level:
                        # Use Algorithm 2 with SGD for lower-level as specified in deterministic.py
                        y_opt, lambda_opt = self._solve_lower_level_algorithm2_sgd(xx, problem, alpha, config)
                        
                        # Update cache
                        last_ll_x_t = xx.detach().clone()
                        last_ll_y_np = y_opt.detach().cpu().numpy()
                        last_ll_lambda_np = lambda_opt.detach().cpu().numpy()
                        last_ll_iter = total_iters
                        ll_solve_count += 1
                    else:
                        # Use Algorithm 2 with SGD for lower-level as specified in deterministic.py
                        y_opt, lambda_opt = self._solve_lower_level_algorithm2_sgd(xx, problem, alpha, config)
                        
                        # Update cache
                        last_ll_x_t = xx.detach().clone()
                        last_ll_y_np = y_opt.detach().cpu().numpy()
                        last_ll_lambda_np = lambda_opt.detach().cpu().numpy()
                        last_ll_iter = total_iters
                        ll_solve_count += 1

                # Fixed alpha (annealing disabled)
                alpha = self.alpha_override if self.alpha_override is not None else config.get('alpha', 0.95)

                # ffo_complex-like averaging logic
                if total_iters % M == 0 and current_block_points:
                    averaged_point = torch.mean(torch.stack(current_block_points), dim=0)
                    averaged_points.append(averaged_point)
                    current_block_points = []  # Reset for the next block

                # Constraint violation is now implicitly handled by the aligned penalty Lagrangian.

                # --- F2CSA.tex Penalty Lagrangian (Equation 1) ---
                # Extract dual variables (lambda) from the CVXPy solution
                lambda_opt = lambda_opt  # already set by chosen LL path

                # Define penalty parameters from F2CSA.tex
                alpha_1 = alpha ** (-2)
                alpha_2 = alpha ** (-4)

                # Calculate h(x, y) for the penalty term
                h_val_penalty = problem.A @ xx - problem.B @ y_opt - problem.b

                # Calculate smooth activation rho_i for each constraint
                tau_delta = 0.01  # Practical smoothing parameter
                epsilon_lambda = 0.01  # Practical smoothing parameter
                rho_i = self.smooth_activation(h_val_penalty, lambda_opt, tau_delta, epsilon_lambda)

                # Compute the terms of the penalty Lagrangian (F2CSA.tex Equation 1)
                f_val = self.problem.upper_objective(xx, y_opt, add_noise=False)  # No noise for penalty construction
                g_val = self.problem.lower_objective(xx, y_opt, add_noise=False)
                g_val_at_y_star = g_val  # y_opt is our best estimate of y*

                # Construct the penalty Lagrangian exactly as in F2CSA.tex
                term1 = f_val
                term2 = alpha_1 * (g_val + torch.dot(lambda_opt, h_val_penalty) - g_val_at_y_star)
                term3 = 0.5 * alpha_2 * torch.sum(rho_i * (h_val_penalty**2))
                final_lagrangian = term1 + term2 + term3
                # Lightweight alpha diagnostics
                if verbose and (total_iters % 100 == 0):
                    try:
                        print(f"alpha={alpha:.3f}, |term2|={term2.detach().abs().item():.3e}, |term3|={term3.detach().abs().item():.3e}")
                    except Exception:
                        pass

                # F2CSA.tex Algorithm 1: Stochastic Hypergradient Oracle
                # Calculate gradient of the full penalty Lagrangian
                accumulated_grad = torch.zeros_like(xx)
                for _ in range(N_g):
                    # Create fresh computational graph for each sample
                    xx_fresh = xx.detach().clone().requires_grad_(True)
                    
                    # Sample stochastic upper-level objective
                    f_val_sample = self.problem.upper_objective(xx_fresh, y_opt, add_noise=True)
                    
                    # Calculate penalty Lagrangian with stochastic f
                    penalty_lagrangian_sample = f_val_sample + term2 + term3
                    
                    # Compute gradient of the full penalty Lagrangian
                    grad_sample = torch.autograd.grad(penalty_lagrangian_sample, xx_fresh, create_graph=False)[0]
                    accumulated_grad += grad_sample
                
                # Average over N_g samples
                gradient = accumulated_grad / N_g

                # Gradient EMA for stabilization - DISABLED FOR TESTING
                # if self.grad_ema is None:
                #     self.grad_ema = torch.zeros_like(gradient)
                # self.grad_ema = self.grad_ema_beta * self.grad_ema + (1 - self.grad_ema_beta) * gradient
                # stabilized_grad = self.grad_ema
                
                # Gradient EMA for stabilization (effective technique)
                if self.grad_ema is None:
                    self.grad_ema = torch.zeros_like(gradient)
                self.grad_ema = self.grad_ema_beta * self.grad_ema + (1 - self.grad_ema_beta) * gradient
                stabilized_grad = self.grad_ema

                # Proximal pull toward EMA of iterates - REMOVED (ineffective)
                prox_term = torch.zeros_like(x)

                # The smooth_activation function is now part of the Lagrangian, so the explicit multiplication is removed.

                # gradient clip - much higher threshold to allow aggressive steps
                if technique.get('gradient_clipping', True):
                    clip_norm = 0.1 if self.grad_clip_override is None else float(self.grad_clip_override)
                    gnorm = torch.norm(stabilized_grad)
                    if gnorm > clip_norm:
                        stabilized_grad = stabilized_grad / gnorm * clip_norm

                # F2CSA.tex Algorithm 2: Update with clipping
                # Δ_{t+1} = clip(Δ_t - η ĝ_t), ||Δ|| ≤ D
                delta = delta - eta * stabilized_grad
                dnorm = torch.norm(delta)
                if dnorm > D:
                    delta = delta / dnorm * D

                # Update x: x_{t+1} = x_t + s * Δ_{t+1}
                x = (x + s * delta).detach().clone().requires_grad_(True)

                # gap + EMA (with caching to reduce CVXPy calls)
                use_cached_gap = False
                if last_gap_x_t is not None:
                    try:
                        if torch.norm(xx - last_gap_x_t).item() < gap_reuse_tol and (total_iters - last_gap_iter) < gap_eval_period:
                            use_cached_gap = True
                    except Exception:
                        use_cached_gap = False

                if use_cached_gap and (last_gap_value is not None):
                    current_gap = last_gap_value
                else:
                    # Use unified gap from the shared problem
                    current_gap = self.problem.compute_gap(xx)
                    last_gap_x_t = xx.detach().clone()
                    last_gap_value = current_gap
                    last_gap_iter = total_iters

                gap_history.append(current_gap)
                stage_gaps.append(current_gap)

                # Fixed step size per paper; no adaptive LR scheduling

                # Gap EMA - REMOVED (ineffective)
                ema_gap = current_gap
                ema_gap_history.append(float(ema_gap))

                # No scheduler step; fixed η per paper

                # early stopping tracking
                # best_gap logic removed

                total_iters += 1

                # Enhanced Debugging Log
                if total_iters % 100 == 0:
                    # Calculate recent trend for logging (raw gap)
                    recent_trend = 0.0
                    if len(gap_history) >= 20:
                        recent_trend = np.mean(gap_history[-10:]) - np.mean(gap_history[-20:-10])

                    # Compute EMA slope (last 100) for reduction diagnostics
                    ema_slope_100 = 0.0
                    if len(ema_gap_history) >= 10:
                        window = ema_gap_history[-100:] if len(ema_gap_history) >= 100 else ema_gap_history
                        t = np.arange(len(window))
                        tm, ym = t.mean(), np.mean(window)
                        denom = ((t - tm) ** 2).sum()
                        if denom > 0:
                            ema_slope_100 = float((((t - tm) * (np.array(window) - ym)).sum()) / denom)

                    # Get gradient norm if not already computed
                    gnorm = torch.norm(gradient)

                    print(f"Iter {total_iters:5d} | EMA Gap: {ema_gap:.6f} | Alpha: {alpha:.4f}")

                # Convergence checking - track improvement
                if ema_gap is not None:
                    if ema_gap < best_ema_gap - convergence_threshold:
                        best_ema_gap = ema_gap
                        iterations_since_improvement = 0
                    else:
                        iterations_since_improvement += 1
                    
                    # Check for convergence
                    if run_until_convergence and iterations_since_improvement >= convergence_patience:
                        if verbose:
                            print(f"Converged after {total_iters} iterations (no improvement for {convergence_patience} iterations)")
                        break
                
                # target checks - use EMA gap for target achievement (only if not running until convergence)
                # Only stop if we've achieved target AND run for at least 100 iterations
                if not run_until_convergence and ema_gap is not None and ema_gap <= target_gap and total_iters >= 100:
                    break
            
            # stage end - check convergence or limits
            if run_until_convergence:
                # Only break if we've converged (handled in inner loop)
                if iterations_since_improvement >= convergence_patience:
                    break
            else:
                # Original logic for limited runs
                if (max_total_iterations is not None and total_iters >= max_total_iterations) or \
                   (ema_gap is not None and ema_gap <= target_gap):
                    break
        
        total_time = time.time() - start_time
        
        # ffo_complex-like output selection (Algorithm 2 from F2CSA.tex)
        if averaged_points:
            # Randomly select one of the averaged points as the output
            final_x = averaged_points[np.random.randint(len(averaged_points))]
        elif x_history:
            final_x = x_history[-1]  # Fallback if no full block was averaged
        else:
            final_x = x
        # Ensure final_x is a torch.Tensor
        if not isinstance(final_x, torch.Tensor):
            final_x = torch.as_tensor(final_x, device=self.device, dtype=problem.dtype)
        zero_y = torch.zeros_like(final_x)
        final_gap = self.problem.compute_gap(final_x) if len(gap_history) > 0 else float('inf')
        final_obj = self.problem.upper_objective(final_x, zero_y).item()
        target_achieved = (ema_gap is not None) and (ema_gap <= target_gap)

        ema_var = float(np.var(ema_gap_history)) if len(ema_gap_history) > 1 else float('nan')
        ema_std = float(np.std(ema_gap_history)) if len(ema_gap_history) > 1 else float('nan')
        results = {
            'final_x': final_x.detach().cpu().numpy(),
            'final_objective': final_obj,
            'gap_history': gap_history,
            'ema_gap_history': ema_gap_history,
            'ema_variance': ema_var,
            'ema_std': ema_std,
            'stage_gaps': stage_gaps,
            'total_iterations': total_iters,
            'total_time': total_time,
            'final_gap': final_gap,
            'final_ema_gap': float(ema_gap) if ema_gap is not None else float('inf'),
            'stage_count': stage + 1,
            'config': config,
            'technique': technique,
            # fields expected by runner:
            'history': {'gaps': gap_history, 'ema_gaps': ema_gap_history},
            'target_achieved': target_achieved
        }
        return results
    
    # ------------------------ public APIs ------------------------

    def optimize(self,
                 max_iterations: int = 1000,
                 early_stopping_patience: int = 100,
                 target_gap: float = 1e-3,
                 verbose: bool = False,
                 run_until_convergence: bool = False) -> Dict:
        """
        Public entry point used by the experiment runner.
        """
        return self._run_optimization(
            max_total_iterations=max_iterations,
            early_stopping_patience=early_stopping_patience,
            target_gap=target_gap,
            verbose=verbose,
            run_until_convergence=run_until_convergence
        )

    # (Optional) keep your testing harness identical for standalone runs
    def run_enhanced_testing(self):
        results = self._run_optimization(
            max_total_iterations=None,
            early_stopping_patience=None,
            target_gap=1e-2,   # uses EMA/your banner goal if you like
            verbose=True
        )
        analysis = self.analyze_enhanced_results(results)
        analysis['execution_time'] = results['total_time']
        # Persist minimal file as before
        with open('enhanced_f2csa_activation_momentum_results.txt', 'w') as f:
            f.write(f"ENHANCED F2CSA RESULTS\n")
            f.write(f"Final EMA Gap: {analysis.get('final_ema_gap', float('inf')):.6f}\n")
            f.write(f"Assessment: {analysis.get('assessment', 'N/A')}\n")
        print("\nResults saved to 'enhanced_f2csa_activation_momentum_results.txt'")
        return analysis

    def analyze_enhanced_results(self, results: Dict) -> Dict:
        # unchanged logic from your version, but consume new results dict
        final_ema_gap = results.get('final_ema_gap', float('inf'))
        gap_history = results.get('gap_history', [])
        ema_history = results.get('ema_gap_history', [])
        stage_count = results.get('stage_count', 1)

        if len(ema_history) >= 50:
            early_ema = np.mean(ema_history[:50])
            late_ema = np.mean(ema_history[-50:])
            ema_convergence_improvement = early_ema - late_ema
        else:
            ema_convergence_improvement = 0.0

        if len(gap_history) >= 50:
            early_avg = np.mean(gap_history[:50])
            late_avg = np.mean(gap_history[-50:])
            convergence_improvement = early_avg - late_avg
        else:
            convergence_improvement = 0.0

        target_achieved = final_ema_gap < 0.01
        if target_achieved and ema_convergence_improvement > 0.2:
            assessment = 'EXCELLENT'
            recommendation = '1E2_TARGET_ACHIEVED_EXCELLENT'
        elif target_achieved and ema_convergence_improvement > 0.15:
            assessment = 'GOOD'
            recommendation = '1E2_TARGET_ACHIEVED_GOOD'
        elif target_achieved and ema_convergence_improvement > 0.1:
            assessment = 'MODERATE'
            recommendation = '1E2_TARGET_ACHIEVED_MODERATE'
        elif target_achieved:
            assessment = 'POOR'
            recommendation = '1E2_TARGET_ACHIEVED_POOR'
        elif final_ema_gap < 0.025:
            assessment = 'CLOSE'
            recommendation = 'CLOSE_TO_1E2_REFINE'
        else:
            assessment = 'POOR'
            recommendation = 'FAR_FROM_1E2_TRY_DIFFERENT'

        return {
            'target_achieved': target_achieved,
            'convergence_improvement': convergence_improvement,
            'ema_convergence_improvement': ema_convergence_improvement,
            'final_gap': results.get('final_gap', float('inf')),
            'final_ema_gap': final_ema_gap,
            'stage_count': stage_count,
            'gap_history': gap_history[-20:],
            'ema_gap_history': ema_history[-20:],
            'config': results.get('config', {}),
            'technique': results.get('technique', {}),
            'enhancements': {
                'smooth_activation': True,
                'stochastic_averaging': True,

                'theoretical_params': True,
                'adaptive_parameter_adjustment': True,
                'N_g': results.get('config', {}).get('N_g', 1)
            },
            'assessment': assessment,
            'recommendation': recommendation
        }


def main():
    print("Starting Enhanced F2CSA Implementation...")
    solver = F2CSAAlgorithm(device='cpu', seed=42)
    # Run the runner-compatible path too:
    res = solver.optimize(max_iterations=50000, early_stopping_patience=200, target_gap=1e-2)
    print(f"Done. Final EMA Gap: {res['final_ema_gap']:.6f}, iters: {res['total_iterations']}")


if __name__ == "__main__":
    main()
