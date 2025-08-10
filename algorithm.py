
import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from problem import StronglyConvexBilevelProblem

class F2CSA:
    """F2CSA with all enhancements from successful configuration"""

    def __init__(self, problem: StronglyConvexBilevelProblem, N_g: int = 5, alpha: float = 0.3, adam_lr: float = 0.005):
        self.problem = problem
        self.N_g = N_g
        self.device = problem.device
        # Gate smoothing controls
        self.tau_factor = 5.0
        self.tau_min = 1e-3

        # Initialize variables
        self.x = torch.randn(problem.dim, device=self.device, dtype=problem.dtype) * 0.1
        self.x.requires_grad_(True)

        # Paper-exact penalty parameters (F2CSA.tex lines 360-361) - REDUCED for stability
        self.alpha = alpha
        # Paper-spec penalties (F2CSA.tex Sec. 2.2): α₁=α^{-2}, α₂=α^{-4}
        self.alpha1 = alpha**(-2)
        self.alpha2 = alpha**(-4)
        self.delta = alpha**3      # δ = α³ (inner accuracy)
        self.alpha_base = alpha

        print(f"🔧 F2CSA: Paper-compliant penalties α₁={self.alpha1:.3f}, α₂={self.alpha2:.3f}, δ={self.delta:.6f}")

        # Variance reduction: EMA for gradient smoothing (F2CSA variance control)
        self.gradient_ema = None
        self.ema_decay = 0.9  # Exponential moving average decay
        print(f"🔧 F2CSA: Variance reduction with EMA decay={self.ema_decay}")

        # Dual smoothing (variance reduction on λ~ used in ρ gates)
        self.lam_ema = None
        self.lam_beta = 0.9
        # Options for diagnostics and paper-pure limit
        self.disable_lam_ema = False
        self.actives_only_gating = False  # when True, set ρ_i=1 for active constraints, 0 otherwise
        # De-bias option at KKT-limit: penalize only inactive constraints (drop quadratic penalty on actives)
        self.penalize_inactive_only = False

        # Emergency protocols
        self.emergency_resets = 0
        self.max_emergency_resets = 3
        self.divergence_threshold = 100.0

        # Convergence acceleration
        self.momentum_beta = 0.9
        self.momentum_x = torch.zeros_like(self.x)
        self.adaptive_lr = adam_lr

        # Optimizer (outer-level)
        self.optimizer = torch.optim.Adam([self.x], lr=self.adaptive_lr)
        self.outer_optimizer_type = 'adam'

        print(f"🔧 F2CSA initialized: N_g={N_g}, α={alpha} (SUCCESSFUL config), Adam lr={self.adaptive_lr}")

    def compute_hypergradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paper oracle (F2CSA.tex Alg. Stochastic Penalty-Based Hypergradient Oracle):
          - α₁=α^{-2}, α₂=α^{-4}, δ=α^3
          - Use approximate (ŷ*, λ~) from LL KKT
          - Define smooth penalty L_{λ~,α}(x,y) and minimize w.r.t. y to δ
          - Return ∇_x L_{λ~,α}(x, ỹ(x)) averaged over N_g samples
        """
        grads = []
        n = self.problem.dim
        device = self.problem.device
        dtype = self.problem.dtype

        # Get approximate LL solution and dual from exact KKT LL solver
        y_star, info = self.problem.solve_lower_level(x)
        lam_tilde = info.get('lambda', torch.zeros(self.problem.num_constraints, device=device, dtype=dtype))
        # Enforce dual feasibility numerically: λ̃ ≥ 0 for By ≥ c inequalities
        lam_tilde = torch.clamp(lam_tilde, min=0.0)

        # Build smooth activation ρ_i(x). For linear constraints, we gate by feasibility at y* and dual sign
        with torch.no_grad():
            h_y_star = self.problem.constraints(x, y_star)
            # Base smoothing
            tau = max(float(self.tau_factor * self.delta), self.tau_min)
            if getattr(self, 'actives_only_gating', False):
                # Paper-pure limit: ρ_i = 1 on active constraints, 0 otherwise
                cons = h_y_star
                active_mask = (cons.abs() <= max(1e-8, tau))
                rho = active_mask.to(h_y_star.dtype)
            else:
                sigma_h = torch.sigmoid(h_y_star / tau)
                # Smooth λ~ with EMA to avoid spiky gating late in training (unless disabled by KKT-limit)
                if getattr(self, 'disable_lam_ema', False):
                    lam_for_gate = lam_tilde
                else:
                    if self.lam_ema is None:
                        self.lam_ema = lam_tilde.clone()
                    else:
                        self.lam_ema = self.lam_beta * self.lam_ema + (1 - self.lam_beta) * lam_tilde
                    lam_for_gate = self.lam_ema
                sigma_lam = torch.sigmoid(lam_for_gate / tau)
                rho = sigma_h * sigma_lam

            # Detect gating flip/noise and adapt smoothing on-the-fly
            rho_std = float(torch.std(rho).item())
            rho_min = float(torch.min(rho).item())
            rho_max = float(torch.max(rho).item())
            if (rho_std > 0.3 and rho_max > 0.95 and rho_min < 0.05):
                # Increase smoothing for this and future steps
                new_tau = min(20.0 * float(self.delta), tau * 1.5)
                self.tau_factor = max(self.tau_factor, new_tau / float(self.delta))
                self.lam_beta = min(0.99, max(self.lam_beta, 0.97))
                tau = max(new_tau, self.tau_min)
                sigma_h = torch.sigmoid(h_y_star / tau)
                sigma_lam = torch.sigmoid(self.lam_ema / tau)
                rho = sigma_h * sigma_lam

        # Precompute inner-system and diagnostics (constant across N_g samples)
        Q = self.problem.Q_lower
        B = self.problem.B
        A = self.problem.A
        c_lower = self.problem.c_lower
        c_upper = self.problem.c_upper
        P = self.problem.P
        diag_rho = torch.diag(rho)

        # δ-regularized inner system for numerical stability (accuracy remains within δ)
        I = torch.eye(Q.shape[0], device=Q.device, dtype=Q.dtype)
        H = self.alpha1 * Q + self.alpha2 * (B.T @ diag_rho @ B) + (self.delta) * I
        d_pen = (c_lower + P.T @ x)
        rhs = -c_upper - self.alpha1 * d_pen + self.alpha1 * (B.T @ lam_tilde) + self.alpha2 * (B.T @ (diag_rho @ (A @ x - self.problem.b)))
        y_tilde_stats = torch.linalg.solve(H, rhs)

        # Collect debug diagnostics
        try:
            H_cond = float(torch.linalg.cond(H).item())
        except Exception:
            H_cond = float('nan')
        # Extra diagnostics: ρ stats, inner feasibility at y_tilde, smoothing/EMA state
        h_y_tilde = self.problem.constraints(x, y_tilde_stats)
        max_pos_h_ytilde = float(torch.clamp(h_y_tilde, min=0).max().item())
        count_active_rho = int((rho > 0.9).sum().item())

        self.last_debug = {
            'rho_min': float(rho.min().item()),
            'rho_mean': float(rho.mean().item()),
            'rho_max': float(rho.max().item()),
            'rho_std': float(torch.std(rho).item()),
            'count_active_rho': count_active_rho,
            'lam_ema_min': float(self.lam_ema.min().item()) if self.lam_ema is not None else float('nan'),
            'lam_ema_mean': float(self.lam_ema.mean().item()) if self.lam_ema is not None else float('nan'),
            'lam_ema_max': float(self.lam_ema.max().item()) if self.lam_ema is not None else float('nan'),
            'tau': float(tau),
            'lam_beta': float(self.lam_beta),
            'H_cond': H_cond,
            'y_tilde_norm': float(torch.norm(y_tilde_stats).item()),
            'rhs_norm': float(torch.norm(rhs).item()),
            'max_pos_h_ytilde': max_pos_h_ytilde,
            'alpha1': float(self.alpha1),
            'alpha2': float(self.alpha2),
            'N_g': int(self.N_g),
        }

        for _ in range(self.N_g):
            # Clone x for gradient
            x_var = x.clone().requires_grad_(True)

            # Define L_{λ~,α}(x,y)
            def L_pen_y(y):
                # Enable stochastic objectives per requirement
                f_xy = self.problem.upper_objective(x_var, y, add_noise=True)
                # Ensure same noise realization for g(x,y) and g(x,y*) difference
                rng_state = torch.get_rng_state()
                g_xy = self.problem.lower_objective(x_var, y, add_noise=True)
                torch.set_rng_state(rng_state)
                g_xystar = self.problem.lower_objective(x_var, y_star, add_noise=True)
                h_xy = self.problem.constraints(x_var, y)
                term1 = self.alpha1 * (g_xy - g_xystar + (lam_tilde @ h_xy))
                if getattr(self, 'penalize_inactive_only', False):
                    term2 = 0.5 * self.alpha2 * torch.sum((1 - rho) * (h_xy**2))
                else:
                    term2 = 0.5 * self.alpha2 * torch.sum(rho * (h_xy**2))
                return f_xy + term1 + term2

            # Inner minimize in y to accuracy δ = α^3 (exact quadratic solve per F2CSA.tex)
            y_tilde = torch.linalg.solve(H, rhs)

            # Compute gradient w.r.t x at (x, y_tilde)
            L_val = L_pen_y(y_tilde.detach())
            g_x = torch.autograd.grad(L_val, x_var, create_graph=False, retain_graph=False)[0]
            grads.append(g_x.detach())

        raw_grad = torch.mean(torch.stack(grads), dim=0)

        # EMA smoothing (optional). If ema_decay is None or 0.0, return raw gradient.
        if getattr(self, 'ema_decay', None) in (None, 0.0):
            return raw_grad
        if self.gradient_ema is None:
            self.gradient_ema = raw_grad.clone()
        else:
            self.gradient_ema = self.ema_decay * self.gradient_ema + (1 - self.ema_decay) * raw_grad
        return self.gradient_ema

    def switch_to_sgd(self, lr: Optional[float] = None):
        """Switch outer optimizer to SGD with conservative lr."""
        if self.outer_optimizer_type != 'sgd':
            if lr is None:
                cur_lr = self.optimizer.param_groups[0]['lr']
                lr = max(cur_lr * 0.3, 1e-4)
            self.optimizer = torch.optim.SGD([self.x], lr=lr)
            self.outer_optimizer_type = 'sgd'

    def emergency_protocol(self, grad_norm: float) -> bool:
        """Emergency protocol for divergence detection"""
        if grad_norm > self.divergence_threshold and self.emergency_resets < self.max_emergency_resets:
            self.emergency_resets += 1
            self.alpha1 = self.alpha_base * 0.5
            self.alpha2 = (self.alpha_base * 0.5)**2
            self.momentum_x.zero_()
            self.adaptive_lr = max(self.adaptive_lr * 0.5, 0.001)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.adaptive_lr
            return True
        return False

    def adaptive_penalty_tuning(self, constraint_violation: float):
        """Adaptive penalty tuning in [0.2, 0.4] range"""
        if constraint_violation > 0.1:
            self.alpha1 = min(self.alpha1 * 1.05, 0.4)
            self.alpha2 = min(self.alpha2 * 1.05, 0.4**2)
        elif constraint_violation < 0.01:
            self.alpha1 = max(self.alpha1 * 0.98, 0.2)
            self.alpha2 = max(self.alpha2 * 0.98, 0.2**2)

    def optimize(self, max_iterations: int = 1000, convergence_threshold: float = 0.1) -> Dict:
        """Run F2CSA optimization"""
        start_time = time.time()
        history = []

        initial_obj = float(self.problem.true_bilevel_objective(self.x))
        best_gap = float('inf')

        for iteration in range(max_iterations):
            # Compute hypergradient
            hypergradient = self.compute_hypergradient(self.x)
            grad_norm = float(torch.norm(hypergradient))

            # Get current metrics (no emergency/fallbacks)
            _, ll_info = self.problem.solve_lower_level(self.x)

            # Compute gap and objective
            bilevel_obj = float(self.problem.true_bilevel_objective(self.x))
            gap = self.problem.compute_gap(self.x)
            best_gap = min(best_gap, gap)

            # Store history
            history.append({
                'iteration': iteration,
                'bilevel_objective': bilevel_obj,
                'gap': gap,
                'gradient_norm': grad_norm,
                'constraint_violation': ll_info['constraint_violation'],
                'alpha1': self.alpha1,
                'alpha2': self.alpha2,
                'emergency_resets': self.emergency_resets,
                'time': time.time() - start_time
            })

            # Progress reporting
            if iteration % 100 == 0:
                print(f"  F2CSA Iter {iteration:4d}: F(x)={bilevel_obj:8.4f}, Gap={gap:8.6f}, α₁={self.alpha1:.3f}")

            # Convergence check
            if gap < convergence_threshold:
                print(f"✅ F2CSA converged at iteration {iteration}: Gap = {gap:.6f}")
                break

            # Momentum-based update
            self.momentum_x = self.momentum_beta * self.momentum_x + (1 - self.momentum_beta) * hypergradient

            # Optimization step (no gradient clipping)
            self.optimizer.zero_grad()
            self.x.grad = self.momentum_x
            self.optimizer.step()

        total_time = time.time() - start_time

        return {
            'algorithm': 'F2CSA',
            'final_objective': bilevel_obj,
            'final_gap': gap,
            'best_gap': best_gap,
            'initial_objective': initial_obj,
            'objective_improvement': bilevel_obj - initial_obj,
            'emergency_resets': self.emergency_resets,
            'total_iterations': iteration + 1,
            'total_time': total_time,
            'converged': gap < convergence_threshold,
            'history': history
        }

class SSIGD:
    """
    [Stochastic] Smoothed Implicit Gradient Descent (SSIGD)

    Implementation following the paper algorithm exactly:
    - Single fixed perturbation q for LL smoothing
    - Implicit gradient computation with Jacobian
    - Stochastic objective samples
    - Proper stepsize schedules
    """

    def __init__(self, problem: StronglyConvexBilevelProblem,
                 stepsize_schedule: str = 'strongly_convex',
                 beta_0: float = 0.01,
                 ll_tolerance: float = 1e-6,
                 seed: int = 42,
                 mu_F: float = None):
        self.problem = problem
        self.stepsize_schedule = stepsize_schedule  # 'constant', 'diminishing', 'strongly_convex'
        self.beta_0 = beta_0
        self.ll_tolerance = ll_tolerance
        self.device = problem.device

        # Strong convexity parameter (estimated if not provided)
        if mu_F is None:
            # Very conservative μ_F for numerical stability with fixed implicit gradient
            # β₀ = 1/(μ_F × 1) should be small (e.g., 0.01-0.05)
            self.mu_F = 50.0  # Very conservative: β₀ = 1/50 = 0.02
        else:
            self.mu_F = mu_F

        print(f"🔧 SSIGD: μ_F={self.mu_F} → β₀={1.0/self.mu_F:.3f} (very conservative for stability)")

        # Initialize variables
        self.x = torch.randn(problem.dim, device=self.device, dtype=problem.dtype) * 0.1
        self.x.requires_grad_(True)

        # CRITICAL: Single fixed perturbation q (kept throughout training)
        # Paper: "A single draw of q suffices and is kept fixed"
        # Paper-compliant scale (fixed magnitude independent of noise level)
        torch.manual_seed(seed)
        self.q = torch.randn(problem.dim, device=self.device, dtype=problem.dtype) * 0.01
        print(f"🔧 SSIGD: Fixed perturbation q sampled with norm {torch.norm(self.q):.6f}")

        # No optimizer - we'll do manual updates with proper stepsize schedule
        print(f"🔧 SSIGD initialized: schedule={stepsize_schedule}, β₀={beta_0}, μ_F={self.mu_F}")

    def solve_perturbed_lower_level(self, x: torch.Tensor) -> torch.Tensor:
        """
        Solve perturbed lower level: min_y {g(x,y) + q^T y | constraints} using exact KKT-based LL solver (paper-compliant)
        """
        y_star, _ = self.problem.solve_lower_level(x, y_linear_offset=self.q)
        return y_star.detach()

    def compute_implicit_gradient(self, x: torch.Tensor, xi_sample: int = None) -> torch.Tensor:
        """
        Compute stochastic implicit gradient using active-set KKT adjoint (SSIGD paper).
        For quadratic LL with linear constraints, the adjoint p solves:
          [Q  B_act^T][p]   = [∇_y f]
          [B_act  0   ][μ]     [   0 ]
        and the implicit component equals −P p.
        """
        # Solve perturbed LL exactly (adds fixed q to linear term)
        y_hat = self.solve_perturbed_lower_level(x)

        # Enable gradients for implicit differentiation of UL only
        x_copy = x.clone().requires_grad_(True)
        y_copy = y_hat.clone().detach().requires_grad_(True)

        # Always use stochastic objective for gradients as requested
        f_val = self.problem.upper_objective(x_copy, y_copy, add_noise=True)

        # Direct UL gradient
        grad_x_direct = torch.autograd.grad(f_val, x_copy, create_graph=False, retain_graph=True)[0]
        grad_y = torch.autograd.grad(f_val, y_copy, create_graph=False, retain_graph=False)[0]

        # Build active-set KKT system using the exact active set from the LL KKT solver
        _, ll_info = self.problem.solve_lower_level(x, y_linear_offset=self.q)
        active_mask = ll_info.get('active_mask', None)
        if active_mask is None:
            # Fallback: detect tight constraints with tol
            cons = self.problem.constraints(x, y_hat).detach()
            act_tol = 1e-8
            active_mask = (cons.abs() <= act_tol)
        B_act = self.problem.B[active_mask, :]

        Q = self.problem.Q_lower
        n = Q.shape[0]
        k = int(B_act.shape[0])

        if k == 0:
            # Unconstrained adjoint: Q p = grad_y
            p = torch.linalg.solve(Q, grad_y)
        else:
            K = torch.zeros(n + k, n + k, device=self.problem.device, dtype=self.problem.dtype)
            K[:n, :n] = Q
            K[:n, n:] = B_act.T
            K[n:, :n] = B_act
            rhs = torch.zeros(n + k, device=self.problem.device, dtype=self.problem.dtype)
            rhs[:n] = grad_y
            sol = torch.linalg.solve(K, rhs)
            p = sol[:n]
        # Implicit component: −P p
        implicit_component = - self.problem.P @ p

        implicit_grad = grad_x_direct + implicit_component
        return implicit_grad

    def compute_moreau_envelope_gradient(self, x: torch.Tensor, rho_hat: float = 1.5) -> float:
        """
        Compute Moreau envelope gradient norm ||∇H_{1/ρ̂}(x)|| for SSIGD convergence

        Following Definition 2 from paper:
        H_λ(x) = min_z {F(z) + (1/2λ)||x-z||²}
        ||∇H_λ(x)|| = (1/λ)||x - prox_{λH}(x)||

        For λ = 1/ρ̂, we have ||∇H_{1/ρ̂}(x)|| = ρ̂||x - prox_{H/ρ̂}(x)||

        We use iterative proximal computation for better accuracy.
        """
        lambda_param = 1.0 / rho_hat

        # Iterative proximal operator computation
        # prox_{λH}(x) = argmin_z {F(z) + (1/2λ)||x-z||²}
        z = x.clone().detach().requires_grad_(True)  # Ensure it's a leaf tensor
        optimizer = torch.optim.LBFGS([z], lr=0.1, max_iter=20)

        def closure():
            optimizer.zero_grad()
            # Objective: F(z) + (1/2λ)||x-z||²
            f_z = self.problem.true_bilevel_objective(z)
            prox_term = (1.0 / (2.0 * lambda_param)) * torch.norm(x - z)**2
            total_obj = f_z + prox_term
            total_obj.backward()
            return total_obj

        optimizer.step(closure)
        prox_x = z.detach()

        # Moreau envelope gradient: ||∇H_λ(x)|| = (1/λ)||x - prox_{λH}(x)||
        moreau_grad_norm = (1.0 / lambda_param) * torch.norm(x - prox_x)

        return float(moreau_grad_norm)

    def compute_ssigd_gap(self, x: torch.Tensor, case: str = 'weakly_convex') -> float:
        """
        Compute SSIGD-specific gap according to paper convergence theorems

        Args:
            case: 'weakly_convex', 'strongly_convex', or 'convex'
        """
        if case == 'weakly_convex':
            # Use Moreau envelope gradient norm (Theorem 1)
            rho_hat = 1.5  # ρ̂ > (3/2)ρ, assuming ρ=1 for strongly convex problems
            return self.compute_moreau_envelope_gradient(x, rho_hat)

        elif case == 'strongly_convex':
            # Use objective function gap F(x) - F* (Theorem 2)
            # Since we don't know F*, use gradient norm as proxy
            return self.problem.compute_gap(x)

        elif case == 'convex':
            # Use objective function gap for ergodic average (Theorem 3)
            # For simplicity, use current point gradient norm
            return self.problem.compute_gap(x)

        else:
            # Default to standard gradient norm
            return self.problem.compute_gap(x)

    def optimize(self, max_iterations: int = 1000, convergence_threshold: float = 0.1,
                 convexity_case: str = 'strongly_convex') -> Dict:
        """
        Run SSIGD optimization following Algorithm 1 from paper exactly:

        For r = 0, 1, ..., T-1:
        1. Compute feasible ŷ(x_r) satisfying active-set/accuracy conditions
        2. Compute stochastic implicit gradient ∇F(x_r; ξ_r)
        3. Update: x_{r+1} ← proj_X(x_r - β_r * ∇F(x_r; ξ_r))
        """
        start_time = time.time()
        history = []

        initial_obj = float(self.problem.true_bilevel_objective(self.x))
        best_gap = float('inf')
        prev_obj = initial_obj  # For monotonic convergence monitoring

        print(f"🚀 SSIGD starting with fixed perturbation q (norm={torch.norm(self.q):.6f})")
        print(f"🔧 SSIGD: μ_F={self.mu_F}, initial β₀={1.0/self.mu_F:.3f}")

        for r in range(max_iterations):  # Using 'r' to match paper notation
            # Step 1: Compute stepsize β_r according to paper Theorem 2 (Line 105)
            # "If F is μ_F-strongly convex, then with β_r = 1/(μ_F(r+1))"
            if self.stepsize_schedule == 'strongly_convex':
                # Paper-exact formula: β_r = 1/(μ_F(r+1))
                beta_r = 1.0 / (self.mu_F * (r + 1))
            elif self.stepsize_schedule == 'diminishing':
                # Convex case: β_r = O(1/√T), paper line 146: β₀/√(r+1)
                beta_r = self.beta_0 / torch.sqrt(torch.tensor(r + 1.0))
            elif self.stepsize_schedule == 'constant':
                # Weakly convex case: β_r = β (constant)
                beta_r = self.beta_0
            else:
                # Default to strongly convex (our problem type)
                beta_r = 1.0 / (self.mu_F * (r + 1))

            # Step 2: Compute stochastic implicit gradient ∇F(x_r; ξ_r)
            # Sample ξ_r for this iteration (stochastic sample)
            xi_r = r  # Use iteration as stochastic sample index
            implicit_grad = self.compute_implicit_gradient(self.x, xi_sample=xi_r)
            grad_norm = torch.norm(implicit_grad)

            # No gradient clipping or artificial monotonicity enforcement

            # Step 3: Projected gradient update
            # x_{r+1} ← proj_X(x_r - β_r * ∇F(x_r; ξ_r))
            with torch.no_grad():
                # Update step
                x_new = self.x - beta_r * implicit_grad

                # Projection onto constraint set X (if needed)
                # For unconstrained case, this is identity
                self.x.copy_(x_new)

            # Compute metrics for monitoring
            _, ll_info = self.problem.solve_lower_level(self.x)
            bilevel_obj = float(self.problem.true_bilevel_objective(self.x))

            # Track objective but do not alter parameters to force monotonicity
            prev_obj = bilevel_obj

            # Use SSIGD-specific gap calculation according to paper
            gap = self.compute_ssigd_gap(self.x, case=convexity_case)
            standard_gap = self.problem.compute_gap(self.x)  # For comparison
            best_gap = min(best_gap, gap)

            # Store history
            history.append({
                'iteration': r,
                'bilevel_objective': bilevel_obj,
                'gap': gap,  # SSIGD-specific gap (Moreau envelope for weakly convex)
                'standard_gap': standard_gap,  # Standard gradient norm gap
                'gradient_norm': grad_norm,
                'stepsize': float(beta_r),
                'constraint_violation': ll_info['constraint_violation'],
                'time': time.time() - start_time
            })

            # Progress reporting
            if r % 100 == 0:
                print(f"  SSIGD Iter {r:4d}: F(x)={bilevel_obj:8.4f}, Gap={gap:8.6f} ({convexity_case}), StdGap={standard_gap:8.6f}, β={beta_r:.6f}")

            # Convergence check using SSIGD-specific gap
            if gap < convergence_threshold:
                print(f"✅ SSIGD converged at iteration {r}: {convexity_case} Gap = {gap:.6f}, Standard Gap = {standard_gap:.6f}")
                break

        total_time = time.time() - start_time

        print(f"🏁 SSIGD completed: {r+1} iterations, final {convexity_case} gap={gap:.6f}, standard gap={standard_gap:.6f}")

        return {
            'algorithm': 'SSIGD',
            'final_objective': bilevel_obj,
            'final_gap': gap,  # SSIGD-specific gap
            'final_standard_gap': standard_gap,  # Standard gradient norm gap
            'best_gap': best_gap,
            'initial_objective': initial_obj,
            'objective_improvement': bilevel_obj - initial_obj,
            'total_iterations': r + 1,
            'total_time': total_time,
            'converged': gap < convergence_threshold,
            'convexity_case': convexity_case,
            'stepsize_schedule': self.stepsize_schedule,
            'final_stepsize': float(beta_r),
            'perturbation_norm': float(torch.norm(self.q)),
            'history': history
        }

class DSBLO:
    """DS-BLO (Doubly Stochastic Bilevel Optimization) Algorithm"""

    def __init__(self, problem: StronglyConvexBilevelProblem, momentum: float = 0.9, sigma: float = 0.01,
                 gamma1: float = 1.0, gamma2: float = 0.1):
        self.problem = problem
        self.momentum = momentum  # β ∈ [0.5, 1) per paper
        self.sigma = sigma
        self.device = problem.device

        # Paper-compliant adaptive stepsize parameters (dsblo_paper.tex line 1528)
        self.gamma1 = gamma1  # γ₁ for η_t = 1/(γ₁||m_t|| + γ₂)
        self.gamma2 = gamma2  # γ₂ for η_t = 1/(γ₁||m_t|| + γ₂)

        # Initialize variables
        self.x = torch.randn(problem.dim, device=self.device, dtype=problem.dtype) * 0.1
        self.x.requires_grad_(True)

        # Momentum vector m_t (paper specification)
        self.momentum_vector = torch.zeros_like(self.x)

        # Variance reduction: Gradient smoothing for stability
        self.gradient_ema = None
        self.ema_decay = 0.95  # Higher decay for DS-BLO stability

        print(f"🔧 DS-BLO: Paper-compliant β={momentum}, σ={sigma}, γ₁={gamma1}, γ₂={gamma2}")
        print(f"🔧 DS-BLO: Variance reduction with EMA decay={self.ema_decay}")

    def solve_perturbed_lower_level(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Solve perturbed lower level: min_y {g(x,y) + q^T y | constraints} using exact KKT-based LL solver
        Keep autograd connection to x so ∂y_q*(x)/∂x contributes to the gradient.
        """
        y_star, _ = self.problem.solve_lower_level(x, y_linear_offset=q, allow_grad=True)
        return y_star  # keep graph for implicit term

    def compute_perturbed_gradient(self, x: torch.Tensor, q: torch.Tensor = None) -> torch.Tensor:
        """Compute gradient with stochastic perturbation following DS-BLO paper (no extra upper noise).
        Important: use the SAME x variable for both the LL solve and the gradient target,
        so that the implicit path ∂y*(x;q)/∂x contributes correctly.
        """
        # Use provided perturbation or sample fresh one
        if q is None:
            q = torch.randn(self.problem.dim, device=self.device, dtype=self.problem.dtype) * self.sigma

        # Ensure x has grad (do not clone for objective; keep same ref as in LL solve)
        x_var = x
        if not x_var.requires_grad:
            x_var = x_var.clone().detach().requires_grad_(True)

        # Solve perturbed lower level with autograd connection to x_var
        y_star = self.solve_perturbed_lower_level(x_var, q)

        # Compute upper objective with perturbed LL solution
        obj = self.problem.upper_objective(x_var, y_star, add_noise=False)

        # Compute gradient wrt x (includes implicit via y_star(x))
        grad = torch.autograd.grad(obj, x_var, retain_graph=False, create_graph=False)[0]
        return grad

    def compute_perturbed_gradient_with_noise(self, x: torch.Tensor, q: torch.Tensor = None) -> torch.Tensor:
        """
        Variant that includes add_noise=True in upper objective (stochastic bilevel setting).
        Keeps the same x variable for both LL solve and gradient target.
        """
        if q is None:
            q = torch.randn(self.problem.dim, device=self.device, dtype=self.problem.dtype) * self.sigma
        x_var = x
        if not x_var.requires_grad:
            x_var = x_var.clone().detach().requires_grad_(True)
        y_star = self.solve_perturbed_lower_level(x_var, q)
        obj = self.problem.upper_objective(x_var, y_star, add_noise=True)
        grad = torch.autograd.grad(obj, x_var, retain_graph=False, create_graph=False)[0]
        return grad

    def optimize(self, max_iterations: int = 1000, convergence_threshold: float = 0.1) -> Dict:
        """Run DS-BLO optimization with enhanced debug diagnostics"""
        start_time = time.time()
        history = []

        initial_obj = float(self.problem.true_bilevel_objective(self.x))
        best_gap = float('inf')
        prev_gap: Optional[float] = None
        prev_grad_norm: Optional[float] = None
        prev_obj: Optional[float] = None

        for iteration in range(max_iterations):
            # Sample fresh perturbation q_t for this iteration (paper requirement)
            q_t = torch.randn(self.problem.dim, device=self.device, dtype=self.problem.dtype) * self.sigma

            # Compute stochastic gradient ĝ_{t+1} = ∇F_{q_{t+1}}(x̄_{t+1}) with Option II (stochastic UL sample)
            raw_gradient = self.compute_perturbed_gradient_with_noise(self.x, q_t)
            raw_grad_norm = float(torch.norm(raw_gradient))

            # Variance reduction: EMA gradient smoothing (DS-BLO variance control)
            if self.gradient_ema is None:
                self.gradient_ema = raw_gradient.clone()
            else:
                self.gradient_ema = self.ema_decay * self.gradient_ema + (1 - self.ema_decay) * raw_gradient

            gradient = self.gradient_ema
            grad_norm = float(torch.norm(gradient))

            # Get current metrics at x_t
            _, ll_info = self.problem.solve_lower_level(self.x)
            bilevel_obj = float(self.problem.true_bilevel_objective(self.x))
            gap = self.problem.compute_gap(self.x)
            best_gap = min(best_gap, gap)

            # Store history
            history.append({
                'iteration': iteration,
                'bilevel_objective': bilevel_obj,
                'gap': gap,
                'gradient_norm': grad_norm,
                'raw_gradient_norm': raw_grad_norm,
                'constraint_violation': ll_info['constraint_violation'],
                'time': time.time() - start_time
            })

            # Progress reporting (coarse)
            if iteration % 100 == 0:
                print(f"  DS-BLO Iter {iteration:4d}: F(x)={bilevel_obj:8.4f}, Gap={gap:8.6f}")

            # Paper-compliant momentum update: m_{t+1} = βm_t + (1-β)ĝ_{t+1}
            self.momentum_vector = self.momentum * self.momentum_vector + (1 - self.momentum) * gradient

            # Paper-compliant adaptive stepsize: η_t = 1/(γ₁||m_t|| + γ₂)
            momentum_norm = torch.norm(self.momentum_vector)
            eta_t = 1.0 / (self.gamma1 * momentum_norm + self.gamma2)

            # Debug diagnostics (fine-grained): show when non-monotonicity occurs, esp. after 5000 iters
            increased_gap = (prev_gap is not None) and (gap > prev_gap * 1.001)
            increased_grad = (prev_grad_norm is not None) and (grad_norm > prev_grad_norm * 1.05)
            step_norm = float((eta_t * momentum_norm).item())
            q_norm = float(torch.norm(q_t).item())
            cos_g_m = float((torch.dot(gradient, self.momentum_vector) / (torch.norm(gradient) * (momentum_norm + 1e-12))).item()) if grad_norm > 0 and float(momentum_norm) > 0 else float('nan')

            debug_period = 50 if iteration >= 5000 else 250
            if (iteration % debug_period == 0) or (iteration >= 5000 and (increased_gap or increased_grad)):
                tag = "⚠️" if (iteration >= 5000 and (increased_gap or increased_grad)) else "↳"
                extra = ""
                try:
                    import os as _os
                    if _os.getenv('DSBLO_DIAG_EXTRA', '0') == '1':
                        # Compare noisy vs noiseless perturbed gradients (same q_t)
                        _g_no_noise = self.compute_perturbed_gradient(self.x, q_t)
                        _gnn_norm = float(torch.norm(_g_no_noise))
                        _cos_noisy_noiseless = float((torch.dot(raw_gradient, _g_no_noise) / (torch.norm(raw_gradient) * (torch.norm(_g_no_noise) + 1e-12))).item()) if raw_grad_norm > 0 and _gnn_norm > 0 else float('nan')
                        extra = f" |g_no_noise|={_gnn_norm:.6e} cos(noisy,noiseless)={_cos_noisy_noiseless:.4f}"
                except Exception as _e:
                    extra = f" diag_error={str(_e)[:60]}"
                print(f"{tag} dsblo-debug: it={iteration} gap={gap:.6f} Δgap={(gap - (prev_gap if prev_gap is not None else gap)):.3e} "
                      f"grad={grad_norm:.6e} (raw={raw_grad_norm:.6e}) Δgrad={(grad_norm - (prev_grad_norm if prev_grad_norm is not None else grad_norm)):.3e} "
                      f"|m|={float(momentum_norm):.6e} eta={float(eta_t):.6e} step={step_norm:.6e} q_norm={q_norm:.6e} ema={self.ema_decay:.2f} cos(g,m)={cos_g_m:.4f} "
                      f"F(x)={bilevel_obj:.6f} cv={ll_info['constraint_violation']:.2e}{extra}")

            # Convergence check
            if gap < convergence_threshold:
                print(f"✅ DS-BLO converged at iteration {iteration}: Gap = {gap:.6f}")
                break

            # Update: x_{t+1} = x_t - η_t * m_t
            with torch.no_grad():
                self.x -= eta_t * self.momentum_vector

            # Track previous values for next-iter diagnostics
            prev_gap = gap
            prev_grad_norm = grad_norm
            prev_obj = bilevel_obj

        total_time = time.time() - start_time

        return {
            'algorithm': 'DS-BLO',
            'final_objective': bilevel_obj,
            'final_gap': gap,
            'best_gap': best_gap,
            'initial_objective': initial_obj,
            'objective_improvement': bilevel_obj - initial_obj,
            'total_iterations': iteration + 1,
            'total_time': total_time,
            'converged': gap < convergence_threshold,
            'history': history
        }
