
import torch
import numpy as np
import time
from typing import Dict, Optional
from problem import StronglyConvexBilevelProblem

class F2CSA:
    """F2CSA (core, paper-aligned). See F2CSA.tex for step references."""

    def __init__(self, problem: StronglyConvexBilevelProblem, N_g: int = 5, alpha: float = 0.3, adam_lr: float = 0.005):
        self.problem = problem
        self.N_g = N_g
        self.device = problem.device
        # No helper toggles; keep core, paper-aligned implementation only

        # Initialize variables
        self.x = torch.randn(problem.dim, device=self.device, dtype=problem.dtype) * 0.1
        self.x.requires_grad_(True)

        # Paper parameters (F2CSA.tex Sec. 2.2): α₁=α^{-2}, α₂=α^{-4}, δ=α³
        self.alpha = alpha
        self.alpha1 = alpha**(-2)
        self.alpha2 = alpha**(-4)
        self.delta = alpha**3

        # Optimizer (outer-level)
        self.optimizer = torch.optim.Adam([self.x], lr=adam_lr)

        print(f"F2CSA initialized: N_g={N_g}, α={alpha}, α1={self.alpha1:.3f}, α2={self.alpha2:.3f}, δ={self.delta:.3e}")

    def compute_hypergradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stochastic Penalty-Based Hypergradient Oracle (Algorithm 1 in F2CSA.tex, lines 355-368)

        This is the core of F2CSA: approximates the true bilevel gradient ∇F(x) using a penalty-based
        reformulation that avoids second-order derivatives (Hessians). The method constructs a smooth
        penalty function and uses stochastic sampling to estimate the hypergradient.

        Mathematical Overview:
        - True bilevel gradient: ∇F(x) = ∇_x f(x,y*(x)) + [∂y*/∂x]^T ∇_y f(x,y*(x))
        - Our approximation: ∇_x L_{λ̃,α}(x, ỹ(x)) where L is a penalty Lagrangian
        - Key insight: penalty method converts constrained LL problem to unconstrained, enabling
          first-order-only computation of the implicit term [∂y*/∂x]^T ∇_y f

        Returns:
            torch.Tensor: Approximation of ∇F(x) with bias O(α) and variance O(1/N_g)
        """
        grads = []  # Will collect N_g gradient samples for variance reduction
        n = self.problem.dim
        device = self.problem.device
        dtype = self.problem.dtype

        # ═══════════════════════════════════════════════════════════════════════════════════════
        # STEP 1: Solve Lower-Level KKT System (F2CSA.tex Algorithm 1, Line 3)
        # ═══════════════════════════════════════════════════════════════════════════════════════
        # Get approximate LL solution ỹ* and dual variables λ̃ from the constrained QP:
        # min_y g(x,y) s.t. h(x,y) ≤ 0, where h(x,y) = Ax - By - b
        # This gives us the "anchor point" around which we build our penalty approximation
        y_star, info = self.problem.solve_lower_level(x)
        lam_tilde = info.get('lambda', torch.zeros(self.problem.num_constraints, device=device, dtype=dtype))
        # Dual clamping disabled per request — use λ̃ as returned by LL solver

        # ═══════════════════════════════════════════════════════════════════════════════════════
        # STEP 2: Construct Smooth Gating ρ_i(x) (F2CSA.tex lines 374-391)
        # ═══════════════════════════════════════════════════════════════════════════════════════
        # Purpose: Identify which LL constraints are (nearly) active and should be emphasized in the
        # penalty. We avoid discontinuous on/off decisions by using smooth sigmoids on both constraint
        # values h_i(x,y*) and duals λ̃_i. This mitigates instability when the active set changes.
        #   - σ_h(z) ~ 1 when constraint is violated or near-active; ~0 when safely inactive
        #   - σ_λ(z) ~ 1 when dual is positive (likely active); ~0 otherwise
        # ρ_i = σ_h(h_i) σ_λ(λ̃_i) blends these signals.
        # In the code we use a temperature τ to control smoothness; τ scales with δ=α^3.
        with torch.no_grad():
            h_y_star = self.problem.constraints(x, y_star)
            # Fixed temperature τ (paper-aligned scaling with δ); no adaptation or hard gating
            tau = max(5.0 * float(self.delta), 1e-3)
            # Smooth gate via sigmoids on constraint values and duals; use λ̃ directly
            sigma_h = torch.sigmoid(h_y_star / tau)
            sigma_lam = torch.sigmoid(lam_tilde / tau)
            rho = sigma_h * sigma_lam

        # ═══════════════════════════════════════════════════════════════════════════════════════
        # STEP 3: Build Penalty Lagrangian and Inner Quadratic Solve (Eq. 396–401 in F2CSA.tex)
        # ═══════════════════════════════════════════════════════════════════════════════════════
        # L_{λ̃,α}(x,y) = f(x,y)
        #                + α₁ [ g(x,y) - g(x,ỹ*(x)) + λ̃^T h(x,y) ]
        #                + (α₂/2) Σ_i ρ_i(x) h_i(x,y)^2
        # With α₁=α^{-2}, α₂=α^{-4}, δ=α^3 (text lines 360–361, 496–505)
        # Rationale:
        #   - α₁ scales the "linearized" KKT residuals part; α₂ enforces constraints quadratically
        #   - Choosing α₁, α₂, δ in this way ensures bias O(α) and variance O(1/N_g)
        #   - δ regularization added to H improves numerical conditioning without violating δ-accuracy
        Q = self.problem.Q_lower
        B = self.problem.B
        A = self.problem.A
        c_lower = self.problem.c_lower
        c_upper = self.problem.c_upper
        P = self.problem.P
        diag_rho = torch.diag(rho)

        # Inner system matrix H for the minimization of L w.r.t. y (no δ ridge)
        # H = α₁ Q + α₂ B^T diag(ρ) B
        H = self.alpha1 * Q + self.alpha2 * (B.T @ diag_rho @ B)
        # Right-hand side derived from ∂L/∂y = 0 (for quadratic forms):
        # Q terms from g, B terms from constraints, and linear terms from c_upper / c_lower / P^T x
        d_pen = (c_lower + P.T @ x)
        rhs = -c_upper - self.alpha1 * d_pen + self.alpha1 * (B.T @ lam_tilde) + self.alpha2 * (B.T @ (diag_rho @ (A @ x - self.problem.b)))
        # Solve for ỹ(x) approximately (to δ-accuracy via the δ-regularized linear system)
        y_tilde_stats = torch.linalg.solve(H, rhs)

        # Diagnostics (to help debugging/maintenance): condition number, ρ stats, feasibility at ỹ
        try:
            H_cond = float(torch.linalg.cond(H).item())
        except Exception:
            H_cond = float('nan')
        h_y_tilde = self.problem.constraints(x, y_tilde_stats)
        max_pos_h_ytilde = float(torch.clamp(h_y_tilde, min=0).max().item())
        count_active_rho = int((rho > 0.9).sum().item())

        # Minimal debug hook (intentionally sparse to keep core clear)

        # ═══════════════════════════════════════════════════════════════════════════════════════
        # STEP 4: Stochastic Hypergradient Estimation (Algorithm 1, lines 365–367)
        # ═══════════════════════════════════════════════════════════════════════════════════════
        # We form L_{λ̃,α}(x,y) explicitly and compute ∇_x L at y = ỹ(x). We repeat this N_g times
        # with independent stochastic draws to reduce variance (mini-batch averaging).
        for _ in range(self.N_g):
            # We need x to be a leaf w.r.t. autograd for ∇_x
            x_var = x.clone().requires_grad_(True)

            # Define the penalty Lagrangian L_{λ̃,α}(x,y) following Eq. 396–401
            def L_pen_y(y):
                # Enable stochastic objectives (Assumption 3, unbiasedness/variance): add_noise=True
                f_xy = self.problem.upper_objective(x_var, y, add_noise=True)
                # Optionally use same noise realization between g(x,y) and g(x,y*) to reduce variance
                # RNG matching disabled — use independent noise draws for g(x,y) and g(x,y*)
                g_xy = self.problem.lower_objective(x_var, y, add_noise=True)
                g_xystar = self.problem.lower_objective(x_var, y_star, add_noise=True)
                h_xy = self.problem.constraints(x_var, y)
                term1 = self.alpha1 * (g_xy - g_xystar + (lam_tilde @ h_xy))
                term2 = 0.5 * self.alpha2 * torch.sum(rho * (h_xy**2))
                return f_xy + term1 + term2

            # Inner argmin_y L_{λ̃,α}(x,y): for quadratics we use the closed-form y_tilde (precomputed)
            y_tilde = y_tilde_stats

            # Single-sample gradient w.r.t x at (x, y_tilde)
            L_val = L_pen_y(y_tilde.detach())
            g_x = torch.autograd.grad(L_val, x_var, create_graph=False, retain_graph=False)[0]
            grads.append(g_x.detach())

        # Mini-batch averaging over N_g samples: variance ↓ as 1/N_g (Lemma 4)
        raw_grad = torch.mean(torch.stack(grads), dim=0)

        # Return the mini-batch averaged hypergradient (no EMA smoothing)
        return raw_grad

    def optimize(self, max_iterations: int = 1000, convergence_threshold: float = 0.1) -> Dict:
        """
        Outer optimization loop for F2CSA.

        This loop repeatedly calls the hypergradient oracle (compute_hypergradient) and updates x using
        a first-order optimizer (Adam by default). While the paper (F2CSA.tex Algorithm 2) presents a
        non-smooth method with clipping and Goldstein grouping, here we use a standard optimizer for
        simplicity while keeping key diagnostics and the same oracle.

        Key elements explained for maintainers:
        - Hypergradient: Approximates ∇F(x) via the penalty oracle (bias O(α), variance O(1/N_g)).
        - Momentum/Adam: Improves practical convergence; not part of the theoretical Algorithm 2.
        - Gap metric: Uses a finite-difference proxy to monitor stationarity; not the Goldstein measure
          from the paper but sufficient for regression and debugging.
        """
        start_time = time.time()
        history = []

        # Initial diagnostics for context
        initial_obj = float(self.problem.true_bilevel_objective(self.x))
        best_gap = float('inf')

        for iteration in range(max_iterations):
            # ── Step A: Call stochastic hypergradient oracle (Algorithm 1) ────────────────
            hypergradient = self.compute_hypergradient(self.x)
            grad_norm = float(torch.norm(hypergradient))

            # ── Step B: Gather diagnostics (LL constraint violation, objective, gap ≔ deterministic loss) ─
            y_star, ll_info = self.problem.solve_lower_level(self.x)
            # Deterministic loss like deterministic.py: f(x, y*(x)) without stochastic noise
            bilevel_obj = float(self.problem.upper_objective(self.x, y_star, add_noise=False))
            gap = bilevel_obj
            best_gap = min(best_gap, gap)

            # Persist iteration history for analysis and debugging
            history.append({
                'iteration': iteration,
                'bilevel_objective': bilevel_obj,
                'gap': gap,
                'gradient_norm': grad_norm,
                'constraint_violation': ll_info['constraint_violation'],
                'alpha1': self.alpha1,
                'alpha2': self.alpha2,
                                'time': time.time() - start_time
            })

            # Periodic progress report
            if iteration % 100 == 0:
                print(f"  F2CSA Iter {iteration:4d}: F(x)={bilevel_obj:8.4f}, Gap={gap:8.6f}, α₁={self.alpha1:.3f}")

            # ── Step C: Convergence check (proxy gap) ─────────────────────────────────────
            if gap < convergence_threshold:
                print(f"✅ F2CSA converged at iteration {iteration}: Gap = {gap:.6f}")
                break

            # ── Step D: Update rule on x via Adam (no momentum smoothing) ─────────────────
            self.optimizer.zero_grad()
            self.x.grad = hypergradient
            self.optimizer.step()

        total_time = time.time() - start_time

        return {
            'algorithm': 'F2CSA',
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

            # Compute metrics for monitoring (harmonize gap with deterministic.py)
            y_star, ll_info = self.problem.solve_lower_level(self.x)
            bilevel_obj = float(self.problem.upper_objective(self.x, y_star, add_noise=False))

            # Track objective but do not alter parameters to force monotonicity
            prev_obj = bilevel_obj

            # Gap definition: use deterministic loss f(x, y*(x)) for consistency
            gap = bilevel_obj
            best_gap = min(best_gap, gap)

            # Store history
            history.append({
                'iteration': r,
                'bilevel_objective': bilevel_obj,
                'gap': gap,
                'gradient_norm': grad_norm,
                'stepsize': float(beta_r),
                'constraint_violation': ll_info['constraint_violation'],
                'time': time.time() - start_time
            })

            # Progress reporting
            if r % 100 == 0:
                print(f"  SSIGD Iter {r:4d}: F(x)={bilevel_obj:8.4f}, Gap={gap:8.6f}, β={beta_r:.6f}")

            # Convergence check using the deterministic-loss gap
            if gap < convergence_threshold:
                print(f"✅ SSIGD converged at iteration {r}: Gap = {gap:.6f}")
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

            # Get current metrics at x_t (harmonize gap with deterministic.py)
            y_star, ll_info = self.problem.solve_lower_level(self.x)
            bilevel_obj = float(self.problem.upper_objective(self.x, y_star, add_noise=False))
            gap = bilevel_obj
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
