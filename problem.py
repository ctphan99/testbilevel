import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import cvxpy as cp

class StronglyConvexBilevelProblem:
    """
    Strongly convex constrained bilevel optimization problem for fair algorithm comparison with enforced strong convexity

    Upper level: min_x F(x,noise) = f(x, y*(x),noise)
    Lower level: y*(x) ∈ argmin_y g(x,y,noise) s.t. h(x,y) ≤ 0

    where:
    - f(x,y) = 0.5 * (x - x_target)^T Q_upper (x - x_target) + c_upper^T y + noise
    - g(x,y) = 0.5 * y^T Q_lower y + (c_lower + P^T x)^T y + noise
    - h(x,y) = Ax - By - b ≤ 0 (linear constraints)
    """

    def __init__(self, dim: int = 100, num_constraints: int = 3, noise_std: float = 0.01, device: str = 'cpu', seed: int = 42, noise_type: str = 'gaussian'):
        self.dim = dim
        self.num_constraints = num_constraints
        self.noise_std = noise_std
        self.noise_type = noise_type  # 'gaussian', 'adversarial', 'custom'
        self.device = device
        self.dtype = torch.float64
        torch.set_default_dtype(self.dtype)


        torch.manual_seed(seed)
        np.random.seed(seed)

        # STRONG CONVEXITY: Critical for F2CSA
        # Upper level parameters with strong convexity
        noise_scale = 0.01 / np.sqrt(dim)
        self.Q_upper = (torch.randn(dim, dim, device=device, dtype=self.dtype) * noise_scale)
        self.Q_upper = self.Q_upper + self.Q_upper.T  # Symmetric
        self.Q_upper += torch.eye(dim, device=device, dtype=self.dtype) * 2.0  # Strong convexity

        # Lower level parameters with strong convexity
        self.Q_lower = (torch.randn(dim, dim, device=device, dtype=self.dtype) * noise_scale)
        self.Q_lower = self.Q_lower + self.Q_lower.T  # Symmetric
        self.Q_lower += torch.eye(dim, device=device, dtype=self.dtype) * 2.0  # Strong convexity

        # Other parameters
        param_scale = 0.1 / np.sqrt(dim)
        self.c_upper = torch.randn(dim, device=device, dtype=self.dtype) * param_scale
        self.c_lower = torch.randn(dim, device=device, dtype=self.dtype) * param_scale
        self.P = torch.randn(dim, dim, device=device, dtype=self.dtype) * param_scale  # Coupling matrix
        self.x_target = torch.randn(dim, device=device, dtype=self.dtype) * 0.1

        # ROBUST FEASIBLE constraint generation: h(x,y) = Ax - By - b ≤ 0
        # Use much smaller coefficients and ensure feasibility at origin
        constraint_scale = 0.05 / np.sqrt(dim)  # Even smaller for better conditioning
        self.A = torch.randn(num_constraints, dim, device=device, dtype=self.dtype) * constraint_scale
        self.B = torch.randn(num_constraints, dim, device=device, dtype=self.dtype) * constraint_scale

        # GUARANTEE feasibility: Set b such that origin satisfies constraints with large slack
        # At origin: Ax - By = 0, so we need b > 0 for feasibility
        self.b = torch.abs(torch.randn(num_constraints, device=device, dtype=self.dtype)) * 0.2 + 0.1  # Always positive

        # Verify feasibility at origin
        origin_constraint = -self.b  # Since Ax - By = 0 at origin
        max_violation = torch.max(origin_constraint)
        print(f"Constraint feasibility at origin: max_violation = {max_violation:.6f} (should be <= 0)")

        # Verify strong convexity
        upper_eigenvals = torch.linalg.eigvals(self.Q_upper).real
        lower_eigenvals = torch.linalg.eigvals(self.Q_lower).real

        print(f"Strongly Convex Bilevel Problem (dim={dim}, constraints={num_constraints})")
        print(f"Upper level strong convexity: lambda_min={upper_eigenvals.min():.3f}, lambda_max={upper_eigenvals.max():.3f}")
        print(f"Lower level strong convexity: lambda_min={lower_eigenvals.min():.3f}, lambda_max={lower_eigenvals.max():.3f}")
        print(f"Condition numbers: Upper={upper_eigenvals.max()/upper_eigenvals.min():.2f}, Lower={lower_eigenvals.max()/lower_eigenvals.min():.2f}")

    def upper_objective(self, x: torch.Tensor, y: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """Upper level objective with optional Gaussian noise"""
        term1 = 0.5 * (x - self.x_target) @ self.Q_upper @ (x - self.x_target)
        term2 = self.c_upper @ y

        if add_noise:
            noise = torch.randn_like(x) * self.noise_std
            return term1 + term2 + noise.sum()
        return term1 + term2

    def lower_objective(self, x: torch.Tensor, y: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """Lower level objective with optional Gaussian noise"""
        term1 = 0.5 * y @ self.Q_lower @ y
        term2 = (self.c_lower + self.P.T @ x) @ y

        if add_noise:
            noise = torch.randn_like(y) * self.noise_std
            return term1 + term2 + noise.sum()
        return term1 + term2

    def constraints(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Linear constraints h(x,y) = Ax - By - b"""
        return self.A @ x - self.B @ y - self.b

    def solve_lower_level(self, x: torch.Tensor, max_iter: int = 200, tol: float = 1e-8,
                           y_linear_offset: Optional[torch.Tensor] = None, allow_grad: bool = False,
                           batch_size: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Solve constrained lower level QP using CVXPy (no fallbacks)

        Problem: min_y 0.5 * y^T Q_lower y + (c_lower + P^T x + (y_linear_offset))^T y
                 s.t.  By ≥ Ax - b

        If y_linear_offset is provided (e.g., perturbation q), it is added to the linear term.
        If batch_size is provided, inject stochasticity by sampling the linear term (mini-batch effect).
        Constraints remain deterministic.
        """
        # Detach x unless explicit gradient-through-solver is requested (not supported here)
        x_det = x.detach()

        # Prepare problem data as numpy
        Q = (self.Q_lower.detach().cpu().numpy() + self.Q_lower.detach().cpu().numpy().T) / 2.0
        d_base = (self.c_lower + self.P.T @ x_det + (y_linear_offset if y_linear_offset is not None else 0.0)).detach().cpu().numpy()
        B = self.B.detach().cpu().numpy()
        c_vec = (self.A @ x_det - self.b).detach().cpu().numpy()

        # Define and solve QP (parameterized if batch_size provided)
        y_var = cp.Variable(self.dim)
        d_param = cp.Parameter(self.dim)
        objective = cp.Minimize(0.5 * cp.quad_form(y_var, Q) + d_param.T @ y_var)
        constraints = [B @ y_var >= c_vec]
        problem = cp.Problem(objective, constraints)
        import time as _time
        _t0 = _time.time()
        if batch_size is None:
            d_param.value = d_base
        else:
            noise_scale = float(self.noise_std) / max(1.0, np.sqrt(batch_size))
            d_param.value = d_base + np.random.randn(*d_base.shape) * noise_scale
        problem.solve(solver=cp.SCS, verbose=False)
        _solve_time = _time.time() - _t0

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"CVXPy LL solve failed: status={problem.status}")

        y_star = torch.tensor(y_var.value, device=self.device, dtype=self.dtype)
        lambda_dual = torch.tensor(constraints[0].dual_value, device=self.device, dtype=self.dtype)

        # Diagnostics info similar to previous outputs
        constraint_violation = float(torch.norm(torch.clamp(torch.tensor(c_vec) - torch.tensor(B) @ y_star.cpu(), min=0)))
        stationarity_residual = (self.Q_lower @ y_star + (self.c_lower + self.P.T @ x_det)) - self.B.T @ lambda_dual
        optimality_gap = float(torch.norm(stationarity_residual))
        info = {
            'iterations': 1,
            'status': problem.status,
            'converged': problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and constraint_violation < tol,
            'constraint_violation': constraint_violation,
            'optimality_gap': optimality_gap,
            'method': 'cvxpy_scs',
            'lambda': lambda_dual.detach(),
            'active_mask': (self.B @ y_star - (self.A @ x_det - self.b)).abs() <= tol,
            'solve_time_sec': _solve_time
        }
        return y_star, info

    def _solve_kkt_analytical(self, Q: torch.Tensor, d: torch.Tensor, B: torch.Tensor, c: torch.Tensor, tol: float) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Solve QP using analytical KKT conditions for small problems

        KKT system:
        Q y - B^T λ = -d
        By ≥ c, λ ≥ 0, λ^T(By - c) = 0
        """
        n_vars = Q.shape[0]
        n_constraints = B.shape[0]

        # Try all possible active sets (2^m possibilities)
        best_y = None
        best_lambda = None
        best_objective = float('inf')

        for active_mask in range(2**n_constraints):
            try:
                # Determine which constraints are active
                active_constraints = []
                for i in range(n_constraints):
                    if (active_mask >> i) & 1:
                        active_constraints.append(i)

                if len(active_constraints) == 0:
                    # No active constraints - unconstrained solution
                    y = -torch.linalg.solve(Q, d)
                    lambda_full = torch.zeros(n_constraints, device=self.device)
                else:
                    # Active constraints: B_active y = c_active
                    B_active = B[active_constraints, :]
                    c_active = c[active_constraints]

                    # Solve KKT system: [Q  -B_active^T] [y     ] = [-d      ]
                    #                   [B_active  0  ] [lambda]   [c_active]

                    kkt_matrix = torch.zeros(n_vars + len(active_constraints), n_vars + len(active_constraints), device=self.device)
                    kkt_matrix[:n_vars, :n_vars] = Q
                    kkt_matrix[:n_vars, n_vars:] = -B_active.T
                    kkt_matrix[n_vars:, :n_vars] = B_active

                    kkt_rhs = torch.zeros(n_vars + len(active_constraints), device=self.device)
                    kkt_rhs[:n_vars] = -d
                    kkt_rhs[n_vars:] = c_active

                    # Solve the system
                    kkt_solution = torch.linalg.solve(kkt_matrix, kkt_rhs)
                    y = kkt_solution[:n_vars]
                    lambda_active = kkt_solution[n_vars:]

                    # Reconstruct full lambda vector
                    lambda_full = torch.zeros(n_constraints, device=self.device)
                    lambda_full[active_constraints] = lambda_active

                # Check KKT conditions
                # 1. Stationarity: Q y - B^T λ + d = 0
                stationarity_residual = Q @ y - B.T @ lambda_full + d
                stationarity_error = torch.norm(stationarity_residual)

                # 2. Primal feasibility: By ≥ c
                constraint_values = B @ y - c
                primal_violation = torch.norm(torch.clamp(-constraint_values, min=0))

                # 3. Dual feasibility: λ ≥ 0
                dual_violation = torch.norm(torch.clamp(-lambda_full, min=0))

                # 4. Complementary slackness: λ^T(By - c) = 0
                complementarity_error = torch.abs(lambda_full @ constraint_values)

                # Check if this is a valid KKT point
                kkt_error = stationarity_error + primal_violation + dual_violation + complementarity_error

                if kkt_error < tol:
                    # Valid KKT point - compute objective
                    objective = 0.5 * y @ Q @ y + d @ y

                    if objective < best_objective:
                        best_objective = objective
                        best_y = y.clone()
                        best_lambda = lambda_full.clone()

            except Exception:
                # This active set didn't work, try next one
                continue

        if best_y is not None:
            return best_y, best_lambda, True
        else:
            return torch.zeros(n_vars, device=self.device), torch.zeros(n_constraints, device=self.device), False

    def _solve_projected_gradient(self, Q: torch.Tensor, d: torch.Tensor, B: torch.Tensor, c: torch.Tensor, max_iter: int, tol: float) -> Tuple[torch.Tensor, Dict]:
        """
        Solve QP using projected gradient descent for larger problems
        """
        n_vars = Q.shape[0]

        # Initialize at unconstrained optimum
        y = -torch.linalg.solve(Q, d)

        # Project onto feasible region
        y = self._project_onto_constraints(y, B, c)

        lr = 0.01

        for i in range(max_iter):
            # Gradient step
            grad = Q @ y + d
            y_new = y - lr * grad

            # Project onto constraints
            y = self._project_onto_constraints(y_new, B, c)

            # Check convergence
            grad_norm = torch.norm(grad)
            constraint_violation = float(torch.norm(torch.clamp(c - B @ y, min=0)))

            if grad_norm < tol and constraint_violation < tol:
                break

        # Active-set KKT refinement for exact stationarity
        act_tol = max(1e-10, tol)
        y_kkt = y.clone()
        By_minus_c = B @ y_kkt - c
        active_mask = (By_minus_c.abs() <= act_tol)

        # Ensure feasibility strictly
        viol = (c - B @ y_kkt)
        if torch.any(viol > act_tol):
            # Add most violated constraints
            add_idx = torch.argmax(viol)
            active_mask[add_idx] = True

        max_refine_iters = 15
        lam_full = torch.zeros(B.shape[0], device=B.device)
        for _ in range(max_refine_iters):
            active_idx = torch.where(active_mask)[0]
            k = int(active_idx.numel())
            if k == 0:
                # Unconstrained step
                y_candidate = -torch.linalg.solve(Q, d)
                # Project if violates
                if torch.any(c - B @ y_candidate > act_tol):
                    worst = torch.argmax(c - B @ y_candidate)
                    active_mask[worst] = True
                    continue
                y_kkt = y_candidate
                lam_full.zero_()
                break

            B_act = B[active_idx, :]
            # Build KKT system
            K = torch.zeros(Q.shape[0] + k, Q.shape[1] + k, device=Q.device)
            K[:Q.shape[0], :Q.shape[1]] = Q
            K[:Q.shape[0], Q.shape[1]:] = -B_act.T
            K[Q.shape[0]:, :Q.shape[1]] = B_act
            rhs = torch.zeros(Q.shape[0] + k, device=Q.device)
            rhs[:Q.shape[0]] = -d
            rhs[Q.shape[0]:] = c[active_idx]

            sol = torch.linalg.solve(K, rhs)
            y_new = sol[:Q.shape[0]]
            lam_act = sol[Q.shape[0]:]

            # Enforce dual feasibility
            if torch.any(lam_act < -1e-12):
                # Drop most negative multiplier and retry
                drop = active_idx[torch.argmin(lam_act)]
                active_mask[drop] = False
                continue

            y_kkt = y_new
            lam_full.zero_()
            lam_full[active_idx] = lam_act

            # Check primal feasibility and complementarity
            By_minus_c = B @ y_kkt - c
            if torch.any(By_minus_c < -act_tol):
                # Add most violated constraint
                add_idx = torch.argmin(By_minus_c)
                active_mask[add_idx] = True
                continue

            # Stable active set found
            break

        # Final residuals
        constraint_violation = float(torch.norm(torch.clamp(c - B @ y_kkt, min=0)))
        stationarity_residual = (Q @ y_kkt + d) - B.T @ lam_full
        kkt_residual = float(torch.norm(stationarity_residual))

        info = {
            'iterations': i + 1,
            'converged': (constraint_violation < tol) and (kkt_residual < tol),
            'constraint_violation': constraint_violation,
            'optimality_gap': kkt_residual,
            'method': 'projected_gradient_active_set_kkt',
            'lambda': lam_full.detach(),
            'active_mask': active_mask.detach()
        }

        return y_kkt, info

    def _project_onto_constraints(self, y: torch.Tensor, B: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Project y onto the feasible region {y : By ≥ c}
        """
        # Simple projection: if constraints are violated, move towards feasibility
        constraint_values = B @ y - c
        violations = torch.clamp(-constraint_values, min=0)

        if torch.norm(violations) < 1e-10:
            return y  # Already feasible

        # Move in direction of constraint normals to restore feasibility
        correction = torch.zeros_like(y)
        for i in range(B.shape[0]):
            if violations[i] > 0:
                # Move in direction of B[i] to satisfy constraint i
                correction += violations[i] * B[i] / (torch.norm(B[i])**2 + 1e-10)

        return y + correction

    def true_bilevel_objective(self, x: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """Compute true bilevel objective F(x) = f(x, y*(x))"""
        y_star, _ = self.solve_lower_level(x)
        return self.upper_objective(x, y_star, add_noise=add_noise)



    def compute_gap(self, x: torch.Tensor) -> float:
        """
        Compute exact bilevel optimality gap using KKT-based LL solution
        Gap = ||∇_x f(x,y*(x)) + [∇y*(x)]^T ∇_y f(x,y*(x))||
        """
        # Solve lower level exactly using KKT conditions
        y_star, ll_info = self.solve_lower_level(x)

        # Report LL solution quality but proceed without fallbacks
        constraint_violation = ll_info.get('constraint_violation', 0)
        ll_converged = ll_info.get('converged', False)
        if not ll_converged or constraint_violation > 1e-6:
            print(f"   ⚠️  LL solution quality: conv={ll_converged}, viol={constraint_violation:.2e}")

        # Compute direct gradient ∇_x f(x,y*)
        x_copy = x.clone().requires_grad_(True)
        y_copy = y_star.clone().requires_grad_(True)

        f_val = self.upper_objective(x_copy, y_copy, add_noise=False)
        grad_x_direct = torch.autograd.grad(f_val, x_copy, create_graph=True, retain_graph=True)[0]
        grad_y = torch.autograd.grad(f_val, y_copy, create_graph=True, retain_graph=True)[0]

        # Compute implicit gradient component ∇y*(x) using finite differences
        eps = 1e-5
        implicit_component = torch.zeros_like(grad_x_direct)

        for i in range(x.shape[0]):
            # Perturb x in direction i
            x_pert = x.clone()
            x_pert[i] += eps
            y_pert, _ = self.solve_lower_level(x_pert)

            # Finite difference approximation of ∂y*/∂x_i
            dy_dxi = (y_pert - y_star) / eps

            # Add contribution to implicit gradient
            implicit_component[i] = torch.dot(dy_dxi, grad_y)

        # Total bilevel gradient
        total_grad = grad_x_direct + implicit_component
        gap_value = float(torch.norm(total_grad))

        print(f"   ✅ Gap: {gap_value:.6f} (direct: {torch.norm(grad_x_direct):.6f}, implicit: {torch.norm(implicit_component):.6f})")

        return gap_value

    def jacobian_dy_dx_if_inactive(self) -> torch.Tensor:
        """
        Closed-form Jacobian ∂y*/∂x for the inactive-constraints regime.
        For g(x,y) = 0.5 y^T Q_lower y + (c_lower + P^T x)^T y, unconstrained optimum is
        y*(x) = -Q_lower^{-1}(c_lower + P^T x) so J = ∂y*/∂x = -Q_lower^{-1} P^T.
        """
        # Solve Q_lower * Z = P^T  =>  Z = Q_lower^{-1} P^T
        Z = torch.linalg.solve(self.Q_lower, self.P.T)
        return -Z

    def implicit_vector_if_inactive(self) -> torch.Tensor:
        """
        Analytical implicit gradient component v = J^T ∇_y f(x,y) with J from inactive regime.
        Since ∇_y f(x,y) = c_upper (constant), v = J^T c_upper = -(P Q_lower^{-1}) c_upper.
        """
        J = self.jacobian_dy_dx_if_inactive()
        return J.T @ self.c_upper

    def stationary_x_star_if_inactive(self) -> torch.Tensor:
        """
        Stationary point solving Q_upper (x - x_target) + v = 0 with v = implicit_vector_if_inactive().
        x* = x_target - Q_upper^{-1} v
        """
        v = self.implicit_vector_if_inactive()
        # Solve Q_upper * s = v  =>  s = Q_upper^{-1} v
        s = torch.linalg.solve(self.Q_upper, v)
        return self.x_target - s

    def print_stationary_analysis(self) -> None:
        """
        Print a concise analytical report explaining the observed constant implicit norm
        and the true stationary point under inactive constraints.
        """
        J = self.jacobian_dy_dx_if_inactive()
        v = J.T @ self.c_upper  # implicit vector
        v_norm = float(torch.norm(v))

        x_star = self.stationary_x_star_if_inactive()
        # Direct part of ∇F at x is Q_upper(x - x_target)
        direct_vec = self.Q_upper @ (x_star - self.x_target)
        direct_norm = float(torch.norm(direct_vec))
        total_vec = direct_vec + v
        total_norm = float(torch.norm(total_vec))

        # Check constraints at (x*, y*(x*)) to confirm inactivity
        y_star, ll_info = self.solve_lower_level(x_star)
        slacks = self.B @ y_star - (self.A @ x_star - self.b)
        min_slack = float(torch.min(slacks))
        cv = float(ll_info.get('constraint_violation', 0.0))

        print("\n🔎 Stationary analysis (inactive-constraints model)")
        print(f"   Expected implicit norm ||J^T c_upper|| = {v_norm:.6f}")
        print(f"   x* solves Q_upper(x - x_target) + v = 0")
        print(f"   At x*: ||direct|| = {direct_norm:.6f}, ||direct + implicit|| = {total_norm:.6f}")
        print(f"   Constraint check at x*: CV={cv:.2e}, min_slack={min_slack:.3e} (inactive if > 0)")


    def compute_true_bilevel_gradient(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the total bilevel gradient at x and its components (direct, implicit).
        total = ∇_x f(x,y*(x)) + [∂y*/∂x]^T ∇_y f(x,y*(x))
        """
        # Solve lower level exactly using KKT conditions
        y_star, _ = self.solve_lower_level(x)
        # Direct component
        x_copy = x.clone().requires_grad_(True)
        y_copy = y_star.clone().requires_grad_(True)
        f_val = self.upper_objective(x_copy, y_copy, add_noise=False)
        grad_x_direct = torch.autograd.grad(f_val, x_copy, create_graph=True, retain_graph=True)[0]
        grad_y = torch.autograd.grad(f_val, y_copy, create_graph=True, retain_graph=True)[0]
        # Finite-difference implicit Jacobian-vector product
        eps = 1e-5
        implicit_component = torch.zeros_like(grad_x_direct)
        for i in range(x.shape[0]):
            x_pert = x.clone()
            x_pert[i] += eps
            y_pert, _ = self.solve_lower_level(x_pert)
            dy_dxi = (y_pert - y_star) / eps
            implicit_component[i] = torch.dot(dy_dxi, grad_y)
        total_grad = grad_x_direct + implicit_component
        return total_grad.detach(), grad_x_direct.detach(), implicit_component.detach()


    # -------------------------------------------------------------------------
    # F2CSA Algorithm 1: Complete penalty Lagrangian solver
    # -------------------------------------------------------------------------
    def solve_f2csa_penalty_lagrangian(self, x: torch.Tensor, y_star: torch.Tensor,
                                       lam_tilde: torch.Tensor, rho: torch.Tensor,
                                       alpha1: float, alpha2: float,
                                       batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Solve the F2CSA penalty Lagrangian minimization over y using CVXPy.

        Implements Eq. 396-401: L_{λ̃,α}(x,y) = f(x,y) + α₁[g(x,y) - g(x,y*) + λ̃ᵀh(x,y)] + (α₂/2)Σᵢρᵢhᵢ(x,y)²
        Subject to: h(x,y) ≤ 0 (original constraints remain active)

        Args:
            x: Current upper-level variable
            y_star: Reference lower-level solution y*(x)
            lam_tilde: Reference dual variables λ̃(x)
            rho: Constraint gating weights ρᵢ(x)
            alpha1, alpha2: F2CSA penalty parameters

        Returns:
            ỹ(x): Optimal y minimizing the penalty Lagrangian
        """
        # Convert to numpy for CVXPy
        x_np = x.detach().cpu().numpy()
        y_star_np = y_star.detach().cpu().numpy()
        lam_np = lam_tilde.detach().cpu().numpy()
        rho_np = rho.detach().cpu().numpy()

        # Problem matrices (symmetrized for numerical stability)
        Ql = (self.Q_lower.detach().cpu().numpy() + self.Q_lower.detach().cpu().numpy().T) / 2.0
        Qu = (self.Q_upper.detach().cpu().numpy() + self.Q_upper.detach().cpu().numpy().T) / 2.0
        P_np = self.P.detach().cpu().numpy()
        A_np = self.A.detach().cpu().numpy()
        B_np = self.B.detach().cpu().numpy()
        b_np = self.b.detach().cpu().numpy()
        c_lower_np = self.c_lower.detach().cpu().numpy()
        c_upper_np = self.c_upper.detach().cpu().numpy()
        x_target_np = self.x_target.detach().cpu().numpy()

        # CVXPy variable and parameters for stochastic objective coefficients
        y_cp = cp.Variable(self.dim)
        q_lin_param = cp.Parameter(self.dim)  # stochastic linear term for g
        c_upper_param = cp.Parameter(self.dim)  # stochastic linear term in f wrt y

        # Build F2CSA penalty objective components
        # f(x,y) = 0.5*(x - x_target)ᵀQ_upper(x - x_target) + c_upperᵀy
        f_const = 0.5 * (x_np - x_target_np).T @ Qu @ (x_np - x_target_np)
        f_cp = f_const + c_upper_param.T @ y_cp

        # g(x,y) = 0.5*yᵀQ_lower*y + (c_lower + Pᵀx)ᵀy
        q_lin_base = c_lower_np + P_np.T @ x_np
        g_cp = 0.5 * cp.quad_form(y_cp, Ql) + q_lin_param.T @ y_cp
        g_opt_cp = 0.5 * y_star_np.T @ Ql @ y_star_np + q_lin_param.T @ y_star_np

        # Stochastic injection for f and g linear terms if batch_size provided
        if batch_size is None:
            q_lin_param.value = q_lin_base
            c_upper_param.value = c_upper_np
        else:
            noise_scale = float(self.noise_std) / max(1.0, np.sqrt(batch_size))
            q_lin_param.value = q_lin_base + np.random.randn(*q_lin_base.shape) * noise_scale
            c_upper_param.value = c_upper_np + np.random.randn(*c_upper_np.shape) * noise_scale

        # h(x,y) = Ax - By - b ≤ 0 (deterministic)
        h_cp = A_np @ x_np - B_np @ y_cp - b_np
        h_opt_cp = A_np @ x_np - B_np @ y_star_np - b_np

        # F2CSA penalty Lagrangian (Eq. 396-401)
        linear_penalty = g_cp - g_opt_cp + lam_np.T @ h_cp
        quadratic_penalty = cp.sum(cp.multiply(rho_np, cp.square(h_cp)))

        objective = cp.Minimize(
            f_cp + alpha1 * linear_penalty + 0.5 * alpha2 * quadratic_penalty
        )

        # Solve with original constraints active
        constraints = [h_cp <= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, verbose=False)

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"F2CSA penalty Lagrangian solve failed: status={problem.status}")

        return torch.tensor(y_cp.value, device=self.device, dtype=self.dtype)


    print(f"   F2CSA: Enhanced with all improvements")
    print(f"   SSIGD: Smoothed gradient approach")
    print(f"   DS-BLO: Doubly stochastic with momentum")
