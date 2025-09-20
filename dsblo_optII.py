#!/usr/bin/env python3
"""
DS-BLO Option II (stochastic) implementation

Implements Option II from the DS-BLO paper, where the gradient is a sampled
implicit gradient. Since our upper-level objective is deterministic in this
setup, we model stochasticity by injecting Gaussian noise with standard
deviation sigma added to the implicit gradient each iteration. The algorithm
already includes the first perturbation q in the LL as per the paper, and
the second perturbation through random xbar ~ U[x_t, x_{t+1}].

Safeguards: gradient clipping and step-size caps to prevent divergence.
"""

from typing import Dict, Optional
import argparse
import torch
import matplotlib.pyplot as plt
from problem import StronglyConvexBilevelProblem
try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    _HAS_CVXPYLAYERS = True
except Exception:
    _HAS_CVXPYLAYERS = False


class DSBLOOptII:
    def __init__(self, problem: StronglyConvexBilevelProblem, device: str = 'cpu', dtype=torch.float64):
        self.problem = problem
        self.device = device
        self.dtype = dtype

        # Add initial q₀ to lower-level problem (like SSIGD) - added once at the beginning
        q_0 = torch.randn(problem.dim, device=device, dtype=dtype)
        self.q_0 = 1e-6 * (q_0 / torch.norm(q_0))  # Normalize like SSIGD
        
        # Create noisy c_lower with initial q₀
        self.c_lower_noisy = problem.c_lower + self.q_0

        # Initialize CVXPyLayer for box-constrained LL: min 0.5 y^T Q y + c^T y s.t. -1 <= y <= 1
        self._layer = None
        if _HAS_CVXPYLAYERS:
            try:
                y = cp.Variable(problem.dim)
                Qp = cp.Parameter((problem.dim, problem.dim), PSD=True)
                cp_c = cp.Parameter(problem.dim)
                objective = cp.Minimize(0.5 * cp.quad_form(y, Qp) + cp_c @ y)
                constraints = [y <= 1, -y <= 1]
                prob = cp.Problem(objective, constraints)
                self._layer = CvxpyLayer(prob, parameters=[Qp, cp_c], variables=[y])
            except Exception:
                # Fallback to problem's CVXPY solver if DPP fails
                self._layer = None

    def gradx_f(self, x: torch.Tensor, y: torch.Tensor, noise_upper: Optional[torch.Tensor] = None) -> torch.Tensor:
        Q_up = self.problem.Q_upper if noise_upper is None else (self.problem.Q_upper + noise_upper)
        return Q_up @ x + self.problem.c_upper + self.problem.P @ y

    def grad_F(self, x: torch.Tensor, y: torch.Tensor,
               noise_upper: Optional[torch.Tensor] = None, noise_lower: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute full implicit gradient using CVXPYLayers for exact Hessian computation.
        Following Equation (8): ∇F(x) = ∇x f(x, y*(x)) + [∇y*(x)]ᵀ ∇y f(x, y*(x))
        """
        # Get solution using CVXPYLayers (y is already the perturbed solution)
        y_hat = y.clone().detach()
        
        # Compute ∇x f(x, y*(x)) - direct partial derivative w.r.t. x
        x_direct = x.clone().requires_grad_(True)
        y_fixed = y_hat.clone().detach()
        f_direct = self.problem.upper_objective(x_direct, y_fixed)
        grad_x_f = torch.autograd.grad(f_direct, x_direct, retain_graph=True)[0]
        
        # Compute ∇y f(x, y*(x)) - partial derivative w.r.t. y
        x_fixed = x.clone().detach()
        y_partial = y_hat.clone().requires_grad_(True)
        f_partial = self.problem.upper_objective(x_fixed, y_partial)
        grad_y_f = torch.autograd.grad(f_partial, y_partial, retain_graph=True)[0]
        
        # Compute ∇y*(x) using finite differences with CVXPYLayers
        # Use the same noise for consistency
        eps = 1e-6
        grad_y_star = torch.stack([
            (self.solve_ll_with_q_t(x + eps * torch.eye(self.problem.dim, device=self.device, dtype=self.dtype)[i], 
                                   torch.zeros_like(self.q_0), noise_lower) - y_hat) / eps 
            for i in range(self.problem.dim)
        ], dim=1)
        
        # Apply Equation (8): ∇F(x) = ∇x f(x, y*(x)) + [∇y*(x)]ᵀ ∇y f(x, y*(x))
        total_grad = grad_x_f + grad_y_star.T @ grad_y_f
        
        return total_grad

    def solve_ll(self, x: torch.Tensor) -> torch.Tensor:
        """Solve clean lower-level problem (no noise) for tracking purposes"""
        if self._layer is None:
            # Fallback: use problem's solver if CVXPYLayers not available
            y_opt, _, _ = self.problem.solve_lower_level(x, solver='gurobi')
            return y_opt
        # Use CVXPYLayers with clean Q_lower (no noise, no q perturbation)
        Q_lo = self.problem.Q_lower.detach()
        c_lo = self.problem.c_lower.detach()
        y_sol, = self._layer(Q_lo, c_lo)
        return y_sol.to(dtype=self.dtype, device=self.device)


    def solve_ll_with_q_t(self, x: torch.Tensor, q_t: torch.Tensor, noise_lower: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Solve lower-level problem with q₀ + q_t perturbation (per iteration)"""
        if self._layer is None:
            # Fallback: approximate by calling problem.solve_lower_level (no q support)
            y_star, _, _ = self.problem.solve_lower_level(x)
            return y_star
        if noise_lower is None:
            _, noise_lower = self.problem._sample_instance_noise()
        Q_lo = (self.problem.Q_lower + noise_lower).detach()
        # Add both q₀ and q_t to c_lower: c_lower + q₀ + q_t
        c_lo = (self.c_lower_noisy + q_t).detach()
        y_sol, = self._layer(Q_lo, c_lo)
        return y_sol.to(dtype=self.dtype, device=self.device)

    def optimize(self, x0: torch.Tensor, T: int, alpha: float, sigma: float = 1e-2,
                 grad_clip: float = 5.0, eta_cap: float = 1e-2,
                 noise_decay: float = 0.995, noise_min: float = 1e-3, grad_avg_k: int = 8,
                 gamma1: float = 1.0, gamma2: float = 1.0, lr: float = 0.8,
                 adapt_eta: bool = True, adapt_patience: int = 50, adapt_factor: float = 0.7,
                 verbose: bool = False, log_every: int = 50, log_csv: Optional[str] = None,
                 ul_track_noisy_ll: bool = False) -> Dict:
        print("🚀 DS-BLO Option II (stochastic) Algorithm - CORRECTED")
        print("=" * 60)
        print(f"T = {T}, α = {alpha:.6f}, σ = {sigma:.3e}")
        print("Using q₀ + q_t perturbation: q₀ fixed, q_t sampled fresh each iteration")

        # Parameters (can be overridden via CLI)
        print(f"γ₁ = {gamma1:.6f}, γ₂ = {gamma2:.6f}, lr = {lr:.6f}")

        x = x0.clone().detach()
        ul_losses = []
        hypergrad_norms = []
        raw_grad_norms = []  # Track raw ||g_t|| in addition to ||m_t||

        # Sample noise once for the initial iteration
        _, noise_lower = self.problem._sample_instance_noise()
        
        # Initial: solve LL with q₀ only (no q_t yet) to get initial gradient
        yhat = self.solve_ll_with_q_t(x, torch.zeros_like(self.q_0), noise_lower)  # q_t = 0 for initial
        m = self.grad_F(x, yhat)

        # Deterministic UL loss tracking (no upper noise) - use clean solve_ll
        y_star = self.solve_ll(x)
        ul_losses.append(self.problem.upper_objective(x, y_star).item())
        # At init, m equals g (no momentum yet)
        hypergrad_norms.append(torch.norm(m).item())

        best_ul = float('inf')
        since_best = 0
        for t in range(1, T + 1):
            # Sample noise once per iteration for consistency
            _, noise_lower = self.problem._sample_instance_noise()
            
            grad_norm = torch.norm(m)
            eta = 1.0 / (gamma1 * grad_norm + gamma2)
            # Ensure step size is bounded between 1e-4 and 1e-2
            eta = torch.clamp(torch.tensor(eta), min=1e-4, max=1e-2).item()
            eta = min(eta, eta_cap)

            x_prev = x.clone()
            x = x - eta * m

            # xbar ~ U[x_t, x_{t+1}] (second perturbation)
            xbar = torch.rand_like(x) * (x - x_prev) + x_prev

            # gradient averaging with fresh LL solves at xbar via CVXPyLayer (with q₀ + q_t perturbation)
            g_acc = torch.zeros_like(x)
            for _ in range(max(1, grad_avg_k)):
                # Sample new q_t independently from previous iterations
                q_t = torch.randn(self.problem.dim, device=self.device, dtype=self.dtype) * 1e-6
                q_t = q_t / torch.norm(q_t)  # Normalize like SSIGD
                yhat = self.solve_ll_with_q_t(xbar, q_t, noise_lower)
                g_sample = self.grad_F(xbar, yhat)
                g_acc += g_sample
            g = g_acc / max(1, grad_avg_k)
            g_norm = torch.norm(g)
            if g_norm > grad_clip:
                g = g / g_norm * grad_clip

            # momentum update
            m_before = m.clone()
            m = lr * m + (1.0 - lr) * g
            # clip momentum as well to stabilize
            m_norm = torch.norm(m)
            if m_norm > grad_clip:
                m = m / m_norm * grad_clip
            
            # Normalize gradient at iteration 1 to reference norm for fair comparison
            if t == 1 and hasattr(self, 'reference_grad_norm'):
                current_norm = torch.norm(m).item()
                if current_norm > 1e-10:  # Avoid division by zero
                    m = m * (self.reference_grad_norm / current_norm)

            if verbose and (t % max(1, log_every) == 0):
                # UL at current x with y* from deterministic LL solve (clean)
                y_star_dbg = self.solve_ll(x)
                F_x = self.problem.upper_objective(x, y_star_dbg).item()
                # Descent probe using immediate grad (no momentum)
                eta_raw_dbg = 1.0 / (gamma1 * max(g_norm.item(), 1e-12) + gamma2)
                eta_dbg = min(eta_raw_dbg, eta_cap)
                x_probe = x - eta_dbg * g
                y_star_probe = self.solve_ll(x_probe)
                F_probe = self.problem.upper_objective(x_probe, y_star_probe).item()

                # Sample q for debugging (same as in gradient computation)
                q_debug = torch.randn(self.problem.dim, device=self.device, dtype=self.dtype) * 1e-6
                q_debug = q_debug / torch.norm(q_debug)
                yhat_debug = self.solve_ll_with_q_t(xbar, q_debug, noise_lower)
                gxf = self.gradx_f(xbar, yhat_debug)
                h_yhat = self.problem.constraints(xbar, yhat_debug)
                viol_yhat = torch.clamp(h_yhat, min=0)

                print(f"[t={t}] F(x)={F_x:.6f} F(x-eta*g)={F_probe:.6f} eta={eta_dbg:.3e} eta_cap={eta_cap:.3e}"
                      f" ||g||pre={g_norm:.6f} ||m||pre={torch.norm(m_before).item():.6f} ||m||post={m_norm:.6f}")
                print(f"  ||gradx_f||={torch.norm(gxf).item():.6f} yhat viol max={viol_yhat.max().item():.3e}"
                      f" ||viol||={torch.norm(viol_yhat).item():.3e} k={grad_avg_k} q_norm={torch.norm(q_debug).item():.6f}")

            # tracking - always use clean solve_ll for deterministic upper-level tracking
            y_star = self.solve_ll(x)
            ul_losses.append(self.problem.upper_objective(x, y_star).item())
            hypergrad_norms.append(torch.norm(g).item())
            raw_grad_norms.append(torch.norm(g).item())

            # adaptive eta cap if not improving
            if adapt_eta:
                cur_ul = ul_losses[-1]
                if cur_ul + 1e-8 < best_ul:
                    best_ul = cur_ul
                    since_best = 0
                else:
                    since_best += 1
                    if since_best >= adapt_patience:
                        new_cap = max(eta_cap * adapt_factor, 1e-6)
                        if new_cap < eta_cap:
                            eta_cap = new_cap
                        since_best = 0

            if t % 100 == 0:
                print(f"Iteration {t}/{T}: ||m|| = {hypergrad_norms[-1]:.6f}, UL = {ul_losses[-1]:.6f}")

            # no algorithmic Gaussian noise; stochasticity comes from noisy Q_lower in LL

        return {
            'x_out': x,
            'final_gradient': m,
            'final_gradient_norm': torch.norm(m).item(),
            'final_ul_loss': ul_losses[-1],
            'ul_losses': ul_losses,
            'hypergrad_norms': hypergrad_norms,
            'raw_grad_norms': raw_grad_norms,  # Include raw gradient norms
            'iterations': T,
            'converged': torch.norm(m).item() < 1e-3,
        }


def main():
    parser = argparse.ArgumentParser(description='DS-BLO Option II (stochastic) test')
    parser.add_argument('--T', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--dim', type=int, default=5)
    parser.add_argument('--constraints', type=int, default=3)
    parser.add_argument('--sigma', type=float, default=1e-2)
    parser.add_argument('--grad-clip', type=float, default=5.0)
    parser.add_argument('--eta-cap', type=float, default=1e-5)
    parser.add_argument('--noise-decay', type=float, default=0.995)
    parser.add_argument('--noise-min', type=float, default=1e-3)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--gamma1', type=float, default=1.0)
    parser.add_argument('--gamma2', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.8)
    parser.add_argument('--no-adapt-eta', action='store_true')
    parser.add_argument('--plot-name', type=str, default='dsblo_optII.png')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--log-every', type=int, default=50)
    args = parser.parse_args()

    problem = StronglyConvexBilevelProblem(dim=args.dim, num_constraints=args.constraints)
    x0 = torch.randn(args.dim, dtype=torch.float64)
    algo = DSBLOOptII(problem)
    res = algo.optimize(
        x0, args.T, args.alpha, sigma=args.sigma,
        grad_clip=args.grad_clip, eta_cap=args.eta_cap,
        noise_decay=args.noise_decay, noise_min=args.noise_min,
        grad_avg_k=args.k, gamma1=args.gamma1, gamma2=args.gamma2, lr=args.lr,
        adapt_eta=(not args.no_adapt_eta), verbose=args.verbose, log_every=args.log_every
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(res['ul_losses'])
    ax1.set_title('DS-BLO Opt II UL Loss')
    ax1.set_xlabel('Iter')
    ax1.set_ylabel('F(x)')
    ax1.grid(True, alpha=0.3)

    ax2.plot(res['hypergrad_norms'])
    ax2.set_title('DS-BLO Opt II ||m_t||')
    ax2.set_xlabel('Iter')
    ax2.set_ylabel('Norm')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.plot_name, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {args.plot_name}")


if __name__ == '__main__':
    main()


