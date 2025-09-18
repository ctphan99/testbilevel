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

from typing import Dict, Optional, Tuple
import argparse
import torch
import matplotlib.pyplot as plt
from problem import StronglyConvexBilevelProblem


class DSBLOOptII:
    def __init__(self, problem: StronglyConvexBilevelProblem, device: str = 'cpu', dtype=torch.float64):
        self.problem = problem
        self.device = device
        self.dtype = dtype

    def active(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        eps = 1e-3
        h = self.problem.constraints(x, y)
        active_indices = []
        for i in range(self.problem.num_constraints):
            if -eps < h[i] <= 0:
                active_indices.append(i)
        if len(active_indices) == 0:
            return None, None
        return self.problem.A[active_indices, :], self.problem.B[active_indices, :]

    def hessyy_g(self, x: torch.Tensor, y: torch.Tensor, noise_lower: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise_lower is None:
            return self.problem.Q_lower
        return self.problem.Q_lower + noise_lower

    def hessxy_g(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # In this problem setup, g(x,y) = 0.5*y^T Q_lower y + c_lower^T y (no x-coupling in LL objective)
        # so ∇^2_{xy} g = 0.
        return torch.zeros((self.problem.dim, self.problem.dim), dtype=self.dtype, device=self.device)

    def gradx_f(self, x: torch.Tensor, y: torch.Tensor, noise_upper: Optional[torch.Tensor] = None) -> torch.Tensor:
        Q_up = self.problem.Q_upper if noise_upper is None else (self.problem.Q_upper + noise_upper)
        return Q_up @ x + self.problem.c_upper + self.problem.P @ y

    def grady_f(self, x: torch.Tensor, y: torch.Tensor, noise_lower: Optional[torch.Tensor] = None) -> torch.Tensor:
        # UL objective: f(x,y) = 0.5*x^T Q_upper x + c_upper^T x + 0.5*y^T P y + x^T P y
        # ∇_y f = P y + P^T x
        return self.problem.P @ y + self.problem.P.T @ x

    def grad_lambdastar(self, x: torch.Tensor, y: torch.Tensor, Aact: torch.Tensor, Bact: torch.Tensor, noise_lower: Optional[torch.Tensor] = None) -> torch.Tensor:
        hessyy_inv = torch.linalg.inv(self.hessyy_g(x, y, noise_lower))
        return -torch.linalg.inv(Aact @ hessyy_inv @ Aact.T) @ (Aact @ hessyy_inv @ self.hessxy_g(x, y) - Bact)

    def grad_ystar(self, x: torch.Tensor, y: torch.Tensor, Aact: Optional[torch.Tensor], Bact: Optional[torch.Tensor], noise_lower: Optional[torch.Tensor] = None) -> torch.Tensor:
        if Aact is None:
            return -torch.linalg.inv(self.hessyy_g(x, y, noise_lower)) @ self.hessxy_g(x, y)
        return torch.linalg.inv(self.hessyy_g(x, y, noise_lower)) @ (-self.hessxy_g(x, y) - Aact.T @ self.grad_lambdastar(x, y, Aact, Bact, noise_lower))

    def grad_F(self, x: torch.Tensor, y: torch.Tensor, Aact: Optional[torch.Tensor], Bact: Optional[torch.Tensor],
               noise_upper: Optional[torch.Tensor] = None, noise_lower: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.gradx_f(x, y, noise_upper) + self.grad_ystar(x, y, Aact, Bact, noise_lower).T @ self.grady_f(x, y, noise_lower)

    def solve_ll(self, x: torch.Tensor) -> torch.Tensor:
        y_star, _ = self.problem.solve_lower_level(x)
        return y_star

    def solve_ll_perturbed(self, x: torch.Tensor, q: torch.Tensor, noise_lower: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Use accurate solver with linear term shift c_lower + q via projection descent
        c_perturbed = self.problem.c_lower + q
        # unconstrained optimum
        Q_lo = self.problem.Q_lower if noise_lower is None else (self.problem.Q_lower + noise_lower)
        y = -torch.linalg.solve(Q_lo, c_perturbed)
        # project to feasible set using simple correction
        h = self.problem.constraints(x, y)
        violations = torch.clamp(h, min=0)
        if torch.norm(violations) < 1e-10:
            return y
        correction = torch.zeros_like(y)
        for i in range(self.problem.num_constraints):
            if violations[i] > 0:
                B_norm_sq = torch.norm(self.problem.B[i]) ** 2
                if B_norm_sq > 1e-10:
                    correction += violations[i] * self.problem.B[i] / B_norm_sq
        return y - correction

    @torch.no_grad()
    def optimize(self, x0: torch.Tensor, T: int, alpha: float, sigma: float = 1e-2,
                 grad_clip: float = 5.0, eta_cap: float = 1e-5,
                 noise_decay: float = 0.995, noise_min: float = 1e-3, grad_avg_k: int = 8,
                 gamma1: float = 1.0, gamma2: float = 1.0, beta: float = 0.8,
                 adapt_eta: bool = True, adapt_patience: int = 50, adapt_factor: float = 0.7,
                 verbose: bool = False, log_every: int = 50, log_csv: Optional[str] = None,
                 ul_track_noisy_ll: bool = False) -> Dict:
        print("🚀 DS-BLO Option II (stochastic) Algorithm")
        print("=" * 60)
        print(f"T = {T}, α = {alpha:.6f}, σ = {sigma:.3e}")

        # Parameters (can be overridden via CLI)
        print(f"γ₁ = {gamma1:.6f}, γ₂ = {gamma2:.6f}, β = {beta:.6f}")

        x = x0.clone().detach()
        ul_losses = []
        hypergrad_norms = []
        x_history = []

        # q1 ~ Q (LL perturbation)
        q = torch.randn(self.problem.dim, dtype=self.dtype, device=self.device)
        q = 1e-6 * (q / torch.norm(q))

        # initial yhat (perturbed) and gradient with problem noise
        noise_up, noise_lo = self.problem._sample_instance_noise()
        yhat = self.solve_ll_perturbed(x, q, noise_lo)
        Aact, Bact = self.active(x, yhat)
        m = self.grad_F(x, yhat, Aact, Bact, noise_up, noise_lo)
        # add stochastic noise (Option II)
        m = m + sigma * torch.randn_like(m)

        if ul_track_noisy_ll:
            noise_up_track, noise_lo_track = self.problem._sample_instance_noise()
            # Solve LL with noisy lower matrix for tracking (analytic + simple projection)
            Q_lo = self.problem.Q_lower + noise_lo_track
            y = -torch.linalg.solve(Q_lo, self.problem.c_lower)
            h = self.problem.constraints(x, y)
            violations = torch.clamp(h, min=0)
            if torch.norm(violations) >= 1e-10:
                correction = torch.zeros_like(y)
                for i in range(self.problem.num_constraints):
                    if violations[i] > 0:
                        B_norm_sq = torch.norm(self.problem.B[i]) ** 2
                        if B_norm_sq > 1e-10:
                            correction += violations[i] * self.problem.B[i] / B_norm_sq
                y = y - correction
            ul_losses.append(self.problem.upper_objective(x, y, noise_up_track).item())
        else:
            noise_up_track, _ = self.problem._sample_instance_noise()
            y_star, _ = self.problem.solve_lower_level(x)
            ul_losses.append(self.problem.upper_objective(x, y_star, noise_up_track).item())
        hypergrad_norms.append(torch.norm(m).item())

        best_ul = float('inf')
        since_best = 0
        for t in range(1, T + 1):
            
            grad_norm = torch.norm(m)
            eta = 1.0 / (gamma1 * grad_norm + gamma2)
            eta = min(eta, eta_cap)

            x_prev = x.clone()
            x = x - eta * m

            # xbar ~ U[x_t, x_{t+1}] (second perturbation)
            xbar = torch.rand_like(x) * (x - x_prev) + x_prev

            # gradient averaging with fresh perturbations and instance noise
            g_acc = torch.zeros_like(x)
            for _ in range(max(1, grad_avg_k)):
                q = torch.randn(self.problem.dim, dtype=self.dtype, device=self.device)
                q = 1e-6 * (q / torch.norm(q))
                noise_up_s, noise_lo_s = self.problem._sample_instance_noise()
                yhat = self.solve_ll_perturbed(xbar, q, noise_lo_s)
                Aact, Bact = self.active(x, yhat)
                g_sample = self.grad_F(xbar, yhat, Aact, Bact, noise_up_s, noise_lo_s)
                g_acc += g_sample
            g = g_acc / max(1, grad_avg_k)
            g_norm = torch.norm(g)
            if g_norm > grad_clip:
                g = g / g_norm * grad_clip
            g = g + sigma * torch.randn_like(g)

            # momentum update
            m_before = m.clone()
            m = beta * m + (1.0 - beta) * g
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
                # UL at current x with true y*
                y_star_dbg, _ = self.problem.solve_lower_level(x)
                F_x = self.problem.upper_objective(x, y_star_dbg).item()
                # Descent probe using immediate grad (no momentum)
                eta_raw_dbg = 1.0 / (gamma1 * max(g_norm.item(), 1e-12) + gamma2)
                eta_dbg = min(eta_raw_dbg, eta_cap)
                x_probe = x - eta_dbg * g
                y_star_probe, _ = self.problem.solve_lower_level(x_probe)
                F_probe = self.problem.upper_objective(x_probe, y_star_probe).item()

                # decomposition terms at (xbar, yhat)
                gxf = self.gradx_f(xbar, yhat, noise_up_s)
                jy = self.grad_ystar(xbar, yhat, Aact, Bact, noise_lo_s).T @ self.grady_f(xbar, yhat, noise_lo_s)
                # LL diagnostics
                h_yhat = self.problem.constraints(xbar, yhat)
                viol_yhat = torch.clamp(h_yhat, min=0)
                H = self.hessyy_g(xbar, yhat, noise_lo_s)
                cond_val = float('nan')
                min_sv = float('nan')
                if Aact is not None:
                    try:
                        Minv = torch.linalg.inv(Aact @ torch.linalg.inv(H) @ Aact.T)
                        s = torch.linalg.svdvals(Minv)
                        if s.numel() > 0:
                            min_sv = float(s.min().item())
                            if min_sv > 0:
                                cond_val = float((s.max() / s.min()).item())
                    except Exception:
                        pass

                print(f"[t={t}] F(x)={F_x:.6f} F(x-eta*g)={F_probe:.6f} eta={eta_dbg:.3e} eta_cap={eta_cap:.3e}"
                      f" ||g||pre={g_norm:.6f} ||m||pre={torch.norm(m_before).item():.6f} ||m||post={m_norm:.6f}")
                print(f"  |A|={(0 if Aact is None else Aact.size(0))} cond(AH^-1A^T)={cond_val} min_sv={min_sv}")
                print(f"  ||gradx_f||={torch.norm(gxf).item():.6f} ||J^T grady_f||={torch.norm(jy).item():.6f}"
                      f" <gradx_f, J^T grady_f>={torch.dot(gxf.view(-1), jy.view(-1)).item():.6f}")
                print(f"  yhat viol max={viol_yhat.max().item():.3e} ||viol||={torch.norm(viol_yhat).item():.3e}"
                      f" ||noise_up||_F={torch.norm(noise_up_s).item():.3e} ||noise_lo||_F={torch.norm(noise_lo_s).item():.3e}"
                      f" sigma={sigma:.2e} k={grad_avg_k}")

            # tracking
            if ul_track_noisy_ll:
                noise_up_track, noise_lo_track = self.problem._sample_instance_noise()
                Q_lo = self.problem.Q_lower + noise_lo_track
                y = -torch.linalg.solve(Q_lo, self.problem.c_lower)
                h = self.problem.constraints(x, y)
                violations = torch.clamp(h, min=0)
                if torch.norm(violations) >= 1e-10:
                    correction = torch.zeros_like(y)
                    for i in range(self.problem.num_constraints):
                        if violations[i] > 0:
                            B_norm_sq = torch.norm(self.problem.B[i]) ** 2
                            if B_norm_sq > 1e-10:
                                correction += violations[i] * self.problem.B[i] / B_norm_sq
                    y = y - correction
                ul_losses.append(self.problem.upper_objective(x, y, noise_up_track).item())
            else:
                noise_up_track, _ = self.problem._sample_instance_noise()
                y_star, _ = self.problem.solve_lower_level(x)
                ul_losses.append(self.problem.upper_objective(x, y_star, noise_up_track).item())
            hypergrad_norms.append(torch.norm(m).item())
            x_history.append(x.clone().detach())

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

            # decay noise
            sigma = max(noise_min, sigma * noise_decay)

        return {
            'x_out': x,
            'final_gradient': m,
            'final_gradient_norm': torch.norm(m).item(),
            'final_ul_loss': ul_losses[-1],
            'ul_losses': ul_losses,
            'hypergrad_norms': hypergrad_norms,
            'x_history': x_history,
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
    parser.add_argument('--beta', type=float, default=0.8)
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
        grad_avg_k=args.k, gamma1=args.gamma1, gamma2=args.gamma2, beta=args.beta,
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


