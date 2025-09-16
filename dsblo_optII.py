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

    def hessyy_g(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.problem.Q_lower

    def hessxy_g(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.problem.P

    def gradx_f(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.problem.Q_upper @ x + self.problem.c_upper + self.problem.P @ y

    def grady_f(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.problem.P.T @ x + self.problem.Q_lower @ y + self.problem.c_lower

    def grad_lambdastar(self, x: torch.Tensor, y: torch.Tensor, Aact: torch.Tensor, Bact: torch.Tensor) -> torch.Tensor:
        hessyy_inv = torch.linalg.inv(self.hessyy_g(x, y))
        return -torch.linalg.inv(Aact @ hessyy_inv @ Aact.T) @ (Aact @ hessyy_inv @ self.hessxy_g(x, y) - Bact)

    def grad_ystar(self, x: torch.Tensor, y: torch.Tensor, Aact: Optional[torch.Tensor], Bact: Optional[torch.Tensor]) -> torch.Tensor:
        if Aact is None:
            return -torch.linalg.inv(self.hessyy_g(x, y)) @ self.hessxy_g(x, y)
        return torch.linalg.inv(self.hessyy_g(x, y)) @ (-self.hessxy_g(x, y) - Aact.T @ self.grad_lambdastar(x, y, Aact, Bact))

    def grad_F(self, x: torch.Tensor, y: torch.Tensor, Aact: Optional[torch.Tensor], Bact: Optional[torch.Tensor]) -> torch.Tensor:
        return self.gradx_f(x, y) + self.grad_ystar(x, y, Aact, Bact).T @ self.grady_f(x, y)

    def solve_ll(self, x: torch.Tensor) -> torch.Tensor:
        y_star, _ = self.problem.solve_lower_level(x)
        return y_star

    def solve_ll_perturbed(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # Use accurate solver with linear term shift c_lower + q via projection descent
        c_perturbed = self.problem.c_lower + q
        # unconstrained optimum
        y = -torch.linalg.solve(self.problem.Q_lower, c_perturbed)
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
                 grad_clip: float = 20.0, eta_cap: float = 1e-4,
                 noise_decay: float = 0.99, noise_min: float = 1e-3, grad_avg_k: int = 4) -> Dict:
        print("🚀 DS-BLO Option II (stochastic) Algorithm")
        print("=" * 60)
        print(f"T = {T}, α = {alpha:.6f}, σ = {sigma:.3e}")

        # Conservative, stability-oriented parameters
        gamma1 = 0.1
        gamma2 = 0.1
        beta = 0.9
        print(f"γ₁ = {gamma1:.6f}, γ₂ = {gamma2:.6f}, β = {beta:.6f}")

        x = x0.clone().detach()
        ul_losses = []
        hypergrad_norms = []
        x_history = []

        # q1 ~ Q (LL perturbation)
        q = torch.randn(self.problem.dim, dtype=self.dtype, device=self.device)
        q = 1e-6 * (q / torch.norm(q))

        # initial yhat (perturbed) and gradient
        yhat = self.solve_ll_perturbed(x, q)
        Aact, Bact = self.active(x, yhat)
        m = self.grad_F(x, yhat, Aact, Bact)
        # add stochastic noise (Option II)
        m = m + sigma * torch.randn_like(m)

        y_star, _ = self.problem.solve_lower_level(x)
        ul_losses.append(self.problem.upper_objective(x, y_star).item())
        hypergrad_norms.append(torch.norm(m).item())

        for t in range(1, T + 1):
            grad_norm = torch.norm(m)
            eta = 1.0 / (gamma1 * grad_norm + gamma2)
            eta = min(eta, eta_cap)

            x_prev = x.clone()
            x = x - eta * m

            # xbar ~ U[x_t, x_{t+1}] (second perturbation)
            xbar = torch.rand_like(x) * (x - x_prev) + x_prev

            # gradient averaging with fresh perturbations
            g_acc = torch.zeros_like(x)
            for _ in range(max(1, grad_avg_k)):
                q = torch.randn(self.problem.dim, dtype=self.dtype, device=self.device)
                q = 1e-6 * (q / torch.norm(q))
                yhat = self.solve_ll_perturbed(xbar, q)
                Aact, Bact = self.active(x, yhat)
                g_sample = self.grad_F(xbar, yhat, Aact, Bact)
                g_acc += g_sample
            g = g_acc / max(1, grad_avg_k)
            if grad_norm > grad_clip:
                g = g / grad_norm * grad_clip
            g = g + sigma * torch.randn_like(g)

            # momentum update
            m = beta * m + (1.0 - beta) * g
            # clip momentum as well to stabilize
            m_norm = torch.norm(m)
            if m_norm > grad_clip:
                m = m / m_norm * grad_clip

            # tracking
            y_star, _ = self.problem.solve_lower_level(x)
            ul_losses.append(self.problem.upper_objective(x, y_star).item())
            hypergrad_norms.append(torch.norm(m).item())
            x_history.append(x.clone().detach())

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
    parser.add_argument('--plot-name', type=str, default='dsblo_optII.png')
    args = parser.parse_args()

    problem = StronglyConvexBilevelProblem(dim=args.dim, num_constraints=args.constraints)
    x0 = torch.randn(args.dim, dtype=torch.float64)
    algo = DSBLOOptII(problem)
    res = algo.optimize(x0, args.T, args.alpha, sigma=args.sigma)

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


