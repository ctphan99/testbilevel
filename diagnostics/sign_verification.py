import torch
from typing import Tuple

from problem import StronglyConvexBilevelProblem


def build_toy_problem() -> StronglyConvexBilevelProblem:
    # Small, deterministic, noise-free toy
    return StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.0, device='cpu', seed=123)


def f2csa_stationarity_components(problem: StronglyConvexBilevelProblem, x: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Gather LL solution and duals
    y_star, info = problem.solve_lower_level(x)
    lam_tilde = info.get('lambda', torch.zeros(problem.num_constraints, device=problem.device, dtype=problem.dtype))
    lam_tilde = torch.clamp(lam_tilde, min=0.0)

    # Gates ρ built as in algorithm.F2CSA (using y*)
    with torch.no_grad():
        h_y_star = problem.constraints(x, y_star)
        tau = 5.0 * (alpha**3)
        tau = max(float(tau), 1e-3)
        sigma_h = torch.sigmoid(h_y_star / tau)
        sigma_lam = torch.sigmoid(lam_tilde / tau)
        rho = sigma_h * sigma_lam

    Q = problem.Q_lower
    B = problem.B
    A = problem.A
    P = problem.P
    c_lower = problem.c_lower
    c_upper = problem.c_upper

    diag_rho = torch.diag(rho)
    alpha1 = alpha**(-2)
    alpha2 = alpha**(-4)
    delta = alpha**3

    I = torch.eye(Q.shape[0], device=Q.device, dtype=Q.dtype)
    H = alpha1 * Q + alpha2 * (B.T @ diag_rho @ B) + delta * I
    d_pen = (c_lower + P.T @ x)
    rhs = -c_upper - alpha1 * d_pen + alpha1 * (B.T @ lam_tilde) + alpha2 * (B.T @ (diag_rho @ (A @ x - problem.b)))

    return y_star, lam_tilde, rho, H, rhs


def f2csa_stationarity_check(problem: StronglyConvexBilevelProblem, alpha: float = 0.25) -> None:
    torch.manual_seed(7)
    x = torch.randn(problem.dim, device=problem.device, dtype=problem.dtype) * 0.1

    y_star, lam_tilde, rho, H, rhs = f2csa_stationarity_components(problem, x, alpha)
    y_tilde = torch.linalg.solve(H, rhs)

    # Build true grad_y L_pen(y) (without the synthetic delta regularizer) at y_tilde
    # L(y) = f(x,y) + alpha1(g(x,y)-g(x,y*)) + alpha1 lam^T h(x,y) + 0.5 alpha2 sum rho h(x,y)^2
    Q = problem.Q_lower
    B = problem.B
    A = problem.A
    P = problem.P
    c_lower = problem.c_lower
    c_upper = problem.c_upper

    d_pen = (c_lower + P.T @ x)
    h = problem.constraints(x, y_tilde)

    # grad_y terms
    grad_y_f = c_upper
    grad_y_g = Q @ y_tilde + d_pen
    grad_y_lam_h = - B.T @ lam_tilde
    grad_y_quad_pen = - (B.T @ (torch.diag(rho) @ h)) * (alpha**(-4))  # note alpha2 will be multiplied below

    alpha1 = alpha**(-2)
    alpha2 = alpha**(-4)

    grad_y_L = grad_y_f + alpha1 * grad_y_g + alpha1 * grad_y_lam_h + alpha2 * (B.T @ (torch.diag(rho) @ (B @ y_tilde - (A @ x - problem.b))))
    # The last line is the expanded version that should match: -B^T diag(rho) h with a plus alpha2 times B^T diag(rho) B y + ...

    # Residual expected due to delta*I regularization: grad_y_L ≈ delta * y_tilde
    delta = alpha**3
    residual = torch.norm(grad_y_L - delta * y_tilde).item()

    print("[F2CSA] Stationarity residual ||∂L/∂y(y_tilde) - δ y_tilde|| = {:.3e}".format(residual))
    print("[F2CSA] ||y_tilde|| = {:.3e}, ||H y_tilde - rhs|| = {:.3e}".format(torch.norm(y_tilde).item(), torch.norm(H @ y_tilde - rhs).item()))


def ssigd_adjoint_check(problem: StronglyConvexBilevelProblem, use_perturbation: bool = True) -> None:
    torch.manual_seed(11)
    x = torch.randn(problem.dim, device=problem.device, dtype=problem.dtype) * 0.1
    q = torch.randn(problem.dim, device=problem.device, dtype=problem.dtype) * 0.01 if use_perturbation else None

    # Solve LL (optionally perturbed)
    y_hat, info = problem.solve_lower_level(x, y_linear_offset=q)

    # Build grad_y f at (x, y_hat)
    x_var = x.clone().requires_grad_(True)
    y_var = y_hat.clone().detach().requires_grad_(True)
    f_val = problem.upper_objective(x_var, y_var, add_noise=False)
    grad_y_f = torch.autograd.grad(f_val, y_var, create_graph=False, retain_graph=False)[0]

    # Active set from LL
    active_mask = info.get('active_mask', None)
    if active_mask is None:
        cons = problem.constraints(x, y_hat).detach()
        active_mask = (cons.abs() <= 1e-8)
    B_act = problem.B[active_mask, :]

    # Solve adjoint [Q B_act^T; B_act 0][p; mu] = [grad_y_f; 0]
    Q = problem.Q_lower
    n = Q.shape[0]
    k = int(B_act.shape[0])
    if k == 0:
        p = torch.linalg.solve(Q, grad_y_f)
    else:
        K = torch.zeros(n + k, n + k, device=problem.device, dtype=problem.dtype)
        K[:n, :n] = Q
        K[:n, n:] = B_act.T
        K[n:, :n] = B_act
        rhs = torch.zeros(n + k, device=problem.device, dtype=problem.dtype)
        rhs[:n] = grad_y_f
        sol = torch.linalg.solve(K, rhs)
        p = sol[:n]

    implicit_adj = - problem.P @ p

    # Finite-difference implicit component (Jacobian-vector product contraction)
    eps = 1e-5
    fd = torch.zeros_like(x)
    with torch.no_grad():
        for i in range(x.shape[0]):
            x_pert = x.clone()
            x_pert[i] += eps
            y_pert, _ = problem.solve_lower_level(x_pert, y_linear_offset=q)
            dy = (y_pert - y_hat) / eps
            fd[i] = torch.dot(dy, grad_y_f)

    err = torch.norm(implicit_adj - fd).item()
    rel = err / (torch.norm(fd).item() + 1e-12)
    print("[SSIGD] Adjoint vs FD implicit mismatch: abs={:.3e}, rel={:.3e}".format(err, rel))


if __name__ == "__main__":
    prob = build_toy_problem()
    print("\n=== F2CSA stationarity sign check (toy) ===")
    f2csa_stationarity_check(prob, alpha=0.25)

    print("\n=== SSIGD adjoint sign check (toy) ===")
    ssigd_adjoint_check(prob, use_perturbation=True)

