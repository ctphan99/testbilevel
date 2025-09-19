import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
import time

try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    _HAS_CVXPYLAYERS = True
except Exception:
    _HAS_CVXPYLAYERS = False


class CorrectSSIGD:
    """
    SSIGD aligned to DS-BLO structure: use CVXPy-based LL solve with noisy Q_lower
    and direct upper gradient ∇x f(x, y*(x)); update via projected step:
    x_{r+1} = proj_X(x_r - beta_r * ∇̂F(x_r)).
    """

    def __init__(self, problem: StronglyConvexBilevelProblem):
        self.prob = problem
        self.device = problem.device
        self.dtype = problem.dtype

        # Build CVXPyLayer LL
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
                self._layer = None

        # SSIGD step 2: sample a fixed q ~ Q for LL perturbation
        self.q = torch.randn(problem.dim, device=self.device, dtype=self.dtype) * (problem.noise_std if hasattr(problem, 'noise_std') else 0.01)

    # Direct UL gradient
    def gradx_f(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.prob.Q_upper @ x + self.prob.c_upper + self.prob.P @ y

    def grad_F(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.gradx_f(x, y)

    def solve_ll(self, x: torch.Tensor) -> torch.Tensor:
        if self._layer is None:
            y_star, _, _ = self.prob.solve_lower_level(x, solver='cvxpy')
            return y_star
        _, noise_lo = self.prob._sample_instance_noise()
        Q_lo = (self.prob.Q_lower + noise_lo).detach()
        c_lo = self.prob.c_lower.detach()
        y_sol, = self._layer(Q_lo, c_lo)
        return y_sol.to(dtype=self.dtype, device=self.device)

    def solve_ll_with_q(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        if self._layer is None:
            y_star, _, _ = self.prob.solve_lower_level(x, solver='cvxpy')
            return y_star
        _, noise_lo = self.prob._sample_instance_noise()
        Q_lo = (self.prob.Q_lower + noise_lo).detach()
        c_lo = (self.prob.c_lower + q).detach()
        y_sol, = self._layer(Q_lo, c_lo)
        return y_sol.to(dtype=self.dtype, device=self.device)

    def proj_X(self, x: torch.Tensor) -> torch.Tensor:
        # If problem defines a projection, use it; otherwise identity
        if hasattr(self.prob, 'project_X') and callable(getattr(self.prob, 'project_X')):
            return self.prob.project_X(x)
        return x

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
            x = self.proj_X(x - lr_t * grad_est)

            # tracking (deterministic UL, LL with noise for y)
            y_star = self.solve_ll(x)
            F = self.prob.upper_objective(x, y_star).item()
            losses.append(F)
            grad_norms.append(torch.norm(grad_est).item())

        return x, losses, grad_norms

def test_correct_ssigd():
    """Test the correct SSIGD implementation"""
    print("=" * 80)
    print("TESTING CORRECT SSIGD IMPLEMENTATION")
    print("=" * 80)
    
    # Create problem instance
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.01,
        strong_convex=True,
        device='cpu'
    )
    
    # Create correct SSIGD
    ssigd = CorrectSSIGD(problem)
    
    # Run SSIGD
    x_final, losses, grad_norms = ssigd.solve(T=100, beta=0.01)
    
    print(f"\nResults:")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Final gradient norm: {grad_norms[-1]:.3f}")
    print(f"  Final x: {x_final}")
    print(f"  Max gradient norm: {max(grad_norms):.3f}")
    print(f"  Loss trajectory: {losses[:5]} ... {losses[-5:]}")
    
    # Check for numerical stability
    if max(grad_norms) > 100:
        print(f"  Warning: Large gradient norms detected - possible numerical issues")
    else:
        print(f"  Gradient norms are reasonable - good numerical stability")
    
    return x_final, losses, grad_norms

if __name__ == "__main__":
    
    # Test correct implementation
    test_correct_ssigd()

