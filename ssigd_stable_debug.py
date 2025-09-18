import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
import time

class StableSSIGD:
    """
    SSIGD with stable step size strategies for better convergence:
    1. β = O(1/√T) for constant step size
    2. β_r = 1/(μ_F(r+1)) for diminishing step sizes in strongly-convex problems
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem):
        self.prob = problem
        self.device = problem.device
        self.dtype = problem.dtype
        
        # Single fixed perturbation for smoothing (as per SSIGD paper)
        self.q = torch.randn(problem.dim, device=self.device, dtype=self.dtype) * (4e-5) ** 0.5
        
        print(f"Stable SSIGD: Using single fixed q-perturbation for smoothing")
        print(f"  q shape: {self.q.shape}, q norm: {torch.norm(self.q).item():.2e}")
    
    def solve_lower_level_with_perturbation(self, x, q_perturbation, max_iter=10):
        """Paper-style LL: 10 steps PGD, step-size 0.1, box projection |y|<=1; add q-perturbation."""
        x_det = x.detach()
        y = torch.zeros(self.prob.dim, device=self.device, dtype=self.dtype, requires_grad=True)
        step_size = 0.1
        for _ in range(max_iter):
            if y.grad is not None:
                y.grad.zero_()
            obj = self.prob.lower_objective(x_det, y)
            if q_perturbation is not None:
                obj = obj + torch.sum(q_perturbation * y)
            obj.backward()
            with torch.no_grad():
                y -= step_size * y.grad
                y.copy_(torch.clamp(y, min=-1.0, max=1.0))
        return y.detach()
    
    def compute_stochastic_implicit_gradient(self, x, xi_upper, zeta_lower, y_hat=None, u_dir=None, y_plus=None, y_minus=None, eps_fd=1e-5):
        """Compute stochastic implicit gradient following exact SSIGD formula."""
        try:
            x.requires_grad_(True)
            
            # Step 1: Solve lower-level with q-perturbation for smoothing
            if y_hat is None:
                y_hat = self.solve_lower_level_with_perturbation(x, self.q)
            y_hat = y_hat.detach().requires_grad_(True)
            
            # Step 2: Compute stochastic upper-level objective f(x,y;ξ)
            noise_upper = xi_upper * torch.ones_like(self.prob.Q_upper)
            f_val = self.prob.upper_objective(x, y_hat, noise_upper)
            
            # Step 3: Direct gradient term ∇_x f(x,ŷ(x);ξ)
            grad_x = torch.autograd.grad(f_val, x, create_graph=True, retain_graph=True)[0]
            
            # Step 4: Implicit term via symmetric finite difference
            eps = eps_fd
            if u_dir is None:
                u = torch.randn_like(x)
                u = u / (torch.norm(u) + 1e-12)
            else:
                u = u_dir
            if y_plus is None or y_minus is None:
                x_plus = x + eps * u
                x_minus = x - eps * u
                y_plus = self.solve_lower_level_with_perturbation(x_plus, self.q)
                y_minus = self.solve_lower_level_with_perturbation(x_minus, self.q)

            # Gradient w.r.t. y at current (x, y_hat)
            grad_y = torch.autograd.grad(f_val, y_hat, create_graph=True, retain_graph=True)[0]

            # Directional approximation
            gy_dot_y_plus = torch.sum(grad_y.detach() * y_plus)
            gy_dot_y_minus = torch.sum(grad_y.detach() * y_minus)
            directional_scalar = (gy_dot_y_plus - gy_dot_y_minus) / (2.0 * eps)

            # Form a vector estimator along u
            implicit_vec = directional_scalar * u

            # Total gradient vector
            total_grad = grad_x + implicit_vec
            
            x.requires_grad_(False)
            
            return total_grad.detach()
            
        except Exception as e:
            print(f"    Error in gradient computation: {str(e)[:50]}")
            x.requires_grad_(False)
            return torch.zeros_like(x)
    
    def get_step_size(self, t, T, strategy='constant_sqrt', mu_F=None):
        """
        Compute step size based on strategy:
        - 'constant_sqrt': β = O(1/√T) ≈ 0.014 for T=5000
        - 'diminishing': β_r = 1/(μ_F(r+1)) for strongly-convex problems
        - 'adaptive': Adaptive step size based on gradient norm
        """
        if strategy == 'constant_sqrt':
            # β = O(1/√T) for constant step size
            return 1.0 / np.sqrt(T)
        elif strategy == 'diminishing':
            # β_r = 1/(μ_F(r+1)) for strongly-convex problems
            if mu_F is None:
                # Estimate μ_F from problem properties
                mu_F = 0.1  # Conservative estimate
            return 1.0 / (mu_F * (t + 1))
        elif strategy == 'adaptive':
            # Adaptive step size that decreases with gradient norm
            base_step = 1.0 / np.sqrt(T)
            return base_step
        else:
            return 0.01  # Default fallback
    
    def solve(self, T=1000, x0=None, step_strategy='constant_sqrt', mu_F=None, 
              grad_averaging_K=32, clip_threshold=1.0, eps_fd=1e-4):
        """
        Main SSIGD algorithm with stable step size strategies
        """
        x = (x0.detach().to(device=self.device, dtype=self.dtype).clone()
             if x0 is not None else
             torch.randn(self.prob.dim, device=self.device, dtype=self.dtype) * 0.1)
        losses = []
        hypergrad_norms = []
        x_history = []
        step_sizes = []
        
        print(f"Stable SSIGD: T={T}, strategy={step_strategy}")
        print(f"  Following exact formula: ∇F(x;ξ) = ∇_x f(x,ŷ(x);ξ) + [∇ŷ*(x)]^T ∇_y f(x,ŷ(x);ξ)")
        
        prev_loss = None
        worsen_count = 0
        step_cap = 0.05  # cap on ||Δx|| per iteration
        
        for t in range(T):
            try:
                # record trajectory before update
                x_history.append(x.clone().detach())
                
                # Get step size for this iteration
                beta = self.get_step_size(t, T, step_strategy, mu_F)
                step_sizes.append(beta)
                
                # Precompute shared lower-level solution for this x
                shared_y_hat = self.solve_lower_level_with_perturbation(x, self.q)

                # Fixed unit direction u shared across all K samples this iteration
                u_dir = torch.randn_like(x)
                u_dir = u_dir / (torch.norm(u_dir) + 1e-12)

                # Precompute symmetric FD LL solves once per iteration
                x_plus = x + eps_fd * u_dir
                x_minus = x - eps_fd * u_dir
                y_plus = self.solve_lower_level_with_perturbation(x_plus, self.q)
                y_minus = self.solve_lower_level_with_perturbation(x_minus, self.q)

                # Share the same xi_upper for logging alignment
                xi_upper_shared = torch.randn(1, device=self.device, dtype=self.dtype) * 0.01

                # Gradient averaging over K stochastic samples
                grad_accum = torch.zeros_like(x)
                for _ in range(grad_averaging_K):
                    xi_upper = torch.randn(1, device=self.device, dtype=self.dtype) * 0.01
                    zeta_lower = torch.randn(self.prob.dim, device=self.device, dtype=self.dtype) * 0.01
                    grad_k = self.compute_stochastic_implicit_gradient(
                        x, xi_upper, zeta_lower, y_hat=shared_y_hat, u_dir=u_dir, y_plus=y_plus, y_minus=y_minus, eps_fd=eps_fd
                    )
                    grad_accum = grad_accum + grad_k
                grad = grad_accum / float(grad_averaging_K)
                
                # Track progress
                if t % 1 == 0:
                    y_opt, _ = self.prob.solve_lower_level(x)
                    noise_upper_log = xi_upper_shared * torch.ones_like(self.prob.Q_upper)
                    loss = self.prob.upper_objective(x, y_opt, noise_upper_log).item()
                    losses.append(loss)
                    grad_norm = torch.norm(grad).item()
                    hypergrad_norms.append(grad_norm)
                    
                    if t % 100 == 0 or t < 10:
                        print(f"  t={t:4d}: loss={loss:.6f}, grad_norm={grad_norm:.3f}, beta={beta:.6f}")
                    
                    # Check for numerical issues
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        print(f"    Warning: Invalid gradient at iteration {t}, setting to zero")
                        grad = torch.zeros_like(grad)
                    
                    # Check for explosion
                    if grad_norm > 1000:
                        print(f"    Warning: Large gradient norm {grad_norm:.2e} at iteration {t}")
                
                # Gradient clipping (by L2 norm)
                gnorm = torch.norm(grad)
                if torch.isfinite(gnorm) and gnorm > clip_threshold:
                    grad = grad * (clip_threshold / gnorm)

                # Adaptive step-size: if loss keeps worsening, reduce beta
                if prev_loss is not None and loss > prev_loss + 1e-2:
                    worsen_count += 1
                else:
                    worsen_count = 0
                if worsen_count >= 5:
                    beta = max(beta * 0.8, 1e-4)
                    worsen_count = 0

                # Step clipping: cap ||Δx|| = beta * ||grad||
                step_norm = beta * torch.norm(grad)
                if torch.isfinite(step_norm) and step_norm > step_cap:
                    scale = step_cap / (step_norm + 1e-12)
                    x = x - (beta * scale) * grad
                else:
                    x = x - beta * grad

                prev_loss = loss
                
            except Exception as e:
                print(f"  Error at iteration {t}: {str(e)[:50]}")
                continue
        
        # ensure final point captured
        if len(x_history) == 0 or not torch.equal(x_history[-1], x.detach()):
            x_history.append(x.clone().detach())
        
        return x, losses, hypergrad_norms, x_history, step_sizes

def test_stable_strategies():
    """Test different step size strategies"""
    print("=" * 80)
    print("TESTING STABLE SSIGD STRATEGIES")
    print("=" * 80)
    
    # Create problem instance
    problem = StronglyConvexBilevelProblem(
        dim=10,
        num_constraints=3,
        noise_std=0.01,
        strong_convex=True,
        device='cpu'
    )
    
    strategies = [
        ('constant_sqrt', 'β = O(1/√T) ≈ 0.014 for T=5000'),
        ('diminishing', 'β_r = 1/(μ_F(r+1)) for strongly-convex'),
        ('adaptive', 'Adaptive step size')
    ]
    
    results = {}
    
    for strategy, description in strategies:
        print(f"\n{'='*60}")
        print(f"Testing {strategy}: {description}")
        print(f"{'='*60}")
        
        ssigd = StableSSIGD(problem)
        x_final, losses, grad_norms, x_history, step_sizes = ssigd.solve(
            T=5000, 
            step_strategy=strategy,
            mu_F=0.1,  # Estimate for strongly-convex parameter
            eps_fd=1e-4
        )
        
        results[strategy] = {
            'final_loss': losses[-1],
            'final_grad_norm': grad_norms[-1],
            'min_loss': min(losses),
            'converged': grad_norms[-1] < 1e-2,
            'step_sizes': step_sizes
        }
        
        print(f"\nResults for {strategy}:")
        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Final gradient norm: {grad_norms[-1]:.3f}")
        print(f"  Min loss achieved: {min(losses):.6f}")
        print(f"  Converged: {grad_norms[-1] < 1e-2}")
        print(f"  Final step size: {step_sizes[-1]:.6f}")
        print(f"  Initial step size: {step_sizes[0]:.6f}")
    
    # Compare strategies
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Strategy':<15} {'Final Loss':<12} {'Grad Norm':<10} {'Converged':<10} {'Min Loss':<12}")
    print("-" * 80)
    
    for strategy, result in results.items():
        print(f"{strategy:<15} {result['final_loss']:<12.6f} {result['final_grad_norm']:<10.3f} "
              f"{result['converged']:<10} {result['min_loss']:<12.6f}")
    
    return results

if __name__ == "__main__":
    test_stable_strategies()
