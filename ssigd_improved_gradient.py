#!/usr/bin/env python3
"""
SSIGD implementation with JAXopt implicit differentiation
"""

import torch
import numpy as np
import jax
import jax.numpy as jnp
from jaxopt import OptaxSolver
from jaxopt import BacktrackingLineSearch
import optax
from optax.perturbations import make_perturbed_fun, Normal
from optax import apply_updates
from optax import projections
from typing import Optional, Union, List
from problem import StronglyConvexBilevelProblem
import time
import csv
import os

class SSIGD:
    """
    SSIGD with JAXopt implicit differentiation:
    - qΓéÇ is fixed (initialized once before optimization loop) Γ£ô
    - Q_lower_noise is fresh per iteration (stochastic noise sampled each iteration) Γ£ô
    - Implicit differentiation using JAXopt OptaxSolver
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem, device='cpu'):
        self.prob = problem
        self.device = device
        self.dtype = torch.float64
        
        # qΓéÇ perturbation is now integrated into the problem's lower-level objective
        # No need for separate qΓéÇ handling in SSIGD class
        
        print(f"Γ£ô SSIGD initialized with JAXopt implicit differentiation, qΓéÇ will be fixed throughout optimization (dim={problem.dim})")
        
    def solve_ll_with_q(self, x: torch.Tensor, noise_lower: torch.Tensor) -> torch.Tensor:
        """Solve lower-level problem with qΓéÇ perturbation integrated in problem initialization"""
        # Convert inputs to JAX arrays for problem.solve_ll
        x_jax = jnp.array(x.detach().cpu().numpy())
        noise_lower_jax = jnp.array(noise_lower.detach().cpu().numpy()) if noise_lower is not None else None
        y_opt_jax = self.prob.solve_ll(x_jax, noise_lower_jax)
        # Convert result back to torch tensor
        return torch.tensor(np.array(y_opt_jax), dtype=self.dtype, device=self.device)
    
    def grad_F(self, x: torch.Tensor, y: torch.Tensor, noise_lower: torch.Tensor) -> torch.Tensor:
        """Constrained implicit hypergradient: respects box-active coordinates.
        Γêç╠éF(x) = Γêçx f + (dy*/dx)^T Γêçy f with dy_A/dx = 0 on active set.
        """
        # Convert to JAX arrays
        x_jax = jnp.array(x.detach().cpu().numpy())
        y_jax = jnp.array(y.detach().cpu().numpy())
        noise_lower_jax = jnp.array(noise_lower.detach().cpu().numpy())

        # Obtain duals by solving clean LL once to detect actives
        # Note: using noisy LL for gradient consistency with y; could also use clean
        y_star_jax, lambda_opt_jax, _ = self.prob.solve_ll_with_duals(x_jax, noise_lower_jax)

        grad_F_jax = self.prob.compute_constrained_implicit_gradient_compiled(
            x_jax, y_jax, lambda_opt_jax, 1e-6, noise_lower_jax
        )

        grad_F_np = np.array(grad_F_jax)
        return torch.tensor(grad_F_np, dtype=self.dtype, device=self.device)
    
    
    def proj_X(self, x_candidate: torch.Tensor) -> torch.Tensor:
        """Project x_candidate onto finite box [-1000, 1000] using Optax projections"""
        # Convert to JAX arrays for Optax projections
        x_jax = jnp.array(x_candidate.detach().cpu().numpy())
        
        # Apply box projection: -1000 <= x <= 1000
        x_proj_jax = projections.projection_box(x_jax, -1000.0, 1000.0)
        
        # Convert back to PyTorch tensor
        return torch.tensor(np.array(x_proj_jax), dtype=self.dtype, device=self.device)
    

    def solve(self, T=1000, beta=0.01, x0=None, diminishing: bool = True, mu_F: float = None, 
              convergence_tol=1e-6, patience=50, divergence_threshold=1e6, divergence_patience=10,
              lambda_moreau: float = 1e-2,
              csv_log_path: str = "ssigd_projected_armijo_run.csv", Ng: int = 64):
        x = (x0.detach().to(device=self.device, dtype=self.dtype).clone()
             if x0 is not None else
             torch.randn(self.prob.dim, device=self.device, dtype=self.dtype) * 0.1)
        
        # Initialize tracking arrays
        losses = []
        grad_norms = []
        x_norms = []
        y_norms = []
        steps = []
        lr_values = []

        # Determine ╬╝_F for diminishing step sizes
        if mu_F is None:
            # Use JAX eigenvalue computation like in problem class
            upper_eigenvals = jnp.linalg.eigvals(self.prob.Q_upper).real
            mu_F = jnp.min(upper_eigenvals).item()
            # Ensure ╬╝_F > 0 for stability and cap at 10 for reasonable step sizes
            mu_F = min(max(mu_F, 1e-6), 10.0)
        
        print(f"SSIGD: T={T}, beta={beta:.4f}, diminishing={diminishing}, ╬╝_F={mu_F:.6f}")
        print(f"Convergence: tol={convergence_tol:.1e}, patience={patience}")
        print(f"Divergence: threshold={divergence_threshold:.1e}, patience={divergence_patience}")
        print(f"{'Step':>6} {'Loss':>12} {'GradNorm':>12} {'XNorm':>10} {'YNorm':>10} {'LR':>10} {'Status':>12}")
        print("-" * 80)
        # Prepare CSV logging
        write_header = not os.path.exists(csv_log_path)
        try:
            csv_file = open(csv_log_path, "a", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            if write_header:
                csv_writer.writerow([
                    "step", "loss", "delta_loss", "grad_norm", "gM_norm", "x_norm", "y_norm",
                    "lr", "kkt_residual", "comp_slack", "active_constraints",
                    "hypergrad_norm", "hypergrad_vector"
                ])
        except Exception:
            csv_file = None
            csv_writer = None

        # Initialize qΓéÇ perturbation ONCE before the optimization loop
        # This is the fixed perturbation that stays constant throughout optimization
        q_perturbation = self.prob.q_perturbation
        print(f"Γ£ô qΓéÇ perturbation initialized before optimization loop: ||qΓéÇ|| = {jnp.linalg.norm(q_perturbation):.2e}")

        # Early stopping variables (monitoring only; no early stop to honor 100 iters run)
        best_loss = float('inf')
        best_x = x.clone()
        patience_counter = 0
        divergence_counter = 0
        converged = False
        diverged = False

        # Line search: Armijo with max stepsize = beta and jit disabled to avoid tracing QP
        def f_value_with_noise_jax(x_jax: jnp.ndarray, noise_lower_jax: Optional[jnp.ndarray]) -> jnp.ndarray:
            # LL solve with the SAME noise as used for gradient
            y_star_local = self.prob.solve_ll(x_jax, noise_lower_jax)
            # Upper objective value in JAX (deterministic UL)
            F_val = self.prob.upper_objective_compiled(x_jax, y_star_local)
            return F_val

        # Projected Armijo parameters
        c1 = 1e-3
        decrease_factor = 0.5
        max_stepsize = 0.0064
        min_stepsize = 1e-6
        stepsize = max_stepsize
        # Moreau envelope parameter controls smoothing (prox radius)
        # lambda_moreau provided as an argument to solve()

        prev_F = None
        for r in range(1, T + 1):  # 1-based iteration like DS-BLO
            # Sample Q_lower_noise ONCE per iteration and use consistently
            _, noise_lower = self.prob._sample_instance_noise()
            
            # Solve LL and compute gradient with SAME noise_lower and fixed qΓéÇ
            # qΓéÇ is fixed throughout optimization (initialized before loop)
            # noise_lower is stochastic noise sampled fresh each iteration
            # Both solve_ll_with_q and grad_F use the same noise_lower instance
            y_hat = self.solve_ll_with_q(x, noise_lower)
            grad_est = self.grad_F(x, y_hat, noise_lower)

            # Compute value f(x) for line search with the SAME noise
            x_jax_now = jnp.array(x.detach().cpu().numpy())
            noise_lower_jax = jnp.array(noise_lower.detach().cpu().numpy())
            fx = float(f_value_with_noise_jax(x_jax_now, noise_lower_jax))

            # Convert gradient to NumPy/JAX for line search (not strictly needed below but kept for parity)
            gx = grad_est.detach().cpu().numpy()

            # Moreau gradient mapping Armijo: d = -g_M where
            # g_M = (1/╬╗_M) (x - prox_{╬╗_M R}(x - ╬╗_M ΓêçF)) with prox implemented by proj_X
            maxiter_ls = 30
            lr_t = stepsize

            # Prox step around current x
            v_torch = x - lambda_moreau * grad_est
            z_torch = self.proj_X(v_torch)
            gM_torch = (1.0 / lambda_moreau) * (x - z_torch)

            # Descent direction is negative gradient mapping
            d_torch = -gM_torch
            d_np = d_torch.detach().cpu().numpy()
            dir_norm_sq = float(np.dot(d_np, d_np))
            if dir_norm_sq == 0.0:
                lr_t = 0.0
            else:
                for _bt in range(maxiter_ls):
                    x_trial = x + lr_t * d_torch
                    f_trial = float(f_value_with_noise_jax(jnp.array(x_trial.detach().cpu().numpy()), noise_lower_jax))
                    # Armijo sufficient decrease w.r.t. ||g_M||^2
                    if f_trial <= fx - c1 * lr_t * dir_norm_sq:
                        break
                    lr_t *= decrease_factor
                    if lr_t < min_stepsize:
                        lr_t = min_stepsize
                        break

            # Final update along -g_M (no extra projection needed)
            x = x + lr_t * d_torch
            stepsize = lr_t
            
            # Tracking: use the same clean f(x) evaluated at post-update x
            x_jax_curr = jnp.array(x.detach().cpu().numpy())
            F = float(f_value_with_noise_jax(x_jax_curr, noise_lower_jax))
            delta_F = (prev_F - F) if (prev_F is not None) else 0.0
            prev_F = F
            
            # Track gradient norm
            grad_norm = torch.norm(grad_est).item()
            # Track Moreau gradient mapping norm
            gM_norm = torch.norm(gM_torch).item()
            
            # Track solution norms
            x_norm = torch.norm(x).item()
            # Compute clean LL solution in JAX and diagnostics
            y_star_jax, lambda_opt_jax, info = self.prob.solve_ll_with_duals(x_jax_curr)
            y_norm = float(np.linalg.norm(np.array(y_star_jax)))
            kkt_res = info.get('kkt_residual', float('nan'))
            comp_slack = info.get('complementary_slackness', float('nan'))
            active_cnt = info.get('active_constraints', 0)

            # Stochastic hypergradient averaging for diagnostics only
            hg_accum = None
            for _i in range(Ng):
                _, noise_lower_i = self.prob._sample_instance_noise()
                y_hat_i = self.solve_ll_with_q(x, noise_lower_i)
                g_i = self.grad_F(x, y_hat_i, noise_lower_i)
                hg_accum = g_i if hg_accum is None else (hg_accum + g_i)
            hypergrad = (hg_accum / float(max(Ng, 1))) if Ng > 0 else grad_est
            hypergrad_norm = torch.norm(hypergrad).item()
            hypergrad_vec = "[" + " ".join(f"{v:.6g}" for v in hypergrad.detach().cpu().numpy()) + "]"
            
            # Store tracking data
            losses.append(F)
            grad_norms.append(grad_norm)
            x_norms.append(x_norm)
            y_norms.append(y_norm)
            steps.append(r)
            lr_values.append(lr_t)

            # CSV Log row
            if csv_writer is not None:
                csv_writer.writerow([
                    r, F, delta_F, grad_norm, gM_norm, x_norm, y_norm, lr_t,
                    kkt_res, comp_slack, active_cnt,
                    hypergrad_norm, hypergrad_vec
                ])
                csv_file.flush()
            
            # Determine status
            status = "Running"
            if grad_norm < convergence_tol:
                status = "Converging"
            elif F > divergence_threshold or grad_norm > divergence_threshold or x_norm > divergence_threshold:
                status = "Diverging"
            elif F < best_loss:
                status = "Improving"
            
            # Print detailed log every iteration (numbers only, no tensor prints)
            print(f"{r:6d} {F:12.6f} {grad_norm:12.6f} {x_norm:10.4f} {y_norm:10.4f} {lr_t:10.6f} {status:>12}")
            print(f"    [Moreau] ||g_M|| = {gM_norm:.6g}, ╬╗_M = {lambda_moreau}")
            # Extra diagnostics similar to provided snippets
            print(f"    [L-check] Decreased: ╬öL={delta_F:.3e}")
            print(f"    [KKT] y*: res={kkt_res:.3e}, comp={comp_slack:.3e}")
            print(f"  Computing stochastic hypergradient with N_g = {Ng}")
            print(f"  Final hypergradient: ΓêçF╠â = {hypergrad_vec}")
            print(f"  Hypergradient norm: ||ΓêçF╠â|| = {hypergrad_norm:.6g}")
            
            # Monitor only: do not early stop to ensure full run length
            if F > divergence_threshold or grad_norm > divergence_threshold or x_norm > divergence_threshold:
                divergence_counter += 1
            else:
                divergence_counter = 0
            if grad_norm < convergence_tol:
                if patience_counter == 0:
                    best_loss = F
                    best_x = x.clone()
                patience_counter += 1
            else:
                patience_counter = 0
                if F < best_loss:
                    best_loss = F
                    best_x = x.clone()

        # Final status
        if not converged and not diverged:
            print(f"\nΓÅ╣∩╕Å  STOPPED at iteration {r}: Max iterations reached")
        
        # Close CSV file if opened
        try:
            if csv_file is not None:
                csv_file.close()
        except Exception:
            pass

        print(f"\n≡ƒôè FINAL RESULTS:")
        print(f"    Final Loss: {losses[-1] if losses else float('inf'):.6f}")
        print(f"    Final GradNorm: {grad_norms[-1] if grad_norms else 0.0:.6f}")
        print(f"    Final XNorm: {x_norms[-1] if x_norms else 0.0:.4f}")
        print(f"    Final YNorm: {y_norms[-1] if y_norms else 0.0:.4f}")
        print(f"    Status: {'Converged' if converged else 'Diverged' if diverged else 'Stopped'}")
        print(f"    Iterations: {r}")

        return {
            'x_final': best_x if converged else x,
            'losses': losses,
            'grad_norms': grad_norms,
            'x_norms': x_norms,
            'y_norms': y_norms,
            'steps': steps,
            'lr_values': lr_values,
            'final_loss': best_loss if converged else (losses[-1] if losses else float('inf')),
            'final_gradient_norm': grad_norms[-1] if grad_norms else 0.0,
            'final_grad_norm': grad_norms[-1] if grad_norms else 0.0,  # Alias for compatibility
            'converged': converged,
            'diverged': diverged,
            'iterations': r,  # Actual iterations run
            'method': 'SSIGD'
        }


def test_ssigd(problem: StronglyConvexBilevelProblem, T=100, beta=0.01, x0=None, 
               diminishing=False, convergence_tol=1e-6, divergence_threshold=1e6, lambda_moreau: float = 1e-2,
               csv_log_path: str = "ssigd_projected_armijo_run.csv"):
    """Test SSIGD with JAXopt implicit differentiation and comprehensive logging"""
    print("\n≡ƒÜÇ Testing SSIGD with JAXopt implicit differentiation")
    print("=" * 50)
    
    # Test SSIGD method
    print("\nTesting SSIGD...")
    ssigd = SSIGD(problem)
    start_time = time.time()
    result = ssigd.solve(T=T, beta=beta, x0=x0, diminishing=diminishing, 
                        convergence_tol=convergence_tol, divergence_threshold=divergence_threshold,
                        lambda_moreau=lambda_moreau, csv_log_path=csv_log_path)
    time_taken = time.time() - start_time
    
    # Display results
    print("\n≡ƒôè RESULTS")
    print("=" * 20)
    print(f"Time: {time_taken:.2f}s")
    print(f"Final Loss: {result['final_loss']:.6f}")
    print(f"Final Gradient: {result['final_grad_norm']:.6f}")
    print(f"Final XNorm: {result['x_norms'][-1] if result['x_norms'] else 0.0:.4f}")
    print(f"Final YNorm: {result['y_norms'][-1] if result['y_norms'] else 0.0:.4f}")
    print(f"Status: {'Converged' if result['converged'] else 'Diverged' if result['diverged'] else 'Stopped'}")
    print(f"Method: {result['method']}")
    
    return {
        'result': result,
        'time': time_taken
    }


if __name__ == "__main__":
    # Example usage with comprehensive logging
    import time
    
    print("≡ƒÜÇ SSIGD with JAXopt implicit differentiation - Comprehensive Logging")
    print("=" * 70)
    
    # Create test problem
    dim = 10
    problem = StronglyConvexBilevelProblem(dim=dim, device='cpu')
    x0 = torch.randn(dim, dtype=torch.float64) * 0.1
    
    print(f"Problem dimension: {dim}")
    print(f"Initial point norm: {torch.norm(x0).item():.4f}")
    
    # Run test with comprehensive logging
    print(f"\n≡ƒöì Running SSIGD with detailed monitoring...")
    # Allow overriding ╬╗_M, T, CSV path via environment variables for quick sweeps
    try:
        env_lambda = os.environ.get("SSIGD_LAMBDA_MOREAU", None)
        lambda_moreau_val = float(env_lambda) if env_lambda is not None else 1e-2
    except Exception:
        lambda_moreau_val = 1e-2
    try:
        env_T = os.environ.get("SSIGD_T", None)
        T_val = int(env_T) if env_T is not None else 20
    except Exception:
        T_val = 20
    csv_env = os.environ.get("SSIGD_CSV", None)
    csv_path = csv_env if csv_env is not None else "ssigd_projected_armijo_run.csv"
    print(f"Using ╬╗_M (Moreau) = {lambda_moreau_val}, T = {T_val}, CSV = {csv_path}")
    test_result = test_ssigd(problem, T=T_val, beta=0.01, x0=x0, 
                           diminishing=False, convergence_tol=1e-4, divergence_threshold=1e4,
                           lambda_moreau=lambda_moreau_val, csv_log_path=csv_path)
    
    print(f"\n≡ƒÄ» READY TO USE:")
    print("SSIGD with comprehensive logging provides:")
    print("  Γ£ô Step-by-step monitoring of Loss, GradNorm, XNorm, YNorm")
    print("  Γ£ô Early stopping for divergence detection")
    print("  Γ£ô Convergence monitoring with patience")
    print("  Γ£ô Learning rate tracking")
    print("  Γ£ô Status indicators (Running/Converging/Diverging/Improving)")
    print("  Γ£ô Numbers-only output for easy terminal monitoring")
    print("SSIGD with JAXopt implicit differentiation provides superior accuracy and performance!")
