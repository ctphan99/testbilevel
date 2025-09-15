#!/usr/bin/env python3
"""
F2CSA Algorithm 2 - WORKING Implementation
Uses the correct hypergradient computation from Algorithm 1
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Optional, Tuple
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
import warnings

warnings.filterwarnings('ignore')

class F2CSAAlgorithm2Working:
    """
    F2CSA Algorithm 2: Nonsmooth Nonconvex Algorithm with Inexact Stochastic Hypergradient Oracle
    WORKING implementation using correct hypergradient computation
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem, device: str = 'cpu', dtype=torch.float64):
        self.problem = problem
        self.device = device
        self.dtype = dtype
        
        # Initialize Algorithm 1 for hypergradient computation
        self.algorithm1 = F2CSAAlgorithm1Final(problem, device=device, dtype=dtype)
        
        # Persistent cache for warm-start
        self.prev_y = None
        self.prev_lambda = None
        self.prev_adam_state = None
        
    def clip_D(self, v: torch.Tensor, D: float) -> torch.Tensor:
        """
        Clipping function: clip_D(v) := min{1, D/||v||} * v
        """
        v_norm = torch.norm(v)
        if v_norm <= D:
            return v
        else:
            return (D / v_norm) * v
    
    def optimize(self, x0: torch.Tensor, T: int, D: float, eta: float, 
                 delta: float, alpha: float, N_g: int = None, 
                 warm_ll: bool = False, keep_adam_state: bool = False,
                 plot_name: str = None, save_warm_name: str = None) -> Dict:
        """
        Run F2CSA Algorithm 2 optimization with WORKING hypergradient computation
        """
        print("🚀 F2CSA Algorithm 2 - WORKING Implementation")
        print("=" * 70)
        print(f"T = {T}, D = {D:.6f}, η = {eta:.6f}")
        print(f"δ = {delta:.6f}, α = {alpha:.6f}")
        print()
        
        # Set default N_g if not provided
        if N_g is None:
            # Balanced N_g for working implementation
            N_g = max(10, min(100, int(1.0 / (alpha**1.5))))
        
        print(f"Using N_g = {N_g} samples for hypergradient estimation")
        print()
        
        # Initialize
        x = x0.clone().detach()
        Delta = torch.zeros_like(x)
        
        # Storage for results
        z_history = []
        x_history = []
        g_history = []
        Delta_history = []
        ul_losses = []
        hypergrad_norms = []
        
        print("Starting optimization...")
        print("-" * 50)
        
        # Main optimization loop
        current_eta = eta
        decay_every = max(50, T // 10)
        decay_factor = 0.95
        window = 50
        ul_tol = 1e-3
        hg_tol = 1e-2
        
        for t in range(1, T + 1):
            # Sample s_t ~ Unif[0, 1]
            s_t = torch.rand(1, device=self.device, dtype=self.dtype).item()
            
            # Update x_t and z_t
            x_t = x + Delta
            z_t = x + s_t * Delta
            
            # Compute hypergradient using Algorithm 1 (WORKING version) with warm-start
            if warm_ll and (self.prev_y is not None or self.prev_lambda is not None):
                # Pass warm-start information to Algorithm 1
                oracle_result = self.algorithm1.oracle_sample(z_t, alpha, N_g, 
                                                             prev_y=self.prev_y, 
                                                             prev_lambda=self.prev_lambda,
                                                             keep_adam_state=keep_adam_state)
            else:
                oracle_result = self.algorithm1.oracle_sample(z_t, alpha, N_g)
            
            # Extract hypergradient from oracle result (returns tuple: grad, y, lambda)
            g_t = oracle_result[0] if isinstance(oracle_result, tuple) else oracle_result

            # Compute upper-level loss at x_t to monitor Algorithm 2 gap (f(x, y*))
            y_star, _ = self.problem.solve_lower_level(x_t)
            ul_loss_t = self.problem.upper_objective(x_t, y_star).item()
            ul_losses.append(ul_loss_t)

            # Update direction with clipping
            Delta = self.clip_D(Delta - current_eta * g_t, D)
            
            # Decay step size periodically
            if t % decay_every == 0:
                current_eta = max(current_eta * decay_factor, eta * 0.2)
            
            # Early stopping: UL loss and hypergrad norm stabilization
            if len(ul_losses) >= window:
                ul_recent = ul_losses[-window:]
                hg_recent = hypergrad_norms[-window:]
                ul_span = max(ul_recent) - min(ul_recent)
                hg_span = max(hg_recent) - min(hg_recent)
                if ul_span < ul_tol and hg_span < hg_tol:
                    print(f"Early stop at iter {t}: UL loss and hypergrad stabilized (spans {ul_span:.3e}, {hg_span:.3e})")
                    T = t
                    break
            
            # Store history
            z_history.append(z_t.clone().detach())
            x_history.append(x_t.clone().detach())
            g_history.append(g_t.clone().detach())
            hypergrad_norms.append(torch.norm(g_t).item())
            Delta_history.append(Delta.clone().detach())
            
            # Update x for next iteration
            x = x_t
            
            # Cache lower-level solution for warm-start (if enabled)
            if warm_ll:
                # Get the lower-level solution from the oracle result
                if isinstance(oracle_result, tuple) and len(oracle_result) >= 3:
                    self.prev_y = oracle_result[1].clone().detach()  # y_tilde
                    self.prev_lambda = oracle_result[2].clone().detach()  # lambda_star
                
                # Cache Adam state if requested
                if keep_adam_state and hasattr(self.algorithm1, 'adam_state'):
                    self.prev_adam_state = self.algorithm1.adam_state.copy() if self.algorithm1.adam_state else None
            
            # Print progress
            if t % max(1, T // 20) == 0 or t <= 10:
                g_norm = torch.norm(g_t).item()
                Delta_norm = torch.norm(Delta).item()
                print(f"Iteration {t:4d}/{T}: ||g_t|| = {g_norm:.6f}, ||Δ_t|| = {Delta_norm:.6f}, ul_loss = {ul_loss_t:.6f}")
        
        print()
        print("Computing output points...")
        
        # Group iterations for Goldstein subdifferential (per spec): K = max(1, total_iters // M)
        total_iters = len(z_history)
        M = max(1, int(delta / D))
        K = max(1, total_iters // M)
        
        print(f"M = {M}, K = {K}")
        
        # Compute candidate points and their upper-level losses f(x, y*) for all K blocks
        candidates = []
        for k in range(1, K + 1):
            start_idx = (k - 1) * M
            end_idx = min(k * M, len(z_history))
            if start_idx < len(z_history):
                z_group = z_history[start_idx:end_idx]
                x_k = torch.stack(z_group).mean(dim=0)
                y_star_k, _ = self.problem.solve_lower_level(x_k)
                ul_loss_k = self.problem.upper_objective(x_k, y_star_k).item()
                candidates.append((x_k, ul_loss_k))
        
        if candidates:
            # Choose the candidate with the smallest upper-level loss
            x_out, best_ul_loss = min(candidates, key=lambda t: t[1])
            print(f"Selected output with min f(x, y*) among {len(candidates)}: f = {best_ul_loss:.6f}")
        else:
            x_out = x_history[-1] if x_history else x0
            print("No candidates formed; falling back to last x.")
        print()
        
        # Compute final gradient norm
        final_oracle_result = self.algorithm1.oracle_sample(x_out, alpha, N_g)
        final_g = final_oracle_result[0] if isinstance(final_oracle_result, tuple) else final_oracle_result
        final_g_norm = torch.norm(final_g).item()
        
        # Compute final upper-level loss f(x, y*) as Algorithm 2 gap
        y_star, _ = self.problem.solve_lower_level(x_out)
        final_ul_loss = self.problem.upper_objective(x_out, y_star).item()
        
        print(f"Final UL loss f(x, y*): {final_ul_loss:.6f}")
        print(f"Final hypergradient norm (diagnostic): {final_g_norm:.6f}")
        print(f"Final point: {x_out}")
        print()
        
        # Plot hypergradient norm and UL loss over iterations
        try:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(hypergrad_norms, color='tab:blue', label='||∇F̃||')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Hypergradient norm', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax2 = ax1.twinx()
            ax2.plot(ul_losses, color='tab:orange', label='f(x, y*)')
            ax2.set_ylabel('UL loss f(x, y*)', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            plt.title('Algorithm 2: Hypergradient norm and UL loss over iterations')
            fig.tight_layout()
            if plot_name is None:
                plot_name = 'algo2_hg_ul_loss.png'
            plt.savefig(plot_name, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved plot to {plot_name}")
        except Exception as e:
            print(f"Plotting failed: {e}")
        
        # Save warm start for next runs
        try:
            if save_warm_name is None:
                save_warm_name = 'algo2_warmstart.npy'
            np.save(save_warm_name, x_out.detach().cpu().numpy())
            print(f"Saved warm start to {save_warm_name}")
        except Exception as e:
            print(f"Failed to save warm start: {e}")
        
        return {
            'x_out': x_out,
            'final_gradient': final_g,
            'final_gradient_norm': final_g_norm,
            'final_ul_loss': final_ul_loss,
            'y_star': y_star,
            'x_history': x_history,
            'z_history': z_history,
            'g_history': g_history,
            'Delta_history': Delta_history,
            'hypergrad_norms': hypergrad_norms,
            'ul_losses': ul_losses,
            'converged': final_g_norm < 1e-3,
            'iterations': T
        }

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='F2CSA Algorithm 2 with warm-start support')
    parser.add_argument('--T', type=int, default=5000, help='Number of iterations')
    parser.add_argument('--D', type=float, default=0.05, help='Clipping parameter')
    parser.add_argument('--eta', type=float, default=0.0001, help='Step size')
    parser.add_argument('--Ng', type=int, default=64, help='Number of gradient samples')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha parameter')
    parser.add_argument('--warm-ll', action='store_true', help='Enable lower-level warm-start')
    parser.add_argument('--keep-adam-state', action='store_true', help='Keep Adam optimizer state')
    parser.add_argument('--plot-name', type=str, default=None, help='Plot filename')
    parser.add_argument('--save-warm-name', type=str, default=None, help='Warm start save filename')
    parser.add_argument('--dim', type=int, default=5, help='Problem dimension')
    parser.add_argument('--constraints', type=int, default=3, help='Number of constraints')
    
    args = parser.parse_args()
    
    # Initialize problem
    problem = StronglyConvexBilevelProblem(
        dim=args.dim, 
        num_constraints=args.constraints, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    algorithm2 = F2CSAAlgorithm2Working(problem)
    
    # Parameters
    alpha = args.alpha
    delta = alpha**3
    
    # Warm start if available
    warm_path = 'algo2_warmstart.npy'
    try:
        x0 = torch.tensor(np.load(warm_path), dtype=torch.float64)
        print(f"Loaded warm start from {warm_path}: x0 shape = {tuple(x0.shape)}")
    except Exception:
        x0 = torch.randn(args.dim, dtype=torch.float64)
        print("No warm start found; using random x0.")
    
    print(f"Test point x0: {x0}")
    print(f"α = {alpha}")
    print(f"T = {args.T}, D = {args.D}, η = {args.eta}")
    print(f"δ = {delta:.6f}, N_g = {args.Ng}")
    print(f"Warm-LL: {args.warm_ll}, Keep Adam: {args.keep_adam_state}")
    print()
    
    # Run Algorithm 2 optimization
    results = algorithm2.optimize(
        x0, args.T, args.D, args.eta, delta, alpha, args.Ng,
        warm_ll=args.warm_ll, keep_adam_state=args.keep_adam_state,
        plot_name=args.plot_name, save_warm_name=args.save_warm_name
    )
    
    # Check convergence
    hg_norms = results['hypergrad_norms']
    print(f"Gradient norms history (first 10): {[f'{g:.1f}' for g in hg_norms[:10]]}")
    print(f"Gradient norms history (last 10): {[f'{g:.1f}' for g in hg_norms[-10:]]}")
    
    # Check for stabilization
    if len(hg_norms) >= 10:
        last_10_norms = np.array(hg_norms[-10:])
        std_dev = np.std(last_10_norms)
        mean_norm = np.mean(last_10_norms)
        print(f"Last 10 gradient norms std dev: {std_dev:.4f}")
        print(f"Last 10 gradient norms range: {np.max(last_10_norms) - np.min(last_10_norms):.4f}")
        if std_dev < 2.0 and (np.max(last_10_norms) - np.min(last_10_norms)) < 5.0:
            print("Status: ✅ CONVERGED (based on heuristic)")
        else:
            print("Status: ❌ NOT CONVERGED (based on heuristic)")
    else:
        print("Not enough iterations to check stability.")
