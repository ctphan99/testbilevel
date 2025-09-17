#!/usr/bin/env python3
"""
F2CSA Optimizer Test: Compare Adam vs SGD (no momentum)
Test whether removing momentum helps with hypergradient accuracy
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Optional, Tuple
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
import warnings

warnings.filterwarnings('ignore')

class F2CSAOptimizerTest:
    """
    F2CSA Algorithm 2 with configurable optimizer (Adam vs SGD)
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem, device: str = 'cpu', dtype=torch.float64):
        self.problem = problem
        self.device = device
        self.dtype = dtype
        
        # Initialize Algorithm 1 for hypergradient computation
        self.algorithm1 = F2CSAAlgorithm1Final(problem, device=device, dtype=dtype)
        
        # Persistent cache for two-level warm-start
        self.prev_y = None
        self.prev_lambda = None
        self.prev_y_lamb_opt = None
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
                 optimizer_type: str = 'adam',
                 warm_ll: bool = False, keep_adam_state: bool = False,
                 plot_name: str = None, save_warm_name: str = None) -> Dict:
        """
        Run F2CSA Algorithm 2 optimization with configurable optimizer
        
        Args:
            optimizer_type: 'adam' or 'sgd' (SGD with momentum=0)
        """
        print(f"🚀 F2CSA Algorithm 2 - Optimizer Test ({optimizer_type.upper()})")
        print("=" * 70)
        print(f"T = {T}, D = {D:.6f}, η = {eta:.6f}")
        print(f"δ = {delta:.6f}, α = {alpha:.6f}")
        print(f"Optimizer: {optimizer_type.upper()}")
        print()
        
        # Set default N_g if not provided
        if N_g is None:
            N_g = max(10, min(100, int(1.0 / (alpha**1.5))))
        
        print(f"Using N_g = {N_g} samples for hypergradient estimation")
        print()
        
        # Initialize x
        x = x0.clone().detach().requires_grad_(True)
        
        # Choose optimizer based on type
        if optimizer_type.lower() == 'sgd':
            # SGD with no momentum
            optimizer = torch.optim.SGD([x], lr=eta, momentum=0)
            print("Using SGD with NO MOMENTUM for potentially more accurate hypergradients")
        else:
            # Default Adam
            optimizer = torch.optim.Adam([x], lr=eta)
            print("Using Adam (with momentum) - standard approach")
        
        # Storage for results
        z_history = []
        x_history = []
        g_history = []
        ul_losses = []
        hypergrad_norms = []
        
        print("Starting optimization...")
        print("-" * 50)
        
        # Main optimization loop
        for t in range(1, T + 1):
            # No external perturbation: use x directly; stochasticity from instance noise
            xx = x
            
            # Get hypergradient from Algorithm 1
            if warm_ll and (self.prev_y is not None or self.prev_lambda is not None):
                oracle_result = self.algorithm1.oracle_sample(xx, alpha, N_g, 
                                                             prev_y=self.prev_y, 
                                                             prev_lambda=self.prev_lambda,
                                                             keep_adam_state=keep_adam_state)
            else:
                oracle_result = self.algorithm1.oracle_sample(xx, alpha, N_g)
            
            # Extract solutions from inner problem
            g_t = oracle_result[0] if isinstance(oracle_result, tuple) else oracle_result
            if isinstance(oracle_result, tuple) and len(oracle_result) >= 3:
                y_opt = oracle_result[1]  # Inner problem solution (y*)
                lambda_opt = oracle_result[2]  # Dual solution (λ*)
            else:
                # Fallback: solve inner problem directly
                y_opt, _ = self.problem.solve_lower_level(xx)
                lambda_opt = torch.zeros(self.problem.num_constraints, device=self.device, dtype=self.dtype)
            
            # Compute upper-level loss for monitoring
            ul_loss_t = self.problem.upper_objective(xx, y_opt).item()
            ul_losses.append(ul_loss_t)
            
            # Update with chosen optimizer 
            x.grad = g_t
            optimizer.step()
            optimizer.zero_grad()
            
            # Store history
            z_history.append(xx.clone().detach())
            x_history.append(x.clone().detach())
            g_history.append(g_t.clone().detach())
            hypergrad_norms.append(torch.norm(g_t).item())
            
            # Warm start persistence
            if warm_ll:
                self.prev_y = y_opt.clone().detach()
                self.prev_lambda = lambda_opt.clone().detach()
                
                if keep_adam_state and hasattr(self.algorithm1, 'adam_state'):
                    self.prev_adam_state = self.algorithm1.adam_state.copy() if self.algorithm1.adam_state else None
            
            # Print progress
            if t % max(1, T // 20) == 0 or t <= 10:
                g_norm = torch.norm(g_t).item()
                print(f"Iteration {t:4d}/{T}: ||g_t|| = {g_norm:.6f}, ul_loss = {ul_loss_t:.6f}")
        
        print()
        print("Computing final output...")
        
        # Final results
        x_out = x.clone().detach()
        
        # Compute final gradient norm
        final_oracle_result = self.algorithm1.oracle_sample(x_out, alpha, N_g)
        final_g = final_oracle_result[0] if isinstance(final_oracle_result, tuple) else final_oracle_result
        final_g_norm = torch.norm(final_g).item()
        
        # Compute final upper-level loss f(x, y*) as Algorithm 2 gap
        y_star, _ = self.problem.solve_lower_level(x_out)
        final_ul_loss = self.problem.upper_objective(x_out, y_star).item()
        
        print(f"Final UL loss f(x, y*): {final_ul_loss:.6f}")
        print(f"Final hypergradient norm: {final_g_norm:.6f}")
        print(f"Optimizer used: {optimizer_type.upper()}")
        print()
        
        # Plot results
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Hypergradient norm
            ax1.plot(hypergrad_norms, color='tab:blue', label=f'||∇F̃|| ({optimizer_type})')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Hypergradient norm')
            ax1.set_title(f'F2CSA with {optimizer_type.upper()}: Hypergradient norm over iterations')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # UL loss
            ax2.plot(ul_losses, color='tab:orange', label=f'f(x, y*) ({optimizer_type})')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('UL loss f(x, y*)')
            ax2.set_title(f'F2CSA with {optimizer_type.upper()}: UL loss over iterations')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if plot_name is None:
                plot_name = f'f2csa_{optimizer_type}_test.png'
            plt.savefig(plot_name, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved plot to {plot_name}")
        except Exception as e:
            print(f"Plotting failed: {e}")
        
        return {
            'optimizer_type': optimizer_type,
            'x_out': x_out,
            'final_gradient': final_g,
            'final_gradient_norm': final_g_norm,
            'final_ul_loss': final_ul_loss,
            'y_star': y_star,
            'x_history': x_history,
            'z_history': z_history,
            'g_history': g_history,
            'hypergrad_norms': hypergrad_norms,
            'ul_losses': ul_losses,
            'converged': final_g_norm < 1e-3,
            'iterations': T
        }

def run_comparison_test():
    """Run both Adam and SGD tests for comparison"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='F2CSA Optimizer Comparison Test')
    parser.add_argument('--T', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--D', type=float, default=0.08, help='Clipping parameter')
    parser.add_argument('--eta', type=float, default=2e-4, help='Step size')
    parser.add_argument('--Ng', type=int, default=32, help='Number of gradient samples')
    parser.add_argument('--alpha', type=float, default=0.6, help='Alpha parameter')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd', 'both'], default='both', 
                        help='Optimizer to test: adam, sgd, or both')
    parser.add_argument('--problem-noise-std', type=float, default=2e-3, help='Instance noise std')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize problem
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=args.problem_noise_std,
        strong_convex=True
    )
    
    # Parameters
    alpha = args.alpha
    delta = alpha**3
    
    # Initialize with same starting point for fair comparison
    x0 = torch.randn(5, dtype=torch.float64)
    print(f"Test starting point x0: {x0}")
    print(f"α = {alpha}, T = {args.T}, D = {args.D}, η = {args.eta}")
    print(f"δ = {delta:.6f}, N_g = {args.Ng}")
    print(f"Problem noise std: {args.problem_noise_std}")
    print(f"Random seed: {args.seed}")
    print()
    
    results = {}
    
    if args.optimizer in ['adam', 'both']:
        print("=" * 80)
        print("TESTING ADAM OPTIMIZER")
        print("=" * 80)
        
        algorithm_adam = F2CSAOptimizerTest(problem)
        results['adam'] = algorithm_adam.optimize(
            x0, args.T, args.D, args.eta, delta, alpha, args.Ng,
            optimizer_type='adam',
            plot_name='f2csa_adam_test.png'
        )
    
    if args.optimizer in ['sgd', 'both']:
        print("=" * 80)
        print("TESTING SGD OPTIMIZER (NO MOMENTUM)")
        print("=" * 80)
        
        algorithm_sgd = F2CSAOptimizerTest(problem)
        results['sgd'] = algorithm_sgd.optimize(
            x0, args.T, args.D, args.eta, delta, alpha, args.Ng,
            optimizer_type='sgd',
            plot_name='f2csa_sgd_test.png'
        )
    
    # Compare results if both were run
    if len(results) == 2:
        print("=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)
        
        adam_result = results['adam']
        sgd_result = results['sgd']
        
        print(f"Adam final UL loss:     {adam_result['final_ul_loss']:.6f}")
        print(f"SGD final UL loss:      {sgd_result['final_ul_loss']:.6f}")
        print(f"UL loss improvement:    {adam_result['final_ul_loss'] - sgd_result['final_ul_loss']:.6f}")
        print()
        
        print(f"Adam final grad norm:   {adam_result['final_gradient_norm']:.6f}")
        print(f"SGD final grad norm:    {sgd_result['final_gradient_norm']:.6f}")
        print(f"Grad norm difference:   {adam_result['final_gradient_norm'] - sgd_result['final_gradient_norm']:.6f}")
        print()
        
        # Compare hypergradient stability
        adam_hg_std = np.std(adam_result['hypergrad_norms'][-100:])  # Last 100 iterations
        sgd_hg_std = np.std(sgd_result['hypergrad_norms'][-100:])
        
        print(f"Adam hypergradient std (last 100 iter): {adam_hg_std:.6f}")
        print(f"SGD hypergradient std (last 100 iter):  {sgd_hg_std:.6f}")
        print(f"Stability improvement (lower is better): {adam_hg_std - sgd_hg_std:.6f}")
        print()
        
        # Plot comparison
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Hypergradient norms comparison
            ax1.plot(adam_result['hypergrad_norms'], color='tab:blue', label='Adam', alpha=0.8)
            ax1.plot(sgd_result['hypergrad_norms'], color='tab:red', label='SGD (no momentum)', alpha=0.8)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Hypergradient norm')
            ax1.set_title('Hypergradient Norm Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # UL loss comparison
            ax2.plot(adam_result['ul_losses'], color='tab:blue', label='Adam', alpha=0.8)
            ax2.plot(sgd_result['ul_losses'], color='tab:red', label='SGD (no momentum)', alpha=0.8)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('UL loss f(x, y*)')
            ax2.set_title('UL Loss Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('f2csa_optimizer_comparison.png', dpi=200, bbox_inches='tight')
            plt.close(fig)
            print("Saved comparison plot to f2csa_optimizer_comparison.png")
        except Exception as e:
            print(f"Comparison plotting failed: {e}")
        
        # Conclusion
        print("=" * 80)
        print("ANALYSIS")
        print("=" * 80)
        
        if sgd_result['final_ul_loss'] < adam_result['final_ul_loss']:
            print("✅ SGD (no momentum) achieved BETTER final UL loss")
            print("   → Removing momentum may help with hypergradient accuracy")
        else:
            print("❌ Adam achieved better final UL loss")
            print("   → Momentum appears beneficial for this problem")
        
        if sgd_hg_std < adam_hg_std:
            print("✅ SGD (no momentum) showed MORE STABLE hypergradients")
            print("   → Less momentum reduces gradient noise accumulation")
        else:
            print("❌ Adam showed more stable hypergradients")
            print("   → Momentum averaging helps with gradient stability")

if __name__ == "__main__":
    run_comparison_test()
