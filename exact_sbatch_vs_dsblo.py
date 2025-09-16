#!/usr/bin/env python3
"""
Exact SBATCH Configuration: F2CSA vs DS-BLO Comparison
Uses the exact same parameters as the SBATCH run for fair comparison
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Optional, Tuple
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
from f2csa_algorithm2_working import F2CSAAlgorithm2Working
from dsblo_conservative import DSBLOConservative
from dsblo_optII import DSBLOOptII
import warnings

warnings.filterwarnings('ignore')

def run_exact_sbatch_vs_dsblo():
    """Run F2CSA vs DS-BLO with exact SBATCH configuration"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Exact SBATCH Configuration: F2CSA vs DS-BLO')
    parser.add_argument('--T', type=int, default=5000, help='Number of iterations')
    parser.add_argument('--D', type=float, default=0.08, help='Clipping parameter')
    parser.add_argument('--eta', type=float, default=2e-4, help='Step size')
    parser.add_argument('--Ng', type=int, default=32, help='Number of gradient samples')
    parser.add_argument('--alpha', type=float, default=0.6, help='Accuracy parameter')
    parser.add_argument('--warm-ll', action='store_true', help='Enable lower-level warm start')
    parser.add_argument('--keep-adam-state', action='store_true', help='Keep Adam optimizer state')
    parser.add_argument('--plot-name', type=str, default='exact_sbatch_vs_dsblo.png', help='Plot filename')
    parser.add_argument('--dim', type=int, default=5, help='Problem dimension')
    parser.add_argument('--constraints', type=int, default=3, help='Number of constraints')
    # removed external perturbation; rely on per-sample instance noise from problem
    parser.add_argument('--dsblo-opt', type=str, choices=['I', 'II'], default='II', help='DS-BLO option (I deterministic, II stochastic)')
    parser.add_argument('--dsblo-sigma', type=float, default=1e-2, help='Stochastic noise std for DS-BLO Option II')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EXACT SBATCH CONFIGURATION: F2CSA vs DS-BLO COMPARISON")
    print("=" * 80)
    print(f"Problem: dim={args.dim}, constraints={args.constraints}")
    print(f"F2CSA Config: T={args.T}, D={args.D}, eta={args.eta}, Ng={args.Ng}, alpha={args.alpha}")
    print(f"Warm start: {args.warm_ll}, Adam state: {args.keep_adam_state}")
    # no explicit perturbation std; stochasticity comes from instance noise
    print()
    
    # Create problem instance
    problem = StronglyConvexBilevelProblem(dim=args.dim, num_constraints=args.constraints)
    
    # Initialize starting point
    x0 = torch.randn(args.dim, dtype=torch.float64)
    print(f"Starting point: {x0}")
    print()
    
    # Compute initial upper-level loss
    y0_star, _ = problem.solve_lower_level(x0)
    initial_ul_loss = problem.upper_objective(x0, y0_star).item()
    print(f"Initial UL loss: {initial_ul_loss:.6f}")
    print()
    
    # Run F2CSA Algorithm 2 with exact SBATCH configuration
    print("=" * 50)
    print("RUNNING F2CSA ALGORITHM 2 (SBATCH CONFIG)")
    print("=" * 50)
    
    algorithm2 = F2CSAAlgorithm2Working(problem)
    delta = 0.216  # Default delta value
    
    f2csa_results = algorithm2.optimize(
        x0, args.T, args.D, args.eta, delta, args.alpha, args.Ng,
        warm_ll=args.warm_ll, keep_adam_state=args.keep_adam_state,
        plot_name=None, save_warm_name=None
    )
    
    print()
    print("F2CSA Results:")
    print(f"  Final UL loss: {f2csa_results['final_ul_loss']:.6f}")
    print(f"  Final gradient norm: {f2csa_results['final_gradient_norm']:.6f}")
    print(f"  Converged: {f2csa_results['converged']}")
    print(f"  Iterations: {f2csa_results['iterations']}")
    print()
    
    # Run DS-BLO with same problem and parameters
    print("=" * 50)
    print("RUNNING DS-BLO (SAME PROBLEM)")
    print("=" * 50)
    
    if args.dsblo_opt == 'II':
        dsblo = DSBLOOptII(problem)
        dsblo_results = dsblo.optimize(x0, args.T, args.alpha, sigma=args.dsblo_sigma)
    else:
        dsblo = DSBLOConservative(problem)
        dsblo_results = dsblo.optimize(x0, args.T, args.alpha)
    
    print()
    print("DS-BLO Results:")
    print(f"  Final UL loss: {dsblo_results['final_ul_loss']:.6f}")
    print(f"  Final gradient norm: {dsblo_results['final_gradient_norm']:.6f}")
    print(f"  Converged: {dsblo_results['converged']}")
    print(f"  Iterations: {dsblo_results['iterations']}")
    print()
    
    # Create comparison plot
    print("Creating comparison plot...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Upper-level loss comparison
    ax1.plot(f2csa_results['ul_losses'], label='F2CSA (SBATCH Config)', linewidth=2)
    ax1.plot(dsblo_results['ul_losses'], label='DS-BLO', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Upper-level Loss')
    ax1.set_title('Upper-level Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Gradient norm comparison
    ax2.plot(f2csa_results['hypergrad_norms'], label='F2CSA (SBATCH Config)', linewidth=2)
    ax2.plot(dsblo_results['hypergrad_norms'], label='DS-BLO', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Hypergradient Norm')
    ax2.set_title('Hypergradient Norm Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: F2CSA trajectory
    x_history = torch.stack(f2csa_results['x_history'])
    ax3.plot(x_history[:, 0], x_history[:, 1], 'b-', alpha=0.7, linewidth=1)
    ax3.scatter(x_history[0, 0], x_history[0, 1], color='green', s=100, label='Start', zorder=5)
    ax3.scatter(x_history[-1, 0], x_history[-1, 1], color='red', s=100, label='End', zorder=5)
    ax3.set_xlabel('x[0]')
    ax3.set_ylabel('x[1]')
    ax3.set_title('F2CSA Trajectory (First 2 Dimensions)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: DS-BLO trajectory
    x_history_dsblo = torch.stack(dsblo_results['x_history'])
    ax4.plot(x_history_dsblo[:, 0], x_history_dsblo[:, 1], 'r-', alpha=0.7, linewidth=1)
    ax4.scatter(x_history_dsblo[0, 0], x_history_dsblo[0, 1], color='green', s=100, label='Start', zorder=5)
    ax4.scatter(x_history_dsblo[-1, 0], x_history_dsblo[-1, 1], color='red', s=100, label='End', zorder=5)
    ax4.set_xlabel('x[0]')
    ax4.set_ylabel('x[1]')
    ax4.set_title('DS-BLO Trajectory (First 2 Dimensions)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.plot_name, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {args.plot_name}")
    
    # Print final comparison summary
    print()
    print("=" * 80)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<25} {'F2CSA (SBATCH)':<20} {'DS-BLO':<20} {'Winner':<15}")
    print("-" * 80)
    print(f"{'Initial UL Loss':<25} {initial_ul_loss:<20.6f} {initial_ul_loss:<20.6f} {'Same':<15}")
    print(f"{'Final UL Loss':<25} {f2csa_results['final_ul_loss']:<20.6f} {dsblo_results['final_ul_loss']:<20.6f} {'F2CSA' if f2csa_results['final_ul_loss'] < dsblo_results['final_ul_loss'] else 'DS-BLO':<15}")
    print(f"{'Final Grad Norm':<25} {f2csa_results['final_gradient_norm']:<20.6f} {dsblo_results['final_gradient_norm']:<20.6f} {'F2CSA' if f2csa_results['final_gradient_norm'] < dsblo_results['final_gradient_norm'] else 'DS-BLO':<15}")
    print(f"{'Converged':<25} {f2csa_results['converged']:<20} {dsblo_results['converged']:<20} {'F2CSA' if f2csa_results['converged'] and not dsblo_results['converged'] else 'DS-BLO' if dsblo_results['converged'] and not f2csa_results['converged'] else 'Both' if f2csa_results['converged'] and dsblo_results['converged'] else 'Neither':<15}")
    print(f"{'Iterations':<25} {f2csa_results['iterations']:<20} {dsblo_results['iterations']:<20} {'Same':<15}")
    print("=" * 80)
    
    return {
        'f2csa_results': f2csa_results,
        'dsblo_results': dsblo_results,
        'initial_ul_loss': initial_ul_loss
    }

if __name__ == "__main__":
    results = run_exact_sbatch_vs_dsblo()
