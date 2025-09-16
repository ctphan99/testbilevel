#!/usr/bin/env python3
"""
Exact SBATCH Configuration Comparison: DS-BLO vs F2CSA
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
from dsblo_corrected_implementation import DSBLOCorrected
import warnings

warnings.filterwarnings('ignore')

def run_exact_sbatch_comparison():
    """Run DS-BLO vs F2CSA with exact SBATCH configuration"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Exact SBATCH Configuration: DS-BLO vs F2CSA')
    parser.add_argument('--T', type=int, default=5000, help='Number of iterations (SBATCH default)')
    parser.add_argument('--D', type=float, default=0.05, help='F2CSA clipping parameter (SBATCH default)')
    parser.add_argument('--eta', type=float, default=1e-4, help='F2CSA step size (SBATCH default)')
    parser.add_argument('--Ng', type=int, default=64, help='F2CSA gradient samples (SBATCH default)')
    parser.add_argument('--alpha', type=float, default=0.6, help='Alpha parameter (SBATCH default)')
    parser.add_argument('--warm-ll', action='store_true', help='Enable lower-level warm-start (SBATCH default)')
    parser.add_argument('--keep-adam-state', action='store_true', help='Keep Adam optimizer state (SBATCH default)')
    parser.add_argument('--dim', type=int, default=3, help='Problem dimension (matching warm start)')
    parser.add_argument('--constraints', type=int, default=2, help='Number of constraints (matching warm start)')
    parser.add_argument('--plot-name', type=str, default='exact_sbatch_comparison.png', help='Plot filename')
    
    args = parser.parse_args()
    
    print("🔬 EXACT SBATCH CONFIGURATION COMPARISON")
    print("=" * 80)
    print("Using EXACT same parameters as SBATCH run:")
    print(f"  T = {args.T} iterations")
    print(f"  D = {args.D} (clipping parameter)")
    print(f"  eta = {args.eta} (step size)")
    print(f"  Ng = {args.Ng} (gradient samples)")
    print(f"  alpha = {args.alpha} (accuracy parameter)")
    print(f"  warm-ll = {args.warm_ll} (lower-level warm-start)")
    print(f"  keep-adam-state = {args.keep_adam_state} (Adam state persistence)")
    print(f"  dim = {args.dim}, constraints = {args.constraints}")
    print()
    
    # Set seed BEFORE creating problem to ensure same parameters
    torch.manual_seed(42)  # Ensure reproducibility
    np.random.seed(42)     # Also set numpy seed for any numpy operations
    
    # Initialize the SAME problem for both algorithms
    problem = StronglyConvexBilevelProblem(
        dim=args.dim, 
        num_constraints=args.constraints, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    # Use warm start if available (same as F2CSA script)
    warm_path = 'algo2_warmstart.npy'
    try:
        x0 = torch.tensor(np.load(warm_path), dtype=torch.float64)
        print(f"✅ Loaded warm start from {warm_path}: x0 shape = {tuple(x0.shape)}")
    except Exception:
        x0 = torch.randn(args.dim, dtype=torch.float64)
        print("⚠️  No warm start found; using random x0.")
    
    print(f"Initial point x0: {x0}")
    print(f"Initial UL loss: {problem.upper_objective(x0, problem.solve_lower_level(x0)[0]).item():.6f}")
    print()
    
    # Initialize algorithms
    dsblo = DSBLOCorrected(problem)
    f2csa = F2CSAAlgorithm2Working(problem)
    
    # F2CSA parameters (exact SBATCH configuration)
    delta = args.alpha**3
    
    print("🚀 Running DS-BLO Corrected...")
    print("-" * 50)
    dsblo_results = dsblo.optimize(x0, args.T, args.alpha)
    
    print("\n🚀 Running F2CSA with EXACT SBATCH configuration...")
    print("-" * 50)
    f2csa_results = f2csa.optimize(
        x0, args.T, args.D, args.eta, delta, args.alpha, args.Ng,
        warm_ll=args.warm_ll, keep_adam_state=args.keep_adam_state,
        plot_name=None, save_warm_name=None  # Don't overwrite existing files
    )
    
    # Create comparison plots
    print("\n📊 Creating comparison plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Upper-level loss comparison
    ax1.plot(dsblo_results['ul_losses'], label='DS-BLO Corrected', color='blue', linewidth=2)
    ax1.plot(f2csa_results['ul_losses'], label='F2CSA (SBATCH Config)', color='red', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Upper-level Loss f(x, y*)')
    ax1.set_title('Upper-level Loss Comparison (Exact SBATCH Config)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Hypergradient norm comparison
    ax2.plot(dsblo_results['hypergrad_norms'], label='DS-BLO Corrected', color='blue', linewidth=2)
    ax2.plot(f2csa_results['hypergrad_norms'], label='F2CSA (SBATCH Config)', color='red', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Hypergradient Norm ||∇F||')
    ax2.set_title('Hypergradient Norm Comparison (Exact SBATCH Config)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Final convergence comparison
    final_losses = [dsblo_results['final_ul_loss'], f2csa_results['final_ul_loss']]
    final_grads = [dsblo_results['final_gradient_norm'], f2csa_results['final_gradient_norm']]
    algorithms = ['DS-BLO Corrected', 'F2CSA (SBATCH)']
    
    ax3.bar(algorithms, final_losses, color=['blue', 'red'], alpha=0.7)
    ax3.set_ylabel('Final Upper-level Loss')
    ax3.set_title('Final Loss Comparison')
    ax3.grid(True, alpha=0.3)
    
    ax4.bar(algorithms, final_grads, color=['blue', 'red'], alpha=0.7)
    ax4.set_ylabel('Final Hypergradient Norm')
    ax4.set_title('Final Gradient Norm Comparison')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(args.plot_name, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Print detailed summary
    print("\n📋 EXACT SBATCH CONFIGURATION COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<30} {'DS-BLO Corrected':<20} {'F2CSA (SBATCH)':<20} {'Winner':<15}")
    print("-" * 80)
    print(f"{'Initial UL Loss':<30} {dsblo_results['ul_losses'][0]:<20.6f} {f2csa_results['ul_losses'][0]:<20.6f} {'Same':<15}")
    print(f"{'Final UL Loss':<30} {dsblo_results['final_ul_loss']:<20.6f} {f2csa_results['final_ul_loss']:<20.6f} {'F2CSA' if f2csa_results['final_ul_loss'] < dsblo_results['final_ul_loss'] else 'DS-BLO':<15}")
    print(f"{'Final Grad Norm':<30} {dsblo_results['final_gradient_norm']:<20.6f} {f2csa_results['final_gradient_norm']:<20.6f} {'F2CSA' if f2csa_results['final_gradient_norm'] < dsblo_results['final_gradient_norm'] else 'DS-BLO':<15}")
    print(f"{'Converged':<30} {dsblo_results['converged']:<20} {f2csa_results['converged']:<20} {'Both' if dsblo_results['converged'] and f2csa_results['converged'] else 'One':<15}")
    print(f"{'Iterations':<30} {dsblo_results['iterations']:<20} {f2csa_results['iterations']:<20} {'Same':<15}")
    
    # Additional F2CSA-specific metrics
    print(f"\n🔍 F2CSA SBATCH Configuration Details:")
    print(f"  Warm-start enabled: {args.warm_ll}")
    print(f"  Adam state persistence: {args.keep_adam_state}")
    print(f"  Clipping parameter D: {args.D}")
    print(f"  Step size eta: {args.eta}")
    print(f"  Gradient samples Ng: {args.Ng}")
    print(f"  Accuracy parameter alpha: {args.alpha}")
    print(f"  Delta (alpha^3): {delta:.6f}")
    
    print(f"\n📊 Plot saved to: {args.plot_name}")
    print("✅ Exact SBATCH configuration comparison completed!")
    
    # Check convergence quality
    if f2csa_results['converged'] and dsblo_results['converged']:
        print("🎉 Both algorithms converged successfully!")
    elif f2csa_results['converged']:
        print("🎯 F2CSA converged, DS-BLO did not")
    elif dsblo_results['converged']:
        print("🎯 DS-BLO converged, F2CSA did not")
    else:
        print("⚠️  Neither algorithm converged within tolerance")

if __name__ == "__main__":
    run_exact_sbatch_comparison()
