#!/usr/bin/env python3
"""
Script to run timing comparison with 3 separate plots:
1. UL Loss comparison
2. Hypergradient norm comparison (without DS-BLO raw gradient)
3. Average time per iteration comparison
"""

import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm2_working import F2CSAAlgorithm2Working
from dsblo_optII import DSBLOOptII
from ssigd_correct_final import CorrectSSIGD

def run_timing_comparison(dim=100, T=5, seed=1234):
    """Run timing comparison for all algorithms"""
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create problem instance
    problem = StronglyConvexBilevelProblem(dim=dim, num_constraints=2*dim, noise_std=0.1)
    
    # Fix CRN for UL loss evaluation across all methods
    crn_upper, crn_lower = problem._sample_instance_noise()
    print(f"Fixed CRN for UL loss evaluation across all methods")
    
    # Generate initial point
    x0 = torch.randn(dim, device='cpu', dtype=torch.float64)
    
    results = {}
    
    # Run F2CSA
    print("=" * 50)
    print("RUNNING F2CSA")
    print("=" * 50)
    algorithm2 = F2CSAAlgorithm2Working(problem)
    algorithm2.crn_upper = crn_upper
    
    start_time = time.time()
    f2csa_results = algorithm2.optimize(
        x0, T, D=0.03, eta=0.1, delta=0.05**3, alpha=0.05, N_g=1,  # K=1, M=1
        warm_ll=False, keep_adam_state=False,
        plot_name=None, save_warm_name=None
    )
    f2csa_time = time.time() - start_time
    
    results['F2CSA'] = {
        'ul_losses': f2csa_results['ul_losses'],
        'hypergrad_norms': f2csa_results['hypergrad_norms'],
        'total_time': f2csa_time,
        'avg_time_per_iter': f2csa_time / T
    }
    
    # Run DS-BLO
    print("=" * 50)
    print("RUNNING DS-BLO")
    print("=" * 50)
    dsblo = DSBLOOptII(problem)
    dsblo.crn_upper = crn_upper
    
    start_time = time.time()
    dsblo_results = dsblo.optimize(
        x0, T, alpha=0.05,
        sigma=0.1, grad_avg_k=1,  # K=1
        gamma1=0.9, gamma2=0.999, beta=0.1, eta_cap=1.0,
        ul_track_noisy_ll=False,
    )
    dsblo_time = time.time() - start_time
    
    results['DS-BLO'] = {
        'ul_losses': dsblo_results['ul_losses'],
        'hypergrad_norms': dsblo_results['hypergrad_norms'],  # Use momentum norms
        'total_time': dsblo_time,
        'avg_time_per_iter': dsblo_time / T
    }
    
    # Run SSIGD
    print("=" * 50)
    print("RUNNING SSIGD")
    print("=" * 50)
    ssigd_algo = CorrectSSIGD(problem)
    ssigd_algo.crn_upper = crn_upper
    
    start_time = time.time()
    x_ssigd, ul_losses_ssigd, hypergrad_norms_ssigd = ssigd_algo.solve(T=T, beta=0.1, x0=x0)
    ssigd_time = time.time() - start_time
    
    results['SSIGD'] = {
        'ul_losses': ul_losses_ssigd,
        'hypergrad_norms': hypergrad_norms_ssigd,
        'total_time': ssigd_time,
        'avg_time_per_iter': ssigd_time / T
    }
    
    return results

def create_plots(results, dim=100, T=5):
    """Create 3 separate plots"""
    
    # Plot 1: UL Loss Comparison
    plt.figure(figsize=(10, 6))
    for algo_name, data in results.items():
        plt.plot(data['ul_losses'], label=algo_name, linewidth=2, marker='o')
    
    plt.xlabel('Iteration')
    plt.ylabel('Upper-Level Loss')
    plt.title(f'Upper-Level Loss Comparison (dim={dim}, T={T})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ul_loss_comparison_dim100.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Hypergradient Norm Comparison (without DS-BLO raw gradient)
    plt.figure(figsize=(10, 6))
    for algo_name, data in results.items():
        plt.plot(data['hypergrad_norms'], label=algo_name, linewidth=2, marker='o')
    
    plt.xlabel('Iteration')
    plt.ylabel('Hypergradient Norm')
    plt.title(f'Hypergradient Norm Comparison (dim={dim}, T={T})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('linear')
    plt.tight_layout()
    plt.savefig('hypergrad_norm_comparison_dim100.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Average Time per Iteration Comparison
    plt.figure(figsize=(10, 6))
    algorithms = list(results.keys())
    avg_times = [results[algo]['avg_time_per_iter'] for algo in algorithms]
    
    bars = plt.bar(algorithms, avg_times, color=['blue', 'red', 'green'], alpha=0.7)
    plt.xlabel('Algorithm')
    plt.ylabel('Average Time per Iteration (seconds)')
    plt.title(f'Average Time per Iteration Comparison (dim={dim}, T={T})')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, avg_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time_val:.3f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('timing_comparison_dim100.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created 3 plots:")
    print("1. ul_loss_comparison_dim100.png")
    print("2. hypergrad_norm_comparison_dim100.png") 
    print("3. timing_comparison_dim100.png")

def main():
    parser = argparse.ArgumentParser(description='Timing comparison with 3 separate plots')
    parser.add_argument('--dim', type=int, default=100, help='Problem dimension')
    parser.add_argument('--T', type=int, default=5, help='Number of iterations')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    
    args = parser.parse_args()
    
    print(f"Running timing comparison: dim={args.dim}, T={args.T}, seed={args.seed}")
    
    # Run comparison
    results = run_timing_comparison(dim=args.dim, T=args.T, seed=args.seed)
    
    # Print timing summary
    print("\n" + "="*60)
    print("TIMING SUMMARY")
    print("="*60)
    for algo_name, data in results.items():
        print(f"{algo_name:>8}: {data['total_time']:.3f}s total, {data['avg_time_per_iter']:.3f}s/iter")
    
    # Create plots
    create_plots(results, dim=args.dim, T=args.T)

if __name__ == "__main__":
    main()
