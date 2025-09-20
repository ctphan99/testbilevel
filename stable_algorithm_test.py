#!/usr/bin/env python3
"""
Stable Algorithm Test - Simple parameter tuning for stability
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
from dsblo_optII import DSBLOOptII
from ssigd_correct_final import CorrectSSIGD
import warnings

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='Stable Algorithm Test')
    parser.add_argument('--dim', type=int, default=100, help='Problem dimension')
    parser.add_argument('--T', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--eta', type=float, default=1e-4, help='F2CSA step size')
    parser.add_argument('--ssigd-beta', type=float, default=0.001, help='SSIGD step size')
    parser.add_argument('--ssigd-mu', type=float, default=10.0, help='SSIGD mu_F parameter')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("STABLE ALGORITHM TEST")
    print("=" * 80)
    print(f"Dimension: {args.dim}, Iterations: {args.T}")
    print(f"F2CSA eta: {args.eta}")
    print(f"SSIGD beta: {args.ssigd_beta}, mu_F: {args.ssigd_mu}")
    print("=" * 80)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=args.dim, device='cpu')
    x0 = torch.randn(args.dim, dtype=torch.float64) * 0.1
    
    # Initial loss
    y0, _, _ = problem.solve_lower_level(x0, solver='cvxpy')
    initial_loss = problem.upper_objective(x0, y0).item()
    print(f"Initial UL Loss: {initial_loss:.6f}")
    
    results = {}
    
    # Run F2CSA with very conservative parameters
    print("\n" + "=" * 60)
    print("RUNNING F2CSA (CONSERVATIVE)")
    print("=" * 60)
    try:
        f2csa = F2CSAAlgorithm1Final(problem, device='cpu', dtype=torch.float64)
        f2csa_result = f2csa.solve(T=args.T, eta=args.eta, alpha=0.6, x0=x0)
        results['F2CSA'] = f2csa_result
        print(f"F2CSA Results: Final loss = {f2csa_result['final_loss']:.6f}, "
              f"Final grad = {f2csa_result['final_grad_norm']:.6f}")
    except Exception as e:
        print(f"F2CSA failed: {e}")
        results['F2CSA'] = {'final_loss': float('inf'), 'final_grad_norm': float('inf'), 
                           'converged': False, 'iterations': 0}
    
    # Run DS-BLO with conservative parameters
    print("\n" + "=" * 60)
    print("RUNNING DS-BLO (CONSERVATIVE)")
    print("=" * 60)
    try:
        dsblo = DSBLOOptII(problem, device='cpu', dtype=torch.float64)
        dsblo_result = dsblo.optimize(x0=x0, T=args.T, alpha=0.6, sigma=0.0, 
                                     gamma1=0.1, gamma2=10.0, beta=0.6,
                                     grad_clip=10.0, eta_cap=0.01)
        results['DS-BLO'] = dsblo_result
        print(f"DS-BLO Results: Final loss = {dsblo_result['final_loss']:.6f}, "
              f"Final grad = {dsblo_result['final_grad_norm']:.6f}")
    except Exception as e:
        print(f"DS-BLO failed: {e}")
        results['DS-BLO'] = {'final_loss': float('inf'), 'final_grad_norm': float('inf'), 
                            'converged': False, 'iterations': 0}
    
    # Run SSIGD with very conservative parameters
    print("\n" + "=" * 60)
    print("RUNNING SSIGD (CONSERVATIVE)")
    print("=" * 60)
    try:
        ssigd = CorrectSSIGD(problem)
        ssigd_result = ssigd.solve(T=args.T, beta=args.ssigd_beta, x0=x0, 
                                  diminishing=True, mu_F=args.ssigd_mu)
        results['SSIGD'] = ssigd_result
        print(f"SSIGD Results: Final loss = {ssigd_result['final_loss']:.6f}, "
              f"Final grad = {ssigd_result['final_grad_norm']:.6f}")
    except Exception as e:
        print(f"SSIGD failed: {e}")
        results['SSIGD'] = {'final_loss': float('inf'), 'final_grad_norm': float('inf'), 
                           'converged': False, 'iterations': 0}
    
    # Create comparison plot
    print("\n" + "=" * 60)
    print("CREATING COMPARISON PLOT")
    print("=" * 60)
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Loss curves
    plt.subplot(2, 2, 1)
    for name, result in results.items():
        if 'losses' in result and result['losses']:
            losses = result['losses']
            # Filter out inf/nan values
            valid_losses = [l for l in losses if np.isfinite(l)]
            if valid_losses:
                plt.plot(valid_losses, label=f"{name} (final: {result['final_loss']:.2f})", linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Upper-Level Loss')
    plt.title('Upper-Level Loss Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Gradient norms
    plt.subplot(2, 2, 2)
    for name, result in results.items():
        if 'grad_norms' in result and result['grad_norms']:
            grad_norms = result['grad_norms']
            # Filter out inf/nan values
            valid_grads = [g for g in grad_norms if np.isfinite(g)]
            if valid_grads:
                plt.plot(valid_grads, label=f"{name} (final: {result['final_grad_norm']:.2f})", linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 3: Final comparison
    plt.subplot(2, 2, 3)
    algorithms = list(results.keys())
    final_losses = [results[alg]['final_loss'] for alg in algorithms]
    final_grads = [results[alg]['final_grad_norm'] for alg in algorithms]
    
    # Filter out inf values for plotting
    final_losses = [l if np.isfinite(l) else 1e6 for l in final_losses]
    final_grads = [g if np.isfinite(g) else 1e6 for g in final_grads]
    
    x_pos = np.arange(len(algorithms))
    width = 0.35
    
    plt.bar(x_pos - width/2, final_losses, width, label='Final Loss', alpha=0.8)
    plt.bar(x_pos + width/2, final_grads, width, label='Final Grad Norm', alpha=0.8)
    
    plt.xlabel('Algorithm')
    plt.ylabel('Value')
    plt.title('Final Performance Comparison')
    plt.xticks(x_pos, algorithms)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 4: Convergence summary
    plt.subplot(2, 2, 4)
    converged = [1 if results[alg]['converged'] else 0 for alg in algorithms]
    iterations = [results[alg]['iterations'] for alg in algorithms]
    
    plt.bar(algorithms, converged, alpha=0.8, color=['green' if c else 'red' for c in converged])
    plt.ylabel('Converged (1=Yes, 0=No)')
    plt.title('Convergence Status')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stable_algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved comparison plot to stable_algorithm_comparison.png")
    
    # Print final results
    print("\n" + "=" * 80)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 80)
    
    print(f"{'Algorithm':<15} {'Final Loss':<15} {'Final Grad':<15} {'Converged':<10} {'Iterations':<10}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<15} {result['final_loss']:<15.6f} {result['final_grad_norm']:<15.6f} "
              f"{result['converged']:<10} {result['iterations']:<10}")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
