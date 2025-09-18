#!/usr/bin/env python3
"""
Comprehensive comparison of SSIGD step size strategies
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from problem import StronglyConvexBilevelProblem
from ssigd_stable_debug import StableSSIGD
import time

def run_strategy_comparison():
    """Compare different SSIGD step size strategies"""
    
    print("=" * 80)
    print("COMPREHENSIVE SSIGD STEP SIZE STRATEGY COMPARISON")
    print("=" * 80)
    
    # Create problem instance
    problem = StronglyConvexBilevelProblem(
        dim=10,
        num_constraints=3,
        noise_std=0.01,
        strong_convex=True,
        device='cpu'
    )
    
    # Initialize starting point
    torch.manual_seed(1234)
    x0 = torch.randn(10, dtype=torch.float64)
    
    print(f"Problem: dim=10, constraints=3")
    print(f"Starting point: {x0}")
    
    # Compute initial upper-level loss
    y0_star, _ = problem.solve_lower_level(x0)
    initial_ul_loss = problem.upper_objective(x0, y0_star).item()
    print(f"Initial UL loss: {initial_ul_loss:.6f}")
    print()
    
    # Define strategies to test
    strategies = [
        {
            'name': 'constant_sqrt',
            'description': 'β = O(1/√T) ≈ 0.014',
            'params': {'step_strategy': 'constant_sqrt', 'mu_F': 0.1}
        },
        {
            'name': 'diminishing_mu01',
            'description': 'β_r = 1/(μ_F(r+1)), μ_F=0.1',
            'params': {'step_strategy': 'diminishing', 'mu_F': 0.1}
        },
        {
            'name': 'diminishing_mu001',
            'description': 'β_r = 1/(μ_F(r+1)), μ_F=0.01',
            'params': {'step_strategy': 'diminishing', 'mu_F': 0.01}
        },
        {
            'name': 'diminishing_mu1',
            'description': 'β_r = 1/(μ_F(r+1)), μ_F=1.0',
            'params': {'step_strategy': 'diminishing', 'mu_F': 1.0}
        },
        {
            'name': 'adaptive',
            'description': 'Adaptive step size',
            'params': {'step_strategy': 'adaptive', 'mu_F': 0.1}
        }
    ]
    
    results = {}
    T = 5000
    
    for strategy in strategies:
        print(f"{'='*60}")
        print(f"Testing {strategy['name']}: {strategy['description']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        ssigd = StableSSIGD(problem)
        x_final, losses, grad_norms, x_history, step_sizes = ssigd.solve(
            T=T,
            x0=x0.clone(),
            step_strategy=strategy['params']['step_strategy'],
            mu_F=strategy['params']['mu_F'],
            eps_fd=1e-4,
            clip_threshold=2.0
        )
        
        end_time = time.time()
        
        # Store results
        results[strategy['name']] = {
            'description': strategy['description'],
            'final_loss': losses[-1],
            'final_grad_norm': grad_norms[-1],
            'min_loss': min(losses),
            'converged': grad_norms[-1] < 1e-2,
            'step_sizes': step_sizes,
            'losses': losses,
            'grad_norms': grad_norms,
            'x_history': x_history,
            'runtime': end_time - start_time,
            'initial_step': step_sizes[0],
            'final_step': step_sizes[-1]
        }
        
        print(f"\nResults for {strategy['name']}:")
        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Final gradient norm: {grad_norms[-1]:.3f}")
        print(f"  Min loss achieved: {min(losses):.6f}")
        print(f"  Converged: {grad_norms[-1] < 1e-2}")
        print(f"  Initial step size: {step_sizes[0]:.6f}")
        print(f"  Final step size: {step_sizes[-1]:.6f}")
        print(f"  Runtime: {end_time - start_time:.2f}s")
        print()
    
    # Create comprehensive comparison plot
    print("Creating comprehensive comparison plot...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot 1: Upper-level loss comparison
    for i, (name, result) in enumerate(results.items()):
        ax1.plot(result['losses'], label=f"{name}: {result['description']}", 
                color=colors[i], linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Upper-level Loss')
    ax1.set_title('Upper-level Loss Comparison')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Gradient norm comparison
    for i, (name, result) in enumerate(results.items()):
        ax2.plot(result['grad_norms'], label=f"{name}", 
                color=colors[i], linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Hypergradient Norm')
    ax2.set_title('Hypergradient Norm Comparison')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Step size evolution
    for i, (name, result) in enumerate(results.items()):
        ax3.plot(result['step_sizes'], label=f"{name}", 
                color=colors[i], linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Step Size (β)')
    ax3.set_title('Step Size Evolution')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Trajectory comparison (first 2 dimensions)
    for i, (name, result) in enumerate(results.items()):
        x_history = torch.stack(result['x_history'])
        ax4.plot(x_history[:, 0], x_history[:, 1], 
                color=colors[i], alpha=0.7, linewidth=1, label=f"{name}")
    ax4.scatter(x0[0], x0[1], color='black', s=100, label='Start', zorder=5)
    ax4.set_xlabel('x[0]')
    ax4.set_ylabel('x[1]')
    ax4.set_title('Trajectory Comparison (First 2 Dimensions)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ssigd_strategies_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved comparison plot to ssigd_strategies_comparison.png")
    
    # Print comprehensive comparison table
    print(f"\n{'='*100}")
    print("COMPREHENSIVE STRATEGY COMPARISON")
    print(f"{'='*100}")
    print(f"{'Strategy':<20} {'Final Loss':<12} {'Grad Norm':<10} {'Converged':<10} {'Min Loss':<12} {'Runtime':<8} {'Init Step':<10} {'Final Step':<10}")
    print("-" * 100)
    
    for name, result in results.items():
        print(f"{name:<20} {result['final_loss']:<12.6f} {result['final_grad_norm']:<10.3f} "
              f"{result['converged']:<10} {result['min_loss']:<12.6f} {result['runtime']:<8.2f} "
              f"{result['initial_step']:<10.6f} {result['final_step']:<10.6f}")
    
    # Find best strategy
    best_strategy = min(results.items(), key=lambda x: x[1]['final_grad_norm'])
    print(f"\n{'='*100}")
    print(f"BEST STRATEGY: {best_strategy[0]}")
    print(f"Description: {best_strategy[1]['description']}")
    print(f"Final gradient norm: {best_strategy[1]['final_grad_norm']:.6f}")
    print(f"Converged: {best_strategy[1]['converged']}")
    print(f"{'='*100}")
    
    return results

if __name__ == "__main__":
    results = run_strategy_comparison()
