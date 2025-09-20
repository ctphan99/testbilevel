#!/usr/bin/env python3
"""
Compare PGD vs Clipping projection methods for SSIGD stability
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from problem import StronglyConvexBilevelProblem
from ssigd_correct_final import CorrectSSIGD

def test_projection_methods():
    """Compare PGD vs Clipping projection methods"""
    
    print("🔬 SSIGD Projection Method Comparison")
    print("=" * 60)
    
    # Set same seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Test parameters
    dim = 50
    T = 1000
    beta = 0.001
    mu_F = 0.1
    
    print(f"Test Parameters:")
    print(f"  Dimension: {dim}")
    print(f"  Iterations: {T}")
    print(f"  Beta (step size): {beta}")
    print(f"  mu_F: {mu_F}")
    print(f"  Seed: {seed}")
    print()
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=dim, device='cpu')
    x0 = torch.randn(dim, dtype=torch.float64) * 0.1
    
    print(f"Problem Info:")
    print(f"  Upper level strong convexity: λ_min={torch.linalg.eigvals(problem.Q_upper).real.min():.3f}")
    print(f"  Lower level strong convexity: λ_min={torch.linalg.eigvals(problem.Q_lower).real.min():.3f}")
    print()
    
    # Initial loss
    y0, _, _ = problem.solve_lower_level(x0, solver='gurobi')
    initial_loss = problem.upper_objective(x0, y0).item()
    print(f"Initial UL Loss: {initial_loss:.6f}")
    print()
    
    results = {}
    
    # Test PGD projection
    print("Testing PGD Projection...")
    torch.manual_seed(seed)  # Reset seed
    np.random.seed(seed)
    problem_pgd = StronglyConvexBilevelProblem(dim=dim, device='cpu')
    x0_pgd = torch.randn(dim, dtype=torch.float64) * 0.1
    
    try:
        ssigd_pgd = CorrectSSIGD(problem_pgd)
        result_pgd = ssigd_pgd.solve(T=T, beta=beta, x0=x0_pgd, diminishing=True, mu_F=mu_F, projection_method='pgd')
        results['PGD'] = result_pgd
        print(f"✓ PGD Results: Final loss = {result_pgd['final_loss']:.6f}, Final grad = {result_pgd['final_grad_norm']:.6f}")
    except Exception as e:
        print(f"✗ PGD failed: {e}")
        results['PGD'] = {'final_loss': float('inf'), 'final_grad_norm': float('inf'), 'losses': [], 'grad_norms': []}
    
    # Test Clipping projection
    print("\nTesting Clipping Projection...")
    torch.manual_seed(seed)  # Reset seed
    np.random.seed(seed)
    problem_clip = StronglyConvexBilevelProblem(dim=dim, device='cpu')
    x0_clip = torch.randn(dim, dtype=torch.float64) * 0.1
    
    try:
        ssigd_clip = CorrectSSIGD(problem_clip)
        result_clip = ssigd_clip.solve(T=T, beta=beta, x0=x0_clip, diminishing=True, mu_F=mu_F, projection_method='clip')
        results['Clipping'] = result_clip
        print(f"✓ Clipping Results: Final loss = {result_clip['final_loss']:.6f}, Final grad = {result_clip['final_grad_norm']:.6f}")
    except Exception as e:
        print(f"✗ Clipping failed: {e}")
        results['Clipping'] = {'final_loss': float('inf'), 'final_grad_norm': float('inf'), 'losses': [], 'grad_norms': []}
    
    # Create comparison plots
    print("\nCreating comparison plots...")
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
    methods = list(results.keys())
    final_losses = [results[method]['final_loss'] for method in methods]
    final_grads = [results[method]['final_grad_norm'] for method in methods]
    
    # Filter out inf values for plotting
    final_losses = [l if np.isfinite(l) else 1e6 for l in final_losses]
    final_grads = [g if np.isfinite(g) else 1e6 for g in final_grads]
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    plt.bar(x_pos - width/2, final_losses, width, label='Final Loss', alpha=0.8)
    plt.bar(x_pos + width/2, final_grads, width, label='Final Grad Norm', alpha=0.8)
    
    plt.xlabel('Projection Method')
    plt.ylabel('Value')
    plt.title('Final Performance Comparison')
    plt.xticks(x_pos, methods)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 4: Stability analysis (loss variance)
    plt.subplot(2, 2, 4)
    stability_scores = []
    for method in methods:
        if 'losses' in results[method] and results[method]['losses']:
            losses = results[method]['losses']
            valid_losses = [l for l in losses if np.isfinite(l)]
            if len(valid_losses) > 100:  # Only consider last 100 iterations for stability
                recent_losses = valid_losses[-100:]
                stability_score = np.std(recent_losses) / np.mean(recent_losses) if np.mean(recent_losses) != 0 else float('inf')
                stability_scores.append(stability_score)
            else:
                stability_scores.append(float('inf'))
        else:
            stability_scores.append(float('inf'))
    
    plt.bar(methods, stability_scores, alpha=0.8, color=['green' if s < 0.1 else 'orange' if s < 0.5 else 'red' for s in stability_scores])
    plt.ylabel('Stability Score (CV of last 100 losses)')
    plt.title('Stability Comparison (Lower = More Stable)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('projection_method_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved comparison plot to projection_method_comparison.png")
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"{'Method':<15} {'Final Loss':<15} {'Final Grad':<15} {'Stability':<15}")
    print("-" * 60)
    
    for i, method in enumerate(methods):
        stability = stability_scores[i] if i < len(stability_scores) else float('inf')
        print(f"{method:<15} {results[method]['final_loss']:<15.6f} {results[method]['final_grad_norm']:<15.6f} {stability:<15.6f}")
    
    # Determine winner
    best_method = None
    best_score = float('inf')
    for i, method in enumerate(methods):
        if np.isfinite(results[method]['final_loss']) and np.isfinite(stability_scores[i]):
            # Combined score: lower loss + higher stability (lower stability score)
            combined_score = results[method]['final_loss'] + stability_scores[i] * 1000
            if combined_score < best_score:
                best_score = combined_score
                best_method = method
    
    if best_method:
        print(f"\n🏆 Most Stable Method: {best_method}")
        print(f"   Final Loss: {results[best_method]['final_loss']:.6f}")
        print(f"   Stability Score: {stability_scores[methods.index(best_method)]:.6f}")
    
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    results = test_projection_methods()
