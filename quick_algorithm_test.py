#!/usr/bin/env python3
"""
Quick test to compare Algorithm 1 vs Algorithm 2 performance
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
from f2csa_algorithm2_working import F2CSAAlgorithm2Working

def quick_test():
    """Quick comparison test"""
    
    print("🔬 Quick Algorithm Comparison")
    print("=" * 50)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.1)
    x0 = torch.randn(5, dtype=torch.float64)
    
    print(f"Test point: {x0}")
    print()
    
    # Test with good parameters
    alpha = 0.05
    
    print("Testing Algorithm 1...")
    algo1 = F2CSAAlgorithm1Final(problem)
    result1 = algo1.optimize(x0, max_iterations=20, alpha=alpha, N_g=10, lr=0.001)
    
    print("\nTesting Algorithm 2...")
    algo2 = F2CSAAlgorithm2Working(problem)
    result2 = algo2.optimize(
        x0, T=20, D=0.05, eta=0.001, 
        delta=alpha**3, alpha=alpha, N_g=10
    )
    
    print("\n📊 Results Comparison:")
    print("-" * 30)
    print(f"Algorithm 1:")
    print(f"  Final loss: {result1['loss_history'][-1]:.6f}")
    print(f"  Final grad norm: {result1['grad_norm_history'][-1]:.6f}")
    print(f"  Converged: {result1['converged']}")
    
    print(f"\nAlgorithm 2:")
    print(f"  Final UL loss: {result2['final_ul_loss']:.6f}")
    print(f"  Final grad norm: {result2['final_gradient_norm']:.6f}")
    print(f"  Converged: {result2['converged']}")
    
    # Determine winner
    if result1['loss_history'][-1] < result2['final_ul_loss']:
        winner = "Algorithm 1"
        improvement = result2['final_ul_loss'] - result1['loss_history'][-1]
    else:
        winner = "Algorithm 2"
        improvement = result1['loss_history'][-1] - result2['final_ul_loss']
    
    print(f"\n🏆 Winner: {winner}")
    print(f"   Improvement: {improvement:.6f}")
    
    return result1, result2

if __name__ == "__main__":
    result1, result2 = quick_test()
