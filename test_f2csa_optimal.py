#!/usr/bin/env python3
"""
Test F2CSA Algorithm 1 with optimal parameters (α = 0.5)
"""

import torch
import numpy as np
from f2csa_algorithm_accurate import F2CSAAlgorithm1
from problem import StronglyConvexBilevelProblem

def test_optimal_f2csa():
    """Test F2CSA Algorithm 1 with optimal parameters"""
    print("Testing F2CSA Algorithm 1 with Optimal Parameters")
    print("=" * 60)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.01, strong_convex=True)
    
    # Initialize algorithm
    algorithm = F2CSAAlgorithm1(problem)
    
    # Test point
    x0 = torch.randn(5, device=problem.device, dtype=problem.dtype) * 0.1
    
    # Optimal parameters based on alpha sweep
    alpha = 0.5  # Optimal from sweep
    N_g = 10     # Increased batch size
    lr = 1e-3    # Learning rate
    max_iterations = 20
    
    print(f"Test point: x0 = {x0}")
    print(f"Parameters: α = {alpha}, N_g = {N_g}, lr = {lr}")
    print(f"Max iterations: {max_iterations}")
    print()
    
    # Test single oracle call first
    print("--- Testing Single Oracle Call ---")
    hypergradient = algorithm.oracle_sample(x0, alpha, N_g)
    grad_norm = torch.norm(hypergradient).item()
    print(f"Single oracle hypergradient norm: {grad_norm:.6f}")
    print()
    
    # Test full optimization
    print("--- Testing Full Optimization ---")
    result = algorithm.optimize(x0, max_iterations=max_iterations, alpha=alpha, N_g=N_g, lr=lr)
    
    print(f"\nFinal Results:")
    print(f"  Final loss: {result['losses'][-1]:.6f}")
    print(f"  Final gradient norm: {result['grad_norms'][-1]:.6f}")
    print(f"  Total loss reduction: {result['losses'][0] - result['losses'][-1]:.6f}")
    print(f"  Converged: {result['converged']}")
    print(f"  Iterations: {result['iterations']}")
    
    # Analyze convergence
    print(f"\nConvergence Analysis:")
    print(f"  Initial loss: {result['losses'][0]:.6f}")
    print(f"  Final loss: {result['losses'][-1]:.6f}")
    print(f"  Loss reduction: {result['losses'][0] - result['losses'][-1]:.6f}")
    print(f"  Relative improvement: {((result['losses'][0] - result['losses'][-1]) / abs(result['losses'][0])) * 100:.2f}%")
    
    print(f"\nGradient Norm Analysis:")
    print(f"  Initial gradient norm: {result['grad_norms'][0]:.6f}")
    print(f"  Final gradient norm: {result['grad_norms'][-1]:.6f}")
    print(f"  Gradient reduction: {result['grad_norms'][0] - result['grad_norms'][-1]:.6f}")
    
    # Check if we achieved δ-accuracy < 0.1
    print(f"\nδ-Accuracy Check:")
    print(f"  δ = α³ = {alpha**3:.6f}")
    if alpha**3 < 0.1:
        print(f"  ✅ δ-accuracy < 0.1 achieved: {alpha**3:.6f} < 0.1")
    else:
        print(f"  ❌ δ-accuracy >= 0.1: {alpha**3:.6f} >= 0.1")
    
    return result

if __name__ == "__main__":
    test_optimal_f2csa()
