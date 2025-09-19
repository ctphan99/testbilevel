#!/usr/bin/env python3
"""
Test script to tune F2CSA parameters until both lower-level and gradient converge
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final

def test_convergence_tuning():
    """Test and tune parameters for convergence"""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create problem
    dim = 5
    problem = StronglyConvexBilevelProblem(dim=dim, noise_std=0.01, device='cpu')
    
    # Create F2CSA algorithm
    f2csa = F2CSAAlgorithm1Final(problem)
    
    # Test point
    x = torch.randn(dim, dtype=torch.float64)
    
    print(f"Testing convergence tuning with x = {x}")
    print(f"Problem dimension: {dim}")
    print("=" * 60)
    
    # Test different alpha values
    alpha_values = [0.1, 0.05, 0.02, 0.01, 0.005]
    
    for alpha in alpha_values:
        print(f"\n--- Testing α = {alpha} (δ = {alpha**3:.2e}) ---")
        
        # Test lower-level convergence
        print("1. Testing lower-level convergence...")
        ll_result = f2csa.test_lower_level_convergence(x, alpha)
        print(f"   Gap: {ll_result['gap']:.2e}")
        print(f"   Converged: {ll_result['converged']}")
        
        # Test hypergradient accuracy
        print("2. Testing hypergradient accuracy...")
        hg_result = f2csa.test_hypergradient_accuracy(x, alpha)
        print(f"   Relative error: {hg_result['relative_error']:.2e}")
        print(f"   Hypergradient norm: {hg_result['hypergradient_norm']:.2e}")
        print(f"   Finite diff norm: {hg_result['finite_diff_norm']:.2e}")
        
        # Test optimization convergence
        print("3. Testing optimization convergence...")
        x_test = x.clone()
        opt_result = f2csa.optimize(x_test, max_iterations=10, alpha=alpha, N_g=20, lr=0.001)
        
        print(f"   Final loss: {opt_result['loss_history'][-1]:.6f}")
        print(f"   Final gradient norm: {opt_result['grad_norm_history'][-1]:.6f}")
        print(f"   Converged: {opt_result['converged']}")
        print(f"   Iterations: {opt_result['iterations']}")
        
        # Check if both lower-level and gradient converged
        ll_converged = ll_result['converged']
        grad_converged = opt_result['converged']
        
        if ll_converged and grad_converged:
            print(f"   ✅ SUCCESS: Both lower-level and gradient converged!")
            print(f"   Optimal α = {alpha}")
            break
        elif ll_converged:
            print(f"   ⚠️  Partial: Lower-level converged, gradient did not")
        elif grad_converged:
            print(f"   ⚠️  Partial: Gradient converged, lower-level did not")
        else:
            print(f"   ❌ Neither converged")
    
    print("\n" + "=" * 60)
    
    # Test with the best alpha found
    best_alpha = 0.01  # Start with a reasonable value
    print(f"\n--- Final Test with α = {best_alpha} ---")
    
    # Run longer optimization
    x_final = torch.randn(dim, dtype=torch.float64)
    print(f"Initial x: {x_final}")
    
    result = f2csa.optimize(x_final, max_iterations=50, alpha=best_alpha, N_g=50, lr=0.001)
    
    print(f"\nFinal Results:")
    print(f"  Final x: {result['x_final']}")
    print(f"  Final loss: {result['loss_history'][-1]:.6f}")
    print(f"  Final gradient norm: {result['grad_norm_history'][-1]:.6f}")
    print(f"  Converged: {result['converged']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  δ-accuracy: {result['delta']:.6f}")
    
    # Plot convergence
    print(f"\nConvergence History:")
    print(f"  Loss: {[f'{l:.6f}' for l in result['loss_history'][-5:]]}")
    print(f"  Grad norm: {[f'{g:.6f}' for g in result['grad_norm_history'][-5:]]}")
    
    return result

if __name__ == "__main__":
    test_convergence_tuning()
