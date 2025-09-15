#!/usr/bin/env python3
"""
Simple convergence test - check if gradient norms stabilize
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
from f2csa_algorithm2 import F2CSAAlgorithm2

def test_convergence_simple():
    """Simple test to check if algorithms converge"""
    print("🔄 SIMPLE CONVERGENCE TEST")
    print("=" * 50)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5, 
        num_constraints=3, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    # Create algorithms
    algorithm1 = F2CSAAlgorithm1Final(problem)
    algorithm2 = F2CSAAlgorithm2(problem)
    
    # Optimal parameters
    alpha = 0.0005
    x0 = torch.randn(5, dtype=torch.float64)
    
    print(f"Using optimal α = {alpha}")
    print(f"Test point: {x0}")
    print()
    
    # Test Algorithm 1 with 100 iterations
    print("1️⃣ ALGORITHM 1 - 100 ITERATIONS")
    print("-" * 30)
    result1 = algorithm1.optimize(x0, max_iterations=100, alpha=alpha, N_g=10, lr=0.001)
    
    # Check if gradient norms stabilize
    grads = result1['grad_norms']
    last_10 = grads[-10:]
    first_10 = grads[:10]
    
    print(f"First 10 gradient norms: {[f'{g:.1f}' for g in first_10]}")
    print(f"Last 10 gradient norms:  {[f'{g:.1f}' for g in last_10]}")
    
    # Check convergence
    grad_std = np.std(last_10)
    grad_range = max(last_10) - min(last_10)
    
    print(f"Last 10 std deviation: {grad_std:.2f}")
    print(f"Last 10 range: {grad_range:.2f}")
    
    if grad_std < 100 and grad_range < 200:
        print("✅ Algorithm 1: CONVERGED (gradient norms stabilized)")
        alg1_converged = True
    else:
        print("❌ Algorithm 1: NOT CONVERGED (gradient norms still changing)")
        alg1_converged = False
    print()
    
    # Test Algorithm 2 with 100 iterations
    print("2️⃣ ALGORITHM 2 - 100 ITERATIONS")
    print("-" * 30)
    result2 = algorithm2.optimize(x0, T=100, D=1.0, eta=0.001, delta=alpha**3, alpha=alpha)
    
    # Check if gradient norms stabilize
    grads = result2['grad_norms']
    last_10 = grads[-10:]
    first_10 = grads[:10]
    
    print(f"First 10 gradient norms: {[f'{g:.1f}' for g in first_10]}")
    print(f"Last 10 gradient norms:  {[f'{g:.1f}' for g in last_10]}")
    
    # Check convergence
    grad_std = np.std(last_10)
    grad_range = max(last_10) - min(last_10)
    
    print(f"Last 10 std deviation: {grad_std:.2f}")
    print(f"Last 10 range: {grad_range:.2f}")
    
    if grad_std < 100 and grad_range < 200:
        print("✅ Algorithm 2: CONVERGED (gradient norms stabilized)")
        alg2_converged = True
    else:
        print("❌ Algorithm 2: NOT CONVERGED (gradient norms still changing)")
        alg2_converged = False
    print()
    
    # Summary
    print("3️⃣ CONVERGENCE SUMMARY")
    print("-" * 30)
    print(f"Algorithm 1 converged: {alg1_converged}")
    print(f"Algorithm 2 converged: {alg2_converged}")
    
    if alg1_converged and alg2_converged:
        print("\n🎉 SUCCESS: Both algorithms converge!")
    elif alg1_converged or alg2_converged:
        print("\n⚠️  PARTIAL: One algorithm converges")
    else:
        print("\n❌ FAILED: Neither algorithm converges")
    
    return {
        'algorithm1_converged': alg1_converged,
        'algorithm2_converged': alg2_converged
    }

if __name__ == "__main__":
    results = test_convergence_simple()
    
    print(f"\n🎯 FINAL RESULT:")
    print(f"Algorithm 1 converged: {results['algorithm1_converged']}")
    print(f"Algorithm 2 converged: {results['algorithm2_converged']}")
    
    if results['algorithm1_converged'] and results['algorithm2_converged']:
        print("✅ Both algorithms show convergence - numbers don't change much!")
    else:
        print("⚠️  Some algorithms may need more iterations to fully converge")
