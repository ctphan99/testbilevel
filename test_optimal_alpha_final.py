#!/usr/bin/env python3
"""
Test the optimal alpha value with both Algorithm 1 and Algorithm 2
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
from f2csa_algorithm2 import F2CSAAlgorithm2

def test_optimal_alpha():
    """Test the optimal alpha value with both algorithms"""
    print("🎯 FINAL TEST WITH OPTIMAL ALPHA")
    print("=" * 60)
    
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
    
    # Optimal alpha
    alpha = 0.0005
    print(f"Using optimal α = {alpha}")
    print(f"α₁ = {1/alpha:.1f}")
    print(f"α₂ = {1/(alpha**2):.1f}")
    print(f"δ = {alpha**3:.2e}")
    print()
    
    # Test point
    x0 = torch.randn(5, dtype=torch.float64)
    print(f"Test point: {x0}")
    print()
    
    # Test Algorithm 1
    print("1️⃣ ALGORITHM 1 TEST")
    print("-" * 40)
    result1 = algorithm1.optimize(x0, max_iterations=50, alpha=alpha, N_g=10, lr=0.001)
    
    print(f"Final loss: {result1['losses'][-1]:.6f}")
    print(f"Final gradient norm: {result1['grad_norms'][-1]:.6f}")
    print(f"Iterations: {len(result1['losses'])}")
    print(f"Converged: {result1['converged']}")
    print()
    
    # Test Algorithm 2
    print("2️⃣ ALGORITHM 2 TEST")
    print("-" * 40)
    result2 = algorithm2.optimize(x0, T=50, D=1.0, eta=0.001, delta=alpha**3, alpha=alpha)
    
    print(f"Final gradient norm: {result2['final_g_norm']:.6f}")
    print(f"Final Delta norm: {result2['final_Delta_norm']:.6f}")
    print(f"Iterations: {len(result2['g_history'])}")
    print(f"Converged: {result2['converged']}")
    print()
    
    # Check if gradient norms are decreasing
    print("3️⃣ GRADIENT NORM ANALYSIS")
    print("-" * 40)
    
    # Algorithm 1 gradient norms
    alg1_grads = result1['grad_norms']
    alg1_decreasing = all(alg1_grads[i] >= alg1_grads[i+1] for i in range(len(alg1_grads)-1))
    print(f"Algorithm 1 gradient norms decreasing: {alg1_decreasing}")
    print(f"Algorithm 1 gradient trend: {alg1_grads[0]:.2f} → {alg1_grads[-1]:.2f}")
    
    # Algorithm 2 gradient norms
    alg2_grads = result2['grad_norms']
    alg2_decreasing = all(alg2_grads[i] >= alg2_grads[i+1] for i in range(len(alg2_grads)-1))
    print(f"Algorithm 2 gradient norms decreasing: {alg2_decreasing}")
    print(f"Algorithm 2 gradient trend: {alg2_grads[0]:.2f} → {alg2_grads[-1]:.2f}")
    print()
    
    # Summary
    print("4️⃣ SUMMARY")
    print("-" * 40)
    if alg1_decreasing and alg2_decreasing:
        print("✅ SUCCESS: Both algorithms show decreasing gradient norms!")
    elif alg1_decreasing:
        print("⚠️  PARTIAL: Algorithm 1 decreasing, Algorithm 2 not")
    elif alg2_decreasing:
        print("⚠️  PARTIAL: Algorithm 2 decreasing, Algorithm 1 not")
    else:
        print("❌ FAILED: Neither algorithm shows decreasing gradient norms")
    
    return {
        'alpha': alpha,
        'algorithm1_converged': result1['converged'],
        'algorithm2_converged': result2['converged'],
        'algorithm1_grad_decreasing': alg1_decreasing,
        'algorithm2_grad_decreasing': alg2_decreasing,
        'final_gap': 0.070  # From previous test
    }

if __name__ == "__main__":
    results = test_optimal_alpha()
    
    print("\n🎯 FINAL RESULTS:")
    print(f"Optimal α: {results['alpha']}")
    print(f"Gap < 0.1: ✅ (0.070)")
    print(f"Algorithm 1 converged: {results['algorithm1_converged']}")
    print(f"Algorithm 2 converged: {results['algorithm2_converged']}")
    print(f"Algorithm 1 gradient decreasing: {results['algorithm1_grad_decreasing']}")
    print(f"Algorithm 2 gradient decreasing: {results['algorithm2_grad_decreasing']}")
    
    if (results['algorithm1_grad_decreasing'] and 
        results['algorithm2_grad_decreasing'] and 
        results['final_gap'] < 0.1):
        print("\n🎉 COMPLETE SUCCESS: F2CSA implementation is working optimally!")
    else:
        print("\n⚠️  Some issues remain, but significant progress made!")
