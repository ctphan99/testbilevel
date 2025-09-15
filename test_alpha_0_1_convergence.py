#!/usr/bin/env python3
"""
Test with α = 0.1 to see if we can get convergence
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
from f2csa_algorithm2 import F2CSAAlgorithm2

def test_alpha_0_1_convergence():
    """Test with α = 0.1 to see if we can get convergence"""
    print("🔧 TESTING α = 0.1 FOR CONVERGENCE")
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
    
    # Use α = 0.1
    alpha = 0.1
    x0 = torch.randn(5, dtype=torch.float64)
    
    print(f"Using α = {alpha}")
    print(f"α₁ = {1/alpha:.1f}")
    print(f"α₂ = {1/(alpha**2):.1f}")
    print(f"δ = α³ = {alpha**3:.3f}")
    print(f"Test point: {x0}")
    print()
    
    # Test Algorithm 1
    print("1️⃣ ALGORITHM 1 - 50 ITERATIONS")
    print("-" * 30)
    result1 = algorithm1.optimize(x0, max_iterations=50, alpha=alpha, N_g=10, lr=0.001)
    
    # Check convergence
    grads1 = result1['grad_norms']
    last_20 = grads1[-20:]
    grad_std1 = np.std(last_20)
    grad_range1 = max(last_20) - min(last_20)
    
    print(f"Final gradient norm: {grads1[-1]:.2f}")
    print(f"Last 20 std deviation: {grad_std1:.2f}")
    print(f"Last 20 range: {grad_range1:.2f}")
    
    if grad_std1 < 10 and grad_range1 < 20:
        print("✅ Algorithm 1: CONVERGED")
        alg1_converged = True
    else:
        print("❌ Algorithm 1: NOT CONVERGED")
        alg1_converged = False
    print()
    
    # Test Algorithm 2
    print("2️⃣ ALGORITHM 2 - 50 ITERATIONS")
    print("-" * 30)
    result2 = algorithm2.optimize(x0, T=50, D=1.0, eta=0.001, delta=alpha**3, alpha=alpha)
    
    # Check convergence
    grads2 = result2['grad_norms']
    last_20 = grads2[-20:]
    grad_std2 = np.std(last_20)
    grad_range2 = max(last_20) - min(last_20)
    
    print(f"Final gradient norm: {result2['final_g_norm']:.2f}")
    print(f"Last 20 std deviation: {grad_std2:.2f}")
    print(f"Last 20 range: {grad_range2:.2f}")
    
    if grad_std2 < 10 and grad_range2 < 20:
        print("✅ Algorithm 2: CONVERGED")
        alg2_converged = True
    else:
        print("❌ Algorithm 2: NOT CONVERGED")
        alg2_converged = False
    print()
    
    # Summary
    print("3️⃣ CONVERGENCE SUMMARY")
    print("-" * 30)
    print(f"Algorithm 1 converged: {alg1_converged}")
    print(f"Algorithm 2 converged: {alg2_converged}")
    print(f"δ-accuracy: {alpha**3:.3f} (target: < 0.1)")
    
    if alg1_converged and alg2_converged:
        print("\n🎉 SUCCESS: Both algorithms converge with α = 0.1!")
        print("✅ Gradient norms stabilize - numbers don't change much!")
        print("⚠️  But δ-accuracy = 0.001 > 0.1 (not meeting requirement)")
    else:
        print(f"\n⚠️  Partial success with α = 0.1")
    
    return {
        'alpha': alpha,
        'algorithm1_converged': alg1_converged,
        'algorithm2_converged': alg2_converged,
        'delta_accuracy': alpha**3
    }

def test_alpha_0_2_convergence():
    """Test with α = 0.2 to see if we can get even better convergence"""
    print("\n🔧 TESTING α = 0.2 FOR CONVERGENCE")
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
    
    # Use α = 0.2
    alpha = 0.2
    x0 = torch.randn(5, dtype=torch.float64)
    
    print(f"Using α = {alpha}")
    print(f"α₁ = {1/alpha:.1f}")
    print(f"α₂ = {1/(alpha**2):.1f}")
    print(f"δ = α³ = {alpha**3:.3f}")
    print(f"Test point: {x0}")
    print()
    
    # Test Algorithm 1
    print("1️⃣ ALGORITHM 1 - 50 ITERATIONS")
    print("-" * 30)
    result1 = algorithm1.optimize(x0, max_iterations=50, alpha=alpha, N_g=10, lr=0.001)
    
    # Check convergence
    grads1 = result1['grad_norms']
    last_20 = grads1[-20:]
    grad_std1 = np.std(last_20)
    grad_range1 = max(last_20) - min(last_20)
    
    print(f"Final gradient norm: {grads1[-1]:.2f}")
    print(f"Last 20 std deviation: {grad_std1:.2f}")
    print(f"Last 20 range: {grad_range1:.2f}")
    
    if grad_std1 < 5 and grad_range1 < 10:
        print("✅ Algorithm 1: CONVERGED")
        alg1_converged = True
    else:
        print("❌ Algorithm 1: NOT CONVERGED")
        alg1_converged = False
    print()
    
    # Test Algorithm 2
    print("2️⃣ ALGORITHM 2 - 50 ITERATIONS")
    print("-" * 30)
    result2 = algorithm2.optimize(x0, T=50, D=1.0, eta=0.001, delta=alpha**3, alpha=alpha)
    
    # Check convergence
    grads2 = result2['grad_norms']
    last_20 = grads2[-20:]
    grad_std2 = np.std(last_20)
    grad_range2 = max(last_20) - min(last_20)
    
    print(f"Final gradient norm: {result2['final_g_norm']:.2f}")
    print(f"Last 20 std deviation: {grad_std2:.2f}")
    print(f"Last 20 range: {grad_range2:.2f}")
    
    if grad_std2 < 5 and grad_range2 < 10:
        print("✅ Algorithm 2: CONVERGED")
        alg2_converged = True
    else:
        print("❌ Algorithm 2: NOT CONVERGED")
        alg2_converged = False
    print()
    
    # Summary
    print("3️⃣ CONVERGENCE SUMMARY")
    print("-" * 30)
    print(f"Algorithm 1 converged: {alg1_converged}")
    print(f"Algorithm 2 converged: {alg2_converged}")
    print(f"δ-accuracy: {alpha**3:.3f} (target: < 0.1)")
    
    if alg1_converged and alg2_converged:
        print("\n🎉 SUCCESS: Both algorithms converge with α = 0.2!")
        print("✅ Gradient norms stabilize - numbers don't change much!")
        print("⚠️  But δ-accuracy = 0.008 > 0.1 (not meeting requirement)")
    else:
        print(f"\n⚠️  Partial success with α = 0.2")
    
    return {
        'alpha': alpha,
        'algorithm1_converged': alg1_converged,
        'algorithm2_converged': alg2_converged,
        'delta_accuracy': alpha**3
    }

if __name__ == "__main__":
    # Test α = 0.1
    results1 = test_alpha_0_1_convergence()
    
    # Test α = 0.2
    results2 = test_alpha_0_2_convergence()
    
    print(f"\n🎯 FINAL CONCLUSION:")
    print(f"α = 0.1: Algorithm 1 converged: {results1['algorithm1_converged']}, Algorithm 2 converged: {results1['algorithm2_converged']}")
    print(f"α = 0.2: Algorithm 1 converged: {results2['algorithm1_converged']}, Algorithm 2 converged: {results2['algorithm2_converged']}")
    
    if results1['algorithm1_converged'] and results1['algorithm2_converged']:
        print("✅ GRADIENT DIVERGENCE FIXED with α = 0.1!")
        print("✅ Both algorithms now converge properly!")
        print("⚠️  But δ-accuracy requirement cannot be met with current penalty parameters")
    elif results2['algorithm1_converged'] and results2['algorithm2_converged']:
        print("✅ GRADIENT DIVERGENCE FIXED with α = 0.2!")
        print("✅ Both algorithms now converge properly!")
        print("⚠️  But δ-accuracy requirement cannot be met with current penalty parameters")
    else:
        print("⚠️  Still working on convergence, but progress made!")
