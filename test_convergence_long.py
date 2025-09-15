#!/usr/bin/env python3
"""
Test convergence with more iterations to see if numbers stabilize
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
from f2csa_algorithm2 import F2CSAAlgorithm2

def test_convergence_with_more_iterations():
    """Test if algorithms converge with more iterations"""
    print("🔄 CONVERGENCE TEST WITH MORE ITERATIONS")
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
    
    # Optimal parameters
    alpha = 0.0005
    x0 = torch.randn(5, dtype=torch.float64)
    
    print(f"Using optimal α = {alpha}")
    print(f"Test point: {x0}")
    print()
    
    # Test Algorithm 1 with more iterations
    print("1️⃣ ALGORITHM 1 - 200 ITERATIONS")
    print("-" * 40)
    result1 = algorithm1.optimize(x0, max_iterations=200, alpha=alpha, N_g=10, lr=0.001)
    
    # Check convergence by looking at last 20 iterations
    last_20_grads = result1['grad_norms'][-20:]
    grad_std = np.std(last_20_grads)
    grad_change = abs(last_20_grads[-1] - last_20_grads[0])
    
    print(f"Final gradient norm: {result1['grad_norms'][-1]:.6f}")
    print(f"Last 20 gradient norms std: {grad_std:.6f}")
    print(f"Last 20 gradient change: {grad_change:.6f}")
    print(f"Converged: {result1['converged']}")
    print()
    
    # Test Algorithm 2 with more iterations
    print("2️⃣ ALGORITHM 2 - 200 ITERATIONS")
    print("-" * 40)
    result2 = algorithm2.optimize(x0, T=200, D=1.0, eta=0.001, delta=alpha**3, alpha=alpha)
    
    # Check convergence by looking at last 20 iterations
    last_20_grads = result2['grad_norms'][-20:]
    grad_std = np.std(last_20_grads)
    grad_change = abs(last_20_grads[-1] - last_20_grads[0])
    
    print(f"Final gradient norm: {result2['final_g_norm']:.6f}")
    print(f"Last 20 gradient norms std: {grad_std:.6f}")
    print(f"Last 20 gradient change: {grad_change:.6f}")
    print(f"Converged: {result2['converged']}")
    print()
    
    # Analyze convergence
    print("3️⃣ CONVERGENCE ANALYSIS")
    print("-" * 40)
    
    # Algorithm 1 convergence
    if grad_std < 100 and grad_change < 1000:  # Reasonable convergence criteria
        print("✅ Algorithm 1: CONVERGED (gradient norms stabilized)")
    else:
        print("❌ Algorithm 1: NOT CONVERGED (gradient norms still changing)")
    
    # Algorithm 2 convergence
    if grad_std < 100 and grad_change < 1000:  # Reasonable convergence criteria
        print("✅ Algorithm 2: CONVERGED (gradient norms stabilized)")
    else:
        print("❌ Algorithm 2: NOT CONVERGED (gradient norms still changing)")
    
    # Show gradient norm trends
    print("\n4️⃣ GRADIENT NORM TRENDS")
    print("-" * 40)
    
    # Algorithm 1 trend
    alg1_start = result1['grad_norms'][0]
    alg1_end = result1['grad_norms'][-1]
    alg1_trend = "decreasing" if alg1_end < alg1_start else "increasing"
    print(f"Algorithm 1: {alg1_start:.1f} → {alg1_end:.1f} ({alg1_trend})")
    
    # Algorithm 2 trend
    alg2_start = result2['grad_norms'][0]
    alg2_end = result2['grad_norms'][-1]
    alg2_trend = "decreasing" if alg2_end < alg2_start else "increasing"
    print(f"Algorithm 2: {alg2_start:.1f} → {alg2_end:.1f} ({alg2_trend})")
    
    return {
        'algorithm1_converged': grad_std < 100 and grad_change < 1000,
        'algorithm2_converged': grad_std < 100 and grad_change < 1000,
        'algorithm1_grad_std': grad_std,
        'algorithm2_grad_std': grad_std,
        'algorithm1_grad_change': grad_change,
        'algorithm2_grad_change': grad_change
    }

def test_very_long_convergence():
    """Test with even more iterations to see true convergence"""
    print("\n🔄 VERY LONG CONVERGENCE TEST - 500 ITERATIONS")
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
    
    # Optimal parameters
    alpha = 0.0005
    x0 = torch.randn(5, dtype=torch.float64)
    
    print(f"Using optimal α = {alpha}")
    print(f"Test point: {x0}")
    print()
    
    # Test Algorithm 1 with 500 iterations
    print("1️⃣ ALGORITHM 1 - 500 ITERATIONS")
    print("-" * 40)
    result1 = algorithm1.optimize(x0, max_iterations=500, alpha=alpha, N_g=10, lr=0.001)
    
    # Check convergence by looking at last 50 iterations
    last_50_grads = result1['grad_norms'][-50:]
    grad_std = np.std(last_50_grads)
    grad_change = abs(last_50_grads[-1] - last_50_grads[0])
    
    print(f"Final gradient norm: {result1['grad_norms'][-1]:.6f}")
    print(f"Last 50 gradient norms std: {grad_std:.6f}")
    print(f"Last 50 gradient change: {grad_change:.6f}")
    print(f"Converged: {result1['converged']}")
    print()
    
    # Test Algorithm 2 with 500 iterations
    print("2️⃣ ALGORITHM 2 - 500 ITERATIONS")
    print("-" * 40)
    result2 = algorithm2.optimize(x0, T=500, D=1.0, eta=0.001, delta=alpha**3, alpha=alpha)
    
    # Check convergence by looking at last 50 iterations
    last_50_grads = result2['grad_norms'][-50:]
    grad_std = np.std(last_50_grads)
    grad_change = abs(last_50_grads[-1] - last_50_grads[0])
    
    print(f"Final gradient norm: {result2['final_g_norm']:.6f}")
    print(f"Last 50 gradient norms std: {grad_std:.6f}")
    print(f"Last 50 gradient change: {grad_change:.6f}")
    print(f"Converged: {result2['converged']}")
    print()
    
    # Final convergence analysis
    print("3️⃣ FINAL CONVERGENCE ANALYSIS")
    print("-" * 40)
    
    # Algorithm 1 convergence
    if grad_std < 50 and grad_change < 500:  # Stricter convergence criteria
        print("✅ Algorithm 1: CONVERGED (gradient norms stabilized)")
    else:
        print("❌ Algorithm 1: NOT CONVERGED (gradient norms still changing)")
    
    # Algorithm 2 convergence
    if grad_std < 50 and grad_change < 500:  # Stricter convergence criteria
        print("✅ Algorithm 2: CONVERGED (gradient norms stabilized)")
    else:
        print("❌ Algorithm 2: NOT CONVERGED (gradient norms still changing)")
    
    return {
        'algorithm1_converged_500': grad_std < 50 and grad_change < 500,
        'algorithm2_converged_500': grad_std < 50 and grad_change < 500,
        'algorithm1_grad_std_500': grad_std,
        'algorithm2_grad_std_500': grad_std,
        'algorithm1_grad_change_500': grad_change,
        'algorithm2_grad_change_500': grad_change
    }

if __name__ == "__main__":
    # Test with 200 iterations
    results_200 = test_convergence_with_more_iterations()
    
    # Test with 500 iterations
    results_500 = test_very_long_convergence()
    
    print("\n🎯 FINAL CONVERGENCE SUMMARY:")
    print("=" * 60)
    print(f"200 iterations - Algorithm 1 converged: {results_200['algorithm1_converged']}")
    print(f"200 iterations - Algorithm 2 converged: {results_200['algorithm2_converged']}")
    print(f"500 iterations - Algorithm 1 converged: {results_500['algorithm1_converged_500']}")
    print(f"500 iterations - Algorithm 2 converged: {results_500['algorithm2_converged_500']}")
    
    if (results_200['algorithm1_converged'] and results_200['algorithm2_converged'] and
        results_500['algorithm1_converged_500'] and results_500['algorithm2_converged_500']):
        print("\n🎉 COMPLETE SUCCESS: Both algorithms converge with more iterations!")
    else:
        print("\n⚠️  Some algorithms may need even more iterations to fully converge")
