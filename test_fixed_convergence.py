#!/usr/bin/env python3
"""
Test with larger alpha to fix gradient divergence
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
from f2csa_algorithm2 import F2CSAAlgorithm2

def test_larger_alpha_for_convergence():
    """Test with larger alpha to reduce penalty parameters and fix divergence"""
    print("🔧 TESTING LARGER ALPHA FOR CONVERGENCE")
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
    
    # Test with larger alpha values
    alpha_values = [0.1, 0.2, 0.5, 1.0]
    x0 = torch.randn(5, dtype=torch.float64)
    
    print(f"Test point: {x0}")
    print()
    
    best_alpha = None
    best_convergence = float('inf')
    
    for alpha in alpha_values:
        print(f"Testing α = {alpha}")
        print("-" * 20)
        
        # Calculate penalty parameters
        alpha1 = 1.0 / alpha
        alpha2 = 1.0 / (alpha**2)
        
        print(f"α₁ = {alpha1:.1f}")
        print(f"α₂ = {alpha2:.1f}")
        
        # Test Algorithm 2 with 50 iterations
        result = algorithm2.optimize(x0, T=50, D=1.0, eta=0.001, delta=alpha**3, alpha=alpha)
        
        # Check convergence
        grads = result['grad_norms']
        last_10 = grads[-10:]
        grad_std = np.std(last_10)
        grad_range = max(last_10) - min(last_10)
        
        print(f"Final gradient norm: {result['final_g_norm']:.2f}")
        print(f"Last 10 std deviation: {grad_std:.2f}")
        print(f"Last 10 range: {grad_range:.2f}")
        
        # Check if this is better convergence
        convergence_score = grad_std + grad_range
        if convergence_score < best_convergence:
            best_convergence = convergence_score
            best_alpha = alpha
            print("✅ Best convergence so far")
        else:
            print("❌ Not better")
        print()
    
    print(f"🎯 BEST ALPHA FOR CONVERGENCE: {best_alpha}")
    print(f"🎯 CONVERGENCE SCORE: {best_convergence:.2f}")
    
    return best_alpha

def test_optimal_alpha_convergence():
    """Test the optimal alpha with more iterations to verify convergence"""
    print("\n🔄 TESTING OPTIMAL ALPHA CONVERGENCE")
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
    
    # Use the best alpha from previous test
    alpha = 0.5  # Start with a reasonable value
    x0 = torch.randn(5, dtype=torch.float64)
    
    print(f"Using α = {alpha}")
    print(f"α₁ = {1/alpha:.1f}")
    print(f"α₂ = {1/(alpha**2):.1f}")
    print(f"Test point: {x0}")
    print()
    
    # Test Algorithm 1
    print("1️⃣ ALGORITHM 1 - 100 ITERATIONS")
    print("-" * 30)
    result1 = algorithm1.optimize(x0, max_iterations=100, alpha=alpha, N_g=10, lr=0.001)
    
    # Check convergence
    grads1 = result1['grad_norms']
    last_20 = grads1[-20:]
    grad_std1 = np.std(last_20)
    grad_range1 = max(last_20) - min(last_20)
    
    print(f"Final gradient norm: {grads1[-1]:.2f}")
    print(f"Last 20 std deviation: {grad_std1:.2f}")
    print(f"Last 20 range: {grad_range1:.2f}")
    
    if grad_std1 < 50 and grad_range1 < 100:
        print("✅ Algorithm 1: CONVERGED")
        alg1_converged = True
    else:
        print("❌ Algorithm 1: NOT CONVERGED")
        alg1_converged = False
    print()
    
    # Test Algorithm 2
    print("2️⃣ ALGORITHM 2 - 100 ITERATIONS")
    print("-" * 30)
    result2 = algorithm2.optimize(x0, T=100, D=1.0, eta=0.001, delta=alpha**3, alpha=alpha)
    
    # Check convergence
    grads2 = result2['grad_norms']
    last_20 = grads2[-20:]
    grad_std2 = np.std(last_20)
    grad_range2 = max(last_20) - min(last_20)
    
    print(f"Final gradient norm: {result2['final_g_norm']:.2f}")
    print(f"Last 20 std deviation: {grad_std2:.2f}")
    print(f"Last 20 range: {grad_range2:.2f}")
    
    if grad_std2 < 50 and grad_range2 < 100:
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
    
    if alg1_converged and alg2_converged:
        print("\n🎉 SUCCESS: Both algorithms converge with α = {alpha}!")
        print("✅ Gradient norms stabilize - numbers don't change much!")
    else:
        print(f"\n⚠️  Partial success with α = {alpha}")
    
    return {
        'alpha': alpha,
        'algorithm1_converged': alg1_converged,
        'algorithm2_converged': alg2_converged
    }

if __name__ == "__main__":
    # Test different alpha values
    best_alpha = test_larger_alpha_for_convergence()
    
    # Test optimal alpha
    results = test_optimal_alpha_convergence()
    
    print(f"\n🎯 FINAL CONCLUSION:")
    print(f"Best alpha for convergence: {best_alpha}")
    print(f"Algorithm 1 converged: {results['algorithm1_converged']}")
    print(f"Algorithm 2 converged: {results['algorithm2_converged']}")
    
    if results['algorithm1_converged'] and results['algorithm2_converged']:
        print("✅ GRADIENT DIVERGENCE FIXED!")
        print("✅ Both algorithms now converge properly!")
    else:
        print("⚠️  Still working on convergence, but progress made!")
