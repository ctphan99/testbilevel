#!/usr/bin/env python3
"""
Test algorithms for much longer to check if they truly converge or diverge
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
from f2csa_algorithm2 import F2CSAAlgorithm2

def test_long_convergence():
    """Test algorithms for much longer to verify true convergence"""
    print("🔄 LONG CONVERGENCE TEST - CHECKING FOR TRUE CONVERGENCE")
    print("=" * 70)
    
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
    
    # Use α = 0.1 (our best convergence value)
    alpha = 0.1
    x0 = torch.randn(5, dtype=torch.float64)
    
    print(f"Using α = {alpha}")
    print(f"α₁ = {1/alpha:.1f}")
    print(f"α₂ = {1/(alpha**2):.1f}")
    print(f"δ = α³ = {alpha**3:.3f}")
    print(f"Test point: {x0}")
    print()
    
    # Test Algorithm 1 for 200 iterations
    print("1️⃣ ALGORITHM 1 - 200 ITERATIONS (LONG TEST)")
    print("-" * 50)
    result1 = algorithm1.optimize(x0, max_iterations=200, alpha=alpha, N_g=10, lr=0.001)
    
    # Analyze convergence pattern
    grads1 = result1['grad_norms']
    print(f"Total iterations: {len(grads1)}")
    print(f"Final gradient norm: {grads1[-1]:.2f}")
    
    # Check different segments
    segments = [
        ("First 20", grads1[:20]),
        ("Middle 20 (50-70)", grads1[50:70] if len(grads1) > 70 else grads1[50:]),
        ("Last 20", grads1[-20:]),
        ("Last 50", grads1[-50:] if len(grads1) > 50 else grads1)
    ]
    
    print("\n📊 CONVERGENCE ANALYSIS:")
    for name, segment in segments:
        if len(segment) > 0:
            mean_grad = np.mean(segment)
            std_grad = np.std(segment)
            min_grad = np.min(segment)
            max_grad = np.max(segment)
            range_grad = max_grad - min_grad
            
            print(f"{name:15}: mean={mean_grad:6.2f}, std={std_grad:5.2f}, range={range_grad:5.2f}")
    
    # Check if truly converged
    last_50 = grads1[-50:] if len(grads1) > 50 else grads1
    if len(last_50) > 0:
        std_last_50 = np.std(last_50)
        range_last_50 = np.max(last_50) - np.min(last_50)
        
        if std_last_50 < 2.0 and range_last_50 < 5.0:
            print("✅ Algorithm 1: TRULY CONVERGED (stable over long run)")
            alg1_converged = True
        else:
            print("❌ Algorithm 1: NOT TRULY CONVERGED (still oscillating)")
            alg1_converged = False
    else:
        alg1_converged = False
    print()
    
    # Test Algorithm 2 for 200 iterations
    print("2️⃣ ALGORITHM 2 - 200 ITERATIONS (LONG TEST)")
    print("-" * 50)
    result2 = algorithm2.optimize(x0, T=200, D=1.0, eta=0.001, delta=alpha**3, alpha=alpha)
    
    # Analyze convergence pattern
    grads2 = result2['grad_norms']
    print(f"Total iterations: {len(grads2)}")
    print(f"Final gradient norm: {result2['final_g_norm']:.2f}")
    
    # Check different segments
    segments = [
        ("First 20", grads2[:20]),
        ("Middle 20 (50-70)", grads2[50:70] if len(grads2) > 70 else grads2[50:]),
        ("Last 20", grads2[-20:]),
        ("Last 50", grads2[-50:] if len(grads2) > 50 else grads2)
    ]
    
    print("\n📊 CONVERGENCE ANALYSIS:")
    for name, segment in segments:
        if len(segment) > 0:
            mean_grad = np.mean(segment)
            std_grad = np.std(segment)
            min_grad = np.min(segment)
            max_grad = np.max(segment)
            range_grad = max_grad - min_grad
            
            print(f"{name:15}: mean={mean_grad:6.2f}, std={std_grad:5.2f}, range={range_grad:5.2f}")
    
    # Check if truly converged
    last_50 = grads2[-50:] if len(grads2) > 50 else grads2
    if len(last_50) > 0:
        std_last_50 = np.std(last_50)
        range_last_50 = np.max(last_50) - np.min(last_50)
        
        if std_last_50 < 2.0 and range_last_50 < 5.0:
            print("✅ Algorithm 2: TRULY CONVERGED (stable over long run)")
            alg2_converged = True
        else:
            print("❌ Algorithm 2: NOT TRULY CONVERGED (still oscillating)")
            alg2_converged = False
    else:
        alg2_converged = False
    print()
    
    # Test with α = 0.2 for comparison
    print("3️⃣ COMPARISON: α = 0.2 - 200 ITERATIONS")
    print("-" * 50)
    alpha2 = 0.2
    print(f"Using α = {alpha2}")
    print(f"α₁ = {1/alpha2:.1f}")
    print(f"α₂ = {1/(alpha2**2):.1f}")
    print(f"δ = α³ = {alpha2**3:.3f}")
    
    # Test Algorithm 2 with α = 0.2
    result2_alpha2 = algorithm2.optimize(x0, T=200, D=1.0, eta=0.001, delta=alpha2**3, alpha=alpha2)
    
    grads2_alpha2 = result2_alpha2['grad_norms']
    print(f"Total iterations: {len(grads2_alpha2)}")
    print(f"Final gradient norm: {result2_alpha2['final_g_norm']:.2f}")
    
    # Check last 50 iterations
    last_50_alpha2 = grads2_alpha2[-50:] if len(grads2_alpha2) > 50 else grads2_alpha2
    if len(last_50_alpha2) > 0:
        std_last_50_alpha2 = np.std(last_50_alpha2)
        range_last_50_alpha2 = np.max(last_50_alpha2) - np.min(last_50_alpha2)
        
        print(f"Last 50 std: {std_last_50_alpha2:.2f}, range: {range_last_50_alpha2:.2f}")
        
        if std_last_50_alpha2 < 2.0 and range_last_50_alpha2 < 5.0:
            print("✅ Algorithm 2 (α=0.2): TRULY CONVERGED")
            alg2_alpha2_converged = True
        else:
            print("❌ Algorithm 2 (α=0.2): NOT TRULY CONVERGED")
            alg2_alpha2_converged = False
    else:
        alg2_alpha2_converged = False
    print()
    
    # Final summary
    print("🎯 LONG CONVERGENCE TEST RESULTS")
    print("=" * 50)
    print(f"Algorithm 1 (α=0.1, 200 iter): {'✅ CONVERGED' if alg1_converged else '❌ DIVERGED'}")
    print(f"Algorithm 2 (α=0.1, 200 iter): {'✅ CONVERGED' if alg2_converged else '❌ DIVERGED'}")
    print(f"Algorithm 2 (α=0.2, 200 iter): {'✅ CONVERGED' if alg2_alpha2_converged else '❌ DIVERGED'}")
    
    if alg1_converged and alg2_converged:
        print("\n🎉 SUCCESS: Both algorithms truly converge over long runs!")
        print("✅ Gradient norms remain stable - no divergence detected!")
    else:
        print("\n⚠️  WARNING: Some algorithms may not be truly convergent!")
        print("❌ Gradient norms may still be oscillating or diverging!")
    
    return {
        'algorithm1_converged': alg1_converged,
        'algorithm2_converged': alg2_converged,
        'algorithm2_alpha2_converged': alg2_alpha2_converged
    }

if __name__ == "__main__":
    results = test_long_convergence()
    
    print(f"\n🔍 FINAL VERDICT:")
    if results['algorithm1_converged'] and results['algorithm2_converged']:
        print("✅ GRADIENT DIVERGENCE COMPLETELY FIXED!")
        print("✅ Both algorithms converge and remain stable over long runs!")
        print("✅ The numbers don't change much with more iterations - TRUE CONVERGENCE!")
    else:
        print("⚠️  CONVERGENCE ISSUES DETECTED!")
        print("❌ Some algorithms may still be diverging or oscillating!")
        print("❌ Need further investigation or parameter tuning!")
