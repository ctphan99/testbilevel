#!/usr/bin/env python3
"""
Test different alpha values to find the optimal one for gap < 0.1
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final

def test_alpha_values():
    """Test different alpha values to find optimal gap"""
    print("🔍 ALPHA OPTIMIZATION TEST")
    print("=" * 60)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5, 
        num_constraints=3, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    # Create algorithm
    algorithm = F2CSAAlgorithm1Final(problem)
    
    # Test point
    x = torch.randn(5, dtype=torch.float64)
    print(f"Test point x: {x}")
    print()
    
    # Test different alpha values
    alpha_values = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    
    results = []
    
    for alpha in alpha_values:
        print(f"Testing α = {alpha}")
        print("-" * 30)
        
        # Calculate parameters
        alpha1 = 1.0 / alpha
        alpha2 = 1.0 / (alpha**2)
        delta = alpha**3
        
        print(f"  α₁ = {alpha1:.1f}")
        print(f"  α₂ = {alpha2:.1f}")
        print(f"  δ = {delta:.2e}")
        
        # Get solutions
        y_star, info = problem.solve_lower_level(x, 'accurate', 1000, 1e-6, alpha)
        lambda_star = info.get('lambda_star', torch.zeros(problem.num_constraints, dtype=torch.float64))
        
        # Get penalty minimizer
        y_penalty = algorithm._minimize_penalty_lagrangian(x, y_star, lambda_star, alpha, delta)
        
        # Calculate gap
        gap = torch.norm(y_penalty - y_star).item()
        
        # Check constraint satisfaction
        h_star = problem.constraints(x, y_star)
        h_penalty = problem.constraints(x, y_penalty)
        constraints_satisfied_star = torch.all(h_star <= 1e-6)
        constraints_satisfied_penalty = torch.all(h_penalty <= 1e-6)
        
        # Get hypergradient
        hypergradient = algorithm.oracle_sample(x, alpha, 10)
        hypergradient_norm = torch.norm(hypergradient).item()
        
        print(f"  Gap: {gap:.3f}")
        print(f"  y* constraints satisfied: {constraints_satisfied_star}")
        print(f"  y~ constraints satisfied: {constraints_satisfied_penalty}")
        print(f"  Hypergradient norm: {hypergradient_norm:.1f}")
        print()
        
        results.append({
            'alpha': alpha,
            'alpha1': alpha1,
            'alpha2': alpha2,
            'delta': delta,
            'gap': gap,
            'constraints_satisfied_star': constraints_satisfied_star,
            'constraints_satisfied_penalty': constraints_satisfied_penalty,
            'hypergradient_norm': hypergradient_norm
        })
    
    # Find optimal alpha
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'α':<8} {'Gap':<8} {'y* OK':<6} {'y~ OK':<6} {'||∇F||':<10}")
    print("-" * 60)
    
    best_alpha = None
    best_gap = float('inf')
    
    for r in results:
        print(f"{r['alpha']:<8.3f} {r['gap']:<8.3f} {r['constraints_satisfied_star']!s:<6} {r['constraints_satisfied_penalty']!s:<6} {r['hypergradient_norm']:<10.1f}")
        
        if r['gap'] < 0.1 and r['constraints_satisfied_penalty']:
            if r['gap'] < best_gap:
                best_gap = r['gap']
                best_alpha = r['alpha']
    
    print()
    if best_alpha is not None:
        print(f"✅ OPTIMAL α = {best_alpha} (gap = {best_gap:.3f})")
    else:
        print("❌ No alpha found with gap < 0.1 and constraints satisfied")
        # Find the best available
        best_available = min(results, key=lambda x: x['gap'])
        print(f"Best available: α = {best_available['alpha']} (gap = {best_available['gap']:.3f})")
    
    return results

def test_optimal_alpha():
    """Test the optimal alpha value"""
    print("\n🎯 TESTING OPTIMAL ALPHA")
    print("=" * 60)
    
    # Based on results, let's test α = 0.01
    alpha = 0.01
    
    problem = StronglyConvexBilevelProblem(
        dim=5, 
        num_constraints=3, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    algorithm = F2CSAAlgorithm1Final(problem)
    
    # Test multiple points
    gaps = []
    hypergradient_norms = []
    
    for i in range(3):
        print(f"\n--- Test Point {i+1} ---")
        x = torch.randn(5, dtype=torch.float64)
        
        # Get solutions
        y_star, info = problem.solve_lower_level(x, 'accurate', 1000, 1e-6, alpha)
        lambda_star = info.get('lambda_star', torch.zeros(problem.num_constraints, dtype=torch.float64))
        delta = alpha**3
        y_penalty = algorithm._minimize_penalty_lagrangian(x, y_star, lambda_star, alpha, delta)
        
        # Calculate gap
        gap = torch.norm(y_penalty - y_star).item()
        gaps.append(gap)
        
        # Check constraints
        h_penalty = problem.constraints(x, y_penalty)
        constraints_ok = torch.all(h_penalty <= 1e-6)
        
        # Get hypergradient
        hypergradient = algorithm.oracle_sample(x, alpha, 10)
        hypergradient_norm = torch.norm(hypergradient).item()
        hypergradient_norms.append(hypergradient_norm)
        
        print(f"Gap: {gap:.3f}")
        print(f"Constraints satisfied: {constraints_ok}")
        print(f"Hypergradient norm: {hypergradient_norm:.1f}")
    
    print(f"\n📊 SUMMARY:")
    print(f"Average gap: {np.mean(gaps):.3f}")
    print(f"Max gap: {np.max(gaps):.3f}")
    print(f"Average hypergradient norm: {np.mean(hypergradient_norms):.1f}")
    print(f"Max hypergradient norm: {np.max(hypergradient_norms):.1f}")
    
    if np.mean(gaps) < 0.1:
        print("✅ SUCCESS: Gap < 0.1 achieved!")
    else:
        print("❌ FAILED: Gap still too large")

if __name__ == "__main__":
    # Test different alpha values
    results = test_alpha_values()
    
    # Test optimal alpha
    test_optimal_alpha()
