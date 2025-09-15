#!/usr/bin/env python3
"""
Final comprehensive validation with α = 0.001 to achieve δ-accuracy < 0.1
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
from f2csa_algorithm2 import F2CSAAlgorithm2
import warnings

warnings.filterwarnings('ignore')

def test_final_validation():
    """Final comprehensive validation with optimal parameters"""
    print("🎯 FINAL VALIDATION: α = 0.001 for δ-accuracy < 0.1")
    print("=" * 60)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5, num_constraints=3, noise_std=0.1, 
        strong_convex=True, device='cpu'
    )
    
    # Initialize algorithms
    algorithm1 = F2CSAAlgorithm1Final(problem)
    algorithm2 = F2CSAAlgorithm2(problem, algorithm1)
    
    alpha = 0.001  # Optimal value found
    delta = alpha**3
    
    print(f"Parameters: α = {alpha}, δ = {delta:.9f}")
    print(f"Penalty parameters: α₁ = {alpha**(-1):.0f}, α₂ = {alpha**(-2):.0f}")
    print()
    
    # Test 1: Lower-level convergence
    print("1️⃣ LOWER-LEVEL CONVERGENCE TEST")
    print("=" * 40)
    
    convergence_results = []
    for test_num in range(5):
        print(f"\nTest {test_num + 1}/5:")
        x = torch.randn(5, dtype=torch.float64)
        
        # Get accurate solution
        y_star, info = problem.solve_lower_level(x, 'accurate', 1000, 1e-8, alpha)
        lambda_star = info.get('lambda_star', torch.zeros(3, dtype=torch.float64))
        
        # Get penalty solution
        y_tilde = algorithm1._minimize_penalty_lagrangian(x, y_star, lambda_star, alpha, delta)
        
        # Compute gap
        gap = torch.norm(y_tilde - y_star).item()
        convergence_results.append(gap)
        
        print(f"  Gap: {gap:.6f} {'✅' if gap < 0.1 else '❌'}")
    
    avg_gap = np.mean(convergence_results)
    print(f"\nAverage gap: {avg_gap:.6f}")
    print(f"Target: < 0.1")
    print(f"Status: {'✅ PASS' if avg_gap < 0.1 else '❌ FAIL'}")
    
    # Test 2: Algorithm 1 optimization
    print(f"\n2️⃣ ALGORITHM 1 OPTIMIZATION TEST")
    print("=" * 40)
    
    x0 = torch.randn(5, dtype=torch.float64)
    print(f"Initial point: {x0}")
    
    result1 = algorithm1.optimize(x0, alpha=alpha, max_iterations=1000, N_g=10, lr=0.001)
    
    print(f"Final loss: {result1['losses'][-1]:.6f}")
    print(f"Final gradient norm: {result1['grad_norms'][-1]:.6f}")
    print(f"Iterations: {len(result1['losses'])}")
    print(f"Converged: {result1['converged']}")
    
    # Test 3: Algorithm 2 optimization
    print(f"\n3️⃣ ALGORITHM 2 OPTIMIZATION TEST")
    print("=" * 40)
    
    x0_2 = torch.randn(5, dtype=torch.float64)
    print(f"Initial point: {x0_2}")
    
    result2 = algorithm2.optimize(x0_2, alpha=alpha, max_iterations=500, N_g=10, lr=0.001)
    
    print(f"Final loss: {result2['losses'][-1]:.6f}")
    print(f"Final gradient norm: {result2['grad_norms'][-1]:.6f}")
    print(f"Iterations: {len(result2['losses'])}")
    print(f"Converged: {result2['converged']}")
    
    # Test 4: Hypergradient accuracy
    print(f"\n4️⃣ HYPERGRADIENT ACCURACY TEST")
    print("=" * 40)
    
    x_test = torch.randn(5, dtype=torch.float64)
    
    # Get hypergradient from Algorithm 1
    hypergrad = algorithm1.oracle_sample(x_test, alpha, N_g=10)
    
    # Compare with finite differences
    eps = 1e-6
    hypergrad_fd = torch.zeros_like(x_test)
    
    for i in range(len(x_test)):
        x_plus = x_test.clone()
        x_plus[i] += eps
        x_minus = x_test.clone()
        x_minus[i] -= eps
        
        y_plus, _ = problem.solve_lower_level(x_plus, 'accurate', 1000, 1e-8, alpha)
        y_minus, _ = problem.solve_lower_level(x_minus, 'accurate', 1000, 1e-8, alpha)
        
        f_plus = problem.upper_objective(x_plus, y_plus)
        f_minus = problem.upper_objective(x_minus, y_minus)
        
        hypergrad_fd[i] = (f_plus - f_minus) / (2 * eps)
    
    relative_error = torch.norm(hypergrad - hypergrad_fd) / torch.norm(hypergrad_fd)
    
    print(f"Hypergradient norm: {torch.norm(hypergrad).item():.6f}")
    print(f"Finite diff norm: {torch.norm(hypergrad_fd).item():.6f}")
    print(f"Relative error: {relative_error.item():.6f}")
    print(f"Status: {'✅ GOOD' if relative_error < 0.1 else '❌ POOR'}")
    
    # Summary
    print(f"\n🎯 FINAL SUMMARY")
    print("=" * 30)
    print(f"Lower-level gap: {avg_gap:.6f} {'✅' if avg_gap < 0.1 else '❌'}")
    print(f"Algorithm 1 converged: {'✅' if result1['converged'] else '❌'}")
    print(f"Algorithm 2 converged: {'✅' if result2['converged'] else '❌'}")
    print(f"Hypergradient error: {relative_error.item():.6f} {'✅' if relative_error < 0.1 else '❌'}")
    
    all_passed = (avg_gap < 0.1 and result1['converged'] and result2['converged'] and relative_error < 0.1)
    print(f"\nOverall status: {'🎉 SUCCESS' if all_passed else '❌ NEEDS WORK'}")
    
    return all_passed

if __name__ == "__main__":
    success = test_final_validation()
    print(f"\nFinal result: {'✅ ALL TESTS PASSED' if success else '❌ SOME TESTS FAILED'}")
