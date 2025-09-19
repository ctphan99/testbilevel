#!/usr/bin/env python3
"""
Comprehensive test to compare Algorithm 1 vs Algorithm 2 performance
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
from f2csa_algorithm2_working import F2CSAAlgorithm2Working

def comprehensive_test():
    """Comprehensive comparison test"""
    
    print("🔬 Comprehensive Algorithm Comparison")
    print("=" * 60)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.1)
    
    # Test multiple random starting points
    num_tests = 5
    results = []
    
    for i in range(num_tests):
        print(f"\n--- Test {i+1}/{num_tests} ---")
        
        # Random starting point
        x0 = torch.randn(5, dtype=torch.float64)
        print(f"Starting point: {x0}")
        
        # Test parameters
        alpha = 0.05
        
        # Test Algorithm 1
        print("Testing Algorithm 1...")
        algo1 = F2CSAAlgorithm1Final(problem)
        result1 = algo1.optimize(x0, max_iterations=20, alpha=alpha, lr=0.001, N_g=20)
        
        # Test Algorithm 2
        print("Testing Algorithm 2...")
        algo2 = F2CSAAlgorithm2Working(problem)
        result2 = algo2.optimize(x0, T=20, D=0.1, eta=0.001, delta=0.001, alpha=alpha, N_g=20)
        
        # Compare results
        final_loss1 = result1['losses'][-1] if result1['losses'] else float('inf')
        final_loss2 = result2['ul_losses'][-1] if result2['ul_losses'] else float('inf')
        
        print(f"Algorithm 1 final loss: {final_loss1:.6f}")
        print(f"Algorithm 2 final loss: {final_loss2:.6f}")
        
        if final_loss2 < final_loss1:
            print("✅ Algorithm 2 wins!")
            winner = "Algorithm 2"
        elif final_loss1 < final_loss2:
            print("✅ Algorithm 1 wins!")
            winner = "Algorithm 1"
        else:
            print("🤝 Tie!")
            winner = "Tie"
        
        results.append({
            'test': i+1,
            'algo1_loss': final_loss1,
            'algo2_loss': final_loss2,
            'winner': winner
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    algo1_wins = sum(1 for r in results if r['winner'] == 'Algorithm 1')
    algo2_wins = sum(1 for r in results if r['winner'] == 'Algorithm 2')
    ties = sum(1 for r in results if r['winner'] == 'Tie')
    
    print(f"Algorithm 1 wins: {algo1_wins}/{num_tests}")
    print(f"Algorithm 2 wins: {algo2_wins}/{num_tests}")
    print(f"Ties: {ties}/{num_tests}")
    
    if algo2_wins > algo1_wins:
        print("\n🏆 Algorithm 2 is the clear winner!")
        print("Key advantages of Algorithm 2:")
        print("• Momentum-like updates with Δ_t")
        print("• Clipping ensures ||Δ_t|| ≤ D")
        print("• Better exploration of solution space")
        print("• More robust convergence")
    elif algo1_wins > algo2_wins:
        print("\n🏆 Algorithm 1 is the clear winner!")
    else:
        print("\n🤝 It's a close competition!")
    
    # Average performance
    avg_loss1 = np.mean([r['algo1_loss'] for r in results])
    avg_loss2 = np.mean([r['algo2_loss'] for r in results])
    
    print(f"\nAverage Algorithm 1 loss: {avg_loss1:.6f}")
    print(f"Average Algorithm 2 loss: {avg_loss2:.6f}")
    
    if avg_loss2 < avg_loss1:
        improvement = ((avg_loss1 - avg_loss2) / avg_loss1) * 100
        print(f"Algorithm 2 is {improvement:.1f}% better on average!")

if __name__ == "__main__":
    comprehensive_test()
