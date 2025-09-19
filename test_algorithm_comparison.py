#!/usr/bin/env python3
"""
Test script to compare Algorithm 1 vs Algorithm 2 performance
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
from f2csa_algorithm2_working import F2CSAAlgorithm2Working
import time

def test_algorithm_comparison():
    """Compare Algorithm 1 vs Algorithm 2 performance"""
    
    print("🔬 Algorithm Comparison Test")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create problem
    dim = 5
    problem = StronglyConvexBilevelProblem(
        dim=dim, 
        num_constraints=3, 
        noise_std=0.1, 
        strong_convex=True, 
        device='cpu'
    )
    
    # Test point
    x0 = torch.randn(dim, dtype=torch.float64)
    print(f"Test point x0: {x0}")
    print(f"Problem dimension: {dim}")
    print()
    
    # Test different alpha values
    alpha_values = [0.1, 0.05, 0.02, 0.01]
    
    results = {}
    
    for alpha in alpha_values:
        print(f"Testing α = {alpha} (δ = {alpha**3:.6f})")
        print("-" * 40)
        
        # Test Algorithm 1
        print("Testing Algorithm 1...")
        algo1 = F2CSAAlgorithm1Final(problem)
        start_time = time.time()
        result1 = algo1.optimize(x0, max_iterations=100, alpha=alpha, N_g=20, lr=0.001)
        time1 = time.time() - start_time
        
        # Test Algorithm 2
        print("Testing Algorithm 2...")
        algo2 = F2CSAAlgorithm2Working(problem)
        start_time = time.time()
        result2 = algo2.optimize(
            x0, T=100, D=0.05, eta=0.001, 
            delta=alpha**3, alpha=alpha, N_g=20
        )
        time2 = time.time() - start_time
        
        # Compare results
        print(f"\nResults for α = {alpha}:")
        print(f"  Algorithm 1:")
        print(f"    Final loss: {result1['loss_history'][-1]:.6f}")
        print(f"    Final grad norm: {result1['grad_norm_history'][-1]:.6f}")
        print(f"    Converged: {result1['converged']}")
        print(f"    Time: {time1:.2f}s")
        
        print(f"  Algorithm 2:")
        print(f"    Final UL loss: {result2['final_ul_loss']:.6f}")
        print(f"    Final grad norm: {result2['final_gradient_norm']:.6f}")
        print(f"    Converged: {result2['converged']}")
        print(f"    Time: {time2:.2f}s")
        
        # Determine winner
        if result1['loss_history'][-1] < result2['final_ul_loss']:
            winner = "Algorithm 1"
            improvement = result2['final_ul_loss'] - result1['loss_history'][-1]
        else:
            winner = "Algorithm 2"
            improvement = result1['loss_history'][-1] - result2['final_ul_loss']
        
        print(f"  Winner: {winner} (improvement: {improvement:.6f})")
        print()
        
        # Store results
        results[alpha] = {
            'algo1': result1,
            'algo2': result2,
            'time1': time1,
            'time2': time2,
            'winner': winner,
            'improvement': improvement
        }
    
    # Summary
    print("📊 Summary")
    print("=" * 60)
    
    algo1_wins = sum(1 for r in results.values() if r['winner'] == 'Algorithm 1')
    algo2_wins = sum(1 for r in results.values() if r['winner'] == 'Algorithm 2')
    
    print(f"Algorithm 1 wins: {algo1_wins}/{len(alpha_values)}")
    print(f"Algorithm 2 wins: {algo2_wins}/{len(alpha_values)}")
    
    # Average performance
    avg_loss1 = np.mean([r['algo1']['loss_history'][-1] for r in results.values()])
    avg_loss2 = np.mean([r['algo2']['final_ul_loss'] for r in results.values()])
    avg_time1 = np.mean([r['time1'] for r in results.values()])
    avg_time2 = np.mean([r['time2'] for r in results.values()])
    
    print(f"\nAverage performance:")
    print(f"  Algorithm 1: loss = {avg_loss1:.6f}, time = {avg_time1:.2f}s")
    print(f"  Algorithm 2: loss = {avg_loss2:.6f}, time = {avg_time2:.2f}s")
    
    # Best overall
    if avg_loss1 < avg_loss2:
        print(f"\n🏆 Overall winner: Algorithm 1")
        print(f"   Better average loss: {avg_loss2 - avg_loss1:.6f}")
    else:
        print(f"\n🏆 Overall winner: Algorithm 2")
        print(f"   Better average loss: {avg_loss1 - avg_loss2:.6f}")
    
    return results

def test_convergence_behavior():
    """Test convergence behavior of both algorithms"""
    
    print("\n🔄 Convergence Behavior Test")
    print("=" * 60)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.1)
    x0 = torch.randn(5, dtype=torch.float64)
    
    # Test with good parameters
    alpha = 0.05
    print(f"Testing convergence with α = {alpha}")
    
    # Algorithm 1
    print("\nAlgorithm 1 convergence:")
    algo1 = F2CSAAlgorithm1Final(problem)
    result1 = algo1.optimize(x0, max_iterations=50, alpha=alpha, N_g=20, lr=0.001)
    
    print(f"  Final loss: {result1['loss_history'][-1]:.6f}")
    print(f"  Final grad norm: {result1['grad_norm_history'][-1]:.6f}")
    print(f"  Converged: {result1['converged']}")
    
    # Algorithm 2
    print("\nAlgorithm 2 convergence:")
    algo2 = F2CSAAlgorithm2Working(problem)
    result2 = algo2.optimize(
        x0, T=50, D=0.05, eta=0.001, 
        delta=alpha**3, alpha=alpha, N_g=20
    )
    
    print(f"  Final UL loss: {result2['final_ul_loss']:.6f}")
    print(f"  Final grad norm: {result2['final_gradient_norm']:.6f}")
    print(f"  Converged: {result2['converged']}")
    
    # Plot convergence
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss convergence
        ax1.plot(result1['loss_history'], label='Algorithm 1', color='blue')
        ax1.plot(result2['ul_losses'], label='Algorithm 2', color='red')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Convergence')
        ax1.legend()
        ax1.grid(True)
        
        # Gradient norm convergence
        ax2.plot(result1['grad_norm_history'], label='Algorithm 1', color='blue')
        ax2.plot(result2['hypergrad_norms'], label='Algorithm 2', color='red')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gradient Norm')
        ax2.set_title('Gradient Norm Convergence')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('algorithm_comparison.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("\nSaved convergence plot to 'algorithm_comparison.png'")
        
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    # Run comparison test
    results = test_algorithm_comparison()
    
    # Run convergence test
    test_convergence_behavior()
    
    print("\n✅ Test completed!")
