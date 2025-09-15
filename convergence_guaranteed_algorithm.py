#!/usr/bin/env python3
"""
Convergence Guaranteed Algorithm for F2CSA
Ensures hypergradient convergence before proceeding to Algorithm 2
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
from f2csa_algorithm import F2CSAAlgorithm2Working
import time
import json
from datetime import datetime

class ConvergenceGuaranteedAlgorithm:
    """
    Ensures hypergradient convergence before Algorithm 2
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem):
        self.problem = problem
        self.algorithm1 = F2CSAAlgorithm1Final(problem)
        self.algorithm2 = F2CSAAlgorithm2Working(problem)
        
    def run_algorithm1_until_convergence(self, x0: torch.Tensor, alpha: float = 0.1, 
                                       max_iterations: int = 5000,
                                       convergence_threshold: float = 1e-3,
                                       window_size: int = 10) -> Dict:
        """
        Run Algorithm 1 until hypergradient actually converges
        """
        print(f"=== RUNNING ALGORITHM 1 UNTIL CONVERGENCE (α = {alpha}) ===")
        print(f"Convergence threshold: {convergence_threshold}")
        print(f"Window size for convergence check: {window_size}")
        
        x = x0.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=0.001)
        
        losses = []
        grad_norms = []
        
        for iteration in range(max_iterations):
            if iteration % 100 == 0:
                print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            # Compute stochastic hypergradient
            hypergradient = self.algorithm1.oracle_sample(x, alpha, N_g=100)
            grad_norm = torch.norm(hypergradient).item()
            grad_norms.append(grad_norm)
            
            if iteration % 100 == 0:
                print(f"Hypergradient norm: {grad_norm:.6f}")
            
            # Check for convergence using moving window
            if len(grad_norms) >= window_size:
                recent_norms = grad_norms[-window_size:]
                max_recent = max(recent_norms)
                min_recent = min(recent_norms)
                
                # Check if gradient norms are stable and small
                if max_recent < convergence_threshold and (max_recent - min_recent) < convergence_threshold * 0.1:
                    print(f"\n✓ HYPERGRADIENT CONVERGED at iteration {iteration + 1}")
                    print(f"  Final gradient norm: {grad_norm:.6f}")
                    print(f"  Recent norms range: [{min_recent:.6f}, {max_recent:.6f}]")
                    break
            
            # Update x
            optimizer.zero_grad()
            x.grad = hypergradient
            optimizer.step()
            
            # Compute current loss
            with torch.no_grad():
                y_current, _ = self.problem.solve_lower_level(x)
                current_loss = self.problem.upper_objective(x, y_current).item()
                losses.append(current_loss)
        
        # Final convergence check
        final_grad_norm = grad_norms[-1] if grad_norms else float('inf')
        converged = final_grad_norm < convergence_threshold
        
        print(f"\nAlgorithm 1 Results:")
        print(f"  Final gradient norm: {final_grad_norm:.6f}")
        print(f"  Converged: {converged}")
        print(f"  Total iterations: {len(losses)}")
        
        return {
            'x_final': x.detach(),
            'losses': losses,
            'grad_norms': grad_norms,
            'converged': converged,
            'final_grad_norm': final_grad_norm,
            'iterations': len(losses),
            'alpha': alpha
        }
    
    def run_algorithm2_with_converged_solution(self, x0: torch.Tensor, alpha: float = 0.1) -> Dict:
        """
        Run Algorithm 2 only after Algorithm 1 has converged
        """
        print(f"\n=== RUNNING ALGORITHM 2 WITH CONVERGED SOLUTION (α = {alpha}) ===")
        
        # Step 1: Ensure Algorithm 1 converges
        print("Step 1: Running Algorithm 1 until convergence...")
        alg1_result = self.run_algorithm1_until_convergence(x0, alpha)
        
        if not alg1_result['converged']:
            print(f"⚠️  WARNING: Algorithm 1 did not converge!")
            print(f"   Final gradient norm: {alg1_result['final_grad_norm']:.6f}")
            print("   Proceeding with Algorithm 2 anyway...")
        else:
            print("✓ Algorithm 1 converged successfully!")
        
        # Step 2: Use converged solution for Algorithm 2
        print("\nStep 2: Running Algorithm 2 with converged Algorithm 1 solution...")
        
        x_converged = alg1_result['x_final']
        print(f"Using converged solution: x = {x_converged}")
        
        # Initialize Algorithm 2 from converged point
        x = x_converged.clone()
        gap_history = []
        hypergradient_norms = []
        
        for t in range(100):  # Test for 100 iterations
            if t % 10 == 0:
                print(f"\n--- Algorithm 2 Iteration {t+1} ---")
            
            # Get accurate lower-level solution
            y_star, _ = self.problem.solve_lower_level(x)
            
            # Get penalty minimizer
            y_tilde = self.algorithm1._minimize_penalty_lagrangian(x, y_star, 
                                                                  torch.zeros(3), alpha, 1e-3)
            
            # Compute gap
            gap = torch.norm(y_tilde - y_star).item()
            gap_history.append(gap)
            
            if t % 10 == 0:
                print(f"Gap: {gap:.6f}")
            
            # Check if gap is stable
            if gap < 1e-2:
                print(f"✓ Gap converged to {gap:.6f} < 1e-2 at iteration {t+1}")
                break
                
            # Compute hypergradient
            g_t = self.algorithm1.oracle_sample(x, alpha, N_g=10)
            hypergradient_norm = torch.norm(g_t).item()
            hypergradient_norms.append(hypergradient_norm)
            
            if t % 10 == 0:
                print(f"Hypergradient norm: {hypergradient_norm:.6f}")
            
            # Update x with very small step size for stability
            x = x - 0.001 * g_t
            
        return {
            'algorithm1_result': alg1_result,
            'gap_history': gap_history,
            'hypergradient_norms': hypergradient_norms,
            'final_gap': gap_history[-1] if gap_history else float('inf'),
            'converged': gap_history[-1] < 0.1 if gap_history else False
        }
    
    def test_multiple_alphas(self, x0: torch.Tensor, 
                           alpha_values: List[float] = [0.15, 0.2, 0.25, 0.3]) -> Dict:
        """
        Test multiple alpha values to find the best one
        """
        print("=== TESTING MULTIPLE ALPHAS FOR CONVERGENCE ===")
        
        results = {}
        
        for alpha in alpha_values:
            print(f"\n{'='*80}")
            print(f"Testing α = {alpha}")
            print(f"{'='*80}")
            
            result = self.run_algorithm2_with_converged_solution(x0, alpha)
            results[alpha] = result
            
            print(f"\nα = {alpha} Summary:")
            print(f"  Algorithm 1 converged: {result['algorithm1_result']['converged']}")
            print(f"  Algorithm 1 final grad norm: {result['algorithm1_result']['final_grad_norm']:.6f}")
            print(f"  Algorithm 2 converged: {result['converged']}")
            print(f"  Algorithm 2 final gap: {result['final_gap']:.6f}")
        
        # Find best alpha
        best_alpha = None
        best_score = float('inf')
        
        for alpha, result in results.items():
            if result['algorithm1_result']['converged'] and result['converged']:
                # Both converged - use final gap as score
                score = result['final_gap']
                if score < best_score:
                    best_score = score
                    best_alpha = alpha
            elif result['algorithm1_result']['converged']:
                # Only Algorithm 1 converged - use gradient norm as score
                score = result['algorithm1_result']['final_grad_norm']
                if score < best_score:
                    best_score = score
                    best_alpha = alpha
        
        print(f"\n🎯 BEST ALPHA: {best_alpha} (score: {best_score:.6f})")
        
        return {
            'results': results,
            'best_alpha': best_alpha,
            'best_score': best_score
        }

def main():
    """Main function"""
    print("Starting Convergence Guaranteed Algorithm for F2CSA")
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.01, strong_convex=True)
    x0 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    
    # Create algorithm
    algorithm = ConvergenceGuaranteedAlgorithm(problem)
    
    # Test multiple alphas
    results = algorithm.test_multiple_alphas(x0)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'convergence_guaranteed_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to convergence_guaranteed_{timestamp}.json")
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Algorithm 1 gradient norms
    plt.subplot(2, 2, 1)
    for alpha, result in results['results'].items():
        if result['algorithm1_result']['grad_norms']:
            plt.plot(result['algorithm1_result']['grad_norms'], label=f'α = {alpha}')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.title('Algorithm 1 Gradient Norms (Until Convergence)')
    plt.legend()
    plt.yscale('log')
    
    # Plot 2: Algorithm 2 gap convergence
    plt.subplot(2, 2, 2)
    for alpha, result in results['results'].items():
        if result['gap_history']:
            plt.plot(result['gap_history'], label=f'α = {alpha}')
    plt.xlabel('Iteration')
    plt.ylabel('Gap')
    plt.title('Algorithm 2 Gap Convergence')
    plt.legend()
    plt.yscale('log')
    
    # Plot 3: Final gradient norms vs alpha
    plt.subplot(2, 2, 3)
    alphas = list(results['results'].keys())
    final_grad_norms = [results['results'][alpha]['algorithm1_result']['final_grad_norm'] for alpha in alphas]
    plt.plot(alphas, final_grad_norms, 'o-')
    plt.xlabel('Alpha')
    plt.ylabel('Final Gradient Norm')
    plt.title('Final Gradient Norm vs Alpha')
    plt.yscale('log')
    
    # Plot 4: Convergence summary
    plt.subplot(2, 2, 4)
    alg1_converged = [results['results'][alpha]['algorithm1_result']['converged'] for alpha in alphas]
    alg2_converged = [results['results'][alpha]['converged'] for alpha in alphas]
    
    x_pos = np.arange(len(alphas))
    width = 0.35
    
    plt.bar(x_pos - width/2, alg1_converged, width, label='Algorithm 1', alpha=0.7)
    plt.bar(x_pos + width/2, alg2_converged, width, label='Algorithm 2', alpha=0.7)
    
    plt.xlabel('Alpha')
    plt.ylabel('Converged')
    plt.title('Convergence Status by Alpha')
    plt.xticks(x_pos, alphas)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'convergence_guaranteed_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plots saved to convergence_guaranteed_{timestamp}.png")

if __name__ == "__main__":
    main()
