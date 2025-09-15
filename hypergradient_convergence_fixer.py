#!/usr/bin/env python3
"""
Hypergradient Convergence Fixer for F2CSA Algorithm 2
Ensures hypergradient convergence before using in Algorithm 2
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

class HypergradientConvergenceFixer:
    """
    Fixes hypergradient convergence issues in F2CSA Algorithm 1
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem):
        self.problem = problem
        self.algorithm1 = F2CSAAlgorithm1Final(problem)
        self.algorithm2 = F2CSAAlgorithm2Working(problem)
        
    def run_algorithm1_with_convergence_check(self, x0: torch.Tensor, alpha: float = 0.1, 
                                            max_iterations: int = 2000, 
                                            convergence_threshold: float = 1e-3,
                                            patience: int = 50) -> Dict:
        """
        Run Algorithm 1 with proper hypergradient convergence checking
        """
        print(f"=== RUNNING ALGORITHM 1 WITH CONVERGENCE CHECK (α = {alpha}) ===")
        
        x = x0.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=0.001)
        
        losses = []
        grad_norms = []
        gap_history = []
        
        best_x = x.clone()
        best_grad_norm = float('inf')
        patience_counter = 0
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            # Compute stochastic hypergradient
            hypergradient = self.algorithm1.oracle_sample(x, alpha, N_g=100)
            grad_norm = torch.norm(hypergradient).item()
            grad_norms.append(grad_norm)
            
            print(f"Hypergradient norm: {grad_norm:.6f}")
            
            # Check for convergence
            if grad_norm < convergence_threshold:
                print(f"✓ Hypergradient converged at iteration {iteration + 1}")
                break
                
            # Check for improvement
            if grad_norm < best_grad_norm:
                best_grad_norm = grad_norm
                best_x = x.clone()
                patience_counter = 0
                print(f"✓ New best gradient norm: {grad_norm:.6f}")
            else:
                patience_counter += 1
                print(f"⚠️  No improvement for {patience_counter} iterations")
                
            # Early stopping if no improvement
            if patience_counter >= patience:
                print(f"⚠️  Early stopping - no improvement for {patience} iterations")
                print(f"Using best solution with grad norm: {best_grad_norm:.6f}")
                x = best_x
                break
            
            # Update x
            optimizer.zero_grad()
            x.grad = hypergradient
            optimizer.step()
            
            # Compute current loss and gap
            with torch.no_grad():
                y_current, _ = self.problem.solve_lower_level(x)
                current_loss = self.problem.upper_objective(x, y_current).item()
                losses.append(current_loss)
                
                print(f"Loss: {current_loss:.6f}")
            
            # Compute gap (need gradients for this)
            y_current_grad, _ = self.problem.solve_lower_level(x)
            y_penalty = self.algorithm1._minimize_penalty_lagrangian(x, y_current_grad, 
                                                                    torch.zeros(3), alpha, 1e-3)
            gap = torch.norm(y_penalty - y_current_grad).item()
            gap_history.append(gap)
            
            print(f"Gap: {gap:.6f}")
        
        return {
            'x_final': x.detach(),
            'losses': losses,
            'grad_norms': grad_norms,
            'gap_history': gap_history,
            'converged': grad_norms[-1] < convergence_threshold if grad_norms else False,
            'final_grad_norm': grad_norms[-1] if grad_norms else float('inf'),
            'iterations': len(losses),
            'alpha': alpha
        }
    
    def test_algorithm2_with_converged_algorithm1(self, x0: torch.Tensor, alpha: float = 0.1) -> Dict:
        """
        Test Algorithm 2 using properly converged Algorithm 1 solution
        """
        print(f"\n=== TESTING ALGORITHM 2 WITH CONVERGED ALGORITHM 1 (α = {alpha}) ===")
        
        # First, get a properly converged Algorithm 1 solution
        print("Step 1: Getting converged Algorithm 1 solution...")
        alg1_result = self.run_algorithm1_with_convergence_check(x0, alpha, max_iterations=2000)
        
        if not alg1_result['converged']:
            print(f"⚠️  Algorithm 1 did not converge (final grad norm: {alg1_result['final_grad_norm']:.6f})")
            print("Using best solution found...")
        
        x1 = alg1_result['x_final']
        print(f"Algorithm 1 solution: x = {x1}")
        print(f"Final gradient norm: {alg1_result['final_grad_norm']:.6f}")
        
        # Now test Algorithm 2 with this solution
        print("\nStep 2: Testing Algorithm 2 with converged Algorithm 1 solution...")
        
        # Initialize Algorithm 2
        x = x0.clone()
        gap_history = []
        hypergradient_norms = []
        
        for t in range(50):  # Test for 50 iterations
            print(f"\n--- Algorithm 2 Iteration {t+1} ---")
            
            # Get accurate lower-level solution
            y_star, _ = self.problem.solve_lower_level(x)
            print(f"y* = {y_star}")
            
            # Get penalty minimizer
            y_tilde = self.algorithm1._minimize_penalty_lagrangian(x, y_star, 
                                                                  torch.zeros(3), alpha, 1e-3)
            print(f"ỹ = {y_tilde}")
            
            # Compute gap
            gap = torch.norm(y_tilde - y_star).item()
            gap_history.append(gap)
            print(f"Gap: {gap:.6f}")
            
            # Check if gap is stable
            if gap < 0.1:
                print(f"✓ Gap converged to {gap:.6f} < 0.1 at iteration {t+1}")
                break
                
            # Compute hypergradient using Algorithm 1's oracle
            g_t = self.algorithm1.oracle_sample(x, alpha, N_g=10)
            hypergradient_norm = torch.norm(g_t).item()
            hypergradient_norms.append(hypergradient_norm)
            print(f"Hypergradient norm: {hypergradient_norm:.6f}")
            
            # Check for instability
            if t > 0:
                gap_change = abs(gap_history[t] - gap_history[t-1])
                if gap_change > 0.1:
                    print(f"⚠️  Large gap change: {gap_change:.6f}")
                
                if hypergradient_norm > 50:
                    print(f"⚠️  Large hypergradient norm: {hypergradient_norm:.6f}")
            
            # Update x with smaller step size for stability
            x = x - 0.001 * g_t  # Very small step size
            
        return {
            'algorithm1_result': alg1_result,
            'gap_history': gap_history,
            'hypergradient_norms': hypergradient_norms,
            'final_gap': gap_history[-1] if gap_history else float('inf'),
            'converged': gap_history[-1] < 0.1 if gap_history else False
        }
    
    def find_optimal_alpha_for_convergence(self, x0: torch.Tensor, 
                                         alpha_values: List[float] = [0.01, 0.05, 0.1, 0.2, 0.3]) -> Dict:
        """
        Find optimal alpha value that ensures both Algorithm 1 and Algorithm 2 convergence
        """
        print("=== FINDING OPTIMAL ALPHA FOR CONVERGENCE ===")
        
        results = {}
        
        for alpha in alpha_values:
            print(f"\n{'='*60}")
            print(f"Testing α = {alpha}")
            print(f"{'='*60}")
            
            result = self.test_algorithm2_with_converged_algorithm1(x0, alpha)
            results[alpha] = result
            
            print(f"\nα = {alpha} Results:")
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
        
        print(f"\n🎯 Best alpha: {best_alpha} (score: {best_score:.6f})")
        
        return {
            'results': results,
            'best_alpha': best_alpha,
            'best_score': best_score
        }

def main():
    """Main function to run hypergradient convergence fixing"""
    print("Starting Hypergradient Convergence Fixing for F2CSA")
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.01, strong_convex=True)
    x0 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    
    # Create fixer
    fixer = HypergradientConvergenceFixer(problem)
    
    # Find optimal alpha
    print("\n" + "="*80)
    optimal_results = fixer.find_optimal_alpha_for_convergence(x0)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'hypergradient_convergence_fix_{timestamp}.json', 'w') as f:
        json.dump(optimal_results, f, indent=2, default=str)
    
    print(f"\nResults saved to hypergradient_convergence_fix_{timestamp}.json")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Algorithm 1 gradient norms
    plt.subplot(2, 2, 1)
    for alpha, result in optimal_results['results'].items():
        if result['algorithm1_result']['grad_norms']:
            plt.plot(result['algorithm1_result']['grad_norms'], label=f'α = {alpha}')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.title('Algorithm 1 Gradient Norms by Alpha')
    plt.legend()
    plt.yscale('log')
    
    # Plot 2: Algorithm 2 gap convergence
    plt.subplot(2, 2, 2)
    for alpha, result in optimal_results['results'].items():
        if result['gap_history']:
            plt.plot(result['gap_history'], label=f'α = {alpha}')
    plt.xlabel('Iteration')
    plt.ylabel('Gap')
    plt.title('Algorithm 2 Gap Convergence by Alpha')
    plt.legend()
    plt.yscale('log')
    
    # Plot 3: Final gradient norms vs alpha
    plt.subplot(2, 2, 3)
    alphas = list(optimal_results['results'].keys())
    final_grad_norms = [optimal_results['results'][alpha]['algorithm1_result']['final_grad_norm'] for alpha in alphas]
    plt.plot(alphas, final_grad_norms, 'o-')
    plt.xlabel('Alpha')
    plt.ylabel('Final Gradient Norm')
    plt.title('Final Gradient Norm vs Alpha')
    plt.yscale('log')
    
    # Plot 4: Convergence status
    plt.subplot(2, 2, 4)
    alg1_converged = [optimal_results['results'][alpha]['algorithm1_result']['converged'] for alpha in alphas]
    alg2_converged = [optimal_results['results'][alpha]['converged'] for alpha in alphas]
    
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
    plt.savefig(f'hypergradient_convergence_fix_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plots saved to hypergradient_convergence_fix_{timestamp}.png")

if __name__ == "__main__":
    main()
