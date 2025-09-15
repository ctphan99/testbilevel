#!/usr/bin/env python3
"""
Focused Gap Debugger for F2CSA Algorithm 2
Targets specific instability sources and gap convergence issues
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

class FocusedGapDebugger:
    """
    Focused debugger for gap stability and Algorithm 2 convergence
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem):
        self.problem = problem
        self.algorithm1 = F2CSAAlgorithm1Final(problem)
        self.algorithm2 = F2CSAAlgorithm2Working(problem)
        
    def debug_gap_calculation(self, x0: torch.Tensor, alpha: float = 0.1, 
                            max_iterations: int = 50) -> Dict:
        """
        Debug gap calculation step by step to identify instability sources
        """
        print(f"=== DEBUGGING GAP CALCULATION (α = {alpha}) ===")
        
        # Get Algorithm 1 solution first
        print("Step 1: Getting Algorithm 1 solution...")
        info1 = self.algorithm1.optimize(x0, max_iterations=1000, alpha=alpha)
        x1 = info1['x_final']
        print(f"Algorithm 1 solution: x = {x1}")
        
        # Get final gap for Algorithm 1
        y1_final, _ = self.problem.solve_lower_level(x1)
        y1_penalty = self.algorithm1._minimize_penalty_lagrangian(x1, y1_final, 
                                                                 torch.zeros(3), alpha, 1e-3)
        gap1 = torch.norm(y1_penalty - y1_final).item()
        print(f"Algorithm 1 final gap: {gap1:.6f}")
        
        # Now test Algorithm 2 with detailed logging
        print("\nStep 2: Testing Algorithm 2 with detailed logging...")
        
        # Initialize Algorithm 2
        x = x0.clone()
        y = self.problem.solve_lower_level(x)
        
        gap_history = []
        hypergradient_norms = []
        penalty_gaps = []
        
        for t in range(max_iterations):
            print(f"\n--- Iteration {t+1} ---")
            
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
                
            # Compute hypergradient
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
            
            # Update x
            x = x - 0.01 * g_t  # Small step size for stability
            
        return {
            'gap_history': gap_history,
            'hypergradient_norms': hypergradient_norms,
            'final_gap': gap_history[-1] if gap_history else float('inf'),
            'converged': gap_history[-1] < 0.1 if gap_history else False,
            'algorithm1_solution': x1,
            'algorithm1_gap': gap1
        }
    
    def test_parameter_sensitivity(self, x0: torch.Tensor, 
                                 alpha_values: List[float] = [0.01, 0.05, 0.1, 0.2]) -> Dict:
        """
        Test sensitivity to alpha parameter
        """
        print("=== TESTING PARAMETER SENSITIVITY ===")
        
        results = {}
        
        for alpha in alpha_values:
            print(f"\nTesting α = {alpha}")
            result = self.debug_gap_calculation(x0, alpha, max_iterations=30)
            results[alpha] = result
            
            print(f"α = {alpha}: Gap = {result['final_gap']:.6f}, "
                  f"Converged = {result['converged']}")
        
        return results
    
    def find_stable_parameters(self, x0: torch.Tensor) -> Dict:
        """
        Find stable parameter configuration for Algorithm 2
        """
        print("=== FINDING STABLE PARAMETERS ===")
        
        # Test different alpha values
        alpha_values = np.linspace(0.01, 0.3, 10)
        best_alpha = None
        best_gap = float('inf')
        
        for alpha in alpha_values:
            print(f"\nTesting α = {alpha:.3f}")
            result = self.debug_gap_calculation(x0, alpha, max_iterations=20)
            
            if result['converged'] and result['final_gap'] < best_gap:
                best_gap = result['final_gap']
                best_alpha = alpha
                print(f"✓ New best: α = {alpha:.3f}, gap = {best_gap:.6f}")
        
        if best_alpha is not None:
            print(f"\n🎯 Best parameters found: α = {best_alpha:.3f}, gap = {best_gap:.6f}")
            
            # Test with best parameters for longer
            print("\nTesting best parameters for longer run...")
            final_result = self.debug_gap_calculation(x0, best_alpha, max_iterations=100)
            
            return {
                'best_alpha': best_alpha,
                'best_gap': best_gap,
                'final_result': final_result
            }
        else:
            print("❌ No stable parameters found")
            return {'best_alpha': None, 'best_gap': float('inf')}

def main():
    """Main function to run focused gap debugging"""
    print("Starting Focused Gap Debugging for F2CSA Algorithm 2")
    
    # Create problem with correct dimensions
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.01, strong_convex=True)
    x0 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    
    # Create debugger
    debugger = FocusedGapDebugger(problem)
    
    # Test parameter sensitivity
    print("\n" + "="*60)
    sensitivity_results = debugger.test_parameter_sensitivity(x0)
    
    # Find stable parameters
    print("\n" + "="*60)
    stable_params = debugger.find_stable_parameters(x0)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'sensitivity_results': sensitivity_results,
        'stable_params': stable_params,
        'timestamp': timestamp
    }
    
    with open(f'focused_gap_debug_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to focused_gap_debug_{timestamp}.json")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Gap convergence for different alphas
    plt.subplot(2, 2, 1)
    for alpha, result in sensitivity_results.items():
        if result['gap_history']:
            plt.plot(result['gap_history'], label=f'α = {alpha}')
    plt.xlabel('Iteration')
    plt.ylabel('Gap')
    plt.title('Gap Convergence by Alpha')
    plt.legend()
    plt.yscale('log')
    
    # Plot 2: Hypergradient norms
    plt.subplot(2, 2, 2)
    for alpha, result in sensitivity_results.items():
        if result['hypergradient_norms']:
            plt.plot(result['hypergradient_norms'], label=f'α = {alpha}')
    plt.xlabel('Iteration')
    plt.ylabel('Hypergradient Norm')
    plt.title('Hypergradient Norms by Alpha')
    plt.legend()
    plt.yscale('log')
    
    # Plot 3: Final gaps vs alpha
    plt.subplot(2, 2, 3)
    alphas = list(sensitivity_results.keys())
    final_gaps = [sensitivity_results[alpha]['final_gap'] for alpha in alphas]
    plt.plot(alphas, final_gaps, 'o-')
    plt.xlabel('Alpha')
    plt.ylabel('Final Gap')
    plt.title('Final Gap vs Alpha')
    plt.yscale('log')
    
    # Plot 4: Convergence status
    plt.subplot(2, 2, 4)
    converged = [sensitivity_results[alpha]['converged'] for alpha in alphas]
    plt.bar(range(len(alphas)), converged)
    plt.xlabel('Alpha Index')
    plt.ylabel('Converged')
    plt.title('Convergence Status by Alpha')
    plt.xticks(range(len(alphas)), [f'{alpha:.2f}' for alpha in alphas])
    
    plt.tight_layout()
    plt.savefig(f'focused_gap_debug_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plots saved to focused_gap_debug_{timestamp}.png")

if __name__ == "__main__":
    main()
