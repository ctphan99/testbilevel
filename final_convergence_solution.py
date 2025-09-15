#!/usr/bin/env python3
"""
Final Convergence Solution for F2CSA
Addresses the remaining convergence issues in Algorithm 1
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

class FinalConvergenceSolution:
    """
    Final solution for F2CSA convergence issues
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem):
        self.problem = problem
        self.algorithm1 = F2CSAAlgorithm1Final(problem)
        self.algorithm2 = F2CSAAlgorithm2Working(problem)
        
    def analyze_convergence_pattern(self, x0: torch.Tensor, alpha: float = 0.1) -> Dict:
        """
        Analyze the convergence pattern to understand why gradient norms oscillate
        """
        print(f"🔍 ANALYZING CONVERGENCE PATTERN")
        print(f"Initial x: {x0}")
        print(f"Alpha: {alpha}")
        print("=" * 60)
        
        x = x0.clone()
        gradient_norms = []
        losses = []
        gaps = []
        step_sizes = []
        
        # Test different step sizes
        step_size_values = [0.001, 0.01, 0.05, 0.1, 0.2]
        results = {}
        
        for step_size in step_size_values:
            print(f"\nTesting step size = {step_size}")
            
            x_test = x0.clone()
            grad_norms = []
            losses_test = []
            
            for iteration in range(50):
                # Compute current loss
                y_current, _ = self.problem.solve_lower_level(x_test)
                current_loss = self.problem.upper_objective(x_test, y_current).item()
                losses_test.append(current_loss)
                
                # Compute gradient
                grad = self.algorithm1.oracle_sample(x_test, alpha, N_g=1)
                grad_norm = torch.norm(grad).item()
                grad_norms.append(grad_norm)
                
                # Update x
                x_test = x_test - step_size * grad
                
                if iteration < 5:  # Show first few iterations
                    print(f"  Iter {iteration+1}: loss = {current_loss:.6f}, ||grad|| = {grad_norm:.6f}")
            
            # Analyze convergence
            final_grad_norm = grad_norms[-1]
            grad_reduction = grad_norms[0] - final_grad_norm
            grad_stability = np.std(grad_norms[-10:]) if len(grad_norms) >= 10 else np.std(grad_norms)
            
            results[step_size] = {
                'final_grad_norm': final_grad_norm,
                'grad_reduction': grad_reduction,
                'grad_stability': grad_stability,
                'gradient_norms': grad_norms,
                'losses': losses_test
            }
            
            print(f"  Final ||grad||: {final_grad_norm:.6f}")
            print(f"  Gradient reduction: {grad_reduction:.6f}")
            print(f"  Gradient stability (std): {grad_stability:.6f}")
        
        return results
    
    def create_adaptive_algorithm1(self, x0: torch.Tensor, alpha: float = 0.1) -> Dict:
        """
        Create an adaptive version of Algorithm 1 that handles convergence issues
        """
        print(f"\n🚀 CREATING ADAPTIVE ALGORITHM 1")
        print(f"Initial x: {x0}")
        print(f"Alpha: {alpha}")
        print("=" * 60)
        
        x = x0.clone()
        gradient_norms = []
        losses = []
        gaps = []
        step_sizes = []
        
        # Adaptive parameters
        initial_step_size = 0.01
        min_step_size = 1e-6
        max_step_size = 0.1
        step_size = initial_step_size
        
        # Convergence tracking
        grad_history = []
        loss_history = []
        
        for iteration in range(1000):
            # Compute current loss
            y_current, _ = self.problem.solve_lower_level(x)
            current_loss = self.problem.upper_objective(x, y_current).item()
            losses.append(current_loss)
            loss_history.append(current_loss)
            
            # Compute gap
            y_penalty = self.algorithm1._minimize_penalty_lagrangian(x, y_current, 
                                                                    torch.zeros(3), alpha, 1e-3)
            gap = torch.norm(y_penalty - y_current).item()
            gaps.append(gap)
            
            # Compute gradient
            grad = self.algorithm1.oracle_sample(x, alpha, N_g=1)
            grad_norm = torch.norm(grad).item()
            gradient_norms.append(grad_norm)
            grad_history.append(grad_norm)
            
            # Adaptive step size based on gradient history
            if iteration > 10:
                recent_grads = grad_history[-10:]
                grad_std = np.std(recent_grads)
                grad_mean = np.mean(recent_grads)
                
                # If gradient is oscillating, reduce step size
                if grad_std > grad_mean * 0.5:  # High variability
                    step_size = max(step_size * 0.9, min_step_size)
                # If gradient is decreasing, increase step size
                elif grad_norm < grad_history[-2] * 0.9:  # Decreasing
                    step_size = min(step_size * 1.05, max_step_size)
            
            # Check convergence
            if grad_norm < 1e-6:
                print(f"✓ Converged at iteration {iteration + 1}")
                break
            
            # Update x
            x = x - step_size * grad
            step_sizes.append(step_size)
            
            # Log progress
            if iteration < 10 or iteration % 100 == 0:
                print(f"Iter {iteration+1}: loss = {current_loss:.6f}, ||grad|| = {grad_norm:.6f}, step = {step_size:.6f}, gap = {gap:.6f}")
        
        return {
            'x_final': x,
            'gradient_norms': gradient_norms,
            'losses': losses,
            'gaps': gaps,
            'step_sizes': step_sizes,
            'final_grad_norm': gradient_norms[-1] if gradient_norms else float('inf'),
            'converged': gradient_norms[-1] < 1e-6 if gradient_norms else False,
            'total_iterations': len(gradient_norms)
        }
    
    def test_algorithm2_with_stable_solution(self, x0: torch.Tensor, alpha: float = 0.1) -> Dict:
        """
        Test Algorithm 2 with a stable solution from Algorithm 1
        """
        print(f"\n🔍 TESTING ALGORITHM 2 WITH STABLE SOLUTION")
        print(f"Initial x: {x0}")
        print(f"Alpha: {alpha}")
        print("=" * 60)
        
        # Run adaptive Algorithm 1
        algorithm1_result = self.create_adaptive_algorithm1(x0, alpha)
        
        print(f"\nAlgorithm 1 Results:")
        print(f"  Converged: {algorithm1_result['converged']}")
        print(f"  Final gradient norm: {algorithm1_result['final_grad_norm']:.6f}")
        print(f"  Total iterations: {algorithm1_result['total_iterations']}")
        
        # Test Algorithm 2 with the solution
        print(f"\nTesting Algorithm 2...")
        try:
            info2 = self.algorithm2.optimize(algorithm1_result['x_final'], alpha=alpha)
            print(f"Algorithm 2 result: {info2}")
            
            return {
                'algorithm1_result': algorithm1_result,
                'algorithm2_result': info2,
                'success': True
            }
        except Exception as e:
            print(f"Algorithm 2 failed: {e}")
            return {
                'algorithm1_result': algorithm1_result,
                'algorithm2_result': None,
                'success': False,
                'error': str(e)
            }
    
    def run_comprehensive_test(self, x0: torch.Tensor, alpha: float = 0.1) -> Dict:
        """
        Run comprehensive test of the final solution
        """
        print(f"🚀 COMPREHENSIVE FINAL SOLUTION TEST")
        print(f"Initial x: {x0}")
        print(f"Alpha: {alpha}")
        print("=" * 80)
        
        # Step 1: Analyze convergence pattern
        convergence_analysis = self.analyze_convergence_pattern(x0, alpha)
        
        # Step 2: Test adaptive Algorithm 1
        adaptive_result = self.create_adaptive_algorithm1(x0, alpha)
        
        # Step 3: Test Algorithm 2 with stable solution
        algorithm2_test = self.test_algorithm2_with_stable_solution(x0, alpha)
        
        return {
            'convergence_analysis': convergence_analysis,
            'adaptive_result': adaptive_result,
            'algorithm2_test': algorithm2_test
        }

def main():
    """Main function"""
    print("🚀 FINAL CONVERGENCE SOLUTION FOR F2CSA")
    print("=" * 80)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.01, strong_convex=True)
    x0 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    
    # Create solution
    solution = FinalConvergenceSolution(problem)
    
    # Run comprehensive test
    results = solution.run_comprehensive_test(x0, alpha=0.1)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"final_convergence_solution_{timestamp}.json"
    
    # Convert tensors to lists for JSON serialization
    def convert_tensors(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors(item) for item in obj]
        return obj
    
    results_serializable = convert_tensors(results)
    
    with open(filename, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n📊 RESULTS SAVED TO: {filename}")
    print(f"Algorithm 1 converged: {results['adaptive_result']['converged']}")
    print(f"Final gradient norm: {results['adaptive_result']['final_grad_norm']:.8f}")
    print(f"Algorithm 2 success: {results['algorithm2_test']['success']}")
    
    if results['algorithm2_test']['success']:
        print(f"Algorithm 2 gap: {results['algorithm2_test']['algorithm2_result'].get('final_gap', 'N/A')}")

if __name__ == "__main__":
    main()
