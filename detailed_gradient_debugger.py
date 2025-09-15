#!/usr/bin/env python3
"""
Detailed Gradient Debugger for F2CSA Algorithm 1
Adds comprehensive logging to understand gradient norm instability
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

class DetailedGradientDebugger:
    """
    Detailed debugger for gradient norm instability in Algorithm 1
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem):
        self.problem = problem
        self.algorithm1 = F2CSAAlgorithm1Final(problem)
        self.algorithm2 = F2CSAAlgorithm2Working(problem)
        
    def debug_gradient_components(self, x0: torch.Tensor, alpha: float = 0.1, 
                                max_iterations: int = 1000) -> Dict:
        """
        Debug gradient components to understand instability
        """
        print(f"🔍 DEBUGGING GRADIENT COMPONENTS")
        print(f"Initial x: {x0}")
        print(f"Alpha: {alpha}")
        print(f"Max iterations: {max_iterations}")
        print("=" * 60)
        
        x = x0.clone().requires_grad_(True)
        gradient_norms = []
        loss_history = []
        gap_history = []
        
        # Detailed logging for first few iterations
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Compute current loss
            y_current, _ = self.problem.solve_lower_level(x)
            current_loss = self.problem.upper_objective(x, y_current)
            loss_history.append(current_loss.item())
            
            print(f"Current loss: {current_loss.item():.8f}")
            
            # Compute gap
            y_penalty = self.algorithm1._minimize_penalty_lagrangian(x, y_current, 
                                                                    torch.zeros(3), alpha, 1e-3)
            gap = torch.norm(y_penalty - y_current).item()
            gap_history.append(gap)
            
            print(f"Current gap: {gap:.8f}")
            
            # Compute gradient with detailed breakdown
            print("Computing gradient...")
            grad = self.algorithm1.oracle_sample(x, alpha, N_g=1)
            grad_norm = torch.norm(grad).item()
            gradient_norms.append(grad_norm)
            
            print(f"Gradient norm: {grad_norm:.8f}")
            print(f"Gradient components: {grad}")
            
            # Check convergence
            if grad_norm < 1e-6:
                print(f"✓ Converged at iteration {iteration + 1}")
                break
                
            # Update x
            x = x - alpha * grad
            print(f"Updated x: {x}")
            
            # Detailed logging for first 10 iterations
            if iteration < 10:
                print(f"  Loss change: {loss_history[-1] - loss_history[-2] if len(loss_history) > 1 else 0:.8f}")
                print(f"  Gap change: {gap_history[-1] - gap_history[-2] if len(gap_history) > 1 else 0:.8f}")
                print(f"  Gradient norm change: {grad_norm - gradient_norms[-2] if len(gradient_norms) > 1 else 0:.8f}")
        
        return {
            'gradient_norms': gradient_norms,
            'loss_history': loss_history,
            'gap_history': gap_history,
            'final_grad_norm': gradient_norms[-1] if gradient_norms else float('inf'),
            'converged': gradient_norms[-1] < 1e-6 if gradient_norms else False,
            'total_iterations': len(gradient_norms)
        }
    
    def debug_hypergradient_computation(self, x: torch.Tensor, alpha: float = 0.1) -> Dict:
        """
        Debug hypergradient computation step by step
        """
        print(f"\n🔍 DEBUGGING HYPERGRADIENT COMPUTATION")
        print(f"Input x: {x}")
        print(f"Alpha: {alpha}")
        print("=" * 60)
        
        # Get lower-level solution
        y_star, _ = self.problem.solve_lower_level(x)
        print(f"Lower-level solution y*: {y_star}")
        
        # Compute penalty Lagrangian
        y_penalty = self.algorithm1._minimize_penalty_lagrangian(x, y_star, 
                                                                torch.zeros(3), alpha, 1e-3)
        print(f"Penalty solution y_penalty: {y_penalty}")
        
        # Compute gap
        gap = torch.norm(y_penalty - y_star).item()
        print(f"Gap: {gap:.8f}")
        
        # Compute hypergradient
        print("\nComputing hypergradient...")
        grad = self.algorithm1.oracle_sample(x, alpha, N_g=1)
        grad_norm = torch.norm(grad).item()
        
        print(f"Hypergradient: {grad}")
        print(f"Hypergradient norm: {grad_norm:.8f}")
        
        # Analyze gradient components
        print(f"\nGradient component analysis:")
        for i, g in enumerate(grad):
            print(f"  Component {i}: {g.item():.8f}")
        
        return {
            'y_star': y_star,
            'y_penalty': y_penalty,
            'gap': gap,
            'gradient': grad,
            'gradient_norm': grad_norm
        }
    
    def run_comprehensive_debug(self, x0: torch.Tensor, alpha: float = 0.1) -> Dict:
        """
        Run comprehensive debugging of Algorithm 1
        """
        print(f"🚀 COMPREHENSIVE ALGORITHM 1 DEBUGGING")
        print(f"Initial x: {x0}")
        print(f"Alpha: {alpha}")
        print("=" * 80)
        
        # Step 1: Debug hypergradient computation
        hypergradient_debug = self.debug_hypergradient_computation(x0, alpha)
        
        # Step 2: Debug gradient components over iterations
        gradient_debug = self.debug_gradient_components(x0, alpha, max_iterations=100)
        
        # Step 3: Check if Algorithm 2 works with current solution
        print(f"\n🔍 TESTING ALGORITHM 2 WITH CURRENT SOLUTION")
        x_final = x0.clone()
        for i in range(100):
            y_current, _ = self.problem.solve_lower_level(x_final)
            grad = self.algorithm1.oracle_sample(x_final, alpha, N_g=1)
            x_final = x_final - alpha * grad
            if torch.norm(grad).item() < 1e-6:
                break
        
        print(f"Final x from Algorithm 1: {x_final}")
        
        # Test Algorithm 2
        info2 = self.algorithm2.optimize(x_final, max_iterations=100, alpha=alpha)
        print(f"Algorithm 2 result: {info2}")
        
        return {
            'hypergradient_debug': hypergradient_debug,
            'gradient_debug': gradient_debug,
            'algorithm2_result': info2,
            'final_x': x_final
        }

def main():
    """Main debugging function"""
    print("🔍 DETAILED GRADIENT DEBUGGING FOR F2CSA ALGORITHM 1")
    print("=" * 80)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.01, strong_convex=True)
    x0 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    
    # Create debugger
    debugger = DetailedGradientDebugger(problem)
    
    # Run comprehensive debug
    results = debugger.run_comprehensive_debug(x0, alpha=0.1)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detailed_gradient_debug_{timestamp}.json"
    
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
    print(f"Final gradient norm: {results['gradient_debug']['final_grad_norm']:.8f}")
    print(f"Converged: {results['gradient_debug']['converged']}")
    print(f"Algorithm 2 gap: {results['algorithm2_result'].get('final_gap', 'N/A')}")

if __name__ == "__main__":
    main()
