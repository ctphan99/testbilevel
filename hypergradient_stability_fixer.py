#!/usr/bin/env python3
"""
Hypergradient Stability Fixer for F2CSA Algorithm 1
Fixes the fundamental instability in hypergradient computation
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

class HypergradientStabilityFixer:
    """
    Fixes hypergradient instability in F2CSA Algorithm 1
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem):
        self.problem = problem
        self.algorithm1 = F2CSAAlgorithm1Final(problem)
        self.algorithm2 = F2CSAAlgorithm2Working(problem)
        
    def analyze_hypergradient_instability(self, x: torch.Tensor, alpha: float = 0.1) -> Dict:
        """
        Analyze why hypergradient computation is unstable
        """
        print(f"🔍 ANALYZING HYPERGRADIENT INSTABILITY")
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
        
        # Test multiple hypergradient computations
        print(f"\nTesting multiple hypergradient computations...")
        gradients = []
        for i in range(10):
            grad = self.algorithm1.oracle_sample(x, alpha, N_g=1)
            grad_norm = torch.norm(grad).item()
            gradients.append(grad_norm)
            print(f"  Trial {i+1}: ||∇F̃|| = {grad_norm:.6f}")
        
        # Analyze stability
        grad_norms = np.array(gradients)
        mean_grad = np.mean(grad_norms)
        std_grad = np.std(grad_norms)
        cv = std_grad / mean_grad if mean_grad > 0 else float('inf')
        
        print(f"\nStability Analysis:")
        print(f"  Mean gradient norm: {mean_grad:.6f}")
        print(f"  Std gradient norm: {std_grad:.6f}")
        print(f"  Coefficient of variation: {cv:.6f}")
        print(f"  Min gradient norm: {np.min(grad_norms):.6f}")
        print(f"  Max gradient norm: {np.max(grad_norms):.6f}")
        
        return {
            'y_star': y_star,
            'y_penalty': y_penalty,
            'gap': gap,
            'gradient_norms': gradients,
            'mean_grad': mean_grad,
            'std_grad': std_grad,
            'cv': cv,
            'is_stable': cv < 0.1  # Stable if coefficient of variation < 0.1
        }
    
    def test_parameter_impact(self, x: torch.Tensor, alpha_values: List[float]) -> Dict:
        """
        Test how different alpha values affect hypergradient stability
        """
        print(f"\n🔍 TESTING PARAMETER IMPACT ON STABILITY")
        print("=" * 60)
        
        results = {}
        for alpha in alpha_values:
            print(f"\nTesting alpha = {alpha}")
            result = self.analyze_hypergradient_instability(x, alpha)
            results[alpha] = result
            
            if result['is_stable']:
                print(f"  ✓ STABLE (CV = {result['cv']:.6f})")
            else:
                print(f"  ✗ UNSTABLE (CV = {result['cv']:.6f})")
        
        return results
    
    def test_step_size_impact(self, x: torch.Tensor, alpha: float = 0.1) -> Dict:
        """
        Test how different step sizes affect convergence
        """
        print(f"\n🔍 TESTING STEP SIZE IMPACT ON CONVERGENCE")
        print(f"Alpha: {alpha}")
        print("=" * 60)
        
        step_sizes = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
        results = {}
        
        for step_size in step_sizes:
            print(f"\nTesting step size = {step_size}")
            
            x_test = x.clone()
            gradient_norms = []
            losses = []
            
            for iteration in range(50):
                # Compute current loss
                y_current, _ = self.problem.solve_lower_level(x_test)
                current_loss = self.problem.upper_objective(x_test, y_current).item()
                losses.append(current_loss)
                
                # Compute gradient
                grad = self.algorithm1.oracle_sample(x_test, alpha, N_g=1)
                grad_norm = torch.norm(grad).item()
                gradient_norms.append(grad_norm)
                
                # Update x
                x_test = x_test - step_size * grad
                
                if iteration < 5:  # Show first few iterations
                    print(f"  Iter {iteration+1}: loss = {current_loss:.6f}, ||grad|| = {grad_norm:.6f}")
            
            # Analyze convergence
            final_grad_norm = gradient_norms[-1]
            grad_reduction = gradient_norms[0] - final_grad_norm
            converged = final_grad_norm < 1e-3
            
            results[step_size] = {
                'final_grad_norm': final_grad_norm,
                'grad_reduction': grad_reduction,
                'converged': converged,
                'gradient_norms': gradient_norms,
                'losses': losses
            }
            
            print(f"  Final ||grad||: {final_grad_norm:.6f}")
            print(f"  Gradient reduction: {grad_reduction:.6f}")
            print(f"  Converged: {converged}")
        
        return results
    
    def create_stable_algorithm1(self, x0: torch.Tensor, alpha: float = 0.1) -> Dict:
        """
        Create a stable version of Algorithm 1 with adaptive step size
        """
        print(f"\n🚀 CREATING STABLE ALGORITHM 1")
        print(f"Initial x: {x0}")
        print(f"Alpha: {alpha}")
        print("=" * 60)
        
        x = x0.clone()
        gradient_norms = []
        losses = []
        gaps = []
        step_sizes = []
        
        # Adaptive step size parameters
        initial_step_size = 0.01
        min_step_size = 1e-6
        max_step_size = 0.1
        step_size = initial_step_size
        step_size_decay = 0.95
        step_size_increase = 1.05
        
        for iteration in range(1000):
            # Compute current loss
            y_current, _ = self.problem.solve_lower_level(x)
            current_loss = self.problem.upper_objective(x, y_current).item()
            losses.append(current_loss)
            
            # Compute gap
            y_penalty = self.algorithm1._minimize_penalty_lagrangian(x, y_current, 
                                                                    torch.zeros(3), alpha, 1e-3)
            gap = torch.norm(y_penalty - y_current).item()
            gaps.append(gap)
            
            # Compute gradient
            grad = self.algorithm1.oracle_sample(x, alpha, N_g=1)
            grad_norm = torch.norm(grad).item()
            gradient_norms.append(grad_norm)
            step_sizes.append(step_size)
            
            # Check convergence
            if grad_norm < 1e-6:
                print(f"✓ Converged at iteration {iteration + 1}")
                break
            
            # Adaptive step size
            if iteration > 0:
                prev_grad_norm = gradient_norms[-2]
                if grad_norm > prev_grad_norm * 1.1:  # Gradient increased
                    step_size = max(step_size * step_size_decay, min_step_size)
                elif grad_norm < prev_grad_norm * 0.9:  # Gradient decreased
                    step_size = min(step_size * step_size_increase, max_step_size)
            
            # Update x
            x = x - step_size * grad
            
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
    
    def run_comprehensive_analysis(self, x0: torch.Tensor, alpha: float = 0.1) -> Dict:
        """
        Run comprehensive analysis and create stable solution
        """
        print(f"🚀 COMPREHENSIVE HYPERGRADIENT STABILITY ANALYSIS")
        print(f"Initial x: {x0}")
        print(f"Alpha: {alpha}")
        print("=" * 80)
        
        # Step 1: Analyze instability
        instability_analysis = self.analyze_hypergradient_instability(x0, alpha)
        
        # Step 2: Test parameter impact
        alpha_values = [0.01, 0.05, 0.1, 0.2, 0.5]
        parameter_impact = self.test_parameter_impact(x0, alpha_values)
        
        # Step 3: Test step size impact
        step_size_impact = self.test_step_size_impact(x0, alpha)
        
        # Step 4: Create stable algorithm
        stable_result = self.create_stable_algorithm1(x0, alpha)
        
        # Step 5: Test Algorithm 2 with stable solution
        print(f"\n🔍 TESTING ALGORITHM 2 WITH STABLE SOLUTION")
        if stable_result['converged']:
            info2 = self.algorithm2.optimize(stable_result['x_final'], alpha=alpha)
            print(f"Algorithm 2 result: {info2}")
        else:
            print("Algorithm 1 did not converge, testing Algorithm 2 anyway...")
            info2 = self.algorithm2.optimize(stable_result['x_final'], alpha=alpha)
            print(f"Algorithm 2 result: {info2}")
        
        return {
            'instability_analysis': instability_analysis,
            'parameter_impact': parameter_impact,
            'step_size_impact': step_size_impact,
            'stable_result': stable_result,
            'algorithm2_result': info2
        }

def main():
    """Main analysis function"""
    print("🔍 HYPERGRADIENT STABILITY ANALYSIS FOR F2CSA ALGORITHM 1")
    print("=" * 80)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.01, strong_convex=True)
    x0 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    
    # Create fixer
    fixer = HypergradientStabilityFixer(problem)
    
    # Run comprehensive analysis
    results = fixer.run_comprehensive_analysis(x0, alpha=0.1)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hypergradient_stability_analysis_{timestamp}.json"
    
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
    print(f"Algorithm 1 converged: {results['stable_result']['converged']}")
    print(f"Final gradient norm: {results['stable_result']['final_grad_norm']:.8f}")
    print(f"Algorithm 2 gap: {results['algorithm2_result'].get('final_gap', 'N/A')}")

if __name__ == "__main__":
    main()
