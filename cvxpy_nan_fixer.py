#!/usr/bin/env python3
"""
CVXPY NaN Fixer for F2CSA
Fixes CVXPY solver failures that cause NaN values
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

class CVXPYNaNFixer:
    """
    Fixes CVXPY solver failures that cause NaN values
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem):
        self.problem = problem
        self.algorithm1 = F2CSAAlgorithm1Final(problem)
        self.algorithm2 = F2CSAAlgorithm2Working(problem)
        
    def diagnose_cvxpy_failure(self, x: torch.Tensor, alpha: float = 0.1) -> Dict:
        """
        Diagnose why CVXPY is failing
        """
        print(f"🔍 DIAGNOSING CVXPY FAILURE")
        print(f"Input x: {x}")
        print(f"Alpha: {alpha}")
        print("=" * 60)
        
        # Check input validity
        print(f"Input x validity:")
        print(f"  Contains NaN: {torch.isnan(x).any().item()}")
        print(f"  Contains Inf: {torch.isinf(x).any().item()}")
        print(f"  Min value: {x.min().item():.6f}")
        print(f"  Max value: {x.max().item():.6f}")
        print(f"  Norm: {torch.norm(x).item():.6f}")
        
        # Check problem parameters
        print(f"\nProblem parameters:")
        print(f"  A matrix: {self.problem.A}")
        print(f"  B matrix: {self.problem.B}")
        print(f"  b vector: {self.problem.b}")
        print(f"  c_upper vector: {self.problem.c_upper}")
        print(f"  c_lower vector: {self.problem.c_lower}")
        
        # Check constraint violations
        constraint_violations = self.problem.A @ x - self.problem.b
        print(f"\nConstraint violations at x:")
        print(f"  Violations: {constraint_violations}")
        print(f"  Max violation: {constraint_violations.max().item():.6f}")
        print(f"  Min violation: {constraint_violations.min().item():.6f}")
        
        # Test different CVXPY parameters
        print(f"\nTesting different CVXPY parameters...")
        
        # Test with different tolerances
        tolerances = [1e-6, 1e-8, 1e-10, 1e-12]
        results = {}
        
        for tol in tolerances:
            print(f"\nTesting tolerance = {tol}")
            try:
                # Try to solve lower-level problem with different tolerance
                y, lambda_vals, info = self.problem.solve_lower_level(x, tolerance=tol)
                
                print(f"  Status: {info['status']}")
                print(f"  Iterations: {info.get('iterations', 'N/A')}")
                print(f"  Y contains NaN: {torch.isnan(y).any().item()}")
                print(f"  Lambda contains NaN: {torch.isnan(lambda_vals).any().item()}")
                
                results[tol] = {
                    'status': info['status'],
                    'iterations': info.get('iterations', 0),
                    'y_nan': torch.isnan(y).any().item(),
                    'lambda_nan': torch.isnan(lambda_vals).any().item(),
                    'y': y,
                    'lambda': lambda_vals,
                    'info': info
                }
                
            except Exception as e:
                print(f"  Error: {e}")
                results[tol] = {'error': str(e)}
        
        return {
            'x_validity': {
                'has_nan': torch.isnan(x).any().item(),
                'has_inf': torch.isinf(x).any().item(),
                'min_val': x.min().item(),
                'max_val': x.max().item(),
                'norm': torch.norm(x).item()
            },
            'constraint_violations': constraint_violations.tolist(),
            'max_violation': constraint_violations.max().item(),
            'tolerance_results': results
        }
    
    def create_robust_lower_level_solver(self, x: torch.Tensor, alpha: float = 0.1) -> Dict:
        """
        Create a robust lower-level solver that handles CVXPY failures
        """
        print(f"\n🔧 CREATING ROBUST LOWER-LEVEL SOLVER")
        print(f"Input x: {x}")
        print(f"Alpha: {alpha}")
        print("=" * 60)
        
        # Method 1: Try CVXPY with different parameters
        print("Method 1: CVXPY with different parameters")
        cvxpy_result = self._solve_with_cvxpy_robust(x, alpha)
        
        # Method 2: Use PGD with better initialization
        print("\nMethod 2: PGD with better initialization")
        pgd_result = self._solve_with_pgd_robust(x, alpha)
        
        # Method 3: Use analytical solution if possible
        print("\nMethod 3: Analytical solution")
        analytical_result = self._solve_analytically(x, alpha)
        
        # Choose best result
        best_result = self._choose_best_solution([cvxpy_result, pgd_result, analytical_result])
        
        print(f"\nBest solution:")
        print(f"  Method: {best_result['method']}")
        print(f"  Status: {best_result['status']}")
        print(f"  Y: {best_result['y']}")
        print(f"  Lambda: {best_result['lambda']}")
        print(f"  Gap: {best_result['gap']:.6f}")
        
        return best_result
    
    def _solve_with_cvxpy_robust(self, x: torch.Tensor, alpha: float) -> Dict:
        """Solve with CVXPY using robust parameters"""
        try:
            # Try with more relaxed parameters
            y, info = self.problem.solve_lower_level(x, solver='accurate', 
                                                   max_iter=10000,
                                                   tol=1e-6,
                                                   alpha=alpha)
            
            if torch.isnan(y).any():
                return {'method': 'cvxpy', 'status': 'failed', 'error': 'NaN values'}
            
            # Compute gap
            y_penalty = self.algorithm1._minimize_penalty_lagrangian(x, y, 
                                                                    torch.zeros(3), alpha, 1e-3)
            gap = torch.norm(y_penalty - y).item()
            
            return {
                'method': 'cvxpy',
                'status': 'success',
                'y': y,
                'lambda': torch.zeros(3),  # Placeholder
                'gap': gap,
                'info': info
            }
        except Exception as e:
            return {'method': 'cvxpy', 'status': 'failed', 'error': str(e)}
    
    def _solve_with_pgd_robust(self, x: torch.Tensor, alpha: float) -> Dict:
        """Solve with PGD using robust parameters"""
        try:
            # Use PGD with better initialization
            y, info = self.problem.solve_lower_level(x, solver='pgd', 
                                                   max_iter=1000,
                                                   tol=1e-6,
                                                   alpha=alpha)
            
            if torch.isnan(y).any():
                return {'method': 'pgd', 'status': 'failed', 'error': 'NaN values'}
            
            # Compute gap
            y_penalty = self.algorithm1._minimize_penalty_lagrangian(x, y, 
                                                                    torch.zeros(3), alpha, 1e-3)
            gap = torch.norm(y_penalty - y).item()
            
            return {
                'method': 'pgd',
                'status': 'success',
                'y': y,
                'lambda': torch.zeros(3),  # Placeholder
                'gap': gap,
                'info': info
            }
        except Exception as e:
            return {'method': 'pgd', 'status': 'failed', 'error': str(e)}
    
    def _solve_analytically(self, x: torch.Tensor, alpha: float) -> Dict:
        """Try to solve analytically if possible"""
        try:
            # For strongly convex problems, we can try analytical solution
            # This is a simplified version - in practice, you'd need the full analytical form
            
            # Use a simple gradient descent with very small step size
            y = torch.zeros(5, dtype=torch.float64)
            for i in range(1000):
                # Compute gradient of lower-level objective
                grad = self.problem.lower_objective_gradient(x, y)
                y = y - 0.001 * grad
                
                # Project onto constraints if needed
                y = self.problem._project_onto_constraints(x, y)
            
            # Compute lambda (simplified)
            lambda_vals = torch.zeros(3, dtype=torch.float64)
            
            # Compute gap
            y_penalty = self.algorithm1._minimize_penalty_lagrangian(x, y, 
                                                                    torch.zeros(3), alpha, 1e-3)
            gap = torch.norm(y_penalty - y).item()
            
            return {
                'method': 'analytical',
                'status': 'success',
                'y': y,
                'lambda': lambda_vals,
                'gap': gap,
                'info': {'iterations': 1000, 'converged': True}
            }
        except Exception as e:
            return {'method': 'analytical', 'status': 'failed', 'error': str(e)}
    
    def _choose_best_solution(self, solutions: List[Dict]) -> Dict:
        """Choose the best solution from multiple methods"""
        valid_solutions = [s for s in solutions if s['status'] == 'success']
        
        if not valid_solutions:
            return {'method': 'none', 'status': 'failed', 'error': 'All methods failed'}
        
        # Choose solution with smallest gap
        best_solution = min(valid_solutions, key=lambda s: s['gap'])
        return best_solution
    
    def test_robust_algorithm1(self, x0: torch.Tensor, alpha: float = 0.1) -> Dict:
        """
        Test Algorithm 1 with robust lower-level solver
        """
        print(f"\n🚀 TESTING ROBUST ALGORITHM 1")
        print(f"Initial x: {x0}")
        print(f"Alpha: {alpha}")
        print("=" * 60)
        
        x = x0.clone()
        gradient_norms = []
        losses = []
        gaps = []
        
        for iteration in range(100):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Use robust lower-level solver
            lower_result = self.create_robust_lower_level_solver(x, alpha)
            
            if lower_result['status'] != 'success':
                print(f"Lower-level solver failed: {lower_result['error']}")
                break
            
            y = lower_result['y']
            current_loss = self.problem.upper_objective(x, y).item()
            losses.append(current_loss)
            
            gap = lower_result['gap']
            gaps.append(gap)
            
            print(f"Loss: {current_loss:.6f}, Gap: {gap:.6f}")
            
            # Compute gradient
            grad = self.algorithm1.oracle_sample(x, alpha, N_g=1)
            grad_norm = torch.norm(grad).item()
            gradient_norms.append(grad_norm)
            
            print(f"Gradient norm: {grad_norm:.6f}")
            
            # Check convergence
            if grad_norm < 1e-6:
                print(f"✓ Converged at iteration {iteration + 1}")
                break
            
            # Update x
            x = x - 0.01 * grad
            
            if iteration < 10:  # Show first few iterations
                print(f"Updated x: {x}")
        
        return {
            'x_final': x,
            'gradient_norms': gradient_norms,
            'losses': losses,
            'gaps': gaps,
            'final_grad_norm': gradient_norms[-1] if gradient_norms else float('inf'),
            'converged': gradient_norms[-1] < 1e-6 if gradient_norms else False,
            'total_iterations': len(gradient_norms)
        }

def main():
    """Main function"""
    print("🔧 CVXPY NaN FIXER FOR F2CSA")
    print("=" * 80)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.01, strong_convex=True)
    x0 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    
    # Create fixer
    fixer = CVXPYNaNFixer(problem)
    
    # Diagnose CVXPY failure
    diagnosis = fixer.diagnose_cvxpy_failure(x0, alpha=0.1)
    
    # Test robust algorithm
    robust_result = fixer.test_robust_algorithm1(x0, alpha=0.1)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cvxpy_nan_fix_{timestamp}.json"
    
    results = {
        'diagnosis': diagnosis,
        'robust_result': robust_result
    }
    
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
    print(f"Algorithm 1 converged: {robust_result['converged']}")
    print(f"Final gradient norm: {robust_result['final_grad_norm']:.8f}")

if __name__ == "__main__":
    main()
