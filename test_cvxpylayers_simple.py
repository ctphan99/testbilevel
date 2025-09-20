#!/usr/bin/env python3
"""
Simple test comparing CVXPYLayers vs Direct CVXPY for exact Hessian computation in SSIGD.
"""

import torch
import numpy as np
import time
import json
from typing import Dict
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem import StronglyConvexBilevelProblem
from ssigd_correct_final import CorrectSSIGD
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class CVXPYLayersSSIGD(CorrectSSIGD):
    """SSIGD implementation using CVXPYLayers for exact Hessian computation."""
    
    def __init__(self, problem: StronglyConvexBilevelProblem, device='cpu'):
        super().__init__(problem, device)
        self.use_cvxpylayers = True
        self._setup_cvxpy_layer_with_noise()
        
    def _setup_cvxpy_layer_with_noise(self):
        """Setup CVXPYLayers for lower-level optimization with noise parameter"""
        try:
            # Create CVXPY problem with noise parameter
            y_cp = cp.Variable(self.prob.dim)
            q_param = cp.Parameter(self.prob.dim)  # Noise parameter
            
            # Objective with noise: (1/2) * y^T * Q_lower * y + (c_lower + q)^T * y
            objective = cp.Minimize(
                0.5 * cp.quad_form(y_cp, self.prob.Q_lower.cpu().numpy()) + 
                cp.sum(cp.multiply(self.prob.c_lower.cpu().numpy() + q_param, y_cp))
            )
            
            # Box constraints: -1 <= y <= 1
            constraints = [y_cp >= -1, y_cp <= 1]
            
            # Create problem and layer
            problem_cp = cp.Problem(objective, constraints)
            self.cvxpy_layer_noise = CvxpyLayer(problem_cp, parameters=[q_param], variables=[y_cp])
            
            print("✓ CVXPYLayers with noise parameter setup successful")
            
        except Exception as e:
            print(f"✗ CVXPYLayers setup failed: {e}")
            self.cvxpy_layer_noise = None
    
    def solve_ll_with_q_cvxpylayers(self, x: torch.Tensor, q_noise: torch.Tensor) -> torch.Tensor:
        """Solve lower-level problem with noise using CVXPYLayers for exact gradients"""
        if self.cvxpy_layer_noise is None:
            return self.solve_ll_with_q(x, q_noise)  # Fallback to direct CVXPY
            
        try:
            # Solve using CVXPYLayers
            solution, = self.cvxpy_layer_noise(q_noise)
            return solution
            
        except Exception as e:
            print(f"CVXPYLayers solve failed: {e}, falling back to direct CVXPY")
            return self.solve_ll_with_q(x, q_noise)
    
    def grad_F_cvxpylayers(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute gradient using CVXPYLayers for exact Hessian computation."""
        # Get noisy solution using CVXPYLayers
        y_hat = self.solve_ll_with_q_cvxpylayers(x, self.q)
        
        # Compute ∇x f(x, y*(x)) - direct partial derivative w.r.t. x
        x_direct = x.clone().requires_grad_(True)
        y_fixed = y_hat.clone().detach()
        f_direct = self.prob.upper_objective(x_direct, y_fixed)
        grad_x_f = torch.autograd.grad(f_direct, x_direct, retain_graph=True)[0]
        
        # Compute ∇y f(x, y*(x)) - partial derivative w.r.t. y
        x_fixed = x.clone().detach()
        y_partial = y_hat.clone().requires_grad_(True)
        f_partial = self.prob.upper_objective(x_fixed, y_partial)
        grad_y_f = torch.autograd.grad(f_partial, y_partial, retain_graph=True)[0]
        
        # Compute ∇y*(x) using finite differences with CVXPYLayers
        eps = 1e-6
        grad_y_star = torch.stack([
            (self.solve_ll_with_q_cvxpylayers(x + eps * torch.eye(self.prob.dim, device=self.device, dtype=self.dtype)[i], self.q) - y_hat) / eps 
            for i in range(self.prob.dim)
        ], dim=1)
        
        # Apply Equation (8): ∇F(x) = ∇x f(x, y*(x)) + [∇y*(x)]ᵀ ∇y f(x, y*(x))
        total_grad = grad_x_f + grad_y_star.T @ grad_y_f
        
        return total_grad
    
    def solve_ll_with_q(self, x: torch.Tensor, q_noise: torch.Tensor) -> torch.Tensor:
        """Override to use CVXPYLayers when available"""
        if self.use_cvxpylayers and self.cvxpy_layer_noise is not None:
            return self.solve_ll_with_q_cvxpylayers(x, q_noise)
        else:
            return super().solve_ll_with_q(x, q_noise)
    
    def grad_F(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Override to use CVXPYLayers gradient computation when available"""
        if self.use_cvxpylayers and self.cvxpy_layer_noise is not None:
            return self.grad_F_cvxpylayers(x, y)
        else:
            return super().grad_F(x, y)


def test_gradient_accuracy(dim=5, num_tests=5):
    """Test gradient computation accuracy between methods"""
    print(f"\n🔍 Testing gradient accuracy with {num_tests} random test points...")
    print(f"Problem dimension: {dim}")
    
    # Create test problem
    problem = StronglyConvexBilevelProblem(dim=dim, device='cpu')
    
    # Create both implementations
    ssigd_direct = CorrectSSIGD(problem, device='cpu')
    ssigd_cvxpylayers = CVXPYLayersSSIGD(problem, device='cpu')
    
    results = {
        'gradient_differences': [],
        'relative_errors': [],
        'cvxpylayers_failures': 0,
        'direct_failures': 0
    }
    
    for i in range(num_tests):
        x = torch.randn(dim, device='cpu', dtype=torch.float64) * 0.1
        
        try:
            # Compute gradients using both methods
            grad_direct = ssigd_direct.grad_F(x, ssigd_direct.solve_ll_with_q(x, ssigd_direct.q))
            grad_cvxpylayers = ssigd_cvxpylayers.grad_F(x, ssigd_cvxpylayers.solve_ll_with_q(x, ssigd_cvxpylayers.q))
            
            # Compute differences
            grad_diff = torch.norm(grad_direct - grad_cvxpylayers).item()
            relative_error = grad_diff / (torch.norm(grad_direct).item() + 1e-10)
            
            results['gradient_differences'].append(grad_diff)
            results['relative_errors'].append(relative_error)
            
            print(f"  Test {i+1}: ||∇F_direct - ∇F_cvxpylayers|| = {grad_diff:.2e}, rel_error = {relative_error:.2e}")
            
        except Exception as e:
            print(f"  Test {i+1} failed: {e}")
            if "cvxpylayers" in str(e).lower():
                results['cvxpylayers_failures'] += 1
            else:
                results['direct_failures'] += 1
    
    # Compute statistics
    if results['gradient_differences']:
        results['mean_grad_diff'] = np.mean(results['gradient_differences'])
        results['std_grad_diff'] = np.std(results['gradient_differences'])
        results['mean_rel_error'] = np.mean(results['relative_errors'])
        results['max_rel_error'] = np.max(results['relative_errors'])
    else:
        results['mean_grad_diff'] = float('inf')
        results['std_grad_diff'] = 0
        results['mean_rel_error'] = float('inf')
        results['max_rel_error'] = float('inf')
    
    return results


def test_convergence_performance(dim=5, T=20, beta=0.01):
    """Test convergence performance of both methods"""
    print(f"\n🚀 Testing convergence performance (T={T}, beta={beta})...")
    
    # Create test problem
    problem = StronglyConvexBilevelProblem(dim=dim, device='cpu')
    
    # Create both implementations
    ssigd_direct = CorrectSSIGD(problem, device='cpu')
    ssigd_cvxpylayers = CVXPYLayersSSIGD(problem, device='cpu')
    
    x0 = torch.randn(dim, device='cpu', dtype=torch.float64) * 0.1
    
    results = {
        'direct': {'time': 0, 'final_loss': 0, 'final_grad': 0, 'converged': False},
        'cvxpylayers': {'time': 0, 'final_loss': 0, 'final_grad': 0, 'converged': False}
    }
    
    # Test direct CVXPY approach
    try:
        start_time = time.time()
        result_direct = ssigd_direct.solve(T=T, beta=beta, x0=x0, diminishing=False)
        results['direct']['time'] = time.time() - start_time
        results['direct']['final_loss'] = result_direct['final_loss']
        results['direct']['final_grad'] = result_direct['final_grad_norm']
        results['direct']['converged'] = result_direct['converged']
        print(f"  Direct CVXPY: {results['direct']['time']:.2f}s, loss={results['direct']['final_loss']:.6f}")
    except Exception as e:
        print(f"  Direct CVXPY failed: {e}")
        results['direct']['time'] = float('inf')
    
    # Test CVXPYLayers approach
    try:
        start_time = time.time()
        result_cvxpylayers = ssigd_cvxpylayers.solve(T=T, beta=beta, x0=x0, diminishing=False)
        results['cvxpylayers']['time'] = time.time() - start_time
        results['cvxpylayers']['final_loss'] = result_cvxpylayers['final_loss']
        results['cvxpylayers']['final_grad'] = result_cvxpylayers['final_grad_norm']
        results['cvxpylayers']['converged'] = result_cvxpylayers['converged']
        print(f"  CVXPYLayers: {results['cvxpylayers']['time']:.2f}s, loss={results['cvxpylayers']['final_loss']:.6f}")
    except Exception as e:
        print(f"  CVXPYLayers failed: {e}")
        results['cvxpylayers']['time'] = float('inf')
    
    return results


def main():
    """Main test function"""
    print("🔬 CVXPYLayers vs Direct CVXPY Accuracy Test")
    print("=" * 50)
    
    # Test parameters
    dim = 5  # Small dimension for testing
    device = 'cpu'
    
    print(f"Problem dimension: {dim}")
    print(f"Device: {device}")
    print(f"Data type: torch.float64")
    
    # Test 1: Gradient accuracy
    grad_results = test_gradient_accuracy(dim=dim, num_tests=5)
    
    # Test 2: Convergence performance
    conv_results = test_convergence_performance(dim=dim, T=20, beta=0.01)
    
    # Print summary
    print("\n📊 SUMMARY RESULTS")
    print("=" * 30)
    
    # Gradient accuracy analysis
    grad_acc = grad_results
    if grad_acc['mean_rel_error'] < 1e-3:
        accuracy_status = "EXCELLENT - Methods agree within 0.1%"
    elif grad_acc['mean_rel_error'] < 1e-2:
        accuracy_status = "GOOD - Methods agree within 1%"
    elif grad_acc['mean_rel_error'] < 1e-1:
        accuracy_status = "FAIR - Methods agree within 10%"
    else:
        accuracy_status = "POOR - Significant disagreement"
    
    print(f"Gradient Accuracy: {accuracy_status}")
    print(f"Mean gradient difference: {grad_acc['mean_grad_diff']:.2e}")
    print(f"Mean relative error: {grad_acc['mean_rel_error']:.2e}")
    print(f"Max relative error: {grad_acc['max_rel_error']:.2e}")
    print(f"CVXPYLayers failures: {grad_acc['cvxpylayers_failures']}")
    print(f"Direct CVXPY failures: {grad_acc['direct_failures']}")
    
    # Performance analysis
    conv_perf = conv_results
    print(f"\nDirect CVXPY time: {conv_perf['direct']['time']:.2f}s")
    print(f"CVXPYLayers time: {conv_perf['cvxpylayers']['time']:.2f}s")
    print(f"Direct final loss: {conv_perf['direct']['final_loss']:.6f}")
    print(f"CVXPYLayers final loss: {conv_perf['cvxpylayers']['final_loss']:.6f}")
    
    # Overall recommendation
    if (grad_acc['mean_rel_error'] < 1e-2 and 
        conv_perf['cvxpylayers']['time'] < conv_perf['direct']['time'] * 1.5):
        recommendation = "USE CVXPYLayers - Better accuracy and reasonable performance"
    elif grad_acc['mean_rel_error'] < 1e-1:
        recommendation = "USE CVXPYLayers - Better accuracy despite performance cost"
    else:
        recommendation = "USE Direct CVXPY - CVXPYLayers may have issues"
    
    print(f"\nRecommendation: {recommendation}")
    
    # Save results
    results = {
        'dimension': dim,
        'device': device,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'gradient_accuracy': grad_results,
        'convergence_performance': conv_results,
        'recommendation': recommendation
    }
    
    output_file = f"cvxpylayers_accuracy_test_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()
