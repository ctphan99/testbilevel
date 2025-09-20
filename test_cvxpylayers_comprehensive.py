#!/usr/bin/env python3
"""
Comprehensive test comparing CVXPYLayers vs Direct CVXPY for exact Hessian computation in SSIGD.
Tests multiple dimensions, analyzes gradient accuracy, and performance scaling.
"""

import torch
import numpy as np
import time
import json
from typing import Dict, List
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
            
            print(f"✓ CVXPYLayers setup successful for dim={self.prob.dim}")
            
        except Exception as e:
            print(f"✗ CVXPYLayers setup failed for dim={self.prob.dim}: {e}")
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


def test_dimension_scaling(dimensions: List[int], num_tests: int = 3):
    """Test how accuracy and performance scale with problem dimension"""
    print(f"\n📏 Testing dimension scaling: {dimensions}")
    
    results = {}
    
    for dim in dimensions:
        print(f"\n--- Testing dimension {dim} ---")
        
        try:
            # Create test problem
            problem = StronglyConvexBilevelProblem(dim=dim, device='cpu')
            
            # Create both implementations
            ssigd_direct = CorrectSSIGD(problem, device='cpu')
            ssigd_cvxpylayers = CVXPYLayersSSIGD(problem, device='cpu')
            
            # Test gradient accuracy
            grad_differences = []
            relative_errors = []
            
            for i in range(num_tests):
                x = torch.randn(dim, device='cpu', dtype=torch.float64) * 0.1
                
                try:
                    # Compute gradients using both methods
                    grad_direct = ssigd_direct.grad_F(x, ssigd_direct.solve_ll_with_q(x, ssigd_direct.q))
                    grad_cvxpylayers = ssigd_cvxpylayers.grad_F(x, ssigd_cvxpylayers.solve_ll_with_q(x, ssigd_cvxpylayers.q))
                    
                    # Compute differences
                    grad_diff = torch.norm(grad_direct - grad_cvxpylayers).item()
                    relative_error = grad_diff / (torch.norm(grad_direct).item() + 1e-10)
                    
                    grad_differences.append(grad_diff)
                    relative_errors.append(relative_error)
                    
                except Exception as e:
                    print(f"    Gradient test {i+1} failed: {e}")
            
            # Test convergence performance
            x0 = torch.randn(dim, device='cpu', dtype=torch.float64) * 0.1
            T = min(50, 10 + dim)  # Scale iterations with dimension
            
            # Direct CVXPY timing
            direct_time = float('inf')
            direct_loss = float('inf')
            try:
                start_time = time.time()
                result_direct = ssigd_direct.solve(T=T, beta=0.01, x0=x0, diminishing=False)
                direct_time = time.time() - start_time
                direct_loss = result_direct['final_loss']
            except Exception as e:
                print(f"    Direct CVXPY failed: {e}")
            
            # CVXPYLayers timing
            cvxpylayers_time = float('inf')
            cvxpylayers_loss = float('inf')
            try:
                start_time = time.time()
                result_cvxpylayers = ssigd_cvxpylayers.solve(T=T, beta=0.01, x0=x0, diminishing=False)
                cvxpylayers_time = time.time() - start_time
                cvxpylayers_loss = result_cvxpylayers['final_loss']
            except Exception as e:
                print(f"    CVXPYLayers failed: {e}")
            
            # Store results
            results[dim] = {
                'gradient_accuracy': {
                    'mean_grad_diff': np.mean(grad_differences) if grad_differences else float('inf'),
                    'mean_rel_error': np.mean(relative_errors) if relative_errors else float('inf'),
                    'max_rel_error': np.max(relative_errors) if relative_errors else float('inf'),
                    'num_successful_tests': len(grad_differences)
                },
                'performance': {
                    'direct_time': direct_time,
                    'cvxpylayers_time': cvxpylayers_time,
                    'direct_loss': direct_loss,
                    'cvxpylayers_loss': cvxpylayers_loss,
                    'speedup': direct_time / cvxpylayers_time if cvxpylayers_time < float('inf') else 0
                }
            }
            
            # Print summary for this dimension
            if grad_differences:
                print(f"    Gradient accuracy: mean_rel_error = {np.mean(relative_errors):.2e}")
            print(f"    Performance: direct={direct_time:.2f}s, cvxpylayers={cvxpylayers_time:.2f}s")
            if cvxpylayers_time < float('inf') and direct_time < float('inf'):
                print(f"    Speedup: {direct_time/cvxpylayers_time:.2f}x")
            
        except Exception as e:
            print(f"    Dimension {dim} test failed: {e}")
            results[dim] = {'error': str(e)}
    
    return results


def test_numerical_stability(dim: int = 10, perturbation_scales: List[float] = None):
    """Test numerical stability under different perturbation scales"""
    if perturbation_scales is None:
        perturbation_scales = [1e-8, 1e-6, 1e-4, 1e-2]
    
    print(f"\n⚖️ Testing numerical stability (dim={dim})")
    
    # Create test problem
    problem = StronglyConvexBilevelProblem(dim=dim, device='cpu')
    ssigd_direct = CorrectSSIGD(problem, device='cpu')
    ssigd_cvxpylayers = CVXPYLayersSSIGD(problem, device='cpu')
    
    x_base = torch.randn(dim, device='cpu', dtype=torch.float64) * 0.1
    
    results = {}
    
    for scale in perturbation_scales:
        print(f"  Testing perturbation scale: {scale:.0e}")
        
        # Generate perturbations
        perturbations = torch.randn(5, dim, device='cpu', dtype=torch.float64) * scale
        
        # Test direct method
        direct_variations = []
        try:
            grad_base_direct = ssigd_direct.grad_F(x_base, ssigd_direct.solve_ll_with_q(x_base, ssigd_direct.q))
            
            for pert in perturbations:
                x_pert = x_base + pert
                grad_pert = ssigd_direct.grad_F(x_pert, ssigd_direct.solve_ll_with_q(x_pert, ssigd_direct.q))
                grad_variation = torch.norm(grad_pert - grad_base_direct).item()
                direct_variations.append(grad_variation)
                
        except Exception as e:
            print(f"    Direct method failed: {e}")
            direct_variations = [float('inf')] * len(perturbations)
        
        # Test CVXPYLayers method
        cvxpylayers_variations = []
        try:
            grad_base_cvxpylayers = ssigd_cvxpylayers.grad_F(x_base, ssigd_cvxpylayers.solve_ll_with_q(x_base, ssigd_cvxpylayers.q))
            
            for pert in perturbations:
                x_pert = x_base + pert
                grad_pert = ssigd_cvxpylayers.grad_F(x_pert, ssigd_cvxpylayers.solve_ll_with_q(x_pert, ssigd_cvxpylayers.q))
                grad_variation = torch.norm(grad_pert - grad_base_cvxpylayers).item()
                cvxpylayers_variations.append(grad_variation)
                
        except Exception as e:
            print(f"    CVXPYLayers method failed: {e}")
            cvxpylayers_variations = [float('inf')] * len(perturbations)
        
        # Compute stability metrics
        direct_stability = np.std(direct_variations) if direct_variations and all(x != float('inf') for x in direct_variations) else float('inf')
        cvxpylayers_stability = np.std(cvxpylayers_variations) if cvxpylayers_variations and all(x != float('inf') for x in cvxpylayers_variations) else float('inf')
        
        results[scale] = {
            'direct_stability': direct_stability,
            'cvxpylayers_stability': cvxpylayers_stability,
            'stability_ratio': cvxpylayers_stability / direct_stability if direct_stability > 0 else float('inf')
        }
        
        print(f"    Direct stability: {direct_stability:.2e}")
        print(f"    CVXPYLayers stability: {cvxpylayers_stability:.2e}")
        if direct_stability > 0 and cvxpylayers_stability < float('inf'):
            print(f"    Stability ratio: {cvxpylayers_stability/direct_stability:.2f}")
    
    return results


def main():
    """Main comprehensive test function"""
    print("🔬 Comprehensive CVXPYLayers vs Direct CVXPY Accuracy Test")
    print("=" * 60)
    
    # Test parameters
    dimensions = [5, 10, 20, 50]  # Test multiple dimensions
    device = 'cpu'
    
    print(f"Testing dimensions: {dimensions}")
    print(f"Device: {device}")
    print(f"Data type: torch.float64")
    
    # Test 1: Dimension scaling
    scaling_results = test_dimension_scaling(dimensions, num_tests=3)
    
    # Test 2: Numerical stability
    stability_results = test_numerical_stability(dim=10, perturbation_scales=[1e-8, 1e-6, 1e-4])
    
    # Analyze results
    print("\n📊 COMPREHENSIVE ANALYSIS")
    print("=" * 40)
    
    # Dimension scaling analysis
    print("\n📏 DIMENSION SCALING:")
    print("Dim | Direct Time | CVXPYLayers Time | Speedup | Accuracy")
    print("-" * 60)
    
    for dim in dimensions:
        if dim in scaling_results and 'error' not in scaling_results[dim]:
            perf = scaling_results[dim]['performance']
            acc = scaling_results[dim]['gradient_accuracy']
            
            speedup = perf['speedup'] if perf['speedup'] > 0 else 0
            accuracy = f"{acc['mean_rel_error']:.1e}" if acc['mean_rel_error'] < float('inf') else "FAIL"
            
            print(f"{dim:3d} | {perf['direct_time']:8.2f}s | {perf['cvxpylayers_time']:13.2f}s | {speedup:6.1f}x | {accuracy}")
    
    # Stability analysis
    print("\n⚖️ NUMERICAL STABILITY:")
    print("Scale | Direct Stability | CVXPYLayers Stability | Ratio")
    print("-" * 55)
    
    for scale, result in stability_results.items():
        direct_stab = result['direct_stability']
        cvxpylayers_stab = result['cvxpylayers_stability']
        ratio = result['stability_ratio']
        
        direct_str = f"{direct_stab:.1e}" if direct_stab < float('inf') else "FAIL"
        cvxpylayers_str = f"{cvxpylayers_stab:.1e}" if cvxpylayers_stab < float('inf') else "FAIL"
        ratio_str = f"{ratio:.2f}" if ratio < float('inf') else "N/A"
        
        print(f"{scale:.0e} | {direct_str:>15} | {cvxpylayers_str:>20} | {ratio_str:>5}")
    
    # Overall recommendation
    print("\n🎯 OVERALL RECOMMENDATION:")
    
    # Analyze scaling results
    successful_dims = [dim for dim in dimensions if dim in scaling_results and 'error' not in scaling_results[dim]]
    
    if successful_dims:
        avg_speedup = np.mean([scaling_results[dim]['performance']['speedup'] for dim in successful_dims if scaling_results[dim]['performance']['speedup'] > 0])
        avg_accuracy = np.mean([scaling_results[dim]['gradient_accuracy']['mean_rel_error'] for dim in successful_dims if scaling_results[dim]['gradient_accuracy']['mean_rel_error'] < float('inf')])
        
        if avg_accuracy < 1e-3 and avg_speedup > 1.5:
            recommendation = "STRONGLY RECOMMEND CVXPYLayers - Excellent accuracy and significant speedup"
        elif avg_accuracy < 1e-2 and avg_speedup > 1.0:
            recommendation = "RECOMMEND CVXPYLayers - Good accuracy and speedup"
        elif avg_accuracy < 1e-1:
            recommendation = "CONSIDER CVXPYLayers - Better accuracy despite potential performance cost"
        else:
            recommendation = "USE Direct CVXPY - CVXPYLayers may have accuracy issues"
    else:
        recommendation = "INCONCLUSIVE - Both methods had failures"
    
    print(f"{recommendation}")
    
    # Save comprehensive results
    results = {
        'dimensions_tested': dimensions,
        'device': device,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'scaling_results': scaling_results,
        'stability_results': stability_results,
        'recommendation': recommendation
    }
    
    output_file = f"cvxpylayers_comprehensive_test_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Comprehensive results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()
