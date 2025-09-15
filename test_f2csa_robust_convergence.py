#!/usr/bin/env python3
"""
Robust Convergence Test for F2CSA Algorithm 1
Uses fixed seeds to ensure consistent convergence results
"""

import torch
import numpy as np
from typing import Dict, Tuple
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
import warnings

warnings.filterwarnings('ignore')

class F2CSARobustConvergenceTest:
    """
    Robust convergence test for F2CSA Algorithm 1 with fixed seeds
    """
    
    def __init__(self, dim: int = 5, num_constraints: int = 3, noise_std: float = 0.1):
        self.dim = dim
        self.num_constraints = num_constraints
        self.noise_std = noise_std
        
        # Set fixed seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create problem instance
        self.problem = StronglyConvexBilevelProblem(
            dim=dim,
            num_constraints=num_constraints,
            noise_std=noise_std,
            strong_convex=True,
            device='cpu'
        )
        
        # Initialize algorithm
        self.algorithm = F2CSAAlgorithm1Final(self.problem)
        
    def test_lower_level_convergence_robust(self, alpha: float = 0.05, max_iterations: int = 1000):
        """
        Test lower-level solution convergence with fixed seeds
        """
        print("🔬 Testing Lower-Level Solution Convergence (Robust)")
        print("=" * 60)
        print(f"α = {alpha}, δ = {alpha**3:.6f} (target: < 0.1)")
        print(f"Max iterations: {max_iterations}")
        print(f"Fixed seed: 42")
        print()
        
        # Test multiple random points with fixed seeds
        convergence_results = []
        
        for test_idx in range(5):
            print(f"📊 Test {test_idx + 1}/5")
            
            # Set specific seed for each test
            torch.manual_seed(42 + test_idx)
            np.random.seed(42 + test_idx)
            
            # Generate random upper-level variable
            x = torch.randn(self.dim, dtype=torch.float64, requires_grad=True)
            
            # Test lower-level solver convergence
            result = self.algorithm.test_lower_level_convergence(x, alpha, max_iterations)
            convergence_results.append(result)
            
            print(f"  ✅ Gap: {result['gap']:.6f} (target: < 0.1)")
            print(f"  ✅ Iterations: {result['iterations']}")
            print(f"  ✅ Converged: {result['converged']}")
            print()
        
        # Analyze convergence
        gaps = [r['gap'] for r in convergence_results]
        iterations = [r['iterations'] for r in convergence_results]
        converged_count = sum(1 for r in convergence_results if r['converged'])
        
        print("📈 Convergence Analysis")
        print("-" * 30)
        print(f"Average gap: {np.mean(gaps):.6f} ± {np.std(gaps):.6f}")
        print(f"Average iterations: {np.mean(iterations):.1f} ± {np.std(iterations):.1f}")
        print(f"Converged: {converged_count}/5 tests")
        print(f"Success rate: {converged_count/5*100:.1f}%")
        print()
        
        return convergence_results
    
    def test_hypergradient_accuracy_robust(self, alpha: float = 0.05):
        """
        Test hypergradient accuracy with fixed seeds
        """
        print("🎯 Testing Hypergradient Accuracy (Robust)")
        print("=" * 60)
        print(f"α = {alpha}, δ = {alpha**3:.6f}")
        print(f"Fixed seed: 42")
        print()
        
        # Test multiple random points with fixed seeds
        hypergradient_results = []
        
        for test_idx in range(5):
            print(f"📊 Test {test_idx + 1}/5")
            
            # Set specific seed for each test
            torch.manual_seed(42 + test_idx)
            np.random.seed(42 + test_idx)
            
            # Generate random upper-level variable
            x = torch.randn(self.dim, dtype=torch.float64, requires_grad=True)
            
            # Test hypergradient accuracy
            result = self.algorithm.test_hypergradient_accuracy(x, alpha)
            hypergradient_results.append(result)
            
            print(f"  ✅ Relative error: {result['relative_error']:.6f}")
            print(f"  ✅ Hypergradient norm: {result['hypergradient_norm']:.6f}")
            print(f"  ✅ Finite diff norm: {result['finite_diff_norm']:.6f}")
            print()
        
        # Analyze hypergradient accuracy
        relative_errors = [r['relative_error'] for r in hypergradient_results]
        hypergradient_norms = [r['hypergradient_norm'] for r in hypergradient_results]
        
        print("📈 Hypergradient Accuracy Analysis")
        print("-" * 40)
        print(f"Average relative error: {np.mean(relative_errors):.6f} ± {np.std(relative_errors):.6f}")
        print(f"Average hypergradient norm: {np.mean(hypergradient_norms):.6f} ± {np.std(hypergradient_norms):.6f}")
        print(f"Max relative error: {np.max(relative_errors):.6f}")
        print()
        
        return hypergradient_results
    
    def test_algorithm_1_optimization_robust(self, alpha: float = 0.05, max_iterations: int = 2000):
        """
        Test Algorithm 1 optimization with fixed seeds
        """
        print("🚀 Testing Algorithm 1 Optimization (Robust)")
        print("=" * 60)
        print(f"α = {alpha}, δ = {alpha**3:.6f}")
        print(f"Max iterations: {max_iterations}")
        print(f"Fixed seed: 42")
        print()
        
        # Set fixed seed
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate initial point
        x0 = torch.randn(self.dim, dtype=torch.float64)
        
        # Run optimization
        result = self.algorithm.optimize(x0, max_iterations=max_iterations, alpha=alpha, N_g=10, lr=0.001)
        
        print("📊 Optimization Results")
        print("-" * 30)
        print(f"Final loss: {result['losses'][-1]:.6f}")
        print(f"Final gradient norm: {result['grad_norms'][-1]:.6f}")
        print(f"Total iterations: {len(result['losses'])}")
        print(f"Converged: {result['converged']}")
        print()
        
        # Analyze convergence
        if len(result['losses']) > 10:
            initial_loss = result['losses'][0]
            final_loss = result['losses'][-1]
            loss_reduction = (initial_loss - final_loss) / initial_loss * 100
            
            print("📈 Convergence Analysis")
            print("-" * 30)
            print(f"Initial loss: {initial_loss:.6f}")
            print(f"Final loss: {final_loss:.6f}")
            print(f"Loss reduction: {loss_reduction:.2f}%")
            print()
            
            # Check if loss is still decreasing
            recent_losses = result['losses'][-10:]
            if len(recent_losses) >= 10:
                loss_std = np.std(recent_losses)
                print(f"Recent loss std: {loss_std:.8f}")
                if loss_std < 1e-6:
                    print("✅ Loss has converged (very small variation)")
                else:
                    print("⚠️  Loss still varying (may need more iterations)")
                print()
        
        return result
    
    def run_robust_test(self):
        """
        Run robust convergence test with fixed seeds
        """
        print("🎯 F2CSA Algorithm 1 Robust Convergence Test")
        print("=" * 70)
        print("Following F2CSA_corrected.tex exactly")
        print("Target: δ-accuracy < 0.1, loss convergence")
        print("Using fixed seeds for reproducibility")
        print()
        
        # Test 1: Lower-level convergence
        print("🔬 PHASE 1: Lower-Level Solution Convergence")
        print("=" * 50)
        convergence_results = self.test_lower_level_convergence_robust(alpha=0.05, max_iterations=1000)
        
        # Test 2: Hypergradient accuracy
        print("🎯 PHASE 2: Hypergradient Accuracy")
        print("=" * 50)
        hypergradient_results = self.test_hypergradient_accuracy_robust(alpha=0.05)
        
        # Test 3: Algorithm 1 optimization
        print("🚀 PHASE 3: Algorithm 1 Optimization")
        print("=" * 50)
        optimization_result = self.test_algorithm_1_optimization_robust(alpha=0.05, max_iterations=2000)
        
        # Final summary
        print("🎉 ROBUST TEST SUMMARY")
        print("=" * 50)
        
        # Check convergence success
        converged_count = sum(1 for r in convergence_results if r['converged'])
        avg_gap = np.mean([r['gap'] for r in convergence_results])
        
        print(f"✅ Lower-level convergence: {converged_count}/5 tests passed")
        print(f"✅ Average gap: {avg_gap:.6f} (target: < 0.1)")
        
        # Check hypergradient accuracy
        avg_relative_error = np.mean([r['relative_error'] for r in hypergradient_results])
        print(f"✅ Average hypergradient error: {avg_relative_error:.6f}")
        
        # Check optimization convergence
        print(f"✅ Optimization converged: {optimization_result['converged']}")
        print(f"✅ Final gradient norm: {optimization_result['grad_norms'][-1]:.6f}")
        
        if converged_count >= 4 and avg_gap < 0.1 and optimization_result['converged']:
            print("\n🎉 SUCCESS: All tests passed! Ready for Algorithm 2 implementation.")
            return True
        else:
            print("\n⚠️  Some tests failed. Need to investigate further.")
            return False

def main():
    """Main test function"""
    test = F2CSARobustConvergenceTest()
    success = test.run_robust_test()
    return success

if __name__ == "__main__":
    main()
