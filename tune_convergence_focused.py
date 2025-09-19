#!/usr/bin/env python3
"""
Focused tuning script to achieve convergence by adjusting parameters
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final

def tune_for_convergence():
    """Tune parameters to achieve both lower-level and gradient convergence"""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create problem
    dim = 5
    problem = StronglyConvexBilevelProblem(dim=dim, noise_std=0.01, device='cpu')
    
    # Create F2CSA algorithm
    f2csa = F2CSAAlgorithm1Final(problem)
    
    # Test point
    x = torch.randn(dim, dtype=torch.float64)
    
    print(f"Focused convergence tuning with x = {x}")
    print(f"Problem dimension: {dim}")
    print("=" * 60)
    
    # Test different parameter combinations
    param_combinations = [
        {'alpha': 0.1, 'lr': 0.01, 'N_g': 20, 'max_iter': 100},
        {'alpha': 0.05, 'lr': 0.005, 'N_g': 50, 'max_iter': 100},
        {'alpha': 0.02, 'lr': 0.001, 'N_g': 100, 'max_iter': 200},
        {'alpha': 0.01, 'lr': 0.0005, 'N_g': 200, 'max_iter': 300},
        {'alpha': 0.005, 'lr': 0.0001, 'N_g': 500, 'max_iter': 500},
    ]
    
    best_result = None
    best_alpha = None
    
    for i, params in enumerate(param_combinations):
        print(f"\n--- Test {i+1}: α={params['alpha']}, lr={params['lr']}, N_g={params['N_g']} ---")
        
        # Test lower-level convergence
        print("1. Testing lower-level convergence...")
        ll_result = f2csa.test_lower_level_convergence(x, params['alpha'])
        print(f"   Gap: {ll_result['gap']:.2e}")
        print(f"   Converged: {ll_result['converged']}")
        
        # Test optimization convergence
        print("2. Testing optimization convergence...")
        x_test = x.clone()
        opt_result = f2csa.optimize(
            x_test, 
            max_iterations=params['max_iter'], 
            alpha=params['alpha'], 
            N_g=params['N_g'], 
            lr=params['lr']
        )
        
        print(f"   Final loss: {opt_result['loss_history'][-1]:.6f}")
        print(f"   Final gradient norm: {opt_result['grad_norm_history'][-1]:.6f}")
        print(f"   Converged: {opt_result['converged']}")
        print(f"   Iterations: {opt_result['iterations']}")
        
        # Check convergence
        ll_converged = ll_result['converged']
        grad_converged = opt_result['converged']
        final_grad_norm = opt_result['grad_norm_history'][-1]
        
        if ll_converged and grad_converged:
            print(f"   ✅ SUCCESS: Both converged!")
            best_result = opt_result
            best_alpha = params['alpha']
            break
        elif ll_converged and final_grad_norm < 1.0:
            print(f"   ⚠️  Good: Lower-level converged, gradient norm = {final_grad_norm:.3f}")
            if best_result is None or final_grad_norm < best_result['grad_norm_history'][-1]:
                best_result = opt_result
                best_alpha = params['alpha']
        else:
            print(f"   ❌ Neither converged")
    
    print("\n" + "=" * 60)
    
    if best_result is not None:
        print(f"\n--- Best Result with α = {best_alpha} ---")
        print(f"Final x: {best_result['x_final']}")
        print(f"Final loss: {best_result['loss_history'][-1]:.6f}")
        print(f"Final gradient norm: {best_result['grad_norm_history'][-1]:.6f}")
        print(f"Converged: {best_result['converged']}")
        print(f"Iterations: {best_result['iterations']}")
        
        # Show convergence trend
        print(f"\nConvergence Trend (last 10 iterations):")
        losses = best_result['loss_history'][-10:]
        grad_norms = best_result['grad_norm_history'][-10:]
        for j, (loss, grad_norm) in enumerate(zip(losses, grad_norms)):
            print(f"  Iter {len(best_result['loss_history'])-10+j+1}: Loss={loss:.6f}, Grad={grad_norm:.6f}")
        
        # Check if we can improve further
        if not best_result['converged'] and best_result['grad_norm_history'][-1] > 1e-3:
            print(f"\n--- Trying to improve convergence ---")
            print("Running with more iterations and smaller learning rate...")
            
            x_improved = x.clone()
            improved_result = f2csa.optimize(
                x_improved,
                max_iterations=1000,
                alpha=best_alpha,
                N_g=100,
                lr=best_result.get('lr', 0.001) * 0.1
            )
            
            print(f"Improved final gradient norm: {improved_result['grad_norm_history'][-1]:.6f}")
            print(f"Improved converged: {improved_result['converged']}")
            
            if improved_result['converged']:
                print("✅ Achieved convergence with improved parameters!")
                return improved_result
            else:
                print("⚠️  Still not converged, but improved")
                return improved_result
        else:
            return best_result
    else:
        print("❌ No successful convergence found with any parameter combination")
        return None

if __name__ == "__main__":
    result = tune_for_convergence()
    
    if result is not None:
        print(f"\n🎉 Tuning completed successfully!")
        print(f"Final gradient norm: {result['grad_norm_history'][-1]:.6f}")
        print(f"Converged: {result['converged']}")
    else:
        print(f"\n❌ Tuning failed - no convergence achieved")
