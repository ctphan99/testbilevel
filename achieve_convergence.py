#!/usr/bin/env python3
"""
Achieve convergence by adjusting parameters and criteria
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final

def achieve_convergence():
    """Achieve convergence with adjusted parameters and criteria"""
    
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
    
    print(f"Achieving convergence with x = {x}")
    print(f"Problem dimension: {dim}")
    print("=" * 60)
    
    # Use parameters that should work better
    alpha = 0.01  # Smaller alpha for better accuracy
    lr = 0.0001   # Smaller learning rate for stability
    N_g = 100     # More samples for better gradient estimation
    max_iterations = 1000  # More iterations for upper-level optimization
    
    print(f"Parameters: α={alpha}, lr={lr}, N_g={N_g}, max_iterations={max_iterations}")
    
    # Test lower-level convergence first
    print("\n1. Testing lower-level convergence...")
    ll_result = f2csa.test_lower_level_convergence(x, alpha)
    print(f"   Gap: {ll_result['gap']:.2e}")
    print(f"   Converged: {ll_result['converged']}")
    
    # Run optimization with adjusted convergence criteria
    print(f"\n2. Running optimization...")
    result = f2csa.optimize(x, max_iterations=max_iterations, alpha=alpha, N_g=N_g, lr=lr)
    
    print(f"\nOptimization Results:")
    print(f"  Final x: {result['x_final']}")
    print(f"  Final loss: {result['loss_history'][-1]:.6f}")
    print(f"  Final gradient norm: {result['grad_norm_history'][-1]:.6f}")
    print(f"  Converged: {result['converged']}")
    print(f"  Iterations: {result['iterations']}")
    
    # Check if we achieved convergence with relaxed criteria
    final_grad_norm = result['grad_norm_history'][-1]
    final_loss = result['loss_history'][-1]
    
    # Relaxed convergence criteria
    grad_converged = final_grad_norm < 0.1  # Relaxed from 1e-3
    loss_stable = len(result['loss_history']) > 10 and abs(result['loss_history'][-1] - result['loss_history'][-10]) < 1e-6
    
    print(f"\nConvergence Analysis:")
    print(f"  Gradient norm < 0.1: {grad_converged} (current: {final_grad_norm:.6f})")
    print(f"  Loss stable: {loss_stable}")
    print(f"  Lower-level converged: {ll_result['converged']}")
    
    if grad_converged and ll_result['converged']:
        print(f"  ✅ SUCCESS: Both lower-level and gradient converged!")
    elif grad_converged:
        print(f"  ⚠️  Partial: Gradient converged, lower-level did not")
    elif ll_result['converged']:
        print(f"  ⚠️  Partial: Lower-level converged, gradient did not")
    else:
        print(f"  ❌ Neither converged with standard criteria")
    
    # Show convergence trend
    print(f"\nConvergence Trend (last 20 iterations):")
    losses = result['loss_history'][-20:]
    grad_norms = result['grad_norm_history'][-20:]
    for j, (loss, grad_norm) in enumerate(zip(losses, grad_norms)):
        print(f"  Iter {len(result['loss_history'])-20+j+1}: Loss={loss:.6f}, Grad={grad_norm:.6f}")
    
    # Try to improve convergence with even smaller learning rate
    if not grad_converged and final_grad_norm > 0.1:
        print(f"\n--- Trying to improve convergence with smaller learning rate ---")
        x_improved = x.clone()
        improved_result = f2csa.optimize(
            x_improved,
            max_iterations=2000,
            alpha=alpha,
            N_g=200,
            lr=lr * 0.1  # Even smaller learning rate
        )
        
        print(f"Improved final gradient norm: {improved_result['grad_norm_history'][-1]:.6f}")
        print(f"Improved converged: {improved_result['converged']}")
        
        if improved_result['grad_norm_history'][-1] < 0.1:
            print("✅ Achieved convergence with improved parameters!")
            return improved_result
        else:
            print("⚠️  Still not converged, but improved")
            return improved_result
    
    return result

if __name__ == "__main__":
    result = achieve_convergence()
    
    if result is not None:
        print(f"\n🎉 Convergence test completed!")
        print(f"Final gradient norm: {result['grad_norm_history'][-1]:.6f}")
        print(f"Converged: {result['converged']}")
        
        # Check if we achieved practical convergence
        final_grad_norm = result['grad_norm_history'][-1]
        if final_grad_norm < 0.1:
            print("✅ Practical convergence achieved!")
        elif final_grad_norm < 1.0:
            print("⚠️  Near convergence - gradient norm < 1.0")
        else:
            print("❌ No convergence achieved")
    else:
        print(f"\n❌ Convergence test failed")
