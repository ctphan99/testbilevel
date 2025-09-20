#!/usr/bin/env python3
"""
Parameter tuning for SSIGD with dimension 100, seed 1234.
Tune parameters to reduce upper-level loss without changing the algorithm.
"""

import torch
import numpy as np
import time
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem import StronglyConvexBilevelProblem
from ssigd_cvxpylayers_enhanced import SSIGD


def tune_ssigd_parameters():
    """Tune SSIGD parameters for dimension 100 with seed 1234"""
    print("🔧 Parameter Tuning for SSIGD")
    print("=" * 50)
    print("Dimension: 100")
    print("Seed: 1234")
    print("Goal: Reduce upper-level loss")
    
    # Set random seed
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Problem setup
    dim = 100
    device = 'cpu'
    
    # Create test problem
    print(f"\nCreating problem with dimension {dim}...")
    problem = StronglyConvexBilevelProblem(dim=dim, device=device)
    
    # Create SSIGD instance
    ssigd = SSIGD(problem, device=device)
    
    # Create initial point
    x0 = torch.randn(dim, device=device, dtype=torch.float64) * 0.01  # Small initial point
    
    # Parameter grid to test
    param_grid = [
        # (T, beta, diminishing, description)
        (50, 0.0001, True, "Very small step, diminishing"),
        (50, 0.0005, True, "Small step, diminishing"),
        (50, 0.001, True, "Medium step, diminishing"),
        (100, 0.0001, True, "Very small step, more iterations"),
        (100, 0.0005, True, "Small step, more iterations"),
        (100, 0.001, True, "Medium step, more iterations"),
        (50, 0.0001, False, "Very small step, fixed"),
        (50, 0.0005, False, "Small step, fixed"),
        (50, 0.001, False, "Medium step, fixed"),
    ]
    
    best_result = None
    best_loss = float('inf')
    best_params = None
    
    print(f"\n🧪 Testing {len(param_grid)} parameter combinations...")
    print("=" * 60)
    
    for i, (T, beta, diminishing, description) in enumerate(param_grid):
        print(f"\n[{i+1}/{len(param_grid)}] Testing: {description}")
        print(f"  T={T}, beta={beta}, diminishing={diminishing}")
        
        try:
            # Run optimization
            start_time = time.time()
            result = ssigd.solve(T=T, beta=beta, x0=x0, diminishing=diminishing)
            time_taken = time.time() - start_time
            
            # Extract results
            final_loss = result['final_loss']
            final_grad = result['final_grad_norm']
            
            print(f"  ✅ Final Loss: {final_loss:.6f}")
            print(f"  ✅ Final Gradient: {final_grad:.6f}")
            print(f"  ✅ Time: {time_taken:.2f}s")
            
            # Check if this is the best result
            if final_loss < best_loss:
                best_loss = final_loss
                best_result = result
                best_params = (T, beta, diminishing, description)
                print(f"  🎯 NEW BEST! Loss improved by {best_loss - final_loss:.6f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            continue
    
    # Display best results
    print(f"\n🏆 BEST RESULTS")
    print("=" * 30)
    if best_result is not None:
        print(f"Best Parameters: {best_params[3]}")
        print(f"T={best_params[0]}, beta={best_params[1]}, diminishing={best_params[2]}")
        print(f"Final Loss: {best_result['final_loss']:.6f}")
        print(f"Final Gradient: {best_result['final_grad_norm']:.6f}")
        print(f"Method: {best_result['method']}")
        
        # Show convergence history
        losses = best_result['losses']
        grad_norms = best_result['grad_norms']
        
        print(f"\n📈 CONVERGENCE HISTORY")
        print("=" * 25)
        print(f"Initial loss: {losses[0]:.6f}")
        print(f"Final loss: {losses[-1]:.6f}")
        print(f"Loss improvement: {losses[0] - losses[-1]:.6f}")
        
        if abs(losses[0]) > 1e-10:
            print(f"Loss reduction: {((losses[0] - losses[-1]) / abs(losses[0]) * 100):.2f}%")
        
        print(f"\nInitial gradient norm: {grad_norms[0]:.6f}")
        print(f"Final gradient norm: {grad_norms[-1]:.6f}")
        
        if grad_norms[0] > 1e-10:
            print(f"Gradient reduction: {((grad_norms[0] - grad_norms[-1]) / grad_norms[0] * 100):.2f}%")
        
        # Show first few and last few iterations
        print(f"\n📋 ITERATION HISTORY")
        print("=" * 25)
        print("Iter | Loss      | Grad Norm")
        print("-" * 25)
        for j in [0, 1, 2, len(losses)-3, len(losses)-2, len(losses)-1]:
            if j < len(losses):
                print(f"{j+1:4d} | {losses[j]:8.4f} | {grad_norms[j]:8.4f}")
        
        print(f"\n🎯 RECOMMENDED PARAMETERS")
        print("=" * 30)
        print(f"T = {best_params[0]}")
        print(f"beta = {best_params[1]}")
        print(f"diminishing = {best_params[2]}")
        print(f"x0_scale = 0.01")
        
    else:
        print("❌ No successful runs found!")
    
    return best_result, best_params


if __name__ == "__main__":
    tune_ssigd_parameters()
