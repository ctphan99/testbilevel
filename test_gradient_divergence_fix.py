#!/usr/bin/env python3
"""
Diagnose and fix gradient divergence issue
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final

def diagnose_gradient_divergence():
    """Diagnose why gradients are diverging"""
    print("🔍 GRADIENT DIVERGENCE DIAGNOSIS")
    print("=" * 50)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5, 
        num_constraints=3, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    # Create algorithm
    algorithm = F2CSAAlgorithm1Final(problem)
    
    # Test with different alpha values
    alpha_values = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    x0 = torch.randn(5, dtype=torch.float64)
    
    print(f"Test point: {x0}")
    print()
    
    for alpha in alpha_values:
        print(f"Testing α = {alpha}")
        print("-" * 20)
        
        # Get hypergradient
        hypergradient = algorithm.oracle_sample(x0, alpha, 10)
        hypergradient_norm = torch.norm(hypergradient).item()
        
        print(f"Hypergradient norm: {hypergradient_norm:.2f}")
        
        # Check if it's reasonable
        if hypergradient_norm < 100:
            print("✅ Reasonable gradient norm")
        elif hypergradient_norm < 1000:
            print("⚠️  Large but manageable gradient norm")
        else:
            print("❌ Diverging gradient norm")
        print()
    
    return alpha_values

def test_learning_rate_effect():
    """Test different learning rates to fix divergence"""
    print("🔧 TESTING DIFFERENT LEARNING RATES")
    print("=" * 50)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5, 
        num_constraints=3, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    # Create algorithm
    algorithm = F2CSAAlgorithm1Final(problem)
    
    # Test with different learning rates
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    alpha = 0.0005
    x0 = torch.randn(5, dtype=torch.float64)
    
    print(f"Using α = {alpha}")
    print(f"Test point: {x0}")
    print()
    
    for lr in learning_rates:
        print(f"Testing learning rate = {lr}")
        print("-" * 25)
        
        # Run a few iterations
        result = algorithm.optimize(x0, max_iterations=10, alpha=alpha, N_g=10, lr=lr)
        
        # Check gradient trend
        grads = result['grad_norms']
        if len(grads) >= 2:
            trend = "increasing" if grads[-1] > grads[0] else "decreasing"
            print(f"Gradient trend: {trend}")
            print(f"Final gradient norm: {grads[-1]:.2f}")
            
            if grads[-1] < grads[0] and grads[-1] < 1000:
                print("✅ Good convergence with this learning rate")
            else:
                print("❌ Still diverging or too large")
        print()
    
    return learning_rates

def test_penalty_parameter_effect():
    """Test if penalty parameters are causing divergence"""
    print("🔧 TESTING PENALTY PARAMETER EFFECTS")
    print("=" * 50)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5, 
        num_constraints=3, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    # Create algorithm
    algorithm = F2CSAAlgorithm1Final(problem)
    
    # Test with different alpha values
    alpha_values = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    x0 = torch.randn(5, dtype=torch.float64)
    
    print(f"Test point: {x0}")
    print()
    
    for alpha in alpha_values:
        print(f"Testing α = {alpha}")
        print("-" * 20)
        
        # Calculate penalty parameters
        alpha1 = 1.0 / alpha
        alpha2 = 1.0 / (alpha**2)
        
        print(f"α₁ = {alpha1:.1f}")
        print(f"α₂ = {alpha2:.1f}")
        
        # Get hypergradient
        hypergradient = algorithm.oracle_sample(x0, alpha, 10)
        hypergradient_norm = torch.norm(hypergradient).item()
        
        print(f"Hypergradient norm: {hypergradient_norm:.2f}")
        
        # Check if penalty parameters are too large
        if alpha2 > 1000000:
            print("❌ Penalty parameters too large - causing divergence")
        elif alpha2 > 100000:
            print("⚠️  Penalty parameters very large")
        else:
            print("✅ Penalty parameters reasonable")
        print()
    
    return alpha_values

def fix_divergence_with_smaller_alpha():
    """Try to fix divergence with smaller alpha"""
    print("🔧 FIXING DIVERGENCE WITH SMALLER ALPHA")
    print("=" * 50)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5, 
        num_constraints=3, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    # Create algorithm
    algorithm = F2CSAAlgorithm1Final(problem)
    
    # Try smaller alpha values
    alpha_values = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    x0 = torch.randn(5, dtype=torch.float64)
    
    print(f"Test point: {x0}")
    print()
    
    best_alpha = None
    best_grad_norm = float('inf')
    
    for alpha in alpha_values:
        print(f"Testing α = {alpha}")
        print("-" * 20)
        
        # Calculate penalty parameters
        alpha1 = 1.0 / alpha
        alpha2 = 1.0 / (alpha**2)
        
        print(f"α₁ = {alpha1:.1f}")
        print(f"α₂ = {alpha2:.1f}")
        
        # Get hypergradient
        hypergradient = algorithm.oracle_sample(x0, alpha, 10)
        hypergradient_norm = torch.norm(hypergradient).item()
        
        print(f"Hypergradient norm: {hypergradient_norm:.2f}")
        
        # Check if this is better
        if hypergradient_norm < best_grad_norm and hypergradient_norm < 1000:
            best_grad_norm = hypergradient_norm
            best_alpha = alpha
            print("✅ Best alpha so far")
        else:
            print("❌ Not better")
        print()
    
    print(f"🎯 BEST ALPHA: {best_alpha} (gradient norm: {best_grad_norm:.2f})")
    
    return best_alpha, best_grad_norm

if __name__ == "__main__":
    # Diagnose the problem
    print("STEP 1: DIAGNOSIS")
    diagnose_gradient_divergence()
    
    print("\nSTEP 2: LEARNING RATE TEST")
    test_learning_rate_effect()
    
    print("\nSTEP 3: PENALTY PARAMETER TEST")
    test_penalty_parameter_effect()
    
    print("\nSTEP 4: FIX WITH SMALLER ALPHA")
    best_alpha, best_grad_norm = fix_divergence_with_smaller_alpha()
    
    print(f"\n🎯 CONCLUSION:")
    if best_alpha is not None:
        print(f"✅ Found working alpha: {best_alpha}")
        print(f"✅ Gradient norm: {best_grad_norm:.2f}")
        print("✅ Divergence issue can be fixed with smaller alpha")
    else:
        print("❌ Could not find working alpha")
        print("❌ Divergence issue persists")
