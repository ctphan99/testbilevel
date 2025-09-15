#!/usr/bin/env python3
"""
Debug Algorithm 2 specifically - why is it diverging even with original parameters?
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm2 import F2CSAAlgorithm2

def debug_algorithm2():
    """Debug Algorithm 2 convergence issues"""
    print("🔍 DEBUGGING ALGORITHM 2 CONVERGENCE")
    print("=" * 60)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5, 
        num_constraints=3, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    # Create Algorithm 2
    algorithm2 = F2CSAAlgorithm2(problem)
    
    # Test with different parameters
    alpha = 0.1
    x0 = torch.randn(5, dtype=torch.float64)
    
    print(f"Using α = {alpha}")
    print(f"Original F2CSA: α₁ = α⁻² = {1/(alpha**2):.1f}")
    print(f"Original F2CSA: α₂ = α⁻⁴ = {1/(alpha**4):.1f}")
    print(f"δ = α³ = {alpha**3:.3f}")
    print(f"Test point: {x0}")
    print()
    
    # Test with different parameter combinations
    test_configs = [
        {"T": 50, "D": 1.0, "eta": 0.001, "delta": alpha**3, "alpha": alpha, "name": "Standard"},
        {"T": 50, "D": 0.1, "eta": 0.001, "delta": alpha**3, "alpha": alpha, "name": "Small D"},
        {"T": 50, "D": 1.0, "eta": 0.0001, "delta": alpha**3, "alpha": alpha, "name": "Small eta"},
        {"T": 50, "D": 1.0, "eta": 0.01, "delta": alpha**3, "alpha": alpha, "name": "Large eta"},
        {"T": 50, "D": 1.0, "eta": 0.001, "delta": alpha**2, "alpha": alpha, "name": "Larger delta"},
    ]
    
    results = []
    
    for i, config in enumerate(test_configs):
        print(f"{i+1}️⃣ TESTING: {config['name']}")
        print(f"   T={config['T']}, D={config['D']}, eta={config['eta']}, delta={config['delta']:.3f}")
        
        try:
            result = algorithm2.optimize(
                x0, 
                T=config['T'], 
                D=config['D'], 
                eta=config['eta'], 
                delta=config['delta'], 
                alpha=config['alpha']
            )
            
            grads = result['grad_norms']
            final_grad = result['final_g_norm']
            
            # Check last 10 iterations for stability
            last_10 = grads[-10:] if len(grads) > 10 else grads
            if len(last_10) > 0:
                std_last_10 = np.std(last_10)
                range_last_10 = np.max(last_10) - np.min(last_10)
                
                print(f"   Final grad: {final_grad:.2f}")
                print(f"   Last 10 std: {std_last_10:.2f}, range: {range_last_10:.2f}")
                
                if std_last_10 < 1.0 and range_last_10 < 3.0:
                    print("   ✅ CONVERGED")
                    converged = True
                else:
                    print("   ❌ DIVERGED")
                    converged = False
            else:
                converged = False
            
            results.append({
                'config': config['name'],
                'converged': converged,
                'final_grad': final_grad,
                'std_last_10': std_last_10 if len(last_10) > 0 else float('inf'),
                'range_last_10': range_last_10 if len(last_10) > 0 else float('inf')
            })
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append({
                'config': config['name'],
                'converged': False,
                'error': str(e)
            })
        
        print()
    
    # Analyze results
    print("📊 ALGORITHM 2 DEBUG RESULTS")
    print("=" * 60)
    
    converged_configs = [r for r in results if r.get('converged', False)]
    diverged_configs = [r for r in results if not r.get('converged', False)]
    
    print(f"Converged: {len(converged_configs)}/{len(results)}")
    print(f"Diverged: {len(diverged_configs)}/{len(results)}")
    print()
    
    if converged_configs:
        print("✅ CONVERGED CONFIGURATIONS:")
        for r in converged_configs:
            print(f"  - {r['config']}: final_grad={r['final_grad']:.2f}, std={r['std_last_10']:.2f}")
        print()
    
    if diverged_configs:
        print("❌ DIVERGED CONFIGURATIONS:")
        for r in diverged_configs:
            if 'error' in r:
                print(f"  - {r['config']}: ERROR - {r['error']}")
            else:
                print(f"  - {r['config']}: final_grad={r['final_grad']:.2f}, std={r['std_last_10']:.2f}")
        print()
    
    # Check if any configuration worked
    if converged_configs:
        print("🎉 SUCCESS: Found working Algorithm 2 configuration!")
        best_config = min(converged_configs, key=lambda x: x['std_last_10'])
        print(f"Best configuration: {best_config['config']}")
        return True
    else:
        print("⚠️  ALL CONFIGURATIONS FAILED!")
        print("Algorithm 2 has fundamental implementation issues.")
        return False

if __name__ == "__main__":
    success = debug_algorithm2()
    
    if success:
        print("\n✅ ALGORITHM 2 DEBUGGING SUCCESSFUL!")
        print("Found working parameter configuration.")
    else:
        print("\n❌ ALGORITHM 2 NEEDS FUNDAMENTAL FIXES!")
        print("Need to investigate the core implementation.")
