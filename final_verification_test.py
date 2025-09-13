#!/usr/bin/env python3
"""
Final verification test of F2CSA algorithm
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm import F2CSAAlgorithm
import warnings
warnings.filterwarnings('ignore')

def final_verification_test():
    """Final verification that F2CSA achieves gap < 0.1 consistently"""
    print("=" * 80)
    print("FINAL VERIFICATION: F2CSA ACHIEVES GAP < 0.1")
    print("=" * 80)
    
    # Test multiple random seeds to verify consistency
    seeds = [42, 123, 456, 789, 999]
    success_count = 0
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Testing with seed: {seed}")
        print(f"{'='*60}")
        
        # Create problem with different seed
        problem = StronglyConvexBilevelProblem(
            dim=5,
            num_constraints=3,
            noise_std=0.001,
            device='cpu',
            seed=seed,
            strong_convex=True
        )
        
        # Use default conservative parameters
        f2csa = F2CSAAlgorithm(
            problem=problem,
            device='cpu',
            seed=seed
        )
        
        try:
            # Run optimization
            result = f2csa.optimize(
                max_iterations=300,
                early_stopping_patience=150,
                target_gap=0.1,
                verbose=False
            )
            
            final_gap = result['final_gap']
            final_ema_gap = result['final_ema_gap']
            iterations = result['total_iterations']
            
            print(f"  Final gap: {final_gap:.6f}")
            print(f"  Final EMA gap: {final_ema_gap:.6f}")
            print(f"  Iterations: {iterations}")
            
            if final_gap < 0.1:
                print(f"  ✅ SUCCESS: Gap < 0.1 achieved!")
                success_count += 1
            else:
                print(f"  ❌ FAILURE: Gap >= 0.1")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
    
    print(f"\n{'='*80}")
    print(f"FINAL VERIFICATION RESULTS")
    print(f"{'='*80}")
    print(f"Success rate: {success_count}/{len(seeds)} ({success_count/len(seeds)*100:.1f}%)")
    
    if success_count == len(seeds):
        print(f"🎉 ALL TESTS PASSED! F2CSA consistently achieves gap < 0.1!")
        return True
    else:
        print(f"⚠️ Some tests failed. F2CSA needs more work.")
        return False

if __name__ == "__main__":
    success = final_verification_test()
    if success:
        print(f"\n🎉 F2CSA algorithm is ready for production use!")
    else:
        print(f"\n⚠️ F2CSA algorithm needs more fixes")
