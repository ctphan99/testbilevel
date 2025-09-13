#!/usr/bin/env python3
"""
Test F2CSA with the fixed conservative default parameters
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm import F2CSAAlgorithm
import warnings
warnings.filterwarnings('ignore')

def test_fixed_f2csa():
    """Test F2CSA with fixed conservative default parameters"""
    print("=" * 80)
    print("TESTING F2CSA WITH FIXED CONSERVATIVE DEFAULT PARAMETERS")
    print("=" * 80)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5,
        num_constraints=3,
        noise_std=0.001,
        device='cpu',
        seed=42,
        strong_convex=True
    )
    
    # Test with default parameters (should be conservative now)
    f2csa = F2CSAAlgorithm(
        problem=problem,
        device='cpu',
        seed=42
        # No overrides - use default conservative parameters
    )
    
    print(f"Testing with default parameters (should be conservative)")
    
    try:
        # Run optimization for 300 iterations
        result = f2csa.optimize(
            max_iterations=300,
            early_stopping_patience=150,
            target_gap=0.1,
            verbose=True
        )
        
        final_gap = result['final_gap']
        final_ema_gap = result['final_ema_gap']
        iterations = result['total_iterations']
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Final gap: {final_gap:.6f}")
        print(f"Final EMA gap: {final_ema_gap:.6f}")
        print(f"Total iterations: {iterations}")
        
        if final_gap < 0.1:
            print(f"✅ SUCCESS: Final gap < 0.1 achieved!")
            return True
        else:
            print(f"❌ FAILURE: Final gap >= 0.1")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_f2csa()
    if success:
        print(f"\n🎉 F2CSA with conservative defaults works!")
    else:
        print(f"\n⚠️ F2CSA still needs more work")