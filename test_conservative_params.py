#!/usr/bin/env python3
"""
Test F2CSA with very conservative parameters to see if we can get convergence
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm import F2CSAAlgorithm
import warnings
warnings.filterwarnings('ignore')

def test_conservative_params():
    """Test F2CSA with very conservative parameters"""
    print("=" * 80)
    print("TESTING F2CSA WITH VERY CONSERVATIVE PARAMETERS")
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
    
    # Test with very conservative parameters
    test_configs = [
        {'alpha': 0.01, 'eta': 0.001, 'D': 0.01, 'Ng': 16, 'name': 'Very Conservative'},
        {'alpha': 0.02, 'eta': 0.002, 'D': 0.02, 'Ng': 8, 'name': 'Conservative'},
        {'alpha': 0.05, 'eta': 0.005, 'D': 0.05, 'Ng': 4, 'name': 'Moderate'},
    ]
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing {config['name']}: α={config['alpha']}, η={config['eta']}, D={config['D']}, Ng={config['Ng']}")
        print(f"{'='*60}")
        
        # Create F2CSA with test parameters
        f2csa = F2CSAAlgorithm(
            problem=problem,
            device='cpu',
            seed=42,
            alpha_override=config['alpha'],
            eta_override=config['eta'],
            D_override=config['D'],
            Ng_override=config['Ng']
        )
        
        try:
            # Run optimization for 200 iterations
            result = f2csa.optimize(
                max_iterations=200,
                early_stopping_patience=100,
                target_gap=0.05,  # Lower target
                verbose=False
            )
            
            final_gap = result['final_gap']
            final_ema_gap = result['final_ema_gap']
            iterations = result['total_iterations']
            
            print(f"  Final gap: {final_gap:.6f}")
            print(f"  Final EMA gap: {final_ema_gap:.6f}")
            print(f"  Iterations: {iterations}")
            
            if final_gap < 0.1:
                print(f"  ✅ SUCCESS: Final gap < 0.1 achieved!")
                return config
            else:
                print(f"  ❌ FAILURE: Final gap >= 0.1")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
    
    print(f"\n{'='*80}")
    print(f"NO CONSERVATIVE CONFIGURATION ACHIEVED FINAL GAP < 0.1")
    print(f"{'='*80}")
    return None

if __name__ == "__main__":
    best_config = test_conservative_params()
    if best_config:
        print(f"\n🎉 Found working configuration: {best_config}")
    else:
        print(f"\n⚠️ Even conservative parameters failed - need to fix algorithm")
