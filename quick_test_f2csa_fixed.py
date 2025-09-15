#!/usr/bin/env python3
"""
Quick F2CSA Test Script - FIXED VERSION
Run a quick test to verify the algorithm works before full parameter tuning
"""

import torch
import numpy as np
import time
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
import warnings
warnings.filterwarnings('ignore')

def quick_test():
    """Run a quick test of F2CSA Algorithm 1 and 2"""
    print("🧪 QUICK F2CSA TEST - FIXED VERSION")
    print("=" * 50)
    
    # Check GPU
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        print("Using CPU")
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5, 
        num_constraints=3, 
        noise_std=0.1, 
        strong_convex=True,
        device=device
    )
    
    print(f"Problem created: dim=5, constraints=3, noise_std=0.1")
    
    # Test Algorithm 1
    print(f"\n🔬 Testing Algorithm 1...")
    algorithm1 = F2CSAAlgorithm1Final(problem)
    test_point = torch.randn(5, device=device, dtype=problem.dtype) * 0.1
    
    try:
        # Test hypergradient computation
        grad = algorithm1.oracle_sample(test_point, alpha=0.1, N_g=10)
        grad_norm = torch.norm(grad).item()
        print(f"✅ Algorithm 1: Hypergradient norm = {grad_norm:.4f}")
        
        # Test gap computation
        gap = problem.compute_gap(test_point)
        print(f"✅ Algorithm 1: Gap = {gap:.6f}")
        
    except Exception as e:
        print(f"❌ Algorithm 1 failed: {e}")
        return False
    
    # Test Algorithm 2
    print(f"\n🚀 Testing Algorithm 2...")
    try:
        from f2csa_algorithm2_final import F2CSAAlgorithm2Final
        algorithm2 = F2CSAAlgorithm2Final(problem, device, problem.dtype)
        
        # Run short optimization
        result = algorithm2.optimize(test_point, T=100, D=0.5, eta=1e-4, 
                                   delta=0.1**3, alpha=0.1, N_g=10)
        
        print(f"✅ Algorithm 2: Final gap = {result['final_gap']:.6f}")
        print(f"✅ Algorithm 2: Convergence = {result['convergence_status']}")
        print(f"✅ Algorithm 2: Iterations = {result['iterations']}")
        
    except Exception as e:
        print(f"❌ Algorithm 2 failed: {e}")
        return False
    
    print(f"\n🎉 QUICK TEST PASSED!")
    print("Ready for full parameter tuning")
    return True

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n✅ All tests passed! You can now run the full parameter tuning:")
        print("   python run_f2csa_tuning.py")
    else:
        print("\n❌ Tests failed! Please check the error messages above.")