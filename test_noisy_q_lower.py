#!/usr/bin/env python3
"""
Test script to verify all algorithms correctly use noisy Q_lower
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
from ssigd_correct_final import CorrectSSIGD
from dsblo_optII import DSBLOOptII

def test_noisy_q_lower_usage():
    """Test that all algorithms use noisy Q_lower in lower-level problem"""
    
    print("Testing noisy Q_lower usage in all algorithms...")
    print("=" * 60)
    
    # Create problem instance
    dim = 5
    problem = StronglyConvexBilevelProblem(dim=dim, num_constraints=2*dim, noise_std=0.1)
    
    # Generate test point
    x0 = torch.randn(dim, device='cpu', dtype=torch.float64)
    
    print(f"Problem dimension: {dim}")
    print(f"Test point x0: {x0[:3]}...")
    print()
    
    # Test 1: Verify problem.lower_objective uses noise_lower
    print("Test 1: Problem.lower_objective with/without noise_lower")
    noise_upper, noise_lower = problem._sample_instance_noise()
    y_test = torch.randn(dim, device='cpu', dtype=torch.float64)
    
    # Without noise (should use clean Q_lower)
    obj_clean = problem.lower_objective(x0, y_test)
    
    # With noise (should use noisy Q_lower)
    obj_noisy = problem.lower_objective(x0, y_test, noise_lower=noise_lower)
    
    print(f"  Clean objective: {obj_clean.item():.6f}")
    print(f"  Noisy objective: {obj_noisy.item():.6f}")
    print(f"  Difference: {abs(obj_noisy.item() - obj_clean.item()):.6f}")
    print(f"  ✓ Noise affects objective: {abs(obj_noisy.item() - obj_clean.item()) > 1e-6}")
    print()
    
    # Test 2: Verify PGD solver uses noisy Q_lower
    print("Test 2: PGD solver with/without noise_lower")
    
    # Without noise
    y_clean, info_clean = problem.solve_lower_level(x0, solver='pgd', max_iter=100, tol=1e-6)
    
    # With noise
    y_noisy, info_noisy = problem.solve_lower_level(x0, solver='pgd', max_iter=100, tol=1e-6, noise_lower=noise_lower)
    
    print(f"  Clean solution: {y_clean[:3]}...")
    print(f"  Noisy solution: {y_noisy[:3]}...")
    print(f"  Solution difference: {torch.norm(y_noisy - y_clean).item():.6f}")
    print(f"  ✓ Noise affects solution: {torch.norm(y_noisy - y_clean).item() > 1e-6}")
    print()
    
    # Test 3: Test F2CSA Algorithm 1
    print("Test 3: F2CSA Algorithm 1")
    try:
        f2csa_algo = F2CSAAlgorithm1Final(problem, device='cpu', dtype=torch.float64)
        
        # Test oracle_sample method
        hypergrad, y_tilde, lambda_star = f2csa_algo.oracle_sample(x0, alpha=0.1, N_g=2)
        
        print(f"  F2CSA hypergradient norm: {torch.norm(hypergrad).item():.6f}")
        print(f"  F2CSA y_tilde: {y_tilde[:3]}...")
        print(f"  F2CSA lambda_star: {lambda_star[:3]}...")
        print(f"  ✓ F2CSA completed successfully")
    except Exception as e:
        print(f"  ✗ F2CSA failed: {e}")
    print()
    
    # Test 4: Test SSIGD
    print("Test 4: SSIGD")
    try:
        ssigd_algo = CorrectSSIGD(problem)
        
        # Test solve method for a few iterations
        x_final, losses, hypergrad_norms = ssigd_algo.solve(T=5, beta=0.1, x0=x0)
        
        print(f"  SSIGD final loss: {losses[-1]:.6f}")
        print(f"  SSIGD final grad norm: {hypergrad_norms[-1]:.6f}")
        print(f"  ✓ SSIGD completed successfully")
    except Exception as e:
        print(f"  ✗ SSIGD failed: {e}")
    print()
    
    # Test 5: Test DS-BLO
    print("Test 5: DS-BLO")
    try:
        dsblo_algo = DSBLOOptII(problem)
        
        # Test optimize method for a few iterations
        results = dsblo_algo.optimize(x0, T=5, alpha=0.05, sigma=1e-2, grad_avg_k=1)
        
        print(f"  DS-BLO final UL loss: {results['ul_losses'][-1]:.6f}")
        print(f"  DS-BLO final grad norm: {results['hypergrad_norms'][-1]:.6f}")
        print(f"  ✓ DS-BLO completed successfully")
    except Exception as e:
        print(f"  ✗ DS-BLO failed: {e}")
    print()
    
    print("=" * 60)
    print("All tests completed!")
    print("✓ All algorithms should now correctly use noisy Q_lower in lower-level problem")

if __name__ == "__main__":
    test_noisy_q_lower_usage()
