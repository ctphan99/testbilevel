#!/usr/bin/env python3
"""
Test SSIGD parameter tuning locally (smaller grid)
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from ssigd_correct_final import CorrectSSIGD

def test_ssigd_batch_local():
    """Test SSIGD with a smaller parameter grid locally"""
    
    print("🔬 SSIGD Local Batch Test")
    print("=" * 50)
    
    # Set random seed
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Test parameters (smaller grid for local testing)
    dim = 50  # Smaller dimension for faster testing
    T = 200   # Fewer iterations for faster testing
    beta_values = [0.001, 0.005, 0.01]
    mu_F_values = [0.1, 0.5, 1.0]
    
    print(f"Test Parameters:")
    print(f"  Dimension: {dim}")
    print(f"  Iterations: {T}")
    print(f"  Beta values: {beta_values}")
    print(f"  mu_F values: {mu_F_values}")
    print(f"  Seed: {seed}")
    print()
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=dim, device='cpu')
    x0 = torch.randn(dim, dtype=torch.float64) * 0.1
    
    # Initial loss
    y0, _, _ = problem.solve_lower_level(x0, solver='gurobi')
    initial_loss = problem.upper_objective(x0, y0).item()
    print(f"Initial UL Loss: {initial_loss:.6f}")
    print()
    
    results = []
    
    print("Testing parameter combinations...")
    print("-" * 50)
    
    for beta in beta_values:
        for mu_F in mu_F_values:
            # Reset seed for each run
            torch.manual_seed(seed)
            np.random.seed(seed)
            problem_test = StronglyConvexBilevelProblem(dim=dim, device='cpu')
            x0_test = torch.randn(dim, dtype=torch.float64) * 0.1
            
            try:
                ssigd = CorrectSSIGD(problem_test)
                result = ssigd.solve(T=T, beta=beta, x0=x0_test, diminishing=True, mu_F=mu_F)
                
                final_loss = result['final_loss']
                final_grad = result['final_grad_norm']
                improvement = initial_loss - final_loss
                reduction_pct = (improvement / initial_loss * 100) if initial_loss != 0 else 0
                
                results.append({
                    'beta': beta,
                    'mu_F': mu_F,
                    'final_loss': final_loss,
                    'final_grad': final_grad,
                    'improvement': improvement,
                    'reduction_pct': reduction_pct
                })
                
                print(f"β={beta:.4f}, μ_F={mu_F:.2f}: Loss={final_loss:.6f}, Improvement={improvement:.6f} ({reduction_pct:.2f}%)")
                
            except Exception as e:
                print(f"β={beta:.4f}, μ_F={mu_F:.2f}: FAILED - {e}")
    
    # Sort by final loss
    results.sort(key=lambda x: x['final_loss'])
    
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    print("Top 3 Parameter Combinations:")
    print(f"{'Rank':<5} {'Beta':<8} {'mu_F':<8} {'Final Loss':<12} {'Improvement':<12} {'Reduction %':<12}")
    print("-" * 60)
    
    for i, result in enumerate(results[:3]):
        print(f"{i+1:<5} {result['beta']:<8.4f} {result['mu_F']:<8.2f} {result['final_loss']:<12.6f} {result['improvement']:<12.6f} {result['reduction_pct']:<12.2f}")
    
    if results:
        best = results[0]
        print(f"\n🏆 Best Parameters:")
        print(f"  Beta: {best['beta']:.6f}")
        print(f"  mu_F: {best['mu_F']:.6f}")
        print(f"  Final Loss: {best['final_loss']:.6f}")
        print(f"  Improvement: {best['improvement']:.6f}")
        print(f"  Reduction: {best['reduction_pct']:.2f}%")
    
    print("=" * 50)

if __name__ == "__main__":
    test_ssigd_batch_local()