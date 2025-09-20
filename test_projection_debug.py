#!/usr/bin/env python3
"""
Debug projection methods for SSIGD - detailed logging to see why X doesn't change
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from ssigd_correct_final import CorrectSSIGD

def test_projection_debug():
    """Debug projection methods with detailed logging"""
    
    print("🔬 SSIGD Projection Method Debug")
    print("=" * 60)
    
    # Set same seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Test parameters
    dim = 50
    T = 100
    beta = 0.001
    mu_F = 0.1
    
    print(f"Test Parameters:")
    print(f"  Dimension: {dim}")
    print(f"  Iterations: {T}")
    print(f"  Beta (step size): {beta}")
    print(f"  mu_F: {mu_F}")
    print(f"  Seed: {seed}")
    print()
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=dim, device='cpu')
    x0 = torch.randn(dim, dtype=torch.float64) * 0.1
    
    print(f"Problem Info:")
    print(f"  Upper level strong convexity: λ_min={torch.linalg.eigvals(problem.Q_upper).real.min():.3f}")
    print(f"  Lower level strong convexity: λ_min={torch.linalg.eigvals(problem.Q_lower).real.min():.3f}")
    print()
    
    # Initial loss
    y0, _, _ = problem.solve_lower_level(x0, solver='gurobi')
    initial_loss = problem.upper_objective(x0, y0).item()
    print(f"Initial UL Loss: {initial_loss:.6f}")
    print()
    
    # Test PGD projection
    print("=" * 60)
    print("TESTING PGD PROJECTION (UNCONSTRAINED)")
    print("=" * 60)
    torch.manual_seed(seed)  # Reset seed
    np.random.seed(seed)
    problem_pgd = StronglyConvexBilevelProblem(dim=dim, device='cpu')
    x0_pgd = torch.randn(dim, dtype=torch.float64) * 0.1
    
    try:
        ssigd_pgd = CorrectSSIGD(problem_pgd)
        result_pgd = ssigd_pgd.solve(T=T, beta=beta, x0=x0_pgd, diminishing=True, mu_F=mu_F, projection_method='pgd')
        print(f"\n✓ PGD Results: Final loss = {result_pgd['final_loss']:.6f}, Final grad = {result_pgd['final_grad_norm']:.6f}")
    except Exception as e:
        print(f"✗ PGD failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Clipping projection
    print("\n" + "=" * 60)
    print("TESTING CLIPPING PROJECTION (UNCONSTRAINED)")
    print("=" * 60)
    torch.manual_seed(seed)  # Reset seed
    np.random.seed(seed)
    problem_clip = StronglyConvexBilevelProblem(dim=dim, device='cpu')
    x0_clip = torch.randn(dim, dtype=torch.float64) * 0.1
    
    try:
        ssigd_clip = CorrectSSIGD(problem_clip)
        result_clip = ssigd_clip.solve(T=T, beta=beta, x0=x0_clip, diminishing=True, mu_F=mu_F, projection_method='clip')
        print(f"\n✓ Clipping Results: Final loss = {result_clip['final_loss']:.6f}, Final grad = {result_clip['final_grad_norm']:.6f}")
    except Exception as e:
        print(f"✗ Clipping failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print("Both methods should produce identical results since both projections")
    print("are identity functions in the unconstrained case (X_bounds=None)")
    print("The step size formula is: min(1/(μ_F * (r+1)), beta)")
    print("=" * 60)

if __name__ == "__main__":
    test_projection_debug()
