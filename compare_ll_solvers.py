#!/usr/bin/env python3
"""
Compare CVXPY vs SGD lower-level solvers for hypergradient accuracy
"""
import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final

def fd_grad(problem, x, eps=1e-6):
    """Compute finite difference gradient for comparison"""
    x = x.detach().clone()
    dim = x.numel()
    grad = torch.zeros_like(x)
    for i in range(dim):
        e = torch.zeros_like(x)
        e[i] = eps
        x_plus = x + e
        x_minus = x - e
        y_plus, _ = problem.solve_lower_level(x_plus)
        y_minus, _ = problem.solve_lower_level(x_minus)
        f_plus = problem.upper_objective(x_plus, y_plus)
        f_minus = problem.upper_objective(x_minus, y_minus)
        grad[i] = (f_plus - f_minus) / (2.0 * eps)
    return grad

def compare_solvers():
    """Compare CVXPY vs SGD solvers for hypergradient accuracy"""
    print("Comparing CVXPY vs SGD Lower-Level Solvers")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    torch.set_default_dtype(torch.float64)
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Create problem instance
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=2e-3)
    algo = F2CSAAlgorithm1Final(problem)
    
    # Test point and parameters
    x = torch.randn(5, dtype=torch.float64)
    alpha = 0.05
    Ng = 32
    
    print(f"Problem: dim={problem.dim}, constraints={problem.num_constraints}, noise_std={problem.noise_std}")
    print(f"Test point x: {x}")
    print(f"Parameters: α={alpha}, N_g={Ng}")
    print()
    
    # Compute finite difference gradient (ground truth)
    print("Computing finite difference gradient...")
    fd = fd_grad(problem, x)
    print(f"FD gradient: {fd}")
    print(f"FD gradient norm: {torch.norm(fd).item():.6e}")
    print()
    
    # Test CVXPY solver
    print("Testing CVXPY solver...")
    try:
        hg_cvx, y_cvx, lambda_cvx = algo.oracle_sample(x, alpha, Ng, force_sgd=False)
        rel_err_cvx = torch.norm(hg_cvx - fd) / (torch.norm(fd) + 1e-12)
        print(f"CVXPY hypergradient: {hg_cvx}")
        print(f"CVXPY relative error: {rel_err_cvx.item():.6e}")
        print(f"CVXPY hypergradient norm: {torch.norm(hg_cvx).item():.6e}")
        print(f"CVXPY lower-level solution: {y_cvx}")
        print(f"CVXPY multipliers: {lambda_cvx}")
        print()
    except Exception as e:
        print(f"CVXPY solver failed: {e}")
        hg_cvx = None
        rel_err_cvx = float('inf')
        print()
    
    # Test SGD solver
    print("Testing SGD solver...")
    try:
        hg_sgd, y_sgd, lambda_sgd = algo.oracle_sample(x, alpha, Ng, force_sgd=True)
        rel_err_sgd = torch.norm(hg_sgd - fd) / (torch.norm(fd) + 1e-12)
        print(f"SGD hypergradient: {hg_sgd}")
        print(f"SGD relative error: {rel_err_sgd.item():.6e}")
        print(f"SGD hypergradient norm: {torch.norm(hg_sgd).item():.6e}")
        print(f"SGD lower-level solution: {y_sgd}")
        print(f"SGD multipliers: {lambda_sgd}")
        print()
    except Exception as e:
        print(f"SGD solver failed: {e}")
        hg_sgd = None
        rel_err_sgd = float('inf')
        print()
    
    # Summary comparison
    print("SUMMARY COMPARISON")
    print("=" * 30)
    print(f"Finite difference gradient norm: {torch.norm(fd).item():.6e}")
    if hg_cvx is not None:
        print(f"CVXPY relative error: {rel_err_cvx.item():.6e}")
        print(f"CVXPY hypergradient norm: {torch.norm(hg_cvx).item():.6e}")
    else:
        print("CVXPY solver: FAILED")
    
    if hg_sgd is not None:
        print(f"SGD relative error: {rel_err_sgd.item():.6e}")
        print(f"SGD hypergradient norm: {torch.norm(hg_sgd).item():.6e}")
    else:
        print("SGD solver: FAILED")
    
    # Determine which solver is more accurate
    if hg_cvx is not None and hg_sgd is not None:
        if rel_err_cvx < rel_err_sgd:
            print(f"\nCVXPY is more accurate (error ratio: {rel_err_sgd/rel_err_cvx:.2f}x)")
        else:
            print(f"\nSGD is more accurate (error ratio: {rel_err_cvx/rel_err_sgd:.2f}x)")
    
    return {
        'fd_grad': fd,
        'cvx_grad': hg_cvx,
        'sgd_grad': hg_sgd,
        'cvx_error': rel_err_cvx,
        'sgd_error': rel_err_sgd
    }

if __name__ == "__main__":
    results = compare_solvers()
