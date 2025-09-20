#!/usr/bin/env python3
"""
Example demonstrating Gurobi integration with bilevel optimization algorithms
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from problem import StronglyConvexBilevelProblem

def run_algorithm_with_gurobi():
    """Example of running an algorithm with Gurobi solver"""
    
    # Create problem
    dim = 10
    problem = StronglyConvexBilevelProblem(dim=dim, device='cpu')
    
    print(f"Running bilevel optimization with Gurobi solver")
    print(f"Problem dimension: {dim}")
    print(f"Constraints: {problem.num_constraints}")
    
    # Initialize
    x = torch.randn(dim, dtype=torch.float64) * 0.1
    learning_rate = 0.01
    max_iterations = 100
    
    # Track progress
    upper_losses = []
    solve_times = []
    
    print(f"\nStarting optimization...")
    print(f"Iteration | Upper Loss | Solve Time | Status")
    print("-" * 50)
    
    for iteration in range(max_iterations):
        # Solve lower-level problem with Gurobi
        import time
        start_time = time.time()
        y_opt, lambda_opt, info = problem.solve_lower_level(x, solver='gurobi')
        solve_time = time.time() - start_time
        
        # Compute upper-level objective
        upper_loss = problem.upper_objective(x, y_opt).item()
        upper_losses.append(upper_loss)
        solve_times.append(solve_time)
        
        # Simple gradient step (example)
        # In real algorithms, you'd use proper gradients
        x = x - learning_rate * torch.randn_like(x) * 0.01
        
        # Print progress
        if iteration % 10 == 0 or iteration < 5:
            status = info.get('status', 'unknown')
            print(f"{iteration:9d} | {upper_loss:10.6f} | {solve_time:10.4f}s | {status}")
    
    print(f"\nOptimization completed!")
    print(f"Average solve time: {np.mean(solve_times):.4f}s")
    print(f"Final upper loss: {upper_losses[-1]:.6f}")
    
    return upper_losses, solve_times

def compare_solvers():
    """Compare different solvers on the same problem"""
    
    problem = StronglyConvexBilevelProblem(dim=5, device='cpu')
    x = torch.randn(5, dtype=torch.float64)
    
    solvers = ['gurobi', 'cvxpy', 'pgd']
    results = {}
    
    print(f"\nComparing solvers on problem dimension {problem.dim}")
    print(f"Test point: {x.numpy()}")
    print("-" * 60)
    
    for solver in solvers:
        try:
            import time
            start_time = time.time()
            y_opt, lambda_opt, info = problem.solve_lower_level(x, solver=solver)
            solve_time = time.time() - start_time
            
            obj_value = problem.lower_objective(x, y_opt).item()
            
            results[solver] = {
                'y_opt': y_opt,
                'obj_value': obj_value,
                'solve_time': solve_time,
                'info': info
            }
            
            print(f"{solver.upper():8s} | {solve_time:8.4f}s | {obj_value:12.6f} | {info.get('status', 'N/A')}")
            
        except Exception as e:
            print(f"{solver.upper():8s} | ERROR: {e}")
    
    return results

def plot_performance():
    """Plot performance comparison"""
    
    # Test different dimensions
    dimensions = [5, 10, 20, 50]
    gurobi_times = []
    cvxpy_times = []
    
    for dim in dimensions:
        problem = StronglyConvexBilevelProblem(dim=dim, device='cpu')
        x = torch.randn(dim, dtype=torch.float64)
        
        # Time Gurobi
        import time
        start = time.time()
        problem.solve_lower_level(x, solver='gurobi')
        gurobi_times.append(time.time() - start)
        
        # Time CVXPY
        start = time.time()
        problem.solve_lower_level(x, solver='cvxpy')
        cvxpy_times.append(time.time() - start)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogy(dimensions, gurobi_times, 'o-', label='Gurobi', linewidth=2)
    plt.semilogy(dimensions, cvxpy_times, 's-', label='CVXPY', linewidth=2)
    plt.xlabel('Problem Dimension')
    plt.ylabel('Solve Time (seconds)')
    plt.title('Solver Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gurobi_performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nPerformance comparison saved to 'gurobi_performance_comparison.png'")

if __name__ == "__main__":
    print("=== Gurobi Integration Example ===\n")
    
    # Run algorithm example
    upper_losses, solve_times = run_algorithm_with_gurobi()
    
    # Compare solvers
    results = compare_solvers()
    
    # Plot performance
    plot_performance()
    
    print("\n=== Example completed ===")
    print("Check 'gurobi_performance_comparison.png' for performance plots")
