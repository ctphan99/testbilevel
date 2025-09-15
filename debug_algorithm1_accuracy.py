#!/usr/bin/env python3
"""
Debug Algorithm 1 accuracy before averaging by n_g
Validate SGD solver and test alternative optimizers
"""

import torch
import numpy as np
import cvxpy as cp
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm import F2CSAAlgorithm
import matplotlib.pyplot as plt
from torch.optim import Adam, RMSprop, Adagrad, SGD

def debug_algorithm1_before_averaging(problem, f2csa, x_test, alpha=0.08, Ng_values=[1, 4, 16, 64]):
    """Debug Algorithm 1's raw output before averaging by n_g"""
    print("=" * 80)
    print("DEBUGGING ALGORITHM 1 BEFORE AVERAGING BY N_G")
    print("=" * 80)
    
    print(f"Testing x: {x_test.detach().numpy()}")
    print(f"Alpha: {alpha}")
    print(f"Ng values to test: {Ng_values}")
    
    # Test different Ng values
    results = {}
    
    for Ng in Ng_values:
        print(f"\n--- Testing Ng = {Ng} ---")
        
        # Get multiple samples to see variance
        samples = []
        for trial in range(10):
            sample = f2csa.oracle_sample(x_test, alpha, Ng)
            samples.append(sample.detach().numpy())
        
        samples = np.array(samples)
        mean_sample = np.mean(samples, axis=0)
        std_sample = np.std(samples, axis=0)
        
        print(f"  Mean gradient: {mean_sample}")
        print(f"  Std gradient: {std_sample}")
        print(f"  Gradient norm: {np.linalg.norm(mean_sample):.6f}")
        print(f"  Gradient std norm: {np.linalg.norm(std_sample):.6f}")
        
        results[Ng] = {
            'samples': samples,
            'mean': mean_sample,
            'std': std_sample,
            'norm': np.linalg.norm(mean_sample),
            'std_norm': np.linalg.norm(std_sample)
        }
    
    return results

def validate_sgd_solver(problem, x_test, alpha=0.08):
    """Validate SGD solver accuracy for lower-level problem"""
    print("\n" + "=" * 80)
    print("VALIDATING SGD SOLVER ACCURACY")
    print("=" * 80)
    
    # Solve with CVXPY (ground truth)
    print("Solving with CVXPY (ground truth)...")
    y_cvxpy, info_cvxpy = problem.solve_lower_level(x_test)
    dual_cvxpy = info_cvxpy.get('lambda', None)
    
    print(f"CVXPY solution y*: {y_cvxpy.detach().numpy()}")
    print(f"CVXPY dual variables: {dual_cvxpy.detach().numpy() if dual_cvxpy is not None else 'None'}")
    
    # Test SGD solver with different configurations
    sgd_configs = [
        {'lr': 0.01, 'steps': 50, 'momentum': 0.9, 'name': 'SGD_01_50_09'},
        {'lr': 0.001, 'steps': 100, 'momentum': 0.9, 'name': 'SGD_001_100_09'},
        {'lr': 0.1, 'steps': 20, 'momentum': 0.0, 'name': 'SGD_1_20_0'},
        {'lr': 0.005, 'steps': 200, 'momentum': 0.95, 'name': 'SGD_005_200_095'},
    ]
    
    sgd_results = {}
    
    for config in sgd_configs:
        print(f"\n--- Testing {config['name']} ---")
        
        # Create SGD solver
        y_sgd = torch.randn(problem.dim, dtype=torch.float64, requires_grad=True)
        optimizer = SGD([y_sgd], lr=config['lr'], momentum=config['momentum'])
        
        # SGD optimization
        for step in range(config['steps']):
            optimizer.zero_grad()
            
            # Lower-level objective
            g_val = 0.5 * y_sgd.T @ problem.Q_lower @ y_sgd + x_test.T @ problem.P @ y_sgd
            
            # Constraint penalty
            h_val = problem.A @ x_test + problem.B @ y_sgd - problem.b
            constraint_penalty = torch.sum(torch.clamp(h_val, min=0) ** 2)
            
            # Total objective
            total_obj = g_val + 1000 * constraint_penalty  # Large penalty for constraints
            
            total_obj.backward()
            optimizer.step()
        
        # Compute constraint violations
        h_val_sgd = problem.A @ x_test + problem.B @ y_sgd - problem.b
        
        print(f"  SGD solution y*: {y_sgd.detach().numpy()}")
        print(f"  Constraint violations: {h_val_sgd.detach().numpy()}")
        print(f"  Max constraint violation: {torch.max(h_val_sgd).item():.6f}")
        
        # Compare with CVXPY
        y_diff = torch.norm(y_sgd - y_cvxpy).item()
        print(f"  Difference from CVXPY: {y_diff:.6f}")
        
        sgd_results[config['name']] = {
            'y_sgd': y_sgd.detach().numpy(),
            'constraint_violations': h_val_sgd.detach().numpy(),
            'max_violation': torch.max(h_val_sgd).item(),
            'diff_from_cvxpy': y_diff
        }
    
    return sgd_results, y_cvxpy.detach().numpy()

def test_alternative_optimizers(problem, x_test, alpha=0.08):
    """Test alternative optimizers for lower-level problem"""
    print("\n" + "=" * 80)
    print("TESTING ALTERNATIVE OPTIMIZERS")
    print("=" * 80)
    
    # Ground truth
    y_cvxpy, info_cvxpy = problem.solve_lower_level(x_test)
    
    # Test different optimizers
    optimizers_config = [
        {'class': Adam, 'params': {'lr': 0.01}, 'name': 'Adam_001'},
        {'class': Adam, 'params': {'lr': 0.001}, 'name': 'Adam_0001'},
        {'class': RMSprop, 'params': {'lr': 0.01}, 'name': 'RMSprop_001'},
        {'class': RMSprop, 'params': {'lr': 0.001}, 'name': 'RMSprop_0001'},
        {'class': Adagrad, 'params': {'lr': 0.01}, 'name': 'Adagrad_001'},
        {'class': Adagrad, 'params': {'lr': 0.1}, 'name': 'Adagrad_01'},
        {'class': SGD, 'params': {'lr': 0.01, 'momentum': 0.9}, 'name': 'SGD_001_m09'},
    ]
    
    optimizer_results = {}
    
    for opt_config in optimizers_config:
        print(f"\n--- Testing {opt_config['name']} ---")
        
        # Initialize
        y_opt = torch.randn(problem.dim, dtype=torch.float64, requires_grad=True)
        optimizer = opt_config['class']([y_opt], **opt_config['params'])
        
        # Optimization loop
        for step in range(100):
            optimizer.zero_grad()
            
            # Lower-level objective
            g_val = 0.5 * y_opt.T @ problem.Q_lower @ y_opt + x_test.T @ problem.P @ y_opt
            
            # Constraint penalty
            h_val = problem.A @ x_test + problem.B @ y_opt - problem.b
            constraint_penalty = torch.sum(torch.clamp(h_val, min=0) ** 2)
            
            # Total objective
            total_obj = g_val + 1000 * constraint_penalty
            
            total_obj.backward()
            optimizer.step()
        
        # Evaluate
        h_val_final = problem.A @ x_test + problem.B @ y_opt - problem.b
        y_diff = torch.norm(y_opt - y_cvxpy).item()
        
        print(f"  Solution y*: {y_opt.detach().numpy()}")
        print(f"  Constraint violations: {h_val_final.detach().numpy()}")
        print(f"  Max constraint violation: {torch.max(h_val_final).item():.6f}")
        print(f"  Difference from CVXPY: {y_diff:.6f}")
        
        optimizer_results[opt_config['name']] = {
            'y_opt': y_opt.detach().numpy(),
            'constraint_violations': h_val_final.detach().numpy(),
            'max_violation': torch.max(h_val_final).item(),
            'diff_from_cvxpy': y_diff
        }
    
    return optimizer_results, y_cvxpy.detach().numpy()

def analyze_lower_level_solver_accuracy(problem, x_test):
    """Comprehensive analysis of lower-level solver accuracy"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE LOWER-LEVEL SOLVER ACCURACY ANALYSIS")
    print("=" * 80)
    
    # Test multiple x values
    x_values = [
        x_test,
        torch.randn(problem.dim, dtype=torch.float64),
        torch.randn(problem.dim, dtype=torch.float64),
        torch.randn(problem.dim, dtype=torch.float64),
    ]
    
    all_results = {}
    
    for i, x in enumerate(x_values):
        print(f"\n--- Testing x[{i}] = {x.detach().numpy()} ---")
        
        # CVXPY solution
        y_cvxpy, info_cvxpy = problem.solve_lower_level(x)
        dual_cvxpy = info_cvxpy.get('lambda', None)
        
        print(f"CVXPY y*: {y_cvxpy.detach().numpy()}")
        print(f"CVXPY duals: {dual_cvxpy.detach().numpy() if dual_cvxpy is not None else 'None'}")
        
        # Test SGD with best configuration
        y_sgd = torch.randn(problem.dim, dtype=torch.float64, requires_grad=True)
        optimizer = SGD([y_sgd], lr=0.001, momentum=0.9)
        
        for step in range(200):
            optimizer.zero_grad()
            g_val = 0.5 * y_sgd.T @ problem.Q_lower @ y_sgd + x.T @ problem.P @ y_sgd
            h_val = problem.A @ x + problem.B @ y_sgd - problem.b
            constraint_penalty = torch.sum(torch.clamp(h_val, min=0) ** 2)
            total_obj = g_val + 1000 * constraint_penalty
            total_obj.backward()
            optimizer.step()
        
        h_val_sgd = problem.A @ x + problem.B @ y_sgd - problem.b
        y_diff = torch.norm(y_sgd - y_cvxpy).item()
        
        print(f"SGD y*: {y_sgd.detach().numpy()}")
        print(f"SGD constraint violations: {h_val_sgd.detach().numpy()}")
        print(f"Difference from CVXPY: {y_diff:.6f}")
        
        all_results[f'x_{i}'] = {
            'x': x.detach().numpy(),
            'y_cvxpy': y_cvxpy.detach().numpy(),
            'y_sgd': y_sgd.detach().numpy(),
            'dual_cvxpy': dual_cvxpy.detach().numpy() if dual_cvxpy is not None else None,
            'constraint_violations_sgd': h_val_sgd.detach().numpy(),
            'diff_from_cvxpy': y_diff
        }
    
    return all_results

def main():
    """Main function"""
    print("DEBUGGING ALGORITHM 1 ACCURACY AND LOWER-LEVEL SOLVER")
    print("=" * 80)
    
    # Create problem with balanced constraint tightening
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, seed=42)
    
    # Apply balanced constraint tightening
    problem.b = problem.b - 0.2
    problem.B = problem.B * 2.5
    problem.Q_lower = problem.Q_lower * 1.8
    
    print(f"Problem setup:")
    print(f"  b: {problem.b.detach().numpy()}")
    print(f"  B norm: {torch.norm(problem.B).item():.6f}")
    print(f"  Q_lower norm: {torch.norm(problem.Q_lower).item():.6f}")
    
    # Create F2CSA algorithm
    f2csa = F2CSAAlgorithm(
        problem=problem,
        alpha_override=0.08,
        eta_override=0.001,
        D_override=0.01,
        Ng_override=64,
        grad_ema_beta_override=0.9,
        prox_weight_override=0.1,
        grad_clip_override=1.0
    )
    
    # Test x
    x_test = torch.randn(problem.dim, dtype=torch.float64)
    
    # 1. Debug Algorithm 1 before averaging
    print("\n" + "=" * 80)
    print("STEP 1: DEBUGGING ALGORITHM 1 BEFORE AVERAGING")
    print("=" * 80)
    algo1_results = debug_algorithm1_before_averaging(problem, f2csa, x_test)
    
    # 2. Validate SGD solver
    print("\n" + "=" * 80)
    print("STEP 2: VALIDATING SGD SOLVER")
    print("=" * 80)
    sgd_results, y_cvxpy = validate_sgd_solver(problem, x_test)
    
    # 3. Test alternative optimizers
    print("\n" + "=" * 80)
    print("STEP 3: TESTING ALTERNATIVE OPTIMIZERS")
    print("=" * 80)
    optimizer_results, y_cvxpy_ref = test_alternative_optimizers(problem, x_test)
    
    # 4. Comprehensive analysis
    print("\n" + "=" * 80)
    print("STEP 4: COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    comprehensive_results = analyze_lower_level_solver_accuracy(problem, x_test)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    
    print("\nAlgorithm 1 Ng Analysis:")
    for Ng, result in algo1_results.items():
        print(f"  Ng={Ng}: norm={result['norm']:.6f}, std_norm={result['std_norm']:.6f}")
    
    print("\nSGD Solver Accuracy:")
    for name, result in sgd_results.items():
        print(f"  {name}: diff={result['diff_from_cvxpy']:.6f}, max_violation={result['max_violation']:.6f}")
    
    print("\nAlternative Optimizers:")
    for name, result in optimizer_results.items():
        print(f"  {name}: diff={result['diff_from_cvxpy']:.6f}, max_violation={result['max_violation']:.6f}")
    
    # Find best solver
    best_solver = min(optimizer_results.items(), key=lambda x: x[1]['diff_from_cvxpy'])
    print(f"\nBest solver: {best_solver[0]} (diff={best_solver[1]['diff_from_cvxpy']:.6f})")
    
    print("\n" + "=" * 80)
    print("DEBUGGING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
