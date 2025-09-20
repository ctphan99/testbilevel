#!/usr/bin/env python3
"""
Tune SSIGD parameters to reduce upper-level loss
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from ssigd_correct_final import CorrectSSIGD

def tune_ssigd_parameters():
    """Tune SSIGD parameters to minimize upper-level loss"""
    
    print("🔬 SSIGD Parameter Tuning")
    print("=" * 60)
    
    # Fixed parameters
    dim = 100
    seed = 1234
    T = 1000  # Long iterations for better convergence
    
    print(f"Fixed Parameters:")
    print(f"  Dimension: {dim}")
    print(f"  Seed: {seed}")
    print(f"  Iterations: {T}")
    print()
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=dim, device='cpu')
    x0 = torch.randn(dim, dtype=torch.float64) * 0.1
    
    print(f"Problem Info:")
    print(f"  Upper level strong convexity: λ_min={torch.linalg.eigvals(problem.Q_upper).real.min():.6f}")
    print(f"  Lower level strong convexity: λ_min={torch.linalg.eigvals(problem.Q_lower).real.min():.6f}")
    print()
    
    # Initial loss
    y0, _, _ = problem.solve_lower_level(x0, solver='gurobi')
    initial_loss = problem.upper_objective(x0, y0).item()
    print(f"Initial UL Loss: {initial_loss:.6f}")
    print()
    
    # Parameter grid to test
    beta_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    mu_F_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    
    print("Parameter Grid:")
    print(f"  Beta values: {beta_values}")
    print(f"  mu_F values: {mu_F_values}")
    print(f"  Total combinations: {len(beta_values) * len(mu_F_values)}")
    print()
    
    best_result = None
    best_loss = float('inf')
    best_params = None
    results = []
    
    print("Testing parameter combinations...")
    print("-" * 60)
    
    for i, beta in enumerate(beta_values):
        for j, mu_F in enumerate(mu_F_values):
            # Reset seed for each run to ensure fair comparison
            torch.manual_seed(seed)
            np.random.seed(seed)
            problem_test = StronglyConvexBilevelProblem(dim=dim, device='cpu')
            x0_test = torch.randn(dim, dtype=torch.float64) * 0.1
            
            try:
                ssigd = CorrectSSIGD(problem_test)
                result = ssigd.solve(T=T, beta=beta, x0=x0_test, diminishing=True, mu_F=mu_F)
                
                final_loss = result['final_loss']
                final_grad = result['final_grad_norm']
                
                results.append({
                    'beta': beta,
                    'mu_F': mu_F,
                    'final_loss': final_loss,
                    'final_grad': final_grad,
                    'converged': True
                })
                
                print(f"β={beta:.4f}, μ_F={mu_F:.2f}: Loss={final_loss:.6f}, Grad={final_grad:.6f}")
                
                if final_loss < best_loss:
                    best_loss = final_loss
                    best_result = result
                    best_params = (beta, mu_F)
                    
            except Exception as e:
                print(f"β={beta:.4f}, μ_F={mu_F:.2f}: FAILED - {e}")
                results.append({
                    'beta': beta,
                    'mu_F': mu_F,
                    'final_loss': float('inf'),
                    'final_grad': float('inf'),
                    'converged': False
                })
    
    print("\n" + "=" * 60)
    print("TUNING RESULTS")
    print("=" * 60)
    
    # Sort results by final loss
    successful_results = [r for r in results if r['converged']]
    successful_results.sort(key=lambda x: x['final_loss'])
    
    print("Top 5 Parameter Combinations:")
    print(f"{'Rank':<5} {'Beta':<8} {'mu_F':<8} {'Final Loss':<12} {'Final Grad':<12}")
    print("-" * 60)
    
    for i, result in enumerate(successful_results[:5]):
        print(f"{i+1:<5} {result['beta']:<8.4f} {result['mu_F']:<8.2f} {result['final_loss']:<12.6f} {result['final_grad']:<12.6f}")
    
    if best_params:
        print(f"\n🏆 Best Parameters:")
        print(f"  Beta: {best_params[0]:.6f}")
        print(f"  mu_F: {best_params[1]:.6f}")
        print(f"  Final Loss: {best_loss:.6f}")
        print(f"  Improvement: {initial_loss - best_loss:.6f}")
        print(f"  Reduction: {((initial_loss - best_loss) / initial_loss * 100):.2f}%")
        
        # Run final test with best parameters
        print(f"\n" + "=" * 60)
        print("FINAL TEST WITH BEST PARAMETERS")
        print("=" * 60)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        problem_final = StronglyConvexBilevelProblem(dim=dim, device='cpu')
        x0_final = torch.randn(dim, dtype=torch.float64) * 0.1
        
        ssigd_final = CorrectSSIGD(problem_final)
        result_final = ssigd_final.solve(T=T, beta=best_params[0], x0=x0_final, diminishing=True, mu_F=best_params[1])
        
        print(f"Final Results:")
        print(f"  Initial Loss: {initial_loss:.6f}")
        print(f"  Final Loss: {result_final['final_loss']:.6f}")
        print(f"  Final Grad: {result_final['final_grad_norm']:.6f}")
        print(f"  Total Improvement: {initial_loss - result_final['final_loss']:.6f}")
        print(f"  Percentage Reduction: {((initial_loss - result_final['final_loss']) / initial_loss * 100):.2f}%")
    
    print("=" * 60)
    
    return best_params, best_loss, results

if __name__ == "__main__":
    best_params, best_loss, results = tune_ssigd_parameters()
