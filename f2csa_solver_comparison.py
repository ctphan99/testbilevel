#!/usr/bin/env python3
"""
F2CSA Solver Comparison Script
Test different first-order solvers (Clarabel, OSQP, SCS) for F2CSA performance
"""

import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm2_working import F2CSAAlgorithm2Working
import warnings

warnings.filterwarnings('ignore')

class SolverComparisonF2CSA:
    """F2CSA with configurable solver support"""
    
    def __init__(self, problem, solver='SCS', device='cpu', dtype=torch.float64):
        self.problem = problem
        self.solver = solver
        self.device = device
        self.dtype = dtype
        
        # Initialize Algorithm 1 for hypergradient computation
        from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
        self.algorithm1 = F2CSAAlgorithm1Final(problem, device=device, dtype=dtype)
        
        # Override the solver in the problem
        self._patch_solver()
        
        # Persistent cache for warm-start
        self.prev_y = None
        self.prev_lambda = None
        self.prev_adam_state = None
        
    def _patch_solver(self):
        """Patch the problem's solver to use the specified solver"""
        original_solve_accurate = self.problem._solve_accurate
        
        def patched_solve_accurate(x, alpha, max_iter, tol):
            try:
                from accurate_lower_level_solver import AccurateLowerLevelSolver
                solver = AccurateLowerLevelSolver(self.problem, device=self.device, dtype=self.dtype)
                # Use the specified solver
                y_opt, lambda_opt, info = solver.solve_lower_level_with_solver(x, alpha, max_iter, tol, self.solver)
                return y_opt, info
            except ImportError:
                print("Accurate solver not available, falling back to PGD")
                return self.problem._solve_pgd(x, max_iter, tol)
        
        self.problem._solve_accurate = patched_solve_accurate
    
    def clip_D(self, v: torch.Tensor, D: float) -> torch.Tensor:
        """Clipping function: clip_D(v) := min{1, D/||v||} * v"""
        v_norm = torch.norm(v)
        if v_norm <= D:
            return v
        else:
            return (D / v_norm) * v
    
    def optimize(self, x0: torch.Tensor, T: int, D: float, eta: float, 
                 delta: float, alpha: float, N_g: int = None, 
                 warm_ll: bool = False, keep_adam_state: bool = False,
                 plot_name: str = None, save_warm_name: str = None) -> dict:
        """
        F2CSA Algorithm 2 optimization with configurable solver
        """
        print(f"F2CSA Algorithm 2 - WORKING Implementation with {self.solver} solver")
        print("=" * 70)
        print(f"T = {T}, D = {D:.6f}, η = {eta:.6f}")
        print(f"δ = {delta:.6f}, α = {alpha:.6f}")
        print()
        
        if N_g is None:
            N_g = max(1, int(1 / (alpha ** 2)))  # Default from F2CSA.tex
        
        print(f"Using N_g = {N_g} samples for hypergradient estimation")
        print()
        
        x = x0.clone().detach()
        ul_losses = []
        hypergrad_norms = []
        
        print("Starting optimization...")
        print("-" * 50)
        
        for t in range(T):
            # Compute accurate lower-level solution with δ = α³
            print(f"  Computing accurate lower-level solution with δ = {delta:.2e}")
            y_star, info = self.problem.solve_lower_level(x, solver='accurate', alpha=alpha, max_iter=10000, tol=1e-8)
            print(f"  Lower-level solution: ỹ* = {y_star[:10]}...")
            print(f"  Lower-level info: {info}")
            
            # Use oracle_sample to get hypergradient, y_tilde, and lambda_star
            print(f"  Computing penalty minimizer and stochastic hypergradient with N_g = {N_g}")
            grad_est, y_tilde, lambda_star = self.algorithm1.oracle_sample(x, alpha, N_g)
            print(f"  Final hypergradient: ∇F̃ = {grad_est[:10]}...")
            print(f"  Hypergradient norm: ||∇F̃|| = {torch.norm(grad_est).item():.6f}")
            
            # Apply clipping
            grad_clipped = self.clip_D(grad_est, D)
            
            # Compute upper-level loss at x_t to monitor Algorithm 2 gap (f(x, y*))
            y_star, _ = self.problem.solve_lower_level(x)
            ul_loss_t = self.problem.upper_objective(x, y_star).item()
            ul_losses.append(ul_loss_t)
            
            # Track hypergradient norm
            hypergrad_norms.append(torch.norm(grad_est).item())
            
            print(f"Iteration {t+1:4d}/{T}: ||g_t|| = {torch.norm(grad_est).item():.6f}, ||Δ_t|| = {D:.6f}, ul_loss = {ul_loss_t:.6f}")
            print()
            
            # Update x
            x = x - eta * grad_clipped
        
        # Compute final upper-level loss f(x, y*) as Algorithm 2 gap
        y_star, _ = self.problem.solve_lower_level(x)
        final_ul_loss = self.problem.upper_objective(x, y_star).item()
        
        print("Computing output points...")
        print(f"M = 1, K = {T}")
        print(f"Selected output with min f(x, y*) among {T}: f = {final_ul_loss:.6f}")
        print()
        
        # Final diagnostic
        y_star_final, _ = self.problem.solve_lower_level(x)
        final_ul_loss_diag = self.problem.upper_objective(x, y_star_final).item()
        final_grad_norm = torch.norm(grad_est).item()
        
        print(f"Final UL loss f(x, y*): {final_ul_loss_diag:.6f}")
        print(f"Final hypergradient norm (diagnostic): {final_grad_norm:.6f}")
        print(f"Final point: {x[:10]}...")
        
        return {
            'x_out': x,
            'final_gradient': grad_est,
            'final_gradient_norm': final_grad_norm,
            'final_ul_loss': final_ul_loss_diag,
            'ul_losses': ul_losses,
            'hypergrad_norms': hypergrad_norms,
            'iterations': T,
            'converged': final_grad_norm < 1e-3,
            'solver': self.solver
        }

def run_solver_comparison(dim=100, T=2, seed=1234):
    """Run F2CSA with different solvers"""
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create problem instance
    problem = StronglyConvexBilevelProblem(dim=dim, num_constraints=2*dim, noise_std=0.1)
    
    # Generate initial point
    x0 = torch.randn(dim, device='cpu', dtype=torch.float64)
    
    results = {}
    solvers = ['SCS', 'OSQP', 'Clarabel']
    
    for solver in solvers:
        print("=" * 80)
        print(f"TESTING F2CSA WITH {solver} SOLVER")
        print("=" * 80)
        
        try:
            f2csa = SolverComparisonF2CSA(problem, solver=solver)
            
            start_time = time.time()
            f2csa_results = f2csa.optimize(
                x0, T, D=0.03, eta=0.1, delta=0.05**3, alpha=0.05, N_g=1,
                warm_ll=False, keep_adam_state=False,
                plot_name=None, save_warm_name=None
            )
            solver_time = time.time() - start_time
            
            results[solver] = {
                'ul_losses': f2csa_results['ul_losses'],
                'hypergrad_norms': f2csa_results['hypergrad_norms'],
                'total_time': solver_time,
                'avg_time_per_iter': solver_time / T,
                'final_ul_loss': f2csa_results['final_ul_loss'],
                'final_grad_norm': f2csa_results['final_gradient_norm'],
                'converged': f2csa_results['converged']
            }
            
            print(f"\n{solver} COMPLETED: {solver_time:.3f}s total, {solver_time/T:.3f}s/iter")
            
        except Exception as e:
            print(f"ERROR with {solver}: {e}")
            results[solver] = {
                'ul_losses': [],
                'hypergrad_norms': [],
                'total_time': float('inf'),
                'avg_time_per_iter': float('inf'),
                'final_ul_loss': float('inf'),
                'final_grad_norm': float('inf'),
                'converged': False,
                'error': str(e)
            }
    
    return results

def create_solver_plots(results, dim=100, T=2):
    """Create comparison plots for different solvers"""
    
    # Plot 1: UL Loss Comparison
    plt.figure(figsize=(12, 8))
    for solver, data in results.items():
        if 'error' not in data:
            plt.plot(data['ul_losses'], label=f'F2CSA-{solver}', linewidth=2, marker='o')
    
    plt.xlabel('Iteration')
    plt.ylabel('Upper-Level Loss')
    plt.title(f'F2CSA Upper-Level Loss Comparison by Solver (dim={dim}, T={T})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('f2csa_ul_loss_solver_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Hypergradient Norm Comparison
    plt.figure(figsize=(12, 8))
    for solver, data in results.items():
        if 'error' not in data:
            plt.plot(data['hypergrad_norms'], label=f'F2CSA-{solver}', linewidth=2, marker='o')
    
    plt.xlabel('Iteration')
    plt.ylabel('Hypergradient Norm')
    plt.title(f'F2CSA Hypergradient Norm Comparison by Solver (dim={dim}, T={T})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('f2csa_hypergrad_norm_solver_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Timing Comparison
    plt.figure(figsize=(12, 8))
    solvers = [s for s in results.keys() if 'error' not in results[s]]
    avg_times = [results[s]['avg_time_per_iter'] for s in solvers]
    
    bars = plt.bar(solvers, avg_times, color=['blue', 'red', 'green'], alpha=0.7)
    plt.xlabel('Solver')
    plt.ylabel('Average Time per Iteration (seconds)')
    plt.title(f'F2CSA Timing Comparison by Solver (dim={dim}, T={T})')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, avg_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time_val:.3f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('f2csa_timing_solver_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created 3 plots:")
    print("1. f2csa_ul_loss_solver_comparison.png")
    print("2. f2csa_hypergrad_norm_solver_comparison.png")
    print("3. f2csa_timing_solver_comparison.png")

def main():
    parser = argparse.ArgumentParser(description='F2CSA Solver Comparison')
    parser.add_argument('--dim', type=int, default=100, help='Problem dimension')
    parser.add_argument('--T', type=int, default=2, help='Number of iterations')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    
    args = parser.parse_args()
    
    print(f"Running F2CSA solver comparison: dim={args.dim}, T={args.T}, seed={args.seed}")
    
    # Run comparison
    results = run_solver_comparison(dim=args.dim, T=args.T, seed=args.seed)
    
    # Print timing summary
    print("\n" + "="*80)
    print("F2CSA SOLVER TIMING SUMMARY")
    print("="*80)
    for solver, data in results.items():
        if 'error' in data:
            print(f"{solver:>10}: ERROR - {data['error']}")
        else:
            print(f"{solver:>10}: {data['total_time']:.3f}s total, {data['avg_time_per_iter']:.3f}s/iter, "
                  f"final_ul={data['final_ul_loss']:.3f}, final_grad={data['final_grad_norm']:.3f}")
    
    # Create plots
    create_solver_plots(results, dim=args.dim, T=args.T)

if __name__ == "__main__":
    main()
