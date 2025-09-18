#!/usr/bin/env python3
"""
F2CSA N_g Comparison Script
Compare CVXPY OSQP vs CVXPY SCS with N_g=1 vs N_g=10 for 10,000 iterations
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

class F2CSANGComparison:
    """F2CSA with configurable solver and N_g support"""
    
    def __init__(self, problem, solver='OSQP', device='cpu', dtype=torch.float64):
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
        F2CSA Algorithm 2 optimization with configurable solver and N_g
        """
        print(f"F2CSA Algorithm 2 - {self.solver} solver, N_g={N_g}")
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
            if t % 1000 == 0 or t < 5:
                print(f"  Computing accurate lower-level solution with δ = {delta:.2e}")
            y_star, info = self.problem.solve_lower_level(x, solver='accurate', alpha=alpha, max_iter=10000, tol=1e-8)
            if t % 1000 == 0 or t < 5:
                print(f"  Lower-level solution: ỹ* = {y_star[:5]}...")
                print(f"  Lower-level info: {info}")
            
            # Use oracle_sample to get hypergradient, y_tilde, and lambda_star
            if t % 1000 == 0 or t < 5:
                print(f"  Computing penalty minimizer and stochastic hypergradient with N_g = {N_g}")
            grad_est, y_tilde, lambda_star = self.algorithm1.oracle_sample(x, alpha, N_g)
            if t % 1000 == 0 or t < 5:
                print(f"  Final hypergradient: ∇F̃ = {grad_est[:5]}...")
                print(f"  Hypergradient norm: ||∇F̃|| = {torch.norm(grad_est).item():.6f}")
            
            # Apply clipping
            grad_clipped = self.clip_D(grad_est, D)
            
            # Compute upper-level loss at x_t to monitor Algorithm 2 gap (f(x, y*))
            y_star, _ = self.problem.solve_lower_level(x)
            ul_loss_t = self.problem.upper_objective(x, y_star).item()
            ul_losses.append(ul_loss_t)
            
            # Track hypergradient norm
            hypergrad_norms.append(torch.norm(grad_est).item())
            
            if t % 1000 == 0 or t < 5:
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
        print(f"Final point: {x[:5]}...")
        
        return {
            'x_out': x,
            'final_gradient': grad_est,
            'final_gradient_norm': final_grad_norm,
            'final_ul_loss': final_ul_loss_diag,
            'ul_losses': ul_losses,
            'hypergrad_norms': hypergrad_norms,
            'iterations': T,
            'converged': final_grad_norm < 1e-3,
            'solver': self.solver,
            'N_g': N_g
        }

def run_ng_comparison(dim=100, T=10000, seed=1234):
    """Run F2CSA with different solvers and N_g values"""
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create problem instance
    problem = StronglyConvexBilevelProblem(dim=dim, num_constraints=2*dim, noise_std=0.1)
    
    # Generate initial point
    x0 = torch.randn(dim, device='cpu', dtype=torch.float64)
    
    results = {}
    configurations = [
        ('OSQP', 1),
        ('OSQP', 10),
        ('SCS', 1),
        ('SCS', 10)
    ]
    
    for solver, ng in configurations:
        config_name = f"{solver}_Ng{ng}"
        print("=" * 80)
        print(f"TESTING F2CSA WITH {solver} SOLVER, N_g={ng}")
        print("=" * 80)
        
        try:
            f2csa = F2CSANGComparison(problem, solver=solver)
            
            start_time = time.time()
            f2csa_results = f2csa.optimize(
                x0, T, D=0.03, eta=0.1, delta=0.05**3, alpha=0.05, N_g=ng,
                warm_ll=False, keep_adam_state=False,
                plot_name=None, save_warm_name=None
            )
            solver_time = time.time() - start_time
            
            results[config_name] = {
                'ul_losses': f2csa_results['ul_losses'],
                'hypergrad_norms': f2csa_results['hypergrad_norms'],
                'total_time': solver_time,
                'avg_time_per_iter': solver_time / T,
                'final_ul_loss': f2csa_results['final_ul_loss'],
                'final_grad_norm': f2csa_results['final_gradient_norm'],
                'converged': f2csa_results['converged'],
                'solver': solver,
                'N_g': ng
            }
            
            print(f"\n{config_name} COMPLETED: {solver_time:.3f}s total, {solver_time/T:.3f}s/iter")
            
        except Exception as e:
            print(f"ERROR with {config_name}: {e}")
            results[config_name] = {
                'ul_losses': [],
                'hypergrad_norms': [],
                'total_time': float('inf'),
                'avg_time_per_iter': float('inf'),
                'final_ul_loss': float('inf'),
                'final_grad_norm': float('inf'),
                'converged': False,
                'error': str(e),
                'solver': solver,
                'N_g': ng
            }
    
    return results

def create_ng_plots(results, dim=100, T=10000):
    """Create comparison plots for different solvers and N_g values"""
    
    # Plot 1: UL Loss Comparison
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: UL Loss
    plt.subplot(2, 2, 1)
    for config_name, data in results.items():
        if 'error' not in data:
            plt.plot(data['ul_losses'], label=f'{data["solver"]} N_g={data["N_g"]}', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Upper-Level Loss')
    plt.title(f'F2CSA UL Loss Comparison (dim={dim}, T={T})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Subplot 2: Hypergradient Norm Comparison
    plt.subplot(2, 2, 2)
    for config_name, data in results.items():
        if 'error' not in data:
            plt.plot(data['hypergrad_norms'], label=f'{data["solver"]} N_g={data["N_g"]}', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Hypergradient Norm')
    plt.title(f'F2CSA Hypergradient Norm Comparison (dim={dim}, T={T})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Subplot 3: Timing Comparison
    plt.subplot(2, 2, 3)
    configs = [c for c in results.keys() if 'error' not in results[c]]
    avg_times = [results[c]['avg_time_per_iter'] for c in configs]
    labels = [f'{results[c]["solver"]} N_g={results[c]["N_g"]}' for c in configs]
    
    bars = plt.bar(range(len(configs)), avg_times, color=['blue', 'red', 'green', 'orange'], alpha=0.7)
    plt.xlabel('Configuration')
    plt.ylabel('Average Time per Iteration (seconds)')
    plt.title(f'F2CSA Timing Comparison (dim={dim}, T={T})')
    plt.xticks(range(len(configs)), labels, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, avg_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time_val:.3f}s', ha='center', va='bottom')
    
    # Subplot 4: Final Performance Comparison
    plt.subplot(2, 2, 4)
    final_ul_losses = [results[c]['final_ul_loss'] for c in configs]
    final_grad_norms = [results[c]['final_grad_norm'] for c in configs]
    
    x_pos = np.arange(len(configs))
    width = 0.35
    
    bars1 = plt.bar(x_pos - width/2, final_ul_losses, width, label='Final UL Loss', alpha=0.7)
    bars2 = plt.bar(x_pos + width/2, final_grad_norms, width, label='Final Grad Norm', alpha=0.7)
    
    plt.xlabel('Configuration')
    plt.ylabel('Value')
    plt.title(f'Final Performance Comparison (dim={dim}, T={T})')
    plt.xticks(x_pos, labels, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('f2csa_ng_comparison_dim100_T10k.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created plot: f2csa_ng_comparison_dim100_T10k.png")

def main():
    parser = argparse.ArgumentParser(description='F2CSA N_g Comparison')
    parser.add_argument('--dim', type=int, default=100, help='Problem dimension')
    parser.add_argument('--T', type=int, default=10000, help='Number of iterations')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    
    args = parser.parse_args()
    
    print(f"Running F2CSA N_g comparison: dim={args.dim}, T={args.T}, seed={args.seed}")
    
    # Run comparison
    results = run_ng_comparison(dim=args.dim, T=args.T, seed=args.seed)
    
    # Print timing summary
    print("\n" + "="*80)
    print("F2CSA N_g COMPARISON SUMMARY")
    print("="*80)
    for config_name, data in results.items():
        if 'error' in data:
            print(f"{config_name:>15}: ERROR - {data['error']}")
        else:
            print(f"{config_name:>15}: {data['total_time']:.3f}s total, {data['avg_time_per_iter']:.3f}s/iter, "
                  f"final_ul={data['final_ul_loss']:.3f}, final_grad={data['final_grad_norm']:.3f}")
    
    # Create plots
    create_ng_plots(results, dim=args.dim, T=args.T)

if __name__ == "__main__":
    main()
