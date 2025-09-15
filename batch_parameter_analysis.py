#!/usr/bin/env python3
"""
Batch Parameter Analysis for F2CSA Algorithm 2
Windows-compatible batch processing using multiprocessing
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import time
import json
from dataclasses import dataclass
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
warnings.filterwarnings('ignore')

# Import our modules
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final

@dataclass
class ParameterConfig:
    """Configuration for parameter analysis"""
    alpha: float
    epsilon: float
    delta: float
    N_g: int
    M: int
    D: float
    eta: float
    T: int
    L_F: float = 1.0
    sigma: float = 1.0

def run_single_configuration(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single parameter configuration (for multiprocessing)"""
    config = ParameterConfig(**config_data['config'])
    x0 = torch.tensor(config_data['x0'])
    
    print(f"Starting α={config.alpha:.3f} (PID: {os.getpid()})")
    
    try:
        # Create problem and algorithm
        problem = StronglyConvexBilevelProblem()
        algorithm1 = F2CSAAlgorithm1Final(problem)
        
        # Initialize Algorithm 2 variables
        x = x0.clone()
        Delta = torch.zeros_like(x)
        
        # Track metrics
        losses = []
        grad_norms = []
        gaps = []
        hypergrad_norms = []
        
        start_time = time.time()
        
        for t in range(1, min(config.T + 1, 1000)):  # Limit iterations for batch processing
            # Sample s_t ~ Unif[0,1]
            s_t = torch.rand(1).item()
            
            # Update positions
            x_t = x + Delta
            z_t = x + s_t * Delta
            
            # Compute hypergradient using Algorithm 1
            hypergrad = algorithm1.oracle_sample(
                z_t, 
                alpha=config.alpha, 
                N_g=config.N_g
            )
            
            # Update direction with clipping
            Delta_new = Delta - config.eta * hypergrad
            Delta = clip_D(Delta_new, config.D)
            
            # Track metrics every 10 iterations
            if t % 10 == 0:
                # Compute true lower-level solution for gap calculation
                y_star, _ = problem.solve_lower_level(z_t)
                
                # Compute approximate solution using penalty minimizer
                y_tilde = algorithm1._minimize_penalty_lagrangian_detailed(
                    z_t, y_star, 
                    algorithm1._solve_lower_level_accurate(z_t, config.alpha)[1]['lambda'],
                    config.alpha, config.delta
                )
                
                # Calculate gap
                gap = torch.norm(y_tilde - y_star).item()
                
                # Calculate loss
                loss = problem.f(z_t, y_star).item()
                
                # Store metrics
                losses.append(loss)
                grad_norms.append(torch.norm(hypergrad).item())
                gaps.append(gap)
                hypergrad_norms.append(torch.norm(hypergrad).item())
            
            # Update x for next iteration
            x = x_t
        
        end_time = time.time()
        
        # Calculate final metrics
        final_loss = losses[-1] if losses else float('inf')
        final_gap = gaps[-1] if gaps else float('inf')
        final_grad_norm = grad_norms[-1] if grad_norms else float('inf')
        
        # Check convergence criteria
        gap_converged = final_gap < 0.1
        grad_stable = len(grad_norms) > 10 and np.std(grad_norms[-10:]) < 0.1 * np.mean(grad_norms[-10:])
        convergence_achieved = gap_converged and grad_stable
        
        result = {
            'config': config_data['config'],
            'success': True,
            'iterations': min(config.T, 1000),
            'runtime': end_time - start_time,
            'final_loss': final_loss,
            'final_gap': final_gap,
            'final_grad_norm': final_grad_norm,
            'convergence_achieved': convergence_achieved,
            'losses': losses,
            'gaps': gaps,
            'grad_norms': grad_norms,
            'hypergrad_norms': hypergrad_norms
        }
        
        print(f"Completed α={config.alpha:.3f}: Gap={final_gap:.6f}, GradNorm={final_grad_norm:.6f}, Converged={convergence_achieved}")
        
        return result
        
    except Exception as e:
        print(f"ERROR in α={config.alpha:.3f}: {e}")
        return {
            'config': config_data['config'],
            'success': False,
            'error': str(e),
            'iterations': 0,
            'final_loss': float('inf'),
            'final_gap': float('inf'),
            'final_grad_norm': float('inf'),
            'convergence_achieved': False
        }

def clip_D(v: torch.Tensor, D: float) -> torch.Tensor:
    """Clip vector to have norm at most D"""
    norm = torch.norm(v)
    if norm <= D:
        return v
    else:
        return v * (D / norm)

class BatchParameterAnalyzer:
    """Batch parameter analysis for F2CSA Algorithm 2"""
    
    def __init__(self, problem: StronglyConvexBilevelProblem, max_workers: int = None):
        self.problem = problem
        self.results = []
        self.max_workers = max_workers or min(mp.cpu_count(), 8)  # Limit to 8 processes
        
    def create_parameter_configs(self) -> List[Dict[str, Any]]:
        """Create various parameter configurations for batch processing"""
        configs = []
        
        # Base parameters from F2CSA.tex
        base_L_F = 1.0
        base_sigma = 1.0
        
        # Test different alpha values (key parameter)
        alpha_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        for alpha in alpha_values:
            # Calculate derived parameters according to F2CSA.tex schedules
            epsilon = alpha  # α = c_α * ε, so ε = α when c_α = 1
            delta = alpha**3  # δ = c_δ * α^3, so δ = α^3 when c_δ = 1
            N_g = max(1, int(base_sigma**2 / alpha**2))  # N_g = c_g * σ^2/α^2
            M = max(1, int(1 / epsilon**2))  # M = ⌊c_M/ε^2⌋
            D = delta * epsilon**2 / base_L_F**2  # D = c_D * δ * ε^2/L_F^2
            eta = delta * epsilon**3 / base_L_F**4  # η = c_η * δ * ε^3/L_F^4
            T = max(100, min(1000, int(1000 / (delta * epsilon**3))))  # T = O(1/(δ*ε^3))
            
            config = {
                'alpha': alpha,
                'epsilon': epsilon,
                'delta': delta,
                'N_g': N_g,
                'M': M,
                'D': D,
                'eta': eta,
                'T': T,
                'L_F': base_L_F,
                'sigma': base_sigma
            }
            configs.append(config)
            
        return configs
    
    def run_batch_analysis(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run batch analysis using multiprocessing"""
        print(f"=== BATCH PARAMETER ANALYSIS ===")
        print(f"Testing {len(configs)} parameter configurations using {self.max_workers} workers...")
        
        # Prepare data for multiprocessing
        x0 = torch.randn(10, requires_grad=False)
        config_data = []
        for config in configs:
            config_data.append({
                'config': config,
                'x0': x0.tolist()  # Convert to list for JSON serialization
            })
        
        results = []
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_config = {
                executor.submit(run_single_configuration, data): data['config']['alpha'] 
                for data in config_data
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_config):
                alpha = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"✓ Completed α={alpha:.3f}")
                except Exception as e:
                    print(f"✗ Failed α={alpha:.3f}: {e}")
                    results.append({
                        'config': {'alpha': alpha},
                        'success': False,
                        'error': str(e),
                        'iterations': 0,
                        'final_loss': float('inf'),
                        'final_gap': float('inf'),
                        'final_grad_norm': float('inf'),
                        'convergence_achieved': False
                    })
        
        # Sort results by alpha
        results.sort(key=lambda x: x['config']['alpha'])
        self.results = results
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the batch results"""
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return {'error': 'No successful runs'}
        
        # Extract data for analysis
        alphas = [r['config']['alpha'] for r in successful_results]
        final_gaps = [r['final_gap'] for r in successful_results]
        final_grad_norms = [r['final_grad_norm'] for r in successful_results]
        convergence_rates = [r['convergence_achieved'] for r in successful_results]
        
        # Find best configurations
        best_gap_idx = np.argmin(final_gaps)
        best_grad_idx = np.argmin(final_grad_norms)
        best_convergence_idx = np.argmax(convergence_rates)
        
        analysis = {
            'total_configs': len(results),
            'successful_configs': len(successful_results),
            'convergence_rate': np.mean(convergence_rates),
            'best_gap_config': {
                'alpha': alphas[best_gap_idx],
                'gap': final_gaps[best_gap_idx],
                'grad_norm': final_grad_norms[best_gap_idx]
            },
            'best_grad_config': {
                'alpha': alphas[best_grad_idx],
                'gap': final_gaps[best_grad_idx],
                'grad_norm': final_grad_norms[best_grad_idx]
            },
            'best_convergence_config': {
                'alpha': alphas[best_convergence_idx],
                'gap': final_gaps[best_convergence_idx],
                'grad_norm': final_grad_norms[best_convergence_idx]
            },
            'parameter_trends': {
                'alphas': alphas,
                'gaps': final_gaps,
                'grad_norms': final_grad_norms,
                'convergence': convergence_rates
            }
        }
        
        return analysis
    
    def plot_results(self, analysis: Dict[str, Any], save_path: str = None):
        """Plot batch analysis results"""
        if 'parameter_trends' not in analysis:
            print("No data to plot")
            return
        
        trends = analysis['parameter_trends']
        alphas = trends['alphas']
        gaps = trends['gaps']
        grad_norms = trends['grad_norms']
        convergence = trends['convergence']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Gap vs Alpha
        ax1.scatter(alphas, gaps, c=convergence, cmap='RdYlGn', alpha=0.7, s=50)
        ax1.axhline(y=0.1, color='r', linestyle='--', label='Target Gap < 0.1')
        ax1.set_xlabel('Alpha (α)')
        ax1.set_ylabel('Final Gap')
        ax1.set_title('Gap vs Alpha Parameter (Batch Results)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Gradient Norm vs Alpha
        ax2.scatter(alphas, grad_norms, c=convergence, cmap='RdYlGn', alpha=0.7, s=50)
        ax2.set_xlabel('Alpha (α)')
        ax2.set_ylabel('Final Gradient Norm')
        ax2.set_title('Gradient Norm vs Alpha Parameter (Batch Results)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Convergence Rate
        convergence_by_alpha = {}
        for i, alpha in enumerate(alphas):
            if alpha not in convergence_by_alpha:
                convergence_by_alpha[alpha] = []
            convergence_by_alpha[alpha].append(convergence[i])
        
        alpha_vals = sorted(convergence_by_alpha.keys())
        conv_rates = [np.mean(convergence_by_alpha[a]) for a in alpha_vals]
        
        ax3.bar(alpha_vals, conv_rates, alpha=0.7)
        ax3.set_xlabel('Alpha (α)')
        ax3.set_ylabel('Convergence Rate')
        ax3.set_title('Convergence Rate by Alpha (Batch Results)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Gap vs Gradient Norm
        scatter = ax4.scatter(gaps, grad_norms, c=alphas, cmap='viridis', alpha=0.7, s=50)
        ax4.axvline(x=0.1, color='r', linestyle='--', label='Target Gap < 0.1')
        ax4.set_xlabel('Final Gap')
        ax4.set_ylabel('Final Gradient Norm')
        ax4.set_title('Gap vs Gradient Norm (Batch Results)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Alpha')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any], filename: str = 'batch_analysis_results.json'):
        """Save results to JSON file"""
        # Convert torch tensors to lists for JSON serialization
        serializable_results = []
        for result in results:
            serializable_result = result.copy()
            if 'losses' in serializable_result:
                serializable_result['losses'] = [float(x) for x in serializable_result['losses']]
            if 'gaps' in serializable_result:
                serializable_result['gaps'] = [float(x) for x in serializable_result['gaps']]
            if 'grad_norms' in serializable_result:
                serializable_result['grad_norms'] = [float(x) for x in serializable_result['grad_norms']]
            if 'hypergrad_norms' in serializable_result:
                serializable_result['hypergrad_norms'] = [float(x) for x in serializable_result['hypergrad_norms']]
            serializable_results.append(serializable_result)
        
        output = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_configs': len(results),
            'successful_configs': len([r for r in results if r['success']]),
            'analysis': analysis,
            'results': serializable_results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {filename}")

def main():
    """Main function to run batch parameter analysis"""
    print("=== F2CSA Algorithm 2 Batch Parameter Analysis ===")
    print(f"Using {mp.cpu_count()} CPU cores")
    
    # Create problem
    problem = StronglyConvexBilevelProblem()
    
    # Create batch analyzer
    analyzer = BatchParameterAnalyzer(problem, max_workers=min(mp.cpu_count(), 8))
    
    # Create parameter configurations
    configs = analyzer.create_parameter_configs()
    print(f"Created {len(configs)} parameter configurations")
    
    # Run batch analysis
    start_time = time.time()
    results = analyzer.run_batch_analysis(configs)
    end_time = time.time()
    
    print(f"\nBatch analysis completed in {end_time - start_time:.2f} seconds")
    
    # Analyze results
    analysis = analyzer.analyze_results(results)
    
    # Print results
    print(f"\n=== BATCH ANALYSIS RESULTS ===")
    if 'error' in analysis:
        print(f"Analysis failed: {analysis['error']}")
        return
    
    print(f"Total configurations: {analysis['total_configs']}")
    print(f"Successful configurations: {analysis['successful_configs']}")
    print(f"Overall convergence rate: {analysis['convergence_rate']:.2%}")
    
    print(f"\nBest Gap Configuration:")
    print(f"  Alpha: {analysis['best_gap_config']['alpha']:.3f}")
    print(f"  Gap: {analysis['best_gap_config']['gap']:.6f}")
    print(f"  Grad Norm: {analysis['best_gap_config']['grad_norm']:.6f}")
    
    print(f"\nBest Convergence Configuration:")
    print(f"  Alpha: {analysis['best_convergence_config']['alpha']:.3f}")
    print(f"  Gap: {analysis['best_convergence_config']['gap']:.6f}")
    print(f"  Grad Norm: {analysis['best_convergence_config']['grad_norm']:.6f}")
    
    # Plot results
    analyzer.plot_results(analysis, 'batch_parameter_analysis.png')
    
    # Save results
    analyzer.save_results(results, analysis, 'batch_analysis_results.json')
    
    print(f"\n=== BATCH PROCESSING COMPLETE ===")
    print(f"Results saved to batch_analysis_results.json")
    print(f"Plot saved to batch_parameter_analysis.png")

if __name__ == "__main__":
    main()
