#!/usr/bin/env python3
"""
Batch Convergence Tester for F2CSA Algorithm 2
Quick batch testing to find stable parameter configurations
"""

import torch
import numpy as np
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
from f2csa_algorithm import F2CSAAlgorithm2Working
import time
import json
from datetime import datetime
import concurrent.futures

class BatchConvergenceTester:
    """
    Batch tester for quick parameter sweep and convergence analysis
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem):
        self.problem = problem
        self.algorithm1 = F2CSAAlgorithm1Final(problem)
        self.algorithm2 = F2CSAAlgorithm2Working(problem)
    
    def test_single_configuration(self, config: Dict) -> Dict:
        """
        Test a single parameter configuration
        """
        alpha = config['alpha']
        eta = config['eta']
        D = config['D']
        T = config['T']
        N_g = config['N_g']
        trial = config.get('trial', 0)
        
        try:
            # Test Algorithm 1 (hypergradient oracle)
            x_test = torch.randn(self.problem.dim, dtype=torch.float64)
            
            # Test hypergradient computation
            start_time = time.time()
            hypergradient = self.algorithm1.oracle_sample(x_test, alpha, N_g)
            algo1_time = time.time() - start_time
            
            # Test gap calculation
            y_star, lambda_star, _ = self.algorithm1._solve_lower_level_accurate(x_test, alpha)
            y_tilde = self.algorithm1._minimize_penalty_lagrangian(x_test, y_star, lambda_star, alpha, alpha**3)
            gap = torch.norm(y_tilde - y_star).item()
            
            algo1_success = gap < 0.1 and torch.norm(hypergradient).item() < 100
            
            # Test Algorithm 2 (optimization)
            x0 = torch.randn(self.problem.dim, dtype=torch.float64)
            delta = alpha**3
            
            start_time = time.time()
            result = self.algorithm2.optimize(x0, T, D, eta, delta, alpha, N_g)
            algo2_time = time.time() - start_time
            
            # Analyze convergence
            grad_norms = result['grad_norms']
            final_grad_norm = grad_norms[-1] if grad_norms else float('inf')
            
            # Check for convergence
            if len(grad_norms) >= 10:
                last_10_norms = np.array(grad_norms[-10:])
                std_dev = np.std(last_10_norms)
                range_norm = np.max(last_10_norms) - np.min(last_10_norms)
                converged = (std_dev < 2.0 and range_norm < 5.0 and final_grad_norm < 10.0)
            else:
                converged = False
                std_dev = float('inf')
                range_norm = float('inf')
            
            return {
                'config': config,
                'algo1_success': algo1_success,
                'algo1_gap': gap,
                'algo1_grad_norm': torch.norm(hypergradient).item(),
                'algo1_time': algo1_time,
                'algo2_success': converged,
                'algo2_final_grad_norm': final_grad_norm,
                'algo2_grad_std': std_dev,
                'algo2_grad_range': range_norm,
                'algo2_time': algo2_time,
                'overall_success': algo1_success and converged,
                'error': None
            }
            
        except Exception as e:
            return {
                'config': config,
                'algo1_success': False,
                'algo1_gap': float('inf'),
                'algo1_grad_norm': float('inf'),
                'algo1_time': 0,
                'algo2_success': False,
                'algo2_final_grad_norm': float('inf'),
                'algo2_grad_std': float('inf'),
                'algo2_grad_range': float('inf'),
                'algo2_time': 0,
                'overall_success': False,
                'error': str(e)
            }
    
    def generate_configurations(self, 
                              alpha_values: List[float] = [0.1, 0.05, 0.01, 0.005, 0.001],
                              eta_values: List[float] = [1e-5, 1e-4, 1e-3, 1e-2],
                              D_values: List[float] = [0.1, 0.5, 1.0, 2.0],
                              T_values: List[int] = [50, 100, 200],
                              N_g_values: List[int] = [5, 10, 20, 50],
                              num_trials: int = 3) -> List[Dict]:
        """
        Generate all parameter configurations to test
        """
        configurations = []
        
        for alpha in alpha_values:
            for eta in eta_values:
                for D in D_values:
                    for T in T_values:
                        for N_g in N_g_values:
                            for trial in range(num_trials):
                                configurations.append({
                                    'alpha': alpha,
                                    'eta': eta,
                                    'D': D,
                                    'T': T,
                                    'N_g': N_g,
                                    'trial': trial,
                                    'delta': alpha**3
                                })
        
        return configurations
    
    def run_batch_test(self, 
                      alpha_values: List[float] = [0.1, 0.05, 0.01, 0.005, 0.001],
                      eta_values: List[float] = [1e-5, 1e-4, 1e-3, 1e-2],
                      D_values: List[float] = [0.1, 0.5, 1.0, 2.0],
                      T_values: List[int] = [50, 100, 200],
                      N_g_values: List[int] = [5, 10, 20, 50],
                      num_trials: int = 3,
                      max_workers: int = None) -> Dict:
        """
        Run batch test with parallel execution
        """
        print("🚀 F2CSA BATCH CONVERGENCE TESTER")
        print("=" * 60)
        
        # Generate configurations
        configurations = self.generate_configurations(
            alpha_values, eta_values, D_values, T_values, N_g_values, num_trials
        )
        
        print(f"Generated {len(configurations)} configurations")
        print(f"Parameter ranges:")
        print(f"  α: {alpha_values}")
        print(f"  η: {eta_values}")
        print(f"  D: {D_values}")
        print(f"  T: {T_values}")
        print(f"  N_g: {N_g_values}")
        print(f"  Trials per config: {num_trials}")
        print()
        
        # Set up parallel execution
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 8)  # Limit to 8 workers
        
        print(f"Running with {max_workers} parallel workers...")
        start_time = time.time()
        
        # Run tests in parallel
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_config = {
                executor.submit(self.test_single_configuration, config): config 
                for config in configurations
            }
            
            # Collect results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_config):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if completed % max(1, len(configurations) // 20) == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        eta = (len(configurations) - completed) / rate if rate > 0 else 0
                        print(f"  Progress: {completed}/{len(configurations)} ({completed/len(configurations)*100:.1f}%) - ETA: {eta:.1f}s")
                        
                except Exception as e:
                    print(f"  Error in parallel execution: {e}")
                    results.append({
                        'config': future_to_config[future],
                        'error': str(e),
                        'overall_success': False
                    })
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Batch test completed in {total_time:.1f} seconds")
        print(f"  Average time per config: {total_time/len(configurations):.3f}s")
        
        # Analyze results
        return self.analyze_batch_results(results)
    
    def analyze_batch_results(self, results: List[Dict]) -> Dict:
        """
        Analyze batch test results
        """
        print("\n📊 ANALYZING BATCH RESULTS")
        print("=" * 50)
        
        # Filter successful results
        successful_results = [r for r in results if r.get('overall_success', False)]
        failed_results = [r for r in results if not r.get('overall_success', False)]
        
        print(f"Total configurations: {len(results)}")
        print(f"Successful: {len(successful_results)} ({len(successful_results)/len(results)*100:.1f}%)")
        print(f"Failed: {len(failed_results)} ({len(failed_results)/len(results)*100:.1f}%)")
        
        if not successful_results:
            print("❌ No successful configurations found!")
            return {
                'total_configs': len(results),
                'successful_configs': 0,
                'success_rate': 0.0,
                'best_configs': [],
                'parameter_analysis': {},
                'recommendations': []
            }
        
        # Group by parameters for analysis
        param_analysis = {}
        
        # Analyze by alpha
        alpha_groups = {}
        for result in successful_results:
            alpha = result['config']['alpha']
            if alpha not in alpha_groups:
                alpha_groups[alpha] = []
            alpha_groups[alpha].append(result)
        
        print(f"\nSuccess rate by α:")
        for alpha in sorted(alpha_groups.keys()):
            alpha_results = alpha_groups[alpha]
            success_rate = len(alpha_results) / len([r for r in results if r['config']['alpha'] == alpha])
            avg_gap = np.mean([r['algo1_gap'] for r in alpha_results])
            avg_grad_norm = np.mean([r['algo2_final_grad_norm'] for r in alpha_results])
            print(f"  α = {alpha}: {success_rate:.2%} success, avg gap = {avg_gap:.4f}, avg grad = {avg_grad_norm:.4f}")
        
        # Find best configurations
        def score_config(result):
            # Score based on gap, gradient norm, and stability
            gap_score = 1.0 / (1.0 + result['algo1_gap'])
            grad_score = 1.0 / (1.0 + result['algo2_final_grad_norm'])
            stability_score = 1.0 / (1.0 + result['algo2_grad_std'])
            return gap_score + grad_score + stability_score
        
        best_configs = sorted(successful_results, key=score_config, reverse=True)[:10]
        
        print(f"\n🏆 TOP 10 CONFIGURATIONS:")
        print("-" * 100)
        print(f"{'Rank':<4} {'α':<8} {'η':<10} {'D':<6} {'T':<4} {'N_g':<4} {'Gap':<8} {'Grad':<8} {'Std':<8} {'Score':<8}")
        print("-" * 100)
        
        for i, config in enumerate(best_configs):
            print(f"{i+1:<4} {config['config']['alpha']:<8.3f} {config['config']['eta']:<10.1e} "
                  f"{config['config']['D']:<6.1f} {config['config']['T']:<4} {config['config']['N_g']:<4} "
                  f"{config['algo1_gap']:<8.4f} {config['algo2_final_grad_norm']:<8.4f} "
                  f"{config['algo2_grad_std']:<8.4f} {score_config(config):<8.4f}")
        
        # Generate recommendations
        recommendations = []
        
        # Best overall
        if best_configs:
            best = best_configs[0]
            recommendations.append({
                'type': 'best_overall',
                'config': best['config'],
                'description': f"Best overall (score: {score_config(best):.4f})"
            })
        
        # Best gap stability
        gap_best = min(successful_results, key=lambda x: x['algo1_gap'])
        recommendations.append({
            'type': 'best_gap',
            'config': gap_best['config'],
            'description': f"Best gap stability (gap: {gap_best['algo1_gap']:.4f})"
        })
        
        # Best convergence
        conv_best = min(successful_results, key=lambda x: x['algo2_final_grad_norm'])
        recommendations.append({
            'type': 'best_convergence',
            'config': conv_best['config'],
            'description': f"Best convergence (grad: {conv_best['algo2_final_grad_norm']:.4f})"
        })
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            config = rec['config']
            print(f"  {i}. {rec['description']}")
            print(f"     α={config['alpha']}, η={config['eta']:.1e}, D={config['D']}, T={config['T']}, N_g={config['N_g']}")
        
        return {
            'total_configs': len(results),
            'successful_configs': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'best_configs': best_configs,
            'parameter_analysis': param_analysis,
            'recommendations': recommendations,
            'all_results': results
        }
    
    def quick_parameter_sweep(self, 
                            alpha_values: List[float] = [0.1, 0.05, 0.01],
                            eta_values: List[float] = [1e-4, 1e-3],
                            D_values: List[float] = [0.5, 1.0],
                            T_values: List[int] = [100],
                            N_g_values: List[int] = [10, 20],
                            num_trials: int = 2) -> Dict:
        """
        Quick parameter sweep for rapid testing
        """
        print("⚡ QUICK PARAMETER SWEEP")
        print("=" * 40)
        
        return self.run_batch_test(
            alpha_values=alpha_values,
            eta_values=eta_values,
            D_values=D_values,
            T_values=T_values,
            N_g_values=N_g_values,
            num_trials=num_trials,
            max_workers=4
        )
    
    def test_recommended_configs(self, recommendations: List[Dict], num_trials: int = 5) -> Dict:
        """
        Test recommended configurations with more trials
        """
        print("🎯 TESTING RECOMMENDED CONFIGURATIONS")
        print("=" * 50)
        
        configs_to_test = []
        for rec in recommendations:
            for trial in range(num_trials):
                configs_to_test.append({
                    **rec['config'],
                    'trial': trial,
                    'recommendation_type': rec['type']
                })
        
        print(f"Testing {len(configs_to_test)} configurations...")
        
        results = []
        for config in configs_to_test:
            result = self.test_single_configuration(config)
            results.append(result)
        
        return self.analyze_batch_results(results)
    
    def save_results(self, results: Dict, filename: str = None):
        """
        Save batch test results
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_test_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy(results)
        
        with open(filename, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"💾 Results saved to {filename}")

def main():
    """
    Main function to run batch convergence testing
    """
    print("🚀 F2CSA BATCH CONVERGENCE TESTER")
    print("=" * 80)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5, 
        num_constraints=3, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    # Initialize tester
    tester = BatchConvergenceTester(problem)
    
    # Run quick parameter sweep first
    print("Step 1: Quick parameter sweep...")
    quick_results = tester.quick_parameter_sweep()
    
    if quick_results['successful_configs'] > 0:
        print(f"\nFound {quick_results['successful_configs']} successful configurations!")
        
        # Test recommended configurations with more trials
        print("\nStep 2: Testing recommended configurations...")
        detailed_results = tester.test_recommended_configs(quick_results['recommendations'])
        
        # Save results
        tester.save_results(detailed_results)
        
        print("\n✅ Batch testing complete!")
    else:
        print("❌ No successful configurations found in quick sweep!")
        print("Consider adjusting parameter ranges or checking algorithm implementation.")

if __name__ == "__main__":
    main()
