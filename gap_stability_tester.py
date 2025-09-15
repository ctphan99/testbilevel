#!/usr/bin/env python3
"""
Gap Stability Tester for F2CSA Algorithm 2
Focuses on achieving gap < 0.1 and stable Algorithm 2 convergence
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
from f2csa_algorithm import F2CSAAlgorithm2Working
import time
import json
from datetime import datetime

class GapStabilityTester:
    """
    Tester focused on gap stability and Algorithm 2 convergence
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem):
        self.problem = problem
        self.algorithm1 = F2CSAAlgorithm1Final(problem)
        self.algorithm2 = F2CSAAlgorithm2Working(problem)
    
    def test_gap_stability(self, alpha: float, num_trials: int = 20) -> Dict:
        """
        Test gap stability for a given alpha value
        """
        print(f"\n🔍 TESTING GAP STABILITY for α = {alpha}")
        print("=" * 60)
        
        delta = alpha**3
        results = []
        
        print(f"Target: gap < 0.1 (δ = {delta:.6f})")
        print(f"Running {num_trials} trials...")
        print()
        
        for trial in range(num_trials):
            print(f"Trial {trial + 1}/{num_trials}:", end=" ")
            
            try:
                x_test = torch.randn(self.problem.dim, dtype=torch.float64)
                
                # Step 1: Get accurate lower-level solution
                y_star, lambda_star, info = self.algorithm1._solve_lower_level_accurate(x_test, alpha)
                
                # Step 2: Compute penalty minimizer
                y_tilde = self.algorithm1._minimize_penalty_lagrangian(x_test, y_star, lambda_star, alpha, delta)
                
                # Step 3: Compute gap
                gap = torch.norm(y_tilde - y_star).item()
                
                # Step 4: Test hypergradient computation
                hypergradient = self.algorithm1.oracle_sample(x_test, alpha, 10)
                grad_norm = torch.norm(hypergradient).item()
                
                # Check constraint violations
                h_val = self.problem.constraints(x_test, y_tilde)
                max_violation = torch.max(torch.clamp(h_val, min=0)).item()
                
                success = gap < 0.1 and grad_norm < 100
                
                results.append({
                    'trial': trial + 1,
                    'gap': gap,
                    'grad_norm': grad_norm,
                    'max_violation': max_violation,
                    'success': success,
                    'y_star': y_star.detach().numpy(),
                    'y_tilde': y_tilde.detach().numpy(),
                    'lambda_star': lambda_star.detach().numpy()
                })
                
                status = "✅" if success else "❌"
                print(f"{status} gap={gap:.4f}, grad={grad_norm:.2f}")
                
            except Exception as e:
                results.append({
                    'trial': trial + 1,
                    'gap': float('inf'),
                    'grad_norm': float('inf'),
                    'max_violation': float('inf'),
                    'success': False,
                    'error': str(e)
                })
                print(f"❌ ERROR: {e}")
        
        # Analyze results
        successful_trials = [r for r in results if r['success']]
        success_rate = len(successful_trials) / len(results)
        
        if successful_trials:
            gaps = [r['gap'] for r in successful_trials]
            grad_norms = [r['grad_norm'] for r in successful_trials]
            
            avg_gap = np.mean(gaps)
            gap_std = np.std(gaps)
            avg_grad_norm = np.mean(grad_norms)
            grad_std = np.std(grad_norms)
        else:
            avg_gap = float('inf')
            gap_std = float('inf')
            avg_grad_norm = float('inf')
            grad_std = float('inf')
        
        print(f"\n📊 GAP STABILITY ANALYSIS:")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Average gap: {avg_gap:.6f}")
        print(f"  Gap std dev: {gap_std:.6f}")
        print(f"  Average grad norm: {avg_grad_norm:.6f}")
        print(f"  Grad norm std: {grad_std:.6f}")
        print(f"  Gap < 0.1: {'✅' if avg_gap < 0.1 else '❌'}")
        
        return {
            'alpha': alpha,
            'delta': delta,
            'success_rate': success_rate,
            'avg_gap': avg_gap,
            'gap_std': gap_std,
            'avg_grad_norm': avg_grad_norm,
            'grad_std': grad_std,
            'trials': results,
            'gap_requirement_met': avg_gap < 0.1,
            'stable': gap_std < 0.05 and grad_std < 10.0
        }
    
    def test_algorithm2_convergence(self, alpha: float, eta: float, D: float, T: int, N_g: int, num_trials: int = 10) -> Dict:
        """
        Test Algorithm 2 convergence for given parameters
        """
        print(f"\n🔍 TESTING ALGORITHM 2 CONVERGENCE")
        print("=" * 60)
        print(f"Parameters: α={alpha}, η={eta:.1e}, D={D}, T={T}, N_g={N_g}")
        print(f"Running {num_trials} trials...")
        print()
        
        delta = alpha**3
        results = []
        
        for trial in range(num_trials):
            print(f"Trial {trial + 1}/{num_trials}:", end=" ")
            
            try:
                x0 = torch.randn(self.problem.dim, dtype=torch.float64)
                
                # Run Algorithm 2
                result = self.algorithm2.optimize(x0, T, D, eta, delta, alpha, N_g)
                
                # Analyze convergence
                grad_norms = result['grad_norms']
                final_grad_norm = grad_norms[-1] if grad_norms else float('inf')
                
                # Check for convergence
                if len(grad_norms) >= 10:
                    last_10_norms = np.array(grad_norms[-10:])
                    std_dev = np.std(last_10_norms)
                    mean_norm = np.mean(last_10_norms)
                    range_norm = np.max(last_10_norms) - np.min(last_10_norms)
                    
                    # Convergence criteria
                    converged = (std_dev < 2.0 and range_norm < 5.0 and final_grad_norm < 10.0)
                else:
                    converged = False
                    std_dev = float('inf')
                    mean_norm = float('inf')
                    range_norm = float('inf')
                
                results.append({
                    'trial': trial + 1,
                    'final_grad_norm': final_grad_norm,
                    'grad_std': std_dev,
                    'grad_mean': mean_norm,
                    'grad_range': range_norm,
                    'converged': converged,
                    'grad_norms': grad_norms,
                    'iterations': len(grad_norms)
                })
                
                status = "✅" if converged else "❌"
                print(f"{status} final_grad={final_grad_norm:.4f}, std={std_dev:.4f}")
                
            except Exception as e:
                results.append({
                    'trial': trial + 1,
                    'final_grad_norm': float('inf'),
                    'grad_std': float('inf'),
                    'grad_mean': float('inf'),
                    'grad_range': float('inf'),
                    'converged': False,
                    'error': str(e)
                })
                print(f"❌ ERROR: {e}")
        
        # Analyze results
        successful_trials = [r for r in results if r['converged']]
        success_rate = len(successful_trials) / len(results)
        
        if successful_trials:
            final_grad_norms = [r['final_grad_norm'] for r in successful_trials]
            grad_stds = [r['grad_std'] for r in successful_trials]
            
            avg_final_grad = np.mean(final_grad_norms)
            avg_grad_std = np.mean(grad_stds)
        else:
            avg_final_grad = float('inf')
            avg_grad_std = float('inf')
        
        print(f"\n📊 ALGORITHM 2 CONVERGENCE ANALYSIS:")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Average final grad norm: {avg_final_grad:.6f}")
        print(f"  Average grad std: {avg_grad_std:.6f}")
        print(f"  Converged: {'✅' if success_rate > 0.8 else '❌'}")
        
        return {
            'alpha': alpha,
            'eta': eta,
            'D': D,
            'T': T,
            'N_g': N_g,
            'success_rate': success_rate,
            'avg_final_grad': avg_final_grad,
            'avg_grad_std': avg_grad_std,
            'trials': results,
            'converged': success_rate > 0.8
        }
    
    def find_stable_parameters(self, 
                             alpha_values: List[float] = [0.1, 0.05, 0.01, 0.005, 0.001],
                             eta_values: List[float] = [1e-5, 1e-4, 1e-3, 1e-2],
                             D_values: List[float] = [0.1, 0.5, 1.0, 2.0],
                             T_values: List[int] = [50, 100, 200],
                             N_g_values: List[int] = [5, 10, 20, 50]) -> Dict:
        """
        Find stable parameters through systematic testing
        """
        print("🔍 FINDING STABLE PARAMETERS")
        print("=" * 60)
        
        # Step 1: Test gap stability for different alpha values
        print("Step 1: Testing gap stability...")
        gap_results = {}
        
        for alpha in alpha_values:
            gap_result = self.test_gap_stability(alpha, num_trials=10)
            gap_results[alpha] = gap_result
            
            if gap_result['gap_requirement_met'] and gap_result['stable']:
                print(f"  ✅ α = {alpha}: Gap stable!")
            else:
                print(f"  ❌ α = {alpha}: Gap not stable")
        
        # Find best alpha values
        stable_alphas = [alpha for alpha, result in gap_results.items() 
                        if result['gap_requirement_met'] and result['stable']]
        
        if not stable_alphas:
            print("❌ No stable alpha values found!")
            return {'error': 'No stable alpha values found'}
        
        print(f"\nStable alpha values: {stable_alphas}")
        
        # Step 2: Test Algorithm 2 convergence for stable alphas
        print("\nStep 2: Testing Algorithm 2 convergence...")
        convergence_results = {}
        
        for alpha in stable_alphas:
            print(f"\nTesting α = {alpha}:")
            convergence_results[alpha] = {}
            
            for eta in eta_values:
                for D in D_values:
                    for T in T_values:
                        for N_g in N_g_values:
                            config_key = f"eta={eta:.1e},D={D},T={T},N_g={N_g}"
                            
                            # Quick test with fewer trials
                            conv_result = self.test_algorithm2_convergence(alpha, eta, D, T, N_g, num_trials=3)
                            convergence_results[alpha][config_key] = conv_result
                            
                            if conv_result['converged']:
                                print(f"  ✅ {config_key}: Converged!")
                            else:
                                print(f"  ❌ {config_key}: Not converged")
        
        # Step 3: Find best configurations
        print("\nStep 3: Finding best configurations...")
        best_configs = []
        
        for alpha in stable_alphas:
            for config_key, result in convergence_results[alpha].items():
                if result['converged']:
                    # Parse configuration
                    parts = config_key.split(',')
                    eta = float(parts[0].split('=')[1])
                    D = float(parts[1].split('=')[1])
                    T = int(parts[2].split('=')[1])
                    N_g = int(parts[3].split('=')[1])
                    
                    # Score configuration
                    gap_score = gap_results[alpha]['avg_gap']
                    grad_score = result['avg_final_grad']
                    stability_score = result['avg_grad_std']
                    
                    total_score = 1.0 / (1.0 + gap_score) + 1.0 / (1.0 + grad_score) + 1.0 / (1.0 + stability_score)
                    
                    best_configs.append({
                        'alpha': alpha,
                        'eta': eta,
                        'D': D,
                        'T': T,
                        'N_g': N_g,
                        'gap_score': gap_score,
                        'grad_score': grad_score,
                        'stability_score': stability_score,
                        'total_score': total_score,
                        'gap_result': gap_results[alpha],
                        'conv_result': result
                    })
        
        # Sort by total score
        best_configs.sort(key=lambda x: x['total_score'], reverse=True)
        
        print(f"\n🏆 TOP 10 STABLE CONFIGURATIONS:")
        print("-" * 100)
        print(f"{'Rank':<4} {'α':<8} {'η':<10} {'D':<6} {'T':<4} {'N_g':<4} {'Gap':<8} {'Grad':<8} {'Std':<8} {'Score':<8}")
        print("-" * 100)
        
        for i, config in enumerate(best_configs[:10]):
            print(f"{i+1:<4} {config['alpha']:<8.3f} {config['eta']:<10.1e} "
                  f"{config['D']:<6.1f} {config['T']:<4} {config['N_g']:<4} "
                  f"{config['gap_score']:<8.4f} {config['grad_score']:<8.4f} "
                  f"{config['stability_score']:<8.4f} {config['total_score']:<8.4f}")
        
        return {
            'gap_results': gap_results,
            'convergence_results': convergence_results,
            'best_configs': best_configs,
            'stable_alphas': stable_alphas,
            'total_stable_configs': len(best_configs)
        }
    
    def validate_best_configuration(self, config: Dict, num_trials: int = 20) -> Dict:
        """
        Validate the best configuration with more trials
        """
        print(f"\n🎯 VALIDATING BEST CONFIGURATION")
        print("=" * 60)
        print(f"Configuration: α={config['alpha']}, η={config['eta']:.1e}, D={config['D']}, T={config['T']}, N_g={config['N_g']}")
        print(f"Running {num_trials} validation trials...")
        print()
        
        # Test gap stability
        gap_result = self.test_gap_stability(config['alpha'], num_trials)
        
        # Test Algorithm 2 convergence
        conv_result = self.test_algorithm2_convergence(
            config['alpha'], config['eta'], config['D'], 
            config['T'], config['N_g'], num_trials
        )
        
        # Overall validation
        overall_success = (gap_result['gap_requirement_met'] and 
                          gap_result['stable'] and 
                          conv_result['converged'])
        
        print(f"\n📊 VALIDATION RESULTS:")
        print(f"  Gap requirement met: {'✅' if gap_result['gap_requirement_met'] else '❌'}")
        print(f"  Gap stable: {'✅' if gap_result['stable'] else '❌'}")
        print(f"  Algorithm 2 converged: {'✅' if conv_result['converged'] else '❌'}")
        print(f"  Overall success: {'✅' if overall_success else '❌'}")
        
        return {
            'config': config,
            'gap_result': gap_result,
            'conv_result': conv_result,
            'overall_success': overall_success,
            'validation_trials': num_trials
        }
    
    def save_results(self, results: Dict, filename: str = None):
        """
        Save test results
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gap_stability_results_{timestamp}.json"
        
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
    Main function to run gap stability testing
    """
    print("🚀 F2CSA GAP STABILITY TESTER")
    print("=" * 80)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5, 
        num_constraints=3, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    # Initialize tester
    tester = GapStabilityTester(problem)
    
    # Find stable parameters
    print("Step 1: Finding stable parameters...")
    results = tester.find_stable_parameters()
    
    if 'error' in results:
        print(f"❌ {results['error']}")
        return
    
    # Validate best configuration
    if results['best_configs']:
        print(f"\nStep 2: Validating best configuration...")
        best_config = results['best_configs'][0]
        validation = tester.validate_best_configuration(best_config)
        
        # Save results
        tester.save_results(results)
        
        print("\n✅ Gap stability testing complete!")
    else:
        print("❌ No stable configurations found!")

if __name__ == "__main__":
    main()
