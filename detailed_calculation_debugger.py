#!/usr/bin/env python3
"""
Detailed Calculation Debugger for F2CSA Algorithm 2
Adds comprehensive logging to identify sources of calculation instability
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
from f2csa_algorithm import F2CSAAlgorithm2Working
import time
import json
from datetime import datetime

class DetailedCalculationDebugger:
    """
    Detailed debugger that logs every calculation step to identify instability sources
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem, verbose: bool = True):
        self.problem = problem
        self.verbose = verbose
        self.algorithm1 = F2CSAAlgorithm1Final(problem)
        self.algorithm2 = F2CSAAlgorithm2Working(problem)
        self.debug_log = []
        
    def log_step(self, step_name: str, data: Dict, iteration: int = None):
        """
        Log a calculation step with detailed information
        """
        log_entry = {
            'timestamp': time.time(),
            'step': step_name,
            'iteration': iteration,
            'data': data
        }
        self.debug_log.append(log_entry)
        
        if self.verbose:
            print(f"[{step_name}] Iter {iteration}: {data}")
    
    def debug_algorithm1_step_by_step(self, x: torch.Tensor, alpha: float, N_g: int = 10) -> Dict:
        """
        Debug Algorithm 1 with detailed step-by-step logging
        """
        print(f"\n🔍 DEBUGGING ALGORITHM 1 STEP-BY-STEP")
        print("=" * 60)
        print(f"Input: x = {x}")
        print(f"Parameters: α = {alpha}, N_g = {N_g}")
        print(f"δ = α³ = {alpha**3:.6f}")
        print()
        
        delta = alpha**3
        step_data = {}
        
        # Step 1: Lower-level solution
        print("Step 1: Solving lower-level problem accurately...")
        start_time = time.time()
        
        try:
            y_star, lambda_star, info = self.algorithm1._solve_lower_level_accurate(x, alpha)
            solve_time = time.time() - start_time
            
            # Log lower-level solution details
            h_val = self.problem.constraints(x, y_star)
            violations = torch.clamp(h_val, min=0)
            max_violation = torch.max(violations).item()
            
            step_data['lower_level'] = {
                'y_star': y_star.detach().numpy().tolist(),
                'lambda_star': lambda_star.detach().numpy().tolist(),
                'constraint_violations': h_val.detach().numpy().tolist(),
                'max_violation': max_violation,
                'solve_time': solve_time,
                'solver_status': info.get('status', 'unknown'),
                'iterations': info.get('iterations', 0)
            }
            
            self.log_step("lower_level_solve", step_data['lower_level'])
            
            print(f"  ✅ y* = {y_star}")
            print(f"  ✅ λ* = {lambda_star}")
            print(f"  ✅ Max violation: {max_violation:.6f}")
            print(f"  ✅ Solve time: {solve_time:.4f}s")
            
        except Exception as e:
            step_data['lower_level'] = {'error': str(e)}
            self.log_step("lower_level_solve", step_data['lower_level'])
            print(f"  ❌ ERROR: {e}")
            return {'error': f"Lower-level solve failed: {e}"}
        
        # Step 2: Penalty minimizer
        print("\nStep 2: Computing penalty minimizer...")
        start_time = time.time()
        
        try:
            y_tilde = self.algorithm1._minimize_penalty_lagrangian(x, y_star, lambda_star, alpha, delta)
            penalty_time = time.time() - start_time
            
            # Compute gap
            gap = torch.norm(y_tilde - y_star).item()
            
            # Check constraint violations at y_tilde
            h_tilde = self.problem.constraints(x, y_tilde)
            violations_tilde = torch.clamp(h_tilde, min=0)
            max_violation_tilde = torch.max(violations_tilde).item()
            
            step_data['penalty_minimizer'] = {
                'y_tilde': y_tilde.detach().numpy().tolist(),
                'gap': gap,
                'constraint_violations': h_tilde.detach().numpy().tolist(),
                'max_violation': max_violation_tilde,
                'penalty_time': penalty_time,
                'gap_requirement_met': gap < 0.1
            }
            
            self.log_step("penalty_minimizer", step_data['penalty_minimizer'])
            
            print(f"  ✅ ỹ = {y_tilde}")
            print(f"  ✅ Gap ||ỹ - y*|| = {gap:.6f}")
            print(f"  ✅ Gap < 0.1: {'✅' if gap < 0.1 else '❌'}")
            print(f"  ✅ Max violation at ỹ: {max_violation_tilde:.6f}")
            print(f"  ✅ Penalty time: {penalty_time:.4f}s")
            
        except Exception as e:
            step_data['penalty_minimizer'] = {'error': str(e)}
            self.log_step("penalty_minimizer", step_data['penalty_minimizer'])
            print(f"  ❌ ERROR: {e}")
            return {'error': f"Penalty minimizer failed: {e}"}
        
        # Step 3: Hypergradient computation
        print(f"\nStep 3: Computing hypergradient with N_g = {N_g} samples...")
        start_time = time.time()
        
        try:
            hypergradient_samples = []
            sample_details = []
            
            for j in range(N_g):
                sample_start = time.time()
                
                # Sample fresh noise
                noise_upper, _ = self.problem._sample_instance_noise()
                
                # Create computational graph
                x_grad = x.clone().detach().requires_grad_(True)
                
                # Compute penalty Lagrangian
                L_val = self.algorithm1._compute_penalty_lagrangian(x_grad, y_tilde, y_star, lambda_star, alpha, delta)
                
                # Add upper-level objective
                f_val = self.problem.upper_objective(x_grad, y_tilde, noise_upper=noise_upper)
                total_val = f_val + L_val
                
                # Compute gradient
                grad_x = torch.autograd.grad(total_val, x_grad, create_graph=True, retain_graph=True)[0]
                hypergradient_samples.append(grad_x.detach())
                
                sample_time = time.time() - sample_start
                grad_norm = torch.norm(grad_x).item()
                
                sample_details.append({
                    'sample': j + 1,
                    'noise_upper': noise_upper.detach().numpy().tolist(),
                    'L_val': L_val.item(),
                    'f_val': f_val.item(),
                    'total_val': total_val.item(),
                    'grad_norm': grad_norm,
                    'computation_time': sample_time
                })
                
                if j < 3 or j % max(1, N_g // 5) == 0:  # Log first 3 and every 20%
                    print(f"    Sample {j+1}/{N_g}: grad_norm = {grad_norm:.6f}, time = {sample_time:.4f}s")
            
            # Average samples
            hypergradient = torch.stack(hypergradient_samples).mean(dim=0)
            hypergradient_norm = torch.norm(hypergradient).item()
            hypergradient_time = time.time() - start_time
            
            # Analyze sample variance
            sample_norms = [torch.norm(sample).item() for sample in hypergradient_samples]
            sample_std = np.std(sample_norms)
            sample_mean = np.mean(sample_norms)
            
            step_data['hypergradient'] = {
                'hypergradient': hypergradient.detach().numpy().tolist(),
                'hypergradient_norm': hypergradient_norm,
                'sample_norms': sample_norms,
                'sample_std': sample_std,
                'sample_mean': sample_mean,
                'computation_time': hypergradient_time,
                'sample_details': sample_details
            }
            
            self.log_step("hypergradient", step_data['hypergradient'])
            
            print(f"  ✅ Hypergradient norm: {hypergradient_norm:.6f}")
            print(f"  ✅ Sample std dev: {sample_std:.6f}")
            print(f"  ✅ Sample range: {min(sample_norms):.6f} - {max(sample_norms):.6f}")
            print(f"  ✅ Hypergradient time: {hypergradient_time:.4f}s")
            
        except Exception as e:
            step_data['hypergradient'] = {'error': str(e)}
            self.log_step("hypergradient", step_data['hypergradient'])
            print(f"  ❌ ERROR: {e}")
            return {'error': f"Hypergradient computation failed: {e}"}
        
        # Summary
        total_time = sum([
            step_data['lower_level']['solve_time'],
            step_data['penalty_minimizer']['penalty_time'],
            step_data['hypergradient']['computation_time']
        ])
        
        summary = {
            'total_time': total_time,
            'gap_requirement_met': step_data['penalty_minimizer']['gap_requirement_met'],
            'hypergradient_stable': step_data['hypergradient']['sample_std'] < 10.0,
            'overall_success': (
                step_data['penalty_minimizer']['gap_requirement_met'] and 
                step_data['hypergradient']['sample_std'] < 10.0
            )
        }
        
        step_data['summary'] = summary
        self.log_step("algorithm1_summary", summary)
        
        print(f"\n📊 ALGORITHM 1 SUMMARY:")
        print(f"  Total time: {total_time:.4f}s")
        print(f"  Gap < 0.1: {'✅' if summary['gap_requirement_met'] else '❌'}")
        print(f"  Hypergradient stable: {'✅' if summary['hypergradient_stable'] else '❌'}")
        print(f"  Overall success: {'✅' if summary['overall_success'] else '❌'}")
        
        return step_data
    
    def debug_algorithm2_step_by_step(self, x0: torch.Tensor, T: int, D: float, eta: float, 
                                    delta: float, alpha: float, N_g: int = 10) -> Dict:
        """
        Debug Algorithm 2 with detailed step-by-step logging
        """
        print(f"\n🔍 DEBUGGING ALGORITHM 2 STEP-BY-STEP")
        print("=" * 60)
        print(f"Input: x0 = {x0}")
        print(f"Parameters: T = {T}, D = {D}, η = {eta}, δ = {delta:.6f}, α = {alpha}, N_g = {N_g}")
        print()
        
        # Initialize
        x = x0.clone().detach()
        Delta = torch.zeros_like(x)
        
        iteration_data = []
        
        print("Starting Algorithm 2 optimization...")
        print("-" * 50)
        
        for t in range(1, T + 1):
            iter_start_time = time.time()
            
            # Sample s_t ~ Unif[0, 1]
            s_t = torch.rand(1, device=self.problem.device, dtype=self.problem.dtype).item()
            
            # Update x_t and z_t
            x_t = x + Delta
            z_t = x + s_t * Delta
            
            # Compute hypergradient using Algorithm 1
            print(f"\nIteration {t}/{T}:")
            print(f"  x_t = {x_t}")
            print(f"  z_t = {z_t}")
            print(f"  s_t = {s_t:.4f}")
            
            # Debug Algorithm 1 for this iteration
            algo1_debug = self.debug_algorithm1_step_by_step(z_t, alpha, N_g)
            
            if 'error' in algo1_debug:
                print(f"  ❌ Algorithm 1 failed: {algo1_debug['error']}")
                iteration_data.append({
                    'iteration': t,
                    'error': algo1_debug['error'],
                    'success': False
                })
                continue
            
            g_t = torch.tensor(algo1_debug['hypergradient']['hypergradient'], dtype=torch.float64)
            g_norm = torch.norm(g_t).item()
            
            # Update direction with clipping
            Delta_new = Delta - eta * g_t
            Delta_clipped = self.algorithm2.clip_D(Delta_new, D)
            
            # Log iteration details
            iter_data = {
                'iteration': t,
                'x_t': x_t.detach().numpy().tolist(),
                'z_t': z_t.detach().numpy().tolist(),
                's_t': s_t,
                'g_t': g_t.detach().numpy().tolist(),
                'g_norm': g_norm,
                'Delta_before': Delta.detach().numpy().tolist(),
                'Delta_after': Delta_clipped.detach().numpy().tolist(),
                'Delta_norm': torch.norm(Delta_clipped).item(),
                'clipping_applied': torch.norm(Delta_new).item() > D,
                'computation_time': time.time() - iter_start_time,
                'algorithm1_debug': algo1_debug
            }
            
            iteration_data.append(iter_data)
            self.log_step("algorithm2_iteration", iter_data, t)
            
            print(f"  ✅ g_t norm: {g_norm:.6f}")
            print(f"  ✅ Δ_t norm: {torch.norm(Delta_clipped).item():.6f}")
            print(f"  ✅ Clipping applied: {'Yes' if iter_data['clipping_applied'] else 'No'}")
            
            # Update for next iteration
            Delta = Delta_clipped
            x = x_t
            
            # Check for early convergence
            if t > 10 and g_norm < 1e-3:
                print(f"  🎯 Early convergence at iteration {t}")
                break
        
        # Analyze convergence
        grad_norms = [iter_data['g_norm'] for iter_data in iteration_data if 'g_norm' in iter_data]
        
        if len(grad_norms) >= 10:
            last_10_norms = np.array(grad_norms[-10:])
            std_dev = np.std(last_10_norms)
            mean_norm = np.mean(last_10_norms)
            range_norm = np.max(last_10_norms) - np.min(last_10_norms)
            
            converged = (std_dev < 2.0 and range_norm < 5.0 and grad_norms[-1] < 10.0)
        else:
            converged = False
            std_dev = float('inf')
            mean_norm = float('inf')
            range_norm = float('inf')
        
        summary = {
            'total_iterations': len(iteration_data),
            'final_grad_norm': grad_norms[-1] if grad_norms else float('inf'),
            'grad_norm_std': std_dev,
            'grad_norm_mean': mean_norm,
            'grad_norm_range': range_norm,
            'converged': converged,
            'successful_iterations': len([d for d in iteration_data if 'g_norm' in d]),
            'failed_iterations': len([d for d in iteration_data if 'error' in d])
        }
        
        self.log_step("algorithm2_summary", summary)
        
        print(f"\n📊 ALGORITHM 2 SUMMARY:")
        print(f"  Total iterations: {summary['total_iterations']}")
        print(f"  Successful iterations: {summary['successful_iterations']}")
        print(f"  Failed iterations: {summary['failed_iterations']}")
        print(f"  Final grad norm: {summary['final_grad_norm']:.6f}")
        print(f"  Grad norm std: {summary['grad_norm_std']:.6f}")
        print(f"  Grad norm range: {summary['grad_norm_range']:.6f}")
        print(f"  Converged: {'✅' if converged else '❌'}")
        
        return {
            'summary': summary,
            'iterations': iteration_data,
            'debug_log': self.debug_log
        }
    
    def identify_instability_sources(self, debug_results: Dict) -> Dict:
        """
        Analyze debug results to identify sources of calculation instability
        """
        print(f"\n🔍 IDENTIFYING INSTABILITY SOURCES")
        print("=" * 50)
        
        instability_analysis = {
            'gap_issues': [],
            'hypergradient_issues': [],
            'algorithm2_issues': [],
            'parameter_issues': [],
            'recommendations': []
        }
        
        # Analyze gap issues
        iterations = debug_results.get('iterations', [])
        for iter_data in iterations:
            if 'algorithm1_debug' in iter_data:
                algo1_debug = iter_data['algorithm1_debug']
                
                # Check gap issues
                if 'penalty_minimizer' in algo1_debug:
                    gap = algo1_debug['penalty_minimizer'].get('gap', float('inf'))
                    if gap >= 0.1:
                        instability_analysis['gap_issues'].append({
                            'iteration': iter_data['iteration'],
                            'gap': gap,
                            'issue': 'Gap >= 0.1 requirement not met'
                        })
                
                # Check hypergradient stability
                if 'hypergradient' in algo1_debug:
                    sample_std = algo1_debug['hypergradient'].get('sample_std', float('inf'))
                    if sample_std > 10.0:
                        instability_analysis['hypergradient_issues'].append({
                            'iteration': iter_data['iteration'],
                            'sample_std': sample_std,
                            'issue': 'High hypergradient sample variance'
                        })
        
        # Analyze Algorithm 2 convergence
        grad_norms = [iter_data.get('g_norm', float('inf')) for iter_data in iterations if 'g_norm' in iter_data]
        if grad_norms:
            grad_norm_std = np.std(grad_norms)
            if grad_norm_std > 5.0:
                instability_analysis['algorithm2_issues'].append({
                    'issue': 'High gradient norm variance',
                    'std': grad_norm_std,
                    'range': max(grad_norms) - min(grad_norms)
                })
        
        # Generate recommendations
        if instability_analysis['gap_issues']:
            instability_analysis['recommendations'].append({
                'issue': 'Gap stability',
                'recommendation': 'Reduce α or improve penalty minimizer convergence',
                'priority': 'HIGH'
            })
        
        if instability_analysis['hypergradient_issues']:
            instability_analysis['recommendations'].append({
                'issue': 'Hypergradient stability',
                'recommendation': 'Increase N_g or improve penalty Lagrangian computation',
                'priority': 'HIGH'
            })
        
        if instability_analysis['algorithm2_issues']:
            instability_analysis['recommendations'].append({
                'issue': 'Algorithm 2 convergence',
                'recommendation': 'Reduce learning rate η or improve gradient stability',
                'priority': 'MEDIUM'
            })
        
        # Print analysis
        print(f"Gap issues: {len(instability_analysis['gap_issues'])}")
        print(f"Hypergradient issues: {len(instability_analysis['hypergradient_issues'])}")
        print(f"Algorithm 2 issues: {len(instability_analysis['algorithm2_issues'])}")
        print(f"Recommendations: {len(instability_analysis['recommendations'])}")
        
        for rec in instability_analysis['recommendations']:
            print(f"  {rec['priority']}: {rec['issue']} - {rec['recommendation']}")
        
        return instability_analysis
    
    def save_debug_log(self, filename: str = None):
        """
        Save detailed debug log to file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detailed_debug_log_{timestamp}.json"
        
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
        
        debug_log_serializable = convert_numpy(self.debug_log)
        
        with open(filename, 'w') as f:
            json.dump(debug_log_serializable, f, indent=2)
        
        print(f"💾 Debug log saved to {filename}")

def main():
    """
    Main function to run detailed calculation debugging
    """
    print("🚀 F2CSA DETAILED CALCULATION DEBUGGER")
    print("=" * 80)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5, 
        num_constraints=3, 
        noise_std=0.1, 
        strong_convex=True
    )
    
    # Initialize debugger
    debugger = DetailedCalculationDebugger(problem, verbose=True)
    
    # Test parameters
    alpha = 0.1
    x0 = torch.randn(5, dtype=torch.float64)
    T = 20  # Reduced for detailed debugging
    D = 1.0
    eta = 0.001
    delta = alpha**3
    N_g = 10
    
    print(f"Test parameters:")
    print(f"  α = {alpha}, δ = {delta:.6f}")
    print(f"  T = {T}, D = {D}, η = {eta}")
    print(f"  N_g = {N_g}")
    print()
    
    # Debug Algorithm 1 first
    print("Step 1: Debugging Algorithm 1...")
    algo1_debug = debugger.debug_algorithm1_step_by_step(x0, alpha, N_g)
    
    if 'error' in algo1_debug:
        print(f"❌ Algorithm 1 debugging failed: {algo1_debug['error']}")
        return
    
    # Debug Algorithm 2
    print("\nStep 2: Debugging Algorithm 2...")
    algo2_debug = debugger.debug_algorithm2_step_by_step(x0, T, D, eta, delta, alpha, N_g)
    
    # Identify instability sources
    print("\nStep 3: Identifying instability sources...")
    instability_analysis = debugger.identify_instability_sources(algo2_debug)
    
    # Save debug log
    debugger.save_debug_log()
    
    print("\n✅ Detailed debugging complete!")

if __name__ == "__main__":
    main()
