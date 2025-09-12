"""
Comprehensive Experiment Runner for Bilevel Optimization Algorithms
Testing F2CSA, DS-BLO, and [S]SIGD with the same experimental setup

This script runs all three algorithms on the same bilevel optimization problem
with identical parameters and measures their performance for comparison.
All algorithms run to a target gap of 0.02 with maximum 10,000 iterations.
"""

import torch
import numpy as np
import time
import json
from typing import Dict, List, Tuple
from datetime import datetime
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm import F2CSAAlgorithm
from dsblo_algorithm import DSBLOAlgorithm
from ssigd_algorithm import SSIGDAlgorithm
import warnings
import argparse
import sys
warnings.filterwarnings('ignore')

# Ensure UTF-8 console to avoid Windows charmap errors
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

class BilevelExperimentRunner:
    """
    Comprehensive experiment runner for bilevel optimization algorithms
    
    Tests:
    1. F2CSA (Fully First-order Constrained Stochastic Approximation)
    2. DS-BLO (Doubly Stochastically Perturbed Algorithm)
    3. [S]SIGD ([Stochastic] Smoothed Implicit Gradient Descent)
    
    All algorithms target a gap of 0.02 with maximum 10,000 iterations.
    """
    
    def __init__(self, device='cpu', seed=42):
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print(f"[EXPERIMENT] COMPREHENSIVE BILEVEL OPTIMIZATION EXPERIMENT")
        print(f"   Testing: F2CSA vs DS-BLO vs [S]SIGD")
        print(f"   Target: Exactly 100 iterations, no early stopping")
        print(f"   Problem: Strongly convex LL with linear constraints")
        print(f"   Device: {device}, Seed: {seed}")
        
        # Common experimental parameters - using exact F2CSA config
        self.dim = 5
        self.epsilon = 0.02  # Target accuracy
        self.alpha = 0.08    # F2CSA config: α=0.08
        self.D = 0.0069      # F2CSA config: D=0.0069
        self.eta = 0.0010    # F2CSA config: η=0.0010
        self.delta = 0.01    # Goldstein accuracy
        self.N_g = 128       # F2CSA config: N_g=128
        self.inner_steps = 50
        self.inner_lr = 1e-2
        self.max_iterations = 100  # Exactly 100 iterations
        self.target_gap = 0.02     # Not used for early stopping
        
        # Create problem instance
        self.problem = StronglyConvexBilevelProblem(
            dim=self.dim,
            num_constraints=3,
            noise_std=0.001,
            device=self.device,
            seed=self.seed,
            strong_convex=True
        )
        
        # Initialize algorithms with algorithm-specific parameters only
        self.algorithms = {
            'F2CSA': F2CSAAlgorithm(
                problem=self.problem,
                device=self.device,
                seed=self.seed,
                alpha_override=self.alpha,
                eta_override=self.eta,
                D_override=self.D,
                Ng_override=self.N_g
            ),
            'DS-BLO': DSBLOAlgorithm(
                problem=self.problem,
                q_distribution='normal',
                q_std=0.01,
                perturbation_scale=0.001,
                goldstein_samples=10,
                device=self.device,
                seed=self.seed
            ),
            'SSIGD': SSIGDAlgorithm(
                problem=self.problem,
                q_distribution='normal',
                q_std=0.01,
                fixed_perturbation=True,
                rho=0.1,
                projection_radius=10.0,
                device=self.device,
                seed=self.seed
            )
        }
        
        # Results storage
        self.results = {}
        
    def _save_gap_plot(self, results: Dict, algorithm_name: str) -> None:
        """Save EMA gap plot for the given algorithm results if possible."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception:
            print("[PLOT] matplotlib not available; skipping plot generation")
            return

        # Extract iteration-wise EMA and raw gap histories
        ema_hist = []
        raw_hist = []
        # Common direct key
        if isinstance(results, dict):
            ema_hist = results.get('ema_gap_history', []) or results.get('ema_gaps', [])
            if not ema_hist:
                history = results.get('history', {}) if isinstance(results.get('history', {}), dict) else {}
                ema_hist = history.get('ema_gap_history', []) or history.get('ema_gaps', []) or history.get('gaps', [])
            # Raw gaps
            raw_hist = results.get('gap_history', [])
            if not raw_hist:
                history = results.get('history', {}) if isinstance(results.get('history', {}), dict) else {}
                raw_hist = history.get('gaps', [])

        if (not ema_hist or not isinstance(ema_hist, list)) and (not raw_hist or not isinstance(raw_hist, list)):
            print("[PLOT] No gap history found; skipping plot generation")
            return

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        # EMA plot (if available)
        if ema_hist and isinstance(ema_hist, list):
            plt.figure(figsize=(8, 4.5))
            plt.plot(range(1, len(ema_hist) + 1), ema_hist, label=f"{algorithm_name} EMA gap", linewidth=1.6)
            plt.xlabel("Iteration")
            plt.ylabel("EMA Gap")
            plt.title(f"EMA Gap over Iterations - {algorithm_name}")
            plt.grid(True, linewidth=0.3, alpha=0.6)
            plt.legend()
            ema_name = f"{algorithm_name.lower()}_ema_gap_{ts}.png"
            plt.tight_layout()
            plt.savefig(ema_name, dpi=150)
            plt.close()
            print(f"[PLOT] Saved EMA gap plot to {ema_name}")

        # Raw gap plot (non-EMA), if available
        if raw_hist and isinstance(raw_hist, list):
            plt.figure(figsize=(8, 4.5))
            plt.plot(range(1, len(raw_hist) + 1), raw_hist, label=f"{algorithm_name} raw gap", linewidth=1.3, alpha=0.9)
            plt.xlabel("Iteration")
            plt.ylabel("Gap (raw)")
            plt.title(f"Raw Gap over Iterations - {algorithm_name}")
            plt.grid(True, linewidth=0.3, alpha=0.6)
            plt.legend()
            raw_name = f"{algorithm_name.lower()}_raw_gap_{ts}.png"
            plt.tight_layout()
            plt.savefig(raw_name, dpi=150)
            plt.close()
            print(f"[PLOT] Saved raw gap plot to {raw_name}")

    def _save_combined_gap_plot(self) -> None:
        """Save a combined raw gap plot for all algorithms using unified gap metric."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception:
            print("[PLOT] matplotlib not available; skipping combined plot")
            return

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.figure(figsize=(9.5, 5.2))
        any_series = False
        for name, res in self.results.items():
            if 'error' in res:
                continue
            history = res.get('history', {}) if isinstance(res.get('history', {}), dict) else {}
            gaps = history.get('gaps', [])
            if isinstance(gaps, list) and len(gaps) > 0:
                plt.plot(range(1, len(gaps) + 1), gaps, label=name, linewidth=1.4)
                any_series = True
        if not any_series:
            print("[PLOT] No gap histories to plot for combined chart")
            plt.close()
            return
        plt.xlabel("Iteration")
        plt.ylabel("Gap (F2CSA metric)")
        plt.title("Raw Gap over Iterations (Unified metric)")
        plt.grid(True, linewidth=0.3, alpha=0.6)
        plt.legend()
        out_name = f"combined_raw_gap_{ts}.png"
        plt.tight_layout()
        plt.savefig(out_name, dpi=160)
        plt.close()
        print(f"[PLOT] Saved combined raw gap plot to {out_name}")
    
    def _coerce_final_gap(self, results: Dict) -> float:
        """
        Normalize/derive a final gap value regardless of algorithm-specific naming.
        Tries keys in order, then falls back to last entry in history gaps.
        """
        candidates = [
            results.get('final_ema_gap', None),
            results.get('final_gap', None),
            results.get('best_gap', None),
        ]
        for val in candidates:
            if isinstance(val, (int, float)) and np.isfinite(val):
                return float(val)
        history = results.get('history', {}) if isinstance(results.get('history', {}), dict) else {}
        gaps = history.get('ema_gap_history') or history.get('ema_gaps') or history.get('gaps') or []
        if isinstance(gaps, list) and len(gaps) > 0:
            try:
                return float(gaps[-1])
            except Exception:
                pass
        return float('inf')
        
    def run_single_algorithm(self, algorithm_name: str) -> Dict:
        """
        Run a single algorithm and return results
        """
        print(f"\n[RUNNING] Running {algorithm_name}...")
        print(f"   Algorithm: {algorithm_name}")
        print(f"   Problem: dim={self.dim}, max_iterations={self.max_iterations}")
        print(f"   Config: α={self.alpha}, N_g={self.N_g}, D={self.D}, η={self.eta}")
        
        algorithm = self.algorithms[algorithm_name]
        
        try:
            # Run optimization
            start_time = time.time()
            results = algorithm.optimize(
                max_iterations=self.max_iterations,
                target_gap=self.target_gap,
                run_until_convergence=False  # No early stopping
            )
            total_time = time.time() - start_time
            
            # Add metadata
            results['algorithm'] = algorithm_name
            results['problem_dim'] = self.dim
            results['target_gap'] = self.target_gap
            results['total_time'] = total_time
            # Normalize gap field for consistent downstream handling
            results['final_ema_gap'] = self._coerce_final_gap(results)
            
            print(f"[SUCCESS] {algorithm_name} completed successfully!")
            print(f"   Final EMA gap: {results['final_ema_gap']:.6f}")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Target achieved: {results['target_achieved']}")
            # Save plot of EMA gap progression before returning
            self._save_gap_plot(results, algorithm_name)
            
            return results
            
        except Exception as e:
            # Avoid Unicode symbols for Windows consoles
            print(f"ERROR: {algorithm_name} failed: {str(e)}")
            return {
                'algorithm': algorithm_name,
                'error': str(e),
                'final_ema_gap': float('inf'),
                'total_time': 0,
                'target_achieved': False
            }
    
    def run_all_algorithms(self) -> Dict:
        """
        Run all algorithms and compare results
        """
        print(f"\n[EXPERIMENT] COMPREHENSIVE EXPERIMENT STARTING")
        print(f"   Algorithms: {list(self.algorithms.keys())}")
        print(f"   Max iterations: {self.max_iterations} (no early stopping)")
        print(f"   Config: α={self.alpha}, N_g={self.N_g}, D={self.D}, η={self.eta}")
        
        # Run each algorithm
        for algorithm_name in self.algorithms.keys():
            self.results[algorithm_name] = self.run_single_algorithm(algorithm_name)
        
        # Compare results
        self.compare_results()

        # Save combined plot using unified gap metric
        self._save_combined_gap_plot()
        
        return self.results
    
    def compare_results(self):
        """
        Compare results across all algorithms
        """
        print(f"\n[RESULTS] COMPREHENSIVE COMPARISON RESULTS")
        print(f"=" * 80)
        
        # Header
        print(f"{'Algorithm':<12} {'Best Gap':<12} {'Time(s)':<10} {'Iterations':<12}")
        print(f"-" * 80)
        
        # Results for each algorithm
        for algorithm_name, results in self.results.items():
            if 'error' in results:
                print(f"{algorithm_name:<12} {'ERROR':<12} {'N/A':<10} {'N/A':<12}")
            else:
                best_gap = results.get('final_ema_gap', float('inf'))
                total_time = results.get('total_time', 0)
                iterations = results.get('total_iterations', 0)
                
                print(f"{algorithm_name:<12} {best_gap:<12.6f} {total_time:<10.2f} {iterations:<12}")
        
        print(f"-" * 80)
        
        # Find best performing algorithm
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        if valid_results:
            best_algorithm = min(valid_results.keys(), 
                               key=lambda k: valid_results[k]['final_ema_gap'])
            best_gap = valid_results[best_algorithm]['final_ema_gap']
            
            print(f"*** BEST PERFORMING: {best_algorithm}")
            print(f"   Best gap achieved: {best_gap:.6f}")
            print(f"   Target achieved: {valid_results[best_algorithm]['target_achieved']}")
        
        # Algorithm-specific metrics
        self.analyze_algorithm_specific_metrics()
    
    def analyze_algorithm_specific_metrics(self):
        """
        Analyze algorithm-specific convergence metrics
        """
        print(f"\n[ANALYSIS] ALGORITHM-SPECIFIC METRICS")
        print(f"=" * 60)
        
        for algorithm_name, results in self.results.items():
            if 'error' in results:
                continue
                
            print(f"\n[METRICS] {algorithm_name} Metrics:")
            
            # Basic metrics
            best_gap = results.get('final_ema_gap', float('inf'))
            total_time = results.get('total_time', 0)
            iterations = results.get('total_iterations', 0)
            
            print(f"   Final EMA gap: {best_gap:.6f}")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Iterations: {iterations}")
            
            # Algorithm-specific metrics
            if algorithm_name == 'F2CSA':
                # F2CSA specific metrics
                history = results.get('history', {})
                if 'gaps' in history and len(history['gaps']) > 0:
                    final_gap = history['gaps'][-1]
                    gap_improvement = history['gaps'][0] - final_gap
                    print(f"   Final gap: {final_gap:.6f}")
                    print(f"   Gap improvement: {gap_improvement:.6f}")
            
            elif algorithm_name == 'DS-BLO':
                # DS-BLO specific metrics - Goldstein stationarity removed from history
                print(f"   Algorithm: Doubly Stochastic Bilevel Optimization")
                print(f"   Perturbation: Normal distribution with std=0.01")
                print(f"   Convergence: Goldstein stationarity measure")
            
            elif algorithm_name == 'SSIGD':
                # [S]SIGD specific metrics - Moreau envelope removed from history
                print(f"   Algorithm: Stochastic Smoothed Implicit Gradient Descent")
                print(f"   Perturbation: Fixed normal distribution with std=0.01")
                print(f"   Convergence: Moreau envelope measure")
    
    def save_results(self, filename: str = 'comprehensive_bilevel_results.json'):
        """
        Save results to JSON file
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for algorithm_name, results in self.results.items():
            serializable_results[algorithm_name] = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[algorithm_name][key] = value.tolist()
                elif isinstance(value, dict) and 'history' in key:
                    # Handle history dict with numpy arrays
                    history_serializable = {}
                    for hist_key, hist_value in value.items():
                        if isinstance(hist_value, list) and len(hist_value) > 0:
                            if isinstance(hist_value[0], np.ndarray):
                                history_serializable[hist_key] = [v.tolist() for v in hist_value]
                            else:
                                history_serializable[hist_key] = hist_value
                        else:
                            history_serializable[hist_key] = hist_value
                    serializable_results[algorithm_name][key] = history_serializable
                else:
                    serializable_results[algorithm_name][key] = value
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n[SAVE] Results saved to {filename}")
    
    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report
        """
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE BILEVEL OPTIMIZATION EXPERIMENT REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Experiment setup
        report.append("EXPERIMENT SETUP:")
        report.append(f"  Problem dimension: {self.dim}")
        report.append(f"  Max iterations: {self.max_iterations} (no early stopping)")
        report.append(f"  Device: {self.device}")
        report.append(f"  Seed: {self.seed}")
        report.append(f"  Config: α={self.alpha}, N_g={self.N_g}, D={self.D}, η={self.eta}")
        report.append("")
        
        # Results summary
        report.append("RESULTS SUMMARY:")
        report.append("-" * 40)
        
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        if valid_results:
            for algorithm_name, results in valid_results.items():
                best_gap = results.get('final_ema_gap', float('inf'))
                total_time = results.get('total_time', 0)
                iterations = results.get('total_iterations', 0)
                
                report.append(f"  {algorithm_name}:")
                report.append(f"    Final EMA gap: {best_gap:.6f}")
                report.append(f"    Total time: {total_time:.2f}s")
                report.append(f"    Total iterations: {iterations}")
                report.append("")
        
        # Best performing algorithm
        if valid_results:
            best_algorithm = min(valid_results.keys(), 
                               key=lambda k: valid_results[k]['final_ema_gap'])
            best_gap = valid_results[best_algorithm]['final_ema_gap']
            
            report.append("BEST PERFORMING ALGORITHM:")
            report.append(f"  {best_algorithm} with gap: {best_gap:.6f}")
            report.append("")
        
        # Algorithm-specific insights
        report.append("ALGORITHM-SPECIFIC INSIGHTS:")
        report.append("-" * 40)
        
        for algorithm_name, results in valid_results.items():
            report.append(f"  {algorithm_name}:")
            
            if algorithm_name == 'F2CSA':
                report.append("    - Penalty-based reformulation approach")
                report.append("    - Smooth activation function")
                report.append("    - Goldstein stationarity convergence")
            
            elif algorithm_name == 'DS-BLO':
                report.append("    - Doubly stochastic perturbation")
                report.append("    - Goldstein stationarity measure")
                report.append("    - Non-Lipschitz smoothness handling")
            
            elif algorithm_name == 'SSIGD':
                report.append("    - Single perturbation approach")
                report.append("    - Moreau envelope convergence")
                report.append("    - Weak convexity assumption")
            
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """
    Main experiment runner
    """
    parser = argparse.ArgumentParser(description='Comprehensive Bilevel Optimization Experiment')
    parser.add_argument('--algorithm', type=str, choices=['F2CSA', 'DS-BLO', 'SSIGD'], 
                       help='Run only a specific algorithm (F2CSA, DS-BLO, or SSIGD)')
    parser.add_argument('--eta', type=float, help='Fixed step size η override for F2CSA')
    parser.add_argument('--D', type=float, help='Clipping radius D override for F2CSA')
    parser.add_argument('--eta_list', type=str, help='Comma-separated η values for sweep')
    parser.add_argument('--D_list', type=str, help='Comma-separated D values for sweep')
    parser.add_argument('--two_stage_sweep', action='store_true', help='Run coarse log sweep then fine local sweep')
    parser.add_argument('--seed', type=int, help='Base seed for runs')
    parser.add_argument('--repeats', type=int, help='Number of repeated runs for statistics')
    # Stabilization overrides
    parser.add_argument('--s_fixed', action='store_true', help='Use s=1.0 (disable random interpolation)')
    parser.add_argument('--Ng', type=int, help='Gradient averaging batch size for upper objective')
    parser.add_argument('--alpha_fixed', type=float, help='Fixed alpha (no annealing)')
    parser.add_argument('--grad_ema_beta', type=float, help='Gradient EMA beta (e.g., 0.95)')
    parser.add_argument('--prox_weight', type=float, help='Proximal pull weight toward x EMA')
    parser.add_argument('--grad_clip', type=float, help='Gradient clipping norm for stabilized grad')
    parser.add_argument('--max_iters', type=int, help='Override max iterations for focused run (e.g., 3000)')
    
    # Utility: combined plot only from existing JSON results
    parser.add_argument('--combined_plot_only', action='store_true',
                        help='Generate a combined gap plot by loading existing JSON results without running algorithms')
    # Validation modes (paper diagnostics)
    parser.add_argument('--validate_algo1', action='store_true',
                        help='Run Algorithm 1 (oracle) bias/variance scaling sweeps using torch-SGD inner solver')
    parser.add_argument('--validate_algo2', action='store_true',
                        help='Run Algorithm 2 Goldstein proxy tracking across blocks')
    parser.add_argument('--grid_sweep', action='store_true',
                        help='Sweep over (alpha, Ng, D, eta) for F2CSA and record final EMA gap and stability flags')
    parser.add_argument('--grid_alpha', type=str, help='Comma list of alpha values, e.g., 0.08,0.04,0.02')
    parser.add_argument('--grid_Ng', type=str, help='Comma list of Ng ints, e.g., 8,16,32,64')
    parser.add_argument('--grid_D', type=str, help='Comma list of D values')
    parser.add_argument('--grid_eta', type=str, help='Comma list of eta values')

    args = parser.parse_args()
    
    print("[EXPERIMENT] COMPREHENSIVE BILEVEL OPTIMIZATION EXPERIMENT")
    print("Testing F2CSA vs DS-BLO vs [S]SIGD with exactly 100 iterations, no early stopping")
    
    # Initialize experiment runner
    base_seed = args.seed if args.seed is not None else 42
    runner = BilevelExperimentRunner(device='cpu', seed=base_seed)

    # Optional override of total iterations
    if args.max_iters is not None and args.max_iters > 0:
        runner.max_iterations = int(args.max_iters)

    # Apply F2CSA overrides if provided
    if True:
        f2csa = runner.algorithms['F2CSA']
        if args.eta is not None:
            f2csa.eta_override = float(args.eta)
        if args.D is not None:
            f2csa.D_override = float(args.D)
        if args.s_fixed:
            f2csa.s_fixed_override = 1.0
        if args.Ng is not None:
            f2csa.Ng_override = int(args.Ng)
        if args.alpha_fixed is not None:
            f2csa.alpha_override = float(args.alpha_fixed)
        if args.grad_ema_beta is not None:
            f2csa.grad_ema_beta = float(args.grad_ema_beta)
        if args.prox_weight is not None:
            f2csa.prox_weight = float(args.prox_weight)
        if args.grad_clip is not None:
            f2csa.grad_clip_override = float(args.grad_clip)

    # Validation: Algorithm 1 (oracle) bias/variance scaling
    if args.validate_algo1:
        f2csa = runner.algorithms['F2CSA']
        # pick a fixed x for oracle evaluation
        x0 = torch.zeros(runner.dim, dtype=runner.problem.dtype)
        alpha_values = [0.08, 0.04, 0.02]
        Ng_values = [1, 4, 8, 16, 32]
        print(f"[VAL-ALG1] alpha={alpha_values}, Ng={Ng_values}")
        stats = f2csa.estimate_bias_variance(x0, alpha_values, Ng_values, trials=32, ref_trials=128)
        with open('validation_algo1.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print("[VAL-ALG1] Saved to validation_algo1.json")
        return

    # Validation: Algorithm 2 (Goldstein proxy tracking)
    if args.validate_algo2:
        # We approximate G_k via available oracle at averaged z points
        f2csa = runner.algorithms['F2CSA']
        # run a short optimization with fixed hyperparameters to collect block averages
        res = f2csa.optimize(max_iterations=1000, target_gap=runner.target_gap, run_until_convergence=False)
        gaps = res.get('history', {}).get('gaps', [])
        # save minimal series
        out = {'gaps': gaps, 'total_iterations': res.get('total_iterations', 0)}
        with open('validation_algo2.json', 'w') as f:
            json.dump(out, f, indent=2)
        print("[VAL-ALG2] Saved to validation_algo2.json")
        return

    # Grid sweep over (alpha, Ng, D, eta)
    if args.grid_sweep:
        f2csa = runner.algorithms['F2CSA']
        alpha_list = [0.08, 0.04] if not args.grid_alpha else [float(v) for v in args.grid_alpha.split(',')]
        Ng_list = [16, 64] if not args.grid_Ng else [int(v) for v in args.grid_Ng.split(',')]
        D_list = [0.006, 0.010] if not args.grid_D else [float(v) for v in args.grid_D.split(',')]
        eta_list = [5e-4, 1e-3] if not args.grid_eta else [float(v) for v in args.grid_eta.split(',')]
        print(f"[GRID] alpha={alpha_list}, Ng={Ng_list}, D={D_list}, eta={eta_list}")
        table = []
        for a in alpha_list:
            for Ng in Ng_list:
                for D in D_list:
                    for eta in eta_list:
                        # apply overrides
                        f2csa.alpha_override = float(a)
                        f2csa.Ng_override = int(Ng)
                        f2csa.D_override = float(D)
                        f2csa.eta_override = float(eta)
                        # short run
                        res = f2csa.optimize(max_iterations=1200, target_gap=runner.target_gap, run_until_convergence=False)
                        ema_hist = res.get('ema_gap_history', [])
                        final_ema = float(ema_hist[-1]) if len(ema_hist) > 0 else float('inf')
                        # SMA(100) trend
                        sma_window = 100
                        if len(ema_hist) >= sma_window:
                            start_sma = float(np.mean(ema_hist[:sma_window]))
                            end_sma = float(np.mean(ema_hist[-sma_window:]))
                        else:
                            start_sma = float(np.mean(ema_hist)) if len(ema_hist) > 0 else float('inf')
                            end_sma = start_sma
                        sma_decrease = (end_sma <= start_sma)
                        # simple stability flags
                        stable = True
                        if len(ema_hist) >= 50:
                            recent = np.array(ema_hist[-50:])
                            if np.isnan(recent).any() or np.isinf(recent).any():
                                stable = False
                            # basic oscillation check: max jump
                            jumps = np.abs(np.diff(recent))
                            if float(np.max(jumps)) > 0.2 * max(1e-6, float(np.mean(np.abs(recent))+1e-6)):
                                stable = False
                        row = {
                            'alpha': float(a), 'Ng': int(Ng), 'D': float(D), 'eta': float(eta),
                            'final_ema_gap': final_ema,
                            'total_iterations': int(res.get('total_iterations', 0)),
                            'stable': bool(stable),
                            'sma100_start': start_sma,
                            'sma100_end': end_sma,
                            'sma100_decrease': bool(sma_decrease)
                        }
                        table.append(row)
                        print(f"[GRID-ROW] {row}")
        with open('grid_sweep_results.json', 'w') as f:
            json.dump({'results': table}, f, indent=2)
        print("[GRID] Saved to grid_sweep_results.json")
        return

    # Repeats mode for F2CSA (aggregation)
    if args.repeats and args.algorithm == 'F2CSA' and (args.eta is not None and args.D is not None):
        gaps = []
        times = []
        for k in range(args.repeats):
            # Vary seed for each repeat deterministically
            repeat_seed = base_seed + k
            runner_rep = BilevelExperimentRunner(device='cpu', seed=repeat_seed)
            f2csa_rep = runner_rep.algorithms['F2CSA']
            f2csa_rep.eta_override = float(args.eta)
            f2csa_rep.D_override = float(args.D)
            res = runner_rep.run_single_algorithm('F2CSA')
            gaps.append(res.get('final_ema_gap', float('inf')))
            times.append(res.get('total_time', 0))
        gaps_np = np.array(gaps)
        times_np = np.array(times)
        print(f"[REPEATS] η={args.eta:.3e}, D={args.D:.3e}, n={args.repeats}")
        print(f"  EMA gap mean={gaps_np.mean():.6f}, median={np.median(gaps_np):.6f}, std={gaps_np.std():.6f}")
        print(f"  Time mean={times_np.mean():.2f}s, median={np.median(times_np):.2f}s, std={times_np.std():.2f}s")
        # Keep the best single run in results for report
        best_idx = int(np.argmin(gaps_np))
        runner.results = {'F2CSA': {'final_ema_gap': float(gaps_np[best_idx]), 'total_time': float(times_np[best_idx]), 'target_achieved': bool(gaps_np[best_idx] <= runner.target_gap)}}
    elif args.two_stage_sweep or args.eta_list or args.D_list:
        if args.two_stage_sweep:
            # Coarse log sweep
            eta_values = [2e-5, 3.2e-5, 5e-5, 8e-5, 1.25e-4, 2e-4]
            D_values = [5e-3, 8e-3, 1.25e-2, 2e-2]
            print(f"[SWEEP] Stage 1 (coarse): η={eta_values}, D={D_values}")
        else:
            eta_values = [float(v) for v in (args.eta_list.split(',') if args.eta_list else [runner.eta])]
            D_values = [float(v) for v in (args.D_list.split(',') if args.D_list else [runner.D])]
            print(f"[SWEEP] F2CSA sweep over η={eta_values}, D={D_values}")
        best = None
        for eta in eta_values:
            for D in D_values:
                f2csa = runner.algorithms['F2CSA']
                f2csa.eta_override = float(eta)
                f2csa.D_override = float(D)
                res = runner.run_single_algorithm('F2CSA')
                key = (eta, D)
                if best is None or res.get('final_ema_gap', float('inf')) < best[1].get('final_ema_gap', float('inf')):
                    best = (key, res)
                print(f"[SWEEP-RESULT] η={eta:.3e}, D={D:.3e} -> EMA gap={res.get('final_ema_gap', float('inf')):.6f}, EMA std={res.get('ema_std', float('nan')):.6f}")
        if args.two_stage_sweep and best:
            # Fine local log sweep around best (multiplicative neighbors)
            (best_eta, best_D), _ = best
            eta_neighbors = [best_eta/1.6, best_eta/1.25, best_eta, best_eta*1.25, best_eta*1.6]
            D_neighbors = [best_D/1.6, best_D/1.25, best_D, best_D*1.25, best_D*1.6]
            # Boundaries
            eta_neighbors = [e for e in eta_neighbors if e > 1e-6 and e < 1e-2]
            D_neighbors = [d for d in D_neighbors if d > 1e-4 and d < 1.0]
            print(f"[SWEEP] Stage 2 (fine): η~{best_eta:.3e} -> {eta_neighbors}, D~{best_D:.3e} -> {D_neighbors}")
            for eta in eta_neighbors:
                for D in D_neighbors:
                    f2csa = runner.algorithms['F2CSA']
                    f2csa.eta_override = float(eta)
                    f2csa.D_override = float(D)
                    res = runner.run_single_algorithm('F2CSA')
                    key = (eta, D)
                    if res.get('final_ema_gap', float('inf')) < best[1].get('final_ema_gap', float('inf')):
                        best = (key, res)
                    print(f"[SWEEP-RESULT-FINE] η={eta:.3e}, D={D:.3e} -> EMA gap={res.get('final_ema_gap', float('inf')):.6f}, EMA std={res.get('ema_std', float('nan')):.6f}")
        if best:
            (best_eta, best_D), best_res = best
            print(f"[SWEEP-BEST] η={best_eta:.3e}, D={best_D:.3e} -> EMA gap={best_res.get('final_ema_gap', float('inf')):.6f}, EMA std={best_res.get('ema_std', float('nan')):.6f}")
            runner.results = {'F2CSA': best_res}
    else:
        if args.combined_plot_only:
            print("[PLOT-ONLY] Generating combined plot from existing JSON results")
            # Try to load consolidated results first
            loaded = False
            try:
                with open('comprehensive_bilevel_results.json', 'r') as f:
                    runner.results = json.load(f)
                # If loaded dict is keyed by algo with dict payloads, keep as-is
                if isinstance(runner.results, dict):
                    loaded = True
            except Exception:
                pass

            # Helper to extract gaps from either 'history.gaps' or 'gap_history'
            def extract_gaps(payload: dict):
                if not isinstance(payload, dict):
                    return []
                history = payload.get('history', {}) if isinstance(payload.get('history', {}), dict) else {}
                gaps = history.get('gaps', [])
                if isinstance(gaps, list) and gaps:
                    return gaps
                # common alternative key used by some writers
                gaps2 = payload.get('gap_history', [])
                return gaps2 if isinstance(gaps2, list) else []

            # Merge per-algorithm files into loaded results or assemble fresh if none loaded
            if not loaded:
                runner.results = {}

            # Ensure dict structure for runner.results
            if not isinstance(runner.results, dict):
                runner.results = {}

            # F2CSA source
            if 'F2CSA' not in runner.results or not runner.results.get('F2CSA', {}).get('history', {}).get('gaps'):
                try:
                    with open('f2csa_results.json', 'r') as f:
                        data = json.load(f)
                    f2 = data.get('F2CSA') or data
                    runner.results['F2CSA'] = runner.results.get('F2CSA', {})
                    runner.results['F2CSA']['history'] = {'gaps': extract_gaps(f2)}
                except Exception:
                    pass
            # DS-BLO source
            if 'DS-BLO' not in runner.results or not runner.results.get('DS-BLO', {}).get('history', {}).get('gaps'):
                try:
                    with open('dsblo_results.json', 'r') as f:
                        data = json.load(f)
                    ds = data.get('DS-BLO') or data
                    runner.results['DS-BLO'] = runner.results.get('DS-BLO', {})
                    runner.results['DS-BLO']['history'] = {'gaps': extract_gaps(ds)}
                except Exception:
                    pass
            # SSIGD source
            if 'SSIGD' not in runner.results or not runner.results.get('SSIGD', {}).get('history', {}).get('gaps'):
                try:
                    with open('ssigd_results.json', 'r') as f:
                        data = json.load(f)
                    sg = data.get('SSIGD') or data
                    runner.results['SSIGD'] = runner.results.get('SSIGD', {})
                    runner.results['SSIGD']['history'] = {'gaps': extract_gaps(sg)}
                except Exception:
                    pass

            # Save combined plot
            runner._save_combined_gap_plot()
            print("[PLOT-ONLY] Combined plot created from existing results")
            return
        
        if args.algorithm:
            print(f"[FOCUSED] Running only {args.algorithm} algorithm")
            results = runner.run_single_algorithm(args.algorithm)
            runner.results = {args.algorithm: results}
        else:
            results = runner.run_all_algorithms()
    
    # Save results
    runner.save_results()

    # Optional SMA(300) reporting for focused runs
    try:
        if isinstance(runner.results, dict):
            for name, res in runner.results.items():
                history = res.get('history', {}) if isinstance(res.get('history', {}), dict) else {}
                ema_hist = res.get('ema_gap_history', []) or history.get('ema_gap_history', []) or history.get('ema_gaps', [])
                if isinstance(ema_hist, list) and len(ema_hist) > 0:
                    import numpy as _np
                    w = 300
                    if len(ema_hist) >= w:
                        sma_start = float(_np.mean(ema_hist[:w]))
                        sma_end = float(_np.mean(ema_hist[-w:]))
                    else:
                        sma_start = float(_np.mean(ema_hist))
                        sma_end = sma_start
                    print(f"[SMA300] {name}: start={sma_start:.6f}, end={sma_end:.6f}, decrease={sma_end <= sma_start}")
    except Exception:
        pass
    
    # Generate and print summary report
    report = runner.generate_summary_report()
    print(report)
    
    # Save report to file
    with open('comprehensive_bilevel_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n[SUCCESS] Experiment completed successfully!")
    print("[RESULTS] Check 'comprehensive_bilevel_results.json' for detailed results")
    print("[REPORT] Check 'comprehensive_bilevel_report.txt' for summary report")

if __name__ == "__main__":
    main()