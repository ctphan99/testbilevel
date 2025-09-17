#!/usr/bin/env python3
"""
Exact SBATCH Configuration: F2CSA vs DS-BLO Comparison
Uses the exact same parameters as the SBATCH run for fair comparison
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Optional, Tuple
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
from f2csa_algorithm2_working import F2CSAAlgorithm2Working
from dsblo_conservative import DSBLOConservative
from dsblo_optII import DSBLOOptII
from ssigd_correct_final import CorrectSSIGD
import warnings

# Legacy DsBlo adapter imports
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'BilevelLinearConstraints'))
import numpy as np
try:
    from algorithms import DsBlo as LegacyDsBlo
except Exception:
    LegacyDsBlo = None

# Try additional known locations for algorithms.py if not found
if LegacyDsBlo is None:
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'BilevelLinearConstraints'),
        r'C:\Users\phant\OneDrive\Documents\BilevelLinearConstraints\BilevelLinearConstraints'
    ]
    for p in candidate_paths:
        if p not in sys.path and os.path.isdir(p):
            sys.path.append(p)
            try:
                from algorithms import DsBlo as LegacyDsBlo  # type: ignore
                break
            except Exception:
                LegacyDsBlo = None
                continue

class LegacyNoisyProblemAdapter:
    def __init__(self, prob):
        self.prob_t = prob
        self.noise_up, self.noise_lo = prob._sample_instance_noise()
        self.Qup = (prob.Q_upper + self.noise_up).cpu().numpy()
        self.Qlo = (prob.Q_lower + self.noise_lo).cpu().numpy()
        self.cu  = prob.c_upper.cpu().numpy().reshape(-1,1)
        self.cl  = prob.c_lower.cpu().numpy().reshape(-1,1)
        self.P   = prob.P.cpu().numpy()
        self.A   = prob.A.cpu().numpy()
        self.B   = prob.B.cpu().numpy()
        self.b   = prob.b.cpu().numpy().reshape(-1,1)
        self.y_dim = self.Qlo.shape[0]

    # UL: f(x,y) = 0.5 x^T (Qup) x + c^T x + 0.5 y^T P y + x^T P y
    def f(self, x, y):
        return float(0.5*x.T@self.Qup@x + self.cu.T@x + 0.5*y.T@self.P@y + x.T@self.P@y)

    # LL: g(x,y) = 0.5 y^T (Qlo) y + c^T y
    def g(self, x, y):
        return float(0.5*y.T@self.Qlo@y + self.cl.T@y)

    def grady_g(self, x, y):
        return self.Qlo@y + self.cl

    # ∇_x f = Qup x + c + P y
    def gradx_f(self, x, y):
        return self.Qup@x + self.cu + self.P@y

    # ∇_y f = P y + P^T x
    def grady_f(self, x, y):
        return self.P@y + self.P.T@x

    def hessyy_g(self, x, y):
        return self.Qlo

    def hessxy_g(self, x, y):
        return np.zeros_like(self.Qlo)

    # projection used for init only; DsBlo replaces with solve_ll(x)
    def projy(self, y0):
        return y0

    def solve_ll(self, x):
        import torch
        xt = torch.from_numpy(x.squeeze()).to(self.prob_t.dtype)
        y_star_t, _ = self.prob_t.solve_lower_level(xt)
        y = y_star_t.cpu().numpy().reshape(-1,1)
        return y

warnings.filterwarnings('ignore')

def run_exact_sbatch_vs_dsblo():
    """Run F2CSA vs DS-BLO with exact SBATCH configuration"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Exact SBATCH Configuration: F2CSA vs DS-BLO')
    parser.add_argument('--T', type=int, default=5000, help='Number of iterations')
    parser.add_argument('--D', type=float, default=0.08, help='Clipping parameter')
    parser.add_argument('--eta', type=float, default=2e-4, help='Step size')
    parser.add_argument('--Ng', type=int, default=32, help='Number of gradient samples')
    parser.add_argument('--alpha', type=float, default=0.6, help='Accuracy parameter')
    parser.add_argument('--warm-ll', action='store_true', help='Enable lower-level warm start')
    parser.add_argument('--keep-adam-state', action='store_true', help='Keep Adam optimizer state')
    parser.add_argument('--plot-name', type=str, default='exact_sbatch_vs_dsblo.png', help='Plot filename')
    parser.add_argument('--dim', type=int, default=5, help='Problem dimension')
    parser.add_argument('--constraints', type=int, default=3, help='Number of constraints')
    # removed external perturbation; rely on per-sample instance noise from problem
    parser.add_argument('--dsblo-opt', type=str, choices=['I', 'II'], default='II', help='DS-BLO option (I deterministic, II stochastic)')
    parser.add_argument('--dsblo-sigma', type=float, default=0.0, help='Extra stochastic noise std for DS-BLO (set 0 to rely on instance noise only)')
    parser.add_argument('--dsblo-gamma1', type=float, default=5.0, help='DS-BLO gamma1 for step size')
    parser.add_argument('--dsblo-gamma2', type=float, default=1.0, help='DS-BLO gamma2 for step size')
    parser.add_argument('--dsblo-beta', type=float, default=0.6, help='DS-BLO momentum beta')
    parser.add_argument('--dsblo-k', type=int, default=16, help='DS-BLO gradient averaging samples')
    parser.add_argument('--dsblo-eta-cap', type=float, default=1e-3, help='DS-BLO eta cap baseline')
    parser.add_argument('--use-crn-ul', action='store_true', help='Plot UL with a single fixed CRN noise for both methods')
    parser.add_argument('--ul-scale', type=str, choices=['linear','symlog'], default='linear', help='Y-scale for UL plots')
    parser.add_argument('--ul-overlay-noisy', action='store_true', help='Overlay MC mean±std with fresh noise per iter')
    parser.add_argument('--ul-overlay-noisy-raw', action='store_true', help='Overlay single-sample raw noisy UL per iter for both methods')
    parser.add_argument('--problem-noise-std', type=float, default=None, help='Instance noise std for problem; default keeps current')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for shared x0')
    parser.add_argument('--legacy-dsblo', action='store_true', help='Run DsBlo (algorithms.py) on a fixed noisy instance')
    parser.add_argument('--only-dsblo', action='store_true', help='Run only DS-BLO and plot its results')
    parser.add_argument('--with-ssigd', action='store_true', help='Include SSIGD in comparison')
    parser.add_argument('--ul-track-noisy-ll', action='store_true', help='Track UL using LL solved with noisy Q_lower each iter (DS-BLO)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EXACT SBATCH CONFIGURATION: F2CSA vs DS-BLO COMPARISON")
    print("=" * 80)
    print(f"Problem: dim={args.dim}, constraints={args.constraints}")
    print(f"F2CSA Config: T={args.T}, D={args.D}, eta={args.eta}, Ng={args.Ng}, alpha={args.alpha}")
    print(f"Warm start: {args.warm_ll}, Adam state: {args.keep_adam_state}")
    # no explicit perturbation std; stochasticity comes from instance noise
    print()
    
    # Seeding for reproducible shared x0 (optional)
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Create problem instance (optionally align noise std)
    noise_std = args.problem_noise_std if args.problem_noise_std is not None else 0.1
    problem = StronglyConvexBilevelProblem(dim=args.dim, num_constraints=args.constraints, noise_std=noise_std)
    
    # Initialize starting point
    x0 = torch.randn(args.dim, dtype=torch.float64)
    print(f"Starting point: {x0}")
    print()
    
    # Compute initial upper-level loss
    y0_star, _ = problem.solve_lower_level(x0)
    initial_ul_loss = problem.upper_objective(x0, y0_star).item()
    print(f"Initial UL loss: {initial_ul_loss:.6f}")
    print()
    
    f2csa_results = None
    if not args.only_dsblo:
        # Run F2CSA Algorithm 2 with exact SBATCH configuration
        print("=" * 50)
        print("RUNNING F2CSA ALGORITHM 2 (SBATCH CONFIG)")
        print("=" * 50)
        
        algorithm2 = F2CSAAlgorithm2Working(problem)
        delta = args.alpha ** 3
        
        f2csa_results = algorithm2.optimize(
            x0, args.T, args.D, args.eta, delta, args.alpha, args.Ng,
            warm_ll=args.warm_ll, keep_adam_state=args.keep_adam_state,
            plot_name=None, save_warm_name=None
        )
        
        print()
        print("F2CSA Results:")
        print(f"  Final UL loss: {f2csa_results['final_ul_loss']:.6f}")
        print(f"  Final gradient norm: {f2csa_results['final_gradient_norm']:.6f}")
        print(f"  Converged: {f2csa_results['converged']}")
        print(f"  Iterations: {f2csa_results['iterations']}")
        print()
    
    # Run DS-BLO with same problem and parameters
    print("=" * 50)
    print("RUNNING DS-BLO (SAME PROBLEM)")
    print("=" * 50)
    
    if args.dsblo_opt == 'II':
        dsblo = DSBLOOptII(problem)
        dsblo_results = dsblo.optimize(
            x0, args.T, args.alpha,
            sigma=args.dsblo_sigma,
            grad_avg_k=args.dsblo_k,
            gamma1=args.dsblo_gamma1,
            gamma2=args.dsblo_gamma2,
            beta=args.dsblo_beta,
            eta_cap=args.dsblo_eta_cap,
            ul_track_noisy_ll=args.ul_track_noisy_ll,
        )
    else:
        dsblo = DSBLOConservative(problem)
        dsblo_results = dsblo.optimize(x0, args.T, args.alpha)
    
    print()
    print("DS-BLO Results:")
    print(f"  Final UL loss: {dsblo_results['final_ul_loss']:.6f}")
    print(f"  Final gradient norm: {dsblo_results['final_gradient_norm']:.6f}")
    print(f"  Converged: {dsblo_results['converged']}")
    print(f"  Iterations: {dsblo_results['iterations']}")
    print()

    dsblo_legacy_results = None
    if args.legacy_dsblo:
        if LegacyDsBlo is None:
            raise RuntimeError('Legacy DsBlo not available: algorithms.py not found')
        x0_np = x0.cpu().numpy().reshape(-1,1)
        y0_np = np.zeros((args.dim,1))
        legacy = LegacyNoisyProblemAdapter(problem)
        dsblo_legacy = LegacyDsBlo(legacy, out_iter=args.T, gamma1=args.dsblo_gamma1, gamma2=args.dsblo_gamma2, beta=args.dsblo_beta)
        dsblo_legacy.run(x0_np, y0_np)
        ul_losses = dsblo_legacy.loss
        hypergrad_norms = dsblo_legacy.gradF
        x_hist = [torch.from_numpy(xx.squeeze()).to(torch.float64) for xx in dsblo_legacy.x_iter]
        dsblo_legacy_results = {
            'final_ul_loss': ul_losses[-1],
            'final_gradient_norm': hypergrad_norms[-1],
            'converged': False,
            'iterations': args.T,
            'ul_losses': ul_losses,
            'hypergrad_norms': hypergrad_norms,
            'x_history': x_hist,
        }

    # Run SSIGD if requested
    ssigd_results = None
    if args.with_ssigd:
        print("=" * 50)
        print("RUNNING SSIGD")
        print("=" * 50)
        ssigd_algo = CorrectSSIGD(problem)
        x_ssigd, ul_losses_ssigd, hypergrad_norms_ssigd = ssigd_algo.solve(T=args.T, beta=args.eta, x0=x0)
        ssigd_results = {
            'final_ul_loss': ul_losses_ssigd[-1],
            'final_gradient_norm': hypergrad_norms_ssigd[-1],
            'converged': hypergrad_norms_ssigd[-1] < 1e-2,  # Heuristic
            'iterations': args.T,
            'ul_losses': ul_losses_ssigd,
            'hypergrad_norms': hypergrad_norms_ssigd,
            'x_history': [x_ssigd] * args.T  # SSIGD only returns final x, so replicate for history
        }
        print()
        print("SSIGD Results:")
        print(f"  Final UL loss: {ssigd_results['final_ul_loss']:.6f}")
        print(f"  Final gradient norm: {ssigd_results['final_gradient_norm']:.6f}")
        print(f"  Converged: {ssigd_results['converged']}")
        print(f"  Iterations: {ssigd_results['iterations']}")
        print()
    
    # Create comparison plot
    print("Creating comparison plot...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Optionally recompute UL with a single fixed CRN noise for both methods
    ul_f2csa = f2csa_results['ul_losses'] if f2csa_results is not None else None
    ul_dsblo = dsblo_results['ul_losses']
    if args.use_crn_ul:
        crn_up, _ = problem._sample_instance_noise()
        # recompute along x histories with fixed noise
        if f2csa_results is not None:
            ul_f2csa_crn = []
            for x_t in f2csa_results['x_history']:
                y_star_t, _ = problem.solve_lower_level(x_t)
                ul_f2csa_crn.append(problem.upper_objective(x_t, y_star_t, crn_up).item())
            ul_f2csa = ul_f2csa_crn
        ul_dsblo_crn = []
        for x_t in dsblo_results['x_history']:
            y_star_t, _ = problem.solve_lower_level(x_t)
            ul_dsblo_crn.append(problem.upper_objective(x_t, y_star_t, crn_up).item())
        ul_dsblo = ul_dsblo_crn

    # Optional noisy overlay: Monte Carlo mean±std with fresh noise per point
    if args.ul_overlay_noisy:
        def mc_ul(x_hist, M=10):
            means, stds = [], []
            for x_t in x_hist:
                vals = []
                for _ in range(M):
                    n_up, _ = problem._sample_instance_noise()
                    y_star_t, _ = problem.solve_lower_level(x_t)
                    vals.append(problem.upper_objective(x_t, y_star_t, n_up).item())
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            return np.array(means), np.array(stds)
        m_f2, s_f2 = (None, None)
        if f2csa_results is not None:
            m_f2, s_f2 = mc_ul(f2csa_results['x_history'])
        m_ds, s_ds = mc_ul(dsblo_results['x_history'])

    # Optional raw noisy overlay: single fresh noise sample per iter (trajectory-like)
    if args.ul_overlay_noisy_raw:
        def raw_ul(x_hist):
            raw_vals = []
            for x_t in x_hist:
                n_up, _ = problem._sample_instance_noise()
                y_star_t, _ = problem.solve_lower_level(x_t)
                raw_vals.append(problem.upper_objective(x_t, y_star_t, n_up).item())
            return np.array(raw_vals)
        raw_f2 = raw_ul(f2csa_results['x_history']) if f2csa_results is not None else None
        raw_ds = raw_ul(dsblo_results['x_history'])
    
    # Plot 1: Upper-level loss comparison
    if ul_f2csa is not None:
        ax1.plot(ul_f2csa, label='F2CSA', linewidth=2)
    ax1.plot(ul_dsblo, label='DS-BLO', linewidth=2)
    if ssigd_results is not None:
        ax1.plot(ssigd_results['ul_losses'], label='SSIGD', linewidth=2)
    if dsblo_legacy_results is not None:
        ax1.plot(dsblo_legacy_results['ul_losses'], label='DS-BLO (Legacy)', linewidth=2, linestyle=':')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Upper-level Loss')
    ax1.set_title('Upper-level Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    if args.ul_scale == 'symlog':
        ax1.set_yscale('symlog', linthresh=1.0)
    if args.ul_overlay_noisy:
        if m_f2 is not None and s_f2 is not None:
            it_f2 = np.arange(len(m_f2))
            ax1.plot(it_f2, m_f2, color='C0', alpha=0.6, linestyle='--', label='F2CSA (noisy mean)')
            ax1.fill_between(it_f2, m_f2 - s_f2, m_f2 + s_f2, color='C0', alpha=0.15)
        it_ds = np.arange(len(m_ds))
        ax1.plot(it_ds, m_ds, color='C1', alpha=0.6, linestyle='--', label='DS-BLO (noisy mean)')
        ax1.fill_between(it_ds, m_ds - s_ds, m_ds + s_ds, color='C1', alpha=0.15)
    if args.ul_overlay_noisy_raw:
        if f2csa_results is not None and raw_f2 is not None:
            ax1.plot(np.arange(len(raw_f2)), raw_f2, color='C0', alpha=0.45, linewidth=1, label='F2CSA (raw noisy)')
        ax1.plot(np.arange(len(raw_ds)), raw_ds, color='C1', alpha=0.45, linewidth=1, label='DS-BLO (raw noisy)')
    
    # Plot 2: Gradient norm comparison
    if f2csa_results is not None:
        ax2.plot(f2csa_results['hypergrad_norms'], label='F2CSA (SBATCH Config)', linewidth=2)
    ax2.plot(dsblo_results['hypergrad_norms'], label='DS-BLO', linewidth=2)
    if ssigd_results is not None:
        ax2.plot(ssigd_results['hypergrad_norms'], label='SSIGD', linewidth=2)
    if dsblo_legacy_results is not None:
        ax2.plot(dsblo_legacy_results['hypergrad_norms'], label='DS-BLO (Legacy)', linewidth=2, linestyle=':')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Hypergradient Norm')
    ax2.set_title('Hypergradient Norm Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: F2CSA trajectory
    if f2csa_results is not None:
        x_history = torch.stack(f2csa_results['x_history'])
        ax3.plot(x_history[:, 0], x_history[:, 1], 'b-', alpha=0.7, linewidth=1)
        ax3.scatter(x_history[0, 0], x_history[0, 1], color='green', s=100, label='Start', zorder=5)
        ax3.scatter(x_history[-1, 0], x_history[-1, 1], color='red', s=100, label='End', zorder=5)
        ax3.set_xlabel('x[0]')
        ax3.set_ylabel('x[1]')
        ax3.set_title('F2CSA Trajectory (First 2 Dimensions)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.axis('off')
    
    # Plot 4: DS-BLO trajectory
    x_history_dsblo = torch.stack(dsblo_results['x_history'])
    ax4.plot(x_history_dsblo[:, 0], x_history_dsblo[:, 1], 'r-', alpha=0.7, linewidth=1, label='DS-BLO')
    ax4.scatter(x_history_dsblo[0, 0], x_history_dsblo[0, 1], color='green', s=100, label='Start', zorder=5)
    ax4.scatter(x_history_dsblo[-1, 0], x_history_dsblo[-1, 1], color='red', s=100, label='End', zorder=5)
    if ssigd_results is not None:
        x_history_ssigd = torch.stack(ssigd_results['x_history'])
        ax4.plot(x_history_ssigd[:, 0], x_history_ssigd[:, 1], 'g-', alpha=0.7, linewidth=1, label='SSIGD')
    if dsblo_legacy_results is not None:
        x_hist_legacy = torch.stack(dsblo_legacy_results['x_history'])
        ax4.plot(x_hist_legacy[:, 0], x_hist_legacy[:, 1], 'm--', alpha=0.7, linewidth=1, label='DS-BLO (Legacy)')
    ax4.set_xlabel('x[0]')
    ax4.set_ylabel('x[1]')
    ax4.set_title('DS-BLO Trajectory (First 2 Dimensions)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.plot_name, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {args.plot_name}")
    
    # Print final comparison summary
    print()
    print("=" * 80)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<25} {'F2CSA (SBATCH)':<20} {'DS-BLO':<20} {'Winner':<15}")
    print("-" * 80)
    print(f"{'Initial UL Loss':<25} {initial_ul_loss:<20.6f} {initial_ul_loss:<20.6f} {'Same':<15}")
    if f2csa_results is not None:
        print(f"{'Final UL Loss':<25} {f2csa_results['final_ul_loss']:<20.6f} {dsblo_results['final_ul_loss']:<20.6f} {'F2CSA' if f2csa_results['final_ul_loss'] < dsblo_results['final_ul_loss'] else 'DS-BLO':<15}")
        print(f"{'Final Grad Norm':<25} {f2csa_results['final_gradient_norm']:<20.6f} {dsblo_results['final_gradient_norm']:<20.6f} {'F2CSA' if f2csa_results['final_gradient_norm'] < dsblo_results['final_gradient_norm'] else 'DS-BLO':<15}")
        print(f"{'Converged':<25} {f2csa_results['converged']:<20} {dsblo_results['converged']:<20} {'F2CSA' if f2csa_results['converged'] and not dsblo_results['converged'] else 'DS-BLO' if dsblo_results['converged'] and not f2csa_results['converged'] else 'Both' if f2csa_results['converged'] and dsblo_results['converged'] else 'Neither':<15}")
        print(f"{'Iterations':<25} {f2csa_results['iterations']:<20} {dsblo_results['iterations']:<20} {'Same':<15}")
    else:
        print(f"{'Final UL Loss':<25} {'-':<20} {dsblo_results['final_ul_loss']:<20.6f} {'DS-BLO':<15}")
        print(f"{'Final Grad Norm':<25} {'-':<20} {dsblo_results['final_gradient_norm']:<20.6f} {'DS-BLO':<15}")
        print(f"{'Converged':<25} {'-':<20} {dsblo_results['converged']:<20} {'DS-BLO':<15}")
        print(f"{'Iterations':<25} {'-':<20} {dsblo_results['iterations']:<20} {'DS-BLO':<15}")
    if dsblo_legacy_results is not None:
        print(f"{'(Legacy) Final UL Loss':<25} {'-':<20} {dsblo_legacy_results['final_ul_loss']:<20.6f} {'DS-BLO (Legacy)':<15}")
        print(f"{'(Legacy) Final Grad Norm':<25} {'-':<20} {dsblo_legacy_results['final_gradient_norm']:<20.6f} {'DS-BLO (Legacy)':<15}")
    if ssigd_results is not None:
        print(f"{'SSIGD Final UL Loss':<25} {'-':<20} {'-':<20} {ssigd_results['final_ul_loss']:<20.6f} {'SSIGD':<15}")
        print(f"{'SSIGD Final Grad Norm':<25} {'-':<20} {'-':<20} {ssigd_results['final_gradient_norm']:<20.6f} {'SSIGD':<15}")
    print("=" * 80)
    
    return {
        'f2csa_results': f2csa_results,
        'dsblo_results': dsblo_results,
        'initial_ul_loss': initial_ul_loss
    }

if __name__ == "__main__":
    results = run_exact_sbatch_vs_dsblo()
