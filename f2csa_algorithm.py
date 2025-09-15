#!/usr/bin/env python3
"""
Canonical entry for Algorithm 2 (warm-LL + Adam carryover)
Configured to reproduce: algo2_warm_D0.05_eta1e-4_Ng64
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm2_working import F2CSAAlgorithm2Working


def run_once(T, D, eta, alpha, Ng, seed, warm_path, warm_ll, keep_adam_state, strong_convex, plot_name):
    torch.manual_seed(seed)
    np.random.seed(seed)

    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.1, strong_convex=strong_convex)
    algo2 = F2CSAAlgorithm2Working(problem)

    if os.path.exists(warm_path):
        try:
            x0 = torch.tensor(np.load(warm_path), dtype=torch.float64)
            print(f"Loaded warm start from {warm_path}: x0 shape = {tuple(x0.shape)}")
        except Exception as e:
            print(f"Failed to load warm start: {e}; using random x0")
            x0 = torch.randn(5, dtype=torch.float64)
    else:
        print(f"Warm start not found at {warm_path}; using random x0")
        x0 = torch.randn(5, dtype=torch.float64)

    delta = alpha ** 3

    results = algo2.optimize(
        x0=x0,
        T=T,
        D=D,
        eta=eta,
        delta=delta,
        alpha=alpha,
        N_g=Ng,
        warm_ll=warm_ll,
        keep_adam_state=keep_adam_state,
    )

    try:
        default_plot = 'algo2_hg_ul_loss.png'
        if plot_name != default_plot and os.path.exists(default_plot):
            os.replace(default_plot, plot_name)
            print(f"Renamed plot to {plot_name}")
    except Exception as e:
        print(f"Post-processing (plot rename) failed: {e}")

    return results


def run_sweep(alpha_list, T, seeds, warm_path, warm_ll, keep_adam_state):
    # Empirical grids (non-theoretical)
    D_list = [0.03, 0.05, 0.08]
    eta_list = [5e-5, 1e-4, 2e-4]
    Ng_list = [32, 64, 96]

    best = {
        'ul_loss': float('inf'),
        'alpha': None,
        'D': None,
        'eta': None,
        'Ng': None,
        'x_out': None,
        'result': None
    }

    for alpha in alpha_list:
        delta = alpha ** 3
        for D in D_list:
            for eta in eta_list:
                for Ng in Ng_list:
                    # Use most recent best warm start if exists
                    if os.path.exists(warm_path):
                        try:
                            x0 = torch.tensor(np.load(warm_path), dtype=torch.float64)
                        except Exception:
                            x0 = torch.randn(5, dtype=torch.float64)
                    else:
                        x0 = torch.randn(5, dtype=torch.float64)

                    # Run across a couple seeds for stability
                    ul_list = []
                    osc_list = []
                    last_hg_list = []
                    pick_res = None
                    for seed in seeds:
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.1, strong_convex=True)
                        algo2 = F2CSAAlgorithm2Working(problem)
                        res = algo2.optimize(x0=x0, T=T, D=D, eta=eta, delta=delta, alpha=alpha, N_g=Ng,
                                             warm_ll=warm_ll, keep_adam_state=keep_adam_state,
                                             plot_name=None, save_warm_name=warm_path)
                        ul = res.get('final_ul_loss', np.inf)
                        ul_list.append(ul)
                        hg = res.get('hypergrad_norms', [])
                        ul_hist = res.get('ul_losses', [])
                        if len(ul_hist) >= 50:
                            span = max(ul_hist[-50:]) - min(ul_hist[-50:])
                        else:
                            span = max(ul_hist) - min(ul_hist) if ul_hist else 1e9
                        osc_list.append(span)
                        last_hg = hg[-1] if hg else 1e9
                        last_hg_list.append(last_hg)
                        pick_res = res  # keep last

                    # Score: prefer low UL loss; break ties by stability (lower span) and lower last hypergrad
                    mean_ul = float(np.mean(ul_list))
                    mean_span = float(np.mean(osc_list))
                    mean_last_hg = float(np.mean(last_hg_list))

                    better = False
                    if mean_ul < best['ul_loss']:
                        better = True
                    elif np.isclose(mean_ul, best['ul_loss']):
                        # tie-breakers
                        prev_res = best['result']
                        prev_span = (max(prev_res['ul_losses'][-50:]) - min(prev_res['ul_losses'][-50:])) if prev_res and len(prev_res['ul_losses']) >= 50 else 1e9
                        if mean_span < prev_span or (np.isclose(mean_span, prev_span) and mean_last_hg < best['result']['hypergrad_norms'][-1]):
                            better = True

                    if better:
                        best.update({'ul_loss': mean_ul, 'alpha': alpha, 'D': D, 'eta': eta, 'Ng': Ng, 'x_out': pick_res['x_out'], 'result': pick_res})
                        # Save best warm start for subsequent runs
                        try:
                            np.save(warm_path, pick_res['x_out'].detach().cpu().numpy())
                            print(f"[BEST UPDATE] α={alpha:.2f}, D={D}, η={eta}, Ng={Ng}, mean f={mean_ul:.6f}, span≈{mean_span:.4e}, last ||∇F̃||≈{mean_last_hg:.3f}")
                        except Exception as e:
                            print(f"Failed to save best warm start: {e}")

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=5000)
    parser.add_argument('--D', type=float, default=0.05)
    parser.add_argument('--eta', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--Ng', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--plot-name', type=str, default='algo2_warm_D0.05_eta1e-4_Ng64.png')
    parser.add_argument('--warm-path', type=str, default='algo2_warmstart.npy')
    parser.add_argument('--no-warm-ll', action='store_true', help='Disable LL warm start (y, lambda)')
    parser.add_argument('--no-keep-adam', action='store_true', help='Disable Adam state carryover for penalty minimizer')
    parser.add_argument('--compare-strong', action='store_true', help='Run both strong_convex=False and True and create overlay plot')
    parser.add_argument('--sweep', action='store_true', help='Run alpha/param sweep')
    args = parser.parse_args()

    warm_ll = not args.no_warm_ll
    keep_adam_state = not args.no_keep_adam

    if args.sweep:
        alpha_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90]
        seeds = [args.seed, args.seed + 7]
        best = run_sweep(alpha_list, args.T, seeds, args.warm_path, warm_ll, keep_adam_state)
        print("\n== SWEEP BEST ==")
        print(f"alpha={best['alpha']}, D={best['D']}, eta={best['eta']}, Ng={best['Ng']}, mean UL≈{best['ul_loss']:.6f}")
        # Re-run from best x_out to refine once more
        try:
            np.save(args.warm_path, best['x_out'].detach().cpu().numpy())
        except Exception:
            pass
        _ = run_once(args.T, best['D'], best['eta'], best['alpha'], best['Ng'], args.seed,
                     args.warm_path, warm_ll, keep_adam_state, True, args.plot_name)
        return

    if not args.compare_strong:
        results = run_once(args.T, args.D, args.eta, args.alpha, args.Ng, args.seed,
                           args.warm_path, warm_ll, keep_adam_state, True, args.plot_name)
        print(f"Final UL loss: {results.get('final_ul_loss')}")
        print(f"Final hypergradient norm: {results.get('final_gradient_norm')}")
        print(f"Saved plot: {args.plot_name}")
        return

    # Compare mode
    res_no = run_once(args.T, args.D, args.eta, args.alpha, args.Ng, args.seed,
                      args.warm_path, warm_ll, keep_adam_state, False, 'algo2_no_strong.png')
    res_yes = run_once(args.T, args.D, args.eta, args.alpha, args.Ng, args.seed,
                       args.warm_path, warm_ll, keep_adam_state, True, 'algo2_yes_strong.png')

    # Overlay plot
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax1.plot(res_no['ul_losses'], label='UL loss (no strong convex)', color='tab:orange')
        ax1.plot(res_yes['ul_losses'], label='UL loss (strong convex)', color='tab:green')
        ax1.set_ylabel('f(x, y*)')
        ax1.legend()
        ax1.grid(True, alpha=0.2)
        ax2.plot(res_no['hypergrad_norms'], label='||∇F̃|| (no strong convex)', color='tab:blue')
        ax2.plot(res_yes['hypergrad_norms'], label='||∇F̃|| (strong convex)', color='tab:red')
        ax2.set_ylabel('||∇F̃||')
        ax2.set_xlabel('Iteration')
        ax2.legend()
        ax2.grid(True, alpha=0.2)
        fig.tight_layout()
        out = 'algo2_compare_strongconvex.png'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved comparison plot to {out}")
    except Exception as e:
        print(f"Failed to create comparison plot: {e}")


if __name__ == "__main__":
    main()
