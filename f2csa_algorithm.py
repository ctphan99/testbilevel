#!/usr/bin/env python3
"""
Canonical entry for Algorithm 2 (warm-LL + Adam carryover)
Configured to reproduce: algo2_warm_D0.05_eta1e-4_Ng64
"""

import os
import argparse
import numpy as np
import torch
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm import F2CSAAlgorithm2Working


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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    warm_ll = not args.no_warm_ll
    keep_adam_state = not args.no_keep_adam

    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, noise_std=0.1, strong_convex=True)
    algo2 = F2CSAAlgorithm2Working(problem)

    # Load warm start x0 if present; otherwise random
    if os.path.exists(args.warm_path):
        try:
            x0 = torch.tensor(np.load(args.warm_path), dtype=torch.float64)
            print(f"Loaded warm start from {args.warm_path}: x0 shape = {tuple(x0.shape)}")
        except Exception as e:
            print(f"Failed to load warm start: {e}; using random x0")
            x0 = torch.randn(5, dtype=torch.float64)
    else:
        print(f"Warm start not found at {args.warm_path}; using random x0")
        x0 = torch.randn(5, dtype=torch.float64)

    delta = args.alpha ** 3

    results = algo2.optimize(
        x0=x0,
        T=args.T,
        D=args.D,
        eta=args.eta,
        delta=delta,
        alpha=args.alpha,
        N_g=args.Ng,
        warm_ll=warm_ll,
        keep_adam_state=keep_adam_state,
    )

    # Rename plot to requested name if created under default
    try:
        default_plot = 'algo2_hg_ul_loss.png'
        if args.plot_name != default_plot and os.path.exists(default_plot):
            os.replace(default_plot, args.plot_name)
            print(f"Renamed plot to {args.plot_name}")
    except Exception as e:
        print(f"Post-processing (plot rename) failed: {e}")

    print(f"Final UL loss: {results.get('final_ul_loss')}")
    print(f"Final hypergradient norm: {results.get('final_gradient_norm')}")
    print(f"Saved plot: {args.plot_name}")


if __name__ == "__main__":
    main()
