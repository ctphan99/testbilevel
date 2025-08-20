import os
import time
import math
import json
from typing import List, Tuple, Dict

import numpy as np
import torch

from problem import StronglyConvexBilevelProblem
from algorithm import F2CSA


def plateau_iter(gaps: List[float], window: int = 100, thresh: float = 1e-3) -> int:
    if len(gaps) <= window:
        return -1
    for t in range(window, len(gaps)):
        if (gaps[t - window] - gaps[t]) < thresh:
            return t
    return -1


def summarize_run(res: Dict, debug: Dict) -> Dict:
    gaps = [float(x['gap']) for x in res['history']]
    grads = [float(x['gradient_norm']) for x in res['history']]
    its = [int(x['iteration']) for x in res['history']]

    t_plateau = plateau_iter(gaps, window=100, thresh=1e-3)
    t_plateau_1e5 = plateau_iter(gaps, window=100, thresh=1e-5)

    summary = {
        'final_gap': float(res['final_gap']),
        'best_gap': float(res['best_gap']),
        'total_iterations': int(res['total_iterations']),
        'converged': bool(res['converged']),
        't_plateau_1e3': int(t_plateau),
        't_plateau_1e5': int(t_plateau_1e5),
        'grad_init': float(grads[0]) if grads else float('nan'),
        'grad_final': float(grads[-1]) if grads else float('nan'),
        'grad_ratio': (float(grads[-1]) / float(grads[0])) if grads and grads[0] != 0 else float('nan'),
    }

    # Attach selected diagnostics from the final iteration
    diag = {
        'H_cond': debug.get('H_cond'),
        'rho_min': debug.get('rho_min'), 'rho_mean': debug.get('rho_mean'), 'rho_max': debug.get('rho_max'), 'rho_std': debug.get('rho_std'),
        'rho_active_count': debug.get('rho_active_count'),
        'raw_grad_norm': debug.get('raw_grad_norm'),
        'sample_grad_norm_mean': debug.get('sample_grad_norm_mean'),
        'sample_grad_norm_std': debug.get('sample_grad_norm_std'),
        'comp_grad_f_norm': debug.get('comp_grad_f_norm'),
        'comp_grad_pen_lin_norm': debug.get('comp_grad_pen1_norm'),
        'comp_grad_pen_quad_norm': debug.get('comp_grad_pen2_norm'),
        'll_status': debug.get('ll_status'), 'll_converged': debug.get('ll_converged'),
        'll_cv': debug.get('ll_cv'), 'll_opt_gap': debug.get('ll_opt_gap'),
    }
    summary['final_diag'] = diag
    return summary


def md_block_from_summary(cfg: Dict, summ: Dict) -> str:
    d = summ['final_diag']
    lines = []
    lines.append(f"### Config: alpha={cfg['alpha']}, N_g={cfg['N_g']}, lr={cfg['lr']}")
    lines.append("")
    lines.append("- Outcome: converged=%s, iters=%d, final_gap=%.6e, best_gap=%.6e" % (
        str(summ['converged']), summ['total_iterations'], summ['final_gap'], summ['best_gap']))
    lines.append("- Plateau (Δ100<1e-3): %s" % (str(summ['t_plateau_1e3']) if summ['t_plateau_1e3'] >= 0 else 'not reached'))
    lines.append("- Plateau (Δ100<1e-5): %s" % (str(summ['t_plateau_1e5']) if summ['t_plateau_1e5'] >= 0 else 'not reached'))
    lines.append("- Grad norm: init=%.3e → final=%.3e (ratio=%.3f)" % (summ['grad_init'], summ['grad_final'], summ['grad_ratio']))
    lines.append("")
    lines.append("- [Inner] cond(H)=%.3e | raw_grad=%.3e" % (d.get('H_cond', float('nan')) or float('nan'), d.get('raw_grad_norm', float('nan')) or float('nan')))
    lines.append("- [Gating] ρ: min=%.3e mean=%.3e max=%.3e std=%.3e; active(>0.9)=%s" % (
        d.get('rho_min', float('nan')) or float('nan'), d.get('rho_mean', float('nan')) or float('nan'), d.get('rho_max', float('nan')) or float('nan'), d.get('rho_std', float('nan')) or float('nan'), str(d.get('rho_active_count'))))
    lines.append("- [Components] |grad_f|=%.3e |grad_pen_lin|=%.3e |grad_pen_quad|=%.3e" % (
        d.get('comp_grad_f_norm', float('nan')) or float('nan'), d.get('comp_grad_pen_lin_norm', float('nan')) or float('nan'), d.get('comp_grad_pen_quad_norm', float('nan')) or float('nan')))
    lines.append("- [LL] status=%s converged=%s cv=%.2e opt_gap=%.2e" % (
        str(d.get('ll_status')), str(d.get('ll_converged')), d.get('ll_cv', float('nan')) or float('nan'), d.get('ll_opt_gap', float('nan')) or float('nan')))
    lines.append("")
    return "\n".join(lines)


def main():
    os.makedirs('logs', exist_ok=True)

    dim = int(os.getenv('PROB_DIM', '20'))
    num_cons = int(os.getenv('PROB_NUM_CONS', '5'))

    alphas = [0.25, 0.35, 0.5]
    Ngs = [30, 100]
    lr = float(os.getenv('F2CSA_LR', '5e-3'))
    max_iters = int(os.getenv('MAX_ITERS', '2000'))
    conv_thr = float(os.getenv('CONV_THR', '1e-3'))

    report_lines = []
    report_lines.append("## F2CSA sweep: effect of alpha and N_g on convergence plateau")
    report_lines.append("")
    report_lines.append(f"- Problem: StronglyConvexBilevelProblem(dim={dim}, constraints={num_cons}, noise_std=1e-4)")
    report_lines.append(f"- Sweep: alpha in {alphas}, N_g in {Ngs}, Adam lr={lr}, max_iters={max_iters}, conv_thr={conv_thr}")
    report_lines.append("")

    results = []

    for alpha in alphas:
        for Ng in Ngs:
            torch.manual_seed(42)
            np.random.seed(42)

            problem = StronglyConvexBilevelProblem(dim=dim, num_constraints=num_cons, device='cpu', seed=42, noise_std=1e-4)
            algo = F2CSA(problem, N_g=Ng, alpha=alpha, adam_lr=lr)

            t0 = time.time()
            res = algo.optimize(max_iterations=max_iters, convergence_threshold=conv_thr)
            elapsed = time.time() - t0

            summ = summarize_run(res, getattr(algo, 'last_debug', {}) or {})
            cfg = {'alpha': alpha, 'N_g': Ng, 'lr': lr, 'time_sec': elapsed}
            results.append({'cfg': cfg, 'summary': summ})

            report_lines.append(md_block_from_summary(cfg, summ))
            report_lines.append("- Time: %.2fs" % (elapsed,))
            report_lines.append("")

    # Aggregate observations
    report_lines.append("## Observations")
    report_lines.append("")
    # Find best final gap
    best = min(results, key=lambda r: r['summary']['final_gap'])
    report_lines.append("- Best final gap: %.6e at alpha=%.2f, N_g=%d" % (best['summary']['final_gap'], best['cfg']['alpha'], best['cfg']['N_g']))
    # Plateau timing
    pt = sorted(results, key=lambda r: (r['summary']['t_plateau_1e3'] if r['summary']['t_plateau_1e3'] >= 0 else 10**9))
    if pt and pt[0]['summary']['t_plateau_1e3'] >= 0:
        report_lines.append("- Earliest plateau (Δ100<1e-3): it=%d at alpha=%.2f, N_g=%d" % (pt[0]['summary']['t_plateau_1e3'], pt[0]['cfg']['alpha'], pt[0]['cfg']['N_g']))
    else:
        report_lines.append("- No runs hit the 1e-3 plateau criterion within max_iters")

    # Write report
    out_path = os.path.join('logs', 'sweep_analysis.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    print(f"Wrote markdown analysis to {out_path}")


if __name__ == '__main__':
    main()

