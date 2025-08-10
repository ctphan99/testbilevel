import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from comprehensive_convergence_test import run_comprehensive_convergence_test


def run_dsblo_with_k(K: int, max_iters: int = 50):
    # Configure environment for a paper-accurate DS-BLO run
    os.environ['ONLY_ALGOS'] = 'DSBLO'
    os.environ['MAX_ITERS'] = str(max_iters)
    os.environ['DSBLO_SIGMA'] = '1e-4'
    os.environ['DSBLO_FIX_SIGMA'] = '1'
    os.environ['DSBLO_PAPER_STEP'] = '1'
    os.environ['DSBLO_OPTION_II'] = '1'
    os.environ['DSBLO_MOMENTUM'] = '0.85'
    os.environ['DSBLO_EMA'] = '0.0'  # no EMA smoothing
    os.environ['DSBLO_GAMMA1'] = '20.0'
    os.environ['DSBLO_GAMMA2'] = '1.0'
    os.environ['DSBLO_NG'] = str(K)   # within-step sample count K

    monitor = run_comprehensive_convergence_test()
    data = monitor.algorithm_data['DSBLO']
    iters = data['iterations']
    gaps = data['gaps']
    return iters, gaps


def main():
    ks = [1, 10, 50]
    max_iters = 400  # extended horizon to make smoothing differences visible

    traces = {}
    for k in ks:
        print(f"\n=== Running DS-BLO with K={k}, iters={max_iters} ===")
        iters, gaps = run_dsblo_with_k(k, max_iters=max_iters)
        traces[k] = (iters, gaps)

    # Plot side-by-side without averaging
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharey=True)
    colors = {1: 'tab:blue', 10: 'tab:orange', 50: 'tab:green'}
    for ax, k in zip(axes, ks):
        iters, gaps = traces[k]
        ax.semilogy(iters, gaps, label=f'K={k}', color=colors.get(k, 'tab:red'), linewidth=2)
        ax.set_title(f'DS-BLO (K={k})')
        ax.set_xlabel('Iteration')
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel('Gap (log scale)')

    fig.suptitle('DS-BLO: Gap vs Iteration (no EMA) — K ∈ {1,10,50}', fontsize=14)
    plt.tight_layout()
    out_png = 'dsblo_k_sweep.png'
    plt.savefig(out_png, dpi=250, bbox_inches='tight')
    print(f"Saved {out_png}")

    # Also create a single overlay plot for direct comparison
    fig2, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
    for k in ks:
        iters, gaps = traces[k]
        ax.semilogy(iters, gaps, label=f'K={k}', color=colors.get(k, 'tab:red'), linewidth=2, alpha=0.9)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gap (log scale)')
    ax.set_title('DS-BLO Overlay: K ∈ {1,10,50} (no EMA)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_overlay = 'dsblo_k_overlay.png'
    plt.savefig(out_overlay, dpi=250, bbox_inches='tight')
    print(f"Saved {out_overlay}")

    # Save per-iteration traces to CSV for post-hoc analysis
    csv_path = 'dsblo_k_traces.csv'
    try:
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['K', 'iteration', 'gap'])
            for k in ks:
                iters, gaps = traces[k]
                for it, gp in zip(iters, gaps):
                    writer.writerow([k, it, gp])
        print(f"Saved {csv_path}")
    except Exception as e:
        print(f"Warning: failed to write CSV traces to {csv_path}: {e}")


if __name__ == '__main__':
    main()

