import os
import re
import time
import argparse
import datetime as dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ITER_RE = re.compile(r"^Iteration\s+(\d+):\s*$")
F2_RE = re.compile(r"^\s*F2CSA:\s+Gap=([0-9eE+\-.]+)")
DS_RE = re.compile(r"^\s*DSBLO:\s+Gap=([0-9eE+\-.]+)")


def parse_args():
    p = argparse.ArgumentParser(description="Live plot F2CSA vs DSBLO from log")
    p.add_argument('--log', default='long_run_50k.log', help='Path to log file to follow')
    p.add_argument('--out-base', default='live_compare', help='Output file base name')
    p.add_argument('--update-every', type=int, default=2000, help='Plot every N iterations')
    p.add_argument('--poll-sec', type=float, default=2.0, help='Polling interval (seconds)')
    return p.parse_args()


def save_plot_csv(f2_hist, ds_hist, last_iter, out_base):
    # Figure
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
    if f2_hist:
        xs_f2, ys_f2 = zip(*f2_hist)
        ax.semilogy(xs_f2, ys_f2, label='F2CSA', color='tab:blue', linewidth=2)
    if ds_hist:
        xs_ds, ys_ds = zip(*ds_hist)
        ax.semilogy(xs_ds, ys_ds, label='DS-BLO (K=10)', color='tab:orange', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gap (log scale)')
    ax.set_title(f'Live gap vs iter — up to {last_iter} (updated {dt.datetime.now().strftime("%H:%M:%S")})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    # Overlay current iteration number inside the axes
    ax.text(0.98, 0.02, f"iter {last_iter}", transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))
    plt.tight_layout()
    latest_png = f'{out_base}_latest.png'
    # Always overwrite a single latest image (no checkpoint archive)
    plt.savefig(latest_png, dpi=220, bbox_inches='tight')
    plt.close(fig)



def follow_and_plot(log_path, out_base, update_every, poll_sec):
    print(f"[live-plot] Following {log_path}; updating every {update_every} iterations")
    while not os.path.exists(log_path):
        print("[live-plot] Waiting for log file...")
        time.sleep(poll_sec)

    # Histories captured at progress blocks (every 50 iters from the main script)
    f2_hist = []  # list of (iter, gap)
    ds_hist = []

    # Next milestone
    next_milestone = update_every

    # Keep state of current "Iteration N:" block
    cur_iter = None
    got_f2 = None
    got_ds = None

    with open(log_path, 'r', encoding='utf-8', errors='replace') as fp:
        # Start from the beginning; then follow new content
        fp.seek(0, os.SEEK_SET)
        while True:
            pos = fp.tell()
            line = fp.readline()
            if not line:
                # No new data; sleep and retry
                time.sleep(poll_sec)
                fp.seek(pos)
                continue

            line = line.strip()

            m_it = ITER_RE.match(line)
            if m_it:
                # If we have previous block values, commit them before starting new block
                if cur_iter is not None:
                    if got_f2 is not None:
                        f2_hist.append((cur_iter, got_f2))
                    if got_ds is not None:
                        ds_hist.append((cur_iter, got_ds))
                    # Check milestone
                    if cur_iter >= next_milestone:
                        try:
                            save_plot_csv(f2_hist, ds_hist, cur_iter, out_base)
                            print(f"[live-plot] Saved plot at iter {cur_iter}")
                        except Exception as e:
                            print(f"[live-plot] Plot error at iter {cur_iter}: {e}")
                        while next_milestone <= cur_iter:
                            next_milestone += update_every
                # Start new block
                cur_iter = int(m_it.group(1))
                got_f2 = None
                got_ds = None
                continue

            if cur_iter is not None:
                m_f2 = F2_RE.match(line)
                if m_f2:
                    try:
                        got_f2 = float(m_f2.group(1))
                    except ValueError:
                        pass
                    continue
                m_ds = DS_RE.match(line)
                if m_ds:
                    try:
                        got_ds = float(m_ds.group(1))
                    except ValueError:
                        pass
                    continue

            # Also, as a safety: occasionally the main script may end without a new block;
            # we don't finalize here to avoid duplicates; next block commit handles it.


if __name__ == '__main__':
    args = parse_args()
    follow_and_plot(args.log, args.out_base, args.update_every, args.poll_sec)

