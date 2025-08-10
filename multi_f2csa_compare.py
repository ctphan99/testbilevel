import os
import re
import subprocess
import shutil
from typing import Tuple, Optional


def run_once(tag: str, ng: int, lr: float, max_iters: int = 2000) -> Tuple[Optional[float], Optional[float]]:
    """
    Runs comprehensive_convergence_test.py once with given F2CSA params and fixed DS-BLO Option II (sigma=1e-4).
    Returns (f2csa_gap, dsblo_gap) parsed from the run log (gap(x_final) lines).
    """
    env = os.environ.copy()
    # Ensure Unicode emojis don't crash on Windows console encoding
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    env["DSBLO_FIX_SIGMA"] = "1"  # keep σ fixed at 1e-4
    env["ONLY_ALGOS"] = "F2CSA,DSBLO"
    env["MAX_ITERS"] = str(max_iters)
    env["F2CSA_NG"] = str(ng)
    env["F2CSA_LR"] = f"{lr}"
    env["F2CSA_FIX_NG"] = "1"
    env["F2CSA_FIXED_LR"] = "1"
    env["RUN_TAG"] = tag

    # Remove old outputs to avoid mixing across runs
    for fname in [
        "comprehensive_convergence_summary.csv",
        "gradient_probe_summary.csv",
        "fd_probe_summary.csv",
        "comprehensive_convergence_test.png",
    ]:
        if os.path.exists(fname):
            try:
                os.remove(fname)
            except OSError:
                pass

    print(f"\n===== RUN {tag}: F2CSA_NG={ng}, F2CSA_LR={lr}, MAX_ITERS={max_iters} =====")
    proc = subprocess.run(
        ["python", "-u", "comprehensive_convergence_test.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    log_path = f"run_{tag}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)

    # Tag artifacts if present
    tag_targets = {
        "comprehensive_convergence_summary.csv": f"comprehensive_convergence_summary_{tag}.csv",
        "gradient_probe_summary.csv": f"gradient_probe_summary_{tag}.csv",
        "fd_probe_summary.csv": f"fd_probe_summary_{tag}.csv",
        "comprehensive_convergence_test.png": f"comprehensive_convergence_test_{tag}.png",
    }
    for src, dst in tag_targets.items():
        if os.path.exists(src):
            try:
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)
            except OSError:
                pass

    # Parse final gap(x_final) from logs (stable section)
    # Example lines:
    #   => F2CSA gap(x_final) = 1.6657250e-02
    #   => DSBLO gap(x_final) = 1.4887739e-01
    f2_gap = None
    ds_gap = None
    pattern = re.compile(r"=>\s*(F2CSA|DSBLO)\s+gap\(x_final\)\s*=\s*([0-9eE+\-.]+)")
    for line in proc.stdout.splitlines():
        m = pattern.search(line)
        if m:
            alg = m.group(1)
            val = float(m.group(2))
            if alg == "F2CSA":
                f2_gap = val
            elif alg == "DSBLO":
                ds_gap = val

    print(f"Parsed final gaps (x_final): F2CSA={f2_gap}, DSBLO={ds_gap}")
    return f2_gap, ds_gap


def main():
    # Config C and D as discussed
    # C: N_g=60, lr=7.5e-4
    # D: N_g=100, lr=7.5e-4
    max_iters = int(os.environ.get("COMPARE_MAX_ITERS", "2000"))

    c_f2, c_ds = run_once(tag="C", ng=60, lr=7.5e-4, max_iters=max_iters)
    d_f2, d_ds = run_once(tag="D", ng=100, lr=7.5e-4, max_iters=max_iters)

    # Report
    def fmt(x):
        return "NA" if x is None else f"{x:.8e}"

    print("\n===== COMPARISON (F2CSA configs C vs D) =====")
    print(f"C: F2CSA gap={fmt(c_f2)} | DSBLO gap={fmt(c_ds)} | |Δ|={fmt(None if (c_f2 is None or c_ds is None) else abs(c_ds - c_f2))}")
    print(f"D: F2CSA gap={fmt(d_f2)} | DSBLO gap={fmt(d_ds)} | |Δ|={fmt(None if (d_f2 is None or d_ds is None) else abs(d_ds - d_f2))}")

    # Which F2CSA config is closer to DS-BLO?
    c_diff = None if (c_f2 is None or c_ds is None) else abs(c_ds - c_f2)
    d_diff = None if (d_f2 is None or d_ds is None) else abs(d_ds - d_f2)

    winner = None
    if c_diff is not None and d_diff is not None:
        if c_diff < d_diff:
            winner = "C"
        elif d_diff < c_diff:
            winner = "D"
        else:
            winner = "TIE"

    if winner:
        print(f"Winner (closer to DS-BLO gap): {winner}")
    else:
        print("Winner: undetermined (missing metrics)")


if __name__ == "__main__":
    main()

