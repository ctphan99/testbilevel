import os
import re
import subprocess
from typing import Dict, Tuple


CONFIGS = {
    "BASE": {"DSBLO_GAMMA1": "50.0", "DSBLO_GAMMA2": "5.0", "DSBLO_MOMENTUM": "0.9", "DSBLO_EMA": "0.9"},
    "DS1":  {"DSBLO_GAMMA1": "30.0", "DSBLO_GAMMA2": "2.0", "DSBLO_MOMENTUM": "0.9", "DSBLO_EMA": "0.85"},
    "DS2":  {"DSBLO_GAMMA1": "20.0", "DSBLO_GAMMA2": "1.0", "DSBLO_MOMENTUM": "0.9", "DSBLO_EMA": "0.85"},
    "DS3":  {"DSBLO_GAMMA1": "10.0", "DSBLO_GAMMA2": "0.5", "DSBLO_MOMENTUM": "0.85", "DSBLO_EMA": "0.8"},
    # Optional: an extra aggressive variant without EMA
    "DS3_NOEMA": {"DSBLO_GAMMA1": "10.0", "DSBLO_GAMMA2": "0.5", "DSBLO_MOMENTUM": "0.85", "DSBLO_EMA": "0.0"},
}


def run_with_config(tag: str, overrides: Dict[str, str], max_iters: int = 2000) -> Tuple[float, float]:
    env = os.environ.copy()
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "DSBLO_FIX_SIGMA": "1",
        "DSBLO_SIGMA": "0.0001",
        "DSBLO_PAPER_STEP": os.environ.get("DSBLO_PAPER_STEP", "1"),
        "DSBLO_OPTION_II": os.environ.get("DSBLO_OPTION_II", "1"),
        "ONLY_ALGOS": "F2CSA,DSBLO",
        "MAX_ITERS": str(max_iters),
        # Keep F2CSA at prior working settings so it improves consistently
        "F2CSA_NG": os.environ.get("F2CSA_NG", "60"),
        "F2CSA_LR": os.environ.get("F2CSA_LR", "1e-3"),
        "F2CSA_FIX_NG": "1",
        "F2CSA_FIXED_LR": "1",
    })
    env.update(overrides)

    proc = subprocess.run([
        "python", "-u", "comprehensive_convergence_test.py"
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", check=False)

    # Parse x_final gaps
    f2_gap = None
    ds_gap = None
    pat = re.compile(r"=>\s*(F2CSA|DSBLO)\s+gap\(x_final\)\s*=\s*([0-9eE+\-.]+)")
    for line in proc.stdout.splitlines():
        m = pat.search(line)
        if m:
            if m.group(1) == "F2CSA":
                f2_gap = float(m.group(2))
            else:
                ds_gap = float(m.group(2))
    return f2_gap, ds_gap


def main():
    max_iters = int(os.environ.get("COMPARE_MAX_ITERS", "2000"))
    results = {}
    for tag, cfg in CONFIGS.items():
        print(f"\n=== Smoke: {tag} ===")
        f2, ds = run_with_config(tag, cfg, max_iters=max_iters)
        results[tag] = (f2, ds)
        print(f"{tag}: F2CSA gap={f2} | DSBLO gap={ds} | |Δ|={None if (f2 is None or ds is None) else abs(ds-f2)}")

    # Pick best DS-BLO (closest to F2CSA)
    diffs = {
        tag: (abs(ds - f2) if (f2 is not None and ds is not None) else None)
        for tag, (f2, ds) in results.items()
    }
    diffs_sorted = sorted([(k, v) for k, v in diffs.items() if v is not None], key=lambda x: x[1])
    print("\n=== Ranking by |DSBLO - F2CSA| ===")
    for tag, diff in diffs_sorted:
        f2, ds = results[tag]
        print(f"{tag}: diff={diff:.6e}, F2CSA={f2:.6e}, DSBLO={ds:.6e}")


if __name__ == "__main__":
    main()

