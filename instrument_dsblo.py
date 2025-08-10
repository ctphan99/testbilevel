import os
import re
import subprocess
from typing import Dict, List, Tuple

# Configurations to test
CONFIGS = {
    "C1_beta0.85_ema0_g120_g21": {"DSBLO_MOMENTUM": "0.85", "DSBLO_EMA": "0.0",  "DSBLO_GAMMA1": "20.0", "DSBLO_GAMMA2": "1.0"},
    "C2_beta0.85_ema085_g120_g21": {"DSBLO_MOMENTUM": "0.85", "DSBLO_EMA": "0.85", "DSBLO_GAMMA1": "20.0", "DSBLO_GAMMA2": "1.0"},
    "C3_beta0.9_ema0_g110_g20p5": {"DSBLO_MOMENTUM": "0.9",  "DSBLO_EMA": "0.0",  "DSBLO_GAMMA1": "10.0", "DSBLO_GAMMA2": "0.5"},
    "C4_beta0.9_ema085_g110_g20p5": {"DSBLO_MOMENTUM": "0.9",  "DSBLO_EMA": "0.85", "DSBLO_GAMMA1": "10.0", "DSBLO_GAMMA2": "0.5"},
}

MAX_ITERS = int(os.environ.get("INSTR_MAX_ITERS", "400"))
SAMPLE_EVERY = int(os.environ.get("INSTR_EVERY", "50"))

DEBUG_RE = re.compile(r"debug:\s*(.*)")
# Expect keys: gap=..., mnorm=..., eta=..., dx_norm=...
PAIR_RE = re.compile(r"(gap|mnorm|eta|dx_norm)=([0-9eE+\-.]+)")


def run_config(tag: str, overrides: Dict[str, str]) -> List[Tuple[int, Dict[str, float]]]:
    env = os.environ.copy()
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "ONLY_ALGOS": "DSBLO",
        "MAX_ITERS": str(MAX_ITERS),
        # DS-BLO paper-exact path and stochastic upper objective
        "DSBLO_PAPER_STEP": os.environ.get("DSBLO_PAPER_STEP", "1"),
        "DSBLO_OPTION_II": os.environ.get("DSBLO_OPTION_II", "1"),
        # Fixed sigma per requirement
        "DSBLO_FIX_SIGMA": "1",
        "DSBLO_SIGMA": "1e-4",
    })
    env.update(overrides)

    proc = subprocess.run([
        "python", "-u", "comprehensive_convergence_test.py"
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", check=False)

    # Parse debug lines and sample every SAMPLE_EVERY iterations
    out = proc.stdout.splitlines()
    samples: List[Tuple[int, Dict[str, float]]] = []
    iter_idx = -1
    for line in out:
        # Count DS-BLO iterations by looking for the DS-BLO status line that appears each loop
        if line.strip().startswith("DSBLO:") or line.strip().startswith("DS-BLO:"):
            iter_idx += 1
        m = DEBUG_RE.search(line)
        if m:
            # Attempt to parse pairs
            data_str = m.group(1)
            vals: Dict[str, float] = {}
            for k, v in PAIR_RE.findall(data_str):
                try:
                    vals[k] = float(v)
                except Exception:
                    pass
            if iter_idx >= 0 and iter_idx % SAMPLE_EVERY == 0 and vals:
                samples.append((iter_idx, vals))
    return samples


def main():
    print(f"Instrumented DS-BLO, MAX_ITERS={MAX_ITERS}, SAMPLE_EVERY={SAMPLE_EVERY}")
    for tag, cfg in CONFIGS.items():
        print(f"\n=== {tag} ===")
        samples = run_config(tag, cfg)
        if not samples:
            print("No samples parsed. Check logs/regex.")
            continue
        for it, vals in samples:
            gap = vals.get("gap")
            mnorm = vals.get("mnorm")
            eta = vals.get("eta")
            dx = vals.get("dx_norm")
            print(f"iter={it:4d} | gap={gap:.6e} | mnorm={mnorm:.3e} | eta={eta:.3e} | dx={dx:.3e}")


if __name__ == "__main__":
    main()

