#!/bin/bash
#SBATCH -J dsblo_autotune
#SBATCH -o dsblo_autotune.%j.out
#SBATCH -e dsblo_autotune.%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00

set -euo pipefail

# User-configurable base problem settings (align noise and start point via comparison script)
T=${T:-5000}                   # iterations per attempt (increase for longer runs)
D=${D:-0.08}
ETA=${ETA:-2e-4}
NG=${NG:-32}
ALPHA=${ALPHA:-0.6}
DIM=${DIM:-5}
CONS=${CONS:-3}
PROBLEM_NOISE_STD=${PROBLEM_NOISE_STD:-2e-3}

# DS-BLO base settings and grids to search
DSBLO_SIGMA=${DSBLO_SIGMA:-2e-3}
GAMMA1_GRID=${GAMMA1_GRID:-"5.0 7.5 3.0"}
GAMMA2=${GAMMA2:-1.0}
BETA_GRID=${BETA_GRID:-"0.6 0.7"}
K_GRID=${K_GRID:-"16 24 32"}
ETA_CAP_GRID=${ETA_CAP_GRID:-"1e-3 2e-3"}
NOISE_DECAY=${NOISE_DECAY:-0.995}
NOISE_MIN=${NOISE_MIN:-1e-4}

# Success threshold: accept when DS-BLO final UL <= F2CSA final UL + TOL
TOL=${TOL:-0.1}

OUT_DIR=${OUT_DIR:-autotune_runs}
mkdir -p "$OUT_DIR"

echo "[INFO] Starting F2CSA vs DS-BLO autotune"
echo "[INFO] Problem noise std: $PROBLEM_NOISE_STD; T=$T, D=$D, eta=$ETA, Ng=$NG, alpha=$ALPHA, dim=$DIM, cons=$CONS"

RESULTS_CSV="$OUT_DIR/results.csv"
echo "gamma1,beta,k,eta_cap,f2csa_final_ul,dsblo_final_ul,win,plot" > "$RESULTS_CSV"

best_plot=""
best_row=""
best_gap=1e18

for g1 in $GAMMA1_GRID; do
  for b in $BETA_GRID; do
    for k in $K_GRID; do
      for ec in $ETA_CAP_GRID; do
        stamp=$(date +%Y%m%d-%H%M%S)-g1_${g1}-b_${b}-k_${k}-ec_${ec}
        plot="$OUT_DIR/exact_sbatch_vs_dsblo_${stamp}.png"

        echo "[INFO] Trying DS-BLO params: gamma1=$g1 beta=$b k=$k eta_cap=$ec"
        # Run the shared comparison so both share the same problem and x0
        # Expect summary lines printed by exact_sbatch_vs_dsblo.py
        PYTHONUNBUFFERED=1 python3 exact_sbatch_vs_dsblo.py \
          --T "$T" --D "$D" --eta "$ETA" --Ng "$NG" --alpha "$ALPHA" \
          --dim "$DIM" --constraints "$CONS" --problem-noise-std "$PROBLEM_NOISE_STD" \
          --dsblo-opt II --dsblo-sigma "$DSBLO_SIGMA" \
          --dsblo-gamma1 "$g1" --dsblo-gamma2 "$GAMMA2" --dsblo-beta "$b" \
          --dsblo-k "$k" --dsblo-eta-cap "$ec" --dsblo-noise-decay "$NOISE_DECAY" --dsblo-noise-min "$NOISE_MIN" \
          --plot-name "$plot" | tee "$OUT_DIR/run_${stamp}.log"

        # Parse final ULs from the summary
        f2csa_ul=$(awk '/Final UL Loss/{print $4}' "$OUT_DIR/run_${stamp}.log" | head -n1)
        dsblo_ul=$(awk '/Final UL Loss/{print $5}' "$OUT_DIR/run_${stamp}.log" | tail -n1)

        # Strip possible trailing characters
        f2csa_ul=${f2csa_ul//,/}
        dsblo_ul=${dsblo_ul//,/}

        # Decide winner
        win="DS-BLO"
        awk "BEGIN{exit !($dsblo_ul <= $f2csa_ul)}" || win="F2CSA"

        echo "$g1,$b,$k,$ec,$f2csa_ul,$dsblo_ul,$win,$plot" >> "$RESULTS_CSV"

        # Keep best (closest) DS-BLO to F2CSA from above
        gap=$(python3 - <<PY
f=${f2csa_ul}; d=${dsblo_ul}
print(abs(d-f))
PY
)
        smaller=$(python3 - <<PY
best=${best_gap}
cur=${gap}
print(1 if float(cur) < float(best) else 0)
PY
)
        if [[ "$smaller" == "1" ]]; then
          best_gap=$gap
          best_plot=$plot
          best_row="$g1,$b,$k,$ec,$f2csa_ul,$dsblo_ul,$win,$plot"
        fi

        # Stop early if DS-BLO within tolerance of F2CSA
        within=$(python3 - <<PY
f=${f2csa_ul}; d=${dsblo_ul}; tol=${TOL}
print(1 if d <= f + tol else 0)
PY
)
        if [[ "$within" == "1" ]]; then
          echo "[SUCCESS] DS-BLO matched F2CSA within TOL=$TOL. Params: gamma1=$g1 beta=$b k=$k eta_cap=$ec"
          echo "[SUCCESS] Best plot: $plot"
          echo "$best_row" > "$OUT_DIR/best.txt"
          exit 0
        fi

      done
    done
  done
done

echo "[DONE] Exhausted grid. Best result gap=$best_gap"
echo "$best_row" > "$OUT_DIR/best.txt"
echo "Best plot: $best_plot"

