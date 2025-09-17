#!/bin/bash
# Phoenix Slurm batch script for F2CSA/DS-BLO combined comparison grid sweep (CPU-AMD)
# Runs a parameter grid across alpha, eta, D, Ng as a Slurm job array

#SBATCH -J f2csa-dsblo-grid
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH -t 24:00:00
#SBATCH -A gts-kwang692
#SBATCH --qos=inferno
#SBATCH --partition=cpu-amd
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
# --array set dynamically below; keep placeholder here
#SBATCH --array=0-8

# Create logs directory
mkdir -p logs

# Environment from library source (prefer venv under python_packages if present)
VENV_DIR=/storage/scratch1/6/cphan36/python_packages
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    export PYTHONNOUSERSITE=1
else
    # Fallback: expose site-packages directly
    export PYTHONPATH=/storage/scratch1/6/cphan36/python_packages/lib/python3.10/site-packages:$PYTHONPATH
fi

# Workdir: cd to the submission directory (where you ran sbatch)
WORK_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$WORK_DIR"

# Ensure Python can find site packages and sibling legacy repo
export PYTHONPATH=/storage/scratch1/6/cphan36/python_packages/lib/python3.10/site-packages:$PYTHONPATH
export PYTHONPATH=$(cd "$WORK_DIR/.." && pwd)/BilevelLinearConstraints:$PYTHONPATH

# Parameter grids
ALPHA_LIST=(0.05 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.60)
ETA_LIST=(2e-4 1e-4 5e-5)
D_LIST=(0.08 0.05 0.03)
NG_LIST=(32 64 96)

NA=${#ALPHA_LIST[@]}
NE=${#ETA_LIST[@]}
ND=${#D_LIST[@]}
NN=${#NG_LIST[@]}
TOTAL=$((NA * NE * ND * NN))

# If controller didn't expand array bounds, print suggestion
if [[ -n "$SLURM_ARRAY_TASK_ID" ]]; then
  : # running inside array
else
  echo "Hint: submit with --array=0-$((TOTAL-1)) to cover all $TOTAL tasks"
fi

# Map array index -> grid indices
TID=${SLURM_ARRAY_TASK_ID}
E_IDX=$(( (TID / 1) % NE ))
D_IDX=$(( (TID / NE) % ND ))
N_IDX=$(( (TID / (NE*ND)) % NN ))
A_IDX=$(( (TID / (NE*ND*NN)) % NA ))

TASK_ALPHA=${ALPHA_LIST[$A_IDX]}
ETA=${ETA_LIST[$E_IDX]}
D=${D_LIST[$D_IDX]}
NG=${NG_LIST[$N_IDX]}

# Fixed params per your test command
T=500
SEED=1234
NOISE_STD=2e-3

# Echo task config
echo "Array task $SLURM_ARRAY_TASK_ID of $((TOTAL-1)): alpha=$TASK_ALPHA, D=$D, eta=$ETA, Ng=$NG, seed=$SEED"
echo "WORK_DIR=$WORK_DIR"
echo "PYTHONPATH=$PYTHONPATH"

# Preflight: verify legacy algorithms.py is importable
python - <<'PY'
import sys, importlib.util, traceback
print('Preflight: checking import of algorithms.DsBlo...')
spec = importlib.util.find_spec('algorithms')
print('find_spec algorithms ->', bool(spec), 'from', getattr(spec, 'origin', None))
try:
    from algorithms import DsBlo  # noqa: F401
    print('Preflight OK: legacy DsBlo importable')
except Exception:
    print('Preflight FAILED: could not import algorithms.DsBlo')
    traceback.print_exc()
    sys.exit(2)
PY
if [[ $? -ne 0 ]]; then
  echo "Aborting task $SLURM_ARRAY_TASK_ID due to missing legacy algorithms.py"
  exit 2
fi

# Run combined comparison: F2CSA + DS-BLO (normal) + DS-BLO (Legacy)
# Add warm-ll/keep-adam-state and use-crn-ul for smoothing and fair visual comparison
srun -n 1 python exact_sbatch_vs_dsblo.py \
  --T $T \
  --D $D \
  --eta $ETA \
  --Ng $NG \
  --alpha $TASK_ALPHA \
  --warm-ll --keep-adam-state \
  --problem-noise-std $NOISE_STD \
  --seed $SEED \
  --dsblo-opt II \
  --dsblo-sigma 0 \
  --dsblo-gamma1 5.0 \
  --dsblo-gamma2 1.0 \
  --dsblo-beta 0.6 \
  --dsblo-k 16 \
  --ul-scale symlog \
  --legacy-dsblo \
  --use-crn-ul \
  --plot-name exact_sbatch_vs_dsblo_alpha${TASK_ALPHA}_D${D}_eta${ETA}_Ng${NG}.png

# To submit FULL GRID: sbatch --array=0-$((TOTAL-1)) phoenix_f2csa_sweep.sbatch
# Example smaller sweep (alphas only): sbatch --array=0-$((NA-1)) phoenix_f2csa_sweep.sbatch
