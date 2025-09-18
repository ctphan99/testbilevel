#!/bin/bash
#SBATCH -J dim-runs-5k
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH -t 06:00:00
#SBATCH -A gts-kwang692
#SBATCH --qos=inferno
#SBATCH --partition=cpu-amd
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2

set -euo pipefail

mkdir -p logs

# Optional env (adjust if you use a venv on Phoenix)
VENV_DIR=/storage/scratch1/6/cphan36/python_packages
if [ -f "$VENV_DIR/bin/activate" ]; then
  source "$VENV_DIR/bin/activate"
  export PYTHONNOUSERSITE=1
else
  export PYTHONPATH=/storage/scratch1/6/cphan36/python_packages/lib/python3.10/site-packages:$PYTHONPATH
fi

# Workdir: submission directory
WORK_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$WORK_DIR"

echo "WORK_DIR=$WORK_DIR"
echo "PYTHONPATH=$PYTHONPATH"

# Dimension set for three runs
DIMS=(10 100 500)
DIM=${DIMS[$SLURM_ARRAY_TASK_ID]}

echo "Running dim=$DIM (array id $SLURM_ARRAY_TASK_ID)"

# Remove any warm start to regenerate
rm -f algo2_warmstart.npy || true

# Launch the run (no srun needed for single task)
python exact_sbatch_vs_dsblo.py \
  --T 5000 \
  --dim $DIM \
  --alpha 0.05 \
  --D 0.03 \
  --eta 0.1 \
  --Ng 32 \
  --with-ssigd \
  --seed 1234 \
  --use-crn-ul \
  --dsblo-k 32 \
  --dsblo-beta 0.3 \
  --dsblo-eta-cap 1e-2 \
  --dsblo-sigma 0 \
  --plot-name comparison_box_quadratic_T5000_dim${DIM}.png

echo "Done dim=$DIM"

# Submit with: sbatch phoenix_dim_runs.sbatch

