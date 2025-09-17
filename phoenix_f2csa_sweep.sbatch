#!/bin/bash
# Phoenix Slurm batch script for F2CSA/DS-BLO combined comparison alpha sweep (CPU-AMD)
# Uses cpu-amd partition (AMD Epyc 7713, 64 cores/node, 512GB RAM)

#SBATCH -J f2csa-dsblo-alpha
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH -t 24:00:00
#SBATCH -A gts-kwang692
#SBATCH --qos=inferno
#SBATCH --partition=cpu-amd
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
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

# Workdir
cd $SLURM_SUBMIT_DIR

# Ensure Python can find site packages
export PYTHONPATH=/storage/scratch1/6/cphan36/python_packages/lib/python3.10/site-packages:$PYTHONPATH

# Alpha sweep values per array task index
ALPHA_LIST=(0.05 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.60)
TASK_IDX=${SLURM_ARRAY_TASK_ID}
TASK_ALPHA=${ALPHA_LIST[$TASK_IDX]}

# Fixed params per your test command
T=500
D=0.08
ETA=2e-4
NG=32
SEED=1234
NOISE_STD=2e-3

# Echo task config
echo "Array task $TASK_IDX: alpha=$TASK_ALPHA, D=$D, eta=$ETA, Ng=$NG, seed=$SEED"

# Run combined comparison: F2CSA + DS-BLO (normal) + DS-BLO (Legacy)
srun -n 1 python exact_sbatch_vs_dsblo.py \
  --T $T \
  --D $D \
  --eta $ETA \
  --Ng $NG \
  --alpha $TASK_ALPHA \
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
  --plot-name exact_sbatch_vs_dsblo_f2csa_dsblo_legacy_normal_alpha${TASK_ALPHA}.png

# To submit: sbatch phoenix_f2csa_sweep.sbatch
