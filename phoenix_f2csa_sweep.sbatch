#!/bin/bash
# Phoenix Slurm batch script for F2CSA Algorithm 2 sweep (CPU-AMD)
# Uses cpu-amd partition (AMD Epyc 7713, 64 cores/node, 512GB RAM)

#SBATCH -J f2csa-sweep
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -t 24:00:00
#SBATCH -A gts-kwang692
#SBATCH --qos=inferno
#SBATCH --partition=cpu-amd
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4

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

# Run parallel tasks - each task runs a different alpha/param combination
# Task 0: alpha=0.05, Task 1: alpha=0.10, etc.
ALPHA_LIST=(0.05 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.60 0.75 0.90)
TASK_ALPHA=${ALPHA_LIST[$SLURM_PROCID]}

# Each task runs with different parameters
D_LIST=(0.03 0.05 0.08)
ETA_LIST=(5e-5 1e-4 2e-4)
NG_LIST=(32 64 96)

# Cycle through parameter combinations based on task ID
D_IDX=$((SLURM_PROCID % 3))
ETA_IDX=$(((SLURM_PROCID / 3) % 3))
NG_IDX=$(((SLURM_PROCID / 9) % 3))

D=${D_LIST[$D_IDX]}
ETA=${ETA_LIST[$ETA_IDX]}
NG=${NG_LIST[$NG_IDX]}

echo "Task $SLURM_PROCID: alpha=$TASK_ALPHA, D=$D, eta=$ETA, Ng=$NG"

# Run single configuration (not sweep) for this task
srun -n 1 --exclusive python f2csa_algorithm2_working.py \
  --T 5000 \
  --D $D \
  --eta $ETA \
  --Ng $NG \
  --alpha $TASK_ALPHA \
  --warm-ll \
  --keep-adam-state \
  --plot-name algo2_task${SLURM_PROCID}_alpha${TASK_ALPHA}_D${D}_eta${ETA}_Ng${NG}.png &

wait

# Example follow-up single long run from the selected best params (uncomment and adjust):
# python f2csa_algorithm2_working.py \
#   --T 10000 --D 0.05 --eta 0.0001 --Ng 64 --alpha 0.1 \
#   --warm-ll --keep-adam-state \
#   --plot-name algo2_best_longrun.png
