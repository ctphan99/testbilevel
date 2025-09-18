#!/bin/bash
#SBATCH -J dsblo
#SBATCH -A gts-kwang692
#SBATCH -q inferno
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=4
#SBATCH --time=6:00:00
#SBATCH -o logs/dsblo_%j.out
#SBATCH -e logs/dsblo_%j.err

# DS-BLO adversarial training job
# Usage examples (set RUN_MODE as an environment variable when submitting):
#   sbatch --export=ALL,RUN_MODE=test   phoenix_dsblo_runs.sbatch
#   sbatch --export=ALL,RUN_MODE=full   phoenix_dsblo_runs.sbatch
#   sbatch --export=ALL,RUN_MODE=train_c10 phoenix_dsblo_runs.sbatch
#   sbatch --export=ALL,RUN_MODE=train_c100 phoenix_dsblo_runs.sbatch
#
# Modes:
#   test         → python run_dsblo_test.py --test-only
#   full         → python run_dsblo_test.py --full-experiment
#   train_c10    → python dsblo_adversarial_training.py --dataset cifar10 --epsilon 0.031372549 --epochs 100
#   train_c100   → python dsblo_adversarial_training.py --dataset cifar100 --epsilon 0.031372549 --epochs 100

set -euo pipefail

mkdir -p logs

# Diagnostics
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-}"

# Load modules if available (adjust to your cluster)
if command -v module &>/dev/null; then
  module purge || true
  module load cuda || true
  module load gcc || true
  module load python || true
fi

# Prefer python3 if python is unavailable
if ! command -v python &>/dev/null; then
  if command -v python3 &>/dev/null; then
    alias python=python3
  fi
fi

# Create and activate venv
VENV_DIR=.venv_dsblo
python -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate" || true

# Install deps
python -m pip install --upgrade pip wheel
if [ -f requirements_dsblo.txt ]; then
  python -m pip install -r requirements_dsblo.txt
else
  python -m pip install torch torchvision numpy matplotlib tqdm Pillow
fi

# Select run mode (default: test)
RUN_MODE=${RUN_MODE:-test}
echo "RUN_MODE=${RUN_MODE}"

case "$RUN_MODE" in
  test)
    echo "Running quick component test..."
    python run_dsblo_test.py --test-only
    ;;
  full)
    echo "Running full experiment..."
    python run_dsblo_test.py --full-experiment
    ;;
  train_c10)
    echo "Training on CIFAR-10 with ε=8/255 for 100 epochs..."
    python dsblo_adversarial_training.py --dataset cifar10 --epsilon 0.031372549 --epochs 100
    ;;
  train_c100)
    echo "Training on CIFAR-100 with ε=8/255 for 100 epochs..."
    python dsblo_adversarial_training.py --dataset cifar100 --epsilon 0.031372549 --epochs 100
    ;;
  *)
    echo "Unknown RUN_MODE=$RUN_MODE; defaulting to test"
    python run_dsblo_test.py --test-only
    ;;
 esac

echo "Job finished."
