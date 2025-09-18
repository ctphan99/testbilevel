#!/bin/bash
# Phoenix Slurm batch script for DS-BLO GPU Test (Quick)
# Quick GPU test to verify DS-BLO adversarial training works on GPU

#SBATCH -J dsblo-gpu-test
#SBATCH -o logs/%x-%A.out
#SBATCH -e logs/%x-%A.err
#SBATCH -t 0:30:00
#SBATCH -A gts-kwang692
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2

# Create logs directory
mkdir -p logs

# Environment setup
VENV_DIR=/storage/scratch1/6/cphan36/python_packages
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    export PYTHONNOUSERSITE=1
else
    # Fallback: expose site-packages directly
    export PYTHONPATH=/storage/scratch1/6/cphan36/python_packages/lib/python3.10/site-packages:$PYTHONPATH
fi

# Workdir: cd to the submission directory
WORK_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$WORK_DIR"

# Ensure Python can find packages
export PYTHONPATH=/storage/scratch1/6/cphan36/python_packages/lib/python3.10/site-packages:$PYTHONPATH

# GPU configuration
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Echo configuration
echo "DS-BLO GPU Test Configuration:"
echo "WORK_DIR=$WORK_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"

# Run GPU test
echo "Running DS-BLO GPU test..."
srun -n 1 --account=gts-kwang692 python test_gpu_dsblo.py

# Check if test passed
if [[ $? -eq 0 ]]; then
    echo "✅ DS-BLO GPU test PASSED!"
    echo "Ready for full training experiments"
else
    echo "❌ DS-BLO GPU test FAILED!"
    echo "Check logs for details"
    exit 1
fi

echo "GPU test completed successfully"
