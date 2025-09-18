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

# GPU configuration
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Echo configuration
echo "DS-BLO GPU Test Configuration:"
echo "WORK_DIR=$WORK_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "PYTHONPATH=$PYTHONPATH"

# Check for GPU info (nvidia-smi might not be available in all environments)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
else
    echo "GPU: nvidia-smi not available, using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# Preflight: verify DS-BLO implementation is importable
python - <<'PY'
import sys, importlib.util, traceback
print('Preflight: checking DS-BLO adversarial training imports...')
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA device count: {torch.cuda.device_count()}')
        print(f'Current device: {torch.cuda.current_device()}')
        print(f'Device name: {torch.cuda.get_device_name()}')
    
    import torchvision
    print(f'Torchvision version: {torchvision.__version__}')
    
    from dsblo_adversarial_training import ResNet18, DSBLOAdversarialTraining, load_cifar_data
    print('Preflight OK: DS-BLO adversarial training modules importable')
except Exception as e:
    print('Preflight FAILED: could not import DS-BLO modules')
    traceback.print_exc()
    sys.exit(2)
PY

if [[ $? -ne 0 ]]; then
  echo "Aborting GPU test due to import failures"
  exit 2
fi

# Run simple GPU test first (no torchvision dependency)
echo "Running simple DS-BLO GPU test..."
srun -n 1 --account=gts-kwang692 python test_gpu_dsblo_simple.py

# If simple test passes, try full test
if [[ $? -eq 0 ]]; then
    echo "Simple test passed, trying full DS-BLO test..."
    srun -n 1 --account=gts-kwang692 python test_gpu_dsblo.py
else
    echo "Simple test failed, skipping full test"
    exit 1
fi

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
