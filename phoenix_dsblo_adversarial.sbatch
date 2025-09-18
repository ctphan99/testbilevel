#!/bin/bash
# Phoenix Slurm batch script for DS-BLO Adversarial Training (GPU-H100)
# Runs DS-BLO adversarial training experiments on CIFAR-10/100 with ResNet-18

#SBATCH -J dsblo-adversarial
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH -t 6:00:00
#SBATCH -A gts-kwang692
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --array=0-5

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

# Parameter grids for DS-BLO adversarial training
DATASET_LIST=(cifar10 cifar100)
EPSILON_LIST=(8/255 16/255)
EPOCHS_LIST=(50 100)

# Map array index to parameters
TID=${SLURM_ARRAY_TASK_ID}
DATASET_IDX=$((TID % 2))
EPSILON_IDX=$((TID / 2 % 2))
EPOCHS_IDX=$((TID / 4 % 2))

DATASET=${DATASET_LIST[$DATASET_IDX]}
EPSILON=${EPSILON_LIST[$EPSILON_IDX]}
EPOCHS=${EPOCHS_LIST[$EPOCHS_IDX]}

# Convert epsilon to decimal for filename (handle both 8/255 and 16/255)
if [[ "$EPSILON" == "8/255" ]]; then
    EPSILON_STR="0314"  # 8/255 ≈ 0.0314
elif [[ "$EPSILON" == "16/255" ]]; then
    EPSILON_STR="0627"  # 16/255 ≈ 0.0627
else
    # Fallback: try bc if available, otherwise use a default
    if command -v bc &> /dev/null; then
        EPSILON_DEC=$(echo "scale=6; $EPSILON" | bc -l)
        EPSILON_STR=$(echo $EPSILON_DEC | sed 's/\.//g' | cut -c1-4)
    else
        EPSILON_STR="0314"  # Default fallback
    fi
fi

# Echo task configuration
echo "Array task $SLURM_ARRAY_TASK_ID: dataset=$DATASET, epsilon=$EPSILON, epochs=$EPOCHS"
echo "WORK_DIR=$WORK_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

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
    
    from dsblo_adversarial_training import ResNet18, DSBLOAdversarialTraining, load_cifar_data
    print('Preflight OK: DS-BLO adversarial training modules importable')
except Exception as e:
    print('Preflight FAILED: could not import DS-BLO modules')
    traceback.print_exc()
    sys.exit(2)
PY

if [[ $? -ne 0 ]]; then
  echo "Aborting task $SLURM_ARRAY_TASK_ID due to import failures"
  exit 2
fi

# Run DS-BLO adversarial training
echo "Starting DS-BLO adversarial training..."
echo "Dataset: $DATASET"
echo "Epsilon: $EPSILON"
echo "Epochs: $EPOCHS"

# Create results directory
RESULTS_DIR="dsblo_results_${DATASET}_eps${EPSILON_STR}_epochs${EPOCHS}"
mkdir -p "$RESULTS_DIR"

# Run the training
srun -n 1 --account=gts-kwang692 python dsblo_adversarial_training.py \
  --dataset "$DATASET" \
  --epsilon "$EPSILON" \
  --epochs "$EPOCHS" \
  --batch_size 128 \
  --patience 10 \
  --save_dir "$RESULTS_DIR" \
  --device cuda

# Check if training completed successfully
if [[ $? -eq 0 ]]; then
    echo "DS-BLO adversarial training completed successfully"
    echo "Results saved to: $RESULTS_DIR"
    
    # List output files
    echo "Output files:"
    ls -la "$RESULTS_DIR"
    
    # Show final results if available
    if [[ -f "$RESULTS_DIR/results.json" ]]; then
        echo "Final results:"
        python -c "
import json
with open('$RESULTS_DIR/results.json', 'r') as f:
    results = json.load(f)
    print(f'Clean Accuracy: {results[\"final_test_clean_acc\"]:.2f}%')
    print(f'Robust Accuracy: {results[\"final_test_robust_acc\"]:.2f}%')
    print(f'Epochs trained: {results[\"epochs_trained\"]}')
"
    fi
else
    echo "DS-BLO adversarial training failed"
    exit 1
fi

echo "Task $SLURM_ARRAY_TASK_ID completed"
