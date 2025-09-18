#!/bin/bash
#SBATCH -J noise-test-5k
#SBATCH -o logs/%x-%A.out
#SBATCH -e logs/%x-%A.err
#SBATCH -t 2:00:00
#SBATCH -A gts-kwang692
#SBATCH --qos=inferno
#SBATCH --partition=cpu-amd
#SBATCH --mem=32G
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

# Workdir: cd to the submission directory (where you ran sbatch)
WORK_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$WORK_DIR"

# Ensure Python can find site packages and sibling legacy repo
export PYTHONPATH=/storage/scratch1/6/cphan36/python_packages/lib/python3.10/site-packages:$PYTHONPATH
export PYTHONPATH=$(cd "$WORK_DIR/.." && pwd)/BilevelLinearConstraints:$PYTHONPATH

# Echo task config
echo "Running noise test with 5k iterations"
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
  echo "Aborting task due to missing legacy algorithms.py"
  exit 2
fi

# Run the comparison with specified parameters
srun -n 1 python exact_sbatch_vs_dsblo.py \
  --T 5000 \
  --alpha 0.05 \
  --D 0.02 \
  --eta 1e-4 \
  --Ng 64 \
  --with-ssigd \
  --warm-ll \
  --plot-name noise_test_5k.png

echo "Job completed successfully!"
