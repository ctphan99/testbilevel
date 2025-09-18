#!/bin/bash
# Phoenix Slurm batch script for noise test with 5k iterations
# Runs exact_sbatch_vs_dsblo.py with specific parameters

#SBATCH -J noise-test-5k
#SBATCH -o logs/%x-%A.out
#SBATCH -e logs/%x-%A.err
#SBATCH -t 12:00:00
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

# Ensure Python can find site packages and legacy repo
export PYTHONPATH=/storage/scratch1/6/cphan36/python_packages/lib/python3.10/site-packages:$PYTHONPATH
export PYTHONPATH=$WORK_DIR/BilevelLinearConstraints:$PYTHONPATH

# Echo task config
echo "Running noise test with 5k iterations"
echo "WORK_DIR=$WORK_DIR"
echo "PYTHONPATH=$PYTHONPATH"

# Preflight: verify main script can run (legacy algorithms.py is optional)
python - <<'PY'
import sys, importlib.util, traceback
print('Preflight: checking main script imports...')
try:
    # Test main imports
    from problem import StronglyConvexBilevelProblem
    from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
    from f2csa_algorithm2_working import F2CSAAlgorithm2Working
    from dsblo_conservative import DSBLOConservative
    from dsblo_optII import DSBLOOptII
    from ssigd_correct_final import CorrectSSIGD
    print('Preflight OK: main script imports successful')
except Exception as e:
    print('Preflight FAILED: could not import main modules')
    traceback.print_exc()
    sys.exit(2)
PY
if [[ $? -ne 0 ]]; then
  echo "Aborting task due to missing main modules"
  exit 2
fi

# Run the specific test command
python exact_sbatch_vs_dsblo.py \
  --T 5000 \
  --alpha 0.05 \
  --D 0.02 \
  --eta 1e-4 \
  --Ng 64 \
  --with-ssigd \
  --warm-ll \
  --plot-name noise_test_5k.png

echo "Task completed successfully"
