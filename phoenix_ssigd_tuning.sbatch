#!/bin/bash
# Phoenix Slurm batch script for SSIGD parameter tuning (CPU-Large)
# Runs a parameter grid across beta and mu_F to find optimal parameters

#SBATCH -J ssigd-tuning
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH -A gts-kwang692
#SBATCH --qos=inferno
#SBATCH --partition=cpu-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-35

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
export PYTHONPATH="/storage/scratch1/6/cphan36/python_packages/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH=$(cd "$WORK_DIR/.." && pwd)/BilevelLinearConstraints:$PYTHONPATH

# Add CVXPYLayer to PYTHONPATH
export PYTHONPATH="/storage/scratch1/6/cphan36/cvxpylayers:$PYTHONPATH"

# Set environment variables
export PYTHONIOENCODING='utf-8'
export OMP_NUM_THREADS=8

# Gurobi license and path setup
export GRB_LICENSE_FILE=/storage/scratch1/6/cphan36/.gurobi/gurobi.lic
export GUROBI_HOME=/storage/scratch1/6/cphan36/gurobi1000/linux64
export PATH=$GUROBI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$GUROBI_HOME/lib/python3.10_utf32:$PYTHONPATH

# Parameter grids for SSIGD tuning
BETA_LIST=(0.0001 0.0005 0.001 0.005 0.01 0.05)
MU_F_LIST=(0.01 0.05 0.1 0.5 1.0 2.0)

NB=${#BETA_LIST[@]}
NM=${#MU_F_LIST[@]}
TOTAL=$((NB * NM))

# If controller didn't expand array bounds, print suggestion
if [[ -n "$SLURM_ARRAY_TASK_ID" ]]; then
  : # running inside array
else
  echo "Hint: submit with --array=0-$((TOTAL-1)) to cover all $TOTAL tasks"
fi

# Map array index -> grid indices
TID=${SLURM_ARRAY_TASK_ID}
B_IDX=$((TID % NB))
M_IDX=$((TID / NB))

BETA=${BETA_LIST[$B_IDX]}
MU_F=${MU_F_LIST[$M_IDX]}

# Fixed params
T=1000
DIM=100
SEED=1234

# Echo task config
echo "Array task $SLURM_ARRAY_TASK_ID of $((TOTAL-1)): beta=$BETA, mu_F=$MU_F, dim=$DIM, T=$T, seed=$SEED"
echo "WORK_DIR=$WORK_DIR"
echo "PYTHONPATH=$PYTHONPATH"
echo "GRB_LICENSE_FILE=$GRB_LICENSE_FILE"

# Preflight: verify required modules are importable
python - <<'PY'
import sys, importlib.util, traceback
print('Preflight: checking imports...')

# Check Gurobi
try:
    import gurobipy as gp
    print('Preflight OK: Gurobi importable')
except Exception as e:
    print('Preflight FAILED: could not import Gurobi:', e)
    sys.exit(2)

# Check required modules
try:
    import torch
    import numpy as np
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    print('Preflight OK: PyTorch, NumPy, CVXPY importable')
except Exception as e:
    print('Preflight FAILED: could not import required modules:', e)
    traceback.print_exc()
    sys.exit(2)

# Check our modules
try:
    from problem import StronglyConvexBilevelProblem
    from ssigd_correct_final import CorrectSSIGD
    print('Preflight OK: project modules importable')
except Exception as e:
    print('Preflight FAILED: could not import project modules:', e)
    traceback.print_exc()
    sys.exit(2)
PY

if [[ $? -ne 0 ]]; then
  echo "Aborting task $SLURM_ARRAY_TASK_ID due to import failures"
  exit 2
fi

# Run SSIGD parameter tuning
srun -n 1 python - <<PY
import torch
import numpy as np
import sys
import os
from problem import StronglyConvexBilevelProblem
from ssigd_correct_final import CorrectSSIGD

def run_ssigd_tuning():
    # Set random seed
    seed = $SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Parameters
    dim = $DIM
    T = $T
    beta = $BETA
    mu_F = $MU_F
    
    print(f"SSIGD Parameter Tuning")
    print(f"Dimension: {dim}, Iterations: {T}")
    print(f"Beta: {beta}, mu_F: {mu_F}, Seed: {seed}")
    print()
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=dim, device='cpu')
    x0 = torch.randn(dim, dtype=torch.float64) * 0.1
    
    # Initial loss
    y0, _, _ = problem.solve_lower_level(x0, solver='gurobi')
    initial_loss = problem.upper_objective(x0, y0).item()
    print(f"Initial UL Loss: {initial_loss:.6f}")
    
    # Run SSIGD
    try:
        ssigd = CorrectSSIGD(problem)
        result = ssigd.solve(T=T, beta=beta, x0=x0, diminishing=True, mu_F=mu_F)
        
        final_loss = result['final_loss']
        final_grad = result['final_grad_norm']
        improvement = initial_loss - final_loss
        reduction_pct = (improvement / initial_loss * 100) if initial_loss != 0 else 0
        
        print(f"Results:")
        print(f"  Final Loss: {final_loss:.6f}")
        print(f"  Final Grad: {final_grad:.6f}")
        print(f"  Improvement: {improvement:.6f}")
        print(f"  Reduction: {reduction_pct:.2f}%")
        
        # Save results to file
        output_file = f"ssigd_results_beta{beta}_muF{mu_F}.txt"
        with open(output_file, 'w') as f:
            f.write(f"beta={beta}\n")
            f.write(f"mu_F={mu_F}\n")
            f.write(f"initial_loss={initial_loss:.6f}\n")
            f.write(f"final_loss={final_loss:.6f}\n")
            f.write(f"final_grad={final_grad:.6f}\n")
            f.write(f"improvement={improvement:.6f}\n")
            f.write(f"reduction_pct={reduction_pct:.2f}\n")
        
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"SSIGD failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_ssigd_tuning()
PY

echo "Task $SLURM_ARRAY_TASK_ID completed: beta=$BETA, mu_F=$MU_F"

# To submit FULL GRID: sbatch --array=0-35 phoenix_ssigd_tuning.sbatch
# This covers 6 beta values × 6 mu_F values = 36 total tasks
