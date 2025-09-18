#!/bin/bash
#SBATCH --job-name=f2csa_ng_comparison
#SBATCH --partition=cpu-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=f2csa_ng_comparison_%j.out
#SBATCH --error=f2csa_ng_comparison_%j.err

# Load required modules (adjust based on your cluster)
module load python/3.9
module load gcc/9.3.0

# Set environment variables
export PYTHONIOENCODING='utf-8'
export OMP_NUM_THREADS=8

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Activate virtual environment if needed (uncomment and adjust path)
# source /path/to/your/venv/bin/activate

# Run the F2CSA N_g comparison
echo "Starting F2CSA N_g comparison..."
echo "Configuration: dim=100, T=10000, seed=1234"
echo "Testing: OSQP vs SCS with N_g=1 vs N_g=10"
echo "=========================================="

python f2csa_ng_comparison.py --dim 100 --T 10000 --seed 1234

echo "=========================================="
echo "Job completed at: $(date)"
echo "Output files:"
echo "- f2csa_ng_comparison_dim100_T10k.png"
echo "- f2csa_ng_comparison_${SLURM_JOB_ID}.out"
echo "- f2csa_ng_comparison_${SLURM_JOB_ID}.err"
