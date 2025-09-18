#!/bin/bash
#SBATCH --job-name=f2csa_ng_comp
#SBATCH --partition=cpu-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=f2csa_ng_comp_%j.out
#SBATCH --error=f2csa_ng_comp_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-email@domain.com

# Load modules and set environment
module purge
module load python/3.9
module load gcc/9.3.0
module load openmpi/4.1.0

# Environment setup
export PYTHONIOENCODING='utf-8'
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

# Job information
echo "=========================================="
echo "F2CSA N_g Comprehensive Comparison"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Python Version: $(python --version)"
echo "=========================================="

# Check available memory and CPU
echo "System Information:"
echo "CPU Info: $(lscpu | grep 'Model name')"
echo "Memory Info: $(free -h)"
echo "Available CPUs: $(nproc)"
echo "=========================================="

# Create results directory
mkdir -p results_${SLURM_JOB_ID}
cd results_${SLURM_JOB_ID}

# Run multiple configurations
echo "Running F2CSA N_g comparison with multiple configurations..."

# Configuration 1: Standard comparison
echo "Configuration 1: Standard comparison (dim=100, T=10000)"
python ../f2csa_ng_comparison.py --dim 100 --T 10000 --seed 1234

# Configuration 2: Higher dimension test
echo "Configuration 2: Higher dimension test (dim=200, T=5000)"
python ../f2csa_ng_comparison.py --dim 200 --T 5000 --seed 5678

# Configuration 3: Longer run
echo "Configuration 3: Longer run (dim=100, T=20000)"
python ../f2csa_ng_comparison.py --dim 100 --T 20000 --seed 9012

echo "=========================================="
echo "All configurations completed at: $(date)"
echo "Results saved in: results_${SLURM_JOB_ID}/"
echo "=========================================="

# Copy results back to main directory
cp *.png ../
cp *.out ../
cp *.err ../

echo "Job completed successfully!"
