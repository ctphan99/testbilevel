#!/bin/bash
# Submission script for F2CSA N_g comparison jobs

echo "F2CSA N_g Comparison Job Submission"
echo "=================================="

# Check if we're on a SLURM cluster
if command -v sbatch &> /dev/null; then
    echo "SLURM detected. Submitting batch job..."
    
    # Submit the basic comparison job
    echo "Submitting basic F2CSA N_g comparison job..."
    sbatch f2csa_ng_comparison.sbatch
    
    echo "Job submitted successfully!"
    echo "Check status with: squeue -u \$USER"
    echo "Check output with: tail -f f2csa_ng_comparison_*.out"
    
elif command -v salloc &> /dev/null; then
    echo "SLURM detected. Starting interactive allocation..."
    
    # Interactive allocation
    salloc --partition=cpu-large \
           --nodes=1 \
           --ntasks=1 \
           --cpus-per-task=8 \
           --mem=32G \
           --time=02:00:00 \
           --job-name=f2csa_ng_interactive
    
else
    echo "SLURM not detected. Running locally..."
    echo "Setting environment and running comparison..."
    
    export PYTHONIOENCODING='utf-8'
    python f2csa_ng_comparison.py --dim 100 --T 10000 --seed 1234
fi
