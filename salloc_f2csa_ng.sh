#!/bin/bash
# Interactive SLURM allocation script for F2CSA N_g comparison
# Usage: ./salloc_f2csa_ng.sh

echo "Requesting CPU-large node allocation..."
echo "Configuration: 1 node, 8 CPUs, 32GB RAM, 2 hours"

# Request allocation
salloc --partition=cpu-large \
       --nodes=1 \
       --ntasks=1 \
       --cpus-per-task=8 \
       --mem=32G \
       --time=02:00:00 \
       --job-name=f2csa_ng_interactive

echo "Allocation completed or failed."
echo "If successful, you can now run:"
echo "  export PYTHONIOENCODING='utf-8'"
echo "  python f2csa_ng_comparison.py --dim 100 --T 10000 --seed 1234"
