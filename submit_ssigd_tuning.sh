#!/bin/bash
# Submit SSIGD parameter tuning batch job

echo "🚀 Submitting SSIGD Parameter Tuning Batch Job"
echo "=============================================="

# Check if batch script exists
if [ ! -f "phoenix_ssigd_tuning.sbatch" ]; then
    echo "❌ Error: phoenix_ssigd_tuning.sbatch not found!"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Submit the batch job
echo "Submitting batch job with array 0-35 (36 total tasks)..."
echo "This will test 6 beta values × 6 mu_F values = 36 parameter combinations"
echo ""

sbatch phoenix_ssigd_tuning.sbatch

echo ""
echo "✅ Batch job submitted successfully!"
echo ""
echo "To monitor job status:"
echo "  squeue -u $USER"
echo ""
echo "To check specific job:"
echo "  squeue -j <JOB_ID>"
echo ""
echo "To view logs:"
echo "  ls logs/"
echo "  tail -f logs/ssigd-tuning-<JOB_ID>_<ARRAY_ID>.out"
echo ""
echo "After all jobs complete, run analysis:"
echo "  python analyze_ssigd_results.py"