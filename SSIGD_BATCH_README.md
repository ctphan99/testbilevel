# SSIGD Parameter Tuning Batch Setup

This branch contains the complete infrastructure for tuning SSIGD parameters using Phoenix cluster batch jobs.

## Files Overview

### Core Batch Files
- **`phoenix_ssigd_tuning.sbatch`** - Main Slurm batch script for parameter tuning
- **`submit_ssigd_tuning.sh`** - Easy submission script with instructions
- **`analyze_ssigd_results.py`** - Results analysis and visualization

### Testing Files
- **`test_ssigd_batch_local.py`** - Local testing with smaller parameter grid
- **`tune_ssigd_parameters.py`** - Parameter tuning implementation

### Updated Implementation
- **`ssigd_correct_final.py`** - Updated SSIGD with proper implicit gradient

## Parameter Grid

The batch job tests **36 parameter combinations**:
- **Beta values**: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
- **mu_F values**: [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

## Usage

### 1. Submit Batch Job
```bash
# Make submission script executable
chmod +x submit_ssigd_tuning.sh

# Submit the batch job
./submit_ssigd_tuning.sh
```

### 2. Monitor Jobs
```bash
# Check job status
squeue -u $USER

# View specific job logs
tail -f logs/ssigd-tuning-<JOB_ID>_<ARRAY_ID>.out
```

### 3. Analyze Results
```bash
# After all jobs complete, run analysis
python analyze_ssigd_results.py
```

## Configuration

### Batch Job Settings
- **Partition**: cpu-large
- **Resources**: 1 node, 8 CPUs, 32GB RAM, 2 hours
- **Array**: 0-35 (36 total tasks)
- **Iterations**: 10,000 per task
- **Dimension**: 100
- **Seed**: 1234

### Environment Setup
- **Gurobi License**: `/storage/scratch1/6/cphan36/.gurobi/gurobi.lic`
- **Python Path**: Includes Gurobi and project modules
- **Threading**: OMP_NUM_THREADS=8

## Expected Output

Each batch job will produce:
- **Result files**: `ssigd_results_beta*_muF*.txt`
- **Log files**: `logs/ssigd-tuning-<JOB_ID>_<ARRAY_ID>.out/err`
- **Analysis plot**: `ssigd_parameter_analysis.png`
- **CSV results**: `ssigd_tuning_results.csv`

## Key Features

1. **Comprehensive Parameter Sweep**: Tests 36 combinations systematically
2. **Robust Error Handling**: Preflight checks and fallback mechanisms
3. **Detailed Logging**: Step-by-step progress tracking
4. **Visual Analysis**: Heatmaps and scatter plots for parameter relationships
5. **Easy Submission**: One-command batch job submission

## Troubleshooting

### Common Issues
1. **Import Errors**: Check PYTHONPATH and Gurobi installation
2. **Memory Issues**: Reduce T (iterations) or dim (dimension)
3. **Timeout**: Increase --time in batch script
4. **License Issues**: Verify GRB_LICENSE_FILE path

### Debug Mode
Run locally first with smaller parameters:
```bash
python test_ssigd_batch_local.py
```

## Results Interpretation

The analysis script will:
1. **Rank parameters** by final loss
2. **Create heatmaps** showing parameter relationships
3. **Identify optimal** beta and mu_F values
4. **Generate visualizations** for easy interpretation

Look for the **best parameters** that minimize the upper-level loss while maintaining convergence.
