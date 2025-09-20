# Gurobi Integration Guide for Bilevel Optimization

## Overview
This guide explains how to use Gurobi as a solver for the lower-level optimization problems in your bilevel optimization framework.

## Installation Status ✅
- Gurobi Optimizer: Installed (Version 12.0.3)
- Gurobi Python Package: Installed (gurobipy 12.0.3)
- License: Active (expires 2026-11-23)

## Available Solvers
Your `StronglyConvexBilevelProblem` class now supports four solvers:

1. **`'gurobi'`** - High-performance quadratic programming solver (RECOMMENDED)
2. **`'cvxpy'`** - General convex optimization with SCS backend
3. **`'pgd'`** - Projected gradient descent (first-order method)
4. **`'accurate'`** - Custom accurate solver implementing F2CSA Algorithm 1

## Basic Usage

### Simple Example
```python
from problem import StronglyConvexBilevelProblem
import torch

# Create problem
problem = StronglyConvexBilevelProblem(dim=10, device='cpu')

# Test point
x = torch.randn(10, dtype=torch.float64)

# Solve lower-level problem with Gurobi
y_opt, lambda_opt, info = problem.solve_lower_level(x, solver='gurobi')

print(f"Status: {info['status']}")
print(f"Objective value: {info['obj_value']}")
print(f"Solve time: {info['solve_time']}")
```

### Performance Comparison
Based on test results:
- **Gurobi**: ~0.05s (30x faster than CVXPY)
- **CVXPY**: ~1.6s
- **PGD**: Variable (depends on convergence)

## Gurobi-Specific Features

### High Precision Settings
The Gurobi solver is configured with:
- `NumericFocus = 3` - Maximum numerical precision
- `OptimalityTol = 1e-8` - High optimality tolerance
- `FeasibilityTol = 1e-8` - High feasibility tolerance
- `OutputFlag = 0` - Suppressed output for clean logs

### Dual Variables
Gurobi provides accurate dual variables (Lagrange multipliers) for:
- Lower bound constraints: `y ≥ -1`
- Upper bound constraints: `y ≤ 1`

### Error Handling
- Automatic fallback to PGD if Gurobi fails
- Graceful handling of numerical issues
- Clear error messages for debugging

## Integration with Your Algorithms

### DS-BLO Algorithm
```python
# In your DS-BLO implementation
y_opt, lambda_opt, info = problem.solve_lower_level(x, solver='gurobi')
```

### F2CSA Algorithm
```python
# In your F2CSA implementation
y_opt, lambda_opt, info = problem.solve_lower_level(x, solver='gurobi')
```

### SSIGD Algorithm
```python
# In your SSIGD implementation
y_opt, lambda_opt, info = problem.solve_lower_level(x, solver='gurobi')
```

## Advanced Configuration

### Custom Gurobi Parameters
To modify Gurobi settings, edit the `_solve_gurobi` method in `problem.py`:

```python
# Example: Enable output for debugging
model.setParam('OutputFlag', 1)

# Example: Set time limit
model.setParam('TimeLimit', 60)

# Example: Use different algorithm
model.setParam('Method', 2)  # 0=primal, 1=dual, 2=barrier
```

### Memory and Performance
For large-scale problems:
```python
# Increase memory limit
model.setParam('MemLimit', 8)  # 8GB

# Use multiple threads
model.setParam('Threads', 4)

# Enable presolve
model.setParam('Presolve', 2)  # Aggressive presolve
```

## Troubleshooting

### Common Issues

1. **License Problems**
   ```python
   import gurobipy as gp
   print(gp.gurobi.version())  # Check version
   ```

2. **Numerical Issues**
   - Increase `NumericFocus` parameter
   - Check problem conditioning
   - Verify positive definiteness

3. **Memory Issues**
   - Reduce problem dimension
   - Use `MemLimit` parameter
   - Consider sparse formulations

### Debug Mode
Enable verbose output:
```python
# In _solve_gurobi method, change:
model.setParam('OutputFlag', 1)
```

## Performance Tips

1. **Use Gurobi for Production**: 30x faster than CVXPY
2. **Warm Starts**: Gurobi automatically uses warm starts
3. **Parallel Processing**: Enable multi-threading for large problems
4. **Memory Management**: Monitor memory usage for large dimensions

## Example: Complete Workflow

```python
import torch
from problem import StronglyConvexBilevelProblem

# Create problem
problem = StronglyConvexBilevelProblem(dim=20, device='cpu')

# Test multiple solvers
x = torch.randn(20, dtype=torch.float64)

solvers = ['gurobi', 'cvxpy', 'pgd']
for solver in solvers:
    y_opt, lambda_opt, info = problem.solve_lower_level(x, solver=solver)
    print(f"{solver}: {info['solve_time']:.4f}s, obj={info.get('obj_value', 'N/A')}")
```

## Next Steps

1. **Update your algorithms** to use `solver='gurobi'` by default
2. **Benchmark performance** on your specific problem sizes
3. **Tune parameters** for your use case
4. **Monitor memory usage** for large-scale problems

## Support

- Gurobi Documentation: https://www.gurobi.com/documentation/
- Gurobi Python API: https://www.gurobi.com/documentation/current/refman/py_python_api_overview.html
- Your project: Check `test_gurobi_integration.py` for examples
