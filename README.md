# Constrained Stochastic Bilevel Optimization

A comprehensive implementation and comparison of state-of-the-art algorithms for constrained stochastic bilevel optimization problems.

## Overview

This repository contains implementations of three major bilevel optimization algorithms:

- **F2CSA** (Fully First-order Constrained Stochastic Approximation) - Enhanced with all theoretical improvements
- **SSIGD** (Stochastic Smooth Inexact Gradient Descent) - Smoothed gradient approach
- **DS-BLO** (Doubly Stochastic Bilevel Optimization) - Doubly stochastic with momentum

## Problem Formulation

We solve constrained bilevel optimization problems of the form:

```
Upper level: min_x F(x,noise) = f(x, y*(x),noise)
Lower level: y*(x) ∈ argmin_y g(x,y,noise) s.t. h(x,y) ≤ 0
```

## Usage

```python
# Create strongly convex bilevel problem
problem = StronglyConvexBilevelProblem(dim=100, num_constraints=3, device='cpu')

# Initialize algorithms
f2csa = F2CSA(problem, N_g=5, alpha=0.3)  # Enhanced configuration
ssigd = SSIGD(problem, smoothing_samples=5, epsilon=0.01)
dsblo = DSBLO(problem, momentum=0.9, sigma=0.01)

# Run optimization
f2csa_result = f2csa.optimize(max_iterations=1000, convergence_threshold=0.1)
ssigd_result = ssigd.optimize(max_iterations=1000, convergence_threshold=0.1)
dsblo_result = dsblo.optimize(max_iterations=1000, convergence_threshold=0.1)

# Compare results
run_comprehensive_comparison()
```


## Citation

If you use this code in your research, please cite:

```bibtex
@software{constraintstochasticbilevel2024,
  title={Constrained Stochastic Bilevel Optimization: Enhanced F2CSA Implementation and Comprehensive Algorithm Comparison},
  author={ctphan99},
  year={2024},
  url={https://github.com/ctphan99/constraintstochasticbilevel}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.
