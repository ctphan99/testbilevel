[33mcommit 681d1fe2194b04151230a6228a5a32bb97a12488[m
Author: ctphan99 <phanthuccac@gmail.com>
Date:   Mon Sep 15 16:44:57 2025 -0400

    Add F2CSA Algorithm 2 implementation with parameter sweep and Phoenix cluster support
    
    - f2csa_algorithm2_working.py: Main Algorithm 2 implementation with hypergradient tracking, UL loss monitoring, and convergence criteria
    - f2csa_algorithm.py: Parameter sweep launcher with alpha range 0.05-0.9 and empirical parameter grid
    - f2csa_algorithm_corrected_final.py: Algorithm 1 (F2CSAAlgorithm1Final) for stochastic hypergradient computation
    - problem.py: StronglyConvexBilevelProblem class with upper/lower objectives and constraints
    - phoenix_f2csa_sweep.sbatch: Slurm batch script for Phoenix cluster with parallel execution
    - algo2_warmstart.npy: Warm start point for faster convergence
    
    Features:
    - Warm start support for x0, lower-level solutions, and Adam optimizer state
    - Comprehensive parameter sweep (alpha, D, eta, N_g)
    - Convergence monitoring with early stopping
    - Hypergradient and UL loss plotting
    - Phoenix cluster parallel execution with custom Python environment

algo2_warmstart.npy
f2csa_algorithm.py
phoenix_f2csa_sweep.sbatch
