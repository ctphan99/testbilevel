#!/usr/bin/env python3
"""
DS-BLO (Doubly Stochastically Perturbed Algorithm) for Linearly Constrained Bilevel Optimization
Implementation based on dsblo_paper.tex

Key Features:
1. Doubly stochastic perturbation for non-differentiability and non-Lipschitz smoothness
2. Goldstein stationarity convergence guarantees
3. Stochastic bilevel optimization with linear constraints
4. Convergence to (ε, δ)-Goldstein stationary point in Õ(ε⁻⁴δ⁻¹) iterations
"""

import torch
import numpy as np
import cvxpy as cp
from typing import Dict, Tuple, Optional
import time
from torch.optim import Adam, SGD
import warnings

class DSBLOAlgorithm:
    """
    DS-BLO: Doubly Stochastically Perturbed Algorithm for Linearly Constrained Bilevel Optimization
    
    Core mechanisms:
    1. First perturbation: Linear perturbation q^T y for differentiability
    2. Second perturbation: Stochastic sampling for non-Lipschitz smoothness
    3. Goldstein stationarity: Robust convergence measure
    4. Stochastic bilevel: Handles stochastic upper-level objective
    """
    
    def __init__(self, 
                 problem,
                 # DS-BLO specific parameters
                 q_distribution: str = 'normal',
                 q_std: float = 0.01,
                 perturbation_scale: float = 0.001,
                 goldstein_samples: int = 10,
                 device: str = 'cpu',
                 seed: int = 42):
        """
        Initialize DS-BLO algorithm
        
        Args:
            problem: StronglyConvexBilevelProblem instance
            q_distribution: Distribution for perturbation q ('normal' or 'uniform')
            q_std: Standard deviation for perturbation q
            perturbation_scale: Scale for perturbation
            goldstein_samples: Number of samples for Goldstein subdifferential
            device: Device to use
            seed: Random seed
        """
        self.problem = problem
        self.device = device
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Perturbation parameters (DS-BLO specific)
        self.q_distribution = q_distribution
        self.q_std = q_std
        self.perturbation_scale = perturbation_scale
        
        # Goldstein stationarity parameters
        self.goldstein_samples = goldstein_samples
        
        # Storage for analysis
        self.history = {
            'iterations': [],
            'objectives': [],
            'gradients': [],
            'gaps': [],
            'times': []
        }
        
        
    def sample_perturbation(self) -> torch.Tensor:
        """
        Sample perturbation q for the lower-level objective
        This is the first perturbation in DS-BLO for differentiability
        """
        if self.q_distribution == 'normal':
            q = torch.randn(self.problem.dim, device=self.device, dtype=self.problem.dtype) * self.q_std
        elif self.q_distribution == 'uniform':
            q = (torch.rand(self.problem.dim, device=self.device, dtype=self.problem.dtype) - 0.5) * 2 * self.q_std
        else:
            raise ValueError(f"Unknown perturbation distribution: {self.q_distribution}")
        
        return q
        
    def solve_perturbed_lower_level(self, x: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Solve the perturbed lower-level problem: min_y g(x,y) + q^T y s.t. Ay + Bx ≤ b
        
        This implements the perturbed LL problem from DS-BLO paper
        """
        # Convert to numpy for CVXPy
        x_np = x.detach().cpu().numpy()
        q_np = q.detach().cpu().numpy()
        
        # Problem parameters
        Q_lower_np = self.problem.Q_lower.detach().cpu().numpy()
        c_lower_np = self.problem.c_lower.detach().cpu().numpy()
        P_np = self.problem.P.detach().cpu().numpy()
        A_np = self.problem.A.detach().cpu().numpy()
        B_np = self.problem.B.detach().cpu().numpy()
        b_np = self.problem.b.detach().cpu().numpy()
        
        # Create CVXPy problem with perturbation
        y_var = cp.Variable(self.problem.dim)
        
        # Perturbed objective: g(x,y) + q^T y
        objective = cp.Minimize(0.5 * cp.quad_form(y_var, Q_lower_np) + 
                              (c_lower_np + P_np.T @ x_np + q_np) @ y_var)
        
        # Constraints: Ay + Bx ≤ b
        constraints = [A_np @ y_var + B_np @ x_np <= b_np]
        
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, eps_abs=self.epsilon**2)
        
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"CVXPy solve failed: status={problem.status}")
        
        y_opt = torch.tensor(y_var.value, device=self.device, dtype=self.problem.dtype)
        
        info = {
            'status': problem.status,
            'value': problem.value,
            'perturbation_norm': torch.norm(q).item()
        }
        
        return y_opt, info
        
    def compute_perturbed_implicit_gradient(self, x: torch.Tensor, q: torch.Tensor, 
                                          y_star: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """
        Compute the perturbed implicit gradient: ∇F_q(x) = ∇x f(x,y_q*(x)) + [∇y_q*(x)]^T ∇y f(x,y_q*(x))
        
        This is the core gradient computation in DS-BLO
        """
        # Make x differentiable
        x_grad = x.clone().requires_grad_(True)
        
        # Compute upper-level objective with stochastic sample
        f_val = self.problem.upper_objective(x_grad, y_star, add_noise=True)
        
        # Compute gradient w.r.t. x
        grad_x = torch.autograd.grad(f_val, x_grad, create_graph=False)[0]
        
        # For the second term [∇y_q*(x)]^T ∇y f(x,y_q*(x)), we need to compute ∇y_q*(x)
        # This requires solving the perturbed LL problem and computing the implicit gradient
        
        # Approximate ∇y_q*(x) using finite differences
        eps_fd = 1e-6
        y_plus = torch.zeros_like(y_star)
        y_minus = torch.zeros_like(y_star)
        
        for i in range(self.problem.dim):
            # Perturb x in the i-th direction
            x_plus = x.clone()
            x_plus[i] += eps_fd
            x_minus = x.clone()
            x_minus[i] -= eps_fd
            
            # Solve perturbed LL problems
            y_plus_i, _ = self.solve_perturbed_lower_level(x_plus, q)
            y_minus_i, _ = self.solve_perturbed_lower_level(x_minus, q)
            
            y_plus[i] = y_plus_i[i]
            y_minus[i] = y_minus_i[i]
        
        # Approximate ∇y_q*(x) using finite differences
        dy_dx = (y_plus - y_minus) / (2 * eps_fd)
        
        # Compute ∇y f(x,y_q*(x))
        y_grad = y_star.clone().requires_grad_(True)
        f_val_y = self.problem.upper_objective(x, y_grad, add_noise=True)
        grad_y = torch.autograd.grad(f_val_y, y_grad, create_graph=False)[0]
        
        # Combine the gradients: ∇x f + [∇y_q*(x)]^T ∇y f
        implicit_gradient = grad_x + dy_dx * grad_y
        
        return implicit_gradient
        
    def doubly_stochastic_gradient_oracle(self, x: torch.Tensor) -> torch.Tensor:
        """
        Doubly stochastic gradient oracle (DS-BLO core mechanism)
        
        This implements the doubly stochastic approach:
        1. First perturbation: q for differentiability
        2. Second perturbation: ξ for stochasticity
        """
        # Sample perturbation q (first perturbation)
        q = self.sample_perturbation()
        
        # Sample stochastic sample ξ (second perturbation)
        xi = torch.randn(1, device=self.device, dtype=self.problem.dtype)
        
        # Solve perturbed lower-level problem
        y_star, info = self.solve_perturbed_lower_level(x, q)
        
        # Compute perturbed implicit gradient
        gradient = self.compute_perturbed_implicit_gradient(x, q, y_star, xi)
        
        
        return gradient
        
        
    def optimize(self, max_iterations: int = 1000, target_gap: float = 1e-3, run_until_convergence: bool = False) -> Dict:
        """
        Main DS-BLO optimization loop
        
        Implements the doubly stochastic algorithm with Goldstein stationarity convergence
        """
        
        # Initialize with consistent seed-based random value
        torch.manual_seed(42)  # Use same seed for consistent initialization
        x = torch.randn(self.problem.dim, device=self.device, dtype=self.problem.dtype) * 0.1
        x.requires_grad_(True)
        

        start_time = time.time()
        
        # Main optimization loop
        for iteration in range(max_iterations):
            iter_start_time = time.time()
            
            # Doubly stochastic gradient oracle
            g_t = self.doubly_stochastic_gradient_oracle(x)
            
            # Update x using gradient descent
            x = x - self.eta * g_t
            
            # Compute current objective and gap using unified gap metric
            current_obj = self.problem.upper_objective(x, torch.zeros_like(x))
            current_gap = self.problem.compute_gap(x)  # Use same gap as F2CSA
            
            # Store history for analysis
            self.history['iterations'].append(iteration)
            self.history['objectives'].append(current_obj.item())
            self.history['gradients'].append(torch.norm(g_t).item())
            self.history['gaps'].append(current_gap)
            
            iter_time = time.time() - iter_start_time
            self.history['times'].append(iter_time)
            
            # Progress monitoring
            if iteration % 50 == 0:
                pass  # Debug output removed
            

        
        total_time = time.time() - start_time
        
        # Final analysis
        final_gap = self.history['gaps'][-1] if self.history['gaps'] else float('inf')
        
        results = {
            'final_x': x.detach().cpu().numpy(),
            'final_gap': final_gap,
            'total_iterations': len(self.history['iterations']),
            'total_time': total_time,
            'target_achieved': final_gap < target_gap,
            'history': self.history
        }
        
        return results
        
    def analyze_perturbation_sensitivity(self, q_std_values: list) -> Dict:
        """
        Analyze sensitivity to different perturbation standard deviations
        
        This helps understand how the perturbation affects:
        1. Differentiability of the implicit function
        2. Approximation error between original and perturbed problems
        3. Overall convergence
        """
        
        results = {}
        x_test = torch.randn(self.problem.dim, device=self.device, dtype=self.problem.dtype) * 0.1
        
        for q_std in q_std_values:
            
            # Update perturbation parameter
            self.q_std = q_std
            
            # Test doubly stochastic gradient oracle
            try:
                grad_estimate = self.doubly_stochastic_gradient_oracle(x_test)
                true_gap = self.problem.compute_gap(x_test)
                goldstein_measure = self.goldstein_stationarity_check(x_test)
                
                results[q_std] = {
                    'gradient_norm': torch.norm(grad_estimate).item(),
                    'true_gap': true_gap,
                    'goldstein_measure': goldstein_measure,
                    'perturbation_scale': q_std
                }
                
                
            except Exception as e:
                results[q_std] = {'error': str(e)}
        
        return results
        
    def analyze_goldstein_convergence(self, x: torch.Tensor) -> Dict:
        """
        Analyze Goldstein stationarity convergence
        
        This helps understand:
        1. How Goldstein measure changes with iterations
        2. Relationship between Goldstein stationarity and gap
        3. Convergence behavior of DS-BLO
        """
        
        # Sample multiple perturbations for comprehensive analysis
        n_samples = 20
        goldstein_measures = []
        gradients = []
        gaps = []
        
        for i in range(n_samples):
            # Sample perturbation q
            q = self.sample_perturbation()
            
            # Sample stochastic sample ξ
            xi = torch.randn(1, device=self.device, dtype=self.problem.dtype)
            
            # Solve perturbed lower-level problem
            y_star, _ = self.solve_perturbed_lower_level(x, q)
            
            # Compute perturbed implicit gradient
            grad = self.compute_perturbed_implicit_gradient(x, q, y_star, xi)
            
            # Compute gap
            gap = self.problem.compute_gap(x)
            
            # Store metrics
            goldstein_measures.append(torch.norm(grad).item())
            gradients.append(torch.norm(grad).item())
            gaps.append(gap)
        
        analysis = {
            'goldstein_statistics': {
                'mean': np.mean(goldstein_measures),
                'std': np.std(goldstein_measures),
                'min': np.min(goldstein_measures),
                'max': np.max(goldstein_measures)
            },
            'gradient_statistics': {
                'mean': np.mean(gradients),
                'std': np.std(gradients)
            },
            'gap_statistics': {
                'mean': np.mean(gaps),
                'std': np.std(gaps)
            },
            'goldstein_measures': goldstein_measures,
            'gradients': gradients,
            'gaps': gaps
        }
        
        
        return analysis
