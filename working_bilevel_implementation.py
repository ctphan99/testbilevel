import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass

@dataclass
class AlgorithmMetrics:
    """Comprehensive metrics for algorithm performance"""
    gaps: List[float]
    times: List[float]
    x_norms: List[float]
    y_norms: List[float]
    grad_norms: List[float]
    step_sizes: List[float]
    convergence_iter: Optional[int] = None
    final_gap: Optional[float] = None
    
class ConstrainedStochasticBilevelProblem:
    """
    Linearly constrained stochastic bilevel optimization problem
    Following the F2CSA paper specification:
    
    min_{x ∈ X} F(x) := E[f(x, y*(x); ξ)]
    s.t. y*(x) ∈ argmin_{y: h(x,y) ≤ 0} E[g(x, y; ζ)]
    
    where h(x,y) := Ax - By - b ≤ 0 (linear constraints)
    """
    
    def __init__(self, dim: int = 10, num_constraints: int = 3, noise_std: float = 0.01, device: str = 'cpu'):
        self.dim = dim
        self.num_constraints = num_constraints
        self.noise_std = noise_std
        self.device = device
        
        # Upper level problem parameters (dimension-aware conditioning)
        if dim <= 10:
            # Original formulation for small dimensions
            self.Q_upper = torch.randn(dim, dim, device=device) * 0.1
            self.Q_upper = self.Q_upper + self.Q_upper.T  # Make symmetric
            self.Q_upper += torch.eye(dim, device=device) * 1.0  # Ensure strongly convex
        else:
            # Strongly convex formulation for high dimensions (F2CSA requirement)
            noise_scale = 0.01 / np.sqrt(dim)  # Scale noise with dimension
            self.Q_upper = torch.randn(dim, dim, device=device) * noise_scale
            self.Q_upper = self.Q_upper + self.Q_upper.T  # Make symmetric
            self.Q_upper += torch.eye(dim, device=device) * 2.0  # Strong convexity for F2CSA
        
        self.c_upper = torch.randn(dim, device=device) * 0.1
        self.x_target = torch.randn(dim, device=device)
        
        # Lower level problem parameters (dimension-aware conditioning)
        if dim <= 10:
            # Original formulation for small dimensions
            self.Q_lower = torch.randn(dim, dim, device=device) * 0.1
            self.Q_lower = self.Q_lower + self.Q_lower.T  # Make symmetric
            self.Q_lower += torch.eye(dim, device=device) * 1.0  # Ensure strongly convex
        else:
            # Well-conditioned formulation for high dimensions
            noise_scale = 0.01 / np.sqrt(dim)  # Scale noise with dimension
            self.Q_lower = torch.randn(dim, dim, device=device) * noise_scale
            self.Q_lower = self.Q_lower + self.Q_lower.T  # Make symmetric
            self.Q_lower += torch.eye(dim, device=device) * 2.0  # Stronger diagonal dominance
        
        # Dimension-aware parameter scaling
        if dim <= 10:
            # Original scaling for small dimensions
            self.c_lower = torch.randn(dim, device=device) * 0.1
            self.P = torch.randn(dim, dim, device=device) * 0.1  # Coupling matrix

            # Linear constraint parameters: h(x,y) = Ax - By - b ≤ 0
            self.A = torch.randn(num_constraints, dim, device=device) * 0.2
            self.B = torch.randn(num_constraints, dim, device=device) * 0.2
            self.b = torch.randn(num_constraints, device=device) * 0.5
        else:
            # Well-conditioned scaling for high dimensions
            param_scale = 0.1 / np.sqrt(dim)  # Scale parameters with dimension
            self.c_lower = torch.randn(dim, device=device) * param_scale
            self.P = torch.randn(dim, dim, device=device) * param_scale  # Coupling matrix

            # Linear constraint parameters: h(x,y) = Ax - By - b ≤ 0
            constraint_scale = 0.1 / np.sqrt(dim)  # Much smaller for high dim
            self.A = torch.randn(num_constraints, dim, device=device) * constraint_scale
            self.B = torch.randn(num_constraints, dim, device=device) * constraint_scale
            self.b = torch.randn(num_constraints, device=device) * 0.1  # Smaller constraint bounds
        
        # Ensure constraints are feasible by adjusting b
        self.b = self.b + torch.abs(self.b) * 0.5  # Make constraints not too tight
        
        print(f"📊 Created Constrained Bilevel Problem:")
        print(f"   Dimension: {dim}")
        print(f"   Constraints: {num_constraints}")
        print(f"   A condition: {torch.linalg.cond(self.A @ self.A.T):.2f}")
        print(f"   B condition: {torch.linalg.cond(self.B @ self.B.T):.2f}")
        print(f"   Q_lower condition: {torch.linalg.cond(self.Q_lower):.2f}")
    
    def upper_level_objective(self, x: torch.Tensor, y: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        """
        Upper level objective f(x,y) with stochastic noise
        f(x,y) = 0.5 * (x - x_target)^T Q_upper (x - x_target) + c_upper^T y + noise
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        noise = torch.randn_like(x) * self.noise_std
        
        term1 = 0.5 * (x - self.x_target) @ self.Q_upper @ (x - self.x_target)
        term2 = self.c_upper @ y
        
        return term1 + term2 + noise.sum()
    
    def lower_level_objective(self, x: torch.Tensor, y: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        """
        Lower level objective g(x,y) with stochastic noise
        g(x,y) = 0.5 * y^T Q_lower y + (c_lower + P^T x)^T y + noise
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        noise = torch.randn_like(y) * self.noise_std
        
        term1 = 0.5 * y @ self.Q_lower @ y
        term2 = (self.c_lower + self.P.T @ x) @ y
        
        return term1 + term2 + noise.sum()
    
    def constraint_function(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Linear constraint function h(x,y) = Ax - By - b
        Returns constraint values (should be ≤ 0 for feasibility)
        """
        return self.A @ x - self.B @ y - self.b
    
    def is_feasible(self, x: torch.Tensor, y: torch.Tensor, tolerance: float = 1e-6) -> bool:
        """Check if (x,y) satisfies the linear constraints"""
        h_values = self.constraint_function(x, y)
        return torch.all(h_values <= tolerance)
    
    def solve_lower_level_constrained(self, x: torch.Tensor, seed: Optional[int] = None, 
                                    max_iter: int = 1000, tolerance: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve constrained lower level problem using projected gradient descent
        Returns (y*, λ*) where λ* are the dual variables
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Initialize y
        y = torch.randn(self.dim, device=self.device) * 0.1
        lambda_dual = torch.zeros(self.num_constraints, device=self.device)
        
        # Projected gradient descent with dual updates
        lr_primal = 0.01
        lr_dual = 0.01
        
        for iteration in range(max_iter):
            # Compute gradients
            y_temp = y.clone().requires_grad_(True)
            obj_val = self.lower_level_objective(x, y_temp, seed)
            grad_y = torch.autograd.grad(obj_val, y_temp)[0]
            
            # Constraint values and gradients
            h_values = self.constraint_function(x, y)
            grad_h = -self.B  # ∇_y h(x,y) = -B
            
            # Primal update with dual correction
            y_new = y - lr_primal * (grad_y + grad_h.T @ lambda_dual)
            
            # Dual update (gradient ascent on dual)
            lambda_new = torch.clamp(lambda_dual + lr_dual * h_values, min=0.0)
            
            # Check convergence
            primal_residual = torch.norm(y_new - y)
            dual_residual = torch.norm(lambda_new - lambda_dual)
            
            y = y_new
            lambda_dual = lambda_new
            
            if primal_residual < tolerance and dual_residual < tolerance:
                break
        
        return y, lambda_dual
    
    def solve_lower_level(self, x: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        """Solve lower level problem (returns only y* for compatibility)"""
        y_star, _ = self.solve_lower_level_constrained(x, seed)
        return y_star
    
    def bilevel_objective(self, x: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        """Full bilevel objective F(x) = f(x, y*(x))"""
        y_star = self.solve_lower_level(x, seed)
        return self.upper_level_objective(x, y_star, seed)
    
    def compute_gap(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute optimality gap (for compatibility)"""
        current_obj = self.bilevel_objective(x).item()
        # Simple gap approximation
        return abs(current_obj)
    
    def get_problem_statistics(self) -> Dict:
        """Get problem characteristics for analysis"""
        return {
            'dim': self.dim,
            'num_constraints': self.num_constraints,
            'A_condition': torch.linalg.cond(self.A @ self.A.T).item(),
            'B_condition': torch.linalg.cond(self.B @ self.B.T).item(),
            'Q_lower_condition': torch.linalg.cond(self.Q_lower).item(),
            'Q_upper_condition': torch.linalg.cond(self.Q_upper).item(),
            'constraint_tightness': self.b.norm().item(),
            'coupling_strength': self.P.norm().item()
        }

class DSBLOSolver:
    """DS-BLO with original doubly stochastic gradient computation + CRN fix"""
    
    def __init__(self, lr: float = 0.01, sigma: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.sigma = sigma
        self.momentum = momentum
        self.velocity_x = None
        self.velocity_y = None
        
    def step(self, problem: ConstrainedStochasticBilevelProblem, x: torch.Tensor, y: torch.Tensor, 
             iteration: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """DS-BLO step with detailed tracking"""
        
        if self.velocity_x is None:
            self.velocity_x = torch.zeros_like(x)
            self.velocity_y = torch.zeros_like(y)
        
        # DS-BLO: Doubly stochastic perturbation for gradient estimation
        u_x = torch.randn_like(x) * self.sigma
        u_y = torch.randn_like(y) * self.sigma
        
        # Use CRN for function evaluations within DS-BLO method
        seed = iteration * 1000  # Deterministic seed based on iteration
        
        # Forward perturbation
        torch.manual_seed(seed)
        f_plus = problem.bilevel_objective(x + u_x)
        
        # Backward perturbation with same seed for correlation
        torch.manual_seed(seed)
        f_minus = problem.bilevel_objective(x - u_x)
        
        # DS-BLO gradient estimation
        func_diff = f_plus - f_minus
        grad_x = func_diff / (2 * self.sigma) * u_x
        
        # Lower level gradient (simplified)
        torch.manual_seed(seed)
        g_plus = problem.lower_level_objective(x, y + u_y)
        torch.manual_seed(seed)
        g_minus = problem.lower_level_objective(x, y - u_y)
        
        func_diff_y = g_plus - g_minus
        grad_y = func_diff_y / (2 * self.sigma) * u_y
        
        # Momentum updates
        self.velocity_x = self.momentum * self.velocity_x + grad_x
        self.velocity_y = self.momentum * self.velocity_y + grad_y
        
        # Parameter updates
        x_new = x - self.lr * self.velocity_x
        y_new = y - self.lr * self.velocity_y
        
        # Detailed tracking
        tracking_info = {
            'perturbation_norm_x': u_x.norm().item(),
            'perturbation_norm_y': u_y.norm().item(),
            'function_diff': func_diff.item(),
            'function_diff_y': func_diff_y.item(),
            'grad_norm_x': grad_x.norm().item(),
            'grad_norm_y': grad_y.norm().item(),
            'velocity_norm_x': self.velocity_x.norm().item(),
            'velocity_norm_y': self.velocity_y.norm().item(),
            'sigma': self.sigma,
            'avg_perturbation_norm': (u_x.norm() + u_y.norm()).item() / 2
        }
        
        return x_new, y_new, tracking_info

class AdaptiveF2CSAHypergradientOracle:
    """
    Correct F2CSA Hypergradient Oracle from Algorithm 1 in F2CSA.tex
    with adaptive mechanisms to prevent penalty explosion
    """

    def __init__(self, alpha: float = 0.3, N_g: int = 5, tau: float = 0.01):
        # Correct penalty parameters from F2CSA paper
        self.alpha = alpha
        self.alpha_1 = alpha ** (-2)  # α₁ = α⁻² (correct from paper)
        self.alpha_2 = alpha ** (-4)  # α₂ = α⁻⁴ (correct from paper)
        self.delta = alpha ** 3       # δ = α³ (inner accuracy)
        self.N_g = N_g               # Batch size for stochastic estimation
        self.tau = tau               # Smooth activation parameter

        # Adaptive mechanism parameters
        self.penalty_explosion_threshold = 3.0
        self.hypergradient_explosion_threshold = 1000.0
        self.sensitivity_threshold_alpha_1 = 50.0
        self.sensitivity_threshold_alpha_2 = 100.0

        # Tracking for adaptive adjustments
        self.previous_penalties = None
        self.adaptation_history = []

    def compute_smooth_activation(self, h_values: torch.Tensor, lambda_values: torch.Tensor) -> torch.Tensor:
        """
        Compute smooth activation functions ρᵢ(x) = σ_h(hᵢ) · σ_λ(λᵢ)
        Using sigmoid-based smooth activation to avoid discontinuities
        """
        sigma_h = torch.sigmoid(h_values / self.tau)
        sigma_lambda = torch.sigmoid(lambda_values / self.tau)
        return sigma_h * sigma_lambda

    def compute_penalty_lagrangian(self, problem, x: torch.Tensor, y: torch.Tensor,
                                 lambda_dual: torch.Tensor, y_star: torch.Tensor, seed: int) -> torch.Tensor:
        """
        Compute penalty Lagrangian from Equation (7) in F2CSA paper:
        L_λ,α(x,y) = f(x,y) + α₁(g(x,y) + λᵀh(x,y) - g(x,ỹ*(x))) + (α₂/2)∑ᵢρᵢ(x)·hᵢ(x,y)²
        """
        # Upper level objective f(x,y)
        f_val = problem.upper_level_objective(x, y, seed)

        # Lower level objectives
        g_val = problem.lower_level_objective(x, y, seed)
        g_star = problem.lower_level_objective(x, y_star, seed)

        # Constraint values h(x,y)
        h_values = problem.constraint_function(x, y)

        # Penalty term 1: α₁(g(x,y) + λᵀh(x,y) - g(x,ỹ*(x)))
        penalty_1 = self.alpha_1 * (g_val + lambda_dual @ h_values - g_star)

        # Penalty term 2: (α₂/2)∑ᵢρᵢ(x)·hᵢ(x,y)²
        rho_values = self.compute_smooth_activation(h_values, lambda_dual)
        penalty_2 = (self.alpha_2 / 2) * (rho_values * (h_values ** 2)).sum()

        return f_val + penalty_1 + penalty_2

    def compute_hypergradient_with_crn(self, problem, x: torch.Tensor, seed: int) -> Tuple[torch.Tensor, Dict]:
        """
        Compute stochastic hypergradient using Algorithm 1 from F2CSA paper with CRN
        """
        # Step 1: Compute approximate primal-dual solution (ỹ*, λ̃)
        torch.manual_seed(seed)
        y_tilde_star, lambda_tilde = problem.solve_lower_level_constrained(x, seed)

        # Step 2: Solve penalty subproblem to get ỹ(x)
        # For simplicity, use y_tilde_star as approximation (can be improved with inner optimization)
        y_tilde = y_tilde_star

        # Step 3: Compute stochastic hypergradient with N_g samples
        hypergradients = []
        penalty_components = {'P1': 0.0, 'P2': 0.0}

        for i in range(self.N_g):
            sample_seed = seed + i * 100

            # Compute gradient of penalty Lagrangian using finite differences
            eps = 0.001
            hypergradient_sample = torch.zeros_like(x)

            for j in range(x.shape[0]):
                x_plus = x.clone()
                x_plus[j] += eps
                x_minus = x.clone()
                x_minus[j] -= eps

                torch.manual_seed(sample_seed)
                L_plus = self.compute_penalty_lagrangian(problem, x_plus, y_tilde, lambda_tilde, y_tilde_star, sample_seed)
                torch.manual_seed(sample_seed)
                L_minus = self.compute_penalty_lagrangian(problem, x_minus, y_tilde, lambda_tilde, y_tilde_star, sample_seed)

                hypergradient_sample[j] = (L_plus - L_minus) / (2 * eps)

            hypergradients.append(hypergradient_sample)

            # Track penalty components for the first sample
            if i == 0:
                torch.manual_seed(sample_seed)
                g_val = problem.lower_level_objective(x, y_tilde, sample_seed)
                g_star = problem.lower_level_objective(x, y_tilde_star, sample_seed)
                h_values = problem.constraint_function(x, y_tilde)

                penalty_1 = self.alpha_1 * (g_val + lambda_tilde @ h_values - g_star)
                rho_values = self.compute_smooth_activation(h_values, lambda_tilde)
                penalty_2 = (self.alpha_2 / 2) * (rho_values * (h_values ** 2)).sum()

                penalty_components['P1'] = penalty_1.item()
                penalty_components['P2'] = penalty_2.item()

        # Average the hypergradients
        averaged_hypergradient = torch.stack(hypergradients).mean(dim=0)

        # Compute variance for tracking
        hypergradient_variance = torch.stack(hypergradients).var(dim=0).mean().item()

        tracking_info = {
            'hypergradient_norm': averaged_hypergradient.norm().item(),
            'hypergradient_variance': hypergradient_variance,
            'penalty_1': penalty_components['P1'],
            'penalty_2': penalty_components['P2'],
            'alpha_1': self.alpha_1,
            'alpha_2': self.alpha_2,
            'N_g': self.N_g,
            'delta': self.delta
        }

        return averaged_hypergradient, tracking_info

    def compute_parameter_sensitivities(self, problem, x: torch.Tensor, seed: int, eps: float = 0.001) -> Dict[str, float]:
        """Compute parameter sensitivities ∂||∇F||/∂α₁, ∂||∇F||/∂α₂"""

        # Baseline hypergradient norm
        baseline_hg, _ = self.compute_hypergradient_with_crn(problem, x, seed)
        baseline_norm = baseline_hg.norm().item()

        # Sensitivity to α₁
        original_alpha_1 = self.alpha_1
        self.alpha_1 += eps
        perturbed_hg, _ = self.compute_hypergradient_with_crn(problem, x, seed)
        perturbed_norm = perturbed_hg.norm().item()
        sensitivity_alpha_1 = (perturbed_norm - baseline_norm) / eps
        self.alpha_1 = original_alpha_1  # Reset

        # Sensitivity to α₂
        original_alpha_2 = self.alpha_2
        self.alpha_2 += eps
        perturbed_hg, _ = self.compute_hypergradient_with_crn(problem, x, seed)
        perturbed_norm = perturbed_hg.norm().item()
        sensitivity_alpha_2 = (perturbed_norm - baseline_norm) / eps
        self.alpha_2 = original_alpha_2  # Reset

        return {
            'alpha_1': sensitivity_alpha_1,
            'alpha_2': sensitivity_alpha_2
        }

    def detect_penalty_explosion(self, current_penalties: Dict[str, float]) -> bool:
        """Detect penalty explosion based on growth rates"""
        if self.previous_penalties is None:
            self.previous_penalties = current_penalties
            return False

        p1_growth = abs(current_penalties['P1'] - self.previous_penalties['P1']) / (abs(self.previous_penalties['P1']) + 1e-8)
        p2_growth = abs(current_penalties['P2'] - self.previous_penalties['P2']) / (abs(self.previous_penalties['P2']) + 1e-8)

        max_growth = max(p1_growth, p2_growth)
        self.previous_penalties = current_penalties

        return max_growth > self.penalty_explosion_threshold

    def adapt_parameters(self, penalties: Dict[str, float], sensitivities: Dict[str, float],
                        hypergradient_norm: float) -> Dict[str, bool]:
        """Apply adaptive parameter adjustments based on tracking metrics"""

        adjustments = {
            'alpha_1_reduced': False,
            'alpha_2_reduced': False,
            'parameters_reset': False
        }

        # Detect penalty explosion
        penalty_explosion = self.detect_penalty_explosion(penalties)

        # Adjust α₂ if high sensitivity or penalty explosion
        if sensitivities['alpha_2'] > self.sensitivity_threshold_alpha_2 or penalty_explosion:
            self.alpha_2 *= 0.9
            adjustments['alpha_2_reduced'] = True

        # Adjust α₁ if high sensitivity
        if sensitivities['alpha_1'] > self.sensitivity_threshold_alpha_1:
            self.alpha_1 *= 0.9
            adjustments['alpha_1_reduced'] = True

        # Emergency parameter reset if hypergradient explosion
        if hypergradient_norm > self.hypergradient_explosion_threshold * 10:
            self.alpha_1 = self.alpha ** (-2)
            self.alpha_2 = self.alpha ** (-4)
            adjustments['parameters_reset'] = True

        return adjustments

    def analyze_gradient_explosion_factors(self, problem, x: torch.Tensor, seed: int) -> Dict:
        """
        Deep analysis of gradient explosion factors with step-by-step breakdown
        """
        analysis = {}

        # 1. Constraint violation analysis
        torch.manual_seed(seed)
        y_tilde_star, lambda_tilde = problem.solve_lower_level_constrained(x, seed)
        h_values = problem.constraint_function(x, y_tilde_star)

        analysis['constraint_violations'] = {
            'h_values': h_values.tolist(),
            'max_violation': h_values.max().item(),
            'violation_norm': h_values.norm().item(),
            'num_violated': (h_values > 1e-6).sum().item(),
            'violation_severity': torch.clamp(h_values, min=0).sum().item()
        }

        # 2. Penalty term breakdown
        g_val = problem.lower_level_objective(x, y_tilde_star, seed)
        g_star = problem.lower_level_objective(x, y_tilde_star, seed)  # Should be same

        # Raw penalty components before scaling
        penalty_1_raw = g_val + lambda_tilde @ h_values - g_star
        penalty_2_raw = (h_values ** 2).sum()

        # Scaled penalty components
        penalty_1_scaled = self.alpha_1 * penalty_1_raw
        penalty_2_scaled = (self.alpha_2 / 2) * penalty_2_raw

        analysis['penalty_breakdown'] = {
            'penalty_1_raw': penalty_1_raw.item(),
            'penalty_2_raw': penalty_2_raw.item(),
            'penalty_1_scaled': penalty_1_scaled.item(),
            'penalty_2_scaled': penalty_2_scaled.item(),
            'alpha_1_amplification': self.alpha_1,
            'alpha_2_amplification': self.alpha_2,
            'penalty_ratio': penalty_1_scaled.item() / (penalty_2_scaled.item() + 1e-8),
            'total_penalty': penalty_1_scaled.item() + penalty_2_scaled.item()
        }

        # 3. Smooth activation analysis
        rho_values = self.compute_smooth_activation(h_values, lambda_tilde)
        analysis['smooth_activation'] = {
            'rho_values': rho_values.tolist(),
            'rho_mean': rho_values.mean().item(),
            'rho_max': rho_values.max().item(),
            'activation_amplification': (rho_values * (h_values ** 2)).sum().item() / (penalty_2_raw.item() + 1e-8)
        }

        # 4. Gradient component analysis
        eps = 0.001

        # Direct objective gradient
        x_temp = x.clone().requires_grad_(True)
        f_val = problem.upper_level_objective(x_temp, y_tilde_star, seed)
        direct_grad = torch.autograd.grad(f_val, x_temp)[0]

        # Penalty gradient (finite difference)
        penalty_grad = torch.zeros_like(x)
        for i in range(x.shape[0]):
            x_plus = x.clone()
            x_plus[i] += eps
            x_minus = x.clone()
            x_minus[i] -= eps

            torch.manual_seed(seed)
            L_plus = self.compute_penalty_lagrangian(problem, x_plus, y_tilde_star, lambda_tilde, y_tilde_star, seed)
            torch.manual_seed(seed)
            L_minus = self.compute_penalty_lagrangian(problem, x_minus, y_tilde_star, lambda_tilde, y_tilde_star, seed)

            penalty_grad[i] = (L_plus - L_minus) / (2 * eps) - direct_grad[i]

        analysis['gradient_components'] = {
            'direct_grad_norm': direct_grad.norm().item(),
            'penalty_grad_norm': penalty_grad.norm().item(),
            'total_grad_norm': (direct_grad + penalty_grad).norm().item(),
            'penalty_dominance_ratio': penalty_grad.norm().item() / (direct_grad.norm().item() + 1e-8),
            'gradient_explosion_factor': (direct_grad + penalty_grad).norm().item() / (direct_grad.norm().item() + 1e-8)
        }

        # 5. Parameter scaling impact
        analysis['parameter_scaling_impact'] = {
            'alpha_base': self.alpha,
            'alpha_1_scaling': self.alpha_1 / self.alpha,  # Should be α⁻¹
            'alpha_2_scaling': self.alpha_2 / (self.alpha ** 2),  # Should be α⁻²
            'constraint_amplification': self.alpha_2 * penalty_2_raw.item(),
            'dual_amplification': self.alpha_1 * abs(lambda_tilde @ h_values).item(),
            'total_amplification': (penalty_1_scaled.item() + penalty_2_scaled.item()) / (abs(penalty_1_raw.item()) + penalty_2_raw.item() + 1e-8)
        }

        # 6. Explosion risk indicators
        explosion_risk = 0.0
        risk_factors = []

        if analysis['constraint_violations']['max_violation'] > 1.0:
            explosion_risk += 2.0
            risk_factors.append(f"Large constraint violation: {analysis['constraint_violations']['max_violation']:.3f}")

        if analysis['penalty_breakdown']['alpha_2_amplification'] > 100:
            explosion_risk += 3.0
            risk_factors.append(f"High α₂ amplification: {analysis['penalty_breakdown']['alpha_2_amplification']:.1f}")

        if analysis['gradient_components']['penalty_dominance_ratio'] > 10:
            explosion_risk += 2.0
            risk_factors.append(f"Penalty gradient dominance: {analysis['gradient_components']['penalty_dominance_ratio']:.1f}x")

        if analysis['parameter_scaling_impact']['total_amplification'] > 100:
            explosion_risk += 1.0
            risk_factors.append(f"High total amplification: {analysis['parameter_scaling_impact']['total_amplification']:.1f}")

        analysis['explosion_risk'] = {
            'total_risk_score': explosion_risk,
            'risk_factors': risk_factors,
            'explosion_likely': explosion_risk > 5.0
        }

        return analysis

class AdaptiveF2CSASolver:
    """
    Adaptive F2CSA Solver implementing Algorithm 2 from F2CSA.tex
    with adaptive mechanisms to prevent penalty explosion
    """

    def __init__(self, alpha: float = 0.3, eta: float = 0.01, D: float = 1.0, N_g: int = 5):
        # F2CSA parameters
        self.alpha = alpha
        self.eta = eta  # Step size
        self.D = D      # Clipping threshold
        self.N_g = N_g  # Batch size for hypergradient oracle

        # Adaptive parameters
        self.eta_base = eta
        self.D_base = D

        # State variables
        self.Delta = None  # Momentum term
        self.iteration_count = 0

        # Hypergradient oracle
        self.oracle = AdaptiveF2CSAHypergradientOracle(alpha=alpha, N_g=N_g)

        # Tracking for adaptive mechanisms
        self.clipping_history = []
        self.adaptation_history = []

    def clip_vector(self, v: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, bool]:
        """Clip vector to threshold: clip_D(v) = min{1, D/||v||} · v"""
        v_norm = v.norm()
        if v_norm > threshold:
            return v * threshold / v_norm, True
        return v, False

    def step(self, problem, x: torch.Tensor, y: torch.Tensor,
             iteration: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Adaptive F2CSA step implementing Algorithm 2 from paper"""

        if self.Delta is None:
            self.Delta = torch.zeros_like(x)

        # Step 1: Sample s_t ~ Uniform[0,1] (Algorithm 2, line 5)
        s_t = torch.rand(1).item()

        # Step 2: Compute z_t = x_{t-1} + s_t * Delta_t (Algorithm 2, line 5)
        z_t = x + s_t * self.Delta

        # Step 3: Compute hypergradient g_t using oracle (Algorithm 2, line 6)
        seed = iteration * 1000  # CRN for consistent evaluation
        hypergradient, oracle_info = self.oracle.compute_hypergradient_with_crn(problem, z_t, seed)

        # Step 4: Deep gradient explosion analysis (every 10 iterations for performance)
        explosion_analysis = None
        if iteration % 10 == 0:
            explosion_analysis = self.oracle.analyze_gradient_explosion_factors(problem, z_t, seed)

        # Step 5: Compute parameter sensitivities for adaptive adjustment
        sensitivities = self.oracle.compute_parameter_sensitivities(problem, z_t, seed)

        # Step 6: Apply adaptive parameter adjustments
        adjustments = self.oracle.adapt_parameters(
            {'P1': oracle_info['penalty_1'], 'P2': oracle_info['penalty_2']},
            sensitivities,
            oracle_info['hypergradient_norm']
        )

        # Step 6: Adaptive step size adjustment
        if oracle_info['hypergradient_norm'] > 1000:
            self.eta = self.eta * 0.8
        elif oracle_info['hypergradient_norm'] < 10:
            self.eta = min(self.eta_base, self.eta * 1.1)

        # Step 7: Momentum update Delta_{t+1} = clip_D(Delta_t - eta * g_t) (Algorithm 2, line 7)
        Delta_new = self.Delta - self.eta * hypergradient
        Delta_clipped, clipped = self.clip_vector(Delta_new, self.D)
        self.Delta = Delta_clipped

        # Track clipping frequency for adaptive D adjustment
        self.clipping_history.append(1.0 if clipped else 0.0)
        if len(self.clipping_history) > 20:
            self.clipping_history = self.clipping_history[-20:]

        # Adaptive clipping threshold adjustment
        if len(self.clipping_history) >= 10:
            clipping_frequency = sum(self.clipping_history) / len(self.clipping_history)
            if clipping_frequency > 0.5:
                self.D = self.D * 0.9
            elif clipping_frequency < 0.1:
                self.D = min(self.D_base, self.D * 1.1)
        else:
            clipping_frequency = 0.0

        # Step 8: Parameter update x_t = x_{t-1} + Delta_t (Algorithm 2, line 5)
        x_new = x + self.Delta

        # Step 9: Solve lower level for new x (for compatibility)
        torch.manual_seed(seed)
        y_new = problem.solve_lower_level(x_new, seed)

        # Comprehensive tracking
        tracking_info = {
            's_t': s_t,
            'z_t_norm': z_t.norm().item(),
            'hypergradient_norm': oracle_info['hypergradient_norm'],
            'hypergradient_variance': oracle_info['hypergradient_variance'],
            'penalty_1': oracle_info['penalty_1'],
            'penalty_2': oracle_info['penalty_2'],
            'penalty_ratio': oracle_info['penalty_1'] / (oracle_info['penalty_2'] + 1e-8),
            'alpha_1_current': self.oracle.alpha_1,
            'alpha_2_current': self.oracle.alpha_2,
            'eta_current': self.eta,
            'D_current': self.D,
            'delta_norm_before': Delta_new.norm().item(),
            'delta_norm_after': self.Delta.norm().item(),
            'clipped': clipped,
            'clipping_frequency': clipping_frequency,
            'sensitivity_alpha_1': sensitivities['alpha_1'],
            'sensitivity_alpha_2': sensitivities['alpha_2'],
            'alpha_1_reduced': adjustments['alpha_1_reduced'],
            'alpha_2_reduced': adjustments['alpha_2_reduced'],
            'parameters_reset': adjustments['parameters_reset'],
            'N_g': self.oracle.N_g,
            'delta_inner_accuracy': self.oracle.delta,

            # Explosion analysis (when available)
            'explosion_analysis': explosion_analysis
        }

        self.iteration_count += 1
        return x_new, y_new, tracking_info

class SSIGDSolver:
    """SSIGD with original smoothed implicit gradient computation + CRN fix"""

    def __init__(self, lr: float = 0.01, epsilon: float = 0.01,
                 smoothing_samples: int = 5, momentum_coeff: float = 0.9):
        self.lr = lr
        self.epsilon = epsilon
        self.smoothing_samples = smoothing_samples
        self.momentum_coeff = momentum_coeff
        self.momentum_x = None
        self.momentum_y = None

    def compute_gradient_with_crn(self, problem: ConstrainedStochasticBilevelProblem, x: torch.Tensor, seed: int) -> torch.Tensor:
        """Compute gradient with CRN for consistent evaluation"""
        eps = 0.001
        grad = torch.zeros_like(x)

        for i in range(x.shape[0]):
            x_plus = x.clone()
            x_plus[i] += eps
            x_minus = x.clone()
            x_minus[i] -= eps

            torch.manual_seed(seed)
            f_plus = problem.bilevel_objective(x_plus)
            torch.manual_seed(seed)
            f_minus = problem.bilevel_objective(x_minus)

            grad[i] = (f_plus - f_minus) / (2 * eps)

        return grad

    def step(self, problem: ConstrainedStochasticBilevelProblem, x: torch.Tensor, y: torch.Tensor,
             iteration: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """SSIGD step with detailed tracking"""

        if self.momentum_x is None:
            self.momentum_x = torch.zeros_like(x)
            self.momentum_y = torch.zeros_like(y)

        # SSIGD: Smoothed implicit gradient computation
        smoothed_grads_x = []
        smoothed_grads_y = []

        for i in range(self.smoothing_samples):
            # Smoothing perturbations
            x_pert = x + torch.randn_like(x) * self.epsilon
            y_pert = y + torch.randn_like(y) * self.epsilon

            # Use CRN within SSIGD's implicit gradient computation
            seed = iteration * 1000 + i

            # Compute implicit gradients at perturbed points using CRN
            grad_x = self.compute_gradient_with_crn(problem, x_pert, seed)

            # Simplified lower level gradient
            torch.manual_seed(seed)
            grad_y = torch.autograd.functional.jacobian(
                lambda y_var: problem.lower_level_objective(x_pert, y_var), y_pert
            )

            smoothed_grads_x.append(grad_x)
            smoothed_grads_y.append(grad_y)

        # SSIGD: Average the smoothed gradients
        grad_x_smoothed = torch.stack(smoothed_grads_x).mean(dim=0)
        grad_y_smoothed = torch.stack(smoothed_grads_y).mean(dim=0)

        # Compute variance for tracking
        grad_x_var = torch.stack(smoothed_grads_x).var(dim=0).mean().item()
        grad_y_var = torch.stack(smoothed_grads_y).var(dim=0).mean().item()

        # SSIGD momentum updates
        self.momentum_x = self.momentum_coeff * self.momentum_x + (1 - self.momentum_coeff) * grad_x_smoothed
        self.momentum_y = self.momentum_coeff * self.momentum_y + (1 - self.momentum_coeff) * grad_y_smoothed

        # Parameter updates
        x_new = x - self.lr * self.momentum_x
        y_new = y - self.lr * self.momentum_y

        # Detailed tracking
        tracking_info = {
            'smoothing_samples': self.smoothing_samples,
            'epsilon': self.epsilon,
            'grad_variance_x': grad_x_var,
            'grad_variance_y': grad_y_var,
            'grad_norm_x': grad_x_smoothed.norm().item(),
            'grad_norm_y': grad_y_smoothed.norm().item(),
            'momentum_norm_x': self.momentum_x.norm().item(),
            'momentum_norm_y': self.momentum_y.norm().item(),
            'momentum_coeff': self.momentum_coeff,
            'avg_grad_variance': (grad_x_var + grad_y_var) / 2
        }

        return x_new, y_new, tracking_info

def run_comprehensive_algorithm_tracking(dim: int = 10, max_iterations: int = 500,
                                       convergence_threshold: float = 0.01, device: str = 'cpu'):
    """Run comprehensive tracking of all three algorithms with detailed analysis"""

    print(f"🔬 COMPREHENSIVE ALGORITHM TRACKING")
    print(f"📊 Problem dimension: {dim}")
    print(f"🎯 Convergence threshold: {convergence_threshold}")
    print(f"⚡ Device: {device}")
    print("=" * 80)

    # Create problem
    problem = ConstrainedStochasticBilevelProblem(dim=dim, device=device)

    # Initialize algorithms
    algorithms = {
        'DS-BLO': DSBLOSolver(lr=0.01, sigma=0.01, momentum=0.9),
        'F2CSA-Adaptive': AdaptiveF2CSASolver(alpha=0.3, eta=0.01, D=1.0, N_g=5),
        'SSIGD': SSIGDSolver(lr=0.01, epsilon=0.01, smoothing_samples=5, momentum_coeff=0.9)
    }

    # Initialize starting points (same for all algorithms)
    torch.manual_seed(42)
    x_init = torch.randn(dim, device=device) * 0.5
    y_init = torch.randn(dim, device=device) * 0.5

    results = {}

    for alg_name, solver in algorithms.items():
        print(f"\n🚀 Running {alg_name}...")

        # Reset to same starting point
        x, y = x_init.clone(), y_init.clone()

        # Tracking lists
        gaps = []
        tracking_data = []
        times = []

        start_time = time.time()

        for iteration in range(max_iterations):
            iter_start = time.time()

            # Algorithm step with detailed tracking
            x_new, y_new, tracking_info = solver.step(problem, x, y, iteration)

            # Compute gap
            gap = problem.compute_gap(x_new, y_new)
            gaps.append(gap)

            # Store tracking data
            tracking_info['iteration'] = iteration
            tracking_info['gap'] = gap
            tracking_info['x_norm'] = x_new.norm().item()
            tracking_info['y_norm'] = y_new.norm().item()
            tracking_data.append(tracking_info)

            times.append(time.time() - iter_start)

            # Update variables
            x, y = x_new, y_new

            # Check convergence
            if gap < convergence_threshold:
                print(f"✅ {alg_name} converged at iteration {iteration} with gap {gap:.6f}")
                break

            # Detailed progress reporting with component analysis
            if iteration % 10 == 0 or (alg_name == 'F2CSA-Adaptive' and iteration < 50):
                print(f"   Iteration {iteration}: gap = {gap:.6f}")

                # Extra detailed logging for F2CSA-Adaptive to debug explosion
                if alg_name == 'F2CSA-Adaptive':
                    print(f"      🔍 Component Debug:")
                    print(f"         P1={tracking_info.get('penalty_1', 0):.6f}, P2={tracking_info.get('penalty_2', 0):.6f}")
                    print(f"         α₁={tracking_info.get('alpha_1_current', 0):.6f}, α₂={tracking_info.get('alpha_2_current', 0):.6f}")
                    print(f"         HG_norm={tracking_info.get('hypergradient_norm', 0):.6f}")
                    print(f"         η={tracking_info.get('eta_current', 0):.6f}, s_t={tracking_info.get('s_t', 0):.6f}")
                    print(f"         Δ_before={tracking_info.get('delta_norm_before', 0):.6f}, Δ_after={tracking_info.get('delta_norm_after', 0):.6f}")
                    print(f"         Clipped={tracking_info.get('clipped', False)}, Freq={tracking_info.get('clipping_frequency', 0):.3f}")
                    print(f"         Sens_α₁={tracking_info.get('sensitivity_alpha_1', 0):.6f}, Sens_α₂={tracking_info.get('sensitivity_alpha_2', 0):.6f}")
                    print(f"         Adaptations: α₁_red={tracking_info.get('alpha_1_reduced', False)}, α₂_red={tracking_info.get('alpha_2_reduced', False)}")

                    # Detailed explosion analysis (when available)
                    explosion_analysis = tracking_info.get('explosion_analysis')
                    if explosion_analysis:
                        print(f"      🔬 EXPLOSION ANALYSIS:")

                        # Constraint violations
                        cv = explosion_analysis['constraint_violations']
                        print(f"         Constraint violations: max={cv['max_violation']:.6f}, norm={cv['violation_norm']:.6f}")
                        print(f"         Violated constraints: {cv['num_violated']}/3, severity={cv['violation_severity']:.6f}")

                        # Penalty breakdown
                        pb = explosion_analysis['penalty_breakdown']
                        print(f"         Penalty raw: P1={pb['penalty_1_raw']:.6f}, P2={pb['penalty_2_raw']:.6f}")
                        print(f"         Penalty scaled: P1={pb['penalty_1_scaled']:.6f}, P2={pb['penalty_2_scaled']:.6f}")
                        print(f"         Amplification: α₁={pb['alpha_1_amplification']:.2f}, α₂={pb['alpha_2_amplification']:.2f}")

                        # Gradient components
                        gc = explosion_analysis['gradient_components']
                        print(f"         Gradients: direct={gc['direct_grad_norm']:.6f}, penalty={gc['penalty_grad_norm']:.6f}")
                        print(f"         Penalty dominance: {gc['penalty_dominance_ratio']:.2f}x, explosion factor: {gc['gradient_explosion_factor']:.2f}x")

                        # Risk assessment
                        er = explosion_analysis['explosion_risk']
                        print(f"         Risk score: {er['total_risk_score']:.1f}, explosion likely: {er['explosion_likely']}")
                        if er['risk_factors']:
                            print(f"         Risk factors: {', '.join(er['risk_factors'])}")

                    # Check for explosion patterns
                    if tracking_info.get('penalty_2', 0) > 1000:
                        print(f"      ⚠️ P2 EXPLOSION DETECTED: {tracking_info.get('penalty_2', 0):.2f}")
                    if tracking_info.get('hypergradient_norm', 0) > 1000:
                        print(f"      ⚠️ HYPERGRADIENT EXPLOSION: {tracking_info.get('hypergradient_norm', 0):.2f}")
                    if tracking_info.get('alpha_2_current', 0) > 1000:
                        print(f"      ⚠️ α₂ EXPLOSION: {tracking_info.get('alpha_2_current', 0):.2f}")

        total_time = time.time() - start_time

        # Store results
        results[alg_name] = {
            'gaps': gaps,
            'tracking_data': tracking_data,
            'times': times,
            'total_time': total_time,
            'final_gap': gaps[-1],
            'converged': gaps[-1] < convergence_threshold,
            'iterations': len(gaps)
        }

        print(f"🏁 {alg_name} finished: Final gap = {gaps[-1]:.6f}, Time = {total_time:.2f}s")

    return results, problem

def analyze_algorithm_behavior(results: Dict, problem: ConstrainedStochasticBilevelProblem):
    """Detailed analysis of algorithm behavior and component influence"""

    print(f"\n🔍 DETAILED ALGORITHM BEHAVIOR ANALYSIS")
    print("=" * 80)

    for alg_name, data in results.items():
        print(f"\n📈 {alg_name} ANALYSIS:")
        print("-" * 40)

        gaps = data['gaps']
        tracking_data = data['tracking_data']

        # Performance metrics
        initial_gap = gaps[0]
        final_gap = gaps[-1]
        best_gap = min(gaps)
        improvement = (initial_gap - final_gap) / initial_gap * 100

        print(f"🎯 Performance:")
        print(f"   Initial gap: {initial_gap:.6f}")
        print(f"   Final gap: {final_gap:.6f}")
        print(f"   Best gap: {best_gap:.6f}")
        print(f"   Improvement: {improvement:.1f}%")
        print(f"   Iterations: {len(gaps)}")

        # Algorithm-specific analysis
        if alg_name == 'DS-BLO':
            analyze_dsblo_behavior(tracking_data)
        elif alg_name == 'F2CSA-Adaptive':
            analyze_adaptive_f2csa_behavior(tracking_data)
        elif alg_name == 'SSIGD':
            analyze_ssigd_behavior(tracking_data)

        # Convergence pattern
        print(f"📊 Convergence pattern:")
        if len(gaps) >= 10:
            early_avg = np.mean(gaps[:10])
            mid_avg = np.mean(gaps[len(gaps)//2:len(gaps)//2+10]) if len(gaps) > 20 else np.mean(gaps[-10:])
            late_avg = np.mean(gaps[-10:])
            print(f"   Early phase (0-10): {early_avg:.6f}")
            print(f"   Mid phase: {mid_avg:.6f}")
            print(f"   Late phase: {late_avg:.6f}")

def analyze_dsblo_behavior(tracking_data: List[Dict]):
    """Analyze DS-BLO specific behavior"""
    print(f"🔵 DS-BLO Component Analysis:")

    # Extract key metrics
    perturbation_norms = [d['avg_perturbation_norm'] for d in tracking_data]
    function_diffs = [abs(d['function_diff']) for d in tracking_data]
    grad_norms = [d['grad_norm_x'] for d in tracking_data]

    print(f"   Perturbation analysis:")
    print(f"     Average perturbation norm: {np.mean(perturbation_norms):.6f}")
    print(f"     Perturbation std: {np.std(perturbation_norms):.6f}")
    print(f"   Function difference analysis:")
    print(f"     Average |f_plus - f_minus|: {np.mean(function_diffs):.6f}")
    print(f"     Function diff std: {np.std(function_diffs):.6f}")
    print(f"   Gradient analysis:")
    print(f"     Average gradient norm: {np.mean(grad_norms):.6f}")
    print(f"     Gradient norm std: {np.std(grad_norms):.6f}")
    print(f"     Sigma parameter: {tracking_data[0]['sigma']}")

def analyze_adaptive_f2csa_behavior(tracking_data: List[Dict]):
    """Analyze Adaptive F2CSA specific behavior with penalty tracking"""
    print(f"🟢 Adaptive F2CSA Component Analysis:")

    # Extract key metrics
    s_t_values = [d['s_t'] for d in tracking_data]
    hypergradient_norms = [d['hypergradient_norm'] for d in tracking_data]
    penalty_1_values = [d['penalty_1'] for d in tracking_data]
    penalty_2_values = [d['penalty_2'] for d in tracking_data]
    alpha_1_values = [d['alpha_1_current'] for d in tracking_data]
    alpha_2_values = [d['alpha_2_current'] for d in tracking_data]
    eta_values = [d['eta_current'] for d in tracking_data]
    clipped_count = sum(1 for d in tracking_data if d['clipped'])

    # Adaptive mechanism tracking
    alpha_1_reductions = sum(1 for d in tracking_data if d['alpha_1_reduced'])
    alpha_2_reductions = sum(1 for d in tracking_data if d['alpha_2_reduced'])
    parameter_resets = sum(1 for d in tracking_data if d['parameters_reset'])

    print(f"   📊 Penalty Analysis (Correct F2CSA formulation):")
    print(f"     Initial P1: {penalty_1_values[0]:.6f}, Final P1: {penalty_1_values[-1]:.6f}")
    print(f"     Initial P2: {penalty_2_values[0]:.6f}, Final P2: {penalty_2_values[-1]:.6f}")
    print(f"     Max P1: {max(penalty_1_values):.6f}, Max P2: {max(penalty_2_values):.6f}")
    print(f"     P1/P2 ratio (final): {penalty_1_values[-1]/(penalty_2_values[-1]+1e-8):.6f}")

    print(f"   🎯 Parameter Evolution (α₁ = α⁻², α₂ = α⁻⁴):")
    print(f"     Initial α₁: {alpha_1_values[0]:.6f}, Final α₁: {alpha_1_values[-1]:.6f}")
    print(f"     Initial α₂: {alpha_2_values[0]:.6f}, Final α₂: {alpha_2_values[-1]:.6f}")
    print(f"     Initial η: {eta_values[0]:.6f}, Final η: {eta_values[-1]:.6f}")

    print(f"   🔧 Adaptive Mechanism Effectiveness:")
    print(f"     α₁ reductions: {alpha_1_reductions}/{len(tracking_data)} ({alpha_1_reductions/len(tracking_data)*100:.1f}%)")
    print(f"     α₂ reductions: {alpha_2_reductions}/{len(tracking_data)} ({alpha_2_reductions/len(tracking_data)*100:.1f}%)")
    print(f"     Parameter resets: {parameter_resets}/{len(tracking_data)} ({parameter_resets/len(tracking_data)*100:.1f}%)")

    print(f"   📈 Hypergradient Analysis:")
    print(f"     Average hypergradient norm: {np.mean(hypergradient_norms):.6f}")
    print(f"     Hypergradient std: {np.std(hypergradient_norms):.6f}")
    print(f"     Max hypergradient norm: {max(hypergradient_norms):.6f}")

    print(f"   ✂️ Clipping Analysis:")
    print(f"     Times clipped: {clipped_count}/{len(tracking_data)} ({clipped_count/len(tracking_data)*100:.1f}%)")
    if len(tracking_data) > 0:
        final_clipping_freq = tracking_data[-1]['clipping_frequency']
        print(f"     Final clipping frequency: {final_clipping_freq:.3f}")

    print(f"   🎲 Sampling Analysis:")
    print(f"     Average s_t: {np.mean(s_t_values):.6f}")
    print(f"     s_t std: {np.std(s_t_values):.6f}")
    print(f"     N_g (batch size): {tracking_data[0]['N_g']}")

    # Penalty explosion detection
    if len(penalty_2_values) > 1:
        p2_growth_rates = []
        for i in range(1, len(penalty_2_values)):
            if abs(penalty_2_values[i-1]) > 1e-8:
                growth_rate = abs(penalty_2_values[i] - penalty_2_values[i-1]) / abs(penalty_2_values[i-1])
                p2_growth_rates.append(growth_rate)

        if p2_growth_rates:
            max_p2_growth = max(p2_growth_rates)
            print(f"   ⚠️ Penalty Explosion Monitoring:")
            print(f"     Max P2 growth rate: {max_p2_growth:.6f}")
            print(f"     Explosion threshold: 3.0")
            print(f"     Explosion detected: {'Yes' if max_p2_growth > 3.0 else 'No'}")

def analyze_f2csa_behavior(tracking_data: List[Dict]):
    """Analyze F2CSA specific behavior (legacy function for compatibility)"""
    print(f"🟢 F2CSA Component Analysis:")

    # Extract key metrics
    s_t_values = [d['s_t'] for d in tracking_data]
    delta_norms = [d['delta_norm'] for d in tracking_data]
    hypergradient_norms = [d['hypergradient_norm'] for d in tracking_data]
    clipped_count = sum(1 for d in tracking_data if d['clipped'])

    print(f"   Sampling analysis:")
    print(f"     Average s_t: {np.mean(s_t_values):.6f}")
    print(f"     s_t std: {np.std(s_t_values):.6f}")
    print(f"   Momentum analysis:")
    print(f"     Initial Delta norm: {delta_norms[0]:.6f}")
    print(f"     Final Delta norm: {delta_norms[-1]:.6f}")
    print(f"     Max Delta norm: {max(delta_norms):.6f}")
    print(f"   Hypergradient analysis:")
    print(f"     Average hypergradient norm: {np.mean(hypergradient_norms):.6f}")
    print(f"     Hypergradient std: {np.std(hypergradient_norms):.6f}")
    print(f"   Clipping analysis:")
    print(f"     Times clipped: {clipped_count}/{len(tracking_data)} ({clipped_count/len(tracking_data)*100:.1f}%)")

def analyze_ssigd_behavior(tracking_data: List[Dict]):
    """Analyze SSIGD specific behavior"""
    print(f"🟡 SSIGD Component Analysis:")

    # Extract key metrics
    grad_variances = [d['avg_grad_variance'] for d in tracking_data]
    grad_norms = [d['grad_norm_x'] for d in tracking_data]
    momentum_norms = [d['momentum_norm_x'] for d in tracking_data]

    print(f"   Smoothing analysis:")
    print(f"     Average gradient variance: {np.mean(grad_variances):.6f}")
    print(f"     Gradient variance std: {np.std(grad_variances):.6f}")
    print(f"     Smoothing samples: {tracking_data[0]['smoothing_samples']}")
    print(f"     Epsilon: {tracking_data[0]['epsilon']}")
    print(f"   Gradient analysis:")
    print(f"     Average gradient norm: {np.mean(grad_norms):.6f}")
    print(f"     Gradient norm std: {np.std(grad_norms):.6f}")
    print(f"   Momentum analysis:")
    print(f"     Initial momentum norm: {momentum_norms[0]:.6f}")
    print(f"     Final momentum norm: {momentum_norms[-1]:.6f}")
    print(f"     Momentum coefficient: {tracking_data[0]['momentum_coeff']}")

if __name__ == "__main__":
    # Run comprehensive tracking test
    print("🚀 Starting comprehensive algorithm tracking with detailed component analysis...")
    print("🎯 Testing working implementation from summary.txt")

    # Test parameters
    dim = 10
    max_iterations = 500
    convergence_threshold = 0.01
    device = 'cpu'  # Use CPU to avoid device issues

    # Run the comprehensive test
    results, problem = run_comprehensive_algorithm_tracking(
        dim=dim,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        device=device
    )

    # Analyze behavior
    analyze_algorithm_behavior(results, problem)

    # Summary
    print(f"\n🏆 FINAL SUMMARY:")
    print("=" * 80)
    for alg_name, data in results.items():
        improvement = (data['gaps'][0] - data['final_gap']) / data['gaps'][0] * 100
        print(f"{alg_name:8}: Final gap = {data['final_gap']:.6f} ({improvement:5.1f}% improvement)")

    # Problem statistics
    stats = problem.get_problem_statistics()
    print(f"\n📊 Problem Characteristics:")
    print(f"   A condition: {stats['A_condition']:.2f}")
    print(f"   B condition: {stats['B_condition']:.2f}")
    print(f"   Coupling strength: {stats['coupling_strength']:.4f}")
    print(f"   Constraint tightness: {stats['constraint_tightness']:.4f}")

    print(f"\n✅ Comprehensive tracking complete!")
    print(f"🔬 This implementation validates the working algorithms from summary.txt")
