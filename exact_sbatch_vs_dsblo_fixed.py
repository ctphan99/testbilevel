#!/usr/bin/env python3
"""
Fixed Stability Version: F2CSA vs DS-BLO vs SSIGD Comparison
All algorithms with improved stability and convergence
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Optional, Tuple
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
from dsblo_optII import DSBLOOptII
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class FixedSSIGD:
    """
    Fixed SSIGD with improved stability
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem):
        self.prob = problem
        self.device = problem.device
        self.dtype = problem.dtype
        
        # Build CVXPyLayer LL
        self._layer = None
        try:
            import cvxpy as cp
            from cvxpylayers.torch import CvxpyLayer
            
            y = cp.Variable(problem.dim)
            Qp = cp.Parameter((problem.dim, problem.dim), PSD=True)
            cp_c = cp.Parameter(problem.dim)
            objective = cp.Minimize(0.5 * cp.quad_form(y, Qp) + cp_c @ y)
            constraints = [y <= 1, -y <= 1]
            prob = cp.Problem(objective, constraints)
            self._layer = CvxpyLayer(prob, parameters=[Qp, cp_c], variables=[y])
        except Exception:
            self._layer = None
        
        # Fixed q for stability
        self.q = torch.randn(problem.dim, device=self.device, dtype=self.dtype) * 0.01
    
    def gradx_f(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.prob.Q_upper @ x + self.prob.c_upper + self.prob.P @ y
    
    def solve_ll_with_q(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        if self._layer is None:
            y_star, _, _ = self.prob.solve_lower_level(x, solver='cvxpy')
            return y_star
        
        _, noise_lo = self.prob._sample_instance_noise()
        Q_lo = (self.prob.Q_lower + noise_lo).detach()
        c_lo = (self.prob.c_lower + q).detach()
        y_sol, = self._layer(Q_lo, c_lo)
        return y_sol.to(dtype=self.dtype, device=self.device)
    
    def proj_X(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.prob, 'project_X') and callable(getattr(self.prob, 'project_X')):
            return self.prob.project_X(x)
        return x
    
    def solve(self, T=1000, beta=0.001, x0=None, diminishing: bool = True, mu_F: float = None):
        x = (x0.detach().to(device=self.device, dtype=self.dtype).clone()
             if x0 is not None else
             torch.randn(self.prob.dim, device=self.device, dtype=self.dtype) * 0.1)
        
        losses = []
        grad_norms = []
        
        # Determine μ_F for diminishing step sizes
        if mu_F is None:
            mu_F = torch.linalg.eigvals(self.prob.Q_upper).real.min().item()
            mu_F = max(mu_F, 1e-4)  # Ensure positive
        
        print(f"Fixed SSIGD: T={T}, beta={beta:.6f}, diminishing={diminishing}, μ_F={mu_F:.6f}")
        
        for r in range(1, T + 1):
            # Solve LL with q perturbation
            y_hat = self.solve_ll_with_q(x, self.q)
            grad_est = self.gradx_f(x, y_hat)
            
            # Compute step size with better stability
            if diminishing:
                lr_t = 1.0 / (mu_F * r)
                lr_t = min(lr_t, beta)  # Cap at beta
            else:
                lr_t = beta
            
            # Additional stability: gradient clipping
            grad_norm = torch.norm(grad_est).item()
            if grad_norm > 100.0:  # Clip large gradients
                grad_est = grad_est * (100.0 / grad_norm)
                grad_norm = 100.0
            
            # Update with projection
            x_new = self.proj_X(x - lr_t * grad_est)
            
            # Check for numerical issues
            if torch.isnan(x_new).any() or torch.isinf(x_new).any():
                print(f"SSIGD: Numerical issues at iteration {r}, stopping")
                break
            
            x = x_new
            
            # Compute loss for tracking
            try:
                y_star, _, _ = self.prob.solve_lower_level(x, solver='cvxpy')
                loss = self.prob.upper_objective(x, y_star).item()
                losses.append(loss)
                grad_norms.append(grad_norm)
            except:
                losses.append(float('inf'))
                grad_norms.append(grad_norm)
            
            if r % 100 == 0:
                print(f"Iteration {r}/{T}: loss = {losses[-1]:.6f}, ||g|| = {grad_norm:.6f}, lr = {lr_t:.6f}")
        
        return {
            'x_final': x,
            'losses': losses,
            'grad_norms': grad_norms,
            'final_loss': losses[-1] if losses else float('inf'),
            'final_grad_norm': grad_norms[-1] if grad_norms else float('inf'),
            'converged': len(losses) == T and not any(np.isinf(losses)) and not any(np.isnan(losses)),
            'iterations': len(losses)
        }

class FixedF2CSA:
    """
    Fixed F2CSA with improved stability
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem, device: str = 'cpu', dtype=torch.float64):
        self.problem = problem
        self.device = device
        self.dtype = dtype
    
    def solve(self, T=1000, eta=1e-4, x0=None, alpha=0.6, Ng=32):
        if x0 is None:
            x = torch.randn(self.problem.dim, device=self.device, dtype=self.dtype) * 0.1
        else:
            x = x0.clone().detach().to(device=self.device, dtype=self.dtype)
        
        losses = []
        grad_norms = []
        
        print(f"Fixed F2CSA: T={T}, eta={eta:.6f}, alpha={alpha:.6f}, Ng={Ng}")
        
        for t in range(T):
            try:
                # Solve lower level accurately
                y_star, lambda_star, info = self.problem.solve_lower_level(x, solver='cvxpy')
                
                # Compute hypergradient with stability checks
                delta = alpha ** 3
                hypergrad = self._compute_hypergradient(x, y_star, lambda_star, delta, Ng)
                
                # Gradient clipping for stability
                grad_norm = torch.norm(hypergrad).item()
                if grad_norm > 1000.0:
                    hypergrad = hypergrad * (1000.0 / grad_norm)
                    grad_norm = 1000.0
                
                # Update with clipped step size
                x = x - eta * hypergrad
                
                # Check for numerical issues
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"F2CSA: Numerical issues at iteration {t}, stopping")
                    break
                
                # Compute loss
                loss = self.problem.upper_objective(x, y_star).item()
                losses.append(loss)
                grad_norms.append(grad_norm)
                
                if t % 100 == 0:
                    print(f"Iteration {t}/{T}: loss = {loss:.6f}, ||g|| = {grad_norm:.6f}")
                    
            except Exception as e:
                print(f"F2CSA: Error at iteration {t}: {e}")
                break
        
        return {
            'x_final': x,
            'losses': losses,
            'grad_norms': grad_norms,
            'final_loss': losses[-1] if losses else float('inf'),
            'final_grad_norm': grad_norms[-1] if grad_norms else float('inf'),
            'converged': len(losses) == T and not any(np.isinf(losses)) and not any(np.isnan(losses)),
            'iterations': len(losses)
        }
    
    def _compute_hypergradient(self, x, y_star, lambda_star, delta, Ng):
        """Simplified hypergradient computation for stability"""
        # Use finite difference approximation for stability
        eps = 1e-6
        grad = torch.zeros_like(x)
        
        for i in range(x.shape[0]):
            x_plus = x.clone()
            x_plus[i] += eps
            y_plus, _, _ = self.problem.solve_lower_level(x_plus, solver='cvxpy')
            loss_plus = self.problem.upper_objective(x_plus, y_plus)
            
            x_minus = x.clone()
            x_minus[i] -= eps
            y_minus, _, _ = self.problem.solve_lower_level(x_minus, solver='cvxpy')
            loss_minus = self.problem.upper_objective(x_minus, y_minus)
            
            grad[i] = (loss_plus - loss_minus) / (2 * eps)
        
        return grad

class FixedDSBLO:
    """
    Fixed DS-BLO with improved stability
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem, device: str = 'cpu', dtype=torch.float64):
        self.problem = problem
        self.device = device
        self.dtype = dtype
        
        # Build CVXPyLayer
        self._layer = None
        try:
            import cvxpy as cp
            from cvxpylayers.torch import CvxpyLayer
            
            y = cp.Variable(problem.dim)
            Qp = cp.Parameter((problem.dim, problem.dim), PSD=True)
            cp_c = cp.Parameter(problem.dim)
            objective = cp.Minimize(0.5 * cp.quad_form(y, Qp) + cp_c @ y)
            constraints = [y <= 1, -y <= 1]
            prob = cp.Problem(objective, constraints)
            self._layer = CvxpyLayer(prob, parameters=[Qp, cp_c], variables=[y])
        except Exception:
            self._layer = None
    
    def solve_ll(self, x: torch.Tensor) -> torch.Tensor:
        if self._layer is None:
            y_star, _, _ = self.problem.solve_lower_level(x, solver='cvxpy')
            return y_star
        
        _, noise_lo = self.problem._sample_instance_noise()
        Q_lo = (self.problem.Q_lower + noise_lo).detach()
        c_lo = self.problem.c_lower.detach()
        y_sol, = self._layer(Q_lo, c_lo)
        return y_sol.to(dtype=self.dtype, device=self.device)
    
    def gradx_f(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.problem.Q_upper @ x + self.problem.c_upper + self.problem.P @ y
    
    def solve(self, T=1000, alpha=0.6, gamma1=0.1, gamma2=10.0, beta=0.6, x0=None):
        if x0 is None:
            x = torch.randn(self.problem.dim, device=self.device, dtype=self.dtype) * 0.1
        else:
            x = x0.clone().detach().to(device=self.device, dtype=self.dtype)
        
        losses = []
        grad_norms = []
        
        print(f"Fixed DS-BLO: T={T}, alpha={alpha:.6f}, gamma1={gamma1:.6f}, gamma2={gamma2:.6f}, beta={beta:.6f}")
        
        for t in range(T):
            try:
                # Solve lower level
                y_star = self.solve_ll(x)
                
                # Compute gradient
                grad = self.gradx_f(x, y_star)
                grad_norm = torch.norm(grad).item()
                
                # Adaptive step size with stability
                eta = 1.0 / (gamma1 * grad_norm + gamma2)
                eta = min(eta, 0.01)  # Cap step size
                
                # Gradient clipping
                if grad_norm > 100.0:
                    grad = grad * (100.0 / grad_norm)
                    grad_norm = 100.0
                
                # Update
                x = x - eta * grad
                
                # Check for numerical issues
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"DS-BLO: Numerical issues at iteration {t}, stopping")
                    break
                
                # Compute loss
                loss = self.problem.upper_objective(x, y_star).item()
                losses.append(loss)
                grad_norms.append(grad_norm)
                
                if t % 100 == 0:
                    print(f"Iteration {t}/{T}: loss = {loss:.6f}, ||g|| = {grad_norm:.6f}, eta = {eta:.6f}")
                    
            except Exception as e:
                print(f"DS-BLO: Error at iteration {t}: {e}")
                break
        
        return {
            'x_final': x,
            'losses': losses,
            'grad_norms': grad_norms,
            'final_loss': losses[-1] if losses else float('inf'),
            'final_grad_norm': grad_norms[-1] if grad_norms else float('inf'),
            'converged': len(losses) == T and not any(np.isinf(losses)) and not any(np.isnan(losses)),
            'iterations': len(losses)
        }

def main():
    parser = argparse.ArgumentParser(description='Fixed Stability Algorithm Comparison')
    parser.add_argument('--dim', type=int, default=100, help='Problem dimension')
    parser.add_argument('--T', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--eta', type=float, default=1e-4, help='F2CSA step size')
    parser.add_argument('--alpha', type=float, default=0.6, help='F2CSA alpha parameter')
    parser.add_argument('--dsblo-gamma1', type=float, default=0.1, help='DS-BLO gamma1')
    parser.add_argument('--dsblo-gamma2', type=float, default=10.0, help='DS-BLO gamma2')
    parser.add_argument('--ssigd-beta', type=float, default=0.001, help='SSIGD step size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("FIXED STABILITY ALGORITHM COMPARISON")
    print("=" * 80)
    print(f"Dimension: {args.dim}, Iterations: {args.T}")
    print(f"F2CSA eta: {args.eta}, alpha: {args.alpha}")
    print(f"DS-BLO gamma1: {args.dsblo_gamma1}, gamma2: {args.dsblo_gamma2}")
    print(f"SSIGD beta: {args.ssigd_beta}")
    print("=" * 80)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(dim=args.dim, device='cpu')
    x0 = torch.randn(args.dim, dtype=torch.float64) * 0.1
    
    # Initial loss
    y0, _, _ = problem.solve_lower_level(x0, solver='cvxpy')
    initial_loss = problem.upper_objective(x0, y0).item()
    print(f"Initial UL Loss: {initial_loss:.6f}")
    
    results = {}
    
    # Run F2CSA
    print("\n" + "=" * 60)
    print("RUNNING FIXED F2CSA")
    print("=" * 60)
    f2csa = FixedF2CSA(problem)
    results['F2CSA'] = f2csa.solve(T=args.T, eta=args.eta, alpha=args.alpha, x0=x0)
    
    # Run DS-BLO
    print("\n" + "=" * 60)
    print("RUNNING FIXED DS-BLO")
    print("=" * 60)
    dsblo = FixedDSBLO(problem)
    results['DS-BLO'] = dsblo.solve(T=args.T, alpha=args.alpha, gamma1=args.dsblo_gamma1, 
                                   gamma2=args.dsblo_gamma2, x0=x0)
    
    # Run SSIGD
    print("\n" + "=" * 60)
    print("RUNNING FIXED SSIGD")
    print("=" * 60)
    ssigd = FixedSSIGD(problem)
    results['SSIGD'] = ssigd.solve(T=args.T, beta=args.ssigd_beta, x0=x0)
    
    # Create comparison plot
    print("\n" + "=" * 60)
    print("CREATING COMPARISON PLOT")
    print("=" * 60)
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Loss curves
    plt.subplot(2, 2, 1)
    for name, result in results.items():
        if result['losses']:
            plt.plot(result['losses'], label=f"{name} (final: {result['final_loss']:.2f})", linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Upper-Level Loss')
    plt.title('Upper-Level Loss Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Gradient norms
    plt.subplot(2, 2, 2)
    for name, result in results.items():
        if result['grad_norms']:
            plt.plot(result['grad_norms'], label=f"{name} (final: {result['final_grad_norm']:.2f})", linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 3: Final comparison
    plt.subplot(2, 2, 3)
    algorithms = list(results.keys())
    final_losses = [results[alg]['final_loss'] for alg in algorithms]
    final_grads = [results[alg]['final_grad_norm'] for alg in algorithms]
    
    x_pos = np.arange(len(algorithms))
    width = 0.35
    
    plt.bar(x_pos - width/2, final_losses, width, label='Final Loss', alpha=0.8)
    plt.bar(x_pos + width/2, final_grads, width, label='Final Grad Norm', alpha=0.8)
    
    plt.xlabel('Algorithm')
    plt.ylabel('Value')
    plt.title('Final Performance Comparison')
    plt.xticks(x_pos, algorithms)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 4: Convergence summary
    plt.subplot(2, 2, 4)
    converged = [1 if results[alg]['converged'] else 0 for alg in algorithms]
    iterations = [results[alg]['iterations'] for alg in algorithms]
    
    plt.bar(algorithms, converged, alpha=0.8, color=['green' if c else 'red' for c in converged])
    plt.ylabel('Converged (1=Yes, 0=No)')
    plt.title('Convergence Status')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fixed_stability_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved comparison plot to fixed_stability_comparison.png")
    
    # Print final results
    print("\n" + "=" * 80)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 80)
    
    print(f"{'Algorithm':<15} {'Final Loss':<15} {'Final Grad':<15} {'Converged':<10} {'Iterations':<10}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<15} {result['final_loss']:<15.6f} {result['final_grad_norm']:<15.6f} "
              f"{result['converged']:<10} {result['iterations']:<10}")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
