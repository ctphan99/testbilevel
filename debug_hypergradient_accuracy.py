#!/usr/bin/env python3
"""
Debug hypergradient accuracy in Algorithm 1
Check if the hypergradient computation is correct before averaging by n_g
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm import F2CSAAlgorithm

def debug_hypergradient_accuracy(problem, f2csa, x_test, alpha=0.08):
    """Debug hypergradient accuracy before averaging by n_g"""
    print("=" * 80)
    print("DEBUGGING HYPERGRADIENT ACCURACY IN ALGORITHM 1")
    print("=" * 80)
    
    print(f"Testing x: {x_test.detach().numpy()}")
    print(f"Alpha: {alpha}")
    
    # Test different Ng values to see how averaging affects accuracy
    Ng_values = [1, 4, 16, 64]
    
    for Ng in Ng_values:
        print(f"\n--- Testing Ng = {Ng} ---")
        
        # Get multiple samples to analyze variance
        samples = []
        for trial in range(10):
            sample = f2csa.oracle_sample(x_test, alpha, Ng)
            samples.append(sample.detach().numpy())
        
        samples = np.array(samples)
        mean_sample = np.mean(samples, axis=0)
        std_sample = np.std(samples, axis=0)
        
        print(f"  Mean hypergradient: {mean_sample}")
        print(f"  Std hypergradient: {std_sample}")
        print(f"  Hypergradient norm: {np.linalg.norm(mean_sample):.6f}")
        print(f"  Hypergradient std norm: {np.linalg.norm(std_sample):.6f}")
        
        # Analyze components
        analyze_hypergradient_components(problem, f2csa, x_test, alpha, Ng)

def analyze_hypergradient_components(problem, f2csa, x_test, alpha, Ng):
    """Analyze the components of the hypergradient computation"""
    print(f"    --- Analyzing hypergradient components for Ng={Ng} ---")
    
    # Manual computation to understand each component
    xx = x_test.detach().clone().requires_grad_(True)
    
    # Get lower-level solution
    y_opt, lambda_opt = f2csa._solve_lower_level_algorithm2_sgd(xx, problem, alpha, {})
    
    # Compute constraint violations
    h_val = problem.A @ xx - problem.B @ y_opt - problem.b
    
    # Penalty Lagrangian terms
    alpha_1 = alpha ** (-2)
    alpha_2 = alpha ** (-4)
    h_val_penalty = h_val
    tau_delta = 0.10
    epsilon_lambda = 0.10
    rho_i = f2csa.smooth_activation(h_val_penalty, lambda_opt, tau_delta, epsilon_lambda)
    
    print(f"      y*: {y_opt.detach().numpy()}")
    print(f"      λ*: {lambda_opt.detach().numpy()}")
    print(f"      h_val: {h_val.detach().numpy()}")
    print(f"      rho_i: {rho_i.detach().numpy()}")
    
    # Compute penalty terms
    f_val = problem.upper_objective(xx, y_opt, add_noise=True)
    g_val = problem.lower_objective(xx, y_opt, add_noise=False)
    g_val_at_y_star = g_val
    
    term1 = f_val
    term2 = alpha_1 * (g_val + torch.dot(lambda_opt, h_val_penalty) - g_val_at_y_star)
    term3 = 0.5 * alpha_2 * torch.sum(rho_i * (h_val_penalty ** 2))
    penalty_term = term2 + term3
    
    print(f"      f_val: {f_val.item():.6f}")
    print(f"      g_val: {g_val.item():.6f}")
    print(f"      term2: {term2.item():.6f}")
    print(f"      term3: {term3.item():.6f}")
    print(f"      penalty_term: {penalty_term.item():.6f}")
    
    # Compute direct gradient (averaged over Ng samples)
    accumulated_grad_f = torch.zeros_like(xx)
    for _ in range(max(1, int(Ng))):
        f_val_sample = problem.upper_objective(xx, y_opt, add_noise=True)
        grad_f_sample = torch.autograd.grad(f_val_sample, xx, create_graph=False)[0]
        accumulated_grad_f += grad_f_sample
    grad_f = accumulated_grad_f / max(1, int(Ng))
    
    # Compute penalty gradient
    grad_penalty = torch.autograd.grad(penalty_term, xx, create_graph=False)[0]
    
    print(f"      grad_f: {grad_f.detach().numpy()}")
    print(f"      grad_penalty: {grad_penalty.detach().numpy()}")
    print(f"      grad_f norm: {torch.norm(grad_f).item():.6f}")
    print(f"      grad_penalty norm: {torch.norm(grad_penalty).item():.6f}")
    
    # Total hypergradient
    g_t = (grad_f + grad_penalty).detach()
    print(f"      total hypergradient: {g_t.numpy()}")
    print(f"      total norm: {torch.norm(g_t).item():.6f}")

def test_hypergradient_consistency(problem, f2csa, x_test, alpha=0.08):
    """Test if hypergradient computation is consistent"""
    print("\n" + "=" * 80)
    print("TESTING HYPERGRADIENT CONSISTENCY")
    print("=" * 80)
    
    # Test with different random seeds to see if results are consistent
    seeds = [42, 123, 456, 789, 999]
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        sample = f2csa.oracle_sample(x_test, alpha, 1)
        print(f"Seed {seed}: {sample.detach().numpy()}")
    
    # Test with different Ng values
    print(f"\nTesting different Ng values:")
    for Ng in [1, 4, 16, 64]:
        torch.manual_seed(42)  # Fixed seed for consistency
        np.random.seed(42)
        
        sample = f2csa.oracle_sample(x_test, alpha, Ng)
        print(f"Ng={Ng}: {sample.detach().numpy()}")

def compare_with_finite_differences(problem, f2csa, x_test, alpha=0.08, eps=1e-5):
    """Compare hypergradient with finite differences"""
    print("\n" + "=" * 80)
    print("COMPARING WITH FINITE DIFFERENCES")
    print("=" * 80)
    
    # Compute hypergradient
    g_hyper = f2csa.oracle_sample(x_test, alpha, 1)
    
    # Compute finite differences
    g_fd = torch.zeros_like(x_test)
    
    for i in range(x_test.shape[0]):
        # Forward difference
        x_plus = x_test.clone()
        x_plus[i] += eps
        
        # Compute objective at x_plus
        y_plus, _ = f2csa._solve_lower_level_algorithm2_sgd(x_plus, problem, alpha, {})
        f_plus = problem.upper_objective(x_plus, y_plus, add_noise=False)
        
        # Compute objective at x
        y_orig, _ = f2csa._solve_lower_level_algorithm2_sgd(x_test, problem, alpha, {})
        f_orig = problem.upper_objective(x_test, y_orig, add_noise=False)
        
        # Finite difference
        g_fd[i] = (f_plus - f_orig) / eps
    
    print(f"Hypergradient: {g_hyper.detach().numpy()}")
    print(f"Finite diff:   {g_fd.detach().numpy()}")
    print(f"Difference:    {(g_hyper - g_fd).detach().numpy()}")
    print(f"Relative error: {torch.norm(g_hyper - g_fd) / torch.norm(g_fd):.6f}")

def main():
    """Main function"""
    print("DEBUGGING HYPERGRADIENT ACCURACY IN ALGORITHM 1")
    print("=" * 80)
    
    # Create problem with balanced constraint tightening
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, seed=42)
    
    # Apply balanced constraint tightening
    problem.b = problem.b - 0.2
    problem.B = problem.B * 2.5
    problem.Q_lower = problem.Q_lower * 1.8
    
    print(f"Problem setup:")
    print(f"  b: {problem.b.detach().numpy()}")
    print(f"  B norm: {torch.norm(problem.B).item():.6f}")
    print(f"  Q_lower norm: {torch.norm(problem.Q_lower).item():.6f}")
    
    # Create F2CSA algorithm
    f2csa = F2CSAAlgorithm(
        problem=problem,
        alpha_override=0.08,
        eta_override=0.001,
        D_override=0.01,
        Ng_override=64,
        grad_ema_beta_override=0.9,
        prox_weight_override=0.1,
        grad_clip_override=1.0
    )
    
    # Test x
    x_test = torch.randn(problem.dim, dtype=torch.float64)
    
    # 1. Debug hypergradient accuracy
    debug_hypergradient_accuracy(problem, f2csa, x_test)
    
    # 2. Test consistency
    test_hypergradient_consistency(problem, f2csa, x_test)
    
    # 3. Compare with finite differences
    compare_with_finite_differences(problem, f2csa, x_test)
    
    print("\n" + "=" * 80)
    print("HYPERGRADIENT DEBUGGING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
