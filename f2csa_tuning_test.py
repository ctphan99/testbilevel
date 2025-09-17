#!/usr/bin/env python3
"""
F2CSA Tuning Test: Quick evaluation of different hyperparameters
Test each tuning strategy to see how it helps F2CSA hypergradient and UL loss in 500 iterations
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Optional, Tuple, List
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
import warnings
import time

warnings.filterwarnings('ignore')

class F2CSATuningTest:
    """
    F2CSA Algorithm 2 with configurable hyperparameters for tuning tests
    """
    
    def __init__(self, problem: StronglyConvexBilevelProblem, device: str = 'cpu', dtype=torch.float64):
        self.problem = problem
        self.device = device
        self.dtype = dtype
        
        # Initialize Algorithm 1 for hypergradient computation
        self.algorithm1 = F2CSAAlgorithm1Final(problem, device=device, dtype=dtype)
    
    def optimize(self, x0: torch.Tensor, T: int, eta: float, Ng: int, alpha: float, 
                 optimizer_type: str = 'sgd', warm_ll: bool = True, keep_adam_state: bool = True,
                 lr_schedule: str = 'none', lr_decay: float = 0.95, lr_step: int = 100,
                 grad_clip: float = 1.0, verbose: bool = False) -> Dict:
        """
        Optimize using F2CSA Algorithm 2 with configurable hyperparameters
        """
        x = x0.clone().detach().requires_grad_(True)
        
        # Create optimizer based on type
        if optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam([x], lr=eta, betas=(0.9, 0.999))
        elif optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD([x], lr=eta, momentum=0)
        elif optimizer_type.lower() == 'sgd_momentum':
            optimizer = torch.optim.SGD([x], lr=eta, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Learning rate scheduler
        if lr_schedule == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)
        elif lr_schedule == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_decay)
        elif lr_schedule == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
        else:
            scheduler = None
        
        # Storage
        x_history = [x.clone().detach()]
        ul_losses = []
        hypergrad_norms = []
        learning_rates = []
        
        # Warm start for lower level
        if warm_ll:
            y_opt, _ = self.problem.solve_lower_level(x)
        
        start_time = time.time()
        
        for t in range(T):
            # Compute hypergradient using Algorithm 1 (noise handled internally)
            hypergrad, y_tilde, lambda_star = self.algorithm1.oracle_sample(
                x, alpha=alpha, N_g=Ng
            )
            
            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([x], grad_clip)
            
            # Update parameters
            optimizer.zero_grad()
            x.grad = hypergrad
            optimizer.step()
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
            
            # Store results
            x_history.append(x.clone().detach())
            
            # Compute UL loss (noise handled internally by problem)
            y_opt, _ = self.problem.solve_lower_level(x)
            ul_loss_t = self.problem.upper_objective(x, y_opt).item()
            ul_losses.append(ul_loss_t)
            
            # Compute hypergradient norm
            hypergrad_norm = torch.norm(hypergrad).item()
            hypergrad_norms.append(hypergrad_norm)
            
            # Store learning rate
            learning_rates.append(optimizer.param_groups[0]['lr'])
            
            # Verbose output
            if verbose and (t % 50 == 0 or t == T-1):
                print(f"t={t:4d} | UL={ul_loss_t:8.4f} | ||g||={hypergrad_norm:8.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")
        
        end_time = time.time()
        
        return {
            'x_history': x_history,
            'ul_losses': ul_losses,
            'hypergrad_norms': hypergrad_norms,
            'learning_rates': learning_rates,
            'final_ul_loss': ul_losses[-1],
            'final_hypergrad_norm': hypergrad_norms[-1],
            'runtime': end_time - start_time,
            'optimizer_type': optimizer_type,
            'Ng': Ng,
            'alpha': alpha,
            'eta': eta,
            'lr_schedule': lr_schedule
        }

def run_tuning_tests(problem: StronglyConvexBilevelProblem, x0: torch.Tensor, T: int = 500):
    """
    Run comprehensive tuning tests for F2CSA
    """
    print("=" * 80)
    print("F2CSA TUNING TEST - 500 ITERATIONS")
    print("=" * 80)
    
    # Test configurations
    test_configs = [
        # Baseline configurations
        {'name': 'Baseline Adam', 'optimizer': 'adam', 'eta': 2e-4, 'Ng': 32, 'alpha': 0.6, 'lr_schedule': 'none'},
        {'name': 'Baseline SGD', 'optimizer': 'sgd', 'eta': 2e-4, 'Ng': 32, 'alpha': 0.6, 'lr_schedule': 'none'},
        
        # Optimizer variations
        {'name': 'SGD Momentum', 'optimizer': 'sgd_momentum', 'eta': 2e-4, 'Ng': 32, 'alpha': 0.6, 'lr_schedule': 'none'},
        
        # Learning rate variations
        {'name': 'SGD + Cosine LR', 'optimizer': 'sgd', 'eta': 2e-4, 'Ng': 32, 'alpha': 0.6, 'lr_schedule': 'cosine'},
        {'name': 'SGD + Step LR', 'optimizer': 'sgd', 'eta': 2e-4, 'Ng': 32, 'alpha': 0.6, 'lr_schedule': 'step', 'lr_decay': 0.95},
        {'name': 'SGD + Exp LR', 'optimizer': 'sgd', 'eta': 2e-4, 'Ng': 32, 'alpha': 0.6, 'lr_schedule': 'exponential', 'lr_decay': 0.995},
        
        # Higher learning rates
        {'name': 'SGD High LR', 'optimizer': 'sgd', 'eta': 5e-4, 'Ng': 32, 'alpha': 0.6, 'lr_schedule': 'none'},
        {'name': 'SGD Very High LR', 'optimizer': 'sgd', 'eta': 1e-3, 'Ng': 32, 'alpha': 0.6, 'lr_schedule': 'none'},
        
        # More gradient samples
        {'name': 'SGD + Ng=64', 'optimizer': 'sgd', 'eta': 2e-4, 'Ng': 64, 'alpha': 0.6, 'lr_schedule': 'none'},
        {'name': 'SGD + Ng=128', 'optimizer': 'sgd', 'eta': 2e-4, 'Ng': 128, 'alpha': 0.6, 'lr_schedule': 'none'},
        
        # Lower alpha values
        {'name': 'SGD + Alpha=0.3', 'optimizer': 'sgd', 'eta': 2e-4, 'Ng': 32, 'alpha': 0.3, 'lr_schedule': 'none'},
        {'name': 'SGD + Alpha=0.4', 'optimizer': 'sgd', 'eta': 2e-4, 'Ng': 32, 'alpha': 0.4, 'lr_schedule': 'none'},
        {'name': 'SGD + Alpha=0.5', 'optimizer': 'sgd', 'eta': 2e-4, 'Ng': 32, 'alpha': 0.5, 'lr_schedule': 'none'},
        
        # Combined best practices
        {'name': 'SGD + Ng=64 + Alpha=0.4', 'optimizer': 'sgd', 'eta': 2e-4, 'Ng': 64, 'alpha': 0.4, 'lr_schedule': 'none'},
        {'name': 'SGD + Ng=128 + Alpha=0.3', 'optimizer': 'sgd', 'eta': 2e-4, 'Ng': 128, 'alpha': 0.3, 'lr_schedule': 'none'},
        {'name': 'SGD + High LR + Ng=64', 'optimizer': 'sgd', 'eta': 5e-4, 'Ng': 64, 'alpha': 0.4, 'lr_schedule': 'none'},
    ]
    
    results = []
    
    for i, config in enumerate(test_configs):
        print(f"\n{i+1:2d}. Testing: {config['name']}")
        print("-" * 60)
        
        # Create algorithm instance
        algo = F2CSATuningTest(problem)
        
        # Run optimization
        result = algo.optimize(
            x0=x0,
            T=T,
            eta=config['eta'],
            Ng=config['Ng'],
            alpha=config['alpha'],
            optimizer_type=config['optimizer'],
            lr_schedule=config.get('lr_schedule', 'none'),
            lr_decay=config.get('lr_decay', 0.95),
            verbose=True
        )
        
        # Store results
        result['config_name'] = config['name']
        results.append(result)
        
        print(f"    Final UL Loss: {result['final_ul_loss']:8.4f}")
        print(f"    Final ||g||:   {result['final_hypergrad_norm']:8.4f}")
        print(f"    Runtime:       {result['runtime']:6.2f}s")
    
    return results

def plot_results(results: List[Dict], save_path: str = 'f2csa_tuning_test.png'):
    """
    Plot comparison of all tuning results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data
    names = [r['config_name'] for r in results]
    final_ul = [r['final_ul_loss'] for r in results]
    final_grad = [r['final_hypergrad_norm'] for r in results]
    runtimes = [r['runtime'] for r in results]
    
    # Plot 1: Final UL Loss
    axes[0,0].bar(range(len(names)), final_ul, color='skyblue', alpha=0.7)
    axes[0,0].set_title('Final UL Loss (Lower is Better)')
    axes[0,0].set_ylabel('UL Loss')
    axes[0,0].set_xticks(range(len(names)))
    axes[0,0].set_xticklabels(names, rotation=45, ha='right')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Final Hypergradient Norm
    axes[0,1].bar(range(len(names)), final_grad, color='lightcoral', alpha=0.7)
    axes[0,1].set_title('Final Hypergradient Norm (Lower is Better)')
    axes[0,1].set_ylabel('||g||')
    axes[0,1].set_xticks(range(len(names)))
    axes[0,1].set_xticklabels(names, rotation=45, ha='right')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Runtime
    axes[1,0].bar(range(len(names)), runtimes, color='lightgreen', alpha=0.7)
    axes[1,0].set_title('Runtime')
    axes[1,0].set_ylabel('Seconds')
    axes[1,0].set_xticks(range(len(names)))
    axes[1,0].set_xticklabels(names, rotation=45, ha='right')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: UL Loss vs Hypergradient Norm scatter
    axes[1,1].scatter(final_grad, final_ul, s=100, alpha=0.7)
    axes[1,1].set_xlabel('Final ||g||')
    axes[1,1].set_ylabel('Final UL Loss')
    axes[1,1].set_title('UL Loss vs Hypergradient Norm')
    axes[1,1].grid(True, alpha=0.3)
    
    # Add labels to scatter points
    for i, name in enumerate(names):
        axes[1,1].annotate(name, (final_grad[i], final_ul[i]), 
                          xytext=(5, 5), textcoords='offset points', 
                          fontsize=8, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nResults saved to: {save_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='F2CSA Tuning Test')
    parser.add_argument('--T', type=int, default=500, help='Number of iterations')
    parser.add_argument('--problem-noise-std', type=float, default=2e-3, help='Problem noise standard deviation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--plot-name', type=str, default='f2csa_tuning_test.png', help='Output plot name')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5, num_constraints=3, noise_std=args.problem_noise_std
    )
    
    # Initialize starting point
    x0 = torch.randn(5, dtype=torch.float64)
    
    print(f"Problem: dim=5, constraints=3, noise_std={args.problem_noise_std}")
    print(f"Starting point: {x0.numpy()}")
    print(f"Testing {args.T} iterations each...")
    
    # Run tuning tests
    results = run_tuning_tests(problem, x0, args.T)
    
    # Plot results
    plot_results(results, args.plot_name)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY - TOP 5 CONFIGURATIONS")
    print("=" * 80)
    
    # Sort by final UL loss (lower is better)
    sorted_results = sorted(results, key=lambda x: x['final_ul_loss'])
    
    for i, result in enumerate(sorted_results[:5]):
        print(f"{i+1}. {result['config_name']}")
        print(f"   Final UL Loss: {result['final_ul_loss']:8.4f}")
        print(f"   Final ||g||:   {result['final_hypergrad_norm']:8.4f}")
        print(f"   Runtime:       {result['runtime']:6.2f}s")
        print()

if __name__ == '__main__':
    main()
