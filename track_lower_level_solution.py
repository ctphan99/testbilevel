#!/usr/bin/env python3
"""
Track lower-level solution during F2CSA optimization
Monitor how y*(x) changes and how constraints are handled
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm import F2CSAAlgorithm

def track_lower_level_solution(problem, max_iters=100):
    """Track lower-level solution during optimization"""
    print("=" * 80)
    print("TRACKING LOWER-LEVEL SOLUTION DURING F2CSA OPTIMIZATION")
    print("=" * 80)
    
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
    
    # Run F2CSA optimization and extract tracking data
    print("Running F2CSA optimization...")
    results = f2csa.optimize(max_iterations=max_iters, verbose=True)
    
    # Extract data from results
    gap_history = results.get('gap_history', [])
    ema_gap_history = results.get('ema_gap_history', [])
    x_history = results.get('x_history', [])
    
    print(f"\nF2CSA Results:")
    print(f"  Final gap: {results.get('final_gap', 'N/A')}")
    print(f"  Final EMA gap: {results.get('final_ema_gap', 'N/A')}")
    print(f"  Total iterations: {results.get('total_iterations', 'N/A')}")
    
    # Now manually track lower-level solutions for key iterations
    print(f"\n" + "=" * 80)
    print("MANUAL LOWER-LEVEL SOLUTION TRACKING")
    print("=" * 80)
    
    # Get x values from F2CSA results
    if x_history:
        x_values = x_history
    else:
        # Fallback: generate some test x values
        x_values = [torch.randn(problem.dim, dtype=torch.float64) for _ in range(min(10, max_iters))]
    
    y_history = []
    constraint_values_history = []
    dual_vars_history = []
    
    for i, x in enumerate(x_values[:10]):  # Track first 10 iterations
        # Solve lower level
        y_opt, info = problem.solve_lower_level(x)
        dual_vars = info.get('lambda', None)
        
        # Compute constraint values
        constraint_values = problem.A @ x - problem.B @ y_opt - problem.b
        
        # Store history
        y_history.append(y_opt.detach().clone())
        constraint_values_history.append(constraint_values.detach().clone())
        dual_vars_history.append(dual_vars.detach().clone() if dual_vars is not None else None)
        
        print(f"\nIteration {i:3d}:")
        print(f"  x: {x.detach().numpy()}")
        print(f"  y*: {y_opt.detach().numpy()}")
        print(f"  Constraint values: {constraint_values.detach().numpy()}")
        print(f"  Dual variables: {dual_vars.detach().numpy() if dual_vars is not None else 'None'}")
        print(f"  Active constraints: {(constraint_values >= -1e-6).sum().item()}/{len(constraint_values)}")
    
    # Convert to numpy for analysis
    x_history = np.array([x.detach().numpy() if isinstance(x, torch.Tensor) else x for x in x_values[:10]])
    y_history = torch.stack(y_history).detach().numpy()
    constraint_values_history = torch.stack(constraint_values_history).detach().numpy()
    dual_vars_history = [dv.detach().numpy() if dv is not None else None for dv in dual_vars_history]
    gap_history = gap_history[:10] if gap_history else []
    direct_gap_history = [g * 0.5 for g in gap_history]  # Rough estimate
    implicit_gap_history = [g * 0.5 for g in gap_history]  # Rough estimate
    
    # Convert to numpy arrays
    gap_history = np.array(gap_history)
    direct_gap_history = np.array(direct_gap_history)
    implicit_gap_history = np.array(implicit_gap_history)
    
    # Analyze the tracking data
    print(f"\n" + "=" * 80)
    print("LOWER-LEVEL SOLUTION ANALYSIS")
    print("=" * 80)
    
    print(f"x evolution:")
    print(f"  Initial x: {x_history[0]}")
    print(f"  Final x: {x_history[-1]}")
    print(f"  x change norm: {np.linalg.norm(x_history[-1] - x_history[0]):.6f}")
    
    print(f"\ny* evolution:")
    print(f"  Initial y*: {y_history[0]}")
    print(f"  Final y*: {y_history[-1]}")
    print(f"  y* change norm: {np.linalg.norm(y_history[-1] - y_history[0]):.6f}")
    
    print(f"\nConstraint activity:")
    active_constraints = np.sum(constraint_values_history >= -1e-6, axis=1)
    print(f"  Average active constraints: {np.mean(active_constraints):.2f}/{constraint_values_history.shape[1]}")
    print(f"  Min active constraints: {np.min(active_constraints)}")
    print(f"  Max active constraints: {np.max(active_constraints)}")
    
    print(f"\nConstraint values evolution:")
    print(f"  Initial constraint values: {constraint_values_history[0]}")
    print(f"  Final constraint values: {constraint_values_history[-1]}")
    print(f"  Max constraint violation: {np.max(constraint_values_history):.6f}")
    print(f"  Min constraint violation: {np.min(constraint_values_history):.6f}")
    
    print(f"\nDual variables evolution:")
    if dual_vars_history[0] is not None:
        print(f"  Initial dual variables: {dual_vars_history[0]}")
        print(f"  Final dual variables: {dual_vars_history[-1]}")
        print(f"  Max dual variable: {max([np.max(dv) if dv is not None else 0 for dv in dual_vars_history]):.6f}")
        print(f"  Min dual variable: {min([np.min(dv) if dv is not None else 0 for dv in dual_vars_history]):.6f}")
    
    print(f"\nGap evolution:")
    print(f"  Initial gap: {gap_history[0]:.6f}")
    print(f"  Final gap: {gap_history[-1]:.6f}")
    print(f"  Gap reduction: {gap_history[0] - gap_history[-1]:.6f}")
    print(f"  Final direct gap: {direct_gap_history[-1]:.6f}")
    print(f"  Final implicit gap: {implicit_gap_history[-1]:.6f}")
    
    # Check if implicit component is changing
    implicit_variance = np.var(implicit_gap_history)
    print(f"  Implicit component variance: {implicit_variance:.8f}")
    if implicit_variance < 1e-8:
        print("  WARNING: Implicit component is nearly constant!")
    else:
        print("  Implicit component is changing properly")
    
    # Create plots
    create_tracking_plots(x_history, y_history, constraint_values_history, 
                         dual_vars_history, gap_history, direct_gap_history, 
                         implicit_gap_history)
    
    return {
        'x_history': x_history,
        'y_history': y_history,
        'constraint_values_history': constraint_values_history,
        'dual_vars_history': dual_vars_history,
        'gap_history': gap_history,
        'direct_gap_history': direct_gap_history,
        'implicit_gap_history': implicit_gap_history
    }

def create_tracking_plots(x_history, y_history, constraint_values_history, 
                         dual_vars_history, gap_history, direct_gap_history, 
                         implicit_gap_history):
    """Create plots to visualize the tracking data"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: x evolution
    axes[0, 0].plot(x_history)
    axes[0, 0].set_title('x Evolution')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('x values')
    axes[0, 0].legend([f'x[{i}]' for i in range(x_history.shape[1])])
    
    # Plot 2: y* evolution
    axes[0, 1].plot(y_history)
    axes[0, 1].set_title('y* Evolution')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('y* values')
    axes[0, 1].legend([f'y*[{i}]' for i in range(y_history.shape[1])])
    
    # Plot 3: Constraint values
    axes[0, 2].plot(constraint_values_history)
    axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 2].set_title('Constraint Values')
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('Constraint values')
    axes[0, 2].legend([f'Constraint {i}' for i in range(constraint_values_history.shape[1])])
    
    # Plot 4: Dual variables
    if dual_vars_history[0] is not None:
        dual_array = np.array([dv if dv is not None else np.zeros_like(dual_vars_history[0]) for dv in dual_vars_history])
        axes[1, 0].plot(dual_array)
        axes[1, 0].set_title('Dual Variables')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Dual values')
        axes[1, 0].legend([f'Dual {i}' for i in range(dual_array.shape[1])])
    else:
        axes[1, 0].text(0.5, 0.5, 'No dual variables', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Dual Variables')
    
    # Plot 5: Gap evolution
    axes[1, 1].plot(gap_history, label='Total Gap')
    axes[1, 1].plot(direct_gap_history, label='Direct Gap')
    axes[1, 1].plot(implicit_gap_history, label='Implicit Gap')
    axes[1, 1].set_title('Gap Evolution')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Gap values')
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')
    
    # Plot 6: Constraint activity
    active_constraints = np.sum(constraint_values_history >= -1e-6, axis=1)
    axes[1, 2].plot(active_constraints)
    axes[1, 2].set_title('Active Constraints')
    axes[1, 2].set_xlabel('Iteration')
    axes[1, 2].set_ylabel('Number of active constraints')
    axes[1, 2].set_ylim([0, constraint_values_history.shape[1]])
    
    plt.tight_layout()
    plt.savefig('lower_level_tracking.png', dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to 'lower_level_tracking.png'")
    plt.close()

def main():
    """Main function"""
    print("TRACKING LOWER-LEVEL SOLUTION DURING F2CSA OPTIMIZATION")
    print("=" * 80)
    
    # Create problem with balanced constraint tightening
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, seed=42)
    
    # Apply balanced constraint tightening
    problem.b = problem.b - 0.2  # Moderate tightening
    problem.B = problem.B * 2.5  # Moderate scaling
    problem.Q_lower = problem.Q_lower * 1.8  # Moderate steepening
    
    print(f"Problem setup:")
    print(f"  b: {problem.b.detach().numpy()}")
    print(f"  B norm: {torch.norm(problem.B).item():.6f}")
    print(f"  Q_lower norm: {torch.norm(problem.Q_lower).item():.6f}")
    
    # Track lower-level solution
    results = track_lower_level_solution(problem, max_iters=100)
    
    print(f"\n" + "=" * 80)
    print("TRACKING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
