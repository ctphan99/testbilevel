
def run_comprehensive_comparison():
    """Run comprehensive comparison of F2CSA, SSIGD, and DS-BLO"""
    print("🔬 COMPREHENSIVE BILEVEL OPTIMIZATION COMPARISON")
    print("=" * 60)
    print("Algorithms: F2CSA (Enhanced) | SSIGD | DS-BLO")
    print("Problem: Strongly Convex Constrained Bilevel Optimization")
    print("=" * 60)

    # Create strongly convex problem
    device = 'cpu'  # Use CPU for stability
    problem = StronglyConvexBilevelProblem(dim=100, num_constraints=3, device=device, seed=42)

    # Algorithm configurations
    algorithms = {
        'F2CSA': {'class': F2CSA, 'params': {'N_g': 5, 'alpha': 0.3}},
        'SSIGD': {'class': SSIGD, 'params': {'smoothing_samples': 5, 'epsilon': 0.01}},
        'DS-BLO': {'class': DSBLO, 'params': {'momentum': 0.9, 'sigma': 0.01}}
    }

    results = {}
    comparison_data = []

    # Run each algorithm
    for alg_name, alg_config in algorithms.items():
        print(f"\n🧪 Running {alg_name}")
        print("-" * 40)

        # Reset random seed for fair comparison
        torch.manual_seed(42)
        np.random.seed(42)

        # Create algorithm instance
        algorithm = alg_config['class'](problem, **alg_config['params'])

        # Run optimization
        result = algorithm.optimize(max_iterations=1000, convergence_threshold=0.1)
        results[alg_name] = result

        # Add to comparison data
        comparison_data.append({
            'Algorithm': alg_name,
            'Final_Objective': result['final_objective'],
            'Final_Gap': result['final_gap'],
            'Best_Gap': result['best_gap'],
            'Objective_Improvement': result['objective_improvement'],
            'Total_Iterations': result['total_iterations'],
            'Total_Time': result['total_time'],
            'Converged': result['converged'],
            'Emergency_Resets': result.get('emergency_resets', 0)
        })

        print(f"📊 {alg_name} Results:")
        print(f"   Final objective: {result['final_objective']:.6f}")
        print(f"   Final gap: {result['final_gap']:.6f}")
        print(f"   Best gap: {result['best_gap']:.6f}")
        print(f"   Iterations: {result['total_iterations']}")
        print(f"   Time: {result['total_time']:.2f}s")
        print(f"   Converged: {'✅ YES' if result['converged'] else '❌ NO'}")
        if 'emergency_resets' in result:
            print(f"   Emergency resets: {result['emergency_resets']}")

    # Create comparison DataFrame
    df = pd.DataFrame(comparison_data)
    df.to_csv('bilevel_algorithms_comparison.csv', index=False)

    print(f"\n🏆 COMPREHENSIVE COMPARISON RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))

    # Find best performers
    best_gap = df.loc[df['Best_Gap'].idxmin()]
    best_time = df.loc[df['Total_Time'].idxmin()]
    most_stable = df.loc[df['Emergency_Resets'].idxmin()]

    print(f"\n🥇 PERFORMANCE RANKINGS:")
    print(f"Best Gap: {best_gap['Algorithm']} (gap={best_gap['Best_Gap']:.6f})")
    print(f"Fastest: {best_time['Algorithm']} (time={best_time['Total_Time']:.2f}s)")
    print(f"Most Stable: {most_stable['Algorithm']} (resets={most_stable['Emergency_Resets']})")

    # Create comprehensive plots
    create_comparison_plots(results)

    return results, df

def create_comparison_plots(results):
    """Create comprehensive comparison plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Bilevel Optimization Algorithms Comparison', fontsize=16, fontweight='bold')

    colors = {'F2CSA': 'blue', 'SSIGD': 'green', 'DS-BLO': 'red'}

    # Plot 1: Gap convergence
    ax1 = axes[0, 0]
    for alg_name, result in results.items():
        history = result['history']
        iterations = [h['iteration'] for h in history]
        gaps = [h['gap'] for h in history]
        ax1.semilogy(iterations, gaps, color=colors[alg_name], linewidth=2, label=alg_name)

    ax1.axhline(y=0.1, color='black', linestyle='--', alpha=0.7, label='Convergence Threshold')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Gap (||∇F||)')
    ax1.set_title('Gap Convergence (Log Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Objective convergence
    ax2 = axes[0, 1]
    for alg_name, result in results.items():
        history = result['history']
        iterations = [h['iteration'] for h in history]
        objectives = [h['bilevel_objective'] for h in history]
        ax2.plot(iterations, objectives, color=colors[alg_name], linewidth=2, label=alg_name)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Bilevel Objective')
    ax2.set_title('Objective Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Time-based convergence
    ax3 = axes[0, 2]
    for alg_name, result in results.items():
        history = result['history']
        times = [h['time'] for h in history]
        gaps = [h['gap'] for h in history]
        ax3.semilogy(times, gaps, color=colors[alg_name], linewidth=2, label=alg_name)

    ax3.axhline(y=0.1, color='black', linestyle='--', alpha=0.7, label='Convergence Threshold')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Gap (||∇F||)')
    ax3.set_title('Time-based Convergence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Constraint violation
    ax4 = axes[1, 0]
    for alg_name, result in results.items():
        history = result['history']
        iterations = [h['iteration'] for h in history]
        violations = [h['constraint_violation'] for h in history]
        ax4.plot(iterations, violations, color=colors[alg_name], linewidth=2, label=alg_name)

    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Constraint Violation')
    ax4.set_title('Constraint Satisfaction')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Final performance comparison
    ax5 = axes[1, 1]
    algorithms = list(results.keys())
    final_gaps = [results[alg]['final_gap'] for alg in algorithms]
    bars = ax5.bar(algorithms, final_gaps, color=[colors[alg] for alg in algorithms], alpha=0.7)

    ax5.set_ylabel('Final Gap')
    ax5.set_title('Final Gap Comparison')
    ax5.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, gap in zip(bars, final_gaps):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_gaps)*0.01,
                f'{gap:.4f}', ha='center', va='bottom', fontweight='bold')

    # Plot 6: Convergence rate comparison
    ax6 = axes[1, 2]
    convergence_rates = []
    for alg in algorithms:
        history = results[alg]['history']
        initial_gap = history[0]['gap']
        final_gap = results[alg]['final_gap']
        rate = (initial_gap - final_gap) / initial_gap * 100
        convergence_rates.append(rate)

    bars = ax6.bar(algorithms, convergence_rates, color=[colors[alg] for alg in algorithms], alpha=0.7)
    ax6.set_ylabel('Gap Reduction (%)')
    ax6.set_title('Convergence Rate Comparison')
    ax6.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, rate in zip(bars, convergence_rates):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(convergence_rates)*0.01,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('bilevel_algorithms_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    results, comparison_df = run_comprehensive_comparison()

    print(f"\n✅ Comprehensive comparison complete!")
    print(f"Files generated:")
    print(f"  - bilevel_algorithms_comparison.csv")
    print(f"  - bilevel_algorithms_comparison.png")
    print(f"\n🎯 All algorithms tested with strongly convex problem setup!")
