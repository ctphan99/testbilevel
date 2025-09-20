#!/usr/bin/env python3
"""
Analyze SSIGD tuning results from batch jobs
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_ssigd_results():
    """Analyze results from SSIGD parameter tuning batch jobs"""
    
    print("🔬 SSIGD Results Analysis")
    print("=" * 50)
    
    # Find all result files
    result_files = glob.glob("ssigd_results_beta*_muF*.txt")
    
    if not result_files:
        print("No result files found. Make sure batch jobs have completed.")
        return
    
    print(f"Found {len(result_files)} result files")
    
    # Parse results
    results = []
    for file in result_files:
        try:
            with open(file, 'r') as f:
                data = {}
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        data[key] = float(value)
                results.append(data)
        except Exception as e:
            print(f"Error parsing {file}: {e}")
    
    if not results:
        print("No valid results found.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    print(f"Loaded {len(df)} results")
    print()
    
    # Sort by final loss
    df_sorted = df.sort_values('final_loss')
    
    print("Top 10 Parameter Combinations:")
    print("-" * 60)
    print(f"{'Rank':<5} {'Beta':<8} {'mu_F':<8} {'Final Loss':<12} {'Improvement':<12} {'Reduction %':<12}")
    print("-" * 60)
    
    for i, (_, row) in enumerate(df_sorted.head(10).iterrows()):
        print(f"{i+1:<5} {row['beta']:<8.4f} {row['mu_F']:<8.2f} {row['final_loss']:<12.6f} {row['improvement']:<12.6f} {row['reduction_pct']:<12.2f}")
    
    # Best parameters
    best = df_sorted.iloc[0]
    print(f"\n🏆 Best Parameters:")
    print(f"  Beta: {best['beta']:.6f}")
    print(f"  mu_F: {best['mu_F']:.6f}")
    print(f"  Final Loss: {best['final_loss']:.6f}")
    print(f"  Improvement: {best['improvement']:.6f}")
    print(f"  Reduction: {best['reduction_pct']:.2f}%")
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    
    # 1. Parameter heatmap
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Pivot for heatmap
    pivot_loss = df.pivot(index='mu_F', columns='beta', values='final_loss')
    pivot_improvement = df.pivot(index='mu_F', columns='beta', values='improvement')
    
    # Loss heatmap
    im1 = axes[0, 0].imshow(pivot_loss.values, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Final Loss Heatmap')
    axes[0, 0].set_xlabel('Beta')
    axes[0, 0].set_ylabel('mu_F')
    axes[0, 0].set_xticks(range(len(pivot_loss.columns)))
    axes[0, 0].set_xticklabels([f'{x:.4f}' for x in pivot_loss.columns], rotation=45)
    axes[0, 0].set_yticks(range(len(pivot_loss.index)))
    axes[0, 0].set_yticklabels([f'{x:.2f}' for x in pivot_loss.index])
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Improvement heatmap
    im2 = axes[0, 1].imshow(pivot_improvement.values, cmap='plasma', aspect='auto')
    axes[0, 1].set_title('Improvement Heatmap')
    axes[0, 1].set_xlabel('Beta')
    axes[0, 1].set_ylabel('mu_F')
    axes[0, 1].set_xticks(range(len(pivot_improvement.columns)))
    axes[0, 1].set_xticklabels([f'{x:.4f}' for x in pivot_improvement.columns], rotation=45)
    axes[0, 1].set_yticks(range(len(pivot_improvement.index)))
    axes[0, 1].set_yticklabels([f'{x:.2f}' for x in pivot_improvement.index])
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 2. Scatter plots
    axes[1, 0].scatter(df['beta'], df['final_loss'], c=df['mu_F'], cmap='viridis', alpha=0.7)
    axes[1, 0].set_xlabel('Beta')
    axes[1, 0].set_ylabel('Final Loss')
    axes[1, 0].set_title('Final Loss vs Beta (colored by mu_F)')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    
    axes[1, 1].scatter(df['mu_F'], df['final_loss'], c=df['beta'], cmap='plasma', alpha=0.7)
    axes[1, 1].set_xlabel('mu_F')
    axes[1, 1].set_ylabel('Final Loss')
    axes[1, 1].set_title('Final Loss vs mu_F (colored by Beta)')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('ssigd_parameter_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved analysis plot to ssigd_parameter_analysis.png")
    
    # Save detailed results
    df_sorted.to_csv('ssigd_tuning_results.csv', index=False)
    print("Saved detailed results to ssigd_tuning_results.csv")
    
    print("\n" + "=" * 50)
    print("Analysis complete!")
    print("=" * 50)

if __name__ == "__main__":
    analyze_ssigd_results()
