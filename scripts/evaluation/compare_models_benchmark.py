"""
Compare Multiple Models using Benchmarking Metrics

This script loads benchmark results from multiple models and compares them
using the naive baseline approach.

Usage:
    python compare_models_benchmark.py --results_dir <directory_with_json_files>
"""

import os
import sys
import json
import argparse
import glob
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from evaluation.benchmark_metrics import compare_models, print_benchmarking_results


def load_all_results(results_dir: str, pattern: str = "benchmark_results_*.json") -> Dict[str, Dict]:
    """
    Load all benchmark results from JSON files in a directory.
    
    Parameters:
    - results_dir: Directory containing benchmark result JSON files
    - pattern: Glob pattern to match result files
    
    Returns:
    - Dictionary mapping model names to their benchmark results
    """
    results_files = glob.glob(os.path.join(results_dir, pattern))
    
    if not results_files:
        print(f"No benchmark result files found in {results_dir}")
        return {}
    
    all_results = {}
    
    for file_path in results_files:
        # Extract model name from filename
        filename = os.path.basename(file_path)
        model_name = filename.replace('benchmark_results_', '').replace('.json', '')
        
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        all_results[model_name] = results
        print(f"✓ Loaded results for: {model_name}")
    
    return all_results


def create_comparison_table(all_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a comparison table with all metrics for all models.
    
    Parameters:
    - all_results: Dictionary mapping model names to benchmark results
    
    Returns:
    - DataFrame with comparison metrics
    """
    rows = []
    
    for model_name, results in all_results.items():
        row = {
            'Model': model_name,
            'R²': results['model_metrics']['R2'],
            'Spearman': results['model_metrics']['Spearman_Correlation'],
            'Pearson': results['model_metrics']['Pearson_Correlation'],
            'MAE': results['model_metrics']['MAE'],
            'Normalized_MAE': results['model_metrics']['Normalized_MAE'],
            'MSE': results['model_metrics']['MSE'],
            'Naive_MAE': results['baseline_metrics']['Naive_MAE'],
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by Normalized MAE (descending - higher is better)
    df = df.sort_values('Normalized_MAE', ascending=False)
    
    return df


def plot_comparison(all_results: Dict[str, Dict], output_dir: str):
    """
    Create visualization plots comparing models.
    
    Parameters:
    - all_results: Dictionary mapping model names to benchmark results
    - output_dir: Directory to save plots
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Benchmarking Comparison', fontsize=16, fontweight='bold')
    
    # Prepare data
    models = list(all_results.keys())
    r2_scores = [all_results[m]['model_metrics']['R2'] for m in models]
    spearman_scores = [all_results[m]['model_metrics']['Spearman_Correlation'] for m in models]
    pearson_scores = [all_results[m]['model_metrics']['Pearson_Correlation'] for m in models]
    mae_scores = [all_results[m]['model_metrics']['MAE'] for m in models]
    normalized_mae_scores = [all_results[m]['model_metrics']['Normalized_MAE'] for m in models]
    mse_scores = [all_results[m]['model_metrics']['MSE'] for m in models]
    
    # Plot 1: R² Score
    ax1 = axes[0, 0]
    bars1 = ax1.barh(models, r2_scores, color='skyblue')
    ax1.axvline(x=0, color='red', linestyle='--', label='Naive Baseline')
    ax1.set_xlabel('R² Score')
    ax1.set_title('R² Score (Higher is Better)')
    ax1.legend()
    
    # Color bars based on performance
    for i, bar in enumerate(bars1):
        if r2_scores[i] > 0.7:
            bar.set_color('green')
        elif r2_scores[i] > 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Plot 2: Correlations
    ax2 = axes[0, 1]
    x = range(len(models))
    width = 0.35
    ax2.bar([i - width/2 for i in x], spearman_scores, width, label='Spearman', alpha=0.8)
    ax2.bar([i + width/2 for i in x], pearson_scores, width, label='Pearson', alpha=0.8)
    ax2.axhline(y=0, color='red', linestyle='--', label='Naive Baseline')
    ax2.set_ylabel('Correlation')
    ax2.set_title('Correlation Coefficients (Higher is Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    
    # Plot 3: Normalized MAE
    ax3 = axes[0, 2]
    bars3 = ax3.barh(models, normalized_mae_scores, color='lightgreen')
    ax3.axvline(x=0, color='red', linestyle='--', label='Naive Baseline')
    ax3.set_xlabel('Normalized MAE')
    ax3.set_title('Normalized MAE (Higher is Better)')
    ax3.legend()
    
    # Color bars based on performance
    for i, bar in enumerate(bars3):
        if normalized_mae_scores[i] > 0.7:
            bar.set_color('green')
        elif normalized_mae_scores[i] > 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Plot 4: MAE
    ax4 = axes[1, 0]
    bars4 = ax4.barh(models, mae_scores, color='salmon')
    ax4.set_xlabel('MAE')
    ax4.set_title('Mean Absolute Error (Lower is Better)')
    
    # Plot 5: MSE
    ax5 = axes[1, 1]
    bars5 = ax5.barh(models, mse_scores, color='plum')
    ax5.set_xlabel('MSE')
    ax5.set_title('Mean Squared Error (Lower is Better)')
    
    # Plot 6: Overall Performance Radar/Summary
    ax6 = axes[1, 2]
    # Normalize all metrics to 0-1 scale for fair comparison
    metrics_normalized = []
    for m in models:
        r2 = all_results[m]['model_metrics']['R2']
        norm_mae = all_results[m]['model_metrics']['Normalized_MAE']
        spear = all_results[m]['model_metrics']['Spearman_Correlation']
        pear = all_results[m]['model_metrics']['Pearson_Correlation']
        
        # Average of key metrics (all 0-1 scale, higher is better)
        avg_score = (r2 + norm_mae + spear + pear) / 4
        metrics_normalized.append(avg_score)
    
    bars6 = ax6.barh(models, metrics_normalized, color='gold')
    ax6.set_xlabel('Overall Performance Score')
    ax6.set_title('Overall Performance (Average of Normalized Metrics)')
    ax6.set_xlim(0, 1)
    
    # Color bars based on performance
    for i, bar in enumerate(bars6):
        if metrics_normalized[i] > 0.7:
            bar.set_color('green')
        elif metrics_normalized[i] > 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'model_comparison_benchmarks.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to {plot_path}")
    
    plt.close()


def generate_report(all_results: Dict[str, Dict], output_dir: str):
    """
    Generate a comprehensive text report of the comparison.
    
    Parameters:
    - all_results: Dictionary mapping model names to benchmark results
    - output_dir: Directory to save report
    """
    report_path = os.path.join(output_dir, 'benchmark_comparison_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("MODEL BENCHMARKING COMPARISON REPORT\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Number of models compared: {len(all_results)}\n")
        f.write(f"Models: {', '.join(all_results.keys())}\n\n")
        
        f.write("="*100 + "\n")
        f.write("BASELINE INFORMATION\n")
        f.write("="*100 + "\n")
        f.write("Naive Baseline Approach:\n")
        f.write("  - R² Baseline: 0 (mean-based predictor explains no variance)\n")
        f.write("  - Spearman Correlation Baseline: 0 (no monotonic relationship)\n")
        f.write("  - Pearson Correlation Baseline: 0 (no linear relationship)\n")
        f.write("  - Normalized MAE: 1 - (MAE / Naive_MAE)\n")
        f.write("    where Naive_MAE is the MAE of predicting the mean for all samples\n\n")
        
        # Sort models by overall performance
        model_scores = []
        for model_name, results in all_results.items():
            r2 = results['model_metrics']['R2']
            norm_mae = results['model_metrics']['Normalized_MAE']
            spear = results['model_metrics']['Spearman_Correlation']
            pear = results['model_metrics']['Pearson_Correlation']
            avg_score = (r2 + norm_mae + spear + pear) / 4
            model_scores.append((model_name, avg_score, results))
        
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        f.write("="*100 + "\n")
        f.write("RANKING BY OVERALL PERFORMANCE\n")
        f.write("="*100 + "\n\n")
        
        for rank, (model_name, score, results) in enumerate(model_scores, 1):
            f.write(f"RANK {rank}: {model_name} (Overall Score: {score:.4f})\n")
            f.write("-"*100 + "\n")
            f.write(f"  Model Metrics:\n")
            for metric, value in results['model_metrics'].items():
                f.write(f"    {metric:<25} {value:.6f}\n")
            f.write(f"  Improvement Over Baseline:\n")
            for metric, value in results['improvement_over_baseline'].items():
                f.write(f"    {metric:<25} {value:.6f}\n")
            f.write("\n")
        
        f.write("="*100 + "\n")
        f.write("DETAILED COMPARISON BY METRIC\n")
        f.write("="*100 + "\n\n")
        
        metrics = ['R2', 'Normalized_MAE', 'Spearman_Correlation', 'Pearson_Correlation', 'MAE', 'MSE']
        for metric in metrics:
            f.write(f"{metric}:\n")
            f.write("-"*100 + "\n")
            
            metric_values = [(m, all_results[m]['model_metrics'][metric]) for m in all_results.keys()]
            
            # Sort appropriately (higher is better for most, lower for MAE/MSE)
            if metric in ['MAE', 'MSE']:
                metric_values.sort(key=lambda x: x[1])  # Lower is better
            else:
                metric_values.sort(key=lambda x: x[1], reverse=True)  # Higher is better
            
            for model, value in metric_values:
                f.write(f"  {model:<30} {value:.6f}\n")
            f.write("\n")
        
        f.write("="*100 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*100 + "\n")
    
    print(f"✓ Comparison report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare multiple models using benchmarking metrics")
    
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing benchmark result JSON files")
    parser.add_argument("--pattern", type=str, default="benchmark_results_*.json",
                       help="Glob pattern to match result files (default: benchmark_results_*.json)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save comparison outputs (default: same as results_dir)")
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all results
    print("Loading benchmark results...")
    all_results = load_all_results(args.results_dir, args.pattern)
    
    if not all_results:
        print("No results found. Exiting.")
        return
    
    print(f"\nLoaded {len(all_results)} model results\n")
    
    # Create comparison table
    print("Creating comparison table...")
    comparison_df = create_comparison_table(all_results)
    
    # Save comparison table
    table_path = os.path.join(output_dir, 'model_comparison_table.csv')
    comparison_df.to_csv(table_path, index=False)
    print(f"✓ Comparison table saved to {table_path}\n")
    
    # Print table
    print("\n" + "="*120)
    print("MODEL COMPARISON TABLE")
    print("="*120)
    print(comparison_df.to_string(index=False))
    print("="*120 + "\n")
    
    # Use comparison function from benchmark_metrics
    print("\n")
    compare_models(all_results, metric='Normalized_MAE')
    
    # Create plots
    print("\nGenerating comparison plots...")
    plot_comparison(all_results, output_dir)
    
    # Generate report
    print("\nGenerating comparison report...")
    generate_report(all_results, output_dir)
    
    print("\n" + "="*120)
    print("COMPARISON COMPLETE!")
    print("="*120)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
