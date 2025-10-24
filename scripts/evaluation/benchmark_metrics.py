"""
Benchmarking Module: Naive Baseline Metrics

This module implements naive baseline calculations and normalized metrics
for benchmarking model performance against simple mean-based predictors.

According to the benchmarking approach described in Section 5.2:
- R² and correlation baselines are 0 (no predictive power)
- MAE is normalized relative to the naive baseline
- Normalized MAE = 1 - (MAE / Naive_MAE)
"""

import numpy as np
import torch
from typing import Dict, Tuple, Union


def compute_mae(predictions: Union[torch.Tensor, np.ndarray], 
                targets: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute Mean Absolute Error (MAE).
    
    Parameters:
    - predictions: Predicted values
    - targets: Actual target values
    
    Returns:
    - float: MAE value
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().detach().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().detach().numpy()
    
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    mae = np.mean(np.abs(predictions - targets))
    return float(mae)


def compute_mse(predictions: Union[torch.Tensor, np.ndarray], 
                targets: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute Mean Squared Error (MSE).
    
    Parameters:
    - predictions: Predicted values
    - targets: Actual target values
    
    Returns:
    - float: MSE value
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().detach().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().detach().numpy()
    
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    mse = np.mean((predictions - targets) ** 2)
    return float(mse)


def compute_naive_baseline_mae(targets: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute the naive baseline MAE.
    
    The naive baseline assumes that the predicted change in traffic volume 
    for each edge is equal to the average observed change in traffic volume 
    across all edges in the test set.
    
    Parameters:
    - targets: Actual target values
    
    Returns:
    - float: Naive baseline MAE
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().detach().numpy()
    
    targets = targets.flatten()
    mean_target = np.mean(targets)
    
    # Naive prediction: predict the mean for all samples
    naive_predictions = np.full_like(targets, mean_target)
    
    naive_mae = np.mean(np.abs(naive_predictions - targets))
    return float(naive_mae)


def compute_normalized_mae(predictions: Union[torch.Tensor, np.ndarray], 
                           targets: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute normalized MAE relative to the naive baseline.
    
    Normalized MAE = 1 - (MAE / Naive_MAE)
    
    This ensures that:
    - 1 represents perfect predictions (zero error)
    - 0 indicates no improvement over the naive mean-based predictor
    - Values can be negative if the model performs worse than the baseline
    
    Parameters:
    - predictions: Predicted values
    - targets: Actual target values
    
    Returns:
    - float: Normalized MAE (ranging from 0 to 1 for good models)
    """
    mae = compute_mae(predictions, targets)
    naive_mae = compute_naive_baseline_mae(targets)
    
    if naive_mae == 0:
        # Edge case: if naive MAE is 0, return 1 if MAE is also 0, else 0
        return 1.0 if mae == 0 else 0.0
    
    normalized_mae = 1.0 - (mae / naive_mae)
    return float(normalized_mae)


def compute_all_metrics(predictions: Union[torch.Tensor, np.ndarray], 
                       targets: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    """
    Compute all evaluation metrics including MAE, MSE, and normalized MAE.
    
    Parameters:
    - predictions: Predicted values
    - targets: Actual target values
    
    Returns:
    - dict: Dictionary containing all metrics
    """
    mae = compute_mae(predictions, targets)
    mse = compute_mse(predictions, targets)
    naive_mae = compute_naive_baseline_mae(targets)
    normalized_mae = compute_normalized_mae(predictions, targets)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'Naive_MAE': naive_mae,
        'Normalized_MAE': normalized_mae,
    }
    
    return metrics


def compute_benchmarking_metrics(predictions: Union[torch.Tensor, np.ndarray], 
                                 targets: Union[torch.Tensor, np.ndarray],
                                 r2: float,
                                 spearman_corr: float,
                                 pearson_corr: float) -> Dict[str, Dict[str, float]]:
    """
    Compute comprehensive benchmarking metrics with naive baselines.
    
    This function computes:
    1. Model metrics (MAE, MSE, R², Spearman, Pearson, Normalized MAE)
    2. Naive baseline metrics (for comparison)
    3. Improvement metrics (how much better than baseline)
    
    Parameters:
    - predictions: Model predicted values
    - targets: Actual target values
    - r2: R² score from model
    - spearman_corr: Spearman correlation coefficient from model
    - pearson_corr: Pearson correlation coefficient from model
    
    Returns:
    - dict: Nested dictionary with model metrics, baselines, and improvements
    """
    # Compute error metrics
    mae = compute_mae(predictions, targets)
    mse = compute_mse(predictions, targets)
    naive_mae = compute_naive_baseline_mae(targets)
    normalized_mae = compute_normalized_mae(predictions, targets)
    
    # Naive baselines for R² and correlations are 0
    # (as they don't explain any variance or show any relationship)
    naive_r2 = 0.0
    naive_spearman = 0.0
    naive_pearson = 0.0
    
    # Compute improvement over baseline
    # For R², Spearman, and Pearson: improvement is simply the value itself
    # (since baseline is 0)
    improvement_r2 = r2 - naive_r2
    improvement_spearman = spearman_corr - naive_spearman
    improvement_pearson = pearson_corr - naive_pearson
    
    # For MAE: improvement is the normalized MAE
    improvement_mae = normalized_mae
    
    # Percentage improvement for interpretability
    # We can express how much better we are doing (in percentage points)
    # For correlations and R², this is straightforward
    # For MAE, we use the normalized value (already 0-1 scale)
    
    results = {
        'model_metrics': {
            'MAE': mae,
            'MSE': mse,
            'R2': r2,
            'Spearman_Correlation': spearman_corr,
            'Pearson_Correlation': pearson_corr,
            'Normalized_MAE': normalized_mae,
        },
        'baseline_metrics': {
            'Naive_MAE': naive_mae,
            'Naive_R2': naive_r2,
            'Naive_Spearman': naive_spearman,
            'Naive_Pearson': naive_pearson,
        },
        'improvement_over_baseline': {
            'R2_Improvement': improvement_r2,
            'Spearman_Improvement': improvement_spearman,
            'Pearson_Improvement': improvement_pearson,
            'Normalized_MAE': improvement_mae,  # This IS the improvement metric
        }
    }
    
    return results


def print_benchmarking_results(results: Dict[str, Dict[str, float]], 
                               model_name: str = "Model"):
    """
    Pretty print benchmarking results with clear formatting.
    
    Parameters:
    - results: Dictionary from compute_benchmarking_metrics
    - model_name: Name of the model for display
    """
    print("\n" + "="*80)
    print(f"BENCHMARKING RESULTS: {model_name}")
    print("="*80)
    
    print("\n📊 MODEL PERFORMANCE METRICS:")
    print("-" * 80)
    for metric, value in results['model_metrics'].items():
        print(f"  {metric:.<30} {value:.6f}")
    
    print("\n📏 NAIVE BASELINE METRICS:")
    print("-" * 80)
    for metric, value in results['baseline_metrics'].items():
        print(f"  {metric:.<30} {value:.6f}")
    
    print("\n📈 IMPROVEMENT OVER NAIVE BASELINE:")
    print("-" * 80)
    for metric, value in results['improvement_over_baseline'].items():
        status = "✓" if value > 0 else "✗"
        print(f"  {status} {metric:.<28} {value:.6f}")
    
    print("\n💡 INTERPRETATION:")
    print("-" * 80)
    
    # R² interpretation
    r2 = results['model_metrics']['R2']
    if r2 > 0.9:
        r2_quality = "Excellent"
    elif r2 > 0.7:
        r2_quality = "Good"
    elif r2 > 0.5:
        r2_quality = "Moderate"
    elif r2 > 0:
        r2_quality = "Poor but better than baseline"
    else:
        r2_quality = "Worse than baseline"
    print(f"  R² Score: {r2_quality} (explains {r2*100:.2f}% of variance)")
    
    # Normalized MAE interpretation
    norm_mae = results['model_metrics']['Normalized_MAE']
    if norm_mae > 0.9:
        mae_quality = "Excellent"
    elif norm_mae > 0.7:
        mae_quality = "Good"
    elif norm_mae > 0.5:
        mae_quality = "Moderate"
    elif norm_mae > 0:
        mae_quality = "Poor but better than baseline"
    else:
        mae_quality = "Worse than baseline"
    print(f"  Normalized MAE: {mae_quality} ({norm_mae*100:.2f}% improvement over naive baseline)")
    
    # Correlation interpretation
    spearman = results['model_metrics']['Spearman_Correlation']
    pearson = results['model_metrics']['Pearson_Correlation']
    avg_corr = (spearman + pearson) / 2
    if avg_corr > 0.9:
        corr_quality = "Very Strong"
    elif avg_corr > 0.7:
        corr_quality = "Strong"
    elif avg_corr > 0.5:
        corr_quality = "Moderate"
    elif avg_corr > 0.3:
        corr_quality = "Weak"
    else:
        corr_quality = "Very Weak"
    print(f"  Correlations: {corr_quality} relationship with ground truth")
    
    print("="*80 + "\n")


def compare_models(model_results: Dict[str, Dict], metric: str = 'Normalized_MAE'):
    """
    Compare multiple models based on a specific metric.
    
    Parameters:
    - model_results: Dictionary mapping model names to their benchmarking results
    - metric: Metric to use for comparison (default: 'Normalized_MAE')
    """
    print("\n" + "="*80)
    print(f"MODEL COMPARISON - Sorted by {metric}")
    print("="*80 + "\n")
    
    # Extract the specified metric for each model
    model_scores = []
    for model_name, results in model_results.items():
        if metric in results['model_metrics']:
            score = results['model_metrics'][metric]
        elif metric in results['improvement_over_baseline']:
            score = results['improvement_over_baseline'][metric]
        else:
            continue
        model_scores.append((model_name, score))
    
    # Sort by score (descending - higher is better for most metrics)
    model_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Print ranking
    print(f"{'Rank':<6} {'Model Name':<30} {metric:<20} {'Status'}")
    print("-" * 80)
    for rank, (model_name, score) in enumerate(model_scores, 1):
        status = "✓ Best" if rank == 1 else "✓ Good" if score > 0.7 else "○ Moderate" if score > 0.5 else "✗ Needs Improvement"
        print(f"{rank:<6} {model_name:<30} {score:<20.6f} {status}")
    
    print("="*80 + "\n")
