# Model Benchmarking with Naive Baselines

This document describes the implementation of naive baseline benchmarking for evaluating model performance in traffic prediction.

## Overview

Since no established baseline exists for this traffic prediction problem, we introduce a **simple mean-based baseline** to assess model performance. This approach is described in **Section 5.2** of the research paper.

## Benchmarking Approach

### Naive Baseline Definition

The naive baseline assumes that the **predicted change in traffic volume for each edge is equal to the average observed change in traffic volume** across all edges in the test set.

### Metrics and Baselines

1. **R² (Coefficient of Determination)**
   - **Naive Baseline**: 0
   - **Reasoning**: A mean-based predictor does not explain any variance beyond predicting the average
   - **Interpretation**: R² > 0 indicates the model explains variance better than the mean

2. **Spearman Correlation**
   - **Naive Baseline**: 0
   - **Reasoning**: No meaningful monotonic relationship with true values
   - **Interpretation**: Positive correlation indicates monotonic relationship with ground truth

3. **Pearson Correlation**
   - **Naive Baseline**: 0
   - **Reasoning**: No meaningful linear relationship with true values
   - **Interpretation**: Positive correlation indicates linear relationship with ground truth

4. **Normalized MAE**
   - **Formula**: `1 - (MAE / Naive_MAE)`
   - **Naive_MAE**: MAE when predicting the mean for all samples
   - **Range**: 0 to 1 (can be negative if model performs worse than baseline)
   - **Interpretation**:
     - 1.0 = Perfect predictions (zero error)
     - 0.0 = No improvement over naive predictor
     - < 0 = Worse than naive predictor

### Why Not Normalized MSE?

Applying the same normalization to MSE would result in a formulation that is **mathematically identical to R²** (see Equation 15 in the paper). Therefore, we do not report a normalized MSE separately.

## Implementation

### Core Module: `benchmark_metrics.py`

Located at: `scripts/evaluation/benchmark_metrics.py`

This module provides:
- `compute_mae()`: Calculate Mean Absolute Error
- `compute_mse()`: Calculate Mean Squared Error
- `compute_naive_baseline_mae()`: Calculate naive MAE baseline
- `compute_normalized_mae()`: Calculate normalized MAE
- `compute_benchmarking_metrics()`: Comprehensive benchmarking with all metrics
- `print_benchmarking_results()`: Pretty-print results with interpretations
- `compare_models()`: Compare multiple models

### Integration with Training

The benchmarking metrics are integrated into the training loop:

1. **Modified Files**:
   - `scripts/gnn/help_functions.py`: Added MAE and normalized MAE computation
   - `scripts/gnn/models/base_gnn.py`: Updated validation to include new metrics
   - `scripts/gnn/models/eign.py`: Updated EIGN validation

2. **New Metrics Logged**:
   - MAE (Mean Absolute Error)
   - Normalized MAE (improvement over baseline)
   - All metrics logged to WandB during training

### Evaluation Scripts

#### 1. Single Model Benchmarking

**Script**: `scripts/evaluation/benchmark_test_model.py`

Evaluate a single trained model with comprehensive benchmarking.

**Usage**:
```bash
python scripts/evaluation/benchmark_test_model.py \
    --model_path data/TR-C_Benchmarks/trans_conv_5_features/trained_model/model.pth \
    --data_dir data/TR-C_Benchmarks/trans_conv_5_features \
    --gnn_arch trans_conv \
    --model_name "TransConv_5Features"
```

**Arguments**:
- `--model_path`: Path to trained model (.pth file)
- `--data_dir`: Directory with test data and scalers
- `--gnn_arch`: Model architecture (trans_conv, gat, gcn, eign, etc.)
- `--model_name`: Display name for the model (optional)
- `--use_signed`: Use signed predictions for EIGN (default: False)
- `--output_dir`: Where to save results (default: data_dir)
- `--device`: cuda or cpu (default: auto-detect)

**Output**:
- Formatted console output with all metrics
- JSON file: `benchmark_results_{arch}.json`
- PyTorch file: `predictions_{arch}.pt`

#### 2. Multi-Model Comparison

**Script**: `scripts/evaluation/compare_models_benchmark.py`

Compare multiple models using benchmarking metrics.

**Usage**:
```bash
python scripts/evaluation/compare_models_benchmark.py \
    --results_dir data/TR-C_Benchmarks/comparison \
    --output_dir data/TR-C_Benchmarks/comparison/reports
```

**Arguments**:
- `--results_dir`: Directory containing `benchmark_results_*.json` files
- `--pattern`: Glob pattern for result files (default: "benchmark_results_*.json")
- `--output_dir`: Where to save comparison outputs (default: results_dir)

**Output**:
- Comparison table (CSV): `model_comparison_table.csv`
- Visualization plots (PNG): `model_comparison_benchmarks.png`
- Text report: `benchmark_comparison_report.txt`

## Complete Workflow

### Step 1: Train Models

Train your models as usual:

```bash
python scripts/training/run_models.py \
    --gnn_arch trans_conv \
    --project_name "TR-C_Benchmarks" \
    --unique_model_description "trans_conv_5_features" \
    --in_channels 5 \
    --use_all_features False \
    --num_epochs 500 \
    --lr 0.003 \
    --early_stopping_patience 25 \
    --use_dropout True \
    --dropout 0.3
```

**New During Training**:
- MAE and Normalized MAE are automatically computed during validation
- All metrics logged to WandB
- Console output includes MAE metrics

### Step 2: Benchmark Individual Models

After training, evaluate each model:

```bash
# Example 1: TransConv
python scripts/evaluation/benchmark_test_model.py \
    --model_path data/TR-C_Benchmarks/trans_conv_5_features/trained_model/model.pth \
    --data_dir data/TR-C_Benchmarks/trans_conv_5_features \
    --gnn_arch trans_conv \
    --model_name "TransConv"

# Example 2: GAT
python scripts/evaluation/benchmark_test_model.py \
    --model_path data/TR-C_Benchmarks/gat_5_features/trained_model/model.pth \
    --data_dir data/TR-C_Benchmarks/gat_5_features \
    --gnn_arch gat \
    --model_name "GAT"

# Example 3: EIGN
python scripts/evaluation/benchmark_test_model.py \
    --model_path data/TR-C_Benchmarks/eign_5_features/trained_model/model.pth \
    --data_dir data/TR-C_Benchmarks/eign_5_features \
    --gnn_arch eign \
    --model_name "EIGN"
```

### Step 3: Compare All Models

Collect all benchmark result JSON files in one directory, then compare:

```bash
# Create comparison directory
mkdir -p data/TR-C_Benchmarks/comparison

# Copy all benchmark results
cp data/TR-C_Benchmarks/*/benchmark_results_*.json data/TR-C_Benchmarks/comparison/

# Run comparison
python scripts/evaluation/compare_models_benchmark.py \
    --results_dir data/TR-C_Benchmarks/comparison \
    --output_dir data/TR-C_Benchmarks/comparison/reports
```

## Example Output

### Console Output (Single Model)

```
================================================================================
BENCHMARKING RESULTS: TransConv_5Features
================================================================================

📊 MODEL PERFORMANCE METRICS:
--------------------------------------------------------------------------------
  R2............................... 0.856432
  Spearman_Correlation............. 0.924567
  Pearson_Correlation.............. 0.931234
  MAE.............................. 12.345678
  Normalized_MAE................... 0.782345
  MSE.............................. 234.567890

📏 NAIVE BASELINE METRICS:
--------------------------------------------------------------------------------
  Naive_MAE........................ 56.789012
  Naive_R2......................... 0.000000
  Naive_Spearman................... 0.000000
  Naive_Pearson.................... 0.000000

📈 IMPROVEMENT OVER NAIVE BASELINE:
--------------------------------------------------------------------------------
  ✓ R2_Improvement................. 0.856432
  ✓ Spearman_Improvement........... 0.924567
  ✓ Pearson_Improvement............ 0.931234
  ✓ Normalized_MAE................. 0.782345

💡 INTERPRETATION:
--------------------------------------------------------------------------------
  R² Score: Good (explains 85.64% of variance)
  Normalized MAE: Good (78.23% improvement over naive baseline)
  Correlations: Very Strong relationship with ground truth
================================================================================
```

### Comparison Output

The comparison script generates:

1. **CSV Table**: Sortable comparison of all models
2. **PNG Plots**: 
   - R² comparison
   - Correlation comparison (Spearman & Pearson)
   - Normalized MAE comparison
   - MAE and MSE comparison
   - Overall performance ranking
3. **Text Report**: Detailed ranking and analysis

## Integration with WandB

All benchmarking metrics are automatically logged to Weights & Biases:

- **During Training**: `mae` and `normalized_mae` logged every epoch
- **Tracked Metrics**:
  - `val_loss`: Validation loss
  - `r^2`: R² score
  - `spearman`: Spearman correlation
  - `pearson`: Pearson correlation
  - `mae`: Mean Absolute Error
  - `normalized_mae`: Normalized MAE (improvement metric)

## Interpreting Results

### Quality Guidelines

| Metric | Excellent | Good | Moderate | Poor |
|--------|-----------|------|----------|------|
| R² | > 0.9 | > 0.7 | > 0.5 | > 0 |
| Normalized MAE | > 0.9 | > 0.7 | > 0.5 | > 0 |
| Correlations | > 0.9 | > 0.7 | > 0.5 | > 0.3 |

### Key Questions

1. **Is the model better than the naive baseline?**
   - Check if R², correlations, and Normalized MAE > 0

2. **How much better?**
   - Normalized MAE directly shows percentage improvement
   - R² shows variance explained beyond the mean

3. **Which model is best?**
   - Compare Normalized MAE across models
   - Consider all metrics together for overall assessment

## Files Modified/Created

### Created Files:
- `scripts/evaluation/benchmark_metrics.py`: Core benchmarking module
- `scripts/evaluation/benchmark_test_model.py`: Single model evaluation
- `scripts/evaluation/compare_models_benchmark.py`: Multi-model comparison
- `docs/benchmarking.md`: This documentation

### Modified Files:
- `scripts/gnn/help_functions.py`: Added MAE functions, updated validation
- `scripts/gnn/models/base_gnn.py`: Integrated MAE in training loop
- `scripts/gnn/models/eign.py`: Integrated MAE for EIGN

## Testing the Implementation

### Quick Test

```bash
# 1. Train a small test model
python scripts/training/run_models.py \
    --gnn_arch trans_conv \
    --project_name "test_benchmark" \
    --unique_model_description "test_run" \
    --num_epochs 10

# 2. Benchmark the model
python scripts/evaluation/benchmark_test_model.py \
    --model_path data/test_benchmark/test_run/trained_model/model.pth \
    --data_dir data/test_benchmark/test_run \
    --gnn_arch trans_conv \
    --model_name "Test Model"
```

## References

This implementation follows the approach described in:
- **Section 5.2**: Benchmarking Model Performance with Naive Baselines
- **Equation 16**: Normalized MAE formula

## Troubleshooting

### Common Issues

1. **"compute_mae_torch not defined" error**
   - Make sure you have the latest version of `scripts/gnn/help_functions.py`

2. **"Missing test_dl.pt" error**
   - Ensure the model was trained with the updated training script
   - Re-run data preprocessing if necessary

3. **Import errors**
   - Check that all paths in `sys.path.append()` are correct
   - Verify you're running from the project root

## Contact

For questions or issues with the benchmarking implementation, please refer to the main project documentation or contact the development team.
