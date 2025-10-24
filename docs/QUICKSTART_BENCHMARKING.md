# Quick Start Guide: Benchmarking Models

This is a quick reference for running the benchmarking functionality.

## Prerequisites

Ensure you have:
- Trained models saved as `.pth` files
- Test data and scalers in the model's data directory
- Required Python packages installed

## Step-by-Step Guide

### 1. Train Your Model (with Benchmarking)

The training script now automatically computes MAE and Normalized MAE during validation:

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

**What's new**: Console output during training now shows:
```
epoch: 50, validation loss: 0.123, lr: 0.003, r^2: 0.856, MAE: 12.345678, Normalized MAE: 0.782345
```

### 2. Evaluate a Single Model

After training, run comprehensive benchmarking:

```bash
python scripts/evaluation/benchmark_test_model.py \
    --model_path data/TR-C_Benchmarks/trans_conv_5_features/trained_model/model.pth \
    --data_dir data/TR-C_Benchmarks/trans_conv_5_features \
    --gnn_arch trans_conv \
    --model_name "TransConv 5 Features"
```

**Output files** (in `data_dir`):
- `benchmark_results_trans_conv.json`: All metrics in JSON format
- `predictions_trans_conv.pt`: Predictions and targets for further analysis

### 3. Compare Multiple Models

First, benchmark each model individually (Step 2), then compare:

```bash
# Organize results
mkdir -p data/TR-C_Benchmarks/comparison
cp data/TR-C_Benchmarks/*/benchmark_results_*.json data/TR-C_Benchmarks/comparison/

# Run comparison
python scripts/evaluation/compare_models_benchmark.py \
    --results_dir data/TR-C_Benchmarks/comparison \
    --output_dir data/TR-C_Benchmarks/comparison/reports
```

**Output files** (in `output_dir`):
- `model_comparison_table.csv`: Comparison table
- `model_comparison_benchmarks.png`: Visualization plots
- `benchmark_comparison_report.txt`: Detailed text report

## Command Templates

### For Different Architectures

**GAT**:
```bash
python scripts/evaluation/benchmark_test_model.py \
    --model_path data/TR-C_Benchmarks/gat_5_features/trained_model/model.pth \
    --data_dir data/TR-C_Benchmarks/gat_5_features \
    --gnn_arch gat \
    --model_name "GAT"
```

**GCN**:
```bash
python scripts/evaluation/benchmark_test_model.py \
    --model_path data/TR-C_Benchmarks/gcn_5_features/trained_model/model.pth \
    --data_dir data/TR-C_Benchmarks/gcn_5_features \
    --gnn_arch gcn \
    --model_name "GCN"
```

**EIGN** (with unsigned predictions):
```bash
python scripts/evaluation/benchmark_test_model.py \
    --model_path data/TR-C_Benchmarks/eign_5_features/trained_model/model.pth \
    --data_dir data/TR-C_Benchmarks/eign_5_features \
    --gnn_arch eign \
    --model_name "EIGN" \
    --use_signed False
```

**GraphSAGE**:
```bash
python scripts/evaluation/benchmark_test_model.py \
    --model_path data/TR-C_Benchmarks/graphsage_5_features/trained_model/model.pth \
    --data_dir data/TR-C_Benchmarks/graphsage_5_features \
    --gnn_arch graphSAGE \
    --model_name "GraphSAGE"
```

## Understanding the Output

### Key Metrics to Look At

1. **Normalized MAE** (0-1, higher is better)
   - Shows improvement over naive baseline
   - 1.0 = perfect, 0.0 = same as baseline, <0 = worse than baseline

2. **R²** (0-1, higher is better)
   - Shows how much variance the model explains
   - 0 = no better than predicting the mean

3. **Correlations** (0-1, higher is better)
   - Spearman: monotonic relationship
   - Pearson: linear relationship

### Quick Interpretation

| Normalized MAE | Interpretation |
|----------------|----------------|
| > 0.9 | Excellent - much better than baseline |
| 0.7 - 0.9 | Good - significant improvement |
| 0.5 - 0.7 | Moderate - some improvement |
| 0.3 - 0.5 | Weak - minor improvement |
| < 0.3 | Poor - barely better than baseline |
| < 0 | Worse than naive baseline |

## Batch Processing Multiple Models

If you have multiple trained models:

```bash
#!/bin/bash

# Define models to benchmark
models=(
    "trans_conv:trans_conv_5_features:TransConv"
    "gat:gat_5_features:GAT"
    "gcn:gcn_5_features:GCN"
    "eign:eign_5_features:EIGN"
    "graphSAGE:graphsage_5_features:GraphSAGE"
)

# Benchmark each model
for model_info in "${models[@]}"; do
    IFS=':' read -r arch desc name <<< "$model_info"
    echo "Benchmarking $name..."
    
    python scripts/evaluation/benchmark_test_model.py \
        --model_path "data/TR-C_Benchmarks/$desc/trained_model/model.pth" \
        --data_dir "data/TR-C_Benchmarks/$desc" \
        --gnn_arch "$arch" \
        --model_name "$name"
done

# Compare all models
echo "Comparing all models..."
mkdir -p data/TR-C_Benchmarks/comparison
cp data/TR-C_Benchmarks/*/benchmark_results_*.json data/TR-C_Benchmarks/comparison/

python scripts/evaluation/compare_models_benchmark.py \
    --results_dir data/TR-C_Benchmarks/comparison \
    --output_dir data/TR-C_Benchmarks/comparison/reports

echo "Done! Check data/TR-C_Benchmarks/comparison/reports for results."
```

Save this as `run_benchmarking.sh` and execute:
```bash
chmod +x run_benchmarking.sh
./run_benchmarking.sh
```

## Viewing Results

### Console Output
Results are printed to console with colored indicators and interpretations.

### WandB Dashboard
During training, view real-time metrics including:
- `mae`: Mean Absolute Error
- `normalized_mae`: Improvement over baseline
- `r^2`, `spearman`, `pearson`: Standard metrics

### Files
- **JSON**: Machine-readable, can be loaded for custom analysis
- **CSV**: Open in Excel/Google Sheets
- **PNG**: Publication-ready plots
- **TXT**: Human-readable detailed report

## Common Use Cases

### Use Case 1: Evaluate One Model
```bash
python scripts/evaluation/benchmark_test_model.py \
    --model_path <path_to_model.pth> \
    --data_dir <data_directory> \
    --gnn_arch <architecture>
```

### Use Case 2: Compare All Trained Models
```bash
python scripts/evaluation/compare_models_benchmark.py \
    --results_dir <directory_with_json_files>
```

### Use Case 3: Monitor Training Progress
Training automatically logs MAE metrics to WandB and console - no extra steps needed!

## Troubleshooting

**Error: "No module named 'benchmark_metrics'"**
```bash
# Make sure you're in the project root directory
cd /Users/trimplingroup/Desktop/Benchmarking
```

**Error: "File not found: test_dl.pt"**
```bash
# The model needs to be trained with the updated scripts
# Re-train the model or check the data_dir path
```

**Error: "Model architecture mismatch"**
```bash
# Make sure --gnn_arch matches the architecture used during training
# Check the unique_model_description used during training
```

## Getting Help

For detailed documentation, see:
- `docs/benchmarking.md`: Comprehensive guide
- `scripts/evaluation/benchmark_metrics.py`: Code documentation

## Summary

**What's implemented:**
✅ Naive baseline comparison (mean-based predictor)
✅ Normalized MAE metric (Equation 16 from paper)
✅ R², Spearman, Pearson with baseline context
✅ Automatic integration with training loop
✅ Comprehensive evaluation and comparison scripts
✅ WandB logging
✅ Publication-ready visualizations

**To use it:**
1. Train models (automatically includes benchmarking)
2. Run `benchmark_test_model.py` for each model
3. Run `compare_models_benchmark.py` to compare all models

That's it! 🎉
