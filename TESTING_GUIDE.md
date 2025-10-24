# Step-by-Step Guide: Running and Testing the Benchmarking Implementation

This guide provides complete instructions for running and testing the newly implemented benchmarking functionality.

## Prerequisites

✅ All code changes have been applied
✅ You're in the project directory: `/Users/trimplingroup/Desktop/Benchmarking`
✅ Python environment is activated

## Part 1: Test with Existing Trained Models (If Available)

### Step 1.1: Locate Your Trained Models

Check what models you have:
```bash
ls -la data/TR-C_Benchmarks/*/trained_model/model.pth
```

### Step 1.2: Benchmark an Existing Model

Pick one model and run benchmarking (replace paths as needed):

```bash
python scripts/evaluation/benchmark_test_model.py \
    --model_path data/TR-C_Benchmarks/trans_conv_5_features/trained_model/model.pth \
    --data_dir data/TR-C_Benchmarks/trans_conv_5_features \
    --gnn_arch trans_conv \
    --model_name "TransConv 5 Features"
```

**Expected Output**:
- Console output with formatted metrics
- File created: `data/TR-C_Benchmarks/trans_conv_5_features/benchmark_results_trans_conv.json`
- File created: `data/TR-C_Benchmarks/trans_conv_5_features/predictions_trans_conv.pt`

### Step 1.3: Verify the Results

Check the JSON file:
```bash
cat data/TR-C_Benchmarks/trans_conv_5_features/benchmark_results_trans_conv.json
```

You should see:
```json
{
    "model_metrics": {
        "MAE": ...,
        "MSE": ...,
        "R2": ...,
        "Spearman_Correlation": ...,
        "Pearson_Correlation": ...,
        "Normalized_MAE": ...
    },
    "baseline_metrics": {
        "Naive_MAE": ...,
        "Naive_R2": 0.0,
        "Naive_Spearman": 0.0,
        "Naive_Pearson": 0.0
    },
    "improvement_over_baseline": {
        "R2_Improvement": ...,
        "Spearman_Improvement": ...,
        "Pearson_Improvement": ...,
        "Normalized_MAE": ...
    }
}
```

## Part 2: Train a New Model with Benchmarking

### Step 2.1: Train a Quick Test Model

Train a small model to test the integration:

```bash
python scripts/training/run_models.py \
    --gnn_arch trans_conv \
    --project_name "benchmark_test" \
    --unique_model_description "test_model" \
    --in_channels 5 \
    --use_all_features False \
    --num_epochs 20 \
    --lr 0.003 \
    --batch_size 8 \
    --early_stopping_patience 10
```

**What to Watch For**:
During training, you should see output like:
```
epoch: 0, validation loss: X.XXX, lr: 0.003, r^2: X.XXX, MAE: XX.XXXXXX, Normalized MAE: X.XXXXXX
epoch: 1, validation loss: X.XXX, lr: 0.003, r^2: X.XXX, MAE: XX.XXXXXX, Normalized MAE: X.XXXXXX
...
```

The MAE and Normalized MAE values are the NEW metrics being tracked!

### Step 2.2: Verify WandB Logging

Check your WandB dashboard. You should see new metrics:
- `mae`: Mean Absolute Error
- `normalized_mae`: Normalized MAE (improvement over baseline)

These appear alongside the existing `r^2`, `spearman`, `pearson` metrics.

### Step 2.3: Benchmark the Newly Trained Model

```bash
python scripts/evaluation/benchmark_test_model.py \
    --model_path data/benchmark_test/test_model/trained_model/model.pth \
    --data_dir data/benchmark_test/test_model \
    --gnn_arch trans_conv \
    --model_name "Test Model"
```

## Part 3: Compare Multiple Models

### Step 3.1: Train Multiple Models (Optional)

If you want to test comparison with fresh models:

```bash
# Model 1: TransConv
python scripts/training/run_models.py \
    --gnn_arch trans_conv \
    --project_name "comparison_test" \
    --unique_model_description "trans_conv" \
    --num_epochs 20

# Model 2: GAT
python scripts/training/run_models.py \
    --gnn_arch gat \
    --project_name "comparison_test" \
    --unique_model_description "gat" \
    --num_epochs 20

# Model 3: GCN
python scripts/training/run_models.py \
    --gnn_arch gcn \
    --project_name "comparison_test" \
    --unique_model_description "gcn" \
    --num_epochs 20
```

### Step 3.2: Benchmark Each Model

```bash
# Benchmark TransConv
python scripts/evaluation/benchmark_test_model.py \
    --model_path data/comparison_test/trans_conv/trained_model/model.pth \
    --data_dir data/comparison_test/trans_conv \
    --gnn_arch trans_conv \
    --model_name "TransConv"

# Benchmark GAT
python scripts/evaluation/benchmark_test_model.py \
    --model_path data/comparison_test/gat/trained_model/model.pth \
    --data_dir data/comparison_test/gat \
    --gnn_arch gat \
    --model_name "GAT"

# Benchmark GCN
python scripts/evaluation/benchmark_test_model.py \
    --model_path data/comparison_test/gcn/trained_model/model.pth \
    --data_dir data/comparison_test/gcn \
    --gnn_arch gcn \
    --model_name "GCN"
```

### Step 3.3: Organize Results for Comparison

```bash
# Create comparison directory
mkdir -p data/comparison_test/comparison

# Copy all benchmark results
cp data/comparison_test/*/benchmark_results_*.json data/comparison_test/comparison/
```

### Step 3.4: Run Comparison

```bash
python scripts/evaluation/compare_models_benchmark.py \
    --results_dir data/comparison_test/comparison \
    --output_dir data/comparison_test/comparison/reports
```

**Expected Outputs**:
1. Console output showing:
   - Comparison table
   - Model rankings
   
2. Files created in `data/comparison_test/comparison/reports/`:
   - `model_comparison_table.csv`
   - `model_comparison_benchmarks.png`
   - `benchmark_comparison_report.txt`

### Step 3.5: View the Results

```bash
# View the comparison table
cat data/comparison_test/comparison/reports/model_comparison_table.csv

# View the detailed report
cat data/comparison_test/comparison/reports/benchmark_comparison_report.txt

# Open the visualization (macOS)
open data/comparison_test/comparison/reports/model_comparison_benchmarks.png
```

## Part 4: Verification Checklist

Use this checklist to ensure everything is working:

### Training Integration
- [ ] Console output shows MAE and Normalized MAE during training
- [ ] WandB logs include `mae` and `normalized_mae` metrics
- [ ] Training completes without errors

### Single Model Benchmarking
- [ ] Script runs without errors
- [ ] Console shows formatted benchmarking results
- [ ] JSON file is created with all metrics
- [ ] Predictions file (.pt) is created
- [ ] All baselines show 0 for R², Spearman, Pearson
- [ ] Normalized MAE is between 0 and 1 (typically)

### Model Comparison
- [ ] Script loads all JSON files
- [ ] Console shows comparison table
- [ ] CSV file is created
- [ ] PNG plot is created with 6 subplots
- [ ] Text report is created
- [ ] Models are ranked correctly

## Part 5: Test Different Architectures

Test with different GNN architectures:

### GAT
```bash
python scripts/evaluation/benchmark_test_model.py \
    --model_path <path_to_gat_model> \
    --data_dir <gat_data_dir> \
    --gnn_arch gat \
    --model_name "GAT"
```

### EIGN (Special case)
```bash
python scripts/evaluation/benchmark_test_model.py \
    --model_path <path_to_eign_model> \
    --data_dir <eign_data_dir> \
    --gnn_arch eign \
    --model_name "EIGN" \
    --use_signed False
```

### GraphSAGE
```bash
python scripts/evaluation/benchmark_test_model.py \
    --model_path <path_to_graphsage_model> \
    --data_dir <graphsage_data_dir> \
    --gnn_arch graphSAGE \
    --model_name "GraphSAGE"
```

## Part 6: Interpreting Results

### Good Performance Indicators
✅ **R² > 0.7**: Model explains >70% of variance (much better than mean)
✅ **Normalized MAE > 0.7**: Model reduces error by >70% vs. naive baseline
✅ **Correlations > 0.7**: Strong relationship with ground truth

### Warning Signs
⚠️ **R² < 0.3**: Model barely better than predicting the mean
⚠️ **Normalized MAE < 0.3**: Model only slightly better than naive baseline
⚠️ **Any metric < 0**: Model is worse than naive baseline!

### Understanding Normalized MAE
- **1.0**: Perfect predictions (zero error)
- **0.8**: 80% improvement over naive baseline
- **0.5**: 50% improvement over naive baseline
- **0.0**: Same performance as naive baseline
- **Negative**: Worse than naive baseline

## Part 7: Troubleshooting

### Error: "compute_mae_torch is not defined"
**Cause**: Old version of `gnn/help_functions.py`
**Solution**: Ensure all file modifications are applied correctly

### Error: "File not found: test_dl.pt"
**Cause**: Model trained before updates, or wrong data_dir
**Solution**: 
- Check data_dir path
- Or retrain the model with updated scripts

### Error: "No benchmark result files found"
**Cause**: Wrong directory or files not created
**Solution**:
- Check results_dir path
- Ensure benchmark_test_model.py ran successfully first
- Look for `benchmark_results_*.json` files

### Values Look Wrong
**Check**:
- Are you using the correct architecture name?
- Is the model path correct?
- Was the model trained successfully?

### WandB Not Showing New Metrics
**Check**:
- Is WandB initialized in your training run?
- Are you looking at a new run (after the updates)?
- Check the WandB project name matches your `--project_name`

## Part 8: Production Use

### For Real Experiments

1. **Train your production models normally**:
```bash
python scripts/training/run_models.py \
    --gnn_arch <your_arch> \
    --project_name "TR-C_Benchmarks" \
    --unique_model_description <your_description> \
    --in_channels 5 \
    --use_all_features False \
    --num_epochs 500 \
    --lr 0.003 \
    --early_stopping_patience 25 \
    --use_dropout True \
    --dropout 0.3
```

2. **After training, benchmark each model**:
```bash
python scripts/evaluation/benchmark_test_model.py \
    --model_path data/TR-C_Benchmarks/<description>/trained_model/model.pth \
    --data_dir data/TR-C_Benchmarks/<description> \
    --gnn_arch <your_arch> \
    --model_name "<Display Name>"
```

3. **When you have multiple models, compare them**:
```bash
# Organize results
mkdir -p data/TR-C_Benchmarks/final_comparison
cp data/TR-C_Benchmarks/*/benchmark_results_*.json data/TR-C_Benchmarks/final_comparison/

# Compare
python scripts/evaluation/compare_models_benchmark.py \
    --results_dir data/TR-C_Benchmarks/final_comparison \
    --output_dir data/TR-C_Benchmarks/final_comparison/reports
```

## Part 9: Success Indicators

You'll know everything is working when:

✅ Training shows MAE metrics in console
✅ WandB dashboard shows mae and normalized_mae graphs
✅ benchmark_test_model.py produces formatted output
✅ JSON files contain all expected metrics
✅ Comparison script produces plots and reports
✅ All metrics make sense (R² between 0-1, etc.)

## Summary

**What You've Implemented**:
1. Naive baseline benchmarking (Section 5.2 of paper)
2. Normalized MAE metric (Equation 16)
3. Automatic integration with training
4. Comprehensive evaluation scripts
5. Model comparison tools
6. Publication-ready visualizations

**How to Use**:
1. Train models → Benchmarking automatic
2. Run `benchmark_test_model.py` → Get comprehensive metrics
3. Run `compare_models_benchmark.py` → Compare all models

**Output**:
- Console: Formatted results with interpretations
- JSON: Machine-readable metrics
- CSV: Comparison tables
- PNG: Visualizations
- TXT: Detailed reports
- WandB: Real-time tracking

## Next Steps

1. ✅ Test with one existing model
2. ✅ Train one new model and verify MAE logging
3. ✅ Benchmark the new model
4. ✅ Train 2-3 models and compare them
5. ✅ Use for real experiments

**Ready to use for your research! 🎉**

## Questions?

Refer to:
- `docs/benchmarking.md` - Comprehensive documentation
- `docs/QUICKSTART_BENCHMARKING.md` - Quick reference
- `BENCHMARKING_IMPLEMENTATION_SUMMARY.md` - Implementation summary
