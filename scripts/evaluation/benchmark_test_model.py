"""
Comprehensive Model Benchmarking Script

This script evaluates trained models using the naive baseline benchmarking approach.
It computes:
- R² score (baseline: 0)
- Spearman correlation (baseline: 0)
- Pearson correlation (baseline: 0)
- MAE and Normalized MAE (relative to naive mean-based predictor)

Usage:
    python benchmark_test_model.py --model_path <path_to_model> --gnn_arch <architecture>
"""

import os
import sys
import json
import argparse
import joblib

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from training.help_functions import create_gnn_model, str_to_bool
from gnn.gnn_io import collate_fn
from gnn.help_functions import (
    compute_r2_torch,
    compute_spearman_pearson,
    compute_mae_torch,
    compute_normalized_mae_torch
)
from evaluation.benchmark_metrics import (
    compute_benchmarking_metrics,
    print_benchmarking_results,
    compare_models
)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def load_model_and_data(model_path, data_dir, gnn_arch, device):
    """
    Load trained model and test data.
    
    Parameters:
    - model_path: Path to the trained model
    - data_dir: Directory containing the test data and scalers
    - gnn_arch: GNN architecture name
    - device: Device to load model on
    
    Returns:
    - model: Loaded model
    - test_loader: Test data loader
    - scalers: Test scalers
    """
    # Load test data
    test_loader_path = os.path.join(data_dir, 'data_created_during_training', 'test_dl.pt')
    test_loader = torch.load(test_loader_path, map_location='cpu')
    
    # Load scalers
    x_scaler_path = os.path.join(data_dir, 'data_created_during_training', 'test_x_scaler.pkl')
    x_scaler = joblib.load(x_scaler_path)
    
    is_eign = (gnn_arch == "eign")
    
    if is_eign:
        x_signed_scaler_path = os.path.join(data_dir, 'data_created_during_training', 'test_x_signed_scaler.pkl')
        x_signed_scaler = joblib.load(x_signed_scaler_path)
        scalers = {
            'x_scaler': x_scaler,
            'x_signed_scaler': x_signed_scaler
        }
    else:
        pos_scaler_path = os.path.join(data_dir, 'data_created_during_training', 'test_pos_scaler.pkl')
        pos_scaler = joblib.load(pos_scaler_path)
        scalers = {
            'x_scaler': x_scaler,
            'pos_scaler': pos_scaler
        }
    
    # Determine in_channels from test data
    sample_data = next(iter(test_loader))
    in_channels = sample_data.x.shape[1]
    
    # Create model configuration (minimal config for inference)
    class Config:
        def __init__(self):
            self.gnn_arch = gnn_arch
            self.in_channels = in_channels
            self.out_channels = 1
            self.use_dropout = False
            self.dropout = 0.0
            self.predict_mode_stats = False
    
    config = Config()
    
    # Create model instance
    model = create_gnn_model(
        gnn_arch=gnn_arch,
        config=config,
        model_kwargs={},
        device=device
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from {model_path}")
    print(f"✓ Architecture: {gnn_arch}")
    print(f"✓ Input channels: {in_channels}")
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    
    return model, test_loader, scalers, is_eign


def evaluate_model(model, test_loader, scalers, device, is_eign=False, use_signed=False):
    """
    Evaluate model on test data and compute all metrics.
    
    Parameters:
    - model: Trained model
    - test_loader: Test data loader
    - scalers: Scalers for denormalization
    - device: Device for computation
    - is_eign: Whether using EIGN architecture
    - use_signed: Whether to use signed predictions (for EIGN)
    
    Returns:
    - predictions: All predictions
    - targets: All ground truth targets
    - metrics: Dictionary of computed metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.inference_mode():
        for data in test_loader:
            data = data.to(device)
            
            if is_eign:
                eign_output = model(
                    x_unsigned=data.x if hasattr(data, "x") and data.x is not None else None,
                    x_signed=data.x_signed if hasattr(data, "x_signed") and data.x_signed is not None else None,
                    edge_index=data.edge_index,
                    is_directed=data.edge_is_directed,
                )
                
                if use_signed:
                    predictions = eign_output.signed
                    targets = data.y_signed
                else:
                    predictions = eign_output.unsigned
                    targets = data.y
            else:
                predictions = model(data)
                targets = data.y
            
            all_predictions.append(predictions)
            all_targets.append(targets)
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    # Compute metrics
    r2 = compute_r2_torch(all_predictions, all_targets).item()
    spearman, pearson = compute_spearman_pearson(all_predictions, all_targets)
    mae = compute_mae_torch(all_predictions, all_targets)
    normalized_mae = compute_normalized_mae_torch(all_predictions, all_targets)
    
    # Compute comprehensive benchmarking metrics
    benchmarking_results = compute_benchmarking_metrics(
        predictions=all_predictions,
        targets=all_targets,
        r2=r2,
        spearman_corr=spearman,
        pearson_corr=pearson
    )
    
    return all_predictions, all_targets, benchmarking_results


def save_results(results, output_path):
    """
    Save benchmarking results to JSON file.
    
    Parameters:
    - results: Benchmarking results dictionary
    - output_path: Path to save JSON file
    """
    # Convert numpy/torch types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    results_native = convert_to_native(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_native, f, indent=4)
    
    print(f"\n✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark trained GNN model with naive baselines")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model file (.pth)")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing the test data and scalers")
    parser.add_argument("--gnn_arch", type=str, required=True,
                       choices=["point_net_transf_gat", "gat", "gcn", "gcn2", "trans_conv", 
                               "pnc", "fc_nn", "graphSAGE", "eign", "xgboost", "trans_encoder"],
                       help="GNN architecture of the trained model")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Descriptive name for the model (for display)")
    parser.add_argument("--use_signed", type=str_to_bool, default=False,
                       help="Use signed predictions for EIGN (default: False)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save results (default: same as data_dir)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}\n")
    
    # Set model name
    model_name = args.model_name if args.model_name else f"{args.gnn_arch}_model"
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.data_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and data
    print("Loading model and test data...")
    model, test_loader, scalers, is_eign = load_model_and_data(
        args.model_path,
        args.data_dir,
        args.gnn_arch,
        device
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    predictions, targets, benchmarking_results = evaluate_model(
        model,
        test_loader,
        scalers,
        device,
        is_eign=is_eign,
        use_signed=args.use_signed
    )
    
    # Print results
    print_benchmarking_results(benchmarking_results, model_name=model_name)
    
    # Save results
    results_file = os.path.join(output_dir, f'benchmark_results_{args.gnn_arch}.json')
    save_results(benchmarking_results, results_file)
    
    # Also save predictions and targets for further analysis
    predictions_file = os.path.join(output_dir, f'predictions_{args.gnn_arch}.pt')
    torch.save({
        'predictions': predictions.cpu(),
        'targets': targets.cpu(),
    }, predictions_file)
    print(f"✓ Predictions saved to {predictions_file}")


if __name__ == '__main__':
    main()
