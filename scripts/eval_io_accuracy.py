#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate Disk I/O prediction accuracy (WriteIO only).

NOTE:
- ReadIO is often ~0 for most samples in typical datasets, making MAPE unstable
  and misleading. This evaluation intentionally uses **WriteIO only**.
"""

import os
import sys
import json
import joblib
import numpy as np
from autotune.utils.resource_parser import parse_collection_data
from autotune.utils.resource_model_loader import load_or_retrain_resource_models
from scripts.evaluate_resource_model import calculate_mape


def main():
    # Default paths
    model_path = 'resource_models/resource_predictor.joblib'
    data_path = 'resource_data/resource_data.json'
    metadata_path = 'resource_models/training_metadata.json'
    default_knob_config = 'scripts/experiment/gen_knobs/mysql_cpu_io_dynamic_15.json'
    default_knob_num = 15
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Please train the model first using train_resource_model.py")
        sys.exit(1)
    
    # Check if test data exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        print("Please collect data first using collect_resource_data.py")
        sys.exit(1)
    
    # Prefer held-out test MAPE from the last training run (matches train_resource_model.py output).
    # For I/O we report WriteIO-only MAPE.
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            test_metrics = metadata.get('test_metrics', {})
            # Try write_io_mape first (new format), then combined_io_mape (old format)
            if 'write_io_mape' in test_metrics:
                mape = float(test_metrics['write_io_mape'])
            elif 'combined_io_mape' in test_metrics:
                mape = float(test_metrics['combined_io_mape'])
            else:
                raise KeyError("No write_io_mape or combined_io_mape found")
            
            n_samples = int(test_metrics.get('write_io_n_samples', metadata.get('n_test', 0)))
            mean_actual = test_metrics.get('write_io_mean_actual')
            mean_predicted = test_metrics.get('write_io_mean_predicted')
            
            print(f"MAPE: {mape:.2f}%")
            print(f"Test Samples: {n_samples}")
            if mean_actual is not None and mean_predicted is not None:
                print(f"Mean Actual: {mean_actual:.2f} MB/s")
                print(f"Mean Predicted: {mean_predicted:.2f} MB/s")
            
            # Print individual values if available
            actual_values = test_metrics.get('write_io_actual_values')
            predicted_values = test_metrics.get('write_io_predicted_values')
            if actual_values and predicted_values:
                print(f"\nIndividual Test Values:")
                print(f"{'Index':<6} {'Actual':>10} {'Predicted':>10} {'Error%':>10}")
                print("-" * 40)
                for i, (act, pred) in enumerate(zip(actual_values, predicted_values)):
                    err_pct = abs(act - pred) / (act + 1e-9) * 100
                    print(f"{i:<6} {act:>10.2f} {pred:>10.2f} {err_pct:>10.2f}")
            return 0
        except Exception as e:
            print(f"Warning: Failed to read held-out MAPE from {metadata_path}: {e}")

    # Fallback: compute MAPE on the provided data file (may be training-set MAPE).
    # Load models
    try:
        loaded = load_or_retrain_resource_models(model_path)
        model_write_io = loaded["model_write_io"]
        workload_encoder = loaded.get("workload_encoder", {})
        unique_workloads = workload_encoder.get('unique_workloads', [])
        n_workload_features = workload_encoder.get('n_features', 0)
        knob_config = loaded.get("knob_config_file") or default_knob_config
        knob_num = int(loaded.get("knob_num") or default_knob_num)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load test data
    try:
        X, Y_cpu, Y_read_io, Y_write_io, workload_info = parse_collection_data(
            data_path, knob_config, knob_num
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Encode workload features
    workload_features = np.zeros((len(workload_info), n_workload_features))
    for i, info in enumerate(workload_info):
        workload_name = info.get('workload_name', info.get('workload_type', 'sysbench'))
        if len(unique_workloads) > 0:
            if workload_name in unique_workloads:
                idx = unique_workloads.index(workload_name)
                workload_features[i, idx] = 1.0
            else:
                workload_features[i, 0] = 1.0
        
        threads = info.get('threads', 40)
        if n_workload_features > 0:
            workload_features[i, -1] = threads / 100.0
    
    # Combine features
    X_combined = np.hstack([X, workload_features])
    
    # Make predictions
    try:
        write_io_pred, _ = model_write_io.predict(X_combined)
        write_io_pred = write_io_pred.flatten()
    except Exception as e:
        print(f"Error making predictions: {e}")
        sys.exit(1)
    
    # Calculate MAPE for WriteIO only
    mape_write_io = calculate_mape(Y_write_io, write_io_pred)
    
    # Print result
    print(f"MAPE: {mape_write_io:.2f}%")
    print(f"Test Samples: {len(Y_write_io)}")
    print(f"Mean Actual: {float(Y_write_io.mean()):.2f} MB/s")
    print(f"Mean Predicted: {float(write_io_pred.mean()):.2f} MB/s")
    
    # Print individual values
    print(f"\nIndividual Test Values:")
    print(f"{'Index':<6} {'Actual':>10} {'Predicted':>10} {'Error%':>10}")
    print("-" * 40)
    for i, (act, pred) in enumerate(zip(Y_write_io.flatten(), write_io_pred.flatten())):
        err_pct = abs(act - pred) / (act + 1e-9) * 100
        print(f"{i:<6} {act:>10.2f} {pred:>10.2f} {err_pct:>10.2f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

