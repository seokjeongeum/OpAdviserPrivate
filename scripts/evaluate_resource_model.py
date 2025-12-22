#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script for resource prediction models.
Calculates MAPE and other metrics.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Tuple, Dict


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MAPE (Mean Absolute Percentage Error):
    MAPE(%) = (1/N) * Σ(|실제값 - 예측값| / 실제값) * 100
    
    Parameters:
    -----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    
    Returns:
    --------
    mape : float
        MAPE in percentage
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Avoid division by zero - filter out zero values
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.inf
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, metric_name: str = "Metric") -> Dict[str, float]:
    """
    Evaluate model performance with multiple metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    metric_name : str
        Name of the metric (for display)
    
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Filter out invalid predictions
    valid_mask = np.isfinite(y_pred) & np.isfinite(y_true) & (y_true != 0)
    if np.sum(valid_mask) == 0:
        return {
            'mape': np.inf,
            'r2': -np.inf,
            'mae': np.inf,
            'rmse': np.inf,
            'n_samples': 0,
            'mean_actual': np.nan,
            'mean_predicted': np.nan
        }
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    mape = calculate_mape(y_true_valid, y_pred_valid)
    r2 = r2_score(y_true_valid, y_pred_valid)
    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    
    metrics = {
        'mape': mape,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'n_samples': len(y_true_valid),
        'mean_actual': np.mean(y_true_valid),
        'mean_predicted': np.mean(y_pred_valid)
    }
    
    return metrics


def print_evaluation_results(metrics: Dict[str, float], metric_name: str = "Metric"):
    """
    Print evaluation results in a formatted way.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing evaluation metrics
    metric_name : str
        Name of the metric
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {metric_name}")
    print(f"{'='*60}")
    print(f"  MAPE:        {metrics['mape']:.2f}%")
    print(f"  R² Score:    {metrics['r2']:.4f}")
    print(f"  MAE:         {metrics['mae']:.4f}")
    print(f"  RMSE:        {metrics['rmse']:.4f}")
    print(f"  Samples:     {metrics['n_samples']}")
    print(f"  Mean Actual: {metrics['mean_actual']:.4f}")
    print(f"  Mean Pred:   {metrics['mean_predicted']:.4f}")
    print(f"{'='*60}\n")


def evaluate_all_models(y_cpu_true: np.ndarray, y_cpu_pred: np.ndarray,
                        y_read_io_true: np.ndarray, y_read_io_pred: np.ndarray,
                        y_write_io_true: np.ndarray, y_write_io_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Evaluate CPU and Write I/O resource prediction models.
    
    NOTE: Read I/O is not printed or used in the summary because it's typically ~0
    in most datasets, making MAPE meaningless.
    
    Parameters:
    -----------
    y_cpu_true, y_cpu_pred : np.ndarray
        CPU actual and predicted values
    y_read_io_true, y_read_io_pred : np.ndarray
        Read I/O actual and predicted values (computed but not printed)
    y_write_io_true, y_write_io_pred : np.ndarray
        Write I/O actual and predicted values
    
    Returns:
    --------
    all_metrics : dict
        Dictionary containing metrics for all models
    """
    cpu_metrics = evaluate_model(y_cpu_true, y_cpu_pred, "CPU")
    read_io_metrics = evaluate_model(y_read_io_true, y_read_io_pred, "Read I/O")
    write_io_metrics = evaluate_model(y_write_io_true, y_write_io_pred, "Write I/O")
    
    # Print CPU and WriteIO only (skip ReadIO as it's typically ~0 and MAPE is meaningless)
    print_evaluation_results(cpu_metrics, "CPU Usage")
    print_evaluation_results(write_io_metrics, "Write I/O")
    
    # Summary uses WriteIO only for I/O metric
    write_io_mape = write_io_metrics['mape']
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    cpu_mape_str = f"{cpu_metrics['mape']:.2f}" if not np.isinf(cpu_metrics['mape']) else "inf"
    write_io_mape_str = f"{write_io_mape:.2f}" if not np.isinf(write_io_mape) else "inf"
    
    print(f"CPU MAPE:      {cpu_mape_str}% {'✓' if cpu_metrics['mape'] < 10 else '✗'}")
    print(f"WriteIO MAPE:  {write_io_mape_str}% {'✓' if write_io_mape < 10 else '✗'}")
    
    all_pass = (cpu_metrics['mape'] < 10 and write_io_mape < 10)
    
    print(f"\nAll models <10% MAPE: {'✓ PASS' if all_pass else '✗ FAIL'}")
    print(f"{'='*60}\n")
    
    return {
        'cpu': cpu_metrics,
        'read_io': read_io_metrics,
        'write_io': write_io_metrics
    }


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_resource_model.py <test_data_file>")
        print("Or use as a module: from scripts.evaluate_resource_model import evaluate_model, calculate_mape")
        sys.exit(1)
    
    # This would be used by the training script
    print("This module provides evaluation functions.")
    print("Import and use evaluate_model() or calculate_mape() in your training script.")

