#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities for loading resource prediction models (CPU / ReadIO / WriteIO).

Why this exists:
- `scripts/train_resource_model.py` stores *training data* in
  `resource_models/resource_predictor.joblib` because the underlying
  `RandomForestWithInstances` objects are not reliably pickleable.
- Some evaluation / runtime code paths historically expected pickled models under
  keys like `model_cpu`, leading to runtime errors like KeyError: 'model_cpu'.

This module provides a single compatibility layer: load the joblib and, if the
models are missing, **retrain them in-memory** from the saved training data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

import joblib
import numpy as np

from autotune.optimizer.surrogate.base.rf_with_instances import RandomForestWithInstances

logger: logging.Logger = logging.getLogger("autotune.utils.resource_model_loader")


class _SklearnRegressorAdapter:
    """
    Adapter for sklearn regressors to match the `RandomForestWithInstances.predict`
    return signature used across this codebase: `(mean, var)`.
    """

    def __init__(self, model: Any) -> None:
        self._model: Any = model

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        pred: np.ndarray = np.asarray(self._model.predict(X), dtype=float).reshape(-1, 1)
        return pred, None


def _train_rf(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    types: np.ndarray,
    bounds: list,
    num_trees: int,
    seed: int = 42,
) -> RandomForestWithInstances:
    """
    Train a `RandomForestWithInstances` resource model.

    Notes:
    - The RF implementation expects y to be a 2D column vector.
    - We intentionally do not attempt to persist the trained model to disk
      (it may not be pickleable depending on SWIG bindings).
    """
    model: RandomForestWithInstances = RandomForestWithInstances(
        types=types,
        bounds=bounds,
        log_y=False,
        num_trees=int(num_trees),
        seed=int(seed),
    )
    y_col: np.ndarray = np.asarray(y_train, dtype=float).reshape(-1, 1)
    model.train(np.asarray(X_train, dtype=float), y_col)
    return model


def load_or_retrain_resource_models(model_path: str) -> Dict[str, Any]:
    """
    Load resource model bundle from `model_path`.

    Returns a dict that always contains:
    - `model_cpu`, `model_read_io`, `model_write_io`
    - `workload_encoder` (may be empty if unavailable)
    - `types`, `bounds` (if available in the bundle)
    - `knob_config_file`, `knob_num` (if available in the bundle)
    """
    model_data: Any = joblib.load(model_path)
    if not isinstance(model_data, Mapping):
        raise TypeError(f"Expected a dict-like object in {model_path}, got {type(model_data)}")

    # Fast path: models are already present (older bundles / custom builds).
    if (
        "model_cpu" in model_data
        and "model_read_io" in model_data
        and "model_write_io" in model_data
    ):
        # If a sklearn backend was used, wrap to preserve the expected `.predict()` signature.
        backend: str = str(model_data.get("model_backend", "rfwi"))
        if backend.startswith("sklearn"):
            return {
                "model_cpu": _SklearnRegressorAdapter(model_data["model_cpu"]),
                "model_read_io": _SklearnRegressorAdapter(model_data["model_read_io"]),
                "model_write_io": _SklearnRegressorAdapter(model_data["model_write_io"]),
                "workload_encoder": model_data.get("workload_encoder", {}),
                "types": model_data.get("types"),
                "bounds": model_data.get("bounds"),
                "knob_config_file": model_data.get("knob_config_file"),
                "knob_num": model_data.get("knob_num"),
            }
        return {
            "model_cpu": model_data["model_cpu"],
            "model_read_io": model_data["model_read_io"],
            "model_write_io": model_data["model_write_io"],
            "workload_encoder": model_data.get("workload_encoder", {}),
            "types": model_data.get("types"),
            "bounds": model_data.get("bounds"),
            "knob_config_file": model_data.get("knob_config_file"),
            "knob_num": model_data.get("knob_num"),
        }

    # Compatibility path: retrain from stored training data produced by
    # `scripts/train_resource_model.py`.
    required_keys: tuple[str, ...] = (
        "X_train",
        "y_cpu_train",
        "y_read_io_train",
        "y_write_io_train",
        "types",
        "bounds",
    )
    missing: list[str] = [k for k in required_keys if k not in model_data]
    if missing:
        raise KeyError(
            "Resource model bundle is missing required keys for retraining: "
            f"{missing}. Available keys: {sorted(model_data.keys())}"
        )

    X_train_cfg: np.ndarray = np.asarray(model_data["X_train"], dtype=float)
    wf_train: Optional[np.ndarray]
    if "wf_train" in model_data and model_data["wf_train"] is not None:
        wf_train = np.asarray(model_data["wf_train"], dtype=float)
        X_train: np.ndarray = np.hstack([X_train_cfg, wf_train])
    else:
        wf_train = None
        X_train = X_train_cfg

    types: np.ndarray = np.asarray(model_data["types"])
    bounds: list = list(model_data["bounds"])
    num_trees: int = int(model_data.get("num_trees", 200))

    logger.info(
        "[Resource Models] Models not found in bundle; retraining in-memory "
        "(samples=%d, features=%d, num_trees=%d).",
        int(X_train.shape[0]),
        int(X_train.shape[1]),
        int(num_trees),
    )

    model_cpu: RandomForestWithInstances = _train_rf(
        X_train=X_train,
        y_train=np.asarray(model_data["y_cpu_train"], dtype=float),
        types=types,
        bounds=bounds,
        num_trees=num_trees,
    )
    model_read_io: RandomForestWithInstances = _train_rf(
        X_train=X_train,
        y_train=np.asarray(model_data["y_read_io_train"], dtype=float),
        types=types,
        bounds=bounds,
        num_trees=num_trees,
    )
    model_write_io: RandomForestWithInstances = _train_rf(
        X_train=X_train,
        y_train=np.asarray(model_data["y_write_io_train"], dtype=float),
        types=types,
        bounds=bounds,
        num_trees=num_trees,
    )

    workload_encoder: Dict[str, Any] = dict(model_data.get("workload_encoder", {}) or {})
    if "n_features" not in workload_encoder:
        workload_encoder["n_features"] = int(wf_train.shape[1]) if wf_train is not None else 0
    if "unique_workloads" not in workload_encoder:
        workload_encoder["unique_workloads"] = []

    return {
        "model_cpu": model_cpu,
        "model_read_io": model_read_io,
        "model_write_io": model_write_io,
        "workload_encoder": workload_encoder,
        "types": types,
        "bounds": bounds,
        "knob_config_file": model_data.get("knob_config_file"),
        "knob_num": model_data.get("knob_num"),
    }


