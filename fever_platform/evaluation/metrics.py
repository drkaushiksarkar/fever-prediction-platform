"""Comprehensive evaluation metrics for fever prediction."""
from typing import Dict

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute full suite of regression metrics."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Safe MAPE (avoid division by zero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
    else:
        mape = float("nan")

    # Normalized RMSE
    range_val = y_true.max() - y_true.min()
    nrmse = rmse / range_val if range_val > 0 else float("nan")

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
        "nrmse": float(nrmse),
        "n_samples": int(len(y_true)),
    }


def compute_per_region_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regions: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics per geographic region."""
    unique_regions = np.unique(regions)
    results = {}
    for region in unique_regions:
        mask = regions == region
        if mask.sum() > 0:
            results[str(region)] = compute_metrics(y_true[mask], y_pred[mask])
    return results
