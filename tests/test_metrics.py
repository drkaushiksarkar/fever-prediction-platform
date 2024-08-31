"""Tests for evaluation metrics."""
import numpy as np
import pytest

from fever_platform.evaluation.metrics import compute_metrics, compute_per_region_metrics


class TestComputeMetrics:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        m = compute_metrics(y, y)
        assert m["mae"] == 0.0
        assert m["rmse"] == 0.0
        assert m["r2"] == 1.0

    def test_known_values(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 2.8])
        m = compute_metrics(y_true, y_pred)
        assert 0 < m["mae"] < 0.3
        assert m["r2"] > 0.9

    def test_per_region(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1.1, 1.9, 3.1, 4.2])
        regions = np.array(["A", "A", "B", "B"])
        results = compute_per_region_metrics(y_true, y_pred, regions)
        assert "A" in results
        assert "B" in results
        assert results["A"]["n_samples"] == 2
