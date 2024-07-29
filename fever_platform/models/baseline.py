"""Baseline models for benchmarking against fusion architecture."""
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from loguru import logger


class BaselineModels:
    """Collection of baseline models for comparison."""

    MODELS = {
        "linear": LinearRegression,
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42
        ),
    }

    def __init__(self):
        self.fitted_models: Dict[str, object] = {}
        self.results: Dict[str, Dict[str, float]] = {}

    def fit_all(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit all baseline models."""
        # Flatten sequences for sklearn models
        X_flat = X_train.reshape(X_train.shape[0], -1) if X_train.ndim == 3 else X_train

        for name, model_fn in self.MODELS.items():
            model = model_fn() if callable(model_fn) else model_fn
            model.fit(X_flat, y_train)
            self.fitted_models[name] = model
            logger.info(f"Fitted baseline: {name}")

    def evaluate_all(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all fitted baselines and return metrics."""
        X_flat = X_test.reshape(X_test.shape[0], -1) if X_test.ndim == 3 else X_test

        for name, model in self.fitted_models.items():
            preds = model.predict(X_flat)
            self.results[name] = {
                "mae": mean_absolute_error(y_test, preds),
                "rmse": np.sqrt(mean_squared_error(y_test, preds)),
                "r2": r2_score(y_test, preds),
            }
            logger.info(f"{name}: MAE={self.results[name]['mae']:.4f}, R2={self.results[name]['r2']:.4f}")

        return self.results

    def comparison_table(self) -> pd.DataFrame:
        """Return formatted comparison DataFrame."""
        return pd.DataFrame(self.results).T.round(4)
