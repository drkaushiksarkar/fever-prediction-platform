"""SHAP-based model explainability for fever prediction models."""
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import shap
from loguru import logger


class FeverExplainer:
    """Generate SHAP explanations for fever prediction models."""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self._explainer = None
        self._shap_values = None

    def compute_shap_values(
        self, X_background: np.ndarray, X_explain: np.ndarray
    ) -> np.ndarray:
        """Compute SHAP values using DeepExplainer."""
        logger.info(
            f"Computing SHAP values: background={X_background.shape}, "
            f"explain={X_explain.shape}"
        )
        self._explainer = shap.DeepExplainer(self.model, X_background)
        self._shap_values = self._explainer.shap_values(X_explain)
        return self._shap_values

    def summary_plot(self, output_path: Optional[Path] = None) -> None:
        """Generate SHAP summary bar plot."""
        if self._shap_values is None:
            raise RuntimeError("Call compute_shap_values first")
        values = self._shap_values
        if isinstance(values, list):
            values = values[0]
        # Aggregate across time steps
        if values.ndim == 3:
            values = np.mean(np.abs(values), axis=1)
        shap.summary_plot(
            values, feature_names=self.feature_names,
            plot_type="bar", show=False,
        )
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"SHAP summary saved to {output_path}")
        plt.close()

    def feature_importance(self) -> dict:
        """Return feature importance as a sorted dictionary."""
        if self._shap_values is None:
            raise RuntimeError("Call compute_shap_values first")
        values = self._shap_values
        if isinstance(values, list):
            values = values[0]
        if values.ndim == 3:
            values = np.mean(np.abs(values), axis=1)
        importance = np.mean(np.abs(values), axis=0)
        ranked = sorted(
            zip(self.feature_names, importance),
            key=lambda x: x[1],
            reverse=True,
        )
        return {name: float(val) for name, val in ranked}
