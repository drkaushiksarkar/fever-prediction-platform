"""Benchmarking framework for comparing model performance."""
from typing import Dict, List

import pandas as pd
from loguru import logger

from fever_platform.evaluation.metrics import compute_metrics


class ModelBenchmarker:
    """Run standardized benchmarks across multiple models."""

    def __init__(self, models: Dict[str, object]):
        self.models = models
        self.results: Dict[str, Dict[str, float]] = {}

    def run_benchmark(self, X_test, y_test) -> pd.DataFrame:
        """Evaluate all models and return comparison table."""
        for name, model in self.models.items():
            preds = model.predict(X_test)
            if hasattr(preds, "flatten"):
                preds = preds.flatten()
            metrics = compute_metrics(y_test, preds)
            self.results[name] = metrics
            logger.info(f"{name}: MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")

        return pd.DataFrame(self.results).T

    def generate_report(self) -> str:
        """Generate markdown comparison report."""
        if not self.results:
            return "No benchmark results available."

        df = pd.DataFrame(self.results).T.round(4)
        lines = ["# Model Benchmark Report\n"]
        lines.append(df.to_markdown())
        lines.append("\n## Best Model")

        best = min(self.results, key=lambda k: self.results[k]["rmse"])
        lines.append(f"\nBest by RMSE: **{best}** (RMSE={self.results[best]['rmse']:.4f})")

        return "\n".join(lines)
