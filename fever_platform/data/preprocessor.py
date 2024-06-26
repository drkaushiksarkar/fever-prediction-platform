"""Feature engineering and preprocessing for fever prediction models."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler, RobustScaler


class FeverPreprocessor:
    """Transform raw data into model-ready features."""

    GEOGRAPHIC_LEVELS = ["DISTRICT", "BLOCK", "CHC", "SC"]
    LAG_FEATURES = ["Fever", "pv_total", "pf_total", "malaria_total"]
    DEFAULT_LAGS = [1, 2, 3, 6, 12]

    def __init__(self, population_floor: int = 3000, scaler_type: str = "robust"):
        self.population_floor = population_floor
        self.scaler_type = scaler_type
        self._scaler: Optional[object] = None
        self._feature_names: List[str] = []

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived epidemiological features."""
        df = df.copy()

        # Floor population to avoid division by zero
        df["Population"] = np.where(
            df["Population"] == 0, self.population_floor, df["Population"]
        )

        # Epidemiological rate features
        df["Fever_proportion"] = df["Fever"] / df["Population"]
        df["PfPR"] = df["pf_total"] / df["Population"]
        df["PvPR"] = df["pv_total"] / df["Population"]
        df["TPR"] = df["malaria_total"] / df["Population"]
        df["Pf_fraction"] = np.where(
            df["malaria_total"] > 0,
            df["pf_total"] / df["malaria_total"],
            0.0,
        )

        # Surveillance coverage
        if "test_total" in df.columns:
            df["Surveillance_proportion"] = df["test_total"] / df["Population"]
            df["Passive_proportion"] = np.where(
                df["test_total"] > 0,
                (df.get("Passive_RDT", 0) + df.get("Passive_Slide", 0)) / df["test_total"],
                0.0,
            )

        # Temporal features
        if "DATE" not in df.columns and "YEAR" in df.columns and "MONTH" in df.columns:
            df["DATE"] = pd.to_datetime(
                df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-01"
            )
        if "MONTH" in df.columns:
            df["monsoon_season"] = ((df["MONTH"] >= 6) & (df["MONTH"] <= 9)).astype(int)
            df["month_sin"] = np.sin(2 * np.pi * df["MONTH"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["MONTH"] / 12)

        logger.info(f"Engineered features: {df.shape[1]} columns")
        return df

    def create_lag_features(
        self,
        df: pd.DataFrame,
        group_cols: Optional[List[str]] = None,
        lags: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Create lagged features for time series modeling."""
        df = df.copy()
        lags = lags or self.DEFAULT_LAGS
        group_cols = group_cols or ["DISTRICT", "BLOCK"]

        for feat in self.LAG_FEATURES:
            if feat not in df.columns:
                continue
            for lag in lags:
                col_name = f"{feat}_lag{lag}"
                df[col_name] = df.groupby(group_cols)[feat].shift(lag)

        before = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.info(f"After lag creation and NaN drop: {before} -> {len(df)} rows")
        return df

    def deidentify(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
        """Replace geographic names with indexed identifiers."""
        columns = columns or [c for c in self.GEOGRAPHIC_LEVELS if c in df.columns]
        df = df.copy()
        mappings: Dict[str, Dict[str, str]] = {}

        for col in columns:
            unique_vals = sorted(df[col].unique())
            mapping = {v: f"index_{i+1}" for i, v in enumerate(unique_vals)}
            mappings[col] = mapping
            index_col = f"{col}_index"
            df[index_col] = df[col].map(mapping)

        logger.info(f"Deidentified {len(columns)} columns: {columns}")
        return df, mappings

    def fit_scaler(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Fit scaler on training data and transform."""
        ScalerClass = RobustScaler if self.scaler_type == "robust" else MinMaxScaler
        self._scaler = ScalerClass()
        self._feature_names = feature_cols
        scaled = self._scaler.fit_transform(df[feature_cols].values)
        return scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted scaler."""
        if self._scaler is None:
            raise RuntimeError("Scaler not fitted. Call fit_scaler first.")
        return self._scaler.transform(df[self._feature_names].values)

    def create_sequences(
        self, data: np.ndarray, target_idx: int, seq_length: int = 12
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences for LSTM input."""
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i - seq_length : i])
            y.append(data[i, target_idx])
        return np.array(X), np.array(y)
