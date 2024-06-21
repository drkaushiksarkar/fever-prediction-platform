"""Data loading and validation for fever/malaria datasets."""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from fever_platform.config import DataConfig


class FeverDataLoader:
    """Load, validate, and merge fever/malaria/weather datasets."""

    REQUIRED_MALARIA_COLS = [
        "YEAR", "MONTH", "DISTRICT", "BLOCK",
        "Population", "Fever", "pv_total", "pf_total", "malaria_total",
    ]
    REQUIRED_WEATHER_COLS = [
        "YEAR", "MONTH", "DISTRICT", "BLOCK",
    ]
    MERGE_KEYS = ["YEAR", "MONTH", "DISTRICT", "BLOCK"]

    def __init__(self, config: DataConfig):
        self.config = config

    def load_master(self) -> pd.DataFrame:
        """Load and validate master dataset."""
        path = self.config.raw_data_path / self.config.master_data_file
        logger.info(f"Loading master data from {path}")
        df = pd.read_csv(path)
        df = self._drop_unnamed(df)
        self._validate_columns(df, self.REQUIRED_MALARIA_COLS, "master")
        logger.info(f"Master data: {len(df)} rows, {len(df.columns)} columns")
        return df

    def load_malaria(self) -> pd.DataFrame:
        """Load and aggregate malaria data at block level."""
        path = self.config.raw_data_path / self.config.malaria_data_file
        logger.info(f"Loading malaria data from {path}")
        df = pd.read_csv(path)
        df = self._drop_unnamed(df)
        self._validate_columns(df, self.REQUIRED_MALARIA_COLS, "malaria")

        agg_cols = ["Population", "Fever", "pv_total", "pf_total", "malaria_total"]
        aggregated = (
            df.groupby(self.MERGE_KEYS)[agg_cols]
            .sum()
            .reset_index()
        )
        logger.info(f"Aggregated malaria data: {len(aggregated)} block-month records")
        return aggregated

    def load_weather(self) -> pd.DataFrame:
        """Load and deduplicate weather data."""
        path = self.config.raw_data_path / self.config.weather_data_file
        logger.info(f"Loading weather data from {path}")
        df = pd.read_csv(path)
        df = self._drop_unnamed(df)
        if "CHC" in df.columns:
            df = df.drop(columns=["CHC"])
        before = len(df)
        df = df.drop_duplicates(keep="first")
        logger.info(f"Weather data: {before} -> {len(df)} after dedup")

        # Remove all-zero weather rows
        weather_cols = [c for c in df.columns if c not in self.MERGE_KEYS]
        mask = (df[weather_cols] != 0).any(axis=1)
        df = df[mask].reset_index(drop=True)
        logger.info(f"Weather data after zero removal: {len(df)} rows")
        return df

    def merge_datasets(
        self,
        malaria: pd.DataFrame,
        weather: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge malaria and weather data on common keys."""
        merged = pd.merge(weather, malaria, on=self.MERGE_KEYS, how="inner")
        logger.info(f"Merged dataset: {len(merged)} rows")
        if len(merged) == 0:
            raise ValueError("Merge produced zero rows -- check key alignment")
        return merged

    def load_and_merge(self) -> pd.DataFrame:
        """Full pipeline: load all sources, merge, validate."""
        malaria = self.load_malaria()
        weather = self.load_weather()
        return self.merge_datasets(malaria, weather)

    @staticmethod
    def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in df.columns if c.startswith("Unnamed")]
        if cols:
            df = df.drop(columns=cols)
        return df

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required: list, name: str) -> None:
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"{name} dataset missing columns: {missing}")
