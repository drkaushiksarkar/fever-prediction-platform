"""Dataset splitting strategies for fever prediction."""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    date_col: str = "DATE",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically to prevent leakage."""
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def split_by_geography(
    df: pd.DataFrame, group_col: str = "DISTRICT"
) -> Dict[str, pd.DataFrame]:
    """Split dataset by geographic unit for per-region models."""
    groups = {}
    for name, group in df.groupby(group_col):
        groups[str(name)] = group.reset_index(drop=True)
    logger.info(f"Split into {len(groups)} geographic groups by {group_col}")
    return groups


def split_by_area_index(
    values: np.ndarray, area_idx: int = 0
) -> Dict[int, np.ndarray]:
    """Split array data by area index column for per-area training."""
    unique_areas = np.unique(values[:, area_idx])
    splits = {}
    for area in unique_areas:
        mask = values[:, area_idx] == area
        splits[int(area)] = values[mask]
    return splits
