"""Tests for data loader."""
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from fever_platform.data.loader import FeverDataLoader
from fever_platform.config import DataConfig


class TestFeverDataLoader:
    def test_drop_unnamed(self):
        df = pd.DataFrame({"Unnamed: 0": [1, 2], "col": [3, 4]})
        result = FeverDataLoader._drop_unnamed(df)
        assert "Unnamed: 0" not in result.columns
        assert "col" in result.columns

    def test_validate_columns_raises(self):
        df = pd.DataFrame({"col": [1]})
        with pytest.raises(ValueError, match="missing columns"):
            FeverDataLoader._validate_columns(df, ["col", "missing"], "test")

    def test_validate_columns_passes(self):
        df = pd.DataFrame({"YEAR": [1], "MONTH": [2]})
        FeverDataLoader._validate_columns(df, ["YEAR", "MONTH"], "test")

    def test_merge_empty_raises(self):
        loader = FeverDataLoader(DataConfig())
        malaria = pd.DataFrame({"YEAR": [2020], "MONTH": [1], "DISTRICT": ["X"], "BLOCK": ["Y"]})
        weather = pd.DataFrame({"YEAR": [2021], "MONTH": [6], "DISTRICT": ["Z"], "BLOCK": ["W"]})
        with pytest.raises(ValueError, match="zero rows"):
            loader.merge_datasets(malaria, weather)
