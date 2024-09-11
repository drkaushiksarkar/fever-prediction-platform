"""Tests for dataset splitting."""
import numpy as np
import pandas as pd
import pytest

from fever_platform.data.splitter import temporal_split, split_by_geography


class TestTemporalSplit:
    def test_split_sizes(self):
        df = pd.DataFrame({
            "DATE": pd.date_range("2020-01-01", periods=100, freq="D"),
            "value": range(100),
        })
        train, val, test = temporal_split(df, 0.7, 0.15)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_no_leakage(self):
        df = pd.DataFrame({
            "DATE": pd.date_range("2020-01-01", periods=100, freq="D"),
            "value": range(100),
        })
        train, val, test = temporal_split(df, 0.7, 0.15)
        assert train["DATE"].max() < val["DATE"].min()
        assert val["DATE"].max() < test["DATE"].min()


class TestSplitByGeography:
    def test_correct_groups(self):
        df = pd.DataFrame({
            "DISTRICT": ["A", "A", "B", "B", "C"],
            "value": [1, 2, 3, 4, 5],
        })
        groups = split_by_geography(df)
        assert len(groups) == 3
        assert len(groups["A"]) == 2
