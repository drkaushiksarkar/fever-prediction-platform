"""Tests for data preprocessing pipeline."""
import numpy as np
import pandas as pd
import pytest

from fever_platform.data.preprocessor import FeverPreprocessor


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 120
    return pd.DataFrame({
        "YEAR": np.repeat(range(2018, 2028), 12)[:n],
        "MONTH": np.tile(range(1, 13), 10)[:n],
        "DISTRICT": np.random.choice(["D1", "D2"], n),
        "BLOCK": np.random.choice(["B1", "B2", "B3"], n),
        "Population": np.random.randint(5000, 50000, n),
        "Fever": np.random.randint(10, 500, n),
        "pv_total": np.random.randint(0, 200, n),
        "pf_total": np.random.randint(0, 300, n),
        "malaria_total": np.random.randint(0, 500, n),
    })


class TestFeverPreprocessor:
    def test_engineer_features_adds_columns(self, sample_data):
        pp = FeverPreprocessor()
        result = pp.engineer_features(sample_data)
        assert "Fever_proportion" in result.columns
        assert "PfPR" in result.columns
        assert "PvPR" in result.columns
        assert "TPR" in result.columns
        assert "monsoon_season" in result.columns
        assert "month_sin" in result.columns

    def test_population_floor(self, sample_data):
        sample_data.loc[0, "Population"] = 0
        pp = FeverPreprocessor(population_floor=5000)
        result = pp.engineer_features(sample_data)
        assert result.loc[0, "Population"] == 5000

    def test_lag_features(self, sample_data):
        pp = FeverPreprocessor()
        featured = pp.engineer_features(sample_data)
        lagged = pp.create_lag_features(featured, lags=[1, 3])
        assert "Fever_lag1" in lagged.columns
        assert "pf_total_lag3" in lagged.columns
        assert lagged.isna().sum().sum() == 0

    def test_deidentify(self, sample_data):
        pp = FeverPreprocessor()
        result, mappings = pp.deidentify(sample_data, ["DISTRICT"])
        assert "DISTRICT_index" in result.columns
        assert "D1" in mappings["DISTRICT"]

    def test_scaler_fit_transform(self, sample_data):
        pp = FeverPreprocessor()
        cols = ["Population", "Fever"]
        scaled = pp.fit_scaler(sample_data, cols)
        assert scaled.shape == (len(sample_data), 2)

    def test_create_sequences(self):
        pp = FeverPreprocessor()
        data = np.random.rand(50, 5)
        X, y = pp.create_sequences(data, target_idx=0, seq_length=10)
        assert X.shape == (40, 10, 5)
        assert y.shape == (40,)
