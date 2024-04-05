# Fever Prediction Platform

Production-grade malaria and fever forecasting using LSTM-Attention fusion models.

## Architecture

The platform implements a multi-head self-attention mechanism fused with
bidirectional LSTM layers for temporal epidemiological forecasting. Key components:

- **Data Pipeline**: Automated ingestion, merging, and preprocessing of
  malaria surveillance data (fever, Plasmodium vivax, Plasmodium falciparum)
  with weather covariates at district, block, and sub-center levels
- **Fusion Model**: LSTM encoder with multi-head self-attention for temporal
  feature weighting, predicting Test Positivity Rate (TPR)
- **Explainability**: SHAP-based feature importance analysis
- **Hyperparameter Optimization**: Bayesian search via Keras Tuner
- **Evaluation**: Per-region benchmarking against baseline models
- **Serving**: FastAPI inference endpoint with health checks

## Quick Start

```bash
pip install -e ".[dev,serve]"
make test
make train
make serve
```

## Project Structure

```
fever_platform/
    config.py              # Pydantic configuration management
    data/
        loader.py          # Multi-source data loading and validation
        preprocessor.py    # Feature engineering (TPR, PfPR, lags, cyclical)
        splitter.py        # Temporal and geographic splitting
    models/
        attention.py       # Multi-Head Self-Attention layer
        fusion.py          # LSTM-Attention fusion architecture
        baseline.py        # Baseline models for benchmarking
        tuner.py           # Bayesian hyperparameter optimization
    explain/
        shap_explainer.py  # SHAP feature importance analysis
    evaluation/
        metrics.py         # MAE, RMSE, R2, MAPE, NRMSE
        benchmarker.py     # Multi-model comparison framework
    training/
        run_pipeline.py    # End-to-end training orchestration
    api/
        server.py          # FastAPI inference server
tests/
    test_preprocessor.py
    test_attention.py
    test_metrics.py
    test_loader.py
    test_splitter.py
```

## Data Format

Input data requires three CSV files with columns:
- **Malaria data**: YEAR, MONTH, DISTRICT, BLOCK, Population, Fever, pv_total, pf_total, malaria_total
- **Weather data**: YEAR, MONTH, DISTRICT, BLOCK, plus weather metric columns
- **Master data**: Combined dataset with all fields

## License

MIT
