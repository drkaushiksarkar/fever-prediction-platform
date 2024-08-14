"""End-to-end training pipeline for fever prediction."""
import json
from pathlib import Path

import click
import numpy as np
import yaml
from loguru import logger

from fever_platform.config import TrainingConfig
from fever_platform.data.loader import FeverDataLoader
from fever_platform.data.preprocessor import FeverPreprocessor
from fever_platform.data.splitter import temporal_split
from fever_platform.models.fusion import FeverFusionModel
from fever_platform.evaluation.metrics import compute_metrics


@click.command()
@click.option("--config", type=click.Path(exists=True), help="YAML config file")
@click.option("--experiment-name", default=None, help="Override experiment name")
def train(config: str, experiment_name: str):
    """Run the full training pipeline."""
    # Load configuration
    if config:
        with open(config) as f:
            cfg_dict = yaml.safe_load(f)
        cfg = TrainingConfig(**cfg_dict)
    else:
        cfg = TrainingConfig()

    if experiment_name:
        cfg.experiment_name = experiment_name

    # Setup directories
    cfg.checkpoint_dir().mkdir(parents=True, exist_ok=True)
    cfg.log_dir().mkdir(parents=True, exist_ok=True)

    logger.add(cfg.log_dir() / "training.log", rotation="100 MB")
    logger.info(f"Starting experiment: {cfg.experiment_name}")

    # Load and preprocess data
    loader = FeverDataLoader(cfg.data)
    merged = loader.load_and_merge()

    preprocessor = FeverPreprocessor(
        population_floor=cfg.data.population_floor,
        scaler_type="minmax",
    )
    featured = preprocessor.engineer_features(merged)
    featured = preprocessor.create_lag_features(featured)

    # Split data
    train_df, val_df, test_df = temporal_split(
        featured, cfg.data.train_ratio, cfg.data.validation_ratio
    )

    # Prepare features
    feature_cols = [
        c for c in train_df.columns
        if c not in ["DATE", "DISTRICT", "BLOCK", "CHC", "SC", cfg.model.target_variable]
        and train_df[c].dtype in ["float64", "int64", "float32"]
    ]

    X_train_scaled = preprocessor.fit_scaler(train_df, feature_cols)
    X_val_scaled = preprocessor.transform(val_df)
    X_test_scaled = preprocessor.transform(test_df)

    target_idx = feature_cols.index(cfg.model.target_variable) if cfg.model.target_variable in feature_cols else -1

    # Create sequences
    X_train, y_train = preprocessor.create_sequences(
        X_train_scaled, target_idx, cfg.model.sequence_length
    )
    X_val, y_val = preprocessor.create_sequences(
        X_val_scaled, target_idx, cfg.model.sequence_length
    )
    X_test, y_test = preprocessor.create_sequences(
        X_test_scaled, target_idx, cfg.model.sequence_length
    )

    logger.info(f"Training shapes: X={X_train.shape}, y={y_train.shape}")

    # Build and train model
    model = FeverFusionModel(
        input_shape=(cfg.model.sequence_length, len(feature_cols)),
        lstm_units_1=cfg.model.lstm_units_1,
        lstm_units_2=cfg.model.lstm_units_2,
        attention_heads=cfg.model.attention_heads,
        embed_size=cfg.model.embed_size,
        dropout_rate=cfg.model.dropout_rate,
        learning_rate=cfg.model.learning_rate,
    )

    history = model.train(
        X_train, y_train, X_val, y_val,
        epochs=cfg.model.epochs,
        batch_size=cfg.model.batch_size,
        patience=cfg.model.early_stopping_patience,
    )

    # Evaluate
    preds = model.predict(X_test).flatten()
    metrics = compute_metrics(y_test, preds)
    logger.info(f"Test metrics: {json.dumps(metrics, indent=2)}")

    # Save
    model_path = cfg.checkpoint_dir() / "best_model"
    model.save(str(model_path))

    metrics_path = cfg.log_dir() / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Pipeline complete. Model: {model_path}, Metrics: {metrics_path}")


if __name__ == "__main__":
    train()
