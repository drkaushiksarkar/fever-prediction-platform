"""Centralized configuration management with Pydantic validation."""
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Data pipeline configuration."""
    raw_data_path: Path = Field(default=Path("data/raw"))
    processed_data_path: Path = Field(default=Path("data/processed"))
    master_data_file: str = "master_data_final.csv"
    malaria_data_file: str = "master_data_malaria.csv"
    weather_data_file: str = "master_data_weather.csv"
    population_floor: int = Field(default=3000, ge=100)
    train_ratio: float = Field(default=0.7, gt=0.0, lt=1.0)
    validation_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)

    @field_validator("train_ratio", "validation_ratio")
    @classmethod
    def check_ratios(cls, v: float) -> float:
        if v <= 0 or v >= 1:
            raise ValueError(f"Ratio must be in (0, 1), got {v}")
        return v


class ModelConfig(BaseModel):
    """LSTM-Attention fusion model configuration."""
    lstm_units_1: int = Field(default=128, ge=16)
    lstm_units_2: int = Field(default=64, ge=16)
    attention_heads: int = Field(default=4, ge=1)
    embed_size: int = Field(default=64, ge=8)
    dropout_rate: float = Field(default=0.3, ge=0.0, le=0.8)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    batch_size: int = Field(default=32, ge=1)
    epochs: int = Field(default=100, ge=1)
    early_stopping_patience: int = Field(default=15, ge=1)
    sequence_length: int = Field(default=12, ge=1)
    target_variable: str = "TPR"


class TrainingConfig(BaseModel):
    """Full training pipeline configuration."""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    output_dir: Path = Field(default=Path("outputs"))
    experiment_name: str = "fever_prediction"
    seed: int = 42
    use_mixed_precision: bool = True
    device: str = "auto"
    s3_bucket: Optional[str] = None
    s3_prefix: str = "fever-platform/experiments"

    def checkpoint_dir(self) -> Path:
        return self.output_dir / self.experiment_name / "checkpoints"

    def log_dir(self) -> Path:
        return self.output_dir / self.experiment_name / "logs"
