"""LSTM-Attention Fusion Model for fever and malaria prediction."""
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Concatenate, LeakyReLU,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from loguru import logger

from fever_platform.models.attention import MultiHeadSelfAttention


class FeverFusionModel:
    """LSTM with Multi-Head Self-Attention fusion for TPR prediction.

    Architecture:
        Input -> LSTM(units_1) -> Dropout -> LSTM(units_2)
            -> MultiHeadSelfAttention -> Concatenate(LSTM_out, Attention_out)
            -> Dense(64) -> LeakyReLU -> Dense(1, sigmoid)

    The attention mechanism allows the model to weight different
    temporal positions based on their relevance to the prediction task,
    while the LSTM layers capture sequential dependencies in the
    epidemiological time series.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        lstm_units_1: int = 128,
        lstm_units_2: int = 64,
        attention_heads: int = 4,
        embed_size: int = 64,
        dropout_rate: float = 0.3,
        learning_rate: float = 1e-3,
    ):
        self.input_shape = input_shape
        self.model = self._build(
            input_shape, lstm_units_1, lstm_units_2,
            attention_heads, embed_size, dropout_rate, learning_rate,
        )
        logger.info(
            f"FeverFusionModel built: {self.model.count_params():,} params, "
            f"input_shape={input_shape}"
        )

    def _build(
        self,
        input_shape: Tuple[int, int],
        lstm_units_1: int,
        lstm_units_2: int,
        attention_heads: int,
        embed_size: int,
        dropout_rate: float,
        learning_rate: float,
    ) -> Model:
        inputs = Input(shape=input_shape)

        # Temporal feature extraction
        x = LSTM(lstm_units_1, return_sequences=True)(inputs)
        x = Dropout(dropout_rate)(x)
        lstm_out = LSTM(lstm_units_2, return_sequences=True)(x)

        # Multi-head self-attention for temporal weighting
        attention_out = MultiHeadSelfAttention(
            embed_size=embed_size, num_heads=attention_heads
        )(lstm_out)

        # Fusion: concatenate LSTM and attention representations
        fused = Concatenate(axis=-1)([lstm_out, attention_out])

        # Prediction head
        x = Dense(64)(fused)
        x = LeakyReLU(negative_slope=0.1)(x)
        x = Dropout(dropout_rate)(x)

        # Global pooling over time dimension
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=inputs, outputs=outputs, name="fever_fusion")
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        return model

    def train(
        self,
        X_train, y_train,
        X_val=None, y_val=None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15,
    ) -> tf.keras.callbacks.History:
        """Train with early stopping and learning rate reduction."""
        callbacks = [
            EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=patience,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5, patience=patience // 3, min_lr=1e-6,
            ),
        ]
        validation_data = (X_val, y_val) if X_val is not None else None
        return self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: str) -> None:
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FeverFusionModel":
        model = tf.keras.models.load_model(
            path, custom_objects={"MultiHeadSelfAttention": MultiHeadSelfAttention}
        )
        instance = cls.__new__(cls)
        instance.model = model
        instance.input_shape = model.input_shape[1:]
        return instance
