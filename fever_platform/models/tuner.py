"""Hyperparameter tuning with Keras Tuner for fever fusion model."""
from typing import Tuple

import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Concatenate, LeakyReLU,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from loguru import logger

from fever_platform.models.attention import MultiHeadSelfAttention


class FeverModelTuner:
    """Bayesian hyperparameter optimization for fusion architecture."""

    def __init__(
        self,
        input_shape: Tuple[int, int],
        max_trials: int = 50,
        executions_per_trial: int = 1,
        directory: str = "tuner_results",
        project_name: str = "fever_fusion",
    ):
        self.input_shape = input_shape
        self.tuner = kt.BayesianOptimization(
            hypermodel=lambda hp: self._build_model(hp),
            objective="val_loss",
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory=directory,
            project_name=project_name,
        )

    def _build_model(self, hp: kt.HyperParameters) -> Model:
        units_1 = hp.Int("lstm_units_1", 32, 256, step=32)
        units_2 = hp.Int("lstm_units_2", 16, 128, step=16)
        heads = hp.Choice("attention_heads", [1, 2, 4, 8])
        embed = hp.Int("embed_size", 16, 128, step=16)
        dropout = hp.Float("dropout", 0.1, 0.5, step=0.1)
        lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")

        # Ensure embed_size is divisible by num_heads
        embed = (embed // heads) * heads

        inputs = Input(shape=self.input_shape)
        x = LSTM(units_1, return_sequences=True)(inputs)
        x = Dropout(dropout)(x)
        lstm_out = LSTM(units_2, return_sequences=True)(x)

        attention_out = MultiHeadSelfAttention(
            embed_size=embed, num_heads=heads
        )(lstm_out)

        fused = Concatenate(axis=-1)([lstm_out, attention_out])
        x = Dense(64)(fused)
        x = LeakyReLU(negative_slope=0.1)(x)
        x = Dropout(dropout)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(lr), loss="mse", metrics=["mae"])
        return model

    def search(self, X_train, y_train, X_val, y_val, epochs: int = 50) -> None:
        """Run hyperparameter search."""
        logger.info("Starting Bayesian hyperparameter search...")
        self.tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ],
        )

    def best_hyperparameters(self) -> dict:
        """Return best hyperparameters found."""
        best = self.tuner.get_best_hyperparameters(1)[0]
        return {p: best.get(p) for p in best.values}

    def best_model(self) -> Model:
        """Return the best model."""
        return self.tuner.get_best_models(1)[0]
