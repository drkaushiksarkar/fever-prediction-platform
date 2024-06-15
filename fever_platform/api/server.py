"""FastAPI server for fever prediction inference."""
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from fever_platform.models.fusion import FeverFusionModel


app = FastAPI(
    title="Fever Prediction Platform",
    description="LSTM-Attention fusion model for malaria and fever forecasting",
    version="1.0.0",
)

# Global model holder
_model = None


class PredictionRequest(BaseModel):
    """Input for fever prediction."""
    sequence: List[List[float]] = Field(
        ..., description="Time series sequence (seq_length x n_features)"
    )


class PredictionResponse(BaseModel):
    """Prediction output."""
    tpr_prediction: float = Field(..., description="Predicted test positivity rate")
    confidence: float = Field(default=0.0, description="Model confidence score")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.on_event("startup")
async def load_model():
    global _model
    model_path = Path("outputs/fever_prediction/checkpoints/best_model")
    if model_path.exists():
        _model = FeverFusionModel.load(str(model_path))
        logger.info("Model loaded successfully")
    else:
        logger.warning(f"No model found at {model_path}")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", model_loaded=_model is not None)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        X = np.array(request.sequence)[np.newaxis, ...]
        pred = _model.predict(X).flatten()[0]
        return PredictionResponse(tpr_prediction=float(pred))
    except Exception as e:
        raise HTTPException(400, f"Prediction error: {str(e)}")
