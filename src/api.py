from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from src.modeling import predict

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"

@asynccontextmanager
async def lifespan(app: FastAPI):
    #if not MODEL_PATH.exists():
            #raise RuntimeError("Model not found. Train the model with: python -m src.train")
    try:     
        app.state.model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise RuntimeError("Model not found. Train the model with: python -m src.train")

    yield
    del app.state.model

class CustomerFeatures(BaseModel):
    recency: int = Field(ge=0)
    total_orders: int = Field(gt=0)
    total_spent: float = Field(gt=0)
    avg_order: float = Field(gt=0)
    purchased_months: int = Field(gt=0)

class PredictResponse(BaseModel):
    label: int = Field(ge=0, le=1)
    prob: float = Field(ge=0, le=1)

app = FastAPI(lifespan=lifespan)

@app.post("/predict", response_model=PredictResponse)
def predict_customer(features: CustomerFeatures, request: Request):
    df = pd.DataFrame([features.model_dump()])
    label, prob = predict(request.app.state.model, df)
    return {'label': int(label[0]), 'prob': float(prob[0])}

@app.get("/health")
def health():
    return {"status": "ok"}