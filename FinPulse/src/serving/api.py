"""FastAPI serving microservice for FinPulse real-time inference."""
import os
import sys
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.serving.schemas import TransactionRequest, PredictionResponse
from src.serving.predictor import ProductionPredictor

predictor: ProductionPredictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    artifacts_dir = os.path.join(FINPULSE_DIR, "models", "artifacts")
    try:
        predictor = ProductionPredictor(artifacts_dir)
        print(f"✅ Production predictor loaded successfully from {artifacts_dir}")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize production predictor: {e}")
    yield

app = FastAPI(
    title="FinPulse AI - Fraud Intelligence Operations API",
    description="Sub-10ms real-time transaction fraud scoring and explainability engine.",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "FinPulse AI",
        "version": "2.0.0",
        "model_loaded": predictor is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(tx: TransactionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model predictor is not initialized.")
    try:
        result = predictor.predict(tx.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.api:app", host="0.0.0.0", port=8000, reload=True)
