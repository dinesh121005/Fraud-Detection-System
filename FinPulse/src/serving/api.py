"""FastAPI serving microservice for FinPulse real-time inference with R6 Security & Deployment Hardening."""
import os
import sys
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.serving.schemas import TransactionRequest, PredictionResponse
from src.serving.predictor import ProductionPredictor
from src.monitoring.metrics import get_metrics, record_error, CONTENT_TYPE_LATEST
from src.monitoring.logger import get_logger

logger = get_logger("FinPulse.API")

predictor: ProductionPredictor = None
MAX_PAYLOAD_BYTES = 1024 * 1024  # 1MB Maximum Payload Limit

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    artifacts_dir = os.path.join(FINPULSE_DIR, "models", "artifacts")
    try:
        predictor = ProductionPredictor(artifacts_dir)
        print(f"[OK] Production predictor loaded successfully from {artifacts_dir}")
    except Exception as e:
        print(f"[WARN] Warning: Could not initialize production predictor: {e}")
    yield

app = FastAPI(
    title="FinPulse AI - Fraud Intelligence Operations API",
    description="Real-time transaction fraud scoring and explainability engine (sub-10ms ML inference, ~25ms warm scoring P50).",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Middleware (Restricted to safe origins)
ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Security Headers & Payload Size Guard Middleware
@app.middleware("http")
async def security_and_payload_middleware(request: Request, call_next):
    # Enforce request payload size limit (Anti-DoS)
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_PAYLOAD_BYTES:
        record_error("api", "validation_error")
        return Response(content='{"detail":"Payload exceeds maximum allowable size (1MB)"}', status_code=413, media_type="application/json")

    response: Response = await call_next(request)

    # Injected Security Headers (OWASP Recommended)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Cache-Control"] = "no-store, max-age=0"

    return response

@app.get("/health")
def health():
    """Liveness probe: returns service status."""
    return {
        "status": "healthy",
        "service": "FinPulse AI",
        "version": "2.0.0",
        "model_loaded": predictor is not None,
        "metrics_enabled": True
    }

@app.get("/ready")
def ready():
    """Readiness probe: verifies model inference engine is fully initialized to process transactions."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Service not ready: Model predictor is not initialized.")
    return {
        "status": "ready",
        "model_version": "finpulse-v3",
        "feature_schema_version": "2.0"
    }

@app.get("/metrics")
def metrics():
    """Prometheus metrics scrape endpoint."""
    metrics_data = get_metrics().generate_exposition()
    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
def predict(tx: TransactionRequest):
    if predictor is None:
        record_error("api", "not_found")
        raise HTTPException(status_code=503, detail="Model predictor is not initialized.")
    try:
        result = predictor.predict(tx.model_dump())
        return result
    except Exception as e:
        record_error("api", "runtime_error")
        logger.error("Internal prediction error encountered", error=str(e), transaction_id=tx.transaction_id)
        # Prevent internal exception message leakage to API callers
        raise HTTPException(status_code=500, detail="Internal transaction scoring error.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.api:app", host="0.0.0.0", port=8000, reload=True)
