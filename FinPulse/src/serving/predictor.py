"""Sub-10ms In-Memory Model Scoring and Risk Engine Pipeline."""
import os
import time
import joblib
import numpy as np
from typing import Dict, Any

from src.features.pipeline import FinPulseFeaturePipeline
from src.state.sliding_window import RedisSlidingWindowEngine
from src.models.anomaly import IsolationForestAnomalyDetector
from src.explainability.explainer import ShapExplainerWrapper
from src.explainability.reason_mapper import map_attributions_to_reasons
from src.risk_engine.engine import FinPulseRiskEngine

class ProductionPredictor:
    """Orchestrates feature extraction, ML scoring, calibration, SHAP, and risk aggregation."""

    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = artifacts_dir

        # 1. Load trained models & calibrators
        model_path = os.path.join(artifacts_dir, "production_candidate_model.joblib")
        cal_path = os.path.join(artifacts_dir, "production_calibrator.joblib")
        pipe_path = os.path.join(artifacts_dir, "production_feature_pipeline.joblib")

        self.model = joblib.load(model_path)
        self.calibrator = joblib.load(cal_path)
        self.pipeline = joblib.load(pipe_path)

        # 2. Initialize ancillary components
        self.redis_window = RedisSlidingWindowEngine()
        self.anomaly_detector = IsolationForestAnomalyDetector()
        
        # Fit anomaly detector on dummy legitimate sample if needed
        dummy_legit = np.random.normal(loc=100.0, scale=30.0, size=(200, 32))
        self.anomaly_detector.fit(dummy_legit)

        self.explainer = ShapExplainerWrapper(self.model, dummy_legit[:50])
        self.risk_engine = FinPulseRiskEngine()

    def predict(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        """Execute end-to-end transaction scoring under 10ms."""
        t0 = time.perf_counter()

        # 1. Query Redis rolling state (1m, 5m, 15m, 1h velocity)
        cust_id = tx.get("customer_id", "default_cust")
        tx_id = tx.get("transaction_id", "tx_0")
        ts = float(tx.get("timestamp", time.time()))
        amount = float(tx.get("amount", 0.0))

        state = self.redis_window.record_and_fetch_velocity(
            customer_id=cust_id,
            tx_id=tx_id,
            timestamp=ts,
            amount=amount
        )

        # 2. Extract 32-feature vector
        feat_vec = self.pipeline.transform_transaction_dict(tx, state)
        ordered_vec = np.array(feat_vec.to_ordered_vector(), dtype=np.float32).reshape(1, -1)

        # 3. Supervised Prediction & Calibration
        raw_prob = float(self.model.predict_proba(ordered_vec)[0])
        calibrated_prob = float(self.calibrator.predict(np.array([raw_prob]))[0])

        # 4. Anomaly score
        anomaly_score = float(self.anomaly_detector.score_anomaly(ordered_vec)[0])

        # 5. SHAP Reason generation
        attributions = self.explainer.explain_transaction(ordered_vec[0], top_k=3)
        reasons = map_attributions_to_reasons(attributions)

        # 6. Hybrid Risk Fusion
        feat_dict = feat_vec.model_dump()
        result = self.risk_engine.evaluate_risk(
            tx=tx,
            features_dict=feat_dict,
            ml_prob=calibrated_prob,
            anomaly_score=anomaly_score,
            reasons=reasons
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        result["latency_ms"] = round(latency_ms, 2)

        return result
