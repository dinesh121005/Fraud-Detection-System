"""Sub-10ms In-Memory Model Scoring and Risk Engine Pipeline with R6.1 Telemetry."""
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
from src.monitoring.metrics import (
    record_transaction,
    record_decision,
    record_stage_latency,
    record_error,
)
from src.monitoring.correlation import set_correlation_context
from src.monitoring.logger import get_logger

logger = get_logger("FinPulse.Predictor")


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
        """Execute end-to-end transaction scoring under 10ms with high-precision telemetry."""
        t0 = time.perf_counter()

        cust_id = tx.get("customer_id", "default_cust")
        tx_id = tx.get("transaction_id", "tx_0")
        ts = float(tx.get("timestamp", time.time()))
        amount = float(tx.get("amount", 0.0))

        # Establish execution correlation context
        set_correlation_context(
            transaction_id=tx_id,
            customer_id=cust_id,
            model_version="finpulse-v3",
            feature_schema_version="2.0"
        )
        record_transaction(status="accepted")

        # 1. Query Redis rolling state (1m, 5m, 15m, 1h velocity)
        t_redis = time.perf_counter()
        try:
            state = self.redis_window.record_and_fetch_velocity(
                customer_id=cust_id,
                tx_id=tx_id,
                timestamp=ts,
                amount=amount
            )
        except Exception as e:
            record_error("redis", "runtime_error")
            logger.error("Redis state access failed", error=str(e), stage="redis")
            raise
        finally:
            record_stage_latency("redis", (time.perf_counter() - t_redis) * 1000.0)

        # 2. Extract 32-feature vector
        t_feat = time.perf_counter()
        try:
            feat_vec = self.pipeline.transform_transaction_dict(tx, state)
            ordered_vec = np.array(feat_vec.to_ordered_vector(), dtype=np.float32).reshape(1, -1)
        except Exception as e:
            record_error("feature_engine", "validation_error")
            logger.error("Feature transformation failed", error=str(e), stage="feature_engine")
            raise
        finally:
            record_stage_latency("feature_engine", (time.perf_counter() - t_feat) * 1000.0)

        # 3. Supervised Prediction, Calibration, Anomaly, and SHAP
        t_model = time.perf_counter()
        try:
            raw_prob = float(self.model.predict_proba(ordered_vec)[0])
            calibrated_prob = float(self.calibrator.predict(np.array([raw_prob]))[0])
            anomaly_score = float(self.anomaly_detector.score_anomaly(ordered_vec)[0])
            attributions = self.explainer.explain_transaction(ordered_vec[0], top_k=3)
            reasons = map_attributions_to_reasons(attributions)
        except Exception as e:
            record_error("model", "runtime_error")
            logger.error("Model scoring failed", error=str(e), stage="model")
            raise
        finally:
            record_stage_latency("model", (time.perf_counter() - t_model) * 1000.0)

        # 4. Hybrid Risk Fusion & Business Rules
        t_risk = time.perf_counter()
        feat_dict = feat_vec.model_dump()
        try:
            result = self.risk_engine.evaluate_risk(
                tx=tx,
                features_dict=feat_dict,
                calibrated_probability=calibrated_prob,
                anomaly_score=anomaly_score,
                reasons=reasons
            )
        except Exception as e:
            record_error("risk_engine", "runtime_error")
            logger.error("Risk evaluation failed", error=str(e), stage="risk_engine")
            raise
        finally:
            record_stage_latency("risk_engine", (time.perf_counter() - t_risk) * 1000.0)

        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        result["latency_ms"] = round(total_latency_ms, 2)
        record_stage_latency("e2e", total_latency_ms)

        # Record decision telemetry with bounded categorical labels
        diag = result.get("diagnostics", {})
        rule_res = result.get("rule_result", {})
        record_decision(
            decision=result["decision"],
            risk_level=result.get("risk_level"),
            hard_block=bool(diag.get("hard_block", False)),
            active_triggers=diag.get("active_triggers", []),
            matched_rules=rule_res.get("matched_rules", []) if isinstance(rule_res, dict) else []
        )

        logger.info(
            "Transaction scored successfully",
            decision=result["decision"],
            risk_score=result["risk_score"],
            latency_ms=result["latency_ms"],
            hard_block=diag.get("hard_block", False)
        )

        return result
