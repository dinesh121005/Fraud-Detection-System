"""In-Memory Low-Latency Model Scoring and Risk Engine Pipeline with R6.1 Telemetry."""
import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.ensemble._iforest")
warnings.filterwarnings("ignore", category=UserWarning, module="shap.explainers._tree")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

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

        # Load authentic legitimate feature vectors from processed dataset for reference background
        finpulse_root = os.path.abspath(os.path.join(artifacts_dir, "..", ".."))
        bg_data_path = os.path.join(finpulse_root, "data", "processed", "sparkov_features.npz")
        legit_background = None
        if os.path.exists(bg_data_path):
            try:
                npz = np.load(bg_data_path)
                X_tr = npz["X_train"]
                y_tr = npz["y_train"]
                legit_idx = np.where(y_tr == 0)[0]
                if len(legit_idx) >= 200:
                    legit_background = X_tr[legit_idx[:200]].astype(np.float32)
            except Exception as ex:
                logger.warning(f"Could not load background data from {bg_data_path}: {ex}")

        if legit_background is None or len(legit_background) < 50:
            legit_background = np.zeros((200, 32), dtype=np.float32)

        self.anomaly_detector.fit(legit_background)
        self.explainer = ShapExplainerWrapper(self.model, legit_background[:50])
        self.risk_engine = FinPulseRiskEngine()

    def predict(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        """Execute end-to-end transaction scoring under 10ms with high-precision telemetry."""
        t0 = time.perf_counter()

        cust_id = tx.get("customer_id", "default_cust")
        tx_id = tx.get("transaction_id", "tx_0")
        ts = float(tx.get("timestamp", time.time()))
        amount = float(tx.get("amount", 0.0))

        # Contextual location, device, and temporal extraction
        loc = tx.get("location")
        lat = float(tx.get("latitude", loc.get("latitude", 0.0) if isinstance(loc, dict) else 0.0))
        lon = float(tx.get("longitude", loc.get("longitude", 0.0) if isinstance(loc, dict) else 0.0))
        hloc = tx.get("home_location")
        hlat = float(tx.get("home_latitude", hloc.get("latitude", 0.0) if isinstance(hloc, dict) else 0.0))
        hlon = float(tx.get("home_longitude", hloc.get("longitude", 0.0) if isinstance(hloc, dict) else 0.0))
        cat = str(tx.get("category", "general"))
        hour = int(tx.get("hour_of_day", int((ts % 86400) // 3600)))
        dev_id = str(tx.get("device_id", "unknown_device"))

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
                amount=amount,
                category=cat,
                hour=hour,
                lat=lat,
                lon=lon,
                device_id=dev_id,
                home_lat=hlat,
                home_lon=hlon
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
            feature_cols = getattr(getattr(self.model, "model", None), "feature_name_", None)
            if feature_cols:
                feat_input = pd.DataFrame(ordered_vec, columns=feature_cols)
                raw_prob = float(self.model.predict_proba(feat_input)[0])
            else:
                raw_prob = float(self.model.predict_proba(ordered_vec)[0])

            calibrated_prob = float(self.calibrator.predict(np.array([raw_prob]))[0])
            anomaly_score = float(self.anomaly_detector.score_anomaly(ordered_vec)[0])
            attributions = self.explainer.explain_transaction(ordered_vec[0], top_k=8)
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
            result["attributions"] = attributions
            result["features_dict"] = feat_dict
            if "diagnostics" in result and isinstance(result["diagnostics"], dict):
                result["diagnostics"]["attributions"] = attributions
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
