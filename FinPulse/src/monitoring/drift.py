"""
FinPulse R7-J — Population Stability Index (PSI) & Distribution Drift Monitor.

Measures data and concept drift across:
- Numerical 32-feature distributions
- Categorical features
- Calibrated fraud probability & hybrid risk score distributions

Standards:
- PSI < 0.10: Stable distribution (No significant change)
- 0.10 <= PSI < 0.25: Moderate drift (Warning / Telemetry monitor)
- PSI >= 0.25: Significant drift (Action required / Retraining trigger)

Includes:
- Robust zero-count Laplace/epsilon smoothing (1e-4)
- Adaptive quantile-based and uniform binning
- Insufficient data guards (requires minimum sample threshold)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Literal

from src.monitoring.logger import get_logger

logger = get_logger("FinPulse.Monitoring.Drift")

DriftSeverity = Literal["STABLE", "MODERATE_DRIFT", "SIGNIFICANT_DRIFT"]


@dataclass(frozen=True)
class FeatureDriftReport:
    """PSI analysis result for a single feature or score."""
    feature_name: str
    psi_value: float
    severity: DriftSeverity
    reference_sample_count: int
    current_sample_count: int
    bin_details: List[Dict[str, float]]
    action_required: bool


@dataclass(frozen=True)
class SystemDriftAssessment:
    """Aggregated drift report across entire feature set and model scores."""
    overall_status: DriftSeverity
    max_psi: float
    mean_psi: float
    features_evaluated: int
    drifting_features_count: int
    feature_reports: Dict[str, FeatureDriftReport]
    retraining_recommended: bool


class PSIDriftMonitor:
    """
    Computes Population Stability Index (PSI) to detect distribution drift
    between a frozen reference baseline and an active inference window.
    """

    MIN_SAMPLE_SIZE = 50
    EPSILON = 1e-4

    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins

    def calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "feature"
    ) -> FeatureDriftReport:
        """
        Calculate PSI between reference and current populations.
        PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
        """
        ref_arr = np.asarray(reference, dtype=np.float64)
        curr_arr = np.asarray(current, dtype=np.float64)

        # Filter out NaN/Inf
        ref_clean = ref_arr[np.isfinite(ref_arr)]
        curr_clean = curr_arr[np.isfinite(curr_arr)]

        if len(ref_clean) < self.MIN_SAMPLE_SIZE or len(curr_clean) < self.MIN_SAMPLE_SIZE:
            logger.warning("Insufficient samples for reliable PSI calculation", feature=feature_name)
            return FeatureDriftReport(
                feature_name=feature_name,
                psi_value=0.0,
                severity="STABLE",
                reference_sample_count=len(ref_clean),
                current_sample_count=len(curr_clean),
                bin_details=[],
                action_required=False
            )

        # Establish bin breakpoints from reference population quantiles
        quantiles = np.linspace(0.0, 1.0, self.num_bins + 1)
        breakpoints = np.unique(np.quantile(ref_clean, quantiles))

        # If too few unique quantiles (e.g. constant/sparse binary feature), use uniform range
        if len(breakpoints) < 3:
            min_val = min(np.min(ref_clean), np.min(curr_clean))
            max_val = max(np.max(ref_clean), np.max(curr_clean))
            if min_val == max_val:
                # Completely invariant constant
                return FeatureDriftReport(
                    feature_name=feature_name,
                    psi_value=0.0,
                    severity="STABLE",
                    reference_sample_count=len(ref_clean),
                    current_sample_count=len(curr_clean),
                    bin_details=[],
                    action_required=False
                )
            breakpoints = np.linspace(min_val, max_val, self.num_bins + 1)

        # Compute empirical frequencies
        ref_counts, _ = np.histogram(ref_clean, bins=breakpoints)
        curr_counts, _ = np.histogram(curr_clean, bins=breakpoints)

        ref_total = len(ref_clean)
        curr_total = len(curr_clean)

        total_psi = 0.0
        bin_details = []

        for i in range(len(ref_counts)):
            ref_pct = (ref_counts[i] / ref_total) + self.EPSILON
            curr_pct = (curr_counts[i] / curr_total) + self.EPSILON

            bin_psi = (curr_pct - ref_pct) * math.log(curr_pct / ref_pct)
            total_psi += bin_psi

            bin_details.append({
                "bin_index": float(i),
                "lower_bound": float(breakpoints[i]),
                "upper_bound": float(breakpoints[i + 1]),
                "expected_pct": round(ref_pct, 4),
                "actual_pct": round(curr_pct, 4),
                "bin_psi": round(bin_psi, 5)
            })

        total_psi = round(max(0.0, total_psi), 4)

        if total_psi >= 0.25:
            severity: DriftSeverity = "SIGNIFICANT_DRIFT"
            action = True
        elif total_psi >= 0.10:
            severity = "MODERATE_DRIFT"
            action = False
        else:
            severity = "STABLE"
            action = False

        return FeatureDriftReport(
            feature_name=feature_name,
            psi_value=total_psi,
            severity=severity,
            reference_sample_count=ref_total,
            current_sample_count=curr_total,
            bin_details=bin_details,
            action_required=action
        )

    def evaluate_batch_drift(
        self,
        reference_data: Dict[str, np.ndarray],
        current_data: Dict[str, np.ndarray]
    ) -> SystemDriftAssessment:
        """Evaluate drift across a dictionary of features and risk scores."""
        reports = {}
        psi_values = []
        drifting_count = 0

        common_keys = [k for k in reference_data.keys() if k in current_data]

        for key in common_keys:
            rep = self.calculate_psi(reference_data[key], current_data[key], feature_name=key)
            reports[key] = rep
            psi_values.append(rep.psi_value)
            if rep.severity == "SIGNIFICANT_DRIFT":
                drifting_count += 1

        if not psi_values:
            max_p = 0.0
            mean_p = 0.0
            overall = "STABLE"
        else:
            max_p = float(np.max(psi_values))
            mean_p = float(np.mean(psi_values))
            if max_p >= 0.25 or (len(psi_values) >= 5 and mean_p >= 0.15):
                overall = "SIGNIFICANT_DRIFT"
            elif max_p >= 0.10:
                overall = "MODERATE_DRIFT"
            else:
                overall = "STABLE"

        return SystemDriftAssessment(
            overall_status=overall,
            max_psi=round(max_p, 4),
            mean_psi=round(mean_p, 4),
            features_evaluated=len(reports),
            drifting_features_count=drifting_count,
            feature_reports=reports,
            retraining_recommended=(overall == "SIGNIFICANT_DRIFT")
        )
