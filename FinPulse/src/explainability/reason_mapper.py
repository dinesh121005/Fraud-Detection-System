"""Maps model feature attributions to standardized banking fraud reason codes."""
from typing import List, Dict, Any

REASON_TEMPLATES = {
    "amount_zscore": "Transaction amount significantly exceeds user historical 30-day average",
    "tx_count_1m": "High transaction velocity detected within 1 minute",
    "tx_count_5m": "Elevated transaction velocity detected (multiple transactions within 5 minutes)",
    "tx_count_15m": "High transaction volume spike over 15-minute window",
    "amount_sum_1h": "Cumulative 1-hour spending exceeds safe threshold",
    "is_new_device": "Transaction initiated from an unrecognized hardware device",
    "is_new_location": "Transaction originated from an unrecognized geographic location",
    "speed_kmh_from_prev_tx": "Impossible travel speed detected between consecutive transactions",
    "amount_to_balance_ratio": "Transaction amount drains over 90% of available origin balance",
    "isolation_forest_score": "Unsupervised anomaly detector flagged abnormal multivariate pattern",
    "auth_factor_verified": "Second-factor biometric or OTP authentication missing",
    "is_night": "Unusual late-night transaction execution"
}

def map_attributions_to_reasons(attributions: List[Dict[str, Any]]) -> List[str]:
    """
    Convert raw feature attributions into human-readable explanation sentences.
    Output schema: ['...', '...', '...']
    """
    reasons = []
    for item in attributions:
        feature = item.get("feature", "")
        attrib = item.get("attribution", 0.0)
        if attrib > 0.01 and feature in REASON_TEMPLATES:
            reasons.append(REASON_TEMPLATES[feature])

    if not reasons:
        reasons.append("Standard behavioral consistency verified; low risk indicators.")

    return reasons[:3]
