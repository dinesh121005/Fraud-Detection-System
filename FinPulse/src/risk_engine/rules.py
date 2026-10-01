"""Deterministic heuristic and velocity business rules engine."""
from typing import Dict, Any, Tuple, List

class DeterministicRuleEngine:
    """
    Evaluates rule violations and hard safety blocks.
    Reuses core domain checks from legacy utils/helpers.py while providing
    hard overrides and normalized rule risk scores in [0.0, 1.0].
    """

    def evaluate_rules(self, tx: Dict[str, Any], features: Dict[str, Any]) -> Tuple[float, List[str], bool]:
        """
        Evaluate business rules.
        Returns:
        - rule_risk: float in [0.0, 1.0]
        - rule_violations: List[str]
        - hard_block: bool (True if critical override triggered)
        """
        rule_risk = 0.0
        violations = []
        hard_block = False

        amount = float(tx.get("amount", 0.0))
        origin_balance = float(tx.get("origin_balance", 0.0))
        auth_verified = int(tx.get("auth_verified", 1))
        speed_kmh = float(features.get("speed_kmh_from_prev_tx", 0.0))
        tx_count_1m = int(features.get("tx_count_1m", 1))

        # 1. Critical Hard Override: Impossible Travel Velocity
        if speed_kmh > 800.0:
            rule_risk = 1.0
            violations.append("CRITICAL: Impossible travel velocity (>800 km/h)")
            hard_block = True

        # 2. Critical Hard Override: Zero-balance or high-drain without auth
        if origin_balance == 0.0 and amount > 10000.0 and auth_verified == 0:
            rule_risk = 1.0
            violations.append("CRITICAL: Zero-balance account high-value drain without authentication")
            hard_block = True
        elif origin_balance > 0.0 and (amount / (origin_balance + 1e-5)) > 0.90 and auth_verified == 0:
            rule_risk = 0.85
            violations.append("CRITICAL: High-value account balance drain without authentication")
            if tx_count_1m >= 3:
                hard_block = True

        # 3. Soft Rule: Origin balance depletion
        if origin_balance > 0.0 and (amount / (origin_balance + 1e-5)) > 0.95:
            rule_risk += 0.25
            violations.append("Transaction drains over 95% of origin balance")

        # 4. Soft Rule: Unverified authentication
        if auth_verified == 0:
            rule_risk += 0.30
            violations.append("Second-factor biometric or OTP authentication missing")

        # 5. Soft Rule: High 1-minute velocity
        if tx_count_1m >= 4:
            rule_risk += 0.25
            violations.append(f"Rapid micro-velocity spike ({tx_count_1m} transactions in 1 minute)")

        return min(rule_risk, 1.0), violations, hard_block
