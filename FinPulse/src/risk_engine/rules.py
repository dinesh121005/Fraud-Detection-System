"""R5.3 — Deterministic Business and Rule Decisioning Engine.

Evaluates transaction context and feature vector signals across 6 categories:
1. Velocity rules (micro-spikes, 5m bursts, 1h surges)
2. Amount rules (balance depletion, extreme z-scores, zero-balance high value)
3. Location & Device rules (impossible travel, multi-user devices, distant new location)
4. Authentication rules (missing 2FA/step-up, unverified high value)
5. Transaction-Context rules (nighttime transfers, high-drain unverified transfers)
6. Hard-Block rules (critical safety overrides that cannot be downgraded by weighted fusion)

Produces structured, deterministic `RuleResult` with:
- rules_signal in [0.0, 1.0]
- matched_rules with categories, severities, and reasons
- hard_block boolean flag
- hard_block_rules list of rule identifiers
- decision recommendation (APPROVE / REVIEW / BLOCK)
- reasons list
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Literal, Callable, Union


RuleCategory = Literal[
    "velocity",
    "amount",
    "location_device",
    "authentication",
    "transaction_context",
    "hard_block",
]

RuleSeverity = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


@dataclass(frozen=True)
class MatchedRule:
    """Individual rule evaluation match evidence."""
    rule_id: str
    category: RuleCategory
    severity: RuleSeverity
    hard_block: bool
    reason: str
    risk_contribution: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert matched rule to serializable dictionary."""
        return {
            "rule_id": self.rule_id,
            "category": self.category,
            "severity": self.severity,
            "hard_block": self.hard_block,
            "reason": self.reason,
            "risk_contribution": self.risk_contribution,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class RuleResult:
    """
    Canonical R5.3 Rule Evaluation Result Contract.
    
    Provides structured diagnostic attribution for rule contributions,
    hard-block overrides, and decision recommendations.
    
    Supports 3-tuple unpacking `(rules_signal, reasons, hard_block)` for
    100% backward compatibility with legacy `DeterministicRuleEngine.evaluate_rules`.
    """
    rules_signal: float
    matched_rules: List[MatchedRule]
    hard_block: bool
    hard_block_rules: List[str]
    decision: Literal["APPROVE", "REVIEW", "BLOCK"]
    reasons: List[str]
    rule_risk: float = field(default=0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule result to standard dictionary."""
        return {
            "rules_signal": round(float(self.rules_signal), 4),
            "matched_rules": [r.to_dict() for r in self.matched_rules],
            "hard_block": bool(self.hard_block),
            "hard_block_rules": list(self.hard_block_rules),
            "decision": self.decision,
            "reasons": list(self.reasons),
            "rule_risk": round(float(self.rule_risk), 4),
        }

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like subscript indexing."""
        return self.to_dict()[key]

    def __contains__(self, key: str) -> bool:
        """Allow 'key in result' membership checks."""
        return key in self.to_dict()

    def __iter__(self):
        """
        Unpack as 3-tuple `(rules_signal, reasons, hard_block)`.
        Preserves backward compatibility with `rule_risk, rule_violations, hard_block = engine.evaluate_rules(...)`.
        """
        yield self.rules_signal
        yield self.reasons
        yield self.hard_block


@dataclass(frozen=True)
class BusinessRule:
    """Individual deterministic business rule definition."""
    rule_id: str
    category: RuleCategory
    severity: RuleSeverity
    risk_contribution: float
    is_hard_block: bool
    reason: str
    condition: Callable[[Dict[str, Any], Dict[str, Any]], bool]

    def evaluate(self, tx: Dict[str, Any], features: Dict[str, Any]) -> Optional[MatchedRule]:
        """Evaluate rule condition against transaction and feature contexts."""
        try:
            if self.condition(tx, features):
                return MatchedRule(
                    rule_id=self.rule_id,
                    category=self.category,
                    severity=self.severity,
                    hard_block=self.is_hard_block,
                    reason=self.reason,
                    risk_contribution=self.risk_contribution,
                )
        except Exception:
            # Deterministic safety: condition exceptions fail closed (rule does not match)
            return None
        return None


class BusinessRuleEngine:
    """
    R5.3 Business / Rule Decisioning Engine.
    
    Evaluates modular, deterministic business rules across 6 categories.
    Distinguishes ordinary risk-score contributions from hard safety blocks.
    """

    def __init__(
        self,
        rules: Optional[List[BusinessRule]] = None,
        custom_rules: Optional[List[BusinessRule]] = None,
    ):
        if rules is not None:
            self._rules = list(rules)
        else:
            self._rules = self._build_default_rules()

        if custom_rules:
            self._rules.extend(custom_rules)

    def register_rule(self, rule: BusinessRule) -> None:
        """Register a new business rule into the engine."""
        # Replace if existing rule_id
        self._rules = [r for r in self._rules if r.rule_id != rule.rule_id]
        self._rules.append(rule)

    def get_rule(self, rule_id: str) -> Optional[BusinessRule]:
        """Lookup a registered rule by its unique identifier."""
        for r in self._rules:
            if r.rule_id == rule_id:
                return r
        return None

    def list_rules(self) -> List[BusinessRule]:
        """Return list of all registered business rules."""
        return list(self._rules)

    def get_rules_by_category(self, category: str) -> List[BusinessRule]:
        """Filter registered rules by category (or 'hard_block' for all hard-block rules)."""
        if category == "hard_block":
            return [r for r in self._rules if r.is_hard_block]
        return [r for r in self._rules if r.category == category]

    def get_hard_block_rules(self) -> List[BusinessRule]:
        """Return all rules flagged as hard-blocks."""
        return [r for r in self._rules if r.is_hard_block]

    def evaluate_rule(
        self, rule_id: str, tx: Dict[str, Any], features: Dict[str, Any]
    ) -> Optional[MatchedRule]:
        """Evaluate a single rule in isolation."""
        rule = self.get_rule(rule_id)
        if rule is None:
            return None
        return rule.evaluate(tx, features)

    def evaluate(
        self, tx: Dict[str, Any], features: Optional[Dict[str, Any]] = None
    ) -> RuleResult:
        """
        Evaluate all registered business rules against transaction context and feature vector.
        
        Returns:
            RuleResult with normalized rules_signal, matched rules list, hard-block status,
            and decision recommendation.
        """
        feat = features or {}
        matched: List[MatchedRule] = []
        hard_block_rules: List[str] = []
        accumulated_risk = 0.0

        for rule in self._rules:
            res = rule.evaluate(tx, feat)
            if res is not None:
                matched.append(res)
                if res.hard_block:
                    hard_block_rules.append(res.rule_id)
                accumulated_risk += res.risk_contribution

        is_hard_block = len(hard_block_rules) > 0

        # Hard blocks force rules_signal to 1.0 and decision to BLOCK
        if is_hard_block:
            accumulated_risk = max(accumulated_risk, 1.0)
            decision: Literal["APPROVE", "REVIEW", "BLOCK"] = "BLOCK"
        else:
            if accumulated_risk >= 0.70:
                decision = "BLOCK"
            elif accumulated_risk >= 0.30:
                decision = "REVIEW"
            else:
                decision = "APPROVE"

        normalized_signal = float(max(0.0, min(1.0, accumulated_risk)))
        reasons = [m.reason for m in matched]

        return RuleResult(
            rules_signal=normalized_signal,
            matched_rules=matched,
            hard_block=is_hard_block,
            hard_block_rules=hard_block_rules,
            decision=decision,
            reasons=reasons,
            rule_risk=normalized_signal,
        )

    def evaluate_rules(
        self, tx: Dict[str, Any], features: Optional[Dict[str, Any]] = None
    ) -> RuleResult:
        """
        Legacy contract adapter for `DeterministicRuleEngine.evaluate_rules`.
        Returns `RuleResult` which unpacks as `(rules_signal, reasons, hard_block)`.
        """
        return self.evaluate(tx, features)

    @staticmethod
    def _build_default_rules() -> List[BusinessRule]:
        """
        Construct authoritative business rules grounded in actual FinPulse
        features and legacy domain checks.
        """
        rules: List[BusinessRule] = []

        # ======================================================================
        # 1. Velocity Rules
        # ======================================================================
        rules.append(
            BusinessRule(
                rule_id="VEL_MICRO_SPIKE",
                category="velocity",
                severity="MEDIUM",
                risk_contribution=0.25,
                is_hard_block=False,
                reason="Rapid micro-velocity spike (>=4 transactions in 1 minute)",
                condition=lambda tx, feat: int(feat.get("tx_count_1m", 1)) >= 4,
            )
        )
        rules.append(
            BusinessRule(
                rule_id="VEL_5M_BURST",
                category="velocity",
                severity="MEDIUM",
                risk_contribution=0.20,
                is_hard_block=False,
                reason="Rapid transaction burst within 5 minutes (>=6 transactions)",
                condition=lambda tx, feat: int(feat.get("tx_count_5m", 1)) >= 6,
            )
        )
        rules.append(
            BusinessRule(
                rule_id="VEL_1H_SURGE",
                category="velocity",
                severity="HIGH",
                risk_contribution=0.25,
                is_hard_block=False,
                reason="High transaction velocity surge within 1 hour (>=15 transactions)",
                condition=lambda tx, feat: int(feat.get("tx_count_1h", 1)) >= 15,
            )
        )

        # ======================================================================
        # 2. Amount Rules
        # ======================================================================
        rules.append(
            BusinessRule(
                rule_id="AMT_BALANCE_DEPLETION",
                category="amount",
                severity="MEDIUM",
                risk_contribution=0.25,
                is_hard_block=False,
                reason="Transaction drains over 95% of origin balance",
                condition=lambda tx, feat: (
                    float(tx.get("origin_balance", 0.0)) > 0.0
                    and (float(tx.get("amount", 0.0)) / (float(tx.get("origin_balance", 0.0)) + 1e-5)) > 0.95
                ),
            )
        )
        rules.append(
            BusinessRule(
                rule_id="AMT_EXTREME_ZSCORE",
                category="amount",
                severity="HIGH",
                risk_contribution=0.30,
                is_hard_block=False,
                reason="Transaction amount exceeds 4 standard deviations from user baseline",
                condition=lambda tx, feat: float(feat.get("amount_zscore", 0.0)) >= 4.0,
            )
        )
        rules.append(
            BusinessRule(
                rule_id="AMT_ZERO_BALANCE_HIGH_VALUE",
                category="amount",
                severity="HIGH",
                risk_contribution=0.30,
                is_hard_block=False,
                reason="High monetary amount requested on zero-balance account",
                condition=lambda tx, feat: (
                    float(tx.get("origin_balance", 0.0)) == 0.0
                    and float(tx.get("amount", 0.0)) > 5000.0
                ),
            )
        )

        # ======================================================================
        # 3. Location / Device Rules
        # ======================================================================
        rules.append(
            BusinessRule(
                rule_id="DEV_SUSPICIOUS_MULTI_USER",
                category="location_device",
                severity="HIGH",
                risk_contribution=0.30,
                is_hard_block=False,
                reason="Suspicious shared device linked to 5 or more distinct users in 24 hours",
                condition=lambda tx, feat: int(feat.get("device_user_count_24h", 1)) >= 5,
            )
        )
        rules.append(
            BusinessRule(
                rule_id="LOC_DEV_ANOMALOUS_COMBINATION",
                category="location_device",
                severity="HIGH",
                risk_contribution=0.25,
                is_hard_block=False,
                reason="Simultaneous new device and distant new location (>500 km from home)",
                condition=lambda tx, feat: (
                    int(feat.get("is_new_device", 0)) == 1
                    and int(feat.get("is_new_location", 0)) == 1
                    and float(feat.get("distance_from_home_km", 0.0)) > 500.0
                ),
            )
        )

        # ======================================================================
        # 4. Authentication Rules
        # ======================================================================
        rules.append(
            BusinessRule(
                rule_id="AUTH_MISSING_STEP_UP",
                category="authentication",
                severity="MEDIUM",
                risk_contribution=0.30,
                is_hard_block=False,
                reason="Second-factor biometric or OTP authentication missing",
                condition=lambda tx, feat: (
                    int(tx.get("auth_verified", 1)) == 0
                    or tx.get("auth_verified") is False
                    or int(feat.get("auth_factor_verified", 1)) == 0
                ),
            )
        )
        rules.append(
            BusinessRule(
                rule_id="AUTH_HIGH_VALUE_UNVERIFIED",
                category="authentication",
                severity="HIGH",
                risk_contribution=0.35,
                is_hard_block=False,
                reason="High-value transaction attempted without 2FA / biometric authentication",
                condition=lambda tx, feat: (
                    (int(tx.get("auth_verified", 1)) == 0 or tx.get("auth_verified") is False)
                    and float(tx.get("amount", 0.0)) > 2500.0
                ),
            )
        )

        # ======================================================================
        # 5. Transaction-Context Rules
        # ======================================================================
        rules.append(
            BusinessRule(
                rule_id="CTX_HIGH_AMOUNT_NIGHT_TRANSFER",
                category="transaction_context",
                severity="MEDIUM",
                risk_contribution=0.20,
                is_hard_block=False,
                reason="High-value nighttime transfer or cash-out transaction",
                condition=lambda tx, feat: (
                    int(feat.get("is_night", 0)) == 1
                    and str(tx.get("payment_type", "")).upper() in ["TRANSFER", "CASH_OUT"]
                    and float(tx.get("amount", 0.0)) > 3000.0
                ),
            )
        )
        rules.append(
            BusinessRule(
                rule_id="CTX_HIGH_DRAIN_UNVERIFIED",
                category="transaction_context",
                severity="HIGH",
                risk_contribution=0.85,
                is_hard_block=False,
                reason="CRITICAL: High-value account balance drain without authentication",
                condition=lambda tx, feat: (
                    float(tx.get("origin_balance", 0.0)) > 0.0
                    and (float(tx.get("amount", 0.0)) / (float(tx.get("origin_balance", 0.0)) + 1e-5)) > 0.90
                    and (int(tx.get("auth_verified", 1)) == 0 or tx.get("auth_verified") is False)
                    and int(feat.get("tx_count_1m", 1)) < 3
                ),
            )
        )

        # ======================================================================
        # 6. Hard-Block Rules
        # ======================================================================
        rules.append(
            BusinessRule(
                rule_id="BLOCK_IMPOSSIBLE_TRAVEL",
                category="location_device",
                severity="CRITICAL",
                risk_contribution=1.0,
                is_hard_block=True,
                reason="CRITICAL: Impossible travel velocity (>800 km/h)",
                condition=lambda tx, feat: float(feat.get("speed_kmh_from_prev_tx", 0.0)) > 800.0,
            )
        )
        rules.append(
            BusinessRule(
                rule_id="BLOCK_ZERO_BALANCE_HIGH_DRAIN_NO_AUTH",
                category="amount",
                severity="CRITICAL",
                risk_contribution=1.0,
                is_hard_block=True,
                reason="CRITICAL: Zero-balance account high-value drain without authentication",
                condition=lambda tx, feat: (
                    float(tx.get("origin_balance", 0.0)) == 0.0
                    and float(tx.get("amount", 0.0)) > 10000.0
                    and (int(tx.get("auth_verified", 1)) == 0 or tx.get("auth_verified") is False)
                ),
            )
        )
        rules.append(
            BusinessRule(
                rule_id="BLOCK_UNVERIFIED_RAPID_DRAIN",
                category="transaction_context",
                severity="CRITICAL",
                risk_contribution=1.0,
                is_hard_block=True,
                reason="CRITICAL: High-value account balance drain without authentication under rapid velocity",
                condition=lambda tx, feat: (
                    float(tx.get("origin_balance", 0.0)) > 0.0
                    and (float(tx.get("amount", 0.0)) / (float(tx.get("origin_balance", 0.0)) + 1e-5)) > 0.90
                    and (int(tx.get("auth_verified", 1)) == 0 or tx.get("auth_verified") is False)
                    and int(feat.get("tx_count_1m", 1)) >= 3
                ),
            )
        )

        return rules


class DeterministicRuleEngine(BusinessRuleEngine):
    """
    Drop-in backward-compatible subclass of BusinessRuleEngine.
    Maintains legacy name and interfaces while leveraging R5.3 structured rule evaluation.
    """
    pass
