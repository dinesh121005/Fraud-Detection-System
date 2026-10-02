"""
FinPulse R7-A — Account Security Events & Account Takeover (ATO) Protection Engine.

Evaluates high-risk identity and credential lifecycle events:
- Login anomalies (rapid multi-IP login, unusual geolocations)
- Password changes / resets
- Device changes & new device registrations
- Profile / contact information modifications (email, phone changes)
- MFA / 2FA resets or downgrades

Provides:
- In-memory / Redis-backed sliding window for account-level security events
- ATO risk score aggregation in [0.0, 1.0] with compounding trigger logic
- Idempotent event ingestion preventing duplicate event counting
- Audit log & correlation context generation
- Guardrail checks integrating seamlessly with transaction workflows
"""

import time
import uuid
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Literal, Set, Tuple

from src.monitoring.logger import get_logger
from src.monitoring.metrics import record_error

logger = get_logger("FinPulse.Workflow.ATO")

AccountEventType = Literal[
    "login_anomaly",
    "password_change",
    "device_change",
    "new_device_registration",
    "profile_update",
    "mfa_reset",
    "credential_stuffing_attempt"
]

ATOAction = Literal["ALLOW", "CHALLENGE_MFA", "SUSPEND_ACCOUNT", "TRIGGER_HOLD"]


@dataclass(frozen=True)
class AccountSecurityEvent:
    """Canonical Account Security Event record."""
    event_id: str
    customer_id: str
    event_type: AccountEventType
    timestamp: float
    ip_address: str = "127.0.0.1"
    device_id: str = "dev_unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        customer_id: str,
        event_type: AccountEventType,
        timestamp: Optional[float] = None,
        ip_address: str = "127.0.0.1",
        device_id: str = "dev_unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> "AccountSecurityEvent":
        ts = timestamp or time.time()
        # Deterministic event ID based on customer, type, timestamp
        raw_key = f"{customer_id}:{event_type}:{ts}:{ip_address}:{device_id}"
        event_id = f"ato_evt_{hashlib.sha256(raw_key.encode('utf-8')).hexdigest()[:16]}"
        return cls(
            event_id=event_id,
            customer_id=customer_id,
            event_type=event_type,
            timestamp=ts,
            ip_address=ip_address,
            device_id=device_id,
            metadata=metadata or {}
        )


@dataclass(frozen=True)
class ATORiskAssessment:
    """Outcome of ATO evaluation for an account."""
    customer_id: str
    ato_risk_score: float  # [0.0, 1.0]
    action: ATOAction
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    triggered_indicators: List[str]
    recent_events_count: int
    requires_transaction_hold: bool


class ATOProtectionEngine:
    """
    Stateful ATO evaluation engine with sliding lookback windows and compounding rules.
    Maintains idempotency and provides transaction guardrails.
    """

    # Event severity baselines in [0.0, 1.0]
    EVENT_SEVERITIES: Dict[AccountEventType, float] = {
        "mfa_reset": 0.50,
        "password_change": 0.40,
        "new_device_registration": 0.35,
        "login_anomaly": 0.30,
        "profile_update": 0.25,
        "device_change": 0.20,
        "credential_stuffing_attempt": 0.60
    }

    def __init__(self, lookback_window_seconds: float = 86400.0):
        self.lookback_window = lookback_window_seconds
        # In-memory store: customer_id -> list of AccountSecurityEvent
        self._history: Dict[str, List[AccountSecurityEvent]] = {}
        # Ingestion idempotency cache
        self._seen_event_ids: Set[str] = set()

    def record_event(self, event: AccountSecurityEvent) -> bool:
        """
        Record an account event idempotently.
        Returns True if new, False if already ingested.
        """
        if event.event_id in self._seen_event_ids:
            logger.info("Duplicate account event ignored", event_id=event.event_id, customer_id=event.customer_id)
            return False

        self._seen_event_ids.add(event.event_id)
        if event.customer_id not in self._history:
            self._history[event.customer_id] = []
        self._history[event.customer_id].append(event)
        return True

    def get_recent_events(self, customer_id: str, current_time: float) -> List[AccountSecurityEvent]:
        """Fetch valid prior events within the lookback window."""
        events = self._history.get(customer_id, [])
        cutoff = current_time - self.lookback_window
        # Strictly chronological and within lookback
        valid = [e for e in events if cutoff <= e.timestamp <= current_time]
        valid.sort(key=lambda x: x.timestamp)
        return valid

    def evaluate_ato_risk(self, customer_id: str, current_time: Optional[float] = None) -> ATORiskAssessment:
        """
        Evaluate account takeover risk for a customer based on historical event patterns.
        Compounding rules:
        - Critical velocity: >= 3 security events in 24h
        - Password change followed by new device registration within 1 hour
        - MFA reset combined with profile update
        """
        now = current_time or time.time()
        recent = self.get_recent_events(customer_id, now)

        if not recent:
            return ATORiskAssessment(
                customer_id=customer_id,
                ato_risk_score=0.0,
                action="ALLOW",
                risk_level="LOW",
                triggered_indicators=[],
                recent_events_count=0,
                requires_transaction_hold=False
            )

        indicators: List[str] = []
        base_score = 0.0

        types_present = [e.event_type for e in recent]
        for e in recent:
            base_score = max(base_score, self.EVENT_SEVERITIES.get(e.event_type, 0.1))

        # Compounding Rule 1: High velocity of sensitive changes
        if len(recent) >= 3:
            base_score += 0.25
            indicators.append(f"Rapid security event cluster ({len(recent)} events in 24h)")

        # Compounding Rule 2: Password change + New device within 1 hour
        pw_events = [e for e in recent if e.event_type == "password_change"]
        dev_events = [e for e in recent if e.event_type in ("new_device_registration", "device_change")]
        if pw_events and dev_events:
            for p in pw_events:
                for d in dev_events:
                    if 0 <= (d.timestamp - p.timestamp) <= 3600:
                        base_score += 0.35
                        indicators.append("Password reset immediately followed by new device registration (<1h)")
                        break

        # Compounding Rule 3: MFA Reset + Profile Update
        if "mfa_reset" in types_present and "profile_update" in types_present:
            base_score += 0.30
            indicators.append("MFA reset accompanied by contact profile modification")

        # Compounding Rule 4: Credential stuffing attempt
        if "credential_stuffing_attempt" in types_present:
            base_score += 0.40
            indicators.append("Prior credential stuffing / automated brute force detected")

        final_score = min(1.0, round(base_score, 4))

        if final_score >= 0.80:
            action: ATOAction = "SUSPEND_ACCOUNT"
            risk_level = "CRITICAL"
            req_hold = True
        elif final_score >= 0.50:
            action: ATOAction = "TRIGGER_HOLD"
            risk_level = "HIGH"
            req_hold = True
        elif final_score >= 0.25:
            action: ATOAction = "CHALLENGE_MFA"
            risk_level = "MEDIUM"
            req_hold = False
        else:
            action: ATOAction = "ALLOW"
            risk_level = "LOW"
            req_hold = False

        return ATORiskAssessment(
            customer_id=customer_id,
            ato_risk_score=final_score,
            action=action,
            risk_level=risk_level,
            triggered_indicators=indicators,
            recent_events_count=len(recent),
            requires_transaction_hold=req_hold
        )

    def check_transaction_guardrails(self, customer_id: str, transaction_amount: float) -> Tuple[bool, Optional[str]]:
        """
        Pre-flight check for financial transactions:
        Returns (is_allowed, optional_block_reason).
        """
        assessment = self.evaluate_ato_risk(customer_id)
        if assessment.action == "SUSPEND_ACCOUNT":
            return False, f"Account suspended due to critical ATO risk ({assessment.ato_risk_score:.2f})"
        if assessment.requires_transaction_hold and transaction_amount > 500.0:
            return False, f"Transaction held: High ATO risk score ({assessment.ato_risk_score:.2f}) on amount ${transaction_amount:.2f}"
        return True, None
