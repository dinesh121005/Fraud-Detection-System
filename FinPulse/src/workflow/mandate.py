"""
FinPulse R7-B — Mandate Registry & Mandate Protection Engine.

Manages recurring debit, standing order, and automated billing mandates:
- Mandate lifecycle: CREATE -> ACTIVATE -> UPDATE -> PAUSE -> CANCEL -> EXPIRE -> EXECUTE
- Enforces strict frequency and amount bounds on automated execution
- Prevents out-of-schedule, unauthorized, or excessive mandate debits
- Emits auditable events and integrates with fraud risk evaluation
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Literal, Tuple

from src.monitoring.logger import get_logger

logger = get_logger("FinPulse.Workflow.Mandate")

MandateStatus = Literal["PENDING", "ACTIVE", "PAUSED", "CANCELLED", "EXPIRED"]
MandateFrequency = Literal["DAILY", "WEEKLY", "MONTHLY", "ONCE"]


@dataclass
class MandateRecord:
    """Persistent Mandate Specification."""
    mandate_id: str
    customer_id: str
    beneficiary_id: str
    max_amount: float
    frequency: MandateFrequency
    created_at: float
    expires_at: float
    status: MandateStatus = "PENDING"
    last_executed_at: Optional[float] = None
    execution_count: int = 0
    total_executed_amount: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, current_time: float) -> bool:
        return current_time > self.expires_at


class MandateProtectionError(Exception):
    """Raised on invalid mandate transitions or violation of mandate constraints."""
    pass


class MandateRegistry:
    """
    Registry and protection engine governing automated recurring mandates.
    Enforces idempotent operations, schedule rules, and fraud ceilings.
    """

    FREQUENCY_INTERVALS: Dict[MandateFrequency, float] = {
        "DAILY": 86400.0,
        "WEEKLY": 7 * 86400.0,
        "MONTHLY": 30 * 86400.0,
        "ONCE": 0.0
    }

    def __init__(self):
        self._mandates: Dict[str, MandateRecord] = {}

    def create_mandate(
        self,
        mandate_id: str,
        customer_id: str,
        beneficiary_id: str,
        max_amount: float,
        frequency: MandateFrequency,
        duration_days: float = 365.0,
        current_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MandateRecord:
        """Create a new mandate in PENDING state."""
        if mandate_id in self._mandates:
            logger.info("Idempotent create: mandate already exists", mandate_id=mandate_id)
            return self._mandates[mandate_id]

        if max_amount <= 0.0:
            raise MandateProtectionError("Mandate max_amount must be greater than zero.")

        now = current_time or time.time()
        record = MandateRecord(
            mandate_id=mandate_id,
            customer_id=customer_id,
            beneficiary_id=beneficiary_id,
            max_amount=max_amount,
            frequency=frequency,
            created_at=now,
            expires_at=now + (duration_days * 86400.0),
            status="PENDING",
            metadata=metadata or {}
        )
        self._mandates[mandate_id] = record
        logger.info("Mandate created", mandate_id=mandate_id, customer_id=customer_id)
        return record

    def activate_mandate(self, mandate_id: str) -> MandateRecord:
        mandate = self._get_mandate(mandate_id)
        if mandate.status not in ("PENDING", "PAUSED"):
            raise MandateProtectionError(f"Cannot activate mandate in status '{mandate.status}'.")
        mandate.status = "ACTIVE"
        logger.info("Mandate activated", mandate_id=mandate_id)
        return mandate

    def pause_mandate(self, mandate_id: str, reason: str = "Customer request") -> MandateRecord:
        mandate = self._get_mandate(mandate_id)
        if mandate.status != "ACTIVE":
            raise MandateProtectionError(f"Cannot pause mandate in status '{mandate.status}'.")
        mandate.status = "PAUSED"
        mandate.metadata["pause_reason"] = reason
        logger.info("Mandate paused", mandate_id=mandate_id, reason=reason)
        return mandate

    def cancel_mandate(self, mandate_id: str, reason: str = "Revocation") -> MandateRecord:
        mandate = self._get_mandate(mandate_id)
        if mandate.status == "CANCELLED":
            return mandate  # Idempotent
        mandate.status = "CANCELLED"
        mandate.metadata["cancel_reason"] = reason
        logger.info("Mandate cancelled", mandate_id=mandate_id, reason=reason)
        return mandate

    def update_mandate_limit(self, mandate_id: str, new_max_amount: float) -> MandateRecord:
        mandate = self._get_mandate(mandate_id)
        if mandate.status not in ("PENDING", "ACTIVE"):
            raise MandateProtectionError(f"Cannot update limit for mandate in status '{mandate.status}'.")
        if new_max_amount <= 0.0:
            raise MandateProtectionError("New max amount must be strictly positive.")
        mandate.max_amount = new_max_amount
        return mandate

    def validate_and_execute(
        self,
        mandate_id: str,
        amount: float,
        current_time: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate incoming automated execution against mandate constraints.
        Enforces:
        - Active status
        - Non-expired validity
        - Amount <= max_amount
        - Minimum frequency elapsed between successive executions
        Returns: (success, reason_if_failed)
        """
        mandate = self._get_mandate(mandate_id)
        now = current_time or time.time()

        # 1. Expiration check
        if mandate.is_expired(now):
            mandate.status = "EXPIRED"
            return False, f"Mandate '{mandate_id}' has expired."

        # 2. Status check
        if mandate.status != "ACTIVE":
            return False, f"Mandate '{mandate_id}' is not ACTIVE (current: {mandate.status})."

        # 3. Amount ceiling check
        if amount > mandate.max_amount:
            return False, f"Execution amount ${amount:.2f} exceeds approved ceiling ${mandate.max_amount:.2f}."

        # 4. Single-use check
        if mandate.frequency == "ONCE" and mandate.execution_count >= 1:
            return False, f"Single-use mandate '{mandate_id}' has already been executed."

        # 5. Frequency interval cadence check
        min_interval = self.FREQUENCY_INTERVALS.get(mandate.frequency, 0.0)
        if min_interval > 0.0 and mandate.last_executed_at is not None:
            elapsed = now - mandate.last_executed_at
            # Allow 5% clock skew tolerance
            if elapsed < (min_interval * 0.95):
                return False, f"Execution too early for {mandate.frequency} mandate. {elapsed:.0f}s elapsed, {min_interval:.0f}s required."

        # Execution approved - record state
        mandate.last_executed_at = now
        mandate.execution_count += 1
        mandate.total_executed_amount += amount

        if mandate.frequency == "ONCE":
            mandate.status = "EXPIRED"

        logger.info("Mandate executed successfully", mandate_id=mandate_id, amount=amount, execution_count=mandate.execution_count)
        return True, None

    def get_mandate(self, mandate_id: str) -> Optional[MandateRecord]:
        return self._mandates.get(mandate_id)

    def _get_mandate(self, mandate_id: str) -> MandateRecord:
        mandate = self._mandates.get(mandate_id)
        if mandate is None:
            raise MandateProtectionError(f"Mandate '{mandate_id}' not found.")
        return mandate
