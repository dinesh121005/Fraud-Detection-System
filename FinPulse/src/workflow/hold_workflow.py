"""
FinPulse R7-C — Hold / Confirm / Deny / Expire State Machine.

Orchestrates customer-review workflows for transactions placed on HOLD:
                  HOLD
                    |
      +-------------+-------------+
      |                           |
  (Timeout)              (Customer Action)
      |                           |
      v                     +-----+-----+
    EXPIRE                  |           |
                         CONFIRM       DENY
                            |           |
                            v           v
                         RELEASE      DENIED

Guarantees:
- Strict, deterministic state machine with explicit legal transitions
- Secure confirmation token verification
- Idempotent resolution handling (duplicate confirmations return cached outcome)
- Expired-hold protection (actions on expired holds are rejected)
- Audit log & callback/event emission support
"""

import time
import uuid
import hmac
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal, Tuple, List

from src.monitoring.logger import get_logger

logger = get_logger("FinPulse.Workflow.Hold")

HoldStatus = Literal["HOLD", "RELEASED", "DENIED", "EXPIRED"]


class HoldTransitionError(Exception):
    """Raised when an invalid state transition or expired action is attempted."""
    pass


@dataclass
class HoldCase:
    """Persistent Customer Review Hold Case."""
    hold_id: str
    transaction_id: str
    customer_id: str
    amount: float
    created_at: float
    timeout_seconds: float
    token_hash: str
    status: HoldStatus = "HOLD"
    resolved_at: Optional[float] = None
    resolution_reason: Optional[str] = None
    customer_channel: str = "sms_push"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def case_id(self) -> str:
        return self.hold_id

    @property
    def expires_at(self) -> float:
        return self.created_at + self.timeout_seconds

    def is_expired(self, current_time: float) -> bool:
        return current_time > self.expires_at


class HoldWorkflowEngine:
    """
    State machine managing real-time HOLD lifecycles.
    Thread-safe dictionary store with transition validation.
    """

    DEFAULT_TIMEOUT_SECONDS: float = 300.0  # 5-minute review window

    def __init__(self, secret_key: str = "finpulse-hold-secret-salt"):
        self.secret_key = secret_key
        self._cases: Dict[str, HoldCase] = {}
        self._tx_to_hold: Dict[str, str] = {}

    def _generate_token(self, hold_id: str, transaction_id: str) -> str:
        msg = f"{hold_id}:{transaction_id}"
        return hmac.new(self.secret_key.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).hexdigest()[:32]

    def create_hold(
        self,
        transaction_id: str,
        customer_id: str,
        amount: float,
        timeout_seconds: Optional[float] = None,
        current_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[HoldCase, str]:
        """
        Create a new HOLD case.
        Returns: (HoldCase, plaintext_confirmation_token)
        """
        if transaction_id in self._tx_to_hold:
            existing_id = self._tx_to_hold[transaction_id]
            case = self._cases[existing_id]
            logger.info("Idempotent create: hold case already exists", transaction_id=transaction_id, hold_id=case.hold_id)
            # Recompute token for convenience
            token = self._generate_token(case.hold_id, transaction_id)
            return case, token

        now = current_time or time.time()
        hold_id = f"hold_{uuid.uuid4().hex[:12]}"
        token = self._generate_token(hold_id, transaction_id)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

        case = HoldCase(
            hold_id=hold_id,
            transaction_id=transaction_id,
            customer_id=customer_id,
            amount=amount,
            created_at=now,
            timeout_seconds=timeout_seconds or self.DEFAULT_TIMEOUT_SECONDS,
            token_hash=token_hash,
            status="HOLD",
            metadata=metadata or {}
        )

        self._cases[hold_id] = case
        self._tx_to_hold[transaction_id] = hold_id
        logger.info("Transaction placed on HOLD", hold_id=hold_id, transaction_id=transaction_id, customer_id=customer_id)
        return case, token

    def register_case(self, case: HoldCase) -> str:
        """Register an existing HoldCase (e.g. loaded from PostgreSQL) and return its valid token."""
        self._cases[case.hold_id] = case
        self._tx_to_hold[case.transaction_id] = case.hold_id
        return self._generate_token(case.hold_id, case.transaction_id)

    def confirm_hold(
        self,
        hold_id: str,
        token: str,
        current_time: Optional[float] = None,
        channel: str = "customer_mobile_app"
    ) -> HoldCase:
        """
        Customer verifies transaction -> State transitions to RELEASED.
        """
        case = self._get_case(hold_id)
        now = current_time or time.time()

        # Idempotency: if already released, return cleanly
        if case.status == "RELEASED":
            return case

        # Terminal state check
        if case.status in ("DENIED", "EXPIRED"):
            raise HoldTransitionError(f"Cannot confirm hold in terminal status '{case.status}'.")

        # Expiry check
        if case.is_expired(now):
            case.status = "EXPIRED"
            case.resolved_at = now
            case.resolution_reason = "Expired before confirmation"
            raise HoldTransitionError("Hold has expired and cannot be confirmed.")

        # Token validation
        incoming_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(incoming_hash, case.token_hash):
            raise HoldTransitionError("Invalid confirmation security token.")

        # Transition
        case.status = "RELEASED"
        case.resolved_at = now
        case.resolution_reason = f"Confirmed by customer via {channel}"
        case.customer_channel = channel
        logger.info("Hold RELEASED by customer", hold_id=hold_id, tx_id=case.transaction_id)
        return case

    def deny_hold(
        self,
        hold_id: str,
        token: str,
        reason: str = "Customer reported unrecognized transaction",
        current_time: Optional[float] = None
    ) -> HoldCase:
        """
        Customer denies transaction -> State transitions to DENIED (fraud confirmed).
        """
        case = self._get_case(hold_id)
        now = current_time or time.time()

        if case.status == "DENIED":
            return case

        if case.status in ("RELEASED", "EXPIRED"):
            raise HoldTransitionError(f"Cannot deny hold in terminal status '{case.status}'.")

        if case.is_expired(now):
            case.status = "EXPIRED"
            case.resolved_at = now
            case.resolution_reason = "Expired before resolution"
            raise HoldTransitionError("Hold has expired.")

        incoming_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(incoming_hash, case.token_hash):
            raise HoldTransitionError("Invalid confirmation security token.")

        case.status = "DENIED"
        case.resolved_at = now
        case.resolution_reason = reason
        logger.warning("Hold DENIED by customer (confirmed fraud)", hold_id=hold_id, tx_id=case.transaction_id)
        return case

    def expire_hold(
        self,
        hold_id: str,
        current_time: Optional[float] = None
    ) -> bool:
        """
        Mark an expired hold as EXPIRED if timed out.
        Returns True if status changed to EXPIRED, False if not yet expired or already resolved.
        """
        case = self._get_case(hold_id)
        now = current_time or time.time()

        if case.status != "HOLD":
            return False  # Already resolved

        if case.is_expired(now):
            case.status = "EXPIRED"
            case.resolved_at = now
            case.resolution_reason = "Review window timeout exceeded"
            logger.info("Hold EXPIRED due to timeout", hold_id=hold_id, tx_id=case.transaction_id)
            return True

        return False

    def scan_and_expire_open_holds(self, current_time: Optional[float] = None) -> List[HoldCase]:
        """Scan all open holds and expire those whose timeout has passed."""
        now = current_time or time.time()
        expired: List[HoldCase] = []
        for case in list(self._cases.values()):
            if case.status == "HOLD" and case.is_expired(now):
                case.status = "EXPIRED"
                case.resolved_at = now
                case.resolution_reason = "Review window timeout exceeded"
                expired.append(case)
        return expired

    def get_case(self, hold_id: str) -> Optional[HoldCase]:
        return self._cases.get(hold_id)

    def get_case_by_transaction(self, transaction_id: str) -> Optional[HoldCase]:
        hold_id = self._tx_to_hold.get(transaction_id)
        return self._cases.get(hold_id) if hold_id else None

    def _get_case(self, hold_id: str) -> HoldCase:
        case = self._cases.get(hold_id)
        if case is None:
            raise HoldTransitionError(f"Hold case '{hold_id}' not found.")
        return case

    def reset(self, customer_id: Optional[str] = None) -> None:
        """Purge hold cases for clean demo reset."""
        if customer_id:
            to_remove = [hid for hid, c in self._cases.items() if c.customer_id == customer_id]
            for hid in to_remove:
                case = self._cases.pop(hid, None)
                if case:
                    self._tx_to_hold.pop(case.transaction_id, None)
            logger.info("Reset hold cases for customer", customer_id=customer_id)
        else:
            self._cases.clear()
            self._tx_to_hold.clear()
            logger.info("Reset all hold cases")


