"""FinPulse R6.2 — Message Accounting & Reconciliation Ledger.

Enforces zero-silent-loss message accounting:
  total_input = successfully_processed + failed_rejected + dlq + pending
Tracks duplicate/replayed transactions independently.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Set, List


class MessageAccountingLedger:
    """
    Strict accounting ledger tracking message lifecycle and verifying zero silent message loss.
    """

    def __init__(self):
        self.input_ids: Set[str] = set()
        self.processed_ids: Set[str] = set()
        self.failed_ids: Dict[str, str] = {}
        self.dlq_ids: Dict[str, str] = {}
        self.pending_ids: Set[str] = set()
        self.duplicate_count: int = 0
        self.input_sequence: List[str] = []

    def record_input(self, tx_id: str) -> bool:
        """Record transaction offered to pipeline. Returns True if duplicate."""
        is_dup = tx_id in self.input_ids
        if is_dup:
            self.duplicate_count += 1
        else:
            self.input_ids.add(tx_id)
        self.input_sequence.append(tx_id)
        return is_dup

    def record_processed(self, tx_id: str) -> None:
        """Record transaction successfully scored and emitted."""
        self.processed_ids.add(tx_id)
        self.pending_ids.discard(tx_id)

    def record_failed(self, tx_id: str, reason: str = "") -> None:
        """Record transaction failure or rejection."""
        self.failed_ids[tx_id] = reason
        self.pending_ids.discard(tx_id)

    def record_dlq(self, tx_id: str, reason: str = "") -> None:
        """Record message routed to dead-letter queue."""
        self.dlq_ids[tx_id] = reason
        self.pending_ids.discard(tx_id)

    def record_pending(self, tx_id: str) -> None:
        """Record message remaining in buffer/queue during snapshot."""
        if tx_id not in self.processed_ids and tx_id not in self.failed_ids:
            self.pending_ids.add(tx_id)

    def reconcile(self) -> Dict[str, Any]:
        """
        Reconcile lifecycle counts and verify zero silent message loss.
        """
        unique_inputs = len(self.input_ids)
        total_inputs = len(self.input_sequence)
        processed = len(self.processed_ids)
        failed = len(self.failed_ids)
        dlq = len(self.dlq_ids)
        pending = len(self.pending_ids)

        accounted_for = processed + failed + dlq + pending
        silent_lost = unique_inputs - accounted_for

        is_balanced = (silent_lost == 0)

        return {
            "total_input_messages": total_inputs,
            "unique_input_transactions": unique_inputs,
            "duplicate_replays": self.duplicate_count,
            "successfully_processed": processed,
            "failed_rejected": failed,
            "dlq_count": dlq,
            "pending_in_queue": pending,
            "silent_lost": silent_lost,
            "is_balanced": is_balanced,
            "loss_rate_percent": (silent_lost / max(1, unique_inputs)) * 100.0,
        }
