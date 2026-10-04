"""
FinPulse Security Case Domain Model (D4, D5).

Represents a formal customer-reported fraud case resulting from:
- Customer explicitly reporting an unauthorized transaction
- Missed / novel attack patterns entering the controlled feedback loop
- Confirmed ground-truth fraud labels linked to payment gateway decisions
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Literal


@dataclass
class SecurityCase:
    """Canonical Security Case record."""
    case_id: str
    transaction_id: str
    customer_id: str
    gateway_decision: str
    customer_report: str  # "UNAUTHORIZED", "DISPUTED", "FRAUD"
    confirmed_label: str  # "FRAUD", "LEGITIMATE"
    attack_type: str      # "UNKNOWN", "ATO", "VELOCITY", "BEHAVIORAL", etc.
    status: str = "CONFIRMED_FRAUD"  # "OPEN", "INVESTIGATING", "CONFIRMED_FRAUD", "RESOLVED"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d
