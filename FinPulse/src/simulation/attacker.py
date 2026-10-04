"""
FinPulse Dynamic Adversarial Attack Runner & Simulation Engine (D2, D3, U1, U2).

Provides a stateful, external adversarial simulation environment:
- Persistent AttackerState abstraction per adversary (ATK-7F31, ATK-92B4, ATK-C812, ATK-44D9, ATK-A71F, ATK-5E20, ATK-U101, ATK-U202)
- Multi-Attacker concurrent tracking & transitions (device, location, recipient, amount, timing)
- Controlled Dynamic Variation: Strategy parameter constraints + Seeded Deterministic Mode / Live Mode
- Formal Template Registry & Classification: Known Predefined Templates vs Unseen Scenarios (U1, U2)
- U1 — Context Mutation (Non-matching single template mutation, moderate context)
- U2 — Compound Novel Attack (Multi-signal moderate compound anomaly)
- Dynamic Adversarial Adaptation: Reacts to real-time FinPulse outcomes (BLOCK / HOLD / APPROVE)
- Complete isolation: External adversarial agent that changes transaction/security context only,
  leaving FinPulse core model, calibrator, 32-feature pipeline, and R5 weights frozen.
"""

import time
import uuid
import math
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Literal

from src.workflow.account_events import (
    ATOProtectionEngine,
    AccountSecurityEvent,
    ATORiskAssessment,
    AccountEventType
)
from src.persistence.sink import IdempotentEventSink
from src.simulation.demo_account import (
    DemoAccountManager,
    DEFAULT_DEMO_CUSTOMER_ID,
    DEFAULT_ATTACKER_RECEIVER_ID,
    KNOWN_LOCATION,
    KNOWN_DEVICE_ID
)
from src.monitoring.logger import get_logger

logger = get_logger("FinPulse.Simulation.Attacker")

# Canonical default constants for backward compatibility
ATTACKER_DEVICE_ID = "DEV_ATTACKER_01"
ATTACKER_LOCATION = {
    "city": "MUMBAI",
    "latitude": 19.0760,
    "longitude": 72.8777
}

# -----------------------------------------------------------------------------
# CITIES & GEOGRAPHIC REPOSITORY
# -----------------------------------------------------------------------------
INDIAN_CITIES = {
    "CHENNAI": {"city": "CHENNAI", "latitude": 13.0827, "longitude": 80.2707, "label": "Chennai Central (Home)"},
    "MUMBAI": {"city": "MUMBAI", "latitude": 19.0760, "longitude": 72.8777, "label": "Mumbai Core (1,030 km)"},
    "BENGALURU": {"city": "BENGALURU", "latitude": 12.9716, "longitude": 77.5946, "label": "Bengaluru Tech Hub (290 km)"},
    "DELHI": {"city": "DELHI", "latitude": 28.6139, "longitude": 77.2090, "label": "Delhi NCR (1,750 km)"},
    "KOLKATA": {"city": "KOLKATA", "latitude": 22.5726, "longitude": 88.3639, "label": "Kolkata East (1,360 km)"},
    "HYDERABAD": {"city": "HYDERABAD", "latitude": 17.3850, "longitude": 78.4867, "label": "Hyderabad Deccan (510 km)"},
    "PUNE": {"city": "PUNE", "latitude": 18.5204, "longitude": 73.8567, "label": "Pune Tech Hub (915 km)"}
}

def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two GPS coordinates in kilometers."""
    r = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(r * c, 1)

# -----------------------------------------------------------------------------
# DIVERSIFIED RECIPIENTS REGISTRY
# -----------------------------------------------------------------------------
ATTACK_RECIPIENTS = {
    "REC_MULE_01": {"id": "REC_MULE_01", "name": "Shadow Mule Corp (Mumbai)", "type": "HIGH_RISK_RECIPIENT", "city": "MUMBAI"},
    "REC_CRYPTO_02": {"id": "REC_CRYPTO_02", "name": "Global P2P Crypto Desk", "type": "HIGH_RISK_RECIPIENT", "city": "DELHI"},
    "REC_OFFSHORE_03": {"id": "REC_OFFSHORE_03", "name": "Unregistered Digital Merchant", "type": "NEW_RECIPIENT", "city": "BENGALURU"},
    "REC_PEER_04": {"id": "REC_PEER_04", "name": "FastCash Peer Remit", "type": "RARE_RECIPIENT", "city": "KOLKATA"},
    "REC_PEER_07": {"id": "REC_PEER_07", "name": "Novel P2P Direct Remit", "type": "NEW_UNINDEXED_RECIPIENT", "city": "HYDERABAD"},
    "REC_VENDOR_09": {"id": "REC_VENDOR_09", "name": "Apex Cloud Infrastructure", "type": "MERCHANT_NEW", "city": "PUNE"},
    "CUST_ATTACKER": {"id": "CUST_ATTACKER", "name": "Attacker Receiver (Primary)", "type": "HIGH_RISK_RECIPIENT", "city": "MUMBAI"}
}

# -----------------------------------------------------------------------------
# PERSISTENT ATTACKER STATE ABSTRACTION
# -----------------------------------------------------------------------------
@dataclass
class AttackerState:
    """
    Persistent state abstraction tracking an external adversarial entity across multiple attempts.
    Maintains independent hardware fingerprints, locations, recipients, and adaptation history.
    """
    actor_id: str
    alias: str
    archetype: str
    objective: str
    current_device: str
    previous_device: Optional[str] = None
    device_history: List[str] = field(default_factory=list)
    current_location: Dict[str, Any] = field(default_factory=lambda: dict(INDIAN_CITIES["MUMBAI"]))
    previous_location: Optional[Dict[str, Any]] = None
    location_history: List[Dict[str, Any]] = field(default_factory=list)
    current_recipient: str = "REC_MULE_01"
    recipient_history: List[str] = field(default_factory=list)
    current_amount: float = 8500.0
    amount_history: List[float] = field(default_factory=list)
    attack_stage: int = 1
    strategy: str = "account_takeover"
    previous_decision: Optional[str] = None
    previous_status: Optional[str] = None
    attempt_count: int = 0
    ip_address: str = "198.51.100.42"
    history: List[Dict[str, Any]] = field(default_factory=list)

    def record_attempt(
        self,
        transaction_id: str,
        amount: float,
        device_id: str,
        location: Dict[str, Any],
        recipient_id: str,
        strategy: str,
        timestamp: float
    ) -> None:
        """Records a new attack attempt and transitions state variables."""
        self.attempt_count += 1
        self.attack_stage += 1
        
        # Track device transitions
        if self.current_device != device_id:
            self.previous_device = self.current_device
            self.current_device = device_id
        if device_id not in self.device_history:
            self.device_history.append(device_id)

        # Track location transitions
        if self.current_location.get("city") != location.get("city"):
            self.previous_location = dict(self.current_location)
            self.current_location = dict(location)
        if location not in self.location_history:
            self.location_history.append(location)

        # Track recipient transitions
        if self.current_recipient != recipient_id:
            self.current_recipient = recipient_id
        if recipient_id not in self.recipient_history:
            self.recipient_history.append(recipient_id)

        # Track amount
        self.current_amount = float(amount)
        self.amount_history.append(float(amount))
        self.strategy = strategy

        self.history.insert(0, {
            "attempt": self.attempt_count,
            "transaction_id": transaction_id,
            "amount": amount,
            "device_id": device_id,
            "location": location,
            "recipient_id": recipient_id,
            "strategy": strategy,
            "timestamp": timestamp,
            "decision": None
        })

    def record_outcome(self, decision: str, status: Optional[str] = None) -> None:
        """Updates the state with the actual authoritative decision from FinPulse."""
        self.previous_decision = decision
        self.previous_status = status or decision
        if self.history:
            self.history[0]["decision"] = decision
            self.history[0]["status"] = status or decision

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# ADVERSARY PROFILES & POOL
# -----------------------------------------------------------------------------
class AttackerProfile:
    """Represents a distinct adversarial entity archetype and baseline configuration."""

    def __init__(
        self,
        actor_id: str,
        name: str,
        archetype: str,
        primary_device: str,
        primary_city: str,
        preferred_strategy: str,
        risk_intent: str,
        description: str,
        ip_address: str = "198.51.100.42"
    ):
        self.actor_id = actor_id
        self.name = name
        self.archetype = archetype
        self.primary_device = primary_device
        self.primary_city = primary_city
        self.location = INDIAN_CITIES.get(primary_city, INDIAN_CITIES["MUMBAI"])
        self.preferred_strategy = preferred_strategy
        self.risk_intent = risk_intent
        self.description = description
        self.ip_address = ip_address

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "name": self.name,
            "archetype": self.archetype,
            "primary_device": self.primary_device,
            "primary_city": self.primary_city,
            "location": self.location,
            "preferred_strategy": self.preferred_strategy,
            "risk_intent": self.risk_intent,
            "description": self.description,
            "ip_address": self.ip_address
        }

ATTACKER_POOL: Dict[str, AttackerProfile] = {
    "ATK-7F31": AttackerProfile(
        actor_id="ATK-7F31",
        name="Morpheus",
        archetype="Credential Thief (ATO)",
        primary_device="DEV_ATTACKER_01",
        primary_city="MUMBAI",
        preferred_strategy="account_takeover",
        risk_intent="CRITICAL_DRAIN",
        description="Compromises victim credentials, changes passwords, rotates hardware to Mumbai, attempts immediate ₹8,500 balance drain.",
        ip_address="198.51.100.42"
    ),
    "ATK-92B4": AttackerProfile(
        actor_id="ATK-92B4",
        name="Hydra",
        archetype="Stolen Session / Device Roamer",
        primary_device="DEV-ATK-4408",
        primary_city="BENGALURU",
        preferred_strategy="new_device_hold",
        risk_intent="STEALTH_VERIFICATION_PROBE",
        description="Operates from nearby tech hub using stolen session tokens and unregistered mobile hardware to test verification friction.",
        ip_address="203.0.113.19"
    ),
    "ATK-C812": AttackerProfile(
        actor_id="ATK-C812",
        name="Phantom",
        archetype="Burst Velocity Drainer",
        primary_device="DEV-ATK-7129",
        primary_city="DELHI",
        preferred_strategy="velocity_surge",
        risk_intent="RAPID_ACCUMULATION",
        description="Fires rapid burst payments (₹500 ➔ ₹700 ➔ ₹900 ➔ ₹1,200) within 60s to evade static per-transaction thresholds.",
        ip_address="198.51.100.88"
    ),
    "ATK-44D9": AttackerProfile(
        actor_id="ATK-44D9",
        name="Vortex",
        archetype="Impossible Travel Roamer",
        primary_device="DEV_ROAMING_X",
        primary_city="DELHI",
        preferred_strategy="impossible_travel",
        risk_intent="PHYSICAL_ANOMALY",
        description="Transits 1,750 km across Indian metropolitan nodes in under 2 minutes at 03:00 AM off-hours.",
        ip_address="198.51.100.77"
    ),
    "ATK-A71F": AttackerProfile(
        actor_id="ATK-A71F",
        name="Specter",
        archetype="Low-and-Slow Stealth Evasion",
        primary_device="DEV_KNOWN_01",
        primary_city="CHENNAI",
        preferred_strategy="low_and_slow",
        risk_intent="SUB_THRESHOLD_PROBING",
        description="Executes subtle sub-₹800 micro-transfers with staggered timing directly inside the victim's home city.",
        ip_address="192.0.2.14"
    ),
    "ATK-5E20": AttackerProfile(
        actor_id="ATK-5E20",
        name="Cipher",
        archetype="Novel SIM-Swap / Edge Exploit",
        primary_device="DEV_EMULATOR_NOVEL",
        primary_city="KOLKATA",
        preferred_strategy="novel_attack",
        risk_intent="ZERO_DAY_FEEDBACK_TARGET",
        description="Subtle novel peer payment with valid token to test the post-authorization dispute and ground-truth retraining feedback loop.",
        ip_address="198.51.100.99"
    ),
    "ATK-U101": AttackerProfile(
        actor_id="ATK-U101",
        name="Chameleon",
        archetype="Unseen Context Mutator (U1)",
        primary_device="DEV-MUT-8842",
        primary_city="HYDERABAD",
        preferred_strategy="u1_context_mutation",
        risk_intent="UNSEEN_MUTATION_PROBE",
        description="Deliberately mutates context (moderate amount, feasible location change, valid auth) to avoid matching predefined attack templates.",
        ip_address="198.51.100.111"
    ),
    "ATK-U202": AttackerProfile(
        actor_id="ATK-U202",
        name="Chimera",
        archetype="Compound Novel Adversary (U2)",
        primary_device="DEV-COMPOUND-77",
        primary_city="PUNE",
        preferred_strategy="u2_compound_novel",
        risk_intent="MULTI_SIGNAL_COMPOUND",
        description="Synthesizes multiple moderate contextual signals without any single extreme trigger to evaluate multi-layer hybrid fusion.",
        ip_address="198.51.100.222"
    )
}

# -----------------------------------------------------------------------------
# STRATEGY TEMPLATES REGISTRY & FORMAL CLASSIFICATION
# -----------------------------------------------------------------------------
class AttackStrategy:
    """Defines an adversarial attack strategy and parameter distributions."""

    def __init__(
        self,
        strategy_id: str,
        name: str,
        category: Literal["KNOWN", "UNSEEN"],
        objective: str,
        amount_range: Tuple[float, float],
        default_amount: float,
        timing_profile: str,
        description: str,
        is_predefined: bool = True
    ):
        self.strategy_id = strategy_id
        self.name = name
        self.category = category
        self.objective = objective
        self.amount_range = amount_range
        self.default_amount = default_amount
        self.timing_profile = timing_profile
        self.description = description
        self.is_predefined = is_predefined

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "category": self.category,
            "objective": self.objective,
            "amount_range": self.amount_range,
            "default_amount": self.default_amount,
            "timing_profile": self.timing_profile,
            "description": self.description,
            "is_predefined": self.is_predefined
        }

# Predefined known templates & distinct unseen scenarios
ATTACK_STRATEGIES: Dict[str, AttackStrategy] = {
    # 1. Smash & Grab (ATO)
    "account_takeover": AttackStrategy(
        strategy_id="account_takeover",
        name="Account Takeover (Smash & Grab)",
        category="KNOWN",
        objective="FULL_BALANCE_DRAIN",
        amount_range=(5000.0, 9500.0),
        default_amount=8500.0,
        timing_profile="Immediate post-credential reset",
        description="Full ATO compromise: password changed, unrecognized device in distant city, failed 2FA, large single drain attempt.",
        is_predefined=True
    ),
    # 2. Velocity Surge
    "velocity_surge": AttackStrategy(
        strategy_id="velocity_surge",
        name="Rapid Velocity Drain Surge",
        category="KNOWN",
        objective="BURST_ACCUMULATION",
        amount_range=(400.0, 2000.0),
        default_amount=1200.0,
        timing_profile="Rapid 10-30s intervals",
        description="Rapid successive transfers escalating in size to overwhelm sliding window accumulators.",
        is_predefined=True
    ),
    # 3. Low and Slow
    "low_and_slow": AttackStrategy(
        strategy_id="low_and_slow",
        name="Low-and-Slow Stealth Evasion",
        category="KNOWN",
        objective="SUB_RADAR_PROBE",
        amount_range=(250.0, 750.0),
        default_amount=450.0,
        timing_profile="Normal daytime intervals",
        description="Low-value transfers staying beneath single-transaction alerting thresholds while testing recipient acceptance.",
        is_predefined=True
    ),
    # 4. Impossible Travel
    "impossible_travel": AttackStrategy(
        strategy_id="impossible_travel",
        name="Impossible Travel Speed Jump",
        category="KNOWN",
        objective="GEOGRAPHIC_HOP",
        amount_range=(1500.0, 6500.0),
        default_amount=4200.0,
        timing_profile="03:00 AM Off-Hours Night",
        description="Hop 1,030–1,750 km across Indian metropolitan hubs at 2,400+ km/h speed with abnormal nighttime execution.",
        is_predefined=True
    ),
    # 5. New Device Hold
    "new_device_hold": AttackStrategy(
        strategy_id="new_device_hold",
        name="New Device Probe (Hold Flow)",
        category="KNOWN",
        objective="VERIFICATION_TEST",
        amount_range=(1000.0, 3000.0),
        default_amount=2500.0,
        timing_profile="Standard business hours",
        description="Home city location with unrecognized device fingerprint, targeting Medium risk to trigger a 2-way verification HOLD.",
        is_predefined=True
    ),
    # 6. Novel SIM-Swap
    "novel_attack": AttackStrategy(
        strategy_id="novel_attack",
        name="Novel SIM-Swap / Edge Pattern",
        category="KNOWN",
        objective="ZERO_DAY_EVASION",
        amount_range=(2000.0, 4500.0),
        default_amount=3500.0,
        timing_profile="Daytime peer transfer",
        description="Subtle novel peer transfer with valid token to evaluate post-settlement customer dispute & retraining feedback loop.",
        is_predefined=True
    ),
    # 7. U1 — Context Mutation (Unseen)
    "u1_context_mutation": AttackStrategy(
        strategy_id="u1_context_mutation",
        name="U1 — Context Mutation",
        category="UNSEEN",
        objective="NON_MATCHING_CONTEXT_PROBE",
        amount_range=(1800.0, 3200.0),
        default_amount=2450.0,
        timing_profile="Unusual off-peak hour (23:45 / 05:15)",
        description="Deliberately mutates observable context: moderate amount, new device, feasible regional distance (510 km), new recipient, valid auth. Not one of the predefined attack templates.",
        is_predefined=False
    ),
    # 8. U2 — Compound Novel Attack (Unseen)
    "u2_compound_novel": AttackStrategy(
        strategy_id="u2_compound_novel",
        name="U2 — Compound Novel Attack",
        category="UNSEEN",
        objective="COMPOUND_MODERATE_SIGNALS",
        amount_range=(2200.0, 3800.0),
        default_amount=2750.0,
        timing_profile="Late night off-hours (02:41 AM)",
        description="Combines multiple moderate anomalies (new device + new recipient + off-hours + moderate distance + slightly elevated amount + small velocity) without any single extreme trigger. Not one of the predefined attack templates.",
        is_predefined=False
    )
}

PREDEFINED_TEMPLATE_IDS = {
    k for k, v in ATTACK_STRATEGIES.items() if v.is_predefined
}

def is_predefined_template(strategy_id: str) -> bool:
    """
    Formal template registry check.
    Returns True if strategy_id matches a predefined known attack template.
    Returns False for unseen variations (e.g. U1 Context Mutation, U2 Compound Novel).
    """
    norm = strategy_id.lower()
    if "u1" in norm or "context_mutation" in norm or "u2" in norm or "compound" in norm:
        return False
    return norm in PREDEFINED_TEMPLATE_IDS or any(k in norm for k in ["takeover", "ato", "velocity", "slow", "travel", "hold", "sim_swap"])

def get_template_classification(strategy_id: str) -> str:
    """Returns substantiable template classification string."""
    if is_predefined_template(strategy_id):
        return "Predefined Attack Template"
    return "Not one of the predefined attack templates"


# -----------------------------------------------------------------------------
# DYNAMIC ATTACKER SIMULATION RUNNER
# -----------------------------------------------------------------------------
class AttackerSimulator:
    """
    Dynamic Attacker Simulator & External Adversarial Runner for FinPulse.
    
    Features:
    1. Stateful AttackerState per adversary with history & transitions.
    2. Controlled Dynamic Variation (Seedable RNG + Strategy Constraints).
    3. Known Strategy Library (ATO, Velocity, Low & Slow, Travel, Device, SIM-Swap).
    4. Unseen Attack Runner:
       - U1: Context Mutation
       - U2: Compound Novel Attack
    5. Adversarial Adaptation: Reacts to actual FinPulse verdicts (BLOCK, HOLD, APPROVE).
    6. Seeded Reproducibility (Demo Mode: 42 / Live Variation Mode).
    7. Clean State Reset mechanism.
    """

    def __init__(
        self,
        ato_engine: Optional[ATOProtectionEngine] = None,
        event_sink: Optional[IdempotentEventSink] = None,
        account_manager: Optional[DemoAccountManager] = None,
        seed: Optional[int] = 42,
        mode: Literal["DEMO_SEEDED", "LIVE_VARIATION"] = "DEMO_SEEDED"
    ):
        self.ato_engine = ato_engine or ATOProtectionEngine()
        self.sink = event_sink or IdempotentEventSink()
        self.account_manager = account_manager or DemoAccountManager()

        # Seed & Randomness state
        self.seed = seed
        self.mode = mode
        self.rng = random.Random(seed if mode == "DEMO_SEEDED" else None)

        # Persistent Multi-Attacker States store
        self.attacker_states: Dict[str, AttackerState] = {}
        self.active_actor_id: str = "ATK-7F31"
        self.active_strategy_id: str = "account_takeover"

        # Global logs and legacy compatibility mirrors
        self.adaptation_log: List[Dict[str, Any]] = []
        self.last_outcome: Optional[str] = None
        self.transaction_counter: int = 0

        # Initialize default attacker states
        self._init_attacker_states()

    def _init_attacker_states(self) -> None:
        """Initializes persistent AttackerState instances for each profile in pool."""
        self.attacker_states.clear()
        for aid, prof in ATTACKER_POOL.items():
            self.attacker_states[aid] = AttackerState(
                actor_id=prof.actor_id,
                alias=prof.name,
                archetype=prof.archetype,
                objective=prof.risk_intent,
                current_device=prof.primary_device,
                current_location=dict(prof.location),
                current_recipient="REC_MULE_01",
                current_amount=ATTACK_STRATEGIES.get(prof.preferred_strategy, ATTACK_STRATEGIES["account_takeover"]).default_amount,
                strategy=prof.preferred_strategy,
                ip_address=prof.ip_address,
                device_history=[prof.primary_device],
                location_history=[dict(prof.location)],
                recipient_history=["REC_MULE_01"]
            )

    # -------------------------------------------------------------------------
    # Seed & Mode Management
    # -------------------------------------------------------------------------
    def set_mode(self, mode: Literal["DEMO_SEEDED", "LIVE_VARIATION"], seed: Optional[int] = 42) -> None:
        """Sets operating mode and resets RNG seed."""
        self.mode = mode
        self.seed = seed
        self.rng = random.Random(seed if mode == "DEMO_SEEDED" else None)
        logger.info("Simulation mode set", mode=mode, seed=seed)

    def set_seed(self, seed: int) -> None:
        """Explicitly set seed for reproducible experiment generation."""
        self.seed = seed
        self.mode = "DEMO_SEEDED"
        self.rng = random.Random(seed)

    def reset_seed(self) -> None:
        """Resets RNG back to initial seed."""
        self.rng = random.Random(self.seed if self.mode == "DEMO_SEEDED" else None)

    # -------------------------------------------------------------------------
    # State Accessors & Compatibility
    # -------------------------------------------------------------------------
    def get_attacker_state(self, actor_id: Optional[str] = None) -> AttackerState:
        """Returns the persistent AttackerState for given actor_id."""
        aid = actor_id or self.active_actor_id
        if aid not in self.attacker_states:
            prof = ATTACKER_POOL.get(aid, ATTACKER_POOL["ATK-7F31"])
            self.attacker_states[aid] = AttackerState(
                actor_id=prof.actor_id,
                alias=prof.name,
                archetype=prof.archetype,
                objective=prof.risk_intent,
                current_device=prof.primary_device,
                current_location=dict(prof.location),
                ip_address=prof.ip_address
            )
        return self.attacker_states[aid]

    def get_attacker_profiles(self) -> Dict[str, AttackerProfile]:
        """Returns dictionary of all adversary profiles."""
        return ATTACKER_POOL

    def get_active_profile(self, actor_id: Optional[str] = None) -> AttackerProfile:
        """Returns the active adversary profile."""
        aid = actor_id or self.active_actor_id
        return ATTACKER_POOL.get(aid, ATTACKER_POOL["ATK-7F31"])

    def set_active_profile(self, actor_id: str) -> AttackerProfile:
        """Sets the active adversary and ensures state is tracked."""
        if actor_id in ATTACKER_POOL:
            self.active_actor_id = actor_id
            state = self.get_attacker_state(actor_id)
            return ATTACKER_POOL[actor_id]
        return ATTACKER_POOL["ATK-7F31"]

    def get_strategies(self) -> Dict[str, AttackStrategy]:
        """Returns dictionary of all attack strategies."""
        return ATTACK_STRATEGIES

    def get_recipient_registry(self) -> Dict[str, Dict[str, Any]]:
        """Returns dictionary of diversified recipients."""
        return ATTACK_RECIPIENTS

    # Backward compatibility properties
    @property
    def location_history(self) -> List[Dict[str, Any]]:
        return self.get_attacker_state().location_history

    @property
    def device_history(self) -> List[str]:
        return self.get_attacker_state().device_history

    @property
    def recipient_history(self) -> List[str]:
        return self.get_attacker_state().recipient_history

    # -------------------------------------------------------------------------
    # Dynamic Adversarial Adaptation Engine
    # -------------------------------------------------------------------------
    def record_outcome_and_adapt(self, result: Any) -> Dict[str, Any]:
        """
        Observes the authoritative gateway verdict (BLOCK / HOLD / APPROVE) and computes
        an intelligent tactical response for the next simulation cycle.
        Updates persistent AttackerState.
        """
        dec = getattr(result, "decision", "APPROVE")
        status = getattr(result, "status", dec)
        amt = getattr(result, "amount", 1000.0)
        self.last_outcome = dec
        self.transaction_counter += 1

        state = self.get_attacker_state()
        state.record_outcome(dec, status)

        adaptation = {
            "attempt_number": self.transaction_counter,
            "actor_id": state.actor_id,
            "previous_decision": dec,
            "previous_status": status,
            "previous_amount": amt,
            "next_recommended_strategy": "low_and_slow",
            "next_recommended_amount": 450.0,
            "tactical_rationale": "",
            "recommended_device_action": "KEEP",
            "recommended_location_action": "SAME_NODE"
        }

        if dec == "BLOCK":
            adaptation["next_recommended_strategy"] = "low_and_slow"
            adaptation["next_recommended_amount"] = round(min(500.0, max(200.0, amt * 0.15)), 2)
            adaptation["recommended_device_action"] = "ROTATE_DEVICE"
            adaptation["recommended_location_action"] = "BLEND_INTO_HOME_CITY"
            adaptation["tactical_rationale"] = (
                f"Adversary intercepted on large transfer (₹{amt:,.2f}) ➔ Shifting tactics to "
                f"Low-and-Slow stealth probing (₹{adaptation['next_recommended_amount']:,.2f}) to test sub-threshold limits."
            )
        elif dec == "HOLD":
            adaptation["next_recommended_strategy"] = "velocity_surge"
            adaptation["next_recommended_amount"] = round(min(1200.0, max(400.0, amt * 0.6)), 2)
            adaptation["recommended_device_action"] = "ROTATE_HARDWARE"
            adaptation["recommended_location_action"] = "SWITCH_NODE"
            adaptation["tactical_rationale"] = (
                "Adversary encountered 2FA verification challenge ➔ Rotating hardware fingerprint "
                "and adjusting velocity to probe unmonitored channels."
            )
        else:  # APPROVE
            adaptation["next_recommended_strategy"] = "velocity_surge"
            adaptation["next_recommended_amount"] = round(amt + 300.0, 2)
            adaptation["recommended_device_action"] = "CONTINUE_SESSION"
            adaptation["recommended_location_action"] = "SAME_NODE"
            adaptation["tactical_rationale"] = (
                f"Transfer of ₹{amt:,.2f} succeeded ➔ Adversary escalating transfer size to "
                f"₹{adaptation['next_recommended_amount']:,.2f} to accelerate balance extraction."
            )

        self.adaptation_log.insert(0, adaptation)
        return adaptation

    def get_latest_adaptation(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent adaptation recommendation."""
        return self.adaptation_log[0] if self.adaptation_log else None

    # -------------------------------------------------------------------------
    # Canonical ATO Simulation (Backward Compatible)
    # -------------------------------------------------------------------------
    def simulate_account_takeover(
        self,
        target_customer_id: str = DEFAULT_DEMO_CUSTOMER_ID,
        new_device: bool = True,
        new_location: bool = True,
        password_change: bool = True,
        auth_anomaly: bool = True,
        custom_device_id: Optional[str] = None,
        custom_location: Optional[Dict[str, Any]] = None,
        current_time: Optional[float] = None
    ) -> List[AccountSecurityEvent]:
        """
        Simulate an account-takeover security event sequence.
        Generates, evaluates, and persists canonical AccountSecurityEvents.
        """
        now = current_time or time.time()
        events_generated: List[AccountSecurityEvent] = []

        active_prof = self.get_active_profile()
        device = custom_device_id or (active_prof.primary_device if new_device else KNOWN_DEVICE_ID)
        loc = custom_location or (active_prof.location if new_location else KNOWN_LOCATION)
        ip = active_prof.ip_address if new_location else "127.0.0.1"

        # 1. Login anomaly
        if auth_anomaly:
            ev_login = AccountSecurityEvent.create(
                customer_id=target_customer_id,
                event_type="login_anomaly",
                timestamp=now - 120.0,
                ip_address=ip,
                device_id=device,
                metadata={"reason": "Unusual IP and device fingerprint", "location": loc, "actor_id": active_prof.actor_id}
            )
            events_generated.append(ev_login)

        # 2. Password change event
        if password_change:
            ev_pw = AccountSecurityEvent.create(
                customer_id=target_customer_id,
                event_type="password_change",
                timestamp=now - 60.0,
                ip_address=ip,
                device_id=device,
                metadata={"reset_method": "account_recovery", "location": loc, "actor_id": active_prof.actor_id}
            )
            events_generated.append(ev_pw)

        # 3. New device registration
        if new_device:
            ev_dev = AccountSecurityEvent.create(
                customer_id=target_customer_id,
                event_type="new_device_registration",
                timestamp=now - 30.0,
                ip_address=ip,
                device_id=device,
                metadata={"device_model": "Linux/Chrome 122", "location": loc, "actor_id": active_prof.actor_id}
            )
            events_generated.append(ev_dev)

        # Ingest into ATO engine and persist into sink
        for ev in events_generated:
            self.ato_engine.record_event(ev)
            try:
                self.sink.persist_account_event({
                    "event_id": ev.event_id,
                    "customer_id": ev.customer_id,
                    "event_type": ev.event_type,
                    "timestamp": ev.timestamp,
                    "ip_address": ev.ip_address,
                    "device_id": ev.device_id,
                    "metadata": ev.metadata
                })
            except Exception as ex:
                logger.warning(f"Could not persist account event: {ex}")

        # Update account status to UNDER_ATTACK
        self.account_manager.update_status(target_customer_id, "UNDER_ATTACK")
        logger.info(
            "Account Takeover simulated successfully",
            customer_id=target_customer_id,
            events_count=len(events_generated)
        )
        return events_generated

    # -------------------------------------------------------------------------
    # Canonical Malicious Transfer Request Formulator
    # -------------------------------------------------------------------------
    def build_malicious_transfer_request(
        self,
        sender_id: str = DEFAULT_DEMO_CUSTOMER_ID,
        receiver_id: str = DEFAULT_ATTACKER_RECEIVER_ID,
        amount: float = 8500.0,
        device_id: Optional[str] = None,
        location: Optional[Dict[str, Any]] = None,
        auth_verified: bool = False,
        tx_id: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """Formulates a canonical malicious transaction attempt."""
        now = timestamp or time.time()
        active_prof = self.get_active_profile()
        dev = device_id or active_prof.primary_device
        loc = location or active_prof.location
        tid = tx_id or f"tx_atk_{int(now * 1000)}"

        account = self.account_manager.get_account(sender_id)

        tx_request = {
            "transaction_id": tid,
            "customer_id": sender_id,
            "merchant_id": receiver_id,
            "amount": float(amount),
            "timestamp": now,
            "payment_type": "TRANSFER",
            "category": "transfer",
            "origin_balance": float(account.balance),
            "dest_balance": float(self.account_manager.get_balance(receiver_id)),
            "auth_verified": bool(auth_verified),
            "device_id": dev,
            "latitude": float(loc.get("latitude", 19.0760)),
            "longitude": float(loc.get("longitude", 72.8777)),
            "location": {
                "latitude": float(loc.get("latitude", 19.0760)),
                "longitude": float(loc.get("longitude", 72.8777)),
                "city": loc.get("city", "MUMBAI")
            },
            "home_latitude": float(account.home_latitude),
            "home_longitude": float(account.home_longitude),
            "home_location": {
                "latitude": float(account.home_latitude),
                "longitude": float(account.home_longitude),
                "city": account.known_city
            },
            "raw_metadata": {
                "simulation": "ADVERSARIAL_ATTACK",
                "actor_id": active_prof.actor_id,
                "attacker_device": dev,
                "city": loc.get("city", "MUMBAI"),
                "strategy": self.active_strategy_id
            }
        }
        return tx_request

    # -------------------------------------------------------------------------
    # Comprehensive Strategy Runner (Known 1-6 + Unseen U1 & U2)
    # -------------------------------------------------------------------------
    def build_scenario_transfer(
        self,
        scenario_type: str,
        sender_id: str = DEFAULT_DEMO_CUSTOMER_ID,
        receiver_id: Optional[str] = None,
        amount: Optional[float] = None,
        custom_params: Optional[Dict[str, Any]] = None,
        actor_id: Optional[str] = None
    ) -> Tuple[List[AccountSecurityEvent], Dict[str, Any]]:
        """
        Executes one of the dynamic attack strategies with controlled variations across
        device, location, recipient, amount, timing, and security events.
        
        Supports:
        - 1. Smash & Grab (ATO)
        - 2. Velocity Surge
        - 3. Low & Slow
        - 4. Impossible Travel
        - 5. New Device Hold
        - 6. Novel SIM-Swap
        - 7. U1 — Context Mutation (Unseen)
        - 8. U2 — Compound Novel Attack (Unseen)
        """
        now = time.time()
        params = custom_params or {}
        events: List[AccountSecurityEvent] = []

        if actor_id:
            self.set_active_profile(actor_id)
        active_prof = self.get_active_profile()
        state = self.get_attacker_state(active_prof.actor_id)
        self.active_strategy_id = scenario_type

        norm_sc = scenario_type.lower()

        # =====================================================================
        # 1. ACCOUNT TAKEOVER / SMASH & GRAB (KNOWN TEMPLATE 1)
        # =====================================================================
        if "takeover" in norm_sc or "ato" in norm_sc or "smash" in norm_sc:
            events = self.simulate_account_takeover(
                target_customer_id=sender_id,
                new_device=params.get("new_device", True),
                new_location=params.get("new_location", True),
                password_change=params.get("password_change", True),
                auth_anomaly=params.get("auth_anomaly", True),
                custom_device_id=params.get("device_id", active_prof.primary_device),
                custom_location=params.get("location", active_prof.location),
                current_time=now
            )
            rec = receiver_id or DEFAULT_ATTACKER_RECEIVER_ID
            amt = amount or (self._sample_amount(ATTACK_STRATEGIES["account_takeover"].amount_range) if self.mode == "DEMO_SEEDED" else 8500.0)
            dev = params.get("device_id", active_prof.primary_device)
            loc = params.get("location", active_prof.location)
            req = self.build_malicious_transfer_request(
                sender_id=sender_id,
                receiver_id=rec,
                amount=amt,
                device_id=dev,
                location=loc,
                auth_verified=False,
                timestamp=now
            )
            req["raw_metadata"]["scenario"] = "ACCOUNT_TAKEOVER_SMASH_GRAB"
            req["raw_metadata"]["template_classification"] = get_template_classification("account_takeover")

        # =====================================================================
        # 2. VELOCITY DRAIN SURGE (KNOWN TEMPLATE 2)
        # =====================================================================
        elif "velocity" in norm_sc or "drain" in norm_sc or "transaction_fraud" in norm_sc:
            rec = receiver_id or "REC_MULE_01"
            amt = amount or (self._sample_amount(ATTACK_STRATEGIES["velocity_surge"].amount_range) if self.mode == "DEMO_SEEDED" else 1200.0)
            dev = params.get("device_id", "DEV-ATK-7129")
            loc = params.get("location", INDIAN_CITIES["DELHI"])
            req = self.build_malicious_transfer_request(
                sender_id=sender_id,
                receiver_id=rec,
                amount=amt,
                device_id=dev,
                location=loc,
                auth_verified=False,
                timestamp=now
            )
            req["raw_metadata"]["scenario"] = "VELOCITY_SURGE_DRAIN"
            req["raw_metadata"]["template_classification"] = get_template_classification("velocity_surge")

        # =====================================================================
        # 3. LOW AND SLOW STEALTH PROBE (KNOWN TEMPLATE 3)
        # =====================================================================
        elif "slow" in norm_sc or "low" in norm_sc or "stealth" in norm_sc:
            rec = receiver_id or "REC_PEER_04"
            amt = amount or (self._sample_amount(ATTACK_STRATEGIES["low_and_slow"].amount_range) if self.mode == "DEMO_SEEDED" else 450.0)
            dev = params.get("device_id", KNOWN_DEVICE_ID)
            loc = params.get("location", KNOWN_LOCATION)
            req = self.build_malicious_transfer_request(
                sender_id=sender_id,
                receiver_id=rec,
                amount=amt,
                device_id=dev,
                location=loc,
                auth_verified=True,
                timestamp=now
            )
            req["raw_metadata"]["scenario"] = "LOW_AND_SLOW_EVASION"
            req["raw_metadata"]["template_classification"] = get_template_classification("low_and_slow")

        # =====================================================================
        # 4. IMPOSSIBLE TRAVEL (KNOWN TEMPLATE 4)
        # =====================================================================
        elif "travel" in norm_sc or "hop" in norm_sc:
            loc_dest = params.get("location", INDIAN_CITIES.get("DELHI", INDIAN_CITIES["MUMBAI"]))
            dist_km = calculate_haversine_distance(
                KNOWN_LOCATION["latitude"], KNOWN_LOCATION["longitude"],
                loc_dest["latitude"], loc_dest["longitude"]
            )
            ev_loc = AccountSecurityEvent.create(
                customer_id=sender_id,
                event_type="impossible_travel",
                timestamp=now - 45.0,
                ip_address=active_prof.ip_address,
                device_id="DEV_ROAMING_X",
                metadata={"speed_kmh": 2400.0, "origin": "CHENNAI", "destination": loc_dest["city"], "distance_km": dist_km}
            )
            self.ato_engine.record_event(ev_loc)
            try:
                self.sink.persist_account_event(ev_loc.to_dict())
            except Exception:
                pass
            events.append(ev_loc)

            rec = receiver_id or "REC_CRYPTO_02"
            amt = amount or (self._sample_amount(ATTACK_STRATEGIES["impossible_travel"].amount_range) if self.mode == "DEMO_SEEDED" else 4200.0)
            dev = params.get("device_id", "DEV_ROAMING_X")
            req = self.build_malicious_transfer_request(
                sender_id=sender_id,
                receiver_id=rec,
                amount=amt,
                device_id=dev,
                location=loc_dest,
                auth_verified=False,
                timestamp=now
            )
            req["hour_of_day"] = 3  # 03:00 AM off-hours execution
            req["raw_metadata"]["scenario"] = "IMPOSSIBLE_TRAVEL_HOP"
            req["raw_metadata"]["distance_km"] = dist_km
            req["raw_metadata"]["template_classification"] = get_template_classification("impossible_travel")

        # =====================================================================
        # 5. NEW DEVICE / VERIFICATION HOLD (KNOWN TEMPLATE 5)
        # =====================================================================
        elif "device" in norm_sc and "hold" in norm_sc:
            dev_new = params.get("device_id", "DEV-ATK-4408")
            rec = receiver_id or "REC_OFFSHORE_03"
            amt = amount or (self._sample_amount(ATTACK_STRATEGIES["new_device_hold"].amount_range) if self.mode == "DEMO_SEEDED" else 2500.0)
            req = {
                "transaction_id": f"tx_hold_{int(now*1000)}",
                "customer_id": sender_id,
                "merchant_id": rec,
                "amount": float(amt),
                "timestamp": now,
                "payment_type": "TRANSFER",
                "category": "transfer",
                "origin_balance": float(self.account_manager.get_balance(sender_id)),
                "dest_balance": 500.0,
                "auth_verified": True,
                "device_id": dev_new,
                "latitude": KNOWN_LOCATION["latitude"],
                "longitude": KNOWN_LOCATION["longitude"],
                "location": {"latitude": KNOWN_LOCATION["latitude"], "longitude": KNOWN_LOCATION["longitude"], "city": KNOWN_LOCATION["city"]},
                "home_latitude": KNOWN_LOCATION["latitude"],
                "home_longitude": KNOWN_LOCATION["longitude"],
                "home_location": {"latitude": KNOWN_LOCATION["latitude"], "longitude": KNOWN_LOCATION["longitude"], "city": KNOWN_LOCATION["city"]},
                "raw_metadata": {
                    "scenario": "NEW_DEVICE_HOLD_PROBE",
                    "device_id": dev_new,
                    "template_classification": get_template_classification("new_device_hold")
                }
            }

        # =====================================================================
        # 7. U1 — CONTEXT MUTATION (UNSEEN SCENARIO 1)
        # =====================================================================
        elif "u1" in norm_sc or "context_mutation" in norm_sc or "mutation" in norm_sc:
            # U1 deliberately avoids matching any single predefined attack template:
            # Moderate amount (₹2,450), new device (DEV-MUT-8842), moderate travel (Hyderabad ~510 km),
            # new recipient (REC_PEER_07), unusual timing (23:45 off-peak), valid auth (no brute force), normal velocity
            loc_u1 = params.get("location", INDIAN_CITIES["HYDERABAD"])
            dev_u1 = params.get("device_id", "DEV-MUT-8842")
            rec_u1 = receiver_id or "REC_PEER_07"
            amt_u1 = amount or (self._sample_amount(ATTACK_STRATEGIES["u1_context_mutation"].amount_range) if self.mode == "DEMO_SEEDED" else 2450.0)

            # U1 sends clean transaction context without injecting artificial ATO security alarm events
            req = {
                "transaction_id": f"tx_u1_{int(now*1000)}",
                "customer_id": sender_id,
                "merchant_id": rec_u1,
                "amount": float(amt_u1),
                "timestamp": now,
                "payment_type": "TRANSFER",
                "category": "peer_transfer",
                "origin_balance": float(self.account_manager.get_balance(sender_id)),
                "dest_balance": 250.0,
                "auth_verified": True,
                "device_id": dev_u1,
                "latitude": float(loc_u1["latitude"]),
                "longitude": float(loc_u1["longitude"]),
                "location": {
                    "latitude": float(loc_u1["latitude"]),
                    "longitude": float(loc_u1["longitude"]),
                    "city": loc_u1["city"]
                },
                "home_latitude": KNOWN_LOCATION["latitude"],
                "home_longitude": KNOWN_LOCATION["longitude"],
                "home_location": {
                    "latitude": KNOWN_LOCATION["latitude"],
                    "longitude": KNOWN_LOCATION["longitude"],
                    "city": KNOWN_LOCATION["city"]
                },
                "hour_of_day": 23,  # Unusual late hour (23:45)
                "raw_metadata": {
                    "scenario": "U1_CONTEXT_MUTATION",
                    "template_classification": "Not one of the predefined attack templates",
                    "context_profile": {
                        "amount_profile": "MODERATE",
                        "device_profile": "UNREGISTERED_HARDWARE",
                        "location_profile": "REGIONAL_HUB_HYDERABAD",
                        "recipient_profile": "NEW_UNINDEXED_PEER",
                        "auth_profile": "VALID_TOKEN",
                        "velocity_profile": "NORMAL"
                    }
                }
            }

        # =====================================================================
        # 8. U2 — COMPOUND NOVEL ATTACK (UNSEEN SCENARIO 2)
        # =====================================================================
        elif "u2" in norm_sc or "compound" in norm_sc:
            # U2 combines multiple moderate signals:
            # new device + new recipient + off-hours (02:41 AM) + moderate distance (Pune ~915 km) + slightly elevated amount (₹2,750) + small velocity
            loc_u2 = params.get("location", INDIAN_CITIES["PUNE"])
            dev_u2 = params.get("device_id", "DEV-COMPOUND-77")
            rec_u2 = receiver_id or "REC_VENDOR_09"
            amt_u2 = amount or (self._sample_amount(ATTACK_STRATEGIES["u2_compound_novel"].amount_range) if self.mode == "DEMO_SEEDED" else 2750.0)

            # Record a subtle pre-flight security note (moderate anomaly, not a hard block)
            ev_comp = AccountSecurityEvent.create(
                customer_id=sender_id,
                event_type="device_fingerprint_drift",
                timestamp=now - 20.0,
                ip_address=active_prof.ip_address,
                device_id=dev_u2,
                metadata={"drift_type": "COMPOUND_MODERATE", "location": loc_u2}
            )
            self.ato_engine.record_event(ev_comp)
            try:
                self.sink.persist_account_event(ev_comp.to_dict())
            except Exception:
                pass
            events.append(ev_comp)

            req = {
                "transaction_id": f"tx_u2_{int(now*1000)}",
                "customer_id": sender_id,
                "merchant_id": rec_u2,
                "amount": float(amt_u2),
                "timestamp": now,
                "payment_type": "TRANSFER",
                "category": "merchant_payment",
                "origin_balance": float(self.account_manager.get_balance(sender_id)),
                "dest_balance": 1500.0,
                "auth_verified": True,
                "device_id": dev_u2,
                "latitude": float(loc_u2["latitude"]),
                "longitude": float(loc_u2["longitude"]),
                "location": {
                    "latitude": float(loc_u2["latitude"]),
                    "longitude": float(loc_u2["longitude"]),
                    "city": loc_u2["city"]
                },
                "home_latitude": KNOWN_LOCATION["latitude"],
                "home_longitude": KNOWN_LOCATION["longitude"],
                "home_location": {
                    "latitude": KNOWN_LOCATION["latitude"],
                    "longitude": KNOWN_LOCATION["longitude"],
                    "city": KNOWN_LOCATION["city"]
                },
                "hour_of_day": 2,  # 02:41 AM off-hours
                "raw_metadata": {
                    "scenario": "U2_COMPOUND_NOVEL_ATTACK",
                    "template_classification": "Not one of the predefined attack templates",
                    "compound_signals": [
                        "NEW_DEVICE_FINGERPRINT",
                        "UNINDEXED_MERCHANT_RECIPIENT",
                        "OFF_HOURS_EXECUTION_02AM",
                        "MODERATE_DISTANCE_PUNE",
                        "MODERATE_AMOUNT_ELEVATION",
                        "SLIGHT_VELOCITY_DRIFT"
                    ]
                }
            }

        # =====================================================================
        # 6. NOVEL SIM-SWAP / EDGE-CASE (KNOWN TEMPLATE 6)
        # =====================================================================
        else:
            rec = receiver_id or "REC_PEER_04"
            amt = amount or (self._sample_amount(ATTACK_STRATEGIES["novel_attack"].amount_range) if self.mode == "DEMO_SEEDED" else 3500.0)
            dev = params.get("device_id", "DEV_EMULATOR_NOVEL")
            loc = params.get("location", {"latitude": 13.0850, "longitude": 80.2720, "city": "CHENNAI"})
            req = {
                "transaction_id": f"tx_novel_{int(now*1000)}",
                "customer_id": sender_id,
                "merchant_id": rec,
                "amount": float(amt),
                "timestamp": now,
                "payment_type": "TRANSFER",
                "category": "peer_to_peer_novel",
                "origin_balance": float(self.account_manager.get_balance(sender_id)),
                "dest_balance": 100.0,
                "auth_verified": True,  # SIM swap bypass
                "device_id": dev,
                "latitude": float(loc["latitude"]),
                "longitude": float(loc["longitude"]),
                "location": {"latitude": float(loc["latitude"]), "longitude": float(loc["longitude"]), "city": loc["city"]},
                "home_latitude": KNOWN_LOCATION["latitude"],
                "home_longitude": KNOWN_LOCATION["longitude"],
                "home_location": {"latitude": KNOWN_LOCATION["latitude"], "longitude": KNOWN_LOCATION["longitude"], "city": KNOWN_LOCATION["city"]},
                "raw_metadata": {
                    "scenario": "NOVEL_ATTACK_SIM_SWAP",
                    "attack_type": "UNKNOWN",
                    "actor_id": "ATK-5E20",
                    "template_classification": get_template_classification("novel_attack")
                }
            }

        # Update persistent attacker state with this attempt
        state.record_attempt(
            transaction_id=req["transaction_id"],
            amount=float(req["amount"]),
            device_id=req.get("device_id", "unknown"),
            location=req.get("location", KNOWN_LOCATION),
            recipient_id=req.get("merchant_id", "unknown"),
            strategy=scenario_type,
            timestamp=now
        )

        return events, req

    def _sample_amount(self, range_tuple: Tuple[float, float]) -> float:
        """Sample a controlled reproducible amount within a strategy's constraint bounds."""
        low, high = range_tuple
        step = 50.0
        steps_count = max(1, int((high - low) / step))
        rand_step = self.rng.randint(0, steps_count)
        sampled = low + (rand_step * step)
        return round(min(high, max(low, sampled)), 2)

    # -------------------------------------------------------------------------
    # Timeline Builder
    # -------------------------------------------------------------------------
    def build_attack_timeline(
        self,
        events: List[AccountSecurityEvent],
        transfer_request: Dict[str, Any],
        result: Any
    ) -> List[Dict[str, Any]]:
        """
        Generate a chronological real-time event timeline using actual
        system timestamps, security events, evaluation metrics, and gateway decision.
        """
        timeline: List[Dict[str, Any]] = []

        # 1. Security events
        for ev in sorted(events, key=lambda x: x.timestamp):
            t_str = time.strftime('%H:%M:%S', time.localtime(ev.timestamp))
            ev_name = ev.event_type.replace("_", " ").title()
            timeline.append({
                "time": t_str,
                "event": f"Security Alarm: {ev_name}",
                "detail": f"Device: {ev.device_id} • IP: {ev.ip_address}",
                "status": "DETECTED",
                "severity": "CRITICAL" if ev.event_type in ["password_change", "login_anomaly", "impossible_travel"] else "HIGH"
            })

        # 2. Transfer initiation
        tx_time = float(transfer_request.get("timestamp", time.time()))
        t_tx_str = time.strftime('%H:%M:%S', time.localtime(tx_time))
        raw_meta = transfer_request.get("raw_metadata", {})
        sc_name = raw_meta.get("scenario", "TRANSFER")
        templ_class = raw_meta.get("template_classification", "Predefined Attack Template")
        timeline.append({
            "time": t_tx_str,
            "event": f"Adversarial Transfer Initiated ({sc_name})",
            "detail": f"₹{float(transfer_request.get('amount', 0)):,.2f} → {transfer_request.get('merchant_id', 'UNKNOWN')} ({templ_class})",
            "status": "INGRESS",
            "severity": "INFO"
        })

        # 3. Gateway Ingress
        timeline.append({
            "time": t_tx_str,
            "event": "Payment Gateway Ingress Pre-Flight",
            "detail": "Gateway intercepted transfer before balance settlement",
            "status": "AUTHORIZING",
            "severity": "INFO"
        })

        # 4. Multi-Layer Evaluation
        lat = getattr(result, "latency_ms", 1.5)
        ato_act = getattr(result, "ato_action", "ALLOW")
        ato_sc = getattr(result, "ato_score", 0.0)
        dec = getattr(result, "decision", "APPROVE")
        money_moved = getattr(result, "money_transferred", False)

        timeline.append({
            "time": t_tx_str,
            "event": "FinPulse 4-Layer Defense Evaluation",
            "detail": f"Inference complete in {lat:.2f} ms • ATO Action: {ato_act} (Score: {ato_sc:.2f})",
            "status": "EVALUATED",
            "severity": "CRITICAL" if ato_act == "SUSPEND_ACCOUNT" else ("HIGH" if dec == "BLOCK" else "MEDIUM")
        })

        # 5. Gateway Verdict
        timeline.append({
            "time": t_tx_str,
            "event": f"Gateway Authorization Verdict: {dec}",
            "detail": f"Outcome: {dec} • Status: {getattr(result, 'status', dec)}",
            "status": dec,
            "severity": "CRITICAL" if dec == "BLOCK" else ("HIGH" if dec == "HOLD" else "LOW")
        })

        # 6. Balance Protection
        amt_str = f"₹{float(transfer_request.get('amount', 0)):,.2f}"
        if not money_moved:
            timeline.append({
                "time": t_tx_str,
                "event": "Money Transferred: ₹0 (Protected)",
                "detail": f"Gateway prevented fund movement. Customer balance untouched ({amt_str} protected)",
                "status": "PROTECTED",
                "severity": "LOW"
            })
        else:
            timeline.append({
                "time": t_tx_str,
                "event": f"Money Transferred: {amt_str}",
                "detail": "Gateway authorized fund transfer to receiver",
                "status": "TRANSFERRED",
                "severity": "INFO"
            })

        return timeline

    # -------------------------------------------------------------------------
    # Clean State Reset
    # -------------------------------------------------------------------------
    def reset(self) -> None:
        """Resets the simulation runner, attacker states, and counters cleanly."""
        self._init_attacker_states()
        self.active_actor_id = "ATK-7F31"
        self.active_strategy_id = "account_takeover"
        self.adaptation_log.clear()
        self.last_outcome = None
        self.transaction_counter = 0
        self.reset_seed()
        logger.info("Attacker simulator reset cleanly")
