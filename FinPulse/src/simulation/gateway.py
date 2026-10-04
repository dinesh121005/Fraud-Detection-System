"""
FinPulse Payment Gateway Simulator & Authorization Boundary (D3, D4, D5).

Orchestrates the authoritative transaction authorization workflow:
1. Transaction Attempt -> Validates Balance
2. Invokes Real-Time FinPulse Fraud Pipeline (Redis + 32-Features + CatBoost + Calibration + Rules)
3. Pre-Authorization ATO Guardrail Check
4. Evaluates Authoritative Decision:
   - APPROVE -> Gateway executes transfer (Sender balance decreases, Receiver increases)
   - HOLD    -> Money NOT transferred -> Creates review case -> Awaits customer verification
   - BLOCK   -> Money NOT transferred -> Rejects transfer -> Emits fraud alert
5. Resolves Customer Verification (Confirm -> Released & Executed / Deny -> Denied & Blocked)
6. Records complete audit trail into PostgreSQL
"""

import time
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Literal

from src.serving.predictor import ProductionPredictor
from src.workflow.hold_workflow import HoldWorkflowEngine, HoldCase
from src.workflow.account_events import ATOProtectionEngine
from src.workflow.security_case import SecurityCase
from src.persistence.sink import IdempotentEventSink
from src.risk_engine.decision_event import DecisionEvent
from src.streaming.publisher import DecisionEventPublisher
from src.simulation.demo_account import (
    DemoAccountManager,
    DEFAULT_DEMO_CUSTOMER_ID,
    DEFAULT_ATTACKER_RECEIVER_ID
)
from src.monitoring.logger import get_logger

logger = get_logger("FinPulse.Simulation.Gateway")

GatewayStatus = Literal[
    "APPROVED",
    "HELD",
    "DECLINED",
    "INSUFFICIENT_FUNDS",
    "RELEASED_AND_EXECUTED",
    "DENIED_AND_BLOCKED"
]


@dataclass
class GatewayTransferResult:
    """Canonical result contract returned by the Payment Gateway Simulator."""
    transaction_id: str
    sender_id: str
    receiver_id: str
    amount: float
    status: GatewayStatus
    authorized: bool
    money_transferred: bool
    decision: Literal["APPROVE", "HOLD", "BLOCK", "REJECTED"]
    risk_score: float
    risk_level: str
    sender_balance_before: float
    sender_balance_after: float
    receiver_balance_before: float
    receiver_balance_after: float
    reasons: list
    matched_rules: list
    latency_ms: float
    hold_case: Optional[HoldCase] = None
    confirmation_token: Optional[str] = None
    ato_score: float = 0.0
    ato_action: str = "ALLOW"
    timestamp: float = field(default_factory=time.time)
    hybrid_decision: Literal["APPROVE", "REVIEW", "BLOCK"] = "APPROVE"
    ml_decision: Literal["APPROVE", "REVIEW", "BLOCK"] = "APPROVE"
    calibrated_probability: float = 0.0
    eval_detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def gateway_decision(self) -> str:
        """Explicit alias for final gateway authorization decision."""
        return self.decision

    @property
    def pipeline_decision(self) -> str:
        """Backward-compatible alias for R5 hybrid risk decision."""
        return self.hybrid_decision

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "transaction_id": self.transaction_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "amount": self.amount,
            "status": self.status,
            "authorized": self.authorized,
            "money_transferred": self.money_transferred,
            "decision": self.decision,
            "gateway_decision": self.decision,
            "hybrid_decision": self.hybrid_decision,
            "pipeline_decision": self.hybrid_decision,
            "ml_decision": self.ml_decision,
            "calibrated_probability": self.calibrated_probability,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "sender_balance_before": self.sender_balance_before,
            "sender_balance_after": self.sender_balance_after,
            "receiver_balance_before": self.receiver_balance_before,
            "receiver_balance_after": self.receiver_balance_after,
            "reasons": list(self.reasons),
            "matched_rules": list(self.matched_rules),
            "latency_ms": self.latency_ms,
            "ato_score": self.ato_score,
            "ato_action": self.ato_action,
            "timestamp": self.timestamp
        }
        if self.hold_case:
            d["hold_id"] = self.hold_case.hold_id
            d["hold_status"] = self.hold_case.status
        return d


class PaymentGatewaySimulator:
    """
    Authoritative Gateway boundary enforcing server-side fraud decisions.
    Never transfers funds before FinPulse authorization grants approval.
    """

    def __init__(
        self,
        predictor: Optional[ProductionPredictor] = None,
        hold_engine: Optional[HoldWorkflowEngine] = None,
        ato_engine: Optional[ATOProtectionEngine] = None,
        event_sink: Optional[IdempotentEventSink] = None,
        account_manager: Optional[DemoAccountManager] = None,
        kafka_publisher: Optional[DecisionEventPublisher] = None
    ):
        finpulse_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        artifacts = os.path.join(finpulse_dir, "models", "artifacts")

        self.predictor = predictor or ProductionPredictor(artifacts_dir=artifacts)
        self.hold_engine = hold_engine or HoldWorkflowEngine()
        self.ato_engine = ato_engine or ATOProtectionEngine()
        self.sink = event_sink or IdempotentEventSink()
        self.account_manager = account_manager or DemoAccountManager()
        try:
            self.kafka_publisher = kafka_publisher or DecisionEventPublisher()
        except Exception as e:
            logger.warning(f"Could not connect DecisionEventPublisher to Kafka: {e}")
            self.kafka_publisher = None

    def process_transfer(self, transfer_request: Dict[str, Any]) -> GatewayTransferResult:
        """
        Main gateway transfer processing pipeline:
        1. Validate request and balance
        2. Pre-flight ATO guardrails
        3. Evaluate via real FinPulse pipeline
        4. Enforce APPROVE / HOLD / BLOCK
        5. Persist auditable records
        """
        t0 = time.perf_counter()
        tx_id = transfer_request.get("transaction_id", f"tx_gw_{int(time.time()*1000)}")
        sender_id = transfer_request.get("customer_id", DEFAULT_DEMO_CUSTOMER_ID)
        receiver_id = transfer_request.get("merchant_id", "CUST_ATTACKER")
        amount = float(transfer_request.get("amount", 0.0))
        ts = float(transfer_request.get("timestamp", time.time()))

        sender_bal_before = self.account_manager.get_balance(sender_id)
        receiver_bal_before = self.account_manager.get_balance(receiver_id)

        # 1. Validation: Amount check
        if amount <= 0.0:
            latency = (time.perf_counter() - t0) * 1000.0
            logger.warning("Gateway rejected: invalid non-positive amount", sender=sender_id, amount=amount)
            return GatewayTransferResult(
                transaction_id=tx_id,
                sender_id=sender_id,
                receiver_id=receiver_id,
                amount=amount,
                status="DECLINED",
                authorized=False,
                money_transferred=False,
                decision="REJECTED",
                risk_score=0.0,
                risk_level="LOW",
                sender_balance_before=sender_bal_before,
                sender_balance_after=sender_bal_before,
                receiver_balance_before=receiver_bal_before,
                receiver_balance_after=receiver_bal_before,
                reasons=["Invalid transaction amount: must be positive monetary amount"],
                matched_rules=[],
                latency_ms=round(latency, 2)
            )

        # 2. Validation: Balance check
        if sender_bal_before < amount:
            latency = (time.perf_counter() - t0) * 1000.0
            logger.warning("Gateway rejected: insufficient balance", sender=sender_id, balance=sender_bal_before, amount=amount)
            return GatewayTransferResult(
                transaction_id=tx_id,
                sender_id=sender_id,
                receiver_id=receiver_id,
                amount=amount,
                status="INSUFFICIENT_FUNDS",
                authorized=False,
                money_transferred=False,
                decision="REJECTED",
                risk_score=0.0,
                risk_level="LOW",
                sender_balance_before=sender_bal_before,
                sender_balance_after=sender_bal_before,
                receiver_balance_before=receiver_bal_before,
                receiver_balance_after=receiver_bal_before,
                reasons=["Insufficient account balance for requested transfer"],
                matched_rules=[],
                latency_ms=round(latency, 2)
            )

        # 2. Pre-authorization ATO Evaluation & Guardrails
        ato_assessment = self.ato_engine.evaluate_ato_risk(sender_id, current_time=ts)
        is_guardrail_allowed, guardrail_reason = self.ato_engine.check_transaction_guardrails(sender_id, amount)

        # Ensure transaction request carries current balance
        transfer_request["origin_balance"] = sender_bal_before

        # 3. Invoke real FinPulse detection pipeline (Redis, Features, CatBoost, Platt, Rules)
        eval_res = self.predictor.predict(transfer_request)
        latency = (time.perf_counter() - t0) * 1000.0

        risk_score = float(eval_res.get("risk_score", 0.0))
        risk_level = eval_res.get("risk_level", "LOW")
        pipeline_decision = eval_res.get("decision", "APPROVE")
        ml_decision = eval_res.get("ml_decision", pipeline_decision)
        calibrated_prob = float(eval_res.get("calibrated_probability", 0.0))
        reasons = list(eval_res.get("top_reasons", []))
        matched_rules = eval_res.get("rule_result", {}).get("matched_rules", [])

        # Persist raw transaction to PostgreSQL
        try:
            self.sink.persist_transaction(transfer_request)
        except Exception as e:
            logger.warning(f"Could not persist transaction: {e}")

        # 4. Synthesize final authorization decision across layers:
        # Layer 1: R4 ML Decision (ml_decision)
        # Layer 2: R5 Hybrid Risk Decision (pipeline_decision)
        # Layer 3: R7-A ATO Security Decision (ato_assessment.action)
        # Layer 4: Gateway Final Authorization Decision (final_decision & workflow_status)
        if ato_assessment.action == "SUSPEND_ACCOUNT" or pipeline_decision == "BLOCK" or (not is_guardrail_allowed and "suspended" in str(guardrail_reason).lower()):
            final_decision = "BLOCK"
            gateway_status: GatewayStatus = "DECLINED"
            workflow_status = "DECLINED_ATO_SUSPENDED" if ato_assessment.action == "SUSPEND_ACCOUNT" else "DECLINED_BLOCK"
            if ato_assessment.action == "SUSPEND_ACCOUNT":
                reasons.insert(0, "CRITICAL: Outbound transfers suspended by Payment Gateway due to active Account Takeover investigation")
        elif ato_assessment.requires_transaction_hold or pipeline_decision == "REVIEW" or (30.0 <= risk_score < 70.0):
            final_decision = "HOLD"
            gateway_status = "HELD"
            workflow_status = "HELD_PENDING_VERIFICATION"
        else:
            final_decision = "APPROVE"
            gateway_status = "APPROVED"
            workflow_status = "APPROVED"

        # Construct diagnostics and metadata capturing complete layered decision lineage
        diag = dict(eval_res.get("diagnostics", {}))
        diag.update({
            "ml_decision": ml_decision,
            "hybrid_decision": pipeline_decision,
            "pipeline_decision": pipeline_decision,
            "ato_action": ato_assessment.action,
            "ato_risk_score": ato_assessment.ato_risk_score,
            "gateway_decision": final_decision,
            "gateway_status": gateway_status,
            "workflow_status": workflow_status
        })

        # Construct and persist canonical DecisionEvent preserving R5 hybrid decision and gateway audit context
        decision_event = DecisionEvent(
            transaction_id=tx_id,
            customer_id=sender_id,
            decision=pipeline_decision,
            risk_score=risk_score,
            risk_level=risk_level,
            ml_decision=ml_decision,
            calibrated_probability=calibrated_prob,
            signals=dict(eval_res.get("signals", {})),
            diagnostics=diag,
            reasons=reasons,
            timestamp=ts,
            model_version=eval_res.get("model_version", "finpulse-v3"),
            matched_rules=matched_rules,
            hard_block=bool(eval_res.get("diagnostics", {}).get("hard_block", False)),
            hard_block_rules=eval_res.get("rule_result", {}).get("hard_block_rules", []),
            latency_ms=round(latency, 2),
            metadata={
                "gateway_decision": final_decision,
                "gateway_status": gateway_status,
                "workflow_status": workflow_status,
                "ato_action": ato_assessment.action,
                "ato_score": ato_assessment.ato_risk_score,
                "hybrid_decision": pipeline_decision,
                "ml_decision": ml_decision
            }
        )
        try:
            self.sink.persist_decision_event(decision_event, workflow_status=workflow_status)
        except Exception as e:
            logger.warning(f"Could not persist decision event: {e}")

        # Publish canonical DecisionEvent to Kafka 'predictions' (and 'fraud-alerts' if BLOCK/alert)
        if self.kafka_publisher:
            try:
                self.kafka_publisher.publish(decision_event, sync=True)
            except Exception as e:
                logger.warning(f"Kafka publication failed: {e}. Gateway authorization proceeds safely.")

        # 5. Enforce Gateway Authorization Action
        if final_decision == "APPROVE":
            # Transfer execution: Balance updates only now
            self.account_manager.execute_transfer(sender_id, receiver_id, amount)
            sender_bal_after = self.account_manager.get_balance(sender_id)
            receiver_bal_after = self.account_manager.get_balance(receiver_id)

            return GatewayTransferResult(
                transaction_id=tx_id,
                sender_id=sender_id,
                receiver_id=receiver_id,
                amount=amount,
                status="APPROVED",
                authorized=True,
                money_transferred=True,
                decision="APPROVE",
                risk_score=risk_score,
                risk_level=risk_level,
                sender_balance_before=sender_bal_before,
                sender_balance_after=sender_bal_after,
                receiver_balance_before=receiver_bal_before,
                receiver_balance_after=receiver_bal_after,
                reasons=reasons,
                matched_rules=matched_rules,
                latency_ms=round(latency, 2),
                ato_score=ato_assessment.ato_risk_score,
                ato_action=ato_assessment.action,
                hybrid_decision=pipeline_decision,
                ml_decision=ml_decision,
                calibrated_probability=calibrated_prob,
                eval_detail=eval_res
            )

        elif final_decision == "HOLD":
            # Money is NOT transferred. Create review case.
            case, token = self.hold_engine.create_hold(
                transaction_id=tx_id,
                customer_id=sender_id,
                amount=amount,
                timeout_seconds=300.0,
                current_time=ts,
                metadata={
                    "receiver_id": receiver_id,
                    "risk_score": risk_score,
                    "ato_score": ato_assessment.ato_risk_score,
                    "device_id": transfer_request.get("device_id", "unknown"),
                    "location": transfer_request.get("location", {})
                }
            )

            # Persist hold case to PostgreSQL
            try:
                self.sink.persist_hold_case({
                    "hold_id": case.hold_id,
                    "transaction_id": case.transaction_id,
                    "customer_id": case.customer_id,
                    "amount": case.amount,
                    "status": case.status,
                    "token_hash": case.token_hash,
                    "timeout_seconds": case.timeout_seconds,
                    "created_at": case.created_at,
                    "expires_at": case.expires_at,
                    "customer_channel": case.customer_channel,
                    "metadata": case.metadata
                })
            except Exception as ex:
                logger.warning(f"Could not persist hold case: {ex}")

            if guardrail_reason and guardrail_reason not in reasons:
                reasons.append(guardrail_reason)

            return GatewayTransferResult(
                transaction_id=tx_id,
                sender_id=sender_id,
                receiver_id=receiver_id,
                amount=amount,
                status="HELD",
                authorized=False,
                money_transferred=False,
                decision="HOLD",
                risk_score=risk_score,
                risk_level=risk_level,
                sender_balance_before=sender_bal_before,
                sender_balance_after=sender_bal_before,
                receiver_balance_before=receiver_bal_before,
                receiver_balance_after=receiver_bal_before,
                reasons=reasons,
                matched_rules=matched_rules,
                latency_ms=round(latency, 2),
                hold_case=case,
                confirmation_token=token,
                ato_score=ato_assessment.ato_risk_score,
                ato_action=ato_assessment.action,
                hybrid_decision=pipeline_decision,
                ml_decision=ml_decision,
                calibrated_probability=calibrated_prob,
                eval_detail=eval_res
            )

        else:  # BLOCK
            # Money is NOT transferred. Evict blocked attempt from Redis sliding window
            if hasattr(self.predictor, "redis_window") and hasattr(self.predictor.redis_window, "remove_transaction"):
                try:
                    self.predictor.redis_window.remove_transaction(sender_id, tx_id)
                except Exception:
                    pass

            # Create fraud alert.
            try:
                self.sink.persist_fraud_alert({
                    "alert_id": f"alt_{int(time.time()*1000)}",
                    "transaction_id": tx_id,
                    "event_id": f"evt_{tx_id}",
                    "customer_id": sender_id,
                    "severity": "CRITICAL" if eval_res.get("hard_block") else "HIGH",
                    "decision": "BLOCK",
                    "reasons": reasons,
                    "notified": True,
                    "status": "OPEN",
                    "created_at": ts
                })
            except Exception as ex:
                logger.warning(f"Could not persist fraud alert: {ex}")

            if guardrail_reason and guardrail_reason not in reasons:
                reasons.append(guardrail_reason)

            return GatewayTransferResult(
                transaction_id=tx_id,
                sender_id=sender_id,
                receiver_id=receiver_id,
                amount=amount,
                status="DECLINED",
                authorized=False,
                money_transferred=False,
                decision="BLOCK",
                risk_score=risk_score,
                risk_level=risk_level,
                sender_balance_before=sender_bal_before,
                sender_balance_after=sender_bal_before,
                receiver_balance_before=receiver_bal_before,
                receiver_balance_after=receiver_bal_before,
                reasons=reasons,
                matched_rules=matched_rules,
                latency_ms=round(latency, 2),
                ato_score=ato_assessment.ato_risk_score,
                ato_action=ato_assessment.action,
                hybrid_decision=pipeline_decision,
                ml_decision=ml_decision,
                calibrated_probability=calibrated_prob,
                eval_detail=eval_res
            )

    def resolve_hold(
        self,
        hold_id: str,
        token: str,
        action: Literal["CONFIRM", "DENY"] = "CONFIRM",
        customer_channel: str = "customer_mobile_app"
    ) -> GatewayTransferResult:
        """
        Customer reviews the held transaction:
        - CONFIRM ("Yes, it's me") -> Release hold -> Execute money transfer -> Balance updates.
        - DENY ("No, this was not me") -> Deny hold -> Confirm fraud -> Permanent block -> Money stays safe.
        """
        t0 = time.perf_counter()
        case = self.hold_engine.get_case(hold_id)
        if not case:
            raise ValueError(f"Hold case '{hold_id}' not found.")

        receiver_id = case.metadata.get("receiver_id", DEFAULT_ATTACKER_RECEIVER_ID)
        sender_bal_before = self.account_manager.get_balance(case.customer_id)
        receiver_bal_before = self.account_manager.get_balance(receiver_id)

        if action == "CONFIRM":
            # State machine transition to RELEASED
            updated_case = self.hold_engine.confirm_hold(hold_id, token, channel=customer_channel)
            try:
                self.sink.update_hold_status(hold_id, "RELEASED", resolution_reason="Customer confirmed via mobile push")
            except Exception:
                pass

            # Update decision record in persistence
            try:
                p = "%s" if self.sink.backend == "postgres" else "?"
                with self.sink._get_cursor() as (cur, _, _):
                    cur.execute(
                        f"UPDATE fraud_decisions SET workflow_status = 'RELEASED', decision = 'APPROVE', updated_at = {p} WHERE transaction_id = {p};",
                        (time.time(), case.transaction_id)
                    )
            except Exception:
                pass

            # Gateway now executes the transfer
            self.account_manager.execute_transfer(case.customer_id, receiver_id, case.amount)
            sender_bal_after = self.account_manager.get_balance(case.customer_id)
            receiver_bal_after = self.account_manager.get_balance(receiver_id)
            latency = (time.perf_counter() - t0) * 1000.0

            return GatewayTransferResult(
                transaction_id=case.transaction_id,
                sender_id=case.customer_id,
                receiver_id=receiver_id,
                amount=case.amount,
                status="RELEASED_AND_EXECUTED",
                authorized=True,
                money_transferred=True,
                decision="APPROVE",
                risk_score=float(case.metadata.get("risk_score", 45.0)),
                risk_level="MEDIUM",
                sender_balance_before=sender_bal_before,
                sender_balance_after=sender_bal_after,
                receiver_balance_before=receiver_bal_before,
                receiver_balance_after=receiver_bal_after,
                reasons=[f"Transaction verified and released by customer via {customer_channel}"],
                matched_rules=[],
                latency_ms=round(latency, 2),
                hold_case=updated_case,
                confirmation_token=token
            )

        else:  # DENY (Fraud confirmed)
            updated_case = self.hold_engine.deny_hold(hold_id, token, reason="Customer reported unauthorized transaction")
            try:
                self.sink.update_hold_status(hold_id, "DENIED", resolution_reason="Customer reported unauthorized transaction")
            except Exception:
                pass

            # Mark decision as BLOCK in system of record
            try:
                p = "%s" if self.sink.backend == "postgres" else "?"
                with self.sink._get_cursor() as (cur, _, _):
                    cur.execute(
                        f"UPDATE fraud_decisions SET workflow_status = 'DENIED', decision = 'BLOCK', updated_at = {p} WHERE transaction_id = {p};",
                        (time.time(), case.transaction_id)
                    )
            except Exception:
                pass

            # Persist alert for confirmed fraud
            try:
                self.sink.persist_fraud_alert({
                    "alert_id": f"alt_confirmed_{int(time.time()*1000)}",
                    "transaction_id": case.transaction_id,
                    "event_id": f"evt_{case.transaction_id}",
                    "customer_id": case.customer_id,
                    "severity": "CRITICAL",
                    "decision": "BLOCK",
                    "reasons": ["Customer explicitly denied transfer: Fraud confirmed"],
                    "notified": True,
                    "status": "CONFIRMED_FRAUD",
                    "created_at": time.time()
                })
            except Exception:
                pass

            # Money was NOT transferred
            latency = (time.perf_counter() - t0) * 1000.0
            return GatewayTransferResult(
                transaction_id=case.transaction_id,
                sender_id=case.customer_id,
                receiver_id=receiver_id,
                amount=case.amount,
                status="DENIED_AND_BLOCKED",
                authorized=False,
                money_transferred=False,
                decision="BLOCK",
                risk_score=95.0,
                risk_level="HIGH",
                sender_balance_before=sender_bal_before,
                sender_balance_after=sender_bal_before,
                receiver_balance_before=receiver_bal_before,
                receiver_balance_after=receiver_bal_before,
                reasons=["Transfer denied by cardholder: Confirmed Fraud. Funds protected."],
                matched_rules=[],
                latency_ms=round(latency, 2),
                hold_case=updated_case,
                confirmation_token=token
            )

    def report_unauthorized_transaction(
        self,
        transaction_id: str,
        customer_id: str = DEFAULT_DEMO_CUSTOMER_ID,
        customer_report: str = "UNAUTHORIZED",
        attack_type: str = "UNKNOWN",
        notes: Optional[str] = None
    ) -> SecurityCase:
        """
        Customer reports an unauthorized / missed fraudulent transaction (D4, D5).
        - Creates a formal Gateway Security Case
        - Persists the case to PostgreSQL / SQLite
        - Attaches delayed ground-truth fraud label (1=Fraud)
        - Emits a CRITICAL fraud alert for operations
        - Updates account status to UNDER_ATTACK
        - Feeds the controlled ML feedback lifecycle
        """
        # Look up existing decision or transaction
        dec = self.sink.get_decision(transaction_id)
        gateway_decision = dec.get("decision", "APPROVE") if dec else "APPROVE"

        now = time.time()
        case_id = f"CASE-2026-{int(now * 1000) % 100000:05d}"

        sec_case = SecurityCase(
            case_id=case_id,
            transaction_id=transaction_id,
            customer_id=customer_id,
            gateway_decision=gateway_decision,
            customer_report=customer_report,
            confirmed_label="FRAUD",
            attack_type=attack_type,
            status="CONFIRMED_FRAUD",
            metadata={
                "notes": notes or "Customer reported unauthorized transaction via mobile wallet",
                "reported_at": now,
                "reported_channel": "customer_wallet",
                "original_gateway_decision": gateway_decision
            },
            created_at=now,
            updated_at=now
        )

        # 1. Persist security case
        try:
            self.sink.persist_security_case(sec_case.to_dict())
        except Exception as ex:
            logger.warning(f"Could not persist security case: {ex}")

        # 2. Attach delayed ground-truth fraud label to fraud_decisions
        try:
            self.sink.attach_delayed_label(transaction_id, fraud_label=1, label_timestamp=now)
        except Exception as ex:
            logger.warning(f"Could not attach delayed label: {ex}")

        # 3. Persist fraud alert
        try:
            self.sink.persist_fraud_alert({
                "alert_id": f"alt_case_{int(now*1000)}",
                "transaction_id": transaction_id,
                "event_id": f"evt_case_{transaction_id}",
                "customer_id": customer_id,
                "severity": "CRITICAL",
                "decision": "BLOCK",
                "reasons": [f"Customer confirmed unauthorized transfer. Case {case_id} recorded. Attack type: {attack_type}"],
                "notified": True,
                "status": "CONFIRMED_FRAUD",
                "created_at": now
            })
        except Exception as ex:
            logger.warning(f"Could not persist fraud alert: {ex}")

        # 4. Update customer account status
        try:
            self.account_manager.update_status(customer_id, "UNDER_ATTACK")
        except Exception:
            pass

        logger.info(
            "Customer reported unauthorized transaction: security case created",
            case_id=case_id,
            tx_id=transaction_id,
            attack_type=attack_type
        )
        return sec_case

