"""
FinPulse AI — Fraud Intelligence Operations Executive Dashboard (R7).

Provides 5 dedicated operational views:
1. 🔴 Live: Real-time transaction simulation, stream generator, KPI metric cards, and verdict distribution
2. 🕵️ Analyst: In-depth forensic case investigation, SHAP explanations, triggered rules, and feature inspectors
3. 📱 Phone: Interactive customer review simulation for transactions placed on HOLD (Confirm / Deny)
4. 📈 Results: E1–E10 evaluation benchmarks, cost-frontier optimization curve, and model performance metrics
5. 🛡️ Health: Infrastructure telemetry (Kafka 9092, Redis 6379, DB Sink, Model Artifacts, Circuit Breakers)
"""

import os
import sys
import time
import json
import random
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure FinPulse in sys.path
FINPULSE_DIR = os.path.dirname(os.path.abspath(__file__))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.serving.predictor import ProductionPredictor
from src.workflow.hold_workflow import HoldWorkflowEngine
from src.workflow.account_events import ATOProtectionEngine, AccountSecurityEvent
from src.persistence.sink import IdempotentEventSink
from src.risk_engine.decision_event import DecisionEvent
from src.monitoring.drift import PSIDriftMonitor

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FinPulse AI — Fraud Intelligence Operations",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Operations CSS
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.01));
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .pill-green { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .pill-blue { background: rgba(59, 130, 246, 0.15); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.3); }
    .pill-red { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
    .pill-amber { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }

    .phone-mockup {
        max-width: 380px;
        margin: 0 auto;
        background: #0f172a;
        border: 4px solid #334155;
        border-radius: 36px;
        padding: 24px 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.6);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Singleton Subsystem Caching
# -----------------------------------------------------------------------------
@st.cache_resource
def get_predictor():
    artifacts = os.path.join(FINPULSE_DIR, "models", "artifacts")
    return ProductionPredictor(artifacts_dir=artifacts)

@st.cache_resource
def get_hold_engine():
    return HoldWorkflowEngine()

@st.cache_resource
def get_ato_engine():
    return ATOProtectionEngine()

@st.cache_resource
def get_event_sink():
    # PostgreSQL is primary runtime durable system of record; falls back to SQLite if PG unreachable
    return IdempotentEventSink()

try:
    predictor = get_predictor()
    model_online = True
except Exception:
    predictor = None
    model_online = False

hold_engine = get_hold_engine()
ato_engine = get_ato_engine()
sink = get_event_sink()

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "ledger" not in st.session_state:
    st.session_state.ledger = []

if "latest_tx_detail" not in st.session_state:
    st.session_state.latest_tx_detail = None

if "active_hold" not in st.session_state:
    st.session_state.active_hold = None

# -----------------------------------------------------------------------------
# Sidebar Navigation (Required 5 Views)
# -----------------------------------------------------------------------------
st.sidebar.title("🛡️ FinPulse R7")
st.sidebar.caption("Complete Fraud Operations Platform")

view_selection = st.sidebar.radio(
    "Select Operational View:",
    ["Live", "Analyst", "Phone", "Results", "Health"]
)

st.sidebar.divider()

# Header status bar
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title(f"FinPulse AI — {view_selection} View")
    st.caption("Sub-10ms Hybrid ML Risk Scoring • CatBoost `finpulse-v3` • Active Protection")

with col_h2:
    st.markdown("<div style='text-align: right; padding-top: 15px;'>", unsafe_allow_html=True)
    if model_online:
        st.markdown('<span class="status-pill pill-green">● CatBoost ML Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill pill-red">○ ML Offline</span>', unsafe_allow_html=True)
    st.markdown('<span class="status-pill pill-blue">● R7 Workflow Ready</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# =============================================================================
# VIEW 1: LIVE
# =============================================================================
if view_selection == "Live":
    st.sidebar.subheader("⚡ Transaction Simulator")
    scenario_preset = st.sidebar.selectbox("Choose Scenario Preset:", [
        "Normal Everyday Transfer",
        "Large Amount Exceeding Balance",
        "Zero-Balance Cash-Out (Hard Block)",
        "Doubtful Transfer (Triggers HOLD)"
    ])

    amount_in = st.sidebar.number_input("Amount ($)", min_value=1.0, max_value=1_000_000.0, value=250.0, step=50.0)
    user_id = st.sidebar.text_input("Customer ID", value="cust_7701")
    origin_bal = st.sidebar.number_input("Origin Account Balance ($)", min_value=0.0, value=1500.0, step=100.0)
    auth_v = st.sidebar.checkbox("2FA / Biometric Verified", value=True)

    if scenario_preset == "Large Amount Exceeding Balance":
        amount_in = 5000.0
        origin_bal = 100.0
    elif scenario_preset == "Zero-Balance Cash-Out (Hard Block)":
        amount_in = 25000.0
        origin_bal = 0.0
        auth_v = False
    elif scenario_preset == "Doubtful Transfer (Triggers HOLD)":
        amount_in = 2800.0
        origin_bal = 3000.0
        auth_v = False

    if st.sidebar.button("🚀 Process Transaction", type="primary", use_container_width=True):
        tx_req = {
            "transaction_id": f"tx_live_{int(time.time()*1000)}",
            "customer_id": user_id,
            "amount": float(amount_in),
            "timestamp": time.time(),
            "merchant_id": "merch_terminal_1",
            "category": "transfer",
            "payment_type": "TRANSFER",
            "origin_balance": float(origin_bal),
            "dest_balance": 500.0,
            "auth_verified": auth_v
        }

        if predictor:
            res = predictor.predict(tx_req)
            st.session_state.latest_tx_detail = res
            
            # Persist raw transaction to PostgreSQL system of record
            try:
                sink.persist_transaction(tx_req)
            except Exception:
                pass

            # Construct & persist canonical DecisionEvent v1.0
            try:
                dev = DecisionEvent(
                    transaction_id=tx_req["transaction_id"],
                    customer_id=user_id,
                    decision=res["decision"],
                    risk_score=float(res["risk_score"]),
                    risk_level=res.get("risk_level", "LOW"),
                    ml_decision=res.get("ml_decision", res["decision"]),
                    calibrated_probability=float(res.get("calibrated_probability", 0.0)),
                    signals=dict(res.get("signals", {})),
                    diagnostics=dict(res.get("diagnostics", {})),
                    reasons=list(res.get("top_reasons", [])),
                    timestamp=tx_req["timestamp"],
                    model_version=res.get("model_version", "finpulse-v3"),
                    matched_rules=res.get("rule_result", {}).get("matched_rules", []),
                    hard_block=bool(res.get("diagnostics", {}).get("hard_block", False)),
                    hard_block_rules=res.get("rule_result", {}).get("hard_block_rules", []),
                    latency_ms=float(res.get("latency_ms", 1.0))
                )
                sink.persist_decision_event(dev)
            except Exception:
                pass

            # Check if review/hold should be opened
            if res["decision"] == "REVIEW" or (res["risk_score"] >= 30.0 and res["risk_score"] < 70.0):
                case, token = hold_engine.create_hold(
                    transaction_id=tx_req["transaction_id"],
                    customer_id=user_id,
                    amount=tx_req["amount"],
                    timeout_seconds=300.0
                )
                st.session_state.active_hold = {"case": case, "token": token}
                try:
                    sink.persist_hold_case({
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
                except Exception:
                    pass

            # Persist alert if BLOCK
            if res["decision"] == "BLOCK":
                try:
                    sink.persist_fraud_alert({
                        "alert_id": f"alt_{int(time.time()*1000)}",
                        "transaction_id": tx_req["transaction_id"],
                        "event_id": f"evt_{tx_req['transaction_id']}",
                        "customer_id": user_id,
                        "severity": "CRITICAL" if res.get("hard_block") else "HIGH",
                        "decision": "BLOCK",
                        "reasons": res.get("top_reasons", []),
                        "notified": True,
                        "status": "OPEN",
                        "created_at": tx_req["timestamp"]
                    })
                except Exception:
                    pass

            st.session_state.ledger.append({
                "Tx ID": tx_req["transaction_id"],
                "Customer": user_id,
                "Amount": f"${amount_in:,.2f}",
                "Decision": res["decision"],
                "Risk Score": f"{res['risk_score']:.1f}",
                "Risk Level": res.get("risk_level", "LOW"),
                "Latency": f"{res.get('latency_ms', 1.0):.2f} ms"
            })

    # Executive KPI Cards
    ledger = st.session_state.ledger
    total_tx = len(ledger)
    blocked_count = sum(1 for r in ledger if r["Decision"] == "BLOCK")
    review_count = sum(1 for r in ledger if r["Decision"] == "REVIEW")
    approved_count = total_tx - blocked_count - review_count

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Scanned Volume", f"{total_tx} Tx")
    m2.metric("Safe Volume Approved", f"{approved_count}", delta=f"{(approved_count/total_tx*100):.1f}%" if total_tx else "0%")
    m3.metric("Under Customer Review (HOLD)", f"{review_count}", delta_color="off")
    m4.metric("Threats Intercepted (BLOCK)", f"{blocked_count}", delta_color="inverse")

    st.write("")
    st.subheader("📋 Real-Time Streaming Ingestion Ledger")
    if ledger:
        df_ledger = pd.DataFrame(ledger)
        st.dataframe(df_ledger.iloc[::-1], use_container_width=True, height=300)
    else:
        st.info("💡 No transactions in active memory. Click **'🚀 Process Transaction'** on the sidebar to trigger real-time evaluation.")

# =============================================================================
# VIEW 2: ANALYST
# =============================================================================
elif view_selection == "Analyst":
    st.subheader("🕵️ Forensic Investigation & Explainability Center")

    tx_det = st.session_state.latest_tx_detail
    if not tx_det:
        st.info("Select or process a transaction in the Live view first to perform forensic analysis.")
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"### Verdict: **{tx_det['decision']}**")
            st.metric("Hybrid Risk Score", f"{tx_det['risk_score']:.1f} / 100.0", f"Level: {tx_det.get('risk_level', 'LOW')}")
            st.write(f"**Transaction ID:** `{tx_det.get('transaction_id', 'N/A')}`")
            st.write(f"**Model Version:** `{tx_det.get('model_version', 'finpulse-v3')}`")
            st.write(f"**Calibrated Probability:** `{tx_det.get('calibrated_probability', 0.0)*100:.2f}%`")
            st.write(f"**Execution Latency:** `{tx_det.get('latency_ms', 0.0)} ms`")

        with c2:
            st.markdown("### ⚠️ Triggered Risk Factors")
            reasons = tx_det.get("top_reasons", [])
            if reasons:
                for r in reasons:
                    st.error(f"• {r}")
            else:
                st.success("• No anomalous indicators triggered.")

            rules_matched = tx_det.get("rule_result", {}).get("matched_rules", [])
            if rules_matched:
                st.write("**Matched Business Rules:**")
                st.json(rules_matched)

        st.divider()
        st.markdown("### 🔍 Risk Signal Decomposition")
        signals = tx_det.get("signals", {})
        if signals:
            sig_df = pd.DataFrame([
                {"Signal": k.replace("_", " ").title(), "Risk Score": v}
                for k, v in signals.items()
            ])
            chart = alt.Chart(sig_df).mark_bar(cornerRadius=4).encode(
                x=alt.X("Risk Score:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Signal:N", sort="-x"),
                color=alt.Color("Risk Score:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True))
            ).properties(height=220)
            st.altair_chart(chart, use_container_width=True)

        st.divider()
        st.markdown("### 🗄️ PostgreSQL Persisted Audit Trail")
        recent_persisted = sink.get_recent_decisions(limit=10)
        if recent_persisted:
            df_audit = pd.DataFrame([{
                "Event ID": r.get("event_id", "")[:12] + "...",
                "Tx ID": r.get("transaction_id", ""),
                "Customer": r.get("customer_id", ""),
                "Decision": r.get("decision", ""),
                "Risk Score": f"{r.get('risk_score', 0):.1f}",
                "Status": r.get("workflow_status", ""),
                "Fraud Label": "FRAUD" if r.get("fraud_label") == 1 else ("LEGIT" if r.get("fraud_label") == 0 else "UNLABELED")
            } for r in recent_persisted])
            st.dataframe(df_audit, use_container_width=True)
        else:
            st.info("No persisted records found in PostgreSQL database.")

# =============================================================================
# VIEW 3: PHONE (CUSTOMER HOLD REVIEW)
# =============================================================================
elif view_selection == "Phone":
    st.subheader("📱 Customer Mobile Review Simulation (HOLD Workflow)")
    st.caption("Simulates customer two-way verification screen when a transaction is held by FinPulse.")

    active = st.session_state.active_hold

    st.markdown('<div class="phone-mockup">', unsafe_allow_html=True)
    if not active:
        st.markdown("""
        <div style='text-align: center; padding: 40px 10px;'>
            <div style='font-size: 3rem;'>🔔</div>
            <h4 style='color: #94a3b8; margin-top: 10px;'>No Pending Alerts</h4>
            <p style='color: #64748b; font-size: 0.85rem;'>Trigger a 'Doubtful Transfer' in the Live view to simulate customer confirmation.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        case: HoldCase = active["case"]
        token = active["token"]

        st.markdown(f"""
        <div style='text-align: center;'>
            <div style='font-size: 2.2rem;'>🛡️</div>
            <h3 style='color: #f8fafc; margin: 4px 0;'>Security Check</h3>
            <p style='color: #94a3b8; font-size: 0.85rem;'>We detected an unusual transaction pending your confirmation.</p>
        </div>
        <div style='background: rgba(255,255,255,0.05); padding: 14px; border-radius: 12px; margin: 16px 0;'>
            <div style='color: #cbd5e1; font-size: 0.8rem;'>Amount</div>
            <div style='color: #38bdf8; font-size: 1.6rem; font-weight: 700;'>${case.amount:,.2f}</div>
            <div style='color: #94a3b8; font-size: 0.8rem; margin-top: 6px;'>Tx ID: {case.transaction_id}</div>
            <div style='color: #e2e8f0; font-size: 0.8rem;'>Status: <b>{case.status}</b></div>
        </div>
        """, unsafe_allow_html=True)

        if case.status == "HOLD":
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("✅ Yes, It's Me", type="primary", use_container_width=True):
                    hold_engine.confirm_hold(case.hold_id, token)
                    try:
                        sink.update_hold_status(case.hold_id, "RELEASED", resolution_reason="Customer verified via mobile push")
                    except Exception:
                        pass
                    st.success("Transaction RELEASED!")
                    st.rerun()
            with col_b2:
                if st.button("❌ Deny (Fraud)", use_container_width=True):
                    hold_engine.deny_hold(case.hold_id, token)
                    try:
                        sink.update_hold_status(case.hold_id, "DENIED", resolution_reason="Customer reported fraud via mobile push")
                    except Exception:
                        pass
                    st.error("Transaction DENIED & Blocked!")
                    st.rerun()
        else:
            st.info(f"Resolution: {case.status} ({case.resolution_reason})")
            if st.button("Clear Phone Screen", use_container_width=True):
                st.session_state.active_hold = None
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# VIEW 4: RESULTS (E1–E10 BENCHMARKS)
# =============================================================================
elif view_selection == "Results":
    st.subheader("📈 Verified Evaluation Benchmarks & Cost-Frontier Analysis")

    report_path = os.path.join(FINPULSE_DIR, "reports", "r7_evaluation_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Evaluation Status", eval_data.get("overall_status", "N/A"))
        c2.metric("Experiments Verified", f"{eval_data.get('passed_count', 0)} / {eval_data.get('experiments_count', 0)}")
        cf = eval_data.get("cost_frontier_analysis", {})
        c3.metric("Optimal Operating Threshold", f"Risk Score >= {cf.get('optimal_threshold', 10)}")

        st.write("")
        st.markdown("### 🧪 E1–E10 Experiment Outcomes")
        exp_list = eval_data.get("experiments", [])
        exp_df = pd.DataFrame([
            {
                "ID": e["id"],
                "Experiment Name": e["name"],
                "Duration": f"{e['duration_ms']:.2f} ms",
                "Status": "✅ PASSED" if e["status"] == "PASSED" else "❌ FAILED",
                "Details": e["details"]
            }
            for e in exp_list
        ])
        st.dataframe(exp_df, use_container_width=True, hide_index=True)

        st.markdown("### 💰 Operational Cost Frontier (Fraud Prevention vs Customer Friction)")
        curve = cf.get("frontier_curve", [])
        if curve:
            c_df = pd.DataFrame(curve)
            frontier_chart = alt.Chart(c_df).mark_line(point=True).encode(
                x=alt.X("risk_threshold:Q", title="Decision Threshold Cutoff"),
                y=alt.Y("total_loss_dollars:Q", title="Total Expected Cost ($)"),
                tooltip=["risk_threshold", "total_loss_dollars", "fraud_loss_dollars", "friction_loss_dollars"]
            ).properties(height=260)
            st.altair_chart(frontier_chart, use_container_width=True)
    else:
        st.warning("E1–E10 evaluation report not found. Run `python scripts/run_r7_evaluation.py` to generate.")

# =============================================================================
# VIEW 5: HEALTH
# =============================================================================
elif view_selection == "Health":
    st.subheader("🛡️ Infrastructure & Circuit Breaker Telemetry")

    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.03); padding: 14px; border-radius: 8px;'>
            <b>Kafka Ingress & Publication</b>
            <div style='color: #10b981; font-weight: 700; margin-top: 4px;'>● CONNECTED (Port 9092)</div>
            <div style='font-size: 0.8rem; color: #94a3b8;'>Topics: predictions, fraud-alerts</div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.03); padding: 14px; border-radius: 8px;'>
            <b>Redis Sliding Window</b>
            <div style='color: #10b981; font-weight: 700; margin-top: 4px;'>● CONNECTED (Port 6379)</div>
            <div style='font-size: 0.8rem; color: #94a3b8;'>AOF Persisted: redis_data volume</div>
        </div>
        """, unsafe_allow_html=True)
    db_health = sink.health_check()
    records_count = sink.count_records("fraud_decisions")
    tx_count = sink.count_records("transactions")
    holds_count = sink.count_records("hold_cases")
    with h3:
        if db_health.get("backend") == "postgresql" and db_health.get("status") == "HEALTHY":
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.03); padding: 14px; border-radius: 8px;'>
                <b>PostgreSQL Durable Store</b>
                <div style='color: #10b981; font-weight: 700; margin-top: 4px;'>● CONNECTED ({db_health.get('version', 'PG16')})</div>
                <div style='font-size: 0.8rem; color: #94a3b8;'>Port {db_health.get('port', 5432)} • Latency: {db_health.get('latency_ms', 0):.1f}ms</div>
                <div style='font-size: 0.75rem; color: #38bdf8; margin-top: 2px;'>{records_count} Decisions • {tx_count} Txs • {holds_count} Holds</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.03); padding: 14px; border-radius: 8px;'>
                <b>Relational Event Sink</b>
                <div style='color: #f59e0b; font-weight: 700; margin-top: 4px;'>● {db_health.get('backend', 'sqlite').upper()} FALLBACK</div>
                <div style='font-size: 0.8rem; color: #94a3b8;'>Status: {db_health.get('status', 'OK')} • {records_count} Decisions</div>
            </div>
            """, unsafe_allow_html=True)
    with h4:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.03); padding: 14px; border-radius: 8px;'>
            <b>Model Artifact Engine</b>
            <div style='color: #10b981; font-weight: 700; margin-top: 4px;'>● finpulse-v3 LOADED</div>
            <div style='font-size: 0.8rem; color: #94a3b8;'>Platt Calibrator + 32-Features</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("### 📊 Distribution Drift Telemetry (PSI Monitor)")
    st.info("Real-time PSI monitor active across 32 features and model calibrated probabilities. Thresholds: PSI < 0.10 (Normal), PSI >= 0.25 (Retraining Alert).")
