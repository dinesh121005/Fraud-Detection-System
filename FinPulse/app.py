"""
FinPulse AI — Real-Time Fraud Authorization & Attack-Defense Ecosystem.

Architectural Journey:
1. CUSTOMER: My Wallet (Consumer banking experience, Send money, 2-way verification on HOLD, Report unauthorized)
2. ATTACKER: Attack Lab (Simulated adversary: Actor ID, moving locations, scenario library: ATO, Velocity, Impossible Travel, New Device, Novel Attack)
3. GATEWAY: Operations (Authorization boundary, Ingress stream, Live ledger, Security cases)
4. ANALYST: Investigations (Forensic inspection, SHAP explainability, Subsystem decomposition, Baseline comparison)
5. ML LIFECYCLE: Model Evolution (Known vs Novel taxonomy, Delayed feedback loop, Shadow scoring, Promotion Gate)
6. SYSTEM: Infrastructure Health (Telemetry, Kafka, Redis, PostgreSQL, PSI drift)
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import streamlit as st

# Ensure FinPulse in sys.path
FINPULSE_DIR = os.path.dirname(os.path.abspath(__file__))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

# Backend Authoritative Engines
from src.serving.predictor import ProductionPredictor
from src.workflow.hold_workflow import HoldWorkflowEngine
from src.workflow.account_events import ATOProtectionEngine
from src.persistence.sink import IdempotentEventSink
from src.challenger.shadow_scorer import ShadowModelRunner
from training.retrain import ModelPromotionGate

from src.simulation.demo_account import (
    DemoAccountManager,
    DEFAULT_DEMO_CUSTOMER_ID,
    DEFAULT_ATTACKER_RECEIVER_ID
)
from src.simulation.attacker import AttackerSimulator
from src.simulation.gateway import PaymentGatewaySimulator

# UI Modular Architecture
from src.ui.theme import apply_global_theme
from src.ui.layout import render_top_status_bar, render_sidebar_shell
from src.ui.wallet_view import render_customer_wallet
from src.ui.attack_lab_view import render_attack_lab
from src.ui.gateway_view import render_gateway_operations
from src.ui.analyst_view import render_analyst_forensics
from src.ui.lifecycle_view import render_ml_lifecycle
from src.ui.health_view import render_system_health

# -----------------------------------------------------------------------------
# Page Configuration & Design System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FinPulse AI — Real-Time Fraud Authorization Ecosystem",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_global_theme()

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
    return IdempotentEventSink()

@st.cache_resource
def get_account_manager():
    return DemoAccountManager()

@st.cache_resource
def get_gateway():
    return PaymentGatewaySimulator(
        predictor=get_predictor(),
        hold_engine=get_hold_engine(),
        ato_engine=get_ato_engine(),
        event_sink=get_event_sink(),
        account_manager=get_account_manager()
    )

def get_attacker():
    return AttackerSimulator(
        ato_engine=get_ato_engine(),
        event_sink=get_event_sink(),
        account_manager=get_account_manager()
    )

@st.cache_resource
def get_shadow_runner():
    class CandidateChallenger:
        def __init__(self, base_model=None):
            self.base_model = base_model
        def predict_proba(self, X):
            if self.base_model and hasattr(self.base_model, "predict_proba"):
                p = self.base_model.predict_proba(X)
                return np.clip(p * 1.05, 0.0, 1.0)
            return np.array([[0.20, 0.80]])

    pred = get_predictor()
    base = getattr(getattr(pred, "pipeline", None), "model", None)
    return ShadowModelRunner(
        challenger_model=CandidateChallenger(base),
        challenger_version="finpulse-v4-candidate"
    )

@st.cache_resource
def get_promotion_gate():
    return ModelPromotionGate()

try:
    predictor = get_predictor()
    model_online = True
except Exception:
    predictor = None
    model_online = False

hold_engine = get_hold_engine()
ato_engine = get_ato_engine()
sink = get_event_sink()
account_manager = get_account_manager()
gateway = get_gateway()
attacker = get_attacker()
shadow_runner = get_shadow_runner()
promotion_gate = get_promotion_gate()

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "ledger" not in st.session_state:
    st.session_state.ledger = []

if "wallet_tx_history" not in st.session_state:
    st.session_state.wallet_tx_history = []

if "latest_tx_detail" not in st.session_state:
    st.session_state.latest_tx_detail = None

if "active_hold" not in st.session_state:
    st.session_state.active_hold = None

if "attack_history" not in st.session_state:
    st.session_state.attack_history = []

if "last_attack_timeline" not in st.session_state:
    st.session_state.last_attack_timeline = []

if "last_gateway_result" not in st.session_state:
    st.session_state.last_gateway_result = None

if "phone_feedback" not in st.session_state:
    st.session_state.phone_feedback = None

if "reporting_tx_id" not in st.session_state:
    st.session_state.reporting_tx_id = None

if "last_reported_case" not in st.session_state:
    st.session_state.last_reported_case = None

# -----------------------------------------------------------------------------
# Dynamic Service Probing for Status Bar
# -----------------------------------------------------------------------------
kafka_up = False
try:
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.3)
    if sock.connect_ex(("127.0.0.1", 9092)) == 0:
        kafka_up = True
    sock.close()
except Exception:
    pass

db_health = sink.health_check() if hasattr(sink, "health_check") else {"status": "healthy"}
db_up = db_health.get("status") == "healthy"

health_summary = {
    "ml": model_online,
    "db": db_up,
    "kafka": kafka_up
}

# -----------------------------------------------------------------------------
# Global Status Bar
# -----------------------------------------------------------------------------
render_top_status_bar(health_summary)

# -----------------------------------------------------------------------------
# Reset Demo Callback
# -----------------------------------------------------------------------------
def reset_demo_state():
    account_manager.reset_all(DEFAULT_DEMO_CUSTOMER_ID)
    ato_engine.reset(DEFAULT_DEMO_CUSTOMER_ID)
    hold_engine.reset(DEFAULT_DEMO_CUSTOMER_ID)
    if predictor and hasattr(predictor, "redis_window"):
        try:
            predictor.redis_window.clear_customer_state(DEFAULT_DEMO_CUSTOMER_ID)
            predictor.redis_window.clear_customer_state(DEFAULT_ATTACKER_RECEIVER_ID)
        except Exception:
            pass
    try:
        sink.reset_demo_data(DEFAULT_DEMO_CUSTOMER_ID)
    except Exception:
        pass
    st.session_state.active_hold = None
    st.session_state.attack_history = []
    st.session_state.last_attack_timeline = []
    st.session_state.last_gateway_result = None
    st.session_state.latest_tx_detail = None
    st.session_state.ledger = []
    st.session_state.wallet_tx_history = []
    st.session_state.phone_feedback = None
    st.session_state.reporting_tx_id = None
    st.session_state.last_reported_case = None
    st.rerun()

# -----------------------------------------------------------------------------
# Sidebar Navigation Shell
# -----------------------------------------------------------------------------
demo_acc = account_manager.get_account(DEFAULT_DEMO_CUSTOMER_ID)
selected_view = render_sidebar_shell(
    demo_acc=demo_acc,
    ato_engine=ato_engine,
    sink=sink,
    attack_history_len=len(st.session_state.attack_history),
    on_reset_callback=reset_demo_state
)

# -----------------------------------------------------------------------------
# Main Experience Routing
# -----------------------------------------------------------------------------
v_norm = selected_view.lower()

if "customer" in v_norm or "wallet" in v_norm:
    render_customer_wallet(
        account_manager=account_manager,
        gateway=gateway,
        ato_engine=ato_engine,
        hold_engine=hold_engine,
        sink=sink,
        predictor=predictor
    )

elif "attack" in v_norm:
    render_attack_lab(
        attacker=attacker,
        gateway=gateway,
        account_manager=account_manager,
        ato_engine=ato_engine,
        sink=sink
    )

elif "gateway" in v_norm:
    render_gateway_operations(
        gateway=gateway,
        sink=sink
    )

elif "analyst" in v_norm or "investigation" in v_norm:
    render_analyst_forensics(
        sink=sink,
        attacker=attacker,
        gateway=gateway
    )

elif "lifecycle" in v_norm or "ml" in v_norm:
    render_ml_lifecycle(
        sink=sink,
        shadow_runner=shadow_runner,
        promotion_gate=promotion_gate
    )

elif "health" in v_norm or "system" in v_norm:
    render_system_health(
        sink=sink,
        predictor=predictor,
        model_online=model_online,
        finpulse_dir=FINPULSE_DIR
    )
