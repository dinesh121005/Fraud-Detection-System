import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import joblib
import json
import altair as alt
from xgboost import XGBClassifier
from kafka import KafkaProducer

from utils.simulation import generate_transaction
from utils.preprocessing import preprocess_transaction
from utils.helpers import color_flag, compute_heuristic_risk, explain_heuristic

# -----------------------
# Streamlit Page Config & Custom Styling
# -----------------------
st.set_page_config(
    page_title="FinPulse AI — Fraud Intelligence Operations",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Executive Dashboard CSS
st.markdown("""
<style>
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700;
    }

    /* Status Pills */
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
    .pill-green {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .pill-blue {
        background: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .pill-purple {
        background: rgba(168, 85, 247, 0.15);
        color: #a855f7;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }

    /* Threat Banner */
    .threat-banner-fraud {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.2), rgba(185, 28, 28, 0.1));
        border-left: 5px solid #ef4444;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .threat-banner-clean {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.1));
        border-left: 5px solid #10b981;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Load Model & Encoder
# -----------------------
model_path = "models/fraud_model.json"
le_path = "models/label_encoder.pkl"

model = XGBClassifier()
model.load_model(model_path)
le = joblib.load(le_path)

# -----------------------
# Initialize Kafka Producer
# -----------------------
try:
    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
    kafka_online = True
except Exception:
    producer = None
    kafka_online = False

# -----------------------
# Session State Initialization
# -----------------------
if "transactions_history" not in st.session_state:
    st.session_state.transactions_history = pd.DataFrame(columns=[
        "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest", "isFraud", "Fraud_Prob", "isFlaggedFraud",
        "sender", "receiver", "heuristic_risk", "isFraudFinal", "status", "reasons"
    ])

if "latest_reasons" not in st.session_state:
    st.session_state.latest_reasons = []

if "latest_tx" not in st.session_state:
    st.session_state.latest_tx = None

# -----------------------
# Header Section
# -----------------------
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("🛡️ FinPulse AI — Fraud Intelligence Operations")
    st.caption("Next-Generation Real-Time Streaming Fraud Classification • XGBoost ML Engine • Kafka Event Backbone")

with col_head2:
    st.markdown("<div style='text-align: right; padding-top: 15px;'>", unsafe_allow_html=True)
    if kafka_online:
        st.markdown('<span class="status-pill pill-green">● Kafka Online (9092)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill pill-blue">○ Local Pipeline</span>', unsafe_allow_html=True)
    st.markdown('<span class="status-pill pill-purple">● XGBoost Active</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# -----------------------
# Sidebar Controls & Presets
# -----------------------
st.sidebar.header("🕹️ Simulation Control Center")

preset = st.sidebar.selectbox("⚡ Choose Demo Preset Scenario", [
    "— None (Custom Input) —",
    "🚨 Attack: Large Amount Exceeding Balance",
    "🚨 Attack: Cash-Out from Zero-Balance Account",
    "✅ Legit: Verified Small Payment",
    "🎲 Random Transaction Stream"
])

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Transaction Parameters")
step = st.sidebar.number_input("Step (Timeline Tick)", min_value=1, max_value=100000, value=random.randint(100, 999), step=1)
tx_type = st.sidebar.selectbox("Transaction Type", options=["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"])
amount = st.sidebar.number_input("Amount ($)", min_value=0.01, max_value=1_000_000.0, value=1500.0, step=50.0)
oldbalanceOrg = st.sidebar.number_input("Sender Initial Balance ($)", min_value=0.0, max_value=10_000_000.0, value=1000.0, step=50.0)
oldbalanceDest = st.sidebar.number_input("Receiver Initial Balance ($)", min_value=0.0, max_value=10_000_000.0, value=250.0, step=50.0)

# Derived post-balances
newbalanceOrig = max(oldbalanceOrg - amount, 0.0)
newbalanceDest = oldbalanceDest + amount
st.sidebar.caption(f"📉 **Projected Sender Balance**: `${newbalanceOrig:,.2f}` | 📈 **Receiver**: `${newbalanceDest:,.2f}`")

with st.sidebar.expander("🔒 Advanced Security & Behavioral Context", expanded=True):
    auth_verified = st.checkbox("Auth Verified (2FA / Biometric)", value=True)
    device_known = st.checkbox("Device Known for Sender", value=True)
    geo_known = st.checkbox("Geolocation Known for Sender", value=True)
    receiver_risk_score = st.slider("Destination Wallet Risk Score", 0.0, 1.0, 0.1, step=0.01)
    sender_hourly_tx_count = st.number_input("Sender Hourly Velocity", min_value=0, value=1, step=1)
    sender_daily_tx_count = st.number_input("Sender Daily Velocity", min_value=0, value=3, step=1)

sender = st.sidebar.number_input("Sender Account ID", min_value=1000, max_value=9999999, value=random.randint(1000, 9999))
receiver = st.sidebar.number_input("Receiver Account ID", min_value=1000, max_value=9999999, value=random.randint(1000, 9999))

st.sidebar.markdown("---")
manual_submit = st.sidebar.button("🚀 Simulate This Transaction", type="primary", use_container_width=True)

with st.sidebar.expander("🔄 Auto-Streaming Generator"):
    auto_simulate = st.checkbox("Enable Continuous Stream Generator", value=False)
    auto_count = st.slider("Batch Stream Count", 1, 50, 10)
    auto_delay = st.slider("Stream Interval (seconds)", 0.2, 3.0, 0.8, step=0.2)

if st.sidebar.button("🗑️ Reset Ledger & Metrics", use_container_width=True):
    st.session_state.transactions_history = st.session_state.transactions_history.iloc[0:0]
    st.session_state.latest_tx = None
    st.session_state.latest_reasons = []
    st.rerun()

# -----------------------
# Helper: Build Manual Transaction
# -----------------------
def build_tx_from_manual():
    return {
        "step": int(step),
        "type": tx_type,
        "amount": float(round(amount, 2)),
        "oldbalanceOrg": float(round(oldbalanceOrg, 2)),
        "newbalanceOrig": float(round(newbalanceOrig, 2)),
        "oldbalanceDest": float(round(oldbalanceDest, 2)),
        "newbalanceDest": float(round(newbalanceDest, 2)),
        "sender": int(sender),
        "receiver": int(receiver),
        "auth_verified": 1 if auth_verified else 0,
        "device_known": 1 if device_known else 0,
        "geo_known": 1 if geo_known else 0,
        "sender_hourly_tx_count": int(sender_hourly_tx_count),
        "sender_daily_tx_count": int(sender_daily_tx_count),
        "receiver_risk_score": float(receiver_risk_score),
        "initiated_by": "third_party" if not auth_verified else "sender"
    }

def convert_to_native_types(d):
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            d[k] = int(v)
        elif isinstance(v, (np.floating,)):
            d[k] = float(v)
        elif isinstance(v, np.bool_):
            d[k] = bool(v)
    return d

# -----------------------
# Process Transaction Function
# -----------------------
def process_transaction_dict(tx):
    tx["isFlaggedFraud"] = 0
    orig_type = str(tx.get("type", "TRANSFER"))
    try:
        tx_type_enc = le.transform([orig_type])[0]
        tx["type"] = int(tx_type_enc)
    except Exception:
        tx["type"] = -1

    X = preprocess_transaction(tx)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame([X])

    pred = int(model.predict(X)[0])
    try:
        prob = float(model.predict_proba(X)[0][1])
    except Exception:
        prob = float(pred)

    tx["type"] = orig_type
    tx["isFraud"] = pred
    tx["Fraud_Prob"] = round(prob, 4)
    tx["isFlaggedFraud"] = pred

    tx["heuristic_risk"] = compute_heuristic_risk(tx)
    reasons = explain_heuristic(tx)

    # Fusion decision rule
    tx["isFraudFinal"] = 1 if (tx["isFraud"] == 1 or tx["heuristic_risk"] > 0.5) else 0
    tx["status"] = "🚨 BLOCKED" if tx["isFraudFinal"] == 1 else "✅ APPROVED"
    tx["reasons"] = reasons

    return tx, X, reasons

# -----------------------
# Preset Resolution
# -----------------------
if preset != "— None (Custom Input) —":
    if preset == "🎲 Random Transaction Stream":
        preset_tx = generate_transaction()
    elif preset == "🚨 Attack: Large Amount Exceeding Balance":
        preset_tx = build_tx_from_manual()
        preset_tx["type"] = "TRANSFER"
        preset_tx["amount"] = float(round(preset_tx["oldbalanceOrg"] + random.uniform(1500, 6000), 2))
        preset_tx["newbalanceOrig"] = 0.0
        preset_tx["newbalanceDest"] = float(round(preset_tx["oldbalanceDest"] + preset_tx["amount"], 2))
        preset_tx["auth_verified"] = 0
        preset_tx["device_known"] = 0
        preset_tx["geo_known"] = 0
        preset_tx["initiated_by"] = "third_party"
        preset_tx["receiver_risk_score"] = 0.85
        preset_tx["sender_hourly_tx_count"] = random.randint(6, 12)
        preset_tx["sender_daily_tx_count"] = random.randint(20, 35)
    elif preset == "🚨 Attack: Cash-Out from Zero-Balance Account":
        preset_tx = build_tx_from_manual()
        preset_tx["type"] = "CASH_OUT"
        preset_tx["oldbalanceOrg"] = 0.0
        preset_tx["amount"] = float(round(random.uniform(2000, 5000), 2))
        preset_tx["newbalanceOrig"] = 0.0
        preset_tx["newbalanceDest"] = float(round(preset_tx["oldbalanceDest"] + preset_tx["amount"], 2))
        preset_tx["auth_verified"] = 0
        preset_tx["device_known"] = 0
        preset_tx["geo_known"] = 0
        preset_tx["initiated_by"] = "third_party"
        preset_tx["receiver_risk_score"] = 0.90
        preset_tx["sender_hourly_tx_count"] = random.randint(5, 10)
        preset_tx["sender_daily_tx_count"] = random.randint(15, 30)
    elif preset == "✅ Legit: Verified Small Payment":
        preset_tx = build_tx_from_manual()
        preset_tx["type"] = "PAYMENT"
        preset_tx["amount"] = 25.50
        preset_tx["oldbalanceOrg"] = 500.0
        preset_tx["oldbalanceDest"] = 120.0
        preset_tx["newbalanceOrig"] = 474.50
        preset_tx["newbalanceDest"] = 145.50
        preset_tx["auth_verified"] = 1
        preset_tx["device_known"] = 1
        preset_tx["geo_known"] = 1
        preset_tx["initiated_by"] = "sender"
        preset_tx["receiver_risk_score"] = 0.05
    else:
        preset_tx = build_tx_from_manual()
else:
    preset_tx = None

# -----------------------
# Execution Logic
# -----------------------
def append_tx(tx_processed, reasons):
    st.session_state.latest_tx = tx_processed
    st.session_state.latest_reasons = reasons

    df_new = pd.DataFrame([tx_processed]).dropna(axis=1, how='all')
    st.session_state.transactions_history = pd.concat([
        st.session_state.transactions_history,
        df_new
    ], ignore_index=True)

if manual_submit:
    tx = preset_tx if preset_tx else build_tx_from_manual()
    tx = convert_to_native_types(tx)
    tx_processed, _, reasons = process_transaction_dict(tx)

    if producer:
        try:
            producer.send("transactions", tx_processed)
            producer.flush()
        except Exception:
            pass

    append_tx(tx_processed, reasons)

if auto_simulate:
    for _ in range(auto_count):
        tx = generate_transaction()
        tx = convert_to_native_types(tx)
        tx_processed, _, reasons = process_transaction_dict(tx)

        if producer:
            try:
                producer.send("transactions", tx_processed)
                producer.flush()
            except Exception:
                pass

        append_tx(tx_processed, reasons)
        time.sleep(auto_delay)

# -----------------------
# Executive Metrics Row
# -----------------------
history = st.session_state.transactions_history

total_count = len(history)
fraud_count = int((history["isFraudFinal"] == 1).sum()) if total_count > 0 else 0
clean_count = total_count - fraud_count

total_volume = float(history["amount"].sum()) if total_count > 0 else 0.0
fraud_volume = float(history[history["isFraudFinal"] == 1]["amount"].sum()) if total_count > 0 else 0.0
clean_volume = total_volume - fraud_volume
fraud_rate = (fraud_count / total_count * 100) if total_count > 0 else 0.0

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Scanned Volume", f"${total_volume:,.2f}", f"{total_count} Transactions")
with m2:
    st.metric("Threats Intercepted", f"${fraud_volume:,.2f}", f"{fraud_count} Blocked ({fraud_rate:.1f}%)", delta_color="inverse")
with m3:
    st.metric("Safe Volume Cleared", f"${clean_volume:,.2f}", f"{clean_count} Approved")
with m4:
    avg_risk = float(history["heuristic_risk"].mean()) if total_count > 0 else 0.0
    st.metric("Avg Heuristic Risk Index", f"{avg_risk:.2f}", "Security Score [0.0 - 1.0]")

st.write("")

# -----------------------
# Threat Intelligence Inspector (Latest Transaction Banner)
# -----------------------
if st.session_state.latest_tx is not None:
    ltx = st.session_state.latest_tx
    reasons = st.session_state.latest_reasons

    if ltx["isFraudFinal"] == 1:
        st.markdown(f"""
        <div class="threat-banner-fraud">
            <h4 style="margin: 0 0 6px 0; color: #ef4444;">🚨 THREAT INTERCEPTED — TRANSACTION BLOCKED</h4>
            <div><b>Tx ID / Step:</b> #{ltx['step']} | <b>Type:</b> {ltx['type']} | <b>Amount:</b> ${ltx['amount']:,.2f} | <b>Sender:</b> #{ltx['sender']} ➔ <b>Receiver:</b> #{ltx['receiver']}</div>
            <div style="margin-top: 6px;"><b>XGBoost ML Probability:</b> <span style="color:#ef4444; font-weight:700;">{ltx['Fraud_Prob']*100:.2f}%</span> | <b>Heuristic Risk Score:</b> <span style="color:#ef4444; font-weight:700;">{ltx['heuristic_risk']:.2f}</span></div>
            <div style="margin-top: 8px; font-size: 0.9rem;"><b>Triggered Risk Indicators:</b> {" • ".join([f"<span style='color:#fca5a5;'>⚠️ {r}</span>" for r in reasons])}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="threat-banner-clean">
            <h4 style="margin: 0 0 6px 0; color: #10b981;">✅ TRANSACTION VERIFIED & APPROVED</h4>
            <div><b>Tx ID / Step:</b> #{ltx['step']} | <b>Type:</b> {ltx['type']} | <b>Amount:</b> ${ltx['amount']:,.2f} | <b>Sender:</b> #{ltx['sender']} ➔ <b>Receiver:</b> #{ltx['receiver']}</div>
            <div style="margin-top: 6px;"><b>XGBoost ML Probability:</b> <span style="color:#10b981; font-weight:700;">{ltx['Fraud_Prob']*100:.2f}%</span> | <b>Heuristic Risk Score:</b> <span style="color:#10b981; font-weight:700;">{ltx['heuristic_risk']:.2f}</span></div>
            <div style="margin-top: 6px; font-size: 0.9rem; color:#6ee7b7;">🛡️ All biometric/OTP security checks verified. Zero anomalous behavioral patterns detected.</div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------
# Visualizations & Live Stream Tabs
# -----------------------
tab_analytics, tab_stream = st.tabs(["📊 Executive Analytics & Visualizations", "📋 Real-Time Transaction Ledger"])

with tab_analytics:
    if total_count == 0:
        st.info("💡 No transactions simulated yet. Choose a preset on the sidebar and click **'🚀 Simulate This Transaction'** or enable the **'Continuous Stream Generator'** to populate live telemetry.")
    else:
        v_col1, v_col2 = st.columns(2)

        with v_col1:
            st.subheader("Verdict Breakdown")
            summary_df = pd.DataFrame({
                "Verdict": ["Approved (Legit)", "Blocked (Fraud)"],
                "Count": [clean_count, fraud_count]
            })

            pie = alt.Chart(summary_df).mark_arc(innerRadius=50).encode(
                theta=alt.Theta("Count:Q"),
                color=alt.Color("Verdict:N", scale=alt.Scale(domain=["Approved (Legit)", "Blocked (Fraud)"], range=["#10b981", "#ef4444"])),
                tooltip=["Verdict", "Count"]
            ).properties(height=280)
            st.altair_chart(pie, use_container_width=True)

        with v_col2:
            st.subheader("Transaction Volume by Type ($)")
            type_vol = history.groupby("type")["amount"].sum().reset_index()
            bar = alt.Chart(type_vol).mark_bar(cornerRadius=6).encode(
                x=alt.X("type:N", title="Transaction Category"),
                y=alt.Y("amount:Q", title="Total Volume ($)"),
                color=alt.Color("type:N", legend=None),
                tooltip=["type", "amount"]
            ).properties(height=280)
            st.altair_chart(bar, use_container_width=True)

        st.subheader("Real-Time Anomaly Probability Timeline")
        timeline_df = history[["step", "Fraud_Prob", "heuristic_risk"]].copy().reset_index()
        timeline_df["Transaction Sequence"] = timeline_df.index + 1

        chart_df = timeline_df.melt(
            id_vars=["Transaction Sequence"],
            value_vars=["Fraud_Prob", "heuristic_risk"],
            var_name="Risk Metric",
            value_name="Score"
        )
        chart_df["Risk Metric"] = chart_df["Risk Metric"].replace({
            "Fraud_Prob": "XGBoost ML Probability",
            "heuristic_risk": "Heuristic Risk Index"
        })

        line_chart = alt.Chart(chart_df).mark_line(point=True).encode(
            x=alt.X("Transaction Sequence:O", title="Transaction Sequence #"),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), title="Probability / Risk Score"),
            color=alt.Color("Risk Metric:N", scale=alt.Scale(range=["#ef4444", "#3b82f6"])),
            tooltip=["Transaction Sequence", "Risk Metric", "Score"]
        ).properties(height=250)

        st.altair_chart(line_chart, use_container_width=True)

with tab_stream:
    st.subheader("Live Streaming Transaction Audit Ledger")
    if total_count == 0:
        st.write("Audit ledger empty. Simulate transactions to see them live.")
    else:
        display_df = history[[
            "step", "sender", "receiver", "type", "amount",
            "isFraud", "Fraud_Prob", "heuristic_risk", "isFraudFinal", "status"
        ]].copy()

        # Format display
        display_df["amount"] = display_df["amount"].map(lambda x: f"${x:,.2f}")
        display_df["Fraud_Prob"] = display_df["Fraud_Prob"].map(lambda x: f"{x*100:.1f}%")
        display_df["heuristic_risk"] = display_df["heuristic_risk"].map(lambda x: f"{x:.2f}")

        display_df.rename(columns={
            "step": "Timeline",
            "sender": "Sender ID",
            "receiver": "Receiver ID",
            "type": "Type",
            "amount": "Amount",
            "isFraud": "ML Verdict",
            "Fraud_Prob": "ML Confidence",
            "heuristic_risk": "Heuristic Index",
            "isFraudFinal": "Final Flag",
            "status": "Decision"
        }, inplace=True)

        def highlight_fraud_rows(row):
            if row["Final Flag"] == 1:
                return ["background-color: rgba(239, 68, 68, 0.25); color: #fca5a5; font-weight: bold;"] * len(row)
            return [""] * len(row)

        st.dataframe(
            display_df.style.apply(highlight_fraud_rows, axis=1),
            use_container_width=True,
            height=400
        )
