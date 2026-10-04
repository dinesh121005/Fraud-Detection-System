"""
FinPulse Global Application Shell & Layout.

Handles:
- Top dynamic ecosystem status bar
- Demo environment banner
- Unified sidebar navigation and customer account card
"""

import time
import streamlit as st
from src.ui.icons import get_icon, render_icon_badge

def render_top_status_bar(health_dict: dict):
    """
    Renders top status bar probing real backend subsystems:
    - Wallet
    - Gateway Authorization
    - CatBoost Model
    - Hybrid Risk Engine
    - ATO Guard
    - PostgreSQL
    - Kafka
    """
    st.markdown(f"""
    <div style='background: linear-gradient(90deg, rgba(14, 165, 233, 0.12) 0%, rgba(168, 85, 247, 0.12) 100%); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 10px; padding: 8px 18px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;'>
        <div style='display: flex; align-items: center; gap: 8px;'>
            {get_icon("shield", size=16, color="#38bdf8")}
            <span style='color: #f8fafc; font-weight: 700; font-size: 0.84rem; letter-spacing: 0.02em;'>DEMO ENVIRONMENT</span>
            <span style='color: #64748b;'>|</span>
            <span style='color: #94a3b8; font-size: 0.8rem;'>Financial Fraud Authorization Ecosystem</span>
        </div>
        <div style='display: flex; align-items: center; gap: 12px; font-size: 0.78rem; color: #cbd5e1;'>
            <span>Synthetic Scenarios • <b>Real FinPulse Decision Engine</b></span>
            <span style='color: #475569;'>•</span>
            <span style='color: #38bdf8;'>Gateway Node: Chennai Core</span>
        </div>
    </div>
    <div style='background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 7px 14px; margin-bottom: 20px; display: flex; gap: 14px; align-items: center; justify-content: center; font-size: 0.76rem; font-weight: 600; color: #cbd5e1; flex-wrap: wrap;'>
        <span style='display: flex; align-items: center; gap: 5px;'>{get_icon("wallet", size=13, color="#10b981")} WALLET <span style='color: #10b981;'>●</span></span>
        <span style='color: #334155;'>•</span>
        <span style='display: flex; align-items: center; gap: 5px;'>{get_icon("shield", size=13, color="#10b981")} GATEWAY <span style='color: #10b981;'>●</span></span>
        <span style='color: #334155;'>•</span>
        <span style='display: flex; align-items: center; gap: 5px;'>{get_icon("brain", size=13, color="#10b981" if health_dict.get("ml") else "#ef4444")} CATBOOST (finpulse-v3) <span style='color: {"#10b981" if health_dict.get("ml") else "#ef4444"};'>●</span></span>
        <span style='color: #334155;'>•</span>
        <span style='display: flex; align-items: center; gap: 5px;'>{get_icon("activity", size=13, color="#10b981")} HYBRID RISK <span style='color: #10b981;'>●</span></span>
        <span style='color: #334155;'>•</span>
        <span style='display: flex; align-items: center; gap: 5px;'>{get_icon("lock", size=13, color="#10b981")} ATO GUARD <span style='color: #10b981;'>●</span></span>
        <span style='color: #334155;'>•</span>
        <span style='display: flex; align-items: center; gap: 5px;'>{get_icon("database", size=13, color="#10b981" if health_dict.get("db") else "#ef4444")} POSTGRESQL <span style='color: {"#10b981" if health_dict.get("db") else "#ef4444"};'>●</span></span>
        <span style='color: #334155;'>•</span>
        <span style='display: flex; align-items: center; gap: 5px;'>{get_icon("server", size=13, color="#10b981" if health_dict.get("kafka") else "#94a3b8")} KAFKA <span style='color: {"#10b981" if health_dict.get("kafka") else "#94a3b8"};'>●</span></span>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_shell(demo_acc, ato_engine, sink, attack_history_len: int, on_reset_callback):
    """
    Renders the professional FinPulse sidebar navigation and demo account status card.
    """
    st.sidebar.markdown(f"""
    <div style='padding: 6px 0 16px 0;'>
        <div style='display: flex; align-items: center; gap: 8px;'>
            {get_icon("shield", size=22, color="#38bdf8")}
            <span style='font-size: 1.25rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;'>FINPULSE</span>
        </div>
        <div style='font-size: 0.76rem; color: #94a3b8; margin-top: 2px;'>Real-Time Fraud Authorization Ecosystem</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div style='font-size: 0.7rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;'>
        EXPERIENCES
    </div>
    """, unsafe_allow_html=True)

    experiences = [
        "Customer Wallet",
        "Attack Lab",
        "Gateway Operations",
        "Fraud Investigation",
        "ML Lifecycle",
        "System Health"
    ]

    selected_view = st.sidebar.radio(
        "Select Experience:",
        experiences,
        index=0,
        label_visibility="collapsed",
        key="nav_main_experience"
    )

    st.sidebar.markdown("<div style='margin: 18px 0 12px 0; border-top: 1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div style='font-size: 0.7rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;'>
        DEMO ACCOUNT
    </div>
    """, unsafe_allow_html=True)

    status_pill_cls = "pill-green" if demo_acc.account_status == "SAFE" else ("pill-amber" if demo_acc.account_status == "UNDER_ATTACK" else "pill-red")

    cases_count = len(sink.get_security_cases(customer_id=demo_acc.customer_id)) if hasattr(sink, "get_security_cases") else 0

    st.sidebar.markdown(f"""
    <div style='background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px 14px; margin-bottom: 12px;'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <span style='font-size: 0.78rem; font-weight: 600; color: #cbd5e1;'><code>{demo_acc.customer_id}</code></span>
            <span class="status-pill {status_pill_cls}">{demo_acc.account_status}</span>
        </div>
        <div style='font-size: 1.4rem; font-weight: 800; color: #38bdf8; margin: 6px 0 4px 0;'>
            ₹{demo_acc.balance:,.2f}
        </div>
        <div style='font-size: 0.74rem; color: #94a3b8; display: flex; justify-content: space-between; margin-top: 4px;'>
            <span>Loc: <code>{demo_acc.known_city}</code></span>
            <span>Device: <code>{demo_acc.known_device_id}</code></span>
        </div>
        <div style='border-top: 1px solid rgba(255,255,255,0.06); margin-top: 8px; padding-top: 6px; font-size: 0.72rem; color: #64748b; display: flex; justify-content: space-between;'>
            <span>Attacks: <b>{attack_history_len}</b></span>
            <span>Security Cases: <b>{cases_count}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("Reset Demo State", width="stretch", type="secondary", key="btn_reset_demo_sidebar"):
        on_reset_callback()

    st.sidebar.markdown("""
    <div style='margin-top: 24px; text-align: center; font-size: 0.72rem; color: #475569;'>
        FinPulse v3.0 • Production Engine
    </div>
    """, unsafe_allow_html=True)

    return selected_view
