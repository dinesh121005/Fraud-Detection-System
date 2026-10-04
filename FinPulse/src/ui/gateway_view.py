"""
FinPulse Payment Gateway Operations Console.

Provides an authoritative payment operations experience:
- Live Gateway & FinPulse Authorization status
- Volume metrics (Approved, Held, Blocked, Total Funds Protected)
- Real-time live transaction stream
- Deep Transaction Authorization Inspector
- Security Case Management for the ML feedback loop
"""

import time
import json
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Optional

from src.ui.icons import get_icon, render_icon_badge
from src.ui.decision_lineage import render_decision_lineage

def render_gateway_operations(gateway, sink):
    """Renders the Payment Gateway Operations view."""
    st.markdown("""
    <div style='margin-bottom: 16px;'>
        <h2 style='margin: 0; font-size: 1.6rem; font-weight: 800; color: #f8fafc; display: flex; align-items: center; gap: 10px;'>
            """ + get_icon("server", size=24, color="#38bdf8") + """
            Payment Gateway Operations & Security Console
        </h2>
        <div style='color: #94a3b8; font-size: 0.85rem; margin-top: 2px;'>
            Authoritative Settlement Boundary • Real-Time Ingress • Live Security Case Ledger
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TOP STATUS & VOLUME METRICS
    # -------------------------------------------------------------------------
    recent_decs = sink.get_recent_decisions(limit=100) if hasattr(sink, "get_recent_decisions") else []
    
    app_cnt = sum(1 for d in recent_decs if d.get("decision") == "APPROVE")
    hold_cnt = sum(1 for d in recent_decs if d.get("decision") in ["HOLD", "REVIEW"])
    block_cnt = sum(1 for d in recent_decs if d.get("decision") == "BLOCK")
    if not recent_decs:
        app_cnt, hold_cnt, block_cnt = 0, 0, 0

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Gateway Core", "ONLINE", "Latency < 5ms (Chennai)")
    with col_m2:
        st.metric("Transactions Approved", app_cnt, "Funds Settled")
    with col_m3:
        st.metric("Placed on Hold", hold_cnt, "Pending 2FA Verification")
    with col_m4:
        st.metric("Threats Blocked", block_cnt, "Funds Fully Protected", delta_color="inverse")

    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

    tab_stream, tab_inspect, tab_cases = st.tabs([
        "Live Transaction Stream",
        "Authorization Inspector",
        "Security Cases (Feedback Loop)"
    ])

    # --- TAB 1: LIVE TRANSACTION STREAM ---
    with tab_stream:
        st.markdown("### Live Transaction Ledger")
        db_txs = sink.get_recent_transactions(limit=25) if hasattr(sink, "get_recent_transactions") else []

        if db_txs:
            rows = []
            for t in db_txs:
                tid = t.get("transaction_id", "")
                dec_rec = sink.get_decision(tid) if hasattr(sink, "get_decision") else None
                dec = dec_rec.get("decision", "APPROVE") if dec_rec else "APPROVE"
                score = float(dec_rec.get("risk_score", 0.0)) if dec_rec else 0.0
                amt = float(t.get("amount", 0.0))

                rows.append({
                    "Time": time.strftime('%H:%M:%S', time.localtime(float(t.get("timestamp", time.time())))),
                    "Tx ID": tid,
                    "Customer": t.get("customer_id", ""),
                    "Recipient": t.get("merchant_id", ""),
                    "Amount": f"₹{amt:,.2f}",
                    "Risk Score": f"{score:.1f}",
                    "Decision": dec,
                    "Funds Transferred": f"₹{amt:,.2f}" if dec == "APPROVE" else "₹0.00 (Protected)"
                })
            df_stream = pd.DataFrame(rows)
            st.dataframe(df_stream, width="stretch", hide_index=True)
        else:
            st.info("No live transactions in stream yet. Execute transfers from Customer Wallet or Attack Lab.")

    # --- TAB 2: AUTHORIZATION INSPECTOR ---
    with tab_inspect:
        st.markdown("### Authoritative Gateway Ingress & Decision Lineage")
        st.caption("Inspect exact pre-settlement authorization parameters, ML calibration, and decision lineage.")

        last_res = st.session_state.get("last_gateway_result")
        if last_res:
            st.markdown(f"""
            <div class="fp-card">
                <div class="fp-card-header">
                    <div class="fp-card-title">
                        {get_icon("shield", size=16, color="#38bdf8")}
                        Authorization Detail: <code>{last_res.transaction_id}</code>
                    </div>
                    <span class="status-pill {"pill-red" if last_res.decision == "BLOCK" else ("pill-amber" if last_res.decision == "HOLD" else "pill-green")}">{last_res.decision}</span>
                </div>
                <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; font-size: 0.85rem; color: #cbd5e1; margin-bottom: 14px;'>
                    <div><b>Sender:</b> <code>{last_res.sender_id}</code></div>
                    <div><b>Receiver:</b> <code>{last_res.receiver_id}</code></div>
                    <div><b>Amount:</b> ₹{last_res.amount:,.2f}</div>
                    <div><b>ML Probability:</b> {last_res.calibrated_probability*100:.2f}%</div>
                    <div><b>Hybrid Risk:</b> {last_res.risk_score:.1f} / 100.0</div>
                    <div><b>ATO Action:</b> <code>{last_res.ato_action}</code></div>
                    <div><b>Latency:</b> {last_res.latency_ms:.2f} ms</div>
                    <div><b>Money Transferred:</b> {"YES" if last_res.money_transferred else "NO (Protected)"}</div>
                    <div><b>Status:</b> <code>{last_res.status}</code></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            render_decision_lineage(last_res)
        else:
            st.info("No recent authorization in memory. Run an attack or wallet transaction to inspect live decision payload.")

    # --- TAB 3: SECURITY CASES ---
    with tab_cases:
        st.markdown("### Persistent Security Cases & Ground-Truth Labels")
        st.caption("Cases generated from customer fraud disputes or confirmed threats. Feeds the controlled ML feedback loop.")

        cases = sink.get_security_cases(limit=30) if hasattr(sink, "get_security_cases") else []
        if cases:
            df_cases = pd.DataFrame([{
                "Case ID": c.get("case_id"),
                "Transaction ID": c.get("transaction_id"),
                "Customer": c.get("customer_id"),
                "Reported Reason": c.get("customer_report"),
                "Confirmed Label": "FRAUD (1)" if c.get("confirmed_label") == 1 else "LEGIT (0)",
                "Attack Type": c.get("attack_type"),
                "Status": c.get("status"),
                "Created At": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(c.get("created_at", time.time()))))
            } for c in cases])
            st.dataframe(df_cases, width="stretch", hide_index=True)
        else:
            st.info("No security cases logged yet. In Customer Wallet, report any transaction as unauthorized to generate a security case.")
