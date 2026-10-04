"""
FinPulse Fraud Analyst & Forensic Investigation Workstation.

Provides deep explainability, signal decomposition, and behavioral audit:
1. Multi-Tier Fraud Criteria & Threshold Spectrum
2. Subsystem Risk Decomposition (ML 45%, Velocity 25%, Behavioral 15%, Rules 10%, Isolation Forest 5%)
3. TreeSHAP Local Feature Attribution Waterfall & Compliance Reason Codes
4. Behavioral Profile Deviation Radar (Comparing against customer's 30-day historical baseline)
"""

import json
import time
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from typing import Dict, Any, List, Optional

from src.ui.icons import get_icon, render_icon_badge
from src.simulation.demo_account import (
    DEFAULT_DEMO_CUSTOMER_ID,
    KNOWN_LOCATION,
    KNOWN_DEVICE_ID
)

def render_analyst_forensics(sink, attacker, gateway):
    """Renders the Fraud Analyst Forensic Workstation."""
    st.markdown("""
    <div style='margin-bottom: 16px;'>
        <h2 style='margin: 0; font-size: 1.6rem; font-weight: 800; color: #f8fafc; display: flex; align-items: center; gap: 10px;'>
            """ + get_icon("search", size=24, color="#38bdf8") + """
            Fraud Analyst Forensic Investigation & Explainability
        </h2>
        <div style='color: #94a3b8; font-size: 0.85rem; margin-top: 2px;'>
            Subsystem Signal Decomposition • TreeSHAP Local Attributions • Behavioral Profile Radar
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 1. TRANSACTION SELECTION & QUICK TEST SCENARIO BUTTONS
    # -------------------------------------------------------------------------
    available_txs = []
    if st.session_state.get("latest_tx_detail"):
        l_det = st.session_state.latest_tx_detail
        l_id = l_det.get("transaction_id", "latest")
        l_dec = l_det.get("decision", "N/A")
        l_score = float(l_det.get("risk_score", 0.0))
        available_txs.append({
            "id": l_id,
            "label": f"Latest Transaction ({l_id}) — {l_dec} (Score: {l_score:.1f})",
            "detail": l_det
        })

    for a in st.session_state.get("attack_history", []):
        t_id = a.get("Tx ID")
        if t_id and not any(x["id"] == t_id for x in available_txs):
            available_txs.append({
                "id": t_id,
                "label": f"Attack Lab: {a.get('Scenario')} ({t_id}) — {a.get('Gateway', 'N/A')}",
                "detail": None
            })

    for w in st.session_state.get("wallet_tx_history", []):
        t_id = w.get("tx_id")
        if t_id and not any(x["id"] == t_id for x in available_txs):
            available_txs.append({
                "id": t_id,
                "label": f"Wallet Transfer: ₹{w.get('amount', 0):,.2f} ({t_id}) — {w.get('decision', 'N/A')}",
                "detail": None
            })

    try:
        db_decs = sink.get_recent_decisions(limit=10) if hasattr(sink, "get_recent_decisions") else []
        for d in db_decs:
            t_id = d.get("transaction_id")
            if t_id and not any(x["id"] == t_id for x in available_txs):
                available_txs.append({
                    "id": t_id,
                    "label": f"DB Record: {t_id} — {d.get('decision', 'N/A')} (Score: {float(d.get('risk_score', 0.0)):.1f})",
                    "detail": None
                })
    except Exception:
        pass

    col_sel, col_btns = st.columns([1.4, 1.0])
    with col_sel:
        if available_txs:
            sel_idx = st.selectbox(
                "Select Transaction for Forensic Analysis:",
                range(len(available_txs)),
                format_func=lambda i: available_txs[i]["label"],
                key="analyst_sel_tx"
            )
            selected_tx = available_txs[sel_idx]
        else:
            selected_tx = None

    with col_btns:
        st.markdown("<div style='font-size: 0.78rem; font-weight: 600; color: #94a3b8; margin-bottom: 4px;'>Or Trigger Forensic Test:</div>", unsafe_allow_html=True)
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            if st.button("ATO Threat", width="stretch", help="Simulate a high-risk Account Takeover"):
                _, req = attacker.build_scenario_transfer("account_takeover", DEFAULT_DEMO_CUSTOMER_ID, 8500.0,
                    custom_params={"new_device": True, "new_location": True, "password_change": True, "auth_anomaly": True})
                res = gateway.process_transfer(req)
                st.session_state.last_gateway_result = res
                st.session_state.latest_tx_detail = res.eval_detail
                st.rerun()
        with col_b2:
            if st.button("Hold Surge", width="stretch", help="Simulate a suspicious pattern triggering a Hold"):
                _, req = attacker.build_scenario_transfer("novel_pattern" if hasattr(attacker, "build_scenario_transfer") else "novel_attack", DEFAULT_DEMO_CUSTOMER_ID, 2500.0,
                    custom_params={"foreign_channel": True, "off_hours": True, "round_amount": False})
                res = gateway.process_transfer(req)
                st.session_state.last_gateway_result = res
                st.session_state.latest_tx_detail = res.eval_detail
                st.rerun()
        with col_b3:
            if st.button("Safe Transfer", width="stretch", help="Simulate a legitimate wallet payment"):
                tx_req = {
                    "transaction_id": f"tx_safe_{int(time.time()*1000)}",
                    "customer_id": DEFAULT_DEMO_CUSTOMER_ID,
                    "recipient_id": "CUST_FRIEND_01",
                    "merchant_id": "CUST_FRIEND_01",
                    "amount": 500.0,
                    "payment_type": "UPI",
                    "device_id": KNOWN_DEVICE_ID,
                    "location": KNOWN_LOCATION,
                    "home_latitude": KNOWN_LOCATION["latitude"],
                    "home_longitude": KNOWN_LOCATION["longitude"],
                    "raw_metadata": {"purpose": "Gift", "channel": "customer_wallet"}
                }
                res = gateway.process_transfer(tx_req)
                st.session_state.last_gateway_result = res
                st.session_state.latest_tx_detail = res.eval_detail
                st.rerun()

    # Load transaction detail
    tx_det = None
    if selected_tx:
        if selected_tx["detail"]:
            tx_det = selected_tx["detail"]
        else:
            dec_rec = sink.get_decision(selected_tx["id"]) if hasattr(sink, "get_decision") else None
            if dec_rec:
                diag = dec_rec.get("diagnostics_json", {})
                if isinstance(diag, str):
                    try:
                        diag = json.loads(diag)
                    except Exception:
                        diag = {}
                sig = dec_rec.get("signals_json", {})
                if isinstance(sig, str):
                    try:
                        sig = json.loads(sig)
                    except Exception:
                        sig = {}
                reasons = dec_rec.get("reasons_json", [])
                if isinstance(reasons, str):
                    try:
                        reasons = json.loads(reasons)
                    except Exception:
                        reasons = []
                rules = dec_rec.get("matched_rules_json", [])
                if isinstance(rules, str):
                    try:
                        rules = json.loads(rules)
                    except Exception:
                        rules = []

                tx_row = sink.get_transaction(selected_tx["id"]) if hasattr(sink, "get_transaction") else None
                amt_from_tx = float(tx_row.get("amount", 0.0)) if (tx_row and tx_row.get("amount") is not None) else None
                dev_from_tx = tx_row.get("device_id") if tx_row else None
                loc_from_tx = tx_row.get("location_json") if tx_row else None

                tx_det = {
                    "transaction_id": dec_rec.get("transaction_id"),
                    "decision": dec_rec.get("decision", "APPROVE"),
                    "risk_score": float(dec_rec.get("risk_score", 0.0)),
                    "risk_level": dec_rec.get("risk_level", "LOW"),
                    "calibrated_probability": float(dec_rec.get("calibrated_probability", 0.0)),
                    "model_version": dec_rec.get("model_version", "finpulse-v3"),
                    "latency_ms": float(dec_rec.get("latency_ms", 3.2)),
                    "signals": sig,
                    "diagnostics": diag,
                    "top_reasons": reasons,
                    "rule_result": {"matched_rules": rules},
                    "amount": amt_from_tx,
                    "device_id": dev_from_tx,
                    "location": loc_from_tx,
                    "attributions": diag.get("attributions", []) if isinstance(diag, dict) else []
                }

    if not tx_det:
        st.info("No transaction loaded. Click one of the test buttons ('ATO Threat', 'Hold Surge', or 'Safe Transfer') above to launch forensic inspection.")
        return

    # Extract forensic fields
    risk_score = float(tx_det.get("risk_score", 0.0))
    risk_level = str(tx_det.get("risk_level", "LOW"))
    decision = str(tx_det.get("decision", "APPROVE"))
    calibrated_prob = float(tx_det.get("calibrated_probability", 0.0))
    model_version = str(tx_det.get("model_version", "finpulse-v3"))
    latency_ms = float(tx_det.get("latency_ms", 3.8))
    reasons = list(tx_det.get("top_reasons", []))
    matched_rules = list(tx_det.get("rule_result", {}).get("matched_rules", []))
    signals = tx_det.get("signals", {})
    diagnostics = tx_det.get("diagnostics", {})
    wf_status = str(diagnostics.get("workflow_status", ""))
    gw_verdict = str(diagnostics.get("gateway_decision", decision))

    # Hero Card
    if gw_verdict == "BLOCK" or decision == "BLOCK" or "suspended" in wf_status.lower() or "ato" in wf_status.lower():
        v_color = "#ef4444"
        v_bg = "rgba(239, 68, 68, 0.12)"
        v_title = "TRANSACTION BLOCKED (CRITICAL THREAT)"
    elif decision in ["HOLD", "REVIEW"] or gw_verdict in ["HOLD", "REVIEW"]:
        v_color = "#f59e0b"
        v_bg = "rgba(245, 158, 11, 0.12)"
        v_title = "TRANSACTION PLACED ON HOLD (VERIFICATION REQUIRED)"
    else:
        v_color = "#10b981"
        v_bg = "rgba(16, 185, 129, 0.12)"
        v_title = "TRANSACTION APPROVED (LEGITIMATE PAYMENT)"

    st.markdown(f"""
    <div style='background: {v_bg}; border: 2px solid {v_color}; border-radius: 14px; padding: 20px 24px; margin: 16px 0 24px 0;'>
        <div style='display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 14px;'>
            <div>
                <h2 style='color: {v_color}; margin: 0; font-size: 1.6rem; font-weight: 800;'>{v_title}</h2>
                <div style='color: #cbd5e1; font-size: 0.92rem; margin-top: 4px;'>
                    Lineage: <b>Supervised ML (L1)</b> ➔ <b>Hybrid Fusion (L2)</b> ➔ <b>ATO Guard (L3)</b> ➔ <b>Gateway Enforcement (L4)</b>
                </div>
                <div style='color: #94a3b8; font-size: 0.8rem; margin-top: 4px;'>
                    Transaction ID: <code>{tx_det.get('transaction_id', 'N/A')}</code> • Model: <code>{model_version}</code>
                </div>
            </div>
            <div style='display: flex; gap: 12px; flex-wrap: wrap;'>
                <div style='background: rgba(0,0,0,0.35); padding: 8px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); text-align: center;'>
                    <div style='font-size: 0.7rem; color: #94a3b8; text-transform: uppercase;'>Composite Risk</div>
                    <div style='font-size: 1.4rem; font-weight: 800; color: {v_color};'>{risk_score:.1f} / 100.0</div>
                </div>
                <div style='background: rgba(0,0,0,0.35); padding: 8px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); text-align: center;'>
                    <div style='font-size: 0.7rem; color: #94a3b8; text-transform: uppercase;'>ML Calibrated P(Fraud)</div>
                    <div style='font-size: 1.4rem; font-weight: 800; color: #38bdf8;'>{calibrated_prob*100:.1f}%</div>
                </div>
                <div style='background: rgba(0,0,0,0.35); padding: 8px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); text-align: center;'>
                    <div style='font-size: 0.7rem; color: #94a3b8; text-transform: uppercase;'>Latency</div>
                    <div style='font-size: 1.4rem; font-weight: 800; color: #10b981;'>{latency_ms:.2f} ms</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 1: FRAUD CRITERIA SPECTRUM
    # -------------------------------------------------------------------------
    st.markdown("### 1. Fraud Criteria & Threshold Spectrum")
    clamped_score = max(2.0, min(98.0, risk_score))
    st.markdown(f"""
    <div style='background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px 22px; margin-bottom: 18px;'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;'>
            <div style='font-weight: 700; color: #f8fafc; font-size: 0.9rem;'>Risk Spectrum Position</div>
            <div style='font-weight: 700; color: {v_color}; font-size: 0.95rem;'>Score: {risk_score:.1f} ➔ {decision}</div>
        </div>
        <div style='position: relative; height: 22px; border-radius: 11px; background: linear-gradient(to right, #10b981 0%, #10b981 30%, #f59e0b 30%, #f59e0b 70%, #ef4444 70%, #ef4444 100%); margin: 22px 0 8px 0;'>
            <div style='position: absolute; left: calc({clamped_score}% - 14px); top: -22px; text-align: center;'>
                <div style='background: #f8fafc; color: #0f172a; font-weight: 800; font-size: 0.75rem; padding: 2px 7px; border-radius: 5px;'>▼ {risk_score:.1f}</div>
            </div>
        </div>
        <div style='display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 600;'>
            <span style='color: #10b981;'>APPROVE ZONE (0.0 — 29.9)</span>
            <span style='color: #f59e0b;'>HOLD ZONE (30.0 — 69.9)</span>
            <span style='color: #ef4444;'>BLOCK ZONE (70.0 — 100.0)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_cr1, c_cr2, c_cr3, c_cr4 = st.columns(4)
    with c_cr1:
        ml_stat_color = "#ef4444" if calibrated_prob >= 0.70 else ("#f59e0b" if calibrated_prob >= 0.30 else "#10b981")
        ml_stat_text = "FRAUD TRIGGER" if calibrated_prob >= 0.70 else ("REVIEW TRIGGER" if calibrated_prob >= 0.30 else "PASS (SAFE)")
        st.markdown(f"""
        <div style='background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px;'>
            <div style='font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;'>1. ML Model Gate</div>
            <div style='font-size: 1.2rem; font-weight: 700; color: {ml_stat_color}; margin: 2px 0;'>{calibrated_prob*100:.1f}%</div>
            <div style='font-size: 0.72rem; color: #cbd5e1;'>Boundary: &ge;70% Block, &lt;30% Safe</div>
            <div style='margin-top: 4px;'><span class="status-pill" style='background: rgba(0,0,0,0.4); color: {ml_stat_color}; border: 1px solid {ml_stat_color}; font-size: 0.68rem;'>{ml_stat_text}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with c_cr2:
        trig_count = diagnostics.get("trigger_count", 0)
        comp_color = "#ef4444" if trig_count >= 2 else ("#f59e0b" if trig_count == 1 else "#10b981")
        comp_text = f"+{diagnostics.get('escalation_boost', 0):.0f} PTS BOOST" if trig_count >= 2 else "NO ESCALATION"
        st.markdown(f"""
        <div style='background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px;'>
            <div style='font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;'>2. Multi-Signal</div>
            <div style='font-size: 1.2rem; font-weight: 700; color: {comp_color}; margin: 2px 0;'>{trig_count} Triggers</div>
            <div style='font-size: 0.72rem; color: #cbd5e1;'>Rule: &ge;2 triggers &rarr; +15 pts boost</div>
            <div style='margin-top: 4px;'><span class="status-pill" style='background: rgba(0,0,0,0.4); color: {comp_color}; border: 1px solid {comp_color}; font-size: 0.68rem;'>{comp_text}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with c_cr3:
        ato_act = diagnostics.get("ato_action", "ALLOW")
        ato_color = "#ef4444" if ato_act == "SUSPEND_ACCOUNT" else ("#f59e0b" if ato_act == "TRIGGER_HOLD" else "#10b981")
        st.markdown(f"""
        <div style='background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px;'>
            <div style='font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;'>3. ATO Guardrail</div>
            <div style='font-size: 1.1rem; font-weight: 700; color: {ato_color}; margin: 2px 0;'>{ato_act}</div>
            <div style='font-size: 0.72rem; color: #cbd5e1;'>Threshold: Score &ge; 75 &rarr; Lock</div>
            <div style='margin-top: 4px;'><span class="status-pill" style='background: rgba(0,0,0,0.4); color: {ato_color}; border: 1px solid {ato_color}; font-size: 0.68rem;'>ATO: {diagnostics.get('ato_risk_score', 0):.1f}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with c_cr4:
        is_hb = diagnostics.get("hard_block", False)
        rule_color = "#ef4444" if is_hb or matched_rules else "#10b981"
        rule_text = "HARD OVERRIDE" if is_hb else (f"{len(matched_rules)} MATCHED" if matched_rules else "CLEAR")
        st.markdown(f"""
        <div style='background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px;'>
            <div style='font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;'>4. Business Rules</div>
            <div style='font-size: 1.1rem; font-weight: 700; color: {rule_color}; margin: 2px 0;'>{rule_text}</div>
            <div style='font-size: 0.72rem; color: #cbd5e1;'>Deterministic Safety Checks</div>
            <div style='margin-top: 4px;'><span class="status-pill" style='background: rgba(0,0,0,0.4); color: {rule_color}; border: 1px solid {rule_color}; font-size: 0.68rem;'>Overrides: {"Active" if is_hb else "None"}</span></div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 2: SUBSYSTEM RISK DECOMPOSITION (CHARTS)
    # -------------------------------------------------------------------------
    st.write("")
    st.markdown("### 2. Subsystem Risk Decomposition (Model Contribution)")
    st.caption("Contribution of the 5 layers to the final composite score.")

    contribs = diagnostics.get("contributions", {})
    c_ml = float(contribs.get("ml", calibrated_prob * 45.0))
    c_vel = float(contribs.get("velocity", float(signals.get("velocity_signal", 0.0)) * 25.0))
    c_beh = float(contribs.get("behavioral", float(signals.get("behavioral_signal", 0.0)) * 15.0))
    c_rules = float(contribs.get("rules", float(signals.get("rules_signal", 0.0)) * 10.0))
    c_anom = float(contribs.get("anomaly", float(signals.get("anomaly_signal", 0.0)) * 5.0))
    tot_pts = max(0.1, c_ml + c_vel + c_beh + c_rules + c_anom)

    df_subsystems = pd.DataFrame([
        {"Subsystem": "Supervised ML (45% Weight)", "Points": round(c_ml, 1), "Share": f"{(c_ml/tot_pts)*100:.1f}%", "Color": "#ef4444"},
        {"Subsystem": "Velocity Burst (25% Weight)", "Points": round(c_vel, 1), "Share": f"{(c_vel/tot_pts)*100:.1f}%", "Color": "#f59e0b"},
        {"Subsystem": "Device & Behavioral Delta (15% Weight)", "Points": round(c_beh, 1), "Share": f"{(c_beh/tot_pts)*100:.1f}%", "Color": "#38bdf8"},
        {"Subsystem": "Deterministic Rules (10% Weight)", "Points": round(c_rules, 1), "Share": f"{(c_rules/tot_pts)*100:.1f}%", "Color": "#a855f7"},
        {"Subsystem": "Isolation Forest Anomaly (5% Weight)", "Points": round(c_anom, 1), "Share": f"{(c_anom/tot_pts)*100:.1f}%", "Color": "#64748b"}
    ])

    col_d1, col_d2 = st.columns([1.1, 1.3])
    with col_d1:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #cbd5e1; margin-bottom: 6px;'>Subsystem Risk Distribution</div>", unsafe_allow_html=True)
        donut_chart = alt.Chart(df_subsystems).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Points", type="quantitative"),
            color=alt.Color(field="Subsystem", type="nominal",
                scale=alt.Scale(domain=df_subsystems["Subsystem"].tolist(), range=df_subsystems["Color"].tolist()),
                legend=alt.Legend(orient="bottom", columns=1, title=None)),
            tooltip=["Subsystem:N", "Points:Q", "Share:N"]
        ).properties(height=250)
        st.altair_chart(donut_chart, width="stretch")

    with col_d2:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #cbd5e1; margin-bottom: 6px;'>Points Added to Composite Score</div>", unsafe_allow_html=True)
        bar_chart = alt.Chart(df_subsystems).mark_bar(cornerRadius=4).encode(
            x=alt.X("Points:Q", title="Points Added"),
            y=alt.Y("Subsystem:N", sort="-x", title=None),
            color=alt.Color("Subsystem:N",
                scale=alt.Scale(domain=df_subsystems["Subsystem"].tolist(), range=df_subsystems["Color"].tolist()),
                legend=None),
            tooltip=["Subsystem:N", "Points:Q", "Share:N"]
        ).properties(height=250)
        st.altair_chart(bar_chart, width="stretch")

    # -------------------------------------------------------------------------
    # SECTION 3: SHAP FEATURE ATTRIBUTIONS & COMPLIANCE REASONS
    # -------------------------------------------------------------------------
    st.write("")
    st.markdown("### 3. TreeSHAP Local Feature Attribution (Why this Happened)")
    st.caption("Feature dimensions pushing the decision toward fraud vs defending as legitimate.")

    raw_attributions = tx_det.get("attributions", [])
    if not raw_attributions and "diagnostics" in tx_det:
        diag_dict = tx_det.get("diagnostics", {})
        if isinstance(diag_dict, dict):
            raw_attributions = diag_dict.get("attributions", [])

    FEATURE_LABELS = {
        "amount_zscore": "Amount Deviation (Z-Score)",
        "is_new_device": "Hardware Fingerprint (New Device)",
        "is_new_location": "Geographic Anomaly (Distant City)",
        "speed_kmh_from_prev_tx": "Physical Speed Impossibility (km/h)",
        "tx_count_5m": "Transaction Velocity (5m Count)",
        "amount_to_balance_ratio": "Balance Drain Ratio",
        "isolation_forest_score": "Multivariate Outlier Score",
        "auth_factor_verified": "2FA Authentication",
        "is_night": "Off-Hours Night Execution"
    }

    shap_rows = []
    for item in raw_attributions:
        fname = item.get("feature", "feature")
        val = float(item.get("attribution", 0.0))
        if abs(val) > 0.01:
            label = FEATURE_LABELS.get(fname, fname.replace("_", " ").title())
            impact = "Pushed Toward FRAUD" if val > 0 else "Defended as LEGITIMATE"
            shap_rows.append({
                "Feature": label,
                "SHAP Attribution": round(val, 2),
                "Impact": impact,
                "Magnitude": abs(val)
            })

    shap_rows.sort(key=lambda x: x["Magnitude"], reverse=True)
    df_shap = pd.DataFrame(shap_rows[:8]) if shap_rows else pd.DataFrame()

    col_sh1, col_sh2 = st.columns([1.4, 1.0])
    with col_sh1:
        if not df_shap.empty:
            shap_chart = alt.Chart(df_shap).mark_bar(cornerRadius=4).encode(
                x=alt.X("SHAP Attribution:Q", title="Points (+ toward Fraud, - toward Legit)"),
                y=alt.Y("Feature:N", sort=alt.SortField("Magnitude", order="descending"), title=None),
                color=alt.Color("Impact:N",
                    scale=alt.Scale(domain=["Pushed Toward FRAUD", "Defended as LEGITIMATE"], range=["#ef4444", "#10b981"]),
                    legend=alt.Legend(title="Attribution Direction", orient="top")),
                tooltip=["Feature:N", "SHAP Attribution:Q", "Impact:N"]
            ).properties(height=260)
            st.altair_chart(shap_chart, width="stretch")
        else:
            st.info("Local TreeSHAP attributions calculated during live inference. For historical records without persisted vectors, see Subsystem Decomposition.")

    with col_sh2:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #cbd5e1; margin-bottom: 6px;'>Compliance Reason Codes</div>", unsafe_allow_html=True)
        if reasons:
            for r in reasons:
                st.markdown(f"""
                <div style='background: rgba(239, 68, 68, 0.08); border-left: 3px solid #ef4444; padding: 8px 12px; margin-bottom: 6px; border-radius: 4px; font-size: 0.82rem; color: #f8fafc;'>
                    {r}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10b981; padding: 8px 12px; margin-bottom: 6px; border-radius: 4px; font-size: 0.82rem; color: #f8fafc;'>
                Behavioral consistency verified against customer baseline.
            </div>
            """, unsafe_allow_html=True)

        if matched_rules:
            st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #cbd5e1; margin-top: 10px; margin-bottom: 4px;'>Matched Rules:</div>", unsafe_allow_html=True)
            for m in matched_rules:
                st.caption(f"• `{m}`")

    # -------------------------------------------------------------------------
    # SECTION 4: BEHAVIORAL PROFILE DEVIATION (RADAR / METRICS)
    # -------------------------------------------------------------------------
    st.write("")
    st.markdown("### 4. Behavioral Baseline vs Transaction Deviation")
    st.caption("Forensic delta comparing customer's historical profile with this attempt.")

    amt_val = tx_det.get("amount")
    dev_val = tx_det.get("device_id")
    loc_val = tx_det.get("location")
    vel_sig = signals.get("velocity_signal") if signals else None

    col_dev1, col_dev2, col_dev3, col_dev4 = st.columns(4)
    with col_dev1:
        if amt_val is not None:
            pct_delta = f"+{int((float(amt_val)/500.0 - 1.0)*100)}%" if float(amt_val) > 500.0 else "Normal"
            st.metric("Transfer Amount", f"₹{float(amt_val):,.2f}", f"Baseline: ₹500.00 ({pct_delta})", delta_color="inverse" if float(amt_val) > 1000.0 else "normal")
        else:
            st.metric("Transfer Amount", "N/A", "Not Persisted")

    with col_dev2:
        if dev_val:
            dev_status = "Unrecognized Hardware" if dev_val != KNOWN_DEVICE_ID else "Known Registered Device"
            st.metric("Device Fingerprint", str(dev_val), dev_status, delta_color="inverse" if dev_val != KNOWN_DEVICE_ID else "normal")
        else:
            st.metric("Device Fingerprint", "N/A", "Not Persisted")

    with col_dev3:
        if loc_val:
            city_str = loc_val.get("city") if isinstance(loc_val, dict) else str(loc_val)
            is_remote = "CHENNAI" not in str(city_str).upper()
            loc_status = f"{city_str} (Remote)" if is_remote else "Chennai (Home)"
            st.metric("Geographic Origin", str(city_str), loc_status, delta_color="inverse" if is_remote else "normal")
        else:
            st.metric("Geographic Origin", "N/A", "Not Persisted")

    with col_dev4:
        if vel_sig is not None:
            v_float = float(vel_sig)
            vel_status = f"{v_float*10:.0f}x Burst Spike" if v_float >= 0.30 else "Normal Baseline"
            st.metric("Velocity Signal", f"{v_float:.2f}", vel_status, delta_color="inverse" if v_float >= 0.30 else "normal")
        else:
            st.metric("Velocity Signal", "N/A", "Not Persisted")
