"""
FinPulse ML Lifecycle & Model Evolution Experience.

Visualizes continuous model governance and the controlled feedback loop:
- Visual End-to-End Retraining Lifecycle Architecture
- Known vs Novel Attack Taxonomy Matrix
- Delayed Ground-Truth Labeled Dataset in PostgreSQL
- Challenger & Shadow Scoring Engine (Non-interfering candidate inference)
- Production Model Promotion Gate (PR-AUC, ROC-AUC, Brier score, FPR@95 Recall)
"""

import time
import numpy as np
import pandas as pd
import streamlit as st

from src.ui.icons import get_icon, render_icon_badge
from src.challenger.shadow_scorer import ShadowModelRunner
from training.retrain import ModelPromotionGate

def render_ml_lifecycle(sink, shadow_runner: ShadowModelRunner, promotion_gate: ModelPromotionGate):
    """Renders the ML Lifecycle & Model Evolution view."""
    st.markdown("""
    <div style='margin-bottom: 16px;'>
        <h2 style='margin: 0; font-size: 1.6rem; font-weight: 800; color: #f8fafc; display: flex; align-items: center; gap: 10px;'>
            """ + get_icon("brain", size=24, color="#38bdf8") + """
            Machine Learning Lifecycle & Controlled Feedback Loop
        </h2>
        <div style='color: #94a3b8; font-size: 0.85rem; margin-top: 2px;'>
            Model Governance • Ground-Truth Delayed Labels • Non-Interfering Shadow Scoring • Promotion Gate
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # PRODUCTION MODEL METRICS OVERVIEW
    # -------------------------------------------------------------------------
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.metric("Production Model", "finpulse-v3", "CatBoost Classifier")
    with c_m2:
        st.metric("Calibration", "Platt Sigmoid", "True Posterior P(Fraud)")
    with c_m3:
        st.metric("Feature Vector", "32 Dimensions", "Sliding Redis + Profile")
    with c_m4:
        st.metric("Drift (PSI)", "0.0136 (Stable)", "Threshold < 0.10", delta_color="normal")

    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

    # Visual Retraining Lifecycle Diagram (Section 34)
    st.markdown(f"""
    <div class="fp-card">
        <div style='font-size: 0.78rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;'>
            Continuous Model Evolution Architecture
        </div>
        <div style='display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; font-size: 0.82rem; font-weight: 600; color: #cbd5e1; background: rgba(0,0,0,0.25); padding: 12px 16px; border-radius: 8px;'>
            <span class="status-pill pill-blue">1. PRODUCTION (finpulse-v3)</span>
            <span style='color: #64748b;'>➔</span>
            <span class="status-pill pill-purple">2. SHADOW SCORING</span>
            <span style='color: #64748b;'>➔</span>
            <span class="status-pill pill-amber">3. CONFIRMED FRAUD LABELS</span>
            <span style='color: #64748b;'>➔</span>
            <span class="status-pill pill-blue">4. RETRAINING DATASET</span>
            <span style='color: #64748b;'>➔</span>
            <span class="status-pill pill-purple">5. CANDIDATE (v4)</span>
            <span style='color: #64748b;'>➔</span>
            <span class="status-pill pill-green">6. PROMOTION GATE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_known, tab_labels, tab_shadow, tab_gate = st.tabs([
        "Known vs Novel Attacks",
        "Confirmed Labels (PostgreSQL)",
        "Shadow Scoring Engine",
        "Promotion Gate"
    ])

    # --- TAB 1: KNOWN VS NOVEL ATTACKS ---
    with tab_known:
        st.markdown("### Known vs Novel Attack Defense Strategy")
        st.caption("How FinPulse handles known signatures versus novel attack edge cases.")

        col_k1, col_k2 = st.columns(2)
        with col_k1:
            st.markdown(f"""
            <div style='background: rgba(16, 185, 129, 0.08); border: 2px solid #10b981; border-radius: 12px; padding: 18px;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <h4 style='color: #10b981; margin: 0; font-size: 1.15rem; font-weight: 800;'>KNOWN ATTACK SIGNATURE</h4>
                    {render_icon_badge("shield", "HIGH CONFIDENCE", "green", size=11)}
                </div>
                <p style='color: #cbd5e1; font-size: 0.85rem; margin-top: 8px;'>
                    <b>Pattern:</b> Recognized ATO signature (e.g. credential compromise + new hardware fingerprint + distant Mumbai location + password change).
                </p>
                <div style='background: rgba(0,0,0,0.3); padding: 10px 14px; border-radius: 8px; font-size: 0.8rem; color: #cbd5e1; line-height: 1.6;'>
                    <div><b>FinPulse Confidence:</b> HIGH (Score: 85 — 100)</div>
                    <div><b>ATO Action:</b> <code>SUSPEND_ACCOUNT</code></div>
                    <div><b>Gateway Verdict:</b> <span style='color: #ef4444; font-weight: 700;'>BLOCK</span></div>
                    <div><b>Funds Movement:</b> <b style='color: #10b981;'>₹0 Moved (Protected at ingress)</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_k2:
            st.markdown(f"""
            <div style='background: rgba(168, 85, 247, 0.08); border: 2px solid #c084fc; border-radius: 12px; padding: 18px;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <h4 style='color: #c084fc; margin: 0; font-size: 1.15rem; font-weight: 800;'>NOVEL / UNKNOWN ATTACK</h4>
                    {render_icon_badge("zap", "FEEDBACK CANDIDATE", "purple", size=11)}
                </div>
                <p style='color: #cbd5e1; font-size: 0.85rem; margin-top: 8px;'>
                    <b>Pattern:</b> Unrecognized attack vector or subtle edge-case (e.g. SIM swap, moderate probe, new merchant category).
                </p>
                <div style='background: rgba(0,0,0,0.3); padding: 10px 14px; border-radius: 8px; font-size: 0.8rem; color: #cbd5e1; line-height: 1.6;'>
                    <div><b>FinPulse Confidence:</b> LOW / MODERATE</div>
                    <div><b>Gateway Verdict:</b> <span style='color: #f59e0b; font-weight: 700;'>HOLD / APPROVE</span></div>
                    <div><b>Customer Action:</b> Reports unauthorized transfer via wallet.</div>
                    <div><b>Lifecycle Action:</b> <b style='color: #c084fc;'>Confirmed label attached ➔ Retraining dataset</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # --- TAB 2: CONFIRMED LABELS DATASET ---
    with tab_labels:
        st.markdown("### Confirmed Fraud Labels in PostgreSQL Sink")
        st.caption("Transactions enriched with ground-truth fraud labels (1) via cardholder reporting or analyst confirmation.")

        labeled_data = sink.get_labeled_dataset() if hasattr(sink, "get_labeled_dataset") else []
        st.write(f"**Total Confirmed Ground-Truth Records:** `{len(labeled_data)}`")

        if labeled_data:
            df_lbl = pd.DataFrame([{
                "Tx ID": d.get("transaction_id"),
                "Customer": d.get("customer_id"),
                "Model Decision": d.get("ml_decision") or d.get("decision") or "APPROVE",
                "Ground-Truth Label": "FRAUD (1)" if d.get("fraud_label") == 1 else "LEGIT (0)",
                "Label Attached": time.strftime(
                    '%Y-%m-%d %H:%M:%S',
                    time.localtime(float(d.get("label_timestamp") or d.get("updated_at") or d.get("created_at") or time.time()))
                )
            } for d in labeled_data])
            st.dataframe(df_lbl, width="stretch", hide_index=True)
        else:
            st.info("No delayed labels in persistence sink yet. Disputing any transaction in My Wallet will immediately attach ground-truth fraud label (1).")

    # --- TAB 3: CHALLENGER & SHADOW SCORING ---
    with tab_shadow:
        st.markdown("### Non-Interfering Challenger & Shadow Scoring Engine")
        st.caption("Executes candidate models concurrently in non-blocking shadow mode alongside the frozen production model.")

        col_sh1, col_sh2 = st.columns(2)
        with col_sh1:
            st.metric("Active Production Model", "finpulse-v3", "Gateway Decision Authority")
        with col_sh2:
            st.metric("Shadow Candidate Model", "finpulse-v4-candidate", "Non-Interfering Shadow Scorer")

        dummy_features = np.zeros(32)
        comp = shadow_runner.evaluate_shadow(
            transaction_id="tx_shadow_demo_01",
            feature_vector_32=dummy_features,
            production_result={"calibrated_probability": 0.05, "risk_score": 12.0}
        )
        if comp:
            st.markdown(f"""
            <div class="fp-card" style='margin-top: 14px;'>
                <div style='font-size: 0.85rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px;'>Shadow Evaluation Sample</div>
                <div style='font-size: 0.82rem; color: #cbd5e1; line-height: 1.6;'>
                    • <b>Shadow Ingestion:</b> <code>{comp.transaction_id}</code><br>
                    • <b>Production Risk Score:</b> <code>{comp.production_risk_score:.1f}</code> vs <b>Candidate:</b> <code>{comp.challenger_risk_score:.1f}</code><br>
                    • <b>Shadow Latency:</b> <code>{comp.challenger_latency_ms:.2f} ms</code> (Zero impact on production SLA)
                </div>
            </div>
            """, unsafe_allow_html=True)

    # --- TAB 4: PROMOTION GATE ---
    with tab_gate:
        st.markdown("### Production Model Promotion Gate")
        st.caption("Enforces strict quality, calibration, and regression checks before promoting candidate models.")

        prod_m = {"pr_auc": 0.885, "roc_auc": 0.942, "brier_score": 0.048, "fpr_at_95_recall": 0.012}
        cand_m = {"pr_auc": 0.892, "roc_auc": 0.948, "brier_score": 0.044, "fpr_at_95_recall": 0.011}

        report = promotion_gate.evaluate_candidate(
            prod_metrics=prod_m,
            cand_metrics=cand_m,
            candidate_version="finpulse-v4-candidate",
            production_version="finpulse-v3"
        )

        st.success(f"Promotion Gate Status: **{report.overall_status}** ({report.summary_reason})")
        df_checks = pd.DataFrame([{
            "Validation Check": c.name,
            "Production Metric": c.production_value,
            "Candidate Metric": c.candidate_value,
            "Threshold Criteria": c.threshold_condition,
            "Gate Status": "PASSED" if c.passed else "FAILED"
        } for c in report.checks])
        st.dataframe(df_checks, width="stretch", hide_index=True)
