"""
FinPulse 4-Layer Decision Lineage Component.

Visually represents how FinPulse processes transactions across 4 authoritative layers:
L1: Supervised ML (CatBoost + Platt Calibration)
L2: Hybrid Risk Fusion (Weighted signals + escalation)
L3: ATO Protection Policy (Account state & guardrails)
L4: Payment Gateway Enforcement (Final fund movement boundary)
"""

from typing import Any, Optional, Dict, List
import streamlit as st
from src.ui.icons import get_icon

def render_decision_lineage(result: Any, show_explanation: bool = True):
    """
    Renders the 4-layer decision pipeline lineage cards with visual arrows and status badges.
    """
    ml_dec = getattr(result, "ml_decision", "APPROVE")
    ml_prob = getattr(result, "calibrated_probability", 0.0)
    hyb_dec = getattr(result, "hybrid_decision", "APPROVE")
    risk_score = getattr(result, "risk_score", 0.0)
    risk_lvl = getattr(result, "risk_level", "LOW")
    ato_act = getattr(result, "ato_action", "ALLOW")
    ato_score = getattr(result, "ato_score", 0.0)
    gw_dec = getattr(result, "decision", "APPROVE")
    gw_status = getattr(result, "status", gw_dec)
    money_moved = getattr(result, "money_transferred", True)

    ml_col = "#10b981" if ml_dec == "APPROVE" else ("#f59e0b" if ml_dec == "REVIEW" else "#ef4444")
    hyb_col = "#10b981" if hyb_dec == "APPROVE" else ("#f59e0b" if hyb_dec == "REVIEW" else "#ef4444")
    ato_col = "#ef4444" if "SUSPEND" in ato_act else ("#f59e0b" if "HOLD" in ato_act else "#10b981")
    gw_col = "#10b981" if gw_dec == "APPROVE" else ("#f59e0b" if gw_dec == "HOLD" else "#ef4444")

    st.markdown(f"""
    <div style='display: flex; flex-direction: column; gap: 4px; margin: 12px 0;'>
        <!-- Layer 1 -->
        <div class="decision-step-box">
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span style='font-size: 0.76rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;'>
                    Layer 1: Supervised ML Model (CatBoost finpulse-v3)
                </span>
                <span class="status-pill" style='background: rgba(0,0,0,0.3); color: {ml_col}; border: 1px solid {ml_col}; font-weight: 700;'>
                    {ml_dec}
                </span>
            </div>
            <div style='font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;'>
                Calibrated P(Fraud): <b>{ml_prob * 100:.2f}%</b> (Platt Sigmoid Scaled)
            </div>
        </div>

        <div class="decision-arrow">↓</div>

        <!-- Layer 2 -->
        <div class="decision-step-box">
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span style='font-size: 0.76rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;'>
                    Layer 2: Hybrid Risk Fusion Engine
                </span>
                <span class="status-pill" style='background: rgba(0,0,0,0.3); color: {hyb_col}; border: 1px solid {hyb_col}; font-weight: 700;'>
                    {hyb_dec}
                </span>
            </div>
            <div style='font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;'>
                Composite Risk Score: <b>{risk_score:.1f} / 100.0</b> (Level: <code>{risk_lvl}</code>)
            </div>
        </div>

        <div class="decision-arrow">↓</div>

        <!-- Layer 3 -->
        <div class="decision-step-box">
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span style='font-size: 0.76rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;'>
                    Layer 3: ATO & Account Security Policy
                </span>
                <span class="status-pill" style='background: rgba(0,0,0,0.3); color: {ato_col}; border: 1px solid {ato_col}; font-weight: 700;'>
                    {ato_act}
                </span>
            </div>
            <div style='font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;'>
                ATO Score: <b>{ato_score:.2f}</b> • Action: <code>{ato_act}</code>
            </div>
        </div>

        <div class="decision-arrow">↓</div>

        <!-- Layer 4 -->
        <div class="decision-step-box" style='border: 1px solid {gw_col}; background: rgba(15, 23, 42, 0.9); box-shadow: 0 4px 16px rgba(0,0,0,0.4);'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span style='font-size: 0.78rem; font-weight: 800; color: {gw_col}; text-transform: uppercase; letter-spacing: 0.05em;'>
                    Layer 4: Authoritative Payment Gateway
                </span>
                <span class="status-pill" style='background: rgba(0,0,0,0.4); color: {gw_col}; border: 1px solid {gw_col}; font-weight: 800;'>
                    {gw_dec}
                </span>
            </div>
            <div style='font-size: 0.85rem; color: #f8fafc; margin-top: 4px; display: flex; justify-content: space-between; align-items: center;'>
                <span>Status: <b>{gw_status}</b></span>
                <span>Funds Movement: <b style='color: {"#ef4444" if not money_moved else "#10b981"};'>{"PREVENTED (₹0 Moved)" if not money_moved else "EXECUTED"}</b></span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if show_explanation and ml_dec == "APPROVE" and gw_dec == "BLOCK":
        st.markdown("""
        <div style='background: rgba(56, 189, 248, 0.08); border-left: 3px solid #38bdf8; border-radius: 6px; padding: 12px 16px; margin-top: 10px; font-size: 0.82rem; color: #cbd5e1; line-height: 1.5;'>
            <b style='color: #38bdf8;'>Why did Layer 1 (ML Model) say APPROVE while Layer 4 (Gateway) decided BLOCK?</b><br>
            The tabular ML model inspects transaction features in isolation. When an attacker crafts a transaction that looks numerically plausible, raw ML alone could be bypassed. FinPulse's <b>4-Layer Architecture</b> caught the behavioral deviation in Layer 2 and intercepted the Account Takeover in Layer 3, commanding the Payment Gateway to block fund movement before any money moved.
        </div>
        """, unsafe_allow_html=True)
