"""
FinPulse Reusable Global Transaction Card Component.

Provides a unified visual card for transactions across:
- Customer Wallet history
- Attack Lab results
- Gateway Operations ledger
- Analyst forensics
"""

import time
from typing import Dict, Any, Optional
from src.ui.icons import get_icon

def render_transaction_card(
    tx_id: str,
    timestamp: str,
    sender: str,
    recipient: str,
    amount: float,
    ml_prob: Optional[float] = None,
    hybrid_score: Optional[float] = None,
    ato_decision: Optional[str] = None,
    gateway_decision: str = "APPROVE",
    funds_moved: bool = True,
    status: Optional[str] = None,
    latency_ms: Optional[float] = None
) -> str:
    """
    Renders the canonical FinPulse transaction card HTML.
    """
    if gateway_decision == "BLOCK" or not funds_moved:
        border_col = "#ef4444"
        bg_col = "rgba(239, 68, 68, 0.08)"
        badge_cls = "pill-red"
        gw_label = "BLOCKED"
        funds_label = "₹0.00 (Protected)"
        funds_col = "#10b981"
    elif gateway_decision == "HOLD" or status in ["HELD", "HELD_PENDING_VERIFICATION"]:
        border_col = "#f59e0b"
        bg_col = "rgba(245, 158, 11, 0.08)"
        badge_cls = "pill-amber"
        gw_label = "HOLD (Verify)"
        funds_label = "₹0.00 (Held for Review)"
        funds_col = "#f59e0b"
    else:
        border_col = "#10b981"
        bg_col = "rgba(16, 185, 129, 0.08)"
        badge_cls = "pill-green"
        gw_label = "APPROVED"
        funds_label = f"₹{amount:,.2f}"
        funds_col = "#38bdf8"

    ml_str = f"{ml_prob * 100:.1f}%" if ml_prob is not None else "—"
    hyb_str = f"{hybrid_score:.1f}" if hybrid_score is not None else "—"
    ato_str = ato_decision if ato_decision else "ALLOW"
    lat_str = f" • {latency_ms:.1f}ms" if latency_ms else ""

    return f"""
    <div style='background: {bg_col}; border: 1px solid {border_col}; border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;'>
        <div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 8px; margin-bottom: 10px;'>
            <span style='font-size: 0.82rem; font-weight: 700; color: #cbd5e1;'><code>{tx_id}</code>{lat_str}</span>
            <span style='font-size: 0.75rem; color: #94a3b8;'>{timestamp}</span>
        </div>
        <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 12px;'>
            <div>
                <div style='font-size: 0.78rem; color: #94a3b8;'>Transfer Flow</div>
                <div style='font-size: 0.95rem; font-weight: 600; color: #f8fafc; margin-top: 2px;'>
                    <code>{sender}</code> ➔ <code>{recipient}</code>
                </div>
            </div>
            <div style='text-align: right;'>
                <div style='font-size: 0.78rem; color: #94a3b8;'>Transfer Amount</div>
                <div style='font-size: 1.5rem; font-weight: 800; color: #f8fafc;'>₹{amount:,.2f}</div>
            </div>
        </div>
        <div style='background: rgba(0,0,0,0.25); border-radius: 8px; padding: 10px 14px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; text-align: center;'>
            <div>
                <div style='font-size: 0.7rem; color: #94a3b8; text-transform: uppercase;'>Layer 1: ML</div>
                <div style='font-size: 0.9rem; font-weight: 700; color: #38bdf8; margin-top: 2px;'>{ml_str}</div>
            </div>
            <div>
                <div style='font-size: 0.7rem; color: #94a3b8; text-transform: uppercase;'>Layer 2: Hybrid</div>
                <div style='font-size: 0.9rem; font-weight: 700; color: #c084fc; margin-top: 2px;'>{hyb_str}</div>
            </div>
            <div>
                <div style='font-size: 0.7rem; color: #94a3b8; text-transform: uppercase;'>Layer 3: ATO</div>
                <div style='font-size: 0.85rem; font-weight: 700; color: {"#ef4444" if "SUSPEND" in ato_str or "BLOCK" in ato_str else "#10b981"}; margin-top: 2px;'>{ato_str}</div>
            </div>
            <div>
                <div style='font-size: 0.7rem; color: #94a3b8; text-transform: uppercase;'>Gateway Verdict</div>
                <div style='font-size: 0.88rem; font-weight: 800; color: {border_col}; margin-top: 2px;'>{gw_label}</div>
            </div>
        </div>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-top: 10px; font-size: 0.78rem; color: #94a3b8;'>
            <span>Funds Movement: <b style='color: {funds_col};'>{funds_label}</b></span>
            <span class="status-pill {badge_cls}">{gateway_decision}</span>
        </div>
    </div>
    """
