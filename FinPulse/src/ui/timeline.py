"""
FinPulse Real-Time Event & Attack Timeline Component.

Renders chronological telemetry nodes from real system events, security alarms,
inference execution, and gateway authorization verdicts.
"""

import streamlit as st
from typing import List, Dict, Any
from src.ui.icons import get_icon

def render_timeline(timeline_items: List[Dict[str, Any]]):
    """
    Renders a clean vertical timeline with severity-coded node dots and micro-badges.
    """
    if not timeline_items:
        st.markdown("""
        <div style='background: rgba(15, 23, 42, 0.4); border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 24px; text-align: center; color: #64748b; font-size: 0.85rem;'>
            No real-time events recorded for this session yet.<br>
            Execute an attack scenario or payment to generate real-time execution telemetry.
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown("<div class='timeline-container'>", unsafe_allow_html=True)
    for item in timeline_items:
        sev = item.get("severity", "INFO")
        dot_class = "critical" if sev == "CRITICAL" else ("warning" if sev in ["HIGH", "MEDIUM"] else "success")
        time_str = item.get("time", "")
        ev_title = item.get("event", "")
        detail = item.get("detail", "")
        status = item.get("status", "")

        status_badge = ""
        if status:
            b_cls = "pill-red" if status in ["BLOCK", "CRITICAL", "DETECTED"] else ("pill-amber" if status in ["HOLD", "REVIEW", "AUTHORIZING"] else "pill-blue")
            status_badge = f"<span class='status-pill {b_cls}' style='font-size: 0.68rem; padding: 2px 8px;'>{status}</span>"

        st.markdown(f"""
        <div class='timeline-item'>
            <div class='timeline-dot {dot_class}'></div>
            <div style='display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px;'>
                <div style='font-weight: 700; color: #f8fafc; font-size: 0.9rem; display: flex; align-items: center; gap: 8px;'>
                    <span>{ev_title}</span>
                    {status_badge}
                </div>
                <span style='font-size: 0.72rem; color: #94a3b8; font-family: monospace;'>{time_str}</span>
            </div>
            <div style='font-size: 0.82rem; color: #94a3b8; margin-top: 3px; line-height: 1.4;'>
                {detail}
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
