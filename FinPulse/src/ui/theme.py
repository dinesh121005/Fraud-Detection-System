"""
FinPulse UI Design System & Theme.

Provides a unified, high-end fintech dark theme:
- Deep slate / obsidian background palette
- Semantic color tokens (Emerald/Green, Amber/Warning, Crimson/Red, Cyan/System, Indigo/Brand)
- Glassmorphic card styling, subtle borders, crisp typography
- Micro-animations, responsive layout utilities
"""

import streamlit as st

def apply_global_theme():
    """Injects the core FinPulse CSS design tokens and component styles into Streamlit."""
    st.markdown("""
    <style>
        /* Global Reset & Google Font Integration */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #f1f5f9;
        }

        code, pre, .mono-font {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Main Container Spacing */
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 1400px;
        }

        /* Metrics Styling */
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            transition: transform 0.15s ease, border-color 0.15s ease;
        }
        div[data-testid="stMetric"]:hover {
            border-color: rgba(56, 189, 248, 0.3);
            transform: translateY(-2px);
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            color: #94a3b8 !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.7rem !important;
            font-weight: 800 !important;
            color: #f8fafc !important;
            letter-spacing: -0.02em !important;
        }

        /* Status Pills */
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }
        .pill-green {
            background: rgba(16, 185, 129, 0.12);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .pill-amber {
            background: rgba(245, 158, 11, 0.12);
            color: #f59e0b;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }
        .pill-red {
            background: rgba(239, 68, 68, 0.12);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        .pill-blue {
            background: rgba(56, 189, 248, 0.12);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }
        .pill-purple {
            background: rgba(168, 85, 247, 0.12);
            color: #c084fc;
            border: 1px solid rgba(168, 85, 247, 0.3);
        }
        .pill-neutral {
            background: rgba(148, 163, 184, 0.12);
            color: #94a3b8;
            border: 1px solid rgba(148, 163, 184, 0.25);
        }

        /* Generic Glass Card */
        .fp-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 20px 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
            margin-bottom: 18px;
        }
        .fp-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 14px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }
        .fp-card-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: #f8fafc;
            display: flex;
            align-items: center;
            gap: 8px;
            letter-spacing: -0.01em;
        }

        /* Wallet Specific Card */
        .wallet-card-hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 18px;
            padding: 26px 30px;
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.45);
            margin-bottom: 22px;
            position: relative;
            overflow: hidden;
        }
        .wallet-card-hero::after {
            content: "";
            position: absolute;
            top: -40px;
            right: -40px;
            width: 140px;
            height: 140px;
            background: radial-gradient(circle, rgba(56, 189, 248, 0.15) 0%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
        }

        /* Attacker Threat Panel */
        .attacker-card {
            background: linear-gradient(135deg, rgba(35, 12, 18, 0.6) 0%, rgba(15, 23, 42, 0.9) 100%);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 14px;
            padding: 20px;
            box-shadow: 0 8px 30px rgba(239, 68, 68, 0.08);
            margin-bottom: 18px;
        }

        /* Timeline Nodes */
        .timeline-container {
            position: relative;
            padding-left: 24px;
            margin: 16px 0;
        }
        .timeline-container::before {
            content: "";
            position: absolute;
            left: 7px;
            top: 6px;
            bottom: 6px;
            width: 2px;
            background: rgba(255, 255, 255, 0.1);
        }
        .timeline-item {
            position: relative;
            margin-bottom: 14px;
        }
        .timeline-dot {
            position: absolute;
            left: -24px;
            top: 4px;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: #1e293b;
            border: 2px solid #38bdf8;
        }
        .timeline-dot.critical { border-color: #ef4444; background: rgba(239, 68, 68, 0.2); }
        .timeline-dot.warning { border-color: #f59e0b; background: rgba(245, 158, 11, 0.2); }
        .timeline-dot.success { border-color: #10b981; background: rgba(16, 185, 129, 0.2); }

        /* Decision Lineage Steps */
        .decision-step-box {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 6px;
            transition: border-color 0.15s ease;
        }
        .decision-step-box:hover {
            border-color: rgba(255, 255, 255, 0.18);
        }
        .decision-arrow {
            text-align: center;
            color: #475569;
            font-size: 1.1rem;
            line-height: 1;
            margin: 2px 0;
            font-weight: 700;
        }

        /* Streamlit Tab Customization */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(15, 23, 42, 0.5);
            padding: 6px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.06);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 0.85rem;
            font-weight: 600;
            color: #94a3b8;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(56, 189, 248, 0.12) !important;
            color: #38bdf8 !important;
            border-bottom: 2px solid #38bdf8 !important;
        }

        /* Buttons & Forms */
        .stButton button {
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.85rem;
            letter-spacing: 0.01em;
            transition: all 0.15s ease;
        }
        .stButton button[kind="primary"] {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
            border: 1px solid #38bdf8;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3);
        }
        .stButton button[kind="primary"]:hover {
            background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
            box-shadow: 0 6px 18px rgba(2, 132, 199, 0.45);
        }

        /* Dataframe & Tables */
        div[data-testid="stDataFrame"] {
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            overflow: hidden;
        }

        /* Sidebar Polish */
        section[data-testid="stSidebar"] {
            background: #090d16;
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }
    </style>
    """, unsafe_allow_html=True)
