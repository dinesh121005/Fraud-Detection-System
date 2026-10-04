"""
FinPulse Attack Simulation Lab & Adversarial Engine (Section 13, 14, 16).

Provides a comprehensive adversarial testing environment:
- Persistent & Adaptable Attacker State (ATK-7F31, ATK-92B4, ATK-C812, ATK-44D9, ATK-A71F, ATK-5E20, ATK-U101, ATK-U202)
- Explicit Visual Distinction: KNOWN ATTACK LIBRARY vs UNSEEN ATTACK TESTS (U1 & U2)
- Seeded / Reproducible Demo Mode (Seed: 42) & Live Variation Mode
- Dynamic Observable Context Telemetry: Hardware Fingerprints, Metropolitans, Recipients, Off-hours Timing
- Real-Time Attack Timeline with live timestamps, security alarms, 4-layer evaluation, and gateway verdicts
- R7 Customer Feedback Linkage for post-authorization fraud report on approved novel transactions
"""

import time
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Optional

from src.ui.icons import get_icon, render_icon_badge
from src.ui.timeline import render_timeline
from src.ui.decision_lineage import render_decision_lineage
from src.simulation.demo_account import (
    DEFAULT_DEMO_CUSTOMER_ID,
    DEFAULT_ATTACKER_RECEIVER_ID,
    KNOWN_LOCATION,
    KNOWN_DEVICE_ID
)
from src.simulation.attacker import (
    AttackerSimulator,
    AttackerState,
    ATTACKER_LOCATION,
    ATTACKER_DEVICE_ID,
    ATTACKER_POOL,
    ATTACK_STRATEGIES,
    ATTACK_RECIPIENTS,
    INDIAN_CITIES,
    calculate_haversine_distance,
    is_predefined_template,
    get_template_classification
)
from src.simulation.gateway import PaymentGatewaySimulator, GatewayTransferResult

def render_attack_lab(
    attacker: AttackerSimulator,
    gateway: PaymentGatewaySimulator,
    account_manager,
    ato_engine,
    sink
):
    """Renders the Attack Lab experience."""
    st.markdown("""
    <div style='margin-bottom: 16px;'>
        <h2 style='margin: 0; font-size: 1.6rem; font-weight: 800; color: #f8fafc; display: flex; align-items: center; gap: 10px;'>
            """ + get_icon("crosshair", size=24, color="#ef4444") + """
            Attack Action Lab & Adversarial Engine
        </h2>
        <div style='color: #94a3b8; font-size: 0.85rem; margin-top: 2px;'>
            External Adversarial Simulation • Known Strategies & Unseen Attacks (U1/U2) • Stateful Dynamic Adaptation
        </div>
    </div>
    """, unsafe_allow_html=True)

    cust = account_manager.get_account(DEFAULT_DEMO_CUSTOMER_ID)

    # -------------------------------------------------------------------------
    # CONTROL BAR: ADVERSARY SELECTION + SIMULATION MODE & SEED
    # -------------------------------------------------------------------------
    profiles = getattr(attacker, "get_attacker_profiles", lambda: ATTACKER_POOL)()
    prof_keys = list(profiles.keys())
    prof_labels = [f"{k} — {profiles[k].name} ({profiles[k].archetype})" for k in prof_keys]

    curr_aid = st.session_state.get("active_actor_id", "ATK-7F31")
    sel_idx = prof_keys.index(curr_aid) if curr_aid in prof_keys else 0

    col_act_sel, col_mode, col_seed = st.columns([2.2, 1.2, 1.0])
    with col_act_sel:
        new_sel_idx = st.selectbox(
            "Adversary Profile:",
            range(len(prof_keys)),
            index=sel_idx,
            format_func=lambda i: prof_labels[i],
            key="sel_attacker_profile_dropdown"
        )
        selected_actor_id = prof_keys[new_sel_idx]
        if selected_actor_id != curr_aid:
            st.session_state.active_actor_id = selected_actor_id
            if hasattr(attacker, "set_active_profile"):
                attacker.set_active_profile(selected_actor_id)
            st.rerun()

    with col_mode:
        curr_mode = getattr(attacker, "mode", "DEMO_SEEDED")
        mode_idx = 0 if curr_mode == "DEMO_SEEDED" else 1
        new_mode = st.selectbox(
            "Simulation Mode:",
            ["Demo Mode (Seeded)", "Live Variation Mode"],
            index=mode_idx,
            key="sel_sim_mode"
        )
        target_mode = "DEMO_SEEDED" if "Demo" in new_mode else "LIVE_VARIATION"
        if target_mode != curr_mode and hasattr(attacker, "set_mode"):
            attacker.set_mode(target_mode, seed=st.session_state.get("sim_seed_val", 42))

    with col_seed:
        seed_val = st.number_input(
            "RNG Seed:",
            min_value=1,
            max_value=999999,
            value=int(getattr(attacker, "seed", 42) or 42),
            step=1,
            key="sim_seed_val",
            disabled=(curr_mode != "DEMO_SEEDED")
        )
        if hasattr(attacker, "set_seed") and curr_mode == "DEMO_SEEDED" and seed_val != getattr(attacker, "seed", 42):
            attacker.set_seed(int(seed_val))

    active_prof = getattr(attacker, "get_active_profile", lambda aid=None: ATTACKER_POOL.get(aid or "ATK-7F31", ATTACKER_POOL["ATK-7F31"]))(st.session_state.get("active_actor_id", "ATK-7F31"))
    state: AttackerState = getattr(attacker, "get_attacker_state", lambda aid=None: None)(active_prof.actor_id)

    # Tactical Adaptation Banner
    latest_adapt = getattr(attacker, "get_latest_adaptation", lambda: None)()
    if latest_adapt:
        st.markdown(f"""
        <div style='background: linear-gradient(90deg, rgba(239, 68, 68, 0.15) 0%, rgba(245, 158, 11, 0.12) 100%); border: 1px solid rgba(239, 68, 68, 0.35); border-radius: 10px; padding: 10px 16px; margin-bottom: 18px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;'>
            <div style='display: flex; align-items: center; gap: 10px;'>
                {get_icon("zap", size=18, color="#ef4444")}
                <div>
                    <span style='font-size: 0.78rem; font-weight: 800; color: #ef4444; text-transform: uppercase;'>Adversarial Adaptation Active</span>
                    <div style='font-size: 0.84rem; color: #cbd5e1; margin-top: 1px;'>{latest_adapt['tactical_rationale']}</div>
                </div>
            </div>
            <div style='display: flex; gap: 8px;'>
                <span class="status-pill pill-amber" style='font-size: 0.72rem;'>Actor: {latest_adapt.get('actor_id', active_prof.actor_id)}</span>
                <span class="status-pill pill-red" style='font-size: 0.72rem;'>Attempt #{latest_adapt['attempt_number']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3-COLUMN ADVERSARIAL MATRIX
    # -------------------------------------------------------------------------
    col_atk, col_prog, col_cust = st.columns([1.15, 1.85, 1.0])

    # --- LEFT COLUMN: PERSISTENT ATTACKER STATE ---
    with col_atk:
        dist_from_home = calculate_haversine_distance(
            KNOWN_LOCATION["latitude"], KNOWN_LOCATION["longitude"],
            state.current_location["latitude"], state.current_location["longitude"]
        ) if state else 1030.0

        prev_dev = state.previous_device if state and state.previous_device else "None (Initial)"
        prev_loc = state.previous_location.get("city", "None") if state and state.previous_location else "None"
        attempts = state.attempt_count if state else 0
        last_dec = state.previous_decision if state and state.previous_decision else "None"

        st.markdown(f"""
        <div class="attacker-card">
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                <div style='font-size: 0.75rem; font-weight: 700; color: #ef4444; text-transform: uppercase; letter-spacing: 0.08em; display: flex; align-items: center; gap: 6px;'>
                    {get_icon("crosshair", size=13, color="#ef4444")} ATTACKER STATE
                </div>
                {render_icon_badge("zap", active_prof.risk_intent, "red", size=11)}
            </div>
            <div style='font-size: 1.22rem; font-weight: 800; color: #f8fafc;'>
                {active_prof.name} <code>({active_prof.actor_id})</code>
            </div>
            <div style='font-size: 0.78rem; color: #94a3b8; margin-top: 2px;'>
                {active_prof.archetype}
            </div>
            <div style='margin-top: 12px; font-size: 0.8rem; color: #cbd5e1; line-height: 1.6; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px;'>
                <div style='color: #94a3b8; font-size: 0.74rem; text-transform: uppercase; font-weight: 700;'>Location State</div>
                <div style='background: rgba(0,0,0,0.3); padding: 6px 10px; border-radius: 6px; margin: 4px 0 8px 0; font-size: 0.78rem;'>
                    <b>Home: Chennai</b> ➔ <span style='color: #ef4444; font-weight: 700;'>{dist_from_home:,.0f} km</span> ➔ <b>{state.current_location.get('city', active_prof.primary_city) if state else active_prof.primary_city}</b>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='color: #94a3b8;'>Current Device:</span>
                    <code>{state.current_device if state else active_prof.primary_device}</code>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='color: #94a3b8;'>Previous Device:</span>
                    <code style='color: #94a3b8;'>{prev_dev}</code>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='color: #94a3b8;'>Previous City:</span>
                    <span style='color: #94a3b8;'>{prev_loc}</span>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='color: #94a3b8;'>Active Recipient:</span>
                    <code>{state.current_recipient if state else "REC_MULE_01"}</code>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='color: #94a3b8;'>Attempt Count:</span>
                    <span style='color: #f59e0b; font-weight: 700;'>#{attempts}</span>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='color: #94a3b8;'>Last Gateway Verdict:</span>
                    <span style='color: {"#ef4444" if last_dec=="BLOCK" else ("#f59e0b" if last_dec=="HOLD" else "#10b981")}; font-weight: 700;'>{last_dec}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- CENTER COLUMN: DUAL-SECTION STRATEGY LIBRARY & UNSEEN LAB ---
    with col_prog:
        rec_keys = list(ATTACK_RECIPIENTS.keys())
        rec_labels = [f"{ATTACK_RECIPIENTS[k]['name']} ({ATTACK_RECIPIENTS[k]['type']})" for k in rec_keys]

        # Top-level Tabs separating Known Library from Unseen Attacks
        cat_tab_known, cat_tab_unseen = st.tabs([
            "🛡️ KNOWN ATTACK LIBRARY (1-6)",
            "🧪 UNSEEN ATTACK LAB (U1 & U2)"
        ])

        # =====================================================================
        # TAB 1: KNOWN ATTACK LIBRARY
        # =====================================================================
        with cat_tab_known:
            st.markdown(f"""
            <div style='font-size: 0.75rem; color: #94a3b8; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;'>
                <span>Predefined attack templates matching known fraud topologies.</span>
                {render_icon_badge("shield", "Predefined Templates", "blue", size=10)}
            </div>
            """, unsafe_allow_html=True)

            sc_tab1, sc_tab2, sc_tab3, sc_tab4, sc_tab5, sc_tab6 = st.tabs([
                "1. Smash & Grab",
                "2. Velocity Surge",
                "3. Low & Slow",
                "4. Impossible Travel",
                "5. New Device Hold",
                "6. Novel SIM-Swap"
            ])

            # 1. SMASH & GRAB
            with sc_tab1:
                st.markdown("""
                <div style='font-size: 0.82rem; color: #cbd5e1; margin-bottom: 8px;'>
                    <b>Objective:</b> High-value single-transfer balance extraction immediately after credential reset from a distant location.
                </div>
                """, unsafe_allow_html=True)
                col_s1_a, col_s1_b = st.columns(2)
                with col_s1_a:
                    s1_rec_idx = st.selectbox("Target Recipient:", range(len(rec_keys)), index=0, format_func=lambda i: rec_labels[i], key="s1_rec")
                    s1_rec = rec_keys[s1_rec_idx]
                with col_s1_b:
                    max_s1 = max(10.0, float(cust.balance))
                    s1_amt = st.number_input("Drain Amount (₹)", min_value=10.0, max_value=max_s1, value=min(8500.0, max_s1), step=500.0, key="s1_amt")

                if st.button("LAUNCH STRATEGY 1: SMASH & GRAB", type="primary", width="stretch", key="btn_exec_s1"):
                    _execute_attack_and_record(
                        attacker=attacker,
                        gateway=gateway,
                        scenario_type="account_takeover",
                        sender_id=DEFAULT_DEMO_CUSTOMER_ID,
                        receiver_id=s1_rec,
                        amount=s1_amt,
                        actor_id=active_prof.actor_id,
                        strategy_label="Smash & Grab (ATO)"
                    )

            # 2. VELOCITY SURGE
            with sc_tab2:
                st.markdown("""
                <div style='font-size: 0.82rem; color: #cbd5e1; margin-bottom: 8px;'>
                    <b>Objective:</b> Rapid burst payments (₹400 ➔ ₹700 ➔ ₹900 ➔ ₹1,200) within 30s to overwhelm sliding-window Redis accumulators.
                </div>
                """, unsafe_allow_html=True)
                col_s2_a, col_s2_b = st.columns(2)
                with col_s2_a:
                    s2_rec_idx = st.selectbox("Target Recipient:", range(len(rec_keys)), index=1, format_func=lambda i: rec_labels[i], key="s2_rec")
                    s2_rec = rec_keys[s2_rec_idx]
                with col_s2_b:
                    max_s2 = max(10.0, float(cust.balance))
                    s2_amt = st.number_input("Velocity Step Amount (₹)", min_value=10.0, max_value=max_s2, value=min(1200.0, max_s2), step=100.0, key="s2_amt")

                if st.button("LAUNCH STRATEGY 2: VELOCITY SURGE", type="primary", width="stretch", key="btn_exec_s2"):
                    _execute_attack_and_record(
                        attacker=attacker,
                        gateway=gateway,
                        scenario_type="velocity_surge",
                        sender_id=DEFAULT_DEMO_CUSTOMER_ID,
                        receiver_id=s2_rec,
                        amount=s2_amt,
                        actor_id=active_prof.actor_id,
                        strategy_label="Velocity Surge"
                    )

            # 3. LOW AND SLOW
            with sc_tab3:
                st.markdown("""
                <div style='font-size: 0.82rem; color: #cbd5e1; margin-bottom: 8px;'>
                    <b>Objective:</b> Stealth sub-threshold probing (₹350 – ₹650) executed inside the victim's home city to test whether FinPulse catches behavioral drift beyond simple amount thresholds.
                </div>
                """, unsafe_allow_html=True)
                col_s3_a, col_s3_b = st.columns(2)
                with col_s3_a:
                    s3_rec_idx = st.selectbox("Target Recipient:", range(len(rec_keys)), index=3, format_func=lambda i: rec_labels[i], key="s3_rec")
                    s3_rec = rec_keys[s3_rec_idx]
                with col_s3_b:
                    max_s3 = max(10.0, float(cust.balance))
                    s3_amt = st.number_input("Stealth Amount (₹)", min_value=10.0, max_value=max_s3, value=min(450.0, max_s3), step=50.0, key="s3_amt")

                if st.button("LAUNCH STRATEGY 3: LOW & SLOW", type="primary", width="stretch", key="btn_exec_s3"):
                    _execute_attack_and_record(
                        attacker=attacker,
                        gateway=gateway,
                        scenario_type="low_and_slow",
                        sender_id=DEFAULT_DEMO_CUSTOMER_ID,
                        receiver_id=s3_rec,
                        amount=s3_amt,
                        actor_id=active_prof.actor_id,
                        strategy_label="Low & Slow"
                    )

            # 4. IMPOSSIBLE TRAVEL
            with sc_tab4:
                st.markdown("""
                <div style='font-size: 0.82rem; color: #cbd5e1; margin-bottom: 8px;'>
                    <b>Objective:</b> Physical speed anomaly: 1,750 km jump (Chennai ➔ Delhi) in 45s (2,400 km/h) executed at 03:00 AM off-hours.
                </div>
                """, unsafe_allow_html=True)
                col_s4_a, col_s4_b = st.columns(2)
                with col_s4_a:
                    s4_rec_idx = st.selectbox("Target Recipient:", range(len(rec_keys)), index=1, format_func=lambda i: rec_labels[i], key="s4_rec")
                    s4_rec = rec_keys[s4_rec_idx]
                with col_s4_b:
                    max_s4 = max(10.0, float(cust.balance))
                    s4_amt = st.number_input("Travel Amount (₹)", min_value=10.0, max_value=max_s4, value=min(4200.0, max_s4), step=200.0, key="s4_amt")

                if st.button("LAUNCH STRATEGY 4: IMPOSSIBLE TRAVEL", type="primary", width="stretch", key="btn_exec_s4"):
                    _execute_attack_and_record(
                        attacker=attacker,
                        gateway=gateway,
                        scenario_type="impossible_travel",
                        sender_id=DEFAULT_DEMO_CUSTOMER_ID,
                        receiver_id=s4_rec,
                        amount=s4_amt,
                        actor_id=active_prof.actor_id,
                        strategy_label="Impossible Travel"
                    )

            # 5. NEW DEVICE HOLD
            with sc_tab5:
                st.markdown("""
                <div style='font-size: 0.82rem; color: #cbd5e1; margin-bottom: 8px;'>
                    <b>Objective:</b> Login from victim's home city using an unfamiliar device (<code>DEV-ATK-4408</code>), producing Medium risk that triggers a <b>Gateway HOLD</b> for customer 2-way verification.
                </div>
                """, unsafe_allow_html=True)
                col_s5_a, col_s5_b = st.columns(2)
                with col_s5_a:
                    s5_rec_idx = st.selectbox("Target Recipient:", range(len(rec_keys)), index=2, format_func=lambda i: rec_labels[i], key="s5_rec")
                    s5_rec = rec_keys[s5_rec_idx]
                with col_s5_b:
                    max_s5 = max(10.0, float(cust.balance))
                    s5_amt = st.number_input("Hold Amount (₹)", min_value=10.0, max_value=max_s5, value=min(2500.0, max_s5), step=250.0, key="s5_amt")

                if st.button("LAUNCH STRATEGY 5: NEW DEVICE HOLD", type="primary", width="stretch", key="btn_exec_s5"):
                    _execute_attack_and_record(
                        attacker=attacker,
                        gateway=gateway,
                        scenario_type="new_device_hold",
                        sender_id=DEFAULT_DEMO_CUSTOMER_ID,
                        receiver_id=s5_rec,
                        amount=s5_amt,
                        actor_id=active_prof.actor_id,
                        strategy_label="New Device Hold"
                    )

            # 6. NOVEL SIM-SWAP
            with sc_tab6:
                st.markdown("""
                <div style='font-size: 0.82rem; color: #cbd5e1; margin-bottom: 8px;'>
                    <b>Objective:</b> Subtle novel attack (SIM swap token bypass) with moderate ₹3,500 amount to test the <b>Customer Fraud Report ➔ Confirmed Ground-Truth Label ➔ Controlled ML Retraining Feedback Loop</b>.
                </div>
                """, unsafe_allow_html=True)
                col_s6_a, col_s6_b = st.columns(2)
                with col_s6_a:
                    s6_rec_idx = st.selectbox("Target Recipient:", range(len(rec_keys)), index=3, format_func=lambda i: rec_labels[i], key="s6_rec")
                    s6_rec = rec_keys[s6_rec_idx]
                with col_s6_b:
                    max_s6 = max(10.0, float(cust.balance))
                    s6_amt = st.number_input("Novel Attack Amount (₹)", min_value=10.0, max_value=max_s6, value=min(3500.0, max_s6), step=250.0, key="s6_amt")

                if st.button("LAUNCH STRATEGY 6: NOVEL SIM-SWAP", type="primary", width="stretch", key="btn_exec_s6"):
                    _execute_attack_and_record(
                        attacker=attacker,
                        gateway=gateway,
                        scenario_type="novel_attack",
                        sender_id=DEFAULT_DEMO_CUSTOMER_ID,
                        receiver_id=s6_rec,
                        amount=s6_amt,
                        actor_id=active_prof.actor_id,
                        strategy_label="Novel SIM-Swap"
                    )

        # =====================================================================
        # TAB 2: UNSEEN ATTACK LAB (U1 & U2)
        # =====================================================================
        with cat_tab_unseen:
            st.markdown(f"""
            <div style='background: rgba(139, 92, 246, 0.08); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div style='font-size: 0.8rem; font-weight: 700; color: #a78bfa;'>
                        UNSEEN ADVERSARIAL EVALUATION
                    </div>
                    {render_icon_badge("alert_triangle", "Not Predefined Template", "purple", size=10)}
                </div>
                <div style='font-size: 0.78rem; color: #cbd5e1; margin-top: 2px;'>
                    Generates novel transaction & security context flowing directly into the frozen production pipeline.
                </div>
            </div>
            """, unsafe_allow_html=True)

            u_tab1, u_tab2 = st.tabs([
                "U1 — Context Mutation",
                "U2 — Compound Novel Attack"
            ])

            # U1 — CONTEXT MUTATION
            with u_tab1:
                st.markdown("""
                <div class="unseen-spec-card" style='background: rgba(0,0,0,0.25); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px; margin-bottom: 10px;'>
                    <div style='font-size: 0.78rem; font-weight: 700; color: #38bdf8;'>SCENARIO: U1 — Context Mutation</div>
                    <div style='font-size: 0.72rem; color: #94a3b8; margin-top: 2px;'>Template Classification: <b>Not one of the predefined attack templates</b></div>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 8px; font-size: 0.76rem;'>
                        <div>• Amount: <span style='color: #f8fafc;'>Moderate (₹1,800 – ₹3,200)</span></div>
                        <div>• Device: <span style='color: #f8fafc;'>Unregistered (<code>DEV-MUT-8842</code>)</span></div>
                        <div>• Location: <span style='color: #f8fafc;'>Regional Hub (Hyderabad, 510 km)</span></div>
                        <div>• Recipient: <span style='color: #f8fafc;'>New Unindexed Peer (<code>REC_PEER_07</code>)</span></div>
                        <div>• Timing: <span style='color: #f8fafc;'>Unusual (23:45 Off-Peak)</span></div>
                        <div>• Auth: <span style='color: #10b981;'>Valid Token</span> • Velocity: <span style='color: #f8fafc;'>Normal</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_u1_a, col_u1_b = st.columns(2)
                with col_u1_a:
                    u1_rec = st.selectbox("Recipient:", ["REC_PEER_07", "REC_VENDOR_09", "REC_OFFSHORE_03"], index=0, key="u1_rec")
                with col_u1_b:
                    max_u1 = max(10.0, float(cust.balance))
                    u1_amt = st.number_input("Amount (₹)", min_value=10.0, max_value=max_u1, value=min(2450.0, max_u1), step=150.0, key="u1_amt")

                if st.button("EXECUTE U1 — CONTEXT MUTATION TEST", type="primary", width="stretch", key="btn_exec_u1"):
                    _execute_attack_and_record(
                        attacker=attacker,
                        gateway=gateway,
                        scenario_type="u1_context_mutation",
                        sender_id=DEFAULT_DEMO_CUSTOMER_ID,
                        receiver_id=u1_rec,
                        amount=u1_amt,
                        actor_id="ATK-U101",
                        strategy_label="U1 — Context Mutation (Unseen)"
                    )

            # U2 — COMPOUND NOVEL ATTACK
            with u_tab2:
                st.markdown("""
                <div class="unseen-spec-card" style='background: rgba(0,0,0,0.25); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px; margin-bottom: 10px;'>
                    <div style='font-size: 0.78rem; font-weight: 700; color: #a78bfa;'>SCENARIO: U2 — Compound Novel Attack</div>
                    <div style='font-size: 0.72rem; color: #94a3b8; margin-top: 2px;'>Template Classification: <b>Not one of the predefined attack templates</b></div>
                    <div style='margin-top: 6px; font-size: 0.76rem; color: #cbd5e1;'>
                        Synthesizes <b>multiple moderate anomalies</b> without any single extreme trigger to evaluate multi-layer hybrid fusion:
                    </div>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 8px; font-size: 0.76rem;'>
                        <div>• Hardware: <span style='color: #f8fafc;'><code>DEV-COMPOUND-77</code></span></div>
                        <div>• Recipient: <span style='color: #f8fafc;'>Unindexed (<code>REC_VENDOR_09</code>)</span></div>
                        <div>• Timing: <span style='color: #f8fafc;'>Off-Hours (02:41 AM)</span></div>
                        <div>• Location: <span style='color: #f8fafc;'>Pune (915 km)</span></div>
                        <div>• Amount: <span style='color: #f8fafc;'>Elevated (₹2,750)</span></div>
                        <div>• Velocity: <span style='color: #f8fafc;'>Slight Elevation (2 tx/10m)</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_u2_a, col_u2_b = st.columns(2)
                with col_u2_a:
                    u2_rec = st.selectbox("Recipient:", ["REC_VENDOR_09", "REC_PEER_07", "REC_CRYPTO_02"], index=0, key="u2_rec")
                with col_u2_b:
                    max_u2 = max(10.0, float(cust.balance))
                    u2_amt = st.number_input("Amount (₹)", min_value=10.0, max_value=max_u2, value=min(2750.0, max_u2), step=150.0, key="u2_amt")

                if st.button("EXECUTE U2 — COMPOUND NOVEL TEST", type="primary", width="stretch", key="btn_exec_u2"):
                    _execute_attack_and_record(
                        attacker=attacker,
                        gateway=gateway,
                        scenario_type="u2_compound_novel",
                        sender_id=DEFAULT_DEMO_CUSTOMER_ID,
                        receiver_id=u2_rec,
                        amount=u2_amt,
                        actor_id="ATK-U202",
                        strategy_label="U2 — Compound Novel Attack (Unseen)"
                    )

    # --- RIGHT COLUMN: TARGET CUSTOMER STATE ---
    with col_cust:
        target_status_badge = "green" if cust.account_status == "SAFE" else ("amber" if cust.account_status == "UNDER_ATTACK" else "red")
        st.markdown(f"""
        <div class="fp-card">
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                <div style='font-size: 0.75rem; font-weight: 700; color: #38bdf8; text-transform: uppercase;'>TARGET ACCOUNT</div>
                {render_icon_badge("shield" if cust.account_status == "SAFE" else "alert_triangle", cust.account_status, target_status_badge, size=11)}
            </div>
            <div style='font-size: 1.1rem; font-weight: 700; color: #f8fafc;'>{cust.customer_id}</div>
            <div style='font-size: 1.6rem; font-weight: 800; color: #38bdf8; margin: 6px 0;'>₹{cust.balance:,.2f}</div>
            <div style='font-size: 0.8rem; color: #94a3b8; line-height: 1.6; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px; margin-top: 6px;'>
                <div>Home City: <code>{cust.known_city}</code></div>
                <div>Registered Device: <code>{cust.known_device_id}</code></div>
                <div style='margin-top: 6px; color: #10b981; font-weight: 600;'>Protected Balance: ₹{cust.balance:,.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ATTACK RESULT HERO BANNER & DECISION LINEAGE
    # -------------------------------------------------------------------------
    last_res: Optional[GatewayTransferResult] = st.session_state.get("last_gateway_result")
    if last_res:
        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

        if last_res.decision == "BLOCK":
            res_bg = "rgba(239, 68, 68, 0.12)"
            res_border = "#ef4444"
            res_title = "ATTACK DETECTED & INTERCEPTED"
            res_subtitle = "Authoritative Gateway Block • ₹0 Moved • Customer Balance Protected"
        elif last_res.decision == "HOLD":
            res_bg = "rgba(245, 158, 11, 0.12)"
            res_border = "#f59e0b"
            res_title = "SUSPICIOUS THREAT PLACED ON HOLD"
            res_subtitle = "Money NOT Transferred • Customer Verification Required on Mobile Wallet"
        else:
            res_bg = "rgba(16, 185, 129, 0.12)"
            res_border = "#10b981"
            res_title = "TRANSACTION AUTHORIZED BY GATEWAY"
            res_subtitle = "Low Risk Verified • Gateway Executed Transfer"

        st.markdown(f"""
        <div style='background: {res_bg}; border: 2px solid {res_border}; border-radius: 14px; padding: 22px 26px; margin-bottom: 20px;'>
            <div style='display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 14px;'>
                <div>
                    <h2 style='color: {res_border}; margin: 0; font-size: 1.8rem; font-weight: 800; letter-spacing: -0.02em;'>
                        {res_title}
                    </h2>
                    <div style='font-size: 0.95rem; color: #cbd5e1; margin-top: 4px;'>{res_subtitle}</div>
                </div>
                <div style='text-align: right; background: rgba(0,0,0,0.35); border-radius: 8px; padding: 8px 16px; border: 1px solid rgba(255,255,255,0.08);'>
                    <div style='font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;'>Attempted Drain</div>
                    <div style='font-size: 1.6rem; font-weight: 800; color: #f8fafc;'>₹{last_res.amount:,.2f}</div>
                </div>
            </div>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-top: 18px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 14px;'>
                <div>
                    <div style='font-size: 0.72rem; color: #94a3b8;'>Gateway Verdict</div>
                    <div style='font-size: 1.15rem; font-weight: 700; color: {res_border};'>`{last_res.decision}`</div>
                </div>
                <div>
                    <div style='font-size: 0.72rem; color: #94a3b8;'>Status Code</div>
                    <div style='font-size: 1.15rem; font-weight: 700; color: #f8fafc;'>`{last_res.status}`</div>
                </div>
                <div>
                    <div style='font-size: 0.72rem; color: #94a3b8;'>Money Transferred?</div>
                    <div style='font-size: 1.15rem; font-weight: 700; color: {"#ef4444" if last_res.money_transferred else "#10b981"};'>
                        {"YES (Deducted)" if last_res.money_transferred else "NO (₹0 Transferred)"}
                    </div>
                </div>
                <div>
                    <div style='font-size: 0.72rem; color: #94a3b8;'>Funds Protected</div>
                    <div style='font-size: 1.15rem; font-weight: 700; color: #10b981;'>
                        ₹{last_res.amount if not last_res.money_transferred else 0:,.2f}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # TIMELINE & 4-LAYER LINEAGE
        # ---------------------------------------------------------------------
        col_t1, col_t2 = st.columns([1.2, 1.0])

        with col_t1:
            st.markdown("""
            <div style='font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;'>
                """ + get_icon("clock", size=18, color="#38bdf8") + """
                Real-Time Attack Timeline
            </div>
            """, unsafe_allow_html=True)
            render_timeline(st.session_state.get("last_attack_timeline", []))

        with col_t2:
            st.markdown("""
            <div style='font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;'>
                """ + get_icon("shield", size=18, color="#38bdf8") + """
                Decision Pipeline Lineage
            </div>
            """, unsafe_allow_html=True)
            render_decision_lineage(last_res)

    # -------------------------------------------------------------------------
    # ATTACK LEDGER
    # -------------------------------------------------------------------------
    if st.session_state.attack_history:
        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        st.markdown("### Adversarial Attack Ledger")
        st.dataframe(pd.DataFrame(st.session_state.attack_history), width="stretch", hide_index=True)


def _execute_attack_and_record(
    attacker: AttackerSimulator,
    gateway: PaymentGatewaySimulator,
    scenario_type: str,
    sender_id: str,
    receiver_id: str,
    amount: float,
    actor_id: str,
    strategy_label: str
) -> None:
    """Helper orchestrating attack execution, gateway processing, adaptation, and state recording."""
    with st.spinner(f"Submitting {strategy_label} to Payment Gateway..."):
        events, req = attacker.build_scenario_transfer(
            scenario_type=scenario_type,
            sender_id=sender_id,
            receiver_id=receiver_id,
            amount=amount,
            actor_id=actor_id
        )
        res = gateway.process_transfer(req)
        if hasattr(attacker, "record_outcome_and_adapt"):
            attacker.record_outcome_and_adapt(res)
        timeline = attacker.build_attack_timeline(events, req, res)

        st.session_state.last_gateway_result = res
        st.session_state.latest_tx_detail = res.eval_detail
        st.session_state.last_attack_timeline = timeline
        if res.hold_case:
            st.session_state.active_hold = {"case": res.hold_case, "token": res.confirmation_token}

        st.session_state.wallet_tx_history.insert(0, {
            "tx_id": res.transaction_id,
            "recipient": req.get("merchant_id", receiver_id),
            "amount": res.amount,
            "decision": res.decision,
            "status": res.status,
            "timestamp": time.strftime('%H:%M:%S', time.localtime(res.timestamp)),
            "money_moved": res.money_transferred
        })

        is_unseen = not is_predefined_template(scenario_type)
        st.session_state.attack_history.insert(0, {
            "Adversary": actor_id,
            "Strategy": strategy_label,
            "Category": "UNSEEN TEST" if is_unseen else "KNOWN TEMPLATE",
            "Tx ID": res.transaction_id,
            "Amount": f"₹{res.amount:,.2f}",
            "ML": res.ml_decision,
            "Hybrid Risk": f"{res.risk_score:.1f}",
            "ATO": res.ato_action,
            "Gateway": res.decision,
            "Funds Transferred": "NO (Protected)" if not res.money_transferred else "YES",
            "Latency": f"{res.latency_ms:.2f} ms",
            "Time": time.strftime('%H:%M:%S', time.localtime(res.timestamp))
        })
        st.rerun()
