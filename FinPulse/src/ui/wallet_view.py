"""
FinPulse Customer Wallet Experience.

Provides a clean, realistic consumer banking interface:
- High-end financial card with available balance & status
- Intuitive Send Money flow with pre-authorization review
- Invariant: Funds move ONLY AFTER authoritative Gateway authorization
- HOLD state interactive verification modal (Yes, It's Me / No, Block It)
- Security unlock via 2FA if account was flagged under attack
- Transaction history with customer-facing "Report Fraud" action that feeds ML lifecycle
"""

import time
import json
from typing import Dict, Any, Optional, List
import streamlit as st

from src.ui.icons import get_icon, render_icon_badge
from src.simulation.demo_account import (
    DEFAULT_DEMO_CUSTOMER_ID,
    KNOWN_LOCATION,
    KNOWN_DEVICE_ID
)
from src.simulation.gateway import GatewayTransferResult
from src.workflow.hold_workflow import HoldCase

def render_customer_wallet(
    account_manager,
    gateway,
    ato_engine,
    hold_engine,
    sink,
    predictor
):
    """Renders the complete Customer Wallet experience."""
    st.markdown("""
    <div style='margin-bottom: 16px;'>
        <h2 style='margin: 0; font-size: 1.6rem; font-weight: 800; color: #f8fafc; display: flex; align-items: center; gap: 10px;'>
            """ + get_icon("wallet", size=24, color="#38bdf8") + """
            My Financial Wallet
        </h2>
        <div style='color: #94a3b8; font-size: 0.85rem; margin-top: 2px;'>
            Secure Consumer Payment Experience • Protected by FinPulse Gateway
        </div>
    </div>
    """, unsafe_allow_html=True)

    cust = account_manager.get_account(DEFAULT_DEMO_CUSTOMER_ID)
    recipients = account_manager.get_recipients()

    status_badge_type = "green" if cust.account_status == "SAFE" else ("amber" if cust.account_status == "UNDER_ATTACK" else "red")
    status_text = "Account Safe" if cust.account_status == "SAFE" else ("Under Threat Lock" if cust.account_status == "UNDER_ATTACK" else "Suspended")

    # -------------------------------------------------------------------------
    # 1. AVAILABLE BALANCE CARD
    # -------------------------------------------------------------------------
    st.markdown(f"""
    <div class="wallet-card-hero">
        <div style='display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;'>
            <div>
                <div style='font-size: 0.75rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.08em; display: flex; align-items: center; gap: 6px;'>
                    {get_icon("shield", size=14, color="#38bdf8")} ACCOUNT PROTECTED
                </div>
                <div style='font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin-top: 4px;'>
                    {cust.customer_id}
                </div>
                <div style='font-size: 0.8rem; color: #94a3b8; margin-top: 2px; display: flex; align-items: center; gap: 6px;'>
                    {get_icon("map_pin", size=13, color="#94a3b8")} Primary Node: Chennai Central
                </div>
            </div>
            <div style='text-align: right;'>
                <div style='font-size: 0.8rem; color: #94a3b8;'>Available Balance</div>
                <div style='font-size: 2.6rem; font-weight: 800; color: #38bdf8; letter-spacing: -0.03em; margin: 2px 0;'>
                    ₹{cust.balance:,.2f}
                </div>
                <div>
                    {render_icon_badge("shield" if cust.account_status == "SAFE" else "alert_triangle", status_text, status_badge_type)}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. ACCOUNT UNDER ATTACK / 2FA RECOVERY BANNER
    # -------------------------------------------------------------------------
    if cust.account_status == "UNDER_ATTACK":
        st.markdown(f"""
        <div style='background: rgba(239, 68, 68, 0.12); border: 2px solid #ef4444; border-radius: 14px; padding: 20px 24px; margin-bottom: 22px;'>
            <div style='display: flex; align-items: flex-start; gap: 14px;'>
                <div>{get_icon("lock", size=28, color="#ef4444")}</div>
                <div style='flex: 1;'>
                    <h3 style='color: #ef4444; margin: 0; font-size: 1.2rem; font-weight: 800;'>
                        ACCOUNT TEMPORARILY LOCKED (THREAT INTERCEPTED)
                    </h3>
                    <div style='color: #f8fafc; font-size: 0.9rem; margin-top: 4px;'>
                        An unauthorized Account Takeover or suspicious login pattern was intercepted by FinPulse Gateway.
                    </div>
                    <div style='color: #cbd5e1; font-size: 0.82rem; margin-top: 4px;'>
                        <b>Protective Action:</b> Outbound transfers are paused to safeguard your ₹{cust.balance:,.2f} balance.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_rec1, col_rec2 = st.columns([1.6, 2.0])
        with col_rec1:
            if st.button("Verify Identity via 2FA & Restore Account", type="primary", width="stretch", key="btn_unlock_wallet"):
                ato_engine.reset(DEFAULT_DEMO_CUSTOMER_ID)
                account_manager.update_status(DEFAULT_DEMO_CUSTOMER_ID, "SAFE")
                if predictor and hasattr(predictor, "redis_window"):
                    try:
                        predictor.redis_window.clear_customer_state(DEFAULT_DEMO_CUSTOMER_ID)
                    except Exception:
                        pass
                st.session_state.last_gateway_result = None
                st.session_state.active_hold = None
                st.success("Identity verified! Account restored to SAFE status.")
                st.rerun()

    # -------------------------------------------------------------------------
    # 3. HOLD VERIFICATION PROMPT
    # -------------------------------------------------------------------------
    active_hold = st.session_state.active_hold
    if active_hold:
        case: HoldCase = active_hold["case"]
        token = active_hold["token"]
        meta_d = case.metadata or {}
        rec_dest = meta_d.get("receiver_id", "Unknown Recipient")
        rec_dev = meta_d.get("device_id", "Unrecognized Device")
        loc_city = meta_d.get("location", {}).get("city", "Mumbai")

        st.markdown(f"""
        <div style='background: rgba(245, 158, 11, 0.12); border: 2px solid #f59e0b; border-radius: 14px; padding: 22px; margin-bottom: 24px;'>
            <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;'>
                <div style='display: flex; align-items: center; gap: 10px;'>
                    {get_icon("alert_triangle", size=24, color="#f59e0b")}
                    <div>
                        <h3 style='color: #f59e0b; margin: 0; font-size: 1.25rem; font-weight: 800;'>PAYMENT ON HOLD — VERIFICATION REQUIRED</h3>
                        <div style='color: #f8fafc; font-size: 0.9rem; margin-top: 2px;'>We detected an unusual payment pending authorization.</div>
                    </div>
                </div>
                <div style='background: rgba(0,0,0,0.4); border-radius: 8px; padding: 6px 16px; border: 1px solid rgba(245, 158, 11, 0.3); font-size: 1.3rem; font-weight: 800; color: #f59e0b;'>
                    ₹{case.amount:,.2f}
                </div>
            </div>
            <div style='margin-top: 14px; font-size: 0.85rem; color: #cbd5e1; line-height: 1.6; background: rgba(0,0,0,0.25); padding: 12px 16px; border-radius: 8px;'>
                <div><b>Recipient:</b> <code>{rec_dest}</code></div>
                <div><b>Device:</b> <code>{rec_dev}</code></div>
                <div><b>Location:</b> {loc_city} (Registered: Chennai)</div>
                <div style='color: #f59e0b; font-weight: 700; margin-top: 4px;'>Status: Funds HELD • ₹0 Transferred • Your balance remains protected</div>
            </div>
            <div style='margin-top: 14px; font-weight: 700; color: #f8fafc;'>Did you initiate this transaction?</div>
        </div>
        """, unsafe_allow_html=True)

        col_vh1, col_vh2, col_vh3 = st.columns([1.2, 1.2, 2])
        with col_vh1:
            if st.button("YES, IT'S ME", type="primary", width="stretch", key="btn_wallet_confirm_hold"):
                try:
                    conf_res = gateway.resolve_hold(case.hold_id, token, action="CONFIRM")
                    st.session_state.last_gateway_result = conf_res
                    st.session_state.active_hold = None
                    if "wallet_tx_history" in st.session_state:
                        for h in st.session_state.wallet_tx_history:
                            if h.get("tx_id") == case.transaction_id:
                                h["decision"] = "APPROVE"
                                h["status"] = "RELEASED_AND_EXECUTED"
                                h["money_moved"] = True
                                break
                    if "attack_history" in st.session_state:
                        for a in st.session_state.attack_history:
                            if a.get("Tx ID") == case.transaction_id:
                                a["Gateway Decision"] = "APPROVE"
                                a["Status"] = "RELEASED_AND_EXECUTED"
                                a["Money Transferred"] = "YES"
                                break
                    st.rerun()
                except Exception as ex:
                    st.error(f"Error resolving hold: {ex}")
        with col_vh2:
            if st.button("NO, BLOCK IT", type="secondary", width="stretch", key="btn_wallet_deny_hold"):
                try:
                    deny_res = gateway.resolve_hold(case.hold_id, token, action="DENY")
                    st.session_state.last_gateway_result = deny_res
                    st.session_state.active_hold = None
                    if "wallet_tx_history" in st.session_state:
                        for h in st.session_state.wallet_tx_history:
                            if h.get("tx_id") == case.transaction_id:
                                h["decision"] = "BLOCK"
                                h["status"] = "DENIED_AND_BLOCKED"
                                h["money_moved"] = False
                                break
                    if "attack_history" in st.session_state:
                        for a in st.session_state.attack_history:
                            if a.get("Tx ID") == case.transaction_id:
                                a["Gateway Decision"] = "BLOCK"
                                a["Status"] = "DENIED_AND_BLOCKED"
                                a["Money Transferred"] = "NO (Protected)"
                                break
                    st.rerun()
                except Exception as ex:
                    st.error(f"Error denying hold: {ex}")

    # -------------------------------------------------------------------------
    # 4. LAST GATEWAY DECISION BANNER
    # -------------------------------------------------------------------------
    last_res: Optional[GatewayTransferResult] = st.session_state.last_gateway_result
    if last_res and not active_hold:
        if last_res.decision == "BLOCK" or last_res.status in ["DECLINED", "DENIED", "DECLINED_ATO_SUSPENDED", "DECLINED_BLOCK", "DENIED_AND_BLOCKED"]:
            st.markdown(f"""
            <div style='background: rgba(239, 68, 68, 0.12); border: 2px solid #ef4444; border-radius: 12px; padding: 18px 22px; margin-bottom: 22px;'>
                <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;'>
                    <div style='display: flex; align-items: center; gap: 12px;'>
                        {get_icon("x", size=24, color="#ef4444")}
                        <div>
                            <h4 style='color: #ef4444; margin: 0; font-size: 1.2rem; font-weight: 800;'>PAYMENT BLOCKED</h4>
                            <div style='color: #cbd5e1; font-size: 0.88rem; margin-top: 3px;'>
                                Transfer of ₹{last_res.amount:,.2f} to <code>{last_res.receiver_id}</code> was rejected to protect your account.
                            </div>
                        </div>
                    </div>
                    <div style='text-align: right;'>
                        <div style='color: #10b981; font-weight: 800; font-size: 1.1rem;'>Funds Transferred: ₹0.00</div>
                        <div style='color: #94a3b8; font-size: 0.8rem;'>Protected Balance: ₹{cust.balance:,.2f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif last_res.decision == "APPROVE" and last_res.money_transferred:
            st.markdown(f"""
            <div style='background: rgba(16, 185, 129, 0.12); border: 2px solid #10b981; border-radius: 12px; padding: 18px 22px; margin-bottom: 22px;'>
                <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;'>
                    <div style='display: flex; align-items: center; gap: 12px;'>
                        {get_icon("check", size=24, color="#10b981")}
                        <div>
                            <h4 style='color: #10b981; margin: 0; font-size: 1.2rem; font-weight: 800;'>PAYMENT APPROVED</h4>
                            <div style='color: #cbd5e1; font-size: 0.88rem; margin-top: 3px;'>
                                ₹{last_res.amount:,.2f} successfully authorized and sent to <code>{last_res.receiver_id}</code>.
                            </div>
                        </div>
                    </div>
                    <div style='text-align: right;'>
                        <div style='color: #38bdf8; font-weight: 800; font-size: 1.15rem;'>New Balance: ₹{cust.balance:,.2f}</div>
                        <div style='color: #10b981; font-size: 0.78rem;'>Payment Completed</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 5. SEND MONEY & REVIEW FLOW
    # -------------------------------------------------------------------------
    col_w1, col_w2 = st.columns([1.2, 1.0])

    with col_w1:
        st.markdown(f"""
        <div class="fp-card">
            <div class="fp-card-header">
                <div class="fp-card-title">
                    {get_icon("arrow_right", size=16, color="#38bdf8")} Send Money
                </div>
                <span style='font-size: 0.75rem; color: #94a3b8;'>Instant Authorization</span>
            </div>
        """, unsafe_allow_html=True)

        rec_opts = list(recipients.keys())
        rec_labels = [f"{recipients[k]['name']} ({k})" for k in rec_opts]
        sel_idx = st.selectbox("Recipient", range(len(rec_opts)), format_func=lambda i: rec_labels[i], key="wallet_sel_rec")
        recipient_id = rec_opts[sel_idx]

        col_amt, col_purp = st.columns(2)
        with col_amt:
            max_send = max(1.0, float(cust.balance))
            send_amt = st.number_input("Amount (₹)", min_value=1.0, max_value=max_send, value=min(500.0, max_send), step=50.0, key="wallet_amt_input")
        with col_purp:
            purpose = st.selectbox("Purpose", ["General Transfer", "Groceries / Shopping", "Family Support", "Emergency", "Rent / Utilities"], key="wallet_purpose")

        # Payment Execution Flow (Section 11)
        if st.button("Review Payment", type="primary", width="stretch", key="btn_wallet_send"):
            with st.spinner("Securing Payment: Verifying context • Screening fraud • Gateway authorization..."):
                tx_id = f"tx_cust_{int(time.time()*1000)}"
                tx_request = {
                    "transaction_id": tx_id,
                    "customer_id": DEFAULT_DEMO_CUSTOMER_ID,
                    "merchant_id": recipient_id,
                    "amount": float(send_amt),
                    "timestamp": time.time(),
                    "payment_type": "TRANSFER",
                    "category": "transfer",
                    "origin_balance": float(cust.balance),
                    "dest_balance": float(account_manager.get_balance(recipient_id)),
                    "auth_verified": True,
                    "device_id": KNOWN_DEVICE_ID,
                    "latitude": KNOWN_LOCATION["latitude"],
                    "longitude": KNOWN_LOCATION["longitude"],
                    "location": KNOWN_LOCATION,
                    "home_latitude": KNOWN_LOCATION["latitude"],
                    "home_longitude": KNOWN_LOCATION["longitude"],
                    "home_location": KNOWN_LOCATION,
                    "raw_metadata": {"purpose": purpose, "channel": "customer_wallet"}
                }

                # Authoritative Gateway Evaluation
                res = gateway.process_transfer(tx_request)
                st.session_state.last_gateway_result = res
                st.session_state.latest_tx_detail = res.eval_detail

                if res.hold_case:
                    st.session_state.active_hold = {"case": res.hold_case, "token": res.confirmation_token}

                # Insert into history
                st.session_state.wallet_tx_history.insert(0, {
                    "tx_id": res.transaction_id,
                    "recipient": recipient_id,
                    "amount": res.amount,
                    "decision": res.decision,
                    "status": res.status,
                    "timestamp": time.strftime('%H:%M:%S', time.localtime(res.timestamp)),
                    "money_moved": res.money_transferred
                })
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with col_w2:
        st.markdown(f"""
        <div class="fp-card">
            <div class="fp-card-header">
                <div class="fp-card-title">
                    {get_icon("shield", size=16, color="#10b981")} FinPulse Protection Guarantee
                </div>
            </div>
            <div style='font-size: 0.84rem; color: #cbd5e1; line-height: 1.6;'>
                <div style='color: #38bdf8; font-weight: 700; margin-bottom: 6px;'>Pre-Settlement Authorization Boundary</div>
                Every payment request is securely screened by FinPulse Gateway before money moves.
                <div style='margin: 10px 0; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; font-family: monospace; font-size: 0.78rem;'>
                    Wallet ➔ Gateway Ingress ➔ Real-Time Defense ➔ Verdict
                </div>
                <div style='color: #94a3b8; font-size: 0.8rem;'>
                    • <b>APPROVE:</b> Safe transaction, funds transferred instantly.<br>
                    • <b>HOLD:</b> Unusual pattern, funds stay in your account pending your confirmation.<br>
                    • <b>BLOCK:</b> Threat detected, transfer rejected, ₹0 moved.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 6. TRANSACTION HISTORY & REPORT FRAUD FLOW
    # -------------------------------------------------------------------------
    st.markdown("""
    <div style='margin: 28px 0 12px 0;'>
        <h3 style='margin: 0; font-size: 1.25rem; font-weight: 700; color: #f8fafc;'>Recent Activity</h3>
        <div style='color: #94a3b8; font-size: 0.8rem;'>Completed transactions and fraud dispute reporting</div>
    </div>
    """, unsafe_allow_html=True)

    # Reporting Fraud Modal / Confirmation
    rep_tx = st.session_state.get("reporting_tx_id")
    if rep_tx:
        st.markdown(f"""
        <div style='background: rgba(239, 68, 68, 0.12); border: 2px solid #ef4444; border-radius: 12px; padding: 18px 22px; margin-bottom: 20px;'>
            <h4 style='color: #ef4444; margin: 0; font-size: 1.15rem; font-weight: 800;'>REPORT UNAUTHORIZED PAYMENT</h4>
            <div style='color: #f8fafc; font-size: 0.9rem; margin-top: 4px;'>
                You are disputing transaction <code>{rep_tx}</code> as unauthorized.
            </div>
            <div style='color: #cbd5e1; font-size: 0.82rem; margin-top: 4px;'>
                This action attaches a confirmed fraud label to the ML feedback pipeline.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_rep1, col_rep2, col_rep3 = st.columns([1.4, 1.2, 2])
        with col_rep1:
            if st.button("CONFIRM FRAUD REPORT", type="primary", width="stretch", key="btn_confirm_report_fraud"):
                sec_case = gateway.report_unauthorized_transaction(
                    transaction_id=rep_tx,
                    customer_id=DEFAULT_DEMO_CUSTOMER_ID,
                    customer_report="UNAUTHORIZED",
                    attack_type="UNKNOWN",
                    notes="Customer reported unauthorized payment via wallet"
                )
                st.session_state.last_reported_case = sec_case
                st.session_state.reporting_tx_id = None
                st.success(f"Security Case {sec_case.case_id} Created! Ground-truth fraud label confirmed (1). Entered into ML feedback pipeline.")
                st.rerun()

        with col_rep2:
            if st.button("Cancel", width="stretch", key="btn_cancel_report"):
                st.session_state.reporting_tx_id = None
                st.rerun()

    # Load history
    w_txs = st.session_state.wallet_tx_history
    if not w_txs:
        db_txs = sink.get_recent_transactions(customer_id=DEFAULT_DEMO_CUSTOMER_ID, limit=8)
        if db_txs:
            w_txs = []
            for t in db_txs:
                tx_id = t.get("transaction_id", "")
                dec_rec = sink.get_decision(tx_id)
                diag = {}
                if dec_rec:
                    d_json = dec_rec.get("diagnostics_json", {})
                    if isinstance(d_json, str):
                        try:
                            diag = json.loads(d_json)
                        except Exception:
                            diag = {}
                    elif isinstance(d_json, dict):
                        diag = d_json

                gw_dec = diag.get("gateway_decision")
                wf_status = dec_rec.get("workflow_status") if dec_rec else "APPROVED"

                if wf_status in ["RELEASED", "RELEASED_AND_EXECUTED"]:
                    decision = "APPROVE"
                    status = "RELEASED_AND_EXECUTED"
                elif wf_status in ["DECLINED_ATO_SUSPENDED", "DECLINED_BLOCK", "DENIED", "DENIED_AND_BLOCKED"]:
                    decision = "BLOCK"
                    status = wf_status
                elif gw_dec in ["BLOCK", "DECLINED"]:
                    decision = "BLOCK"
                    status = diag.get("gateway_status", "DECLINED")
                elif gw_dec in ["HOLD", "HELD"]:
                    decision = "HOLD"
                    status = diag.get("gateway_status", "HELD")
                else:
                    decision = "APPROVE"
                    status = diag.get("gateway_status", "APPROVED")

                money_moved = (decision == "APPROVE" and status in ["APPROVED", "RELEASED", "RELEASED_AND_EXECUTED"])

                w_txs.append({
                    "tx_id": tx_id,
                    "recipient": t.get("merchant_id", "UNKNOWN"),
                    "amount": float(t.get("amount", 0.0)),
                    "decision": decision,
                    "status": status,
                    "timestamp": time.strftime('%H:%M:%S', time.localtime(float(t.get("timestamp", time.time())))),
                    "money_moved": money_moved
                })

    if w_txs:
        for idx, t in enumerate(w_txs[:8]):
            c_tx1, c_tx2, c_tx3, c_tx4, c_tx5 = st.columns([2, 1.5, 1.5, 1.5, 1.5])
            c_tx1.markdown(f"<code>{t['tx_id']}</code>")
            c_tx2.write(f"To: **{t['recipient']}**")
            c_tx3.write(f"**₹{t['amount']:,.2f}**")

            if t["decision"] == "BLOCK" or t["status"] in ["DECLINED", "DENIED", "DECLINED_ATO_SUSPENDED", "DECLINED_BLOCK", "DENIED_AND_BLOCKED"]:
                c_tx4.markdown(render_icon_badge("x", "BLOCKED", "red"), unsafe_allow_html=True)
                with c_tx5:
                    st.caption("🔒 Funds Safe (₹0)")
            elif t["decision"] == "HOLD" or t["status"] in ["HELD", "HELD_PENDING_VERIFICATION"]:
                c_tx4.markdown(render_icon_badge("alert_triangle", "ON HOLD", "amber"), unsafe_allow_html=True)
                with c_tx5:
                    st.caption("Verify Above")
            else:
                c_tx4.markdown(render_icon_badge("check", "APPROVED", "green"), unsafe_allow_html=True)
                with c_tx5:
                    if st.button("Report Fraud", key=f"btn_rep_{t['tx_id']}_{idx}", width="stretch"):
                        st.session_state.reporting_tx_id = t["tx_id"]
                        st.rerun()
    else:
        st.info("No transaction history recorded yet. Use the Send Money form above to execute your first transfer.")
