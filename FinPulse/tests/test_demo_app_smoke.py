"""
Smoke tests for FinPulse Streamlit App and Demo Hardening components.
"""

import os
import sys
import pytest

FINPULSE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.simulation.demo_account import DemoAccountManager, DEFAULT_DEMO_CUSTOMER_ID
from src.workflow.account_events import ATOProtectionEngine
from src.workflow.hold_workflow import HoldWorkflowEngine
from src.persistence.sink import IdempotentEventSink
from src.simulation.gateway import PaymentGatewaySimulator
from src.simulation.attacker import AttackerSimulator


def test_demo_reset_cleanliness():
    acc_mgr = DemoAccountManager()
    ato_eng = ATOProtectionEngine()
    hold_eng = HoldWorkflowEngine()
    sink = IdempotentEventSink(db_path=":memory:")

    # Create active hold case before reset to verify hold_eng.reset
    hold_eng.create_hold("tx_smoke_1", DEFAULT_DEMO_CUSTOMER_ID, 8500.0)
    hold_eng.create_hold("tx_smoke_2", "CUST_OTHER", 5000.0)

    # Trigger reset for demo customer
    acc_mgr.reset_all(DEFAULT_DEMO_CUSTOMER_ID)
    ato_eng.reset(DEFAULT_DEMO_CUSTOMER_ID)
    hold_eng.reset(DEFAULT_DEMO_CUSTOMER_ID)
    sink.reset_demo_data(DEFAULT_DEMO_CUSTOMER_ID)

    assert hold_eng.get_case_by_transaction("tx_smoke_1") is None
    assert hold_eng.get_case_by_transaction("tx_smoke_2") is not None

    # Trigger global reset
    hold_eng.reset()
    assert hold_eng.get_case_by_transaction("tx_smoke_2") is None


    acc = acc_mgr.get_account(DEFAULT_DEMO_CUSTOMER_ID)
    assert acc.balance == 10000.0
    assert acc.account_status == "SAFE"
    assert acc.known_device_id == "DEV_KNOWN_01"
    assert acc.known_city == "CHENNAI"
    assert len(ato_eng.get_recent_events(DEFAULT_DEMO_CUSTOMER_ID, 9999999999.0)) == 0


def test_app_py_compilation():
    app_path = os.path.join(FINPULSE_DIR, "app.py")
    with open(app_path, "r", encoding="utf-8") as f:
        code = f.read()
    compiled = compile(code, app_path, "exec")
    assert compiled is not None
