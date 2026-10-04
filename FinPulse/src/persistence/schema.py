"""
FinPulse Schema DDL & Migration Definitions.

Contains schema definitions for PostgreSQL (production runtime) and SQLite (test fallback).
Enforces strict primary key and uniqueness constraints for guaranteed write idempotency.
"""

# =============================================================================
# PostgreSQL DDL Schema
# =============================================================================
POSTGRES_SCHEMA_SQL = """
-- 1. Fraud Decisions & Canonical DecisionEvents
CREATE TABLE IF NOT EXISTS fraud_decisions (
    event_id VARCHAR(64) PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL UNIQUE,
    customer_id VARCHAR(64) NOT NULL,
    decision VARCHAR(16) NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL,
    risk_level VARCHAR(16) NOT NULL,
    ml_decision VARCHAR(16) NOT NULL,
    calibrated_probability DOUBLE PRECISION NOT NULL,
    model_version VARCHAR(32) NOT NULL,
    schema_version VARCHAR(16) NOT NULL,
    signals_json JSONB NOT NULL,
    diagnostics_json JSONB NOT NULL,
    reasons_json JSONB NOT NULL,
    matched_rules_json JSONB NOT NULL,
    hard_block BOOLEAN NOT NULL,
    latency_ms DOUBLE PRECISION,
    workflow_status VARCHAR(32) DEFAULT 'FINAL',
    fraud_label INTEGER DEFAULT NULL,
    label_timestamp DOUBLE PRECISION DEFAULT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decisions_cust ON fraud_decisions(customer_id);
CREATE INDEX IF NOT EXISTS idx_decisions_created ON fraud_decisions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_label ON fraud_decisions(fraud_label);
CREATE INDEX IF NOT EXISTS idx_decisions_status ON fraud_decisions(workflow_status);

-- 2. Raw Ingested Transactions
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(64) PRIMARY KEY,
    customer_id VARCHAR(64) NOT NULL,
    merchant_id VARCHAR(64) NOT NULL,
    amount DOUBLE PRECISION NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL,
    payment_type VARCHAR(32) DEFAULT 'TRANSFER',
    category VARCHAR(32) DEFAULT 'general',
    origin_balance DOUBLE PRECISION DEFAULT 0.0,
    dest_balance DOUBLE PRECISION DEFAULT 0.0,
    auth_verified BOOLEAN DEFAULT TRUE,
    device_id VARCHAR(64) DEFAULT 'unknown_device',
    location_json JSONB DEFAULT '{}'::jsonb,
    created_at DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tx_cust ON transactions(customer_id, timestamp DESC);

-- 3. HOLD Cases & Review Workflow Lifecycle
CREATE TABLE IF NOT EXISTS hold_cases (
    hold_id VARCHAR(64) PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL UNIQUE,
    customer_id VARCHAR(64) NOT NULL,
    amount DOUBLE PRECISION NOT NULL,
    status VARCHAR(16) NOT NULL,
    token_hash VARCHAR(64) NOT NULL,
    timeout_seconds DOUBLE PRECISION NOT NULL DEFAULT 300.0,
    created_at DOUBLE PRECISION NOT NULL,
    expires_at DOUBLE PRECISION NOT NULL,
    resolved_at DOUBLE PRECISION DEFAULT NULL,
    resolution_reason TEXT DEFAULT NULL,
    customer_channel VARCHAR(32) DEFAULT 'sms_push',
    metadata_json JSONB DEFAULT '{}'::jsonb,
    updated_at DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hold_cust ON hold_cases(customer_id);
CREATE INDEX IF NOT EXISTS idx_hold_status ON hold_cases(status);
CREATE INDEX IF NOT EXISTS idx_hold_expires ON hold_cases(expires_at);

-- 4. Fraud Alerts
CREATE TABLE IF NOT EXISTS fraud_alerts (
    alert_id VARCHAR(64) PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL,
    event_id VARCHAR(64),
    customer_id VARCHAR(64) NOT NULL,
    severity VARCHAR(16) NOT NULL,
    decision VARCHAR(16) NOT NULL,
    reasons_json JSONB NOT NULL,
    notified BOOLEAN DEFAULT FALSE,
    status VARCHAR(32) DEFAULT 'OPEN',
    created_at DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alerts_tx ON fraud_alerts(transaction_id);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON fraud_alerts(severity);

-- 5. Account Security Events & ATO Tracking
CREATE TABLE IF NOT EXISTS account_security_events (
    event_id VARCHAR(64) PRIMARY KEY,
    customer_id VARCHAR(64) NOT NULL,
    event_type VARCHAR(64) NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL,
    ip_address VARCHAR(45) DEFAULT '127.0.0.1',
    device_id VARCHAR(64) DEFAULT 'dev_unknown',
    metadata_json JSONB DEFAULT '{}'::jsonb,
    created_at DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_acc_ev_cust ON account_security_events(customer_id, timestamp DESC);

-- 6. Recurring Mandates
CREATE TABLE IF NOT EXISTS mandates (
    mandate_id VARCHAR(64) PRIMARY KEY,
    customer_id VARCHAR(64) NOT NULL,
    beneficiary_id VARCHAR(64) NOT NULL,
    max_amount DOUBLE PRECISION NOT NULL,
    frequency VARCHAR(32) NOT NULL,
    status VARCHAR(16) NOT NULL,
    execution_count INTEGER DEFAULT 0,
    total_executed_amount DOUBLE PRECISION DEFAULT 0.0,
    last_executed_at DOUBLE PRECISION DEFAULT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mandates_cust ON mandates(customer_id);

-- 7. Replay & Evaluation Audit Records
CREATE TABLE IF NOT EXISTS replay_evaluation_records (
    replay_id VARCHAR(64) PRIMARY KEY,
    scenario_type VARCHAR(32) NOT NULL,
    transactions_count INTEGER NOT NULL,
    matured_labels_count INTEGER DEFAULT 0,
    started_at DOUBLE PRECISION NOT NULL,
    completed_at DOUBLE PRECISION DEFAULT NULL,
    metrics_json JSONB DEFAULT '{}'::jsonb,
    created_at DOUBLE PRECISION NOT NULL
);

-- 8. Gateway Security Cases & Feedback Loop (D4, D5)
CREATE TABLE IF NOT EXISTS security_cases (
    case_id VARCHAR(64) PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL,
    customer_id VARCHAR(64) NOT NULL,
    gateway_decision VARCHAR(16) NOT NULL,
    customer_report VARCHAR(32) NOT NULL,
    confirmed_label VARCHAR(16) NOT NULL,
    attack_type VARCHAR(64) NOT NULL,
    status VARCHAR(32) DEFAULT 'CONFIRMED_FRAUD',
    metadata_json JSONB DEFAULT '{}'::jsonb,
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cases_cust ON security_cases(customer_id);
CREATE INDEX IF NOT EXISTS idx_cases_tx ON security_cases(transaction_id);
CREATE INDEX IF NOT EXISTS idx_cases_type ON security_cases(attack_type);
"""

# =============================================================================
# SQLite DDL Schema (Test Fallback)
# =============================================================================
SQLITE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS fraud_decisions (
    event_id VARCHAR(64) PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL UNIQUE,
    customer_id VARCHAR(64) NOT NULL,
    decision VARCHAR(16) NOT NULL,
    risk_score REAL NOT NULL,
    risk_level VARCHAR(16) NOT NULL,
    ml_decision VARCHAR(16) NOT NULL,
    calibrated_probability REAL NOT NULL,
    model_version VARCHAR(32) NOT NULL,
    schema_version VARCHAR(16) NOT NULL,
    signals_json TEXT NOT NULL,
    diagnostics_json TEXT NOT NULL,
    reasons_json TEXT NOT NULL,
    matched_rules_json TEXT NOT NULL,
    hard_block INTEGER NOT NULL,
    latency_ms REAL,
    workflow_status VARCHAR(32) DEFAULT 'FINAL',
    fraud_label INTEGER DEFAULT NULL,
    label_timestamp REAL DEFAULT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decisions_cust ON fraud_decisions(customer_id);
CREATE INDEX IF NOT EXISTS idx_decisions_created ON fraud_decisions(created_at);
CREATE INDEX IF NOT EXISTS idx_decisions_label ON fraud_decisions(fraud_label);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(64) PRIMARY KEY,
    customer_id VARCHAR(64) NOT NULL,
    merchant_id VARCHAR(64) NOT NULL,
    amount REAL NOT NULL,
    timestamp REAL NOT NULL,
    payment_type VARCHAR(32) DEFAULT 'TRANSFER',
    category VARCHAR(32) DEFAULT 'general',
    origin_balance REAL DEFAULT 0.0,
    dest_balance REAL DEFAULT 0.0,
    auth_verified INTEGER DEFAULT 1,
    device_id VARCHAR(64) DEFAULT 'unknown_device',
    location_json TEXT DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tx_cust ON transactions(customer_id, timestamp);

CREATE TABLE IF NOT EXISTS hold_cases (
    hold_id VARCHAR(64) PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL UNIQUE,
    customer_id VARCHAR(64) NOT NULL,
    amount REAL NOT NULL,
    status VARCHAR(16) NOT NULL,
    token_hash VARCHAR(64) NOT NULL,
    timeout_seconds REAL NOT NULL DEFAULT 300.0,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    resolved_at REAL DEFAULT NULL,
    resolution_reason TEXT DEFAULT NULL,
    customer_channel VARCHAR(32) DEFAULT 'sms_push',
    metadata_json TEXT DEFAULT '{}',
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hold_cust ON hold_cases(customer_id);
CREATE INDEX IF NOT EXISTS idx_hold_status ON hold_cases(status);

CREATE TABLE IF NOT EXISTS fraud_alerts (
    alert_id VARCHAR(64) PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL,
    event_id VARCHAR(64),
    customer_id VARCHAR(64) NOT NULL,
    severity VARCHAR(16) NOT NULL,
    decision VARCHAR(16) NOT NULL,
    reasons_json TEXT NOT NULL,
    notified INTEGER DEFAULT 0,
    status VARCHAR(32) DEFAULT 'OPEN',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS account_security_events (
    event_id VARCHAR(64) PRIMARY KEY,
    customer_id VARCHAR(64) NOT NULL,
    event_type VARCHAR(64) NOT NULL,
    timestamp REAL NOT NULL,
    ip_address VARCHAR(45) DEFAULT '127.0.0.1',
    device_id VARCHAR(64) DEFAULT 'dev_unknown',
    metadata_json TEXT DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS mandates (
    mandate_id VARCHAR(64) PRIMARY KEY,
    customer_id VARCHAR(64) NOT NULL,
    beneficiary_id VARCHAR(64) NOT NULL,
    max_amount REAL NOT NULL,
    frequency VARCHAR(32) NOT NULL,
    status VARCHAR(16) NOT NULL,
    execution_count INTEGER DEFAULT 0,
    total_executed_amount REAL DEFAULT 0.0,
    last_executed_at REAL DEFAULT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS replay_evaluation_records (
    replay_id VARCHAR(64) PRIMARY KEY,
    scenario_type VARCHAR(32) NOT NULL,
    transactions_count INTEGER NOT NULL,
    matured_labels_count INTEGER DEFAULT 0,
    started_at REAL NOT NULL,
    completed_at REAL DEFAULT NULL,
    metrics_json TEXT DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS security_cases (
    case_id VARCHAR(64) PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL,
    customer_id VARCHAR(64) NOT NULL,
    gateway_decision VARCHAR(16) NOT NULL,
    customer_report VARCHAR(32) NOT NULL,
    confirmed_label VARCHAR(16) NOT NULL,
    attack_type VARCHAR(64) NOT NULL,
    status VARCHAR(32) DEFAULT 'CONFIRMED_FRAUD',
    metadata_json TEXT DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cases_cust ON security_cases(customer_id);
CREATE INDEX IF NOT EXISTS idx_cases_tx ON security_cases(transaction_id);
CREATE INDEX IF NOT EXISTS idx_cases_type ON security_cases(attack_type);
"""
