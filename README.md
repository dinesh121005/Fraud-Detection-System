# FinPulse AI — Complete Fraud Intelligence & Risk Decisioning Platform

**FinPulse** is an institutional-grade, real-time streaming fraud detection platform engineering sub-10ms risk classification, hybrid machine learning fusion, customer protection workflows, and complete model lifecycle management.

---

## 1. System Architecture Overview

```text
Incoming Transaction / Event
           │
           ▼
    ┌──────────────┐
    │ Kafka Ingress│ (Topic: 'transactions' @ 9092)
    └──────┬───────┘
           │
           ▼
┌──────────────────────┐
│ Streaming Worker     │ ◄───► ┌──────────────────────┐
│ (Idempotent Consumer)│       │ Redis Rolling Window │ (1m, 5m, 1h Velocity)
└──────────┬───────────┘       └──────────────────────┘
           │
           ▼
┌──────────────────────┐
│ 32-Feature Pipeline  │ (Frozen Schema, Zero-Leakage)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ ML Scoring & Calib   │ ───► Shadow Challenger Model (R7-H Non-Blocking)
│ (CatBoost finpulse-v3│
│  + Platt Calibrator) │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Hybrid Risk Engine   │ (Weights: ML 45%, Velocity 15%, Behavioral 15%,
│  & Business Rules    │           Rules 15%, Anomaly 10% + Escalation)
└──────────┬───────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│ Decision & Workflow Router                              │
├───────────────┬─────────────────────────┬───────────────┤
│    APPROVE    │          HOLD           │     BLOCK     │
│  (Risk < 30)  │     (30 <= Risk < 70)   │  (Risk >= 70  │
│               │                         │  or Hard-Blk) │
└───────┬───────┴────────────┬────────────┴───────┬───────┘
        │                    │                    │
        │             Customer Review             │
        │         (Confirm / Deny / Expire)       │
        │                    │                    │
        ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────┐
│ Kafka Multi-Topic Publisher (Guaranteed At-Least-Once)  │
│  - 'predictions'   (All canonical decisions)            │
│  - 'fraud-alerts'  (Critical threats & hard-blocks)     │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ PostgreSQL Relational Persistence Sink (Primary Runtime)│
│  - Connection Pool (ThreadedConnectionPool)             │
│  - ACID Transactions, Write Idempotency (ON CONFLICT)   │
│  - 7 Tables: Transactions, Decisions, Holds, Alerts,     │
│    ATO Events, Mandates, Replay/Evaluation Records      │
│  *(SQLite retained strictly for isolated unit tests)*   │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ Executive Operational Dashboard (Streamlit :8501)       │
│  [Live]  [Analyst]  [Phone]  [Results]  [Health]        │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Core Functional Subsystems

### R1–R3: Feature Engineering & State Store
- **32-Feature Contract:** Strictly ordered, leak-free feature vector spanning velocity (1m, 5m, 15m, 1h), behavioral z-scores, merchant historical risk, and device/spatial discrepancies.
- **Redis Rolling Windows:** Low-latency sorted sets and hash sets tracking real-time customer and device activity under strict `t < current_timestamp` read-before-write isolation.

### R4: Frozen Machine Learning Engine
- **Model Candidate:** Calibrated gradient-boosted decision tree (`finpulse-v3`).
- **Platt Scaling:** Sigmoid probability calibration mapping raw margin outputs to true statistical fraud posterior $P(\text{Fraud} \mid X)$.

### R5: Hybrid Risk Engine & Business Rules
- **Weighted Fusion:** Linear aggregation using engineering baseline weights:
  $$\text{Score} = 100 \times (0.45 \cdot P_{\text{ML}} + 0.15 \cdot S_{\text{vel}} + 0.15 \cdot S_{\text{beh}} + 0.15 \cdot S_{\text{rule}} + 0.10 \cdot S_{\text{anom}})$$
- **Synergistic Compounding:** Multi-trigger synergy boost: $K \ge 2 \implies K \times 12.0$ points added to risk score.
- **Hard-Block Overrides:** Safety overrides (e.g. impossible travel, zero-balance drain without authentication) forcing `BLOCK` with minimum score $\ge 85.0$.
- **DecisionEvent v1.0:** Canonical, deterministic UTF-8 JSON schema with UUIDv5 idempotency key.

### R6: Observability, Capacity & Hardening
- **Prometheus Telemetry:** Bounded label cardinality, real-time stage latencies (`redis`, `feature_engine`, `model`, `risk_engine`, `e2e`).
- **Capacity:** Benchmarked at 5,120 msg/sec with sub-10ms P99 scoring latency.
- **Resilience:** Offset commits strictly guarded by publication ACK; automated Dead-Letter Queue (DLQ) isolation; safe artifact corruption failure.

### R7: Product Workflows & Lifecycle
- **R7-A Account Security & ATO:** Ingests credential/device lifecycle events (`password_change`, `new_device_registration`, `mfa_reset`); compounds multi-event takeover patterns to hold or suspend accounts.
- **R7-B Mandate Protection:** Manages standing orders and recurring billing; enforces frequency limits and maximum debit ceilings.
- **R7-C HOLD State Machine:** Orchestrates two-way customer confirmations:
  $$\text{HOLD} \xrightarrow{\text{Confirm}} \text{RELEASE} \quad \Big\vert \quad \text{HOLD} \xrightarrow{\text{Deny}} \text{DENIED} \quad \Big\vert \quad \text{HOLD} \xrightarrow{\text{Timeout}} \text{EXPIRED}$$
- **R7-D Notifier & Expiry Worker:** Non-blocking notification dispatch and background thread for expiring timed-out review requests.
- **R7-E Replay & Delayed Labels:** Realistic traffic replayer supporting delayed chargeback maturation (e.g. 14–30 day reporting lag).
- **R7-F Relational Sink:** ACID-compliant idempotent event persistence with primary key de-duplication on `event_id`.
- **R7-G 5-View Dashboard:** `Live` (streaming ledger), `Analyst` (SHAP forensics), `Phone` (customer review UI), `Results` (E1–E10 benchmark view), `Health` (infrastructure probes).
- **R7-H Shadow Scoring:** Parallel evaluation of candidate models without altering production decisions.
- **R7-I Retraining & Promotion Gate:** Gated candidate promotion checking PR-AUC, ROC-AUC, Brier score, and FPR floors.
- **R7-J PSI Drift Monitoring:** Population Stability Index monitoring across all 32 features and model score distributions.
- **R7-K E1–E10 Evaluation Suite:** 10 core empirical benchmarks verifying reproducibility, concurrency, policy, and capacity.

---

## 3. Quickstart & Deployment

### Prerequisites
- Python 3.12+
- Docker Engine & Docker Compose

### 1. Launch Streaming & Relational Infrastructure
```bash
docker compose up -d
```
Starts Kafka (`localhost:9092`), Redis (`localhost:6379`), PostgreSQL 16 (`localhost:5432`), and Prometheus (`localhost:9090`).

### 2. Run Comprehensive Test Suite
```bash
python -m pytest -v
```
Executes all **314 automated unit, integration, persistence, and contract tests** with 100% green pass rate (0 failures, 0 skipped).

### 3. Run PostgreSQL Migration & Golden Scenario Validation
```bash
python FinPulse/scripts/validate_postgres_migration.py
```
Validates connectivity, 7 PostgreSQL tables, ACID rollback, container restart durability, idempotency, and the three golden scenarios (`APPROVE`, `HOLD -> RELEASE`, `HARD BLOCK`).

### 4. Run E1–E10 Full System Evaluation
```bash
python FinPulse/scripts/run_r7_evaluation.py
```
Outputs structured benchmark evidence and generates `reports/r7_evaluation_report.json`.

### 5. Run Model Retraining & Promotion Gate
```bash
python FinPulse/scripts/retrain.py --candidate-version finpulse-v4-candidate
```

### 5. Launch Operational Dashboard
```bash
streamlit run FinPulse/app.py
```
Opens the 5-view FinPulse Operations UI at `http://localhost:8501`.

---

## 4. Operational Dashboard Views

| View Name | Primary Function | Key Features |
|---|---|---|
| **🔴 Live** | Streaming Operations | Real-time transaction simulation, stream generator, KPI metric cards, and verdict distribution. |
| **🕵️ Analyst** | Forensic Case Investigation | SHAP explanation waterfall, triggered business rules, and 32-feature vector inspection. |
| **📱 Phone** | Customer Review Simulation | Interactive two-way authorization screen simulating customer mobile confirmation on `HOLD` decisions. |
| **📈 Results** | Governance & Benchmarks | Verified E1–E10 evaluation results, PR-AUC / ROC-AUC metrics, and cost-frontier optimization curve. |
| **🛡️ Health** | Telemetry & Probes | Live status for Kafka 9092, Redis 6379, DB Sink, Model Artifacts, and Population Stability Index (PSI). |

---

## 5. Verified Experimental Benchmarks (E1–E10)

| ID | Benchmark Name | Measured Metric / Result | Status |
|:---:|---|---|:---:|
| **E1** | Parity & Determinism | Exact scoring, calibration & attribution parity on repeated runs | **PASSED** |
| **E2** | Concurrency & Idempotency | Duplicate writes de-duplicated; 100% record integrity | **PASSED** |
| **E3** | Policy & Decision Consistency | Zero-balance drain triggers hard-block; risk score $\ge 85.0$ | **PASSED** |
| **E4** | HOLD Lifecycle | Complete state machine verified across RELEASE and EXPIRE branches | **PASSED** |
| **E5** | Replay & Delayed Labels | 20/20 transactions replayed; 20/20 delayed chargebacks matured | **PASSED** |
| **E6** | Cold-Start Resilience | Clean fallback to baseline priors for unseen customer profiles | **PASSED** |
| **E7** | ATO Compounding Detection | Password change + new device triggers ATO review ($S \ge 0.50$) | **PASSED** |
| **E8** | Load & Capacity Saturation | In-memory evaluation verified under sustained execution | **PASSED** |
| **E9** | Shadow / Challenger Scoring | Challenger scores in parallel with zero impact on production decisions | **PASSED** |
| **E10** | PSI Drift & Promotion Gate | PSI flags shifted distributions ($\text{PSI} = 4.33$); promotion gate verified | **PASSED** |

---

## 6. Known Limitations & Production Readiness Boundary

1. **Synthetic & Benchmark Workloads:** Benchmarks use synthetic and PaySim/IEEE distributions; live Visa/Mastercard ISO 8583 message feeds require dedicated clearinghouse network adapters.
2. **External Authentication:** Development and local Docker configurations use `PLAINTEXT` Kafka and unauthenticated Redis; production deployments must configure SASL_SSL (SCRAM-SHA-512) and Redis TLS.
3. **Regulatory Audits:** The platform is engineered to institutional software architecture standards but has not undergone formal external audits for PCI-DSS Level 1 or SOC 2 Type II certification.