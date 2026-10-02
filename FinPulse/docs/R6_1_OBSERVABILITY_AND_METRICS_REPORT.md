# FinPulse R6.1 — Observability & Metrics Technical Report

## 1. Overview & Architectural Boundaries

**FinPulse R6.1** represents the initial **Production Hardening & Operational Validation** phase. It instruments the complete, frozen fraud-decision pipeline with low-cardinality Prometheus telemetry, sub-millisecond stage-level latency histograms, structured JSON logging with correlation contexts, and operational dashboard definitions.

### Decision Pipeline Invariance Guarantee
As mandated by the frozen pipeline boundary, zero changes were made to fraud-decision mathematics:
- R4 CatBoost inference and Platt scaling calibration parameters remain untouched.
- R4 ML threshold boundaries ($p_{low} = 0.1580, p_{high} = 0.5516$) remain untouched.
- R5.1 hybrid risk weights (ML: 45%, Velocity: 15%, Behavioral: 15%, Rules: 15%, Anomaly: 10%) remain untouched.
- R5.1 decision thresholds ($\tau_{\text{review}} = 30.0, \tau_{\text{block}} = 70.0$) remain untouched.
- Multi-signal compounding policy ($K \ge 2 \implies K \times 0.12$) remains untouched.
- R5.2 diagnostic attribution calculations remain untouched.
- R5.3 business rules catalog and hard-block semantics remain untouched.
- R5.4 `DecisionEvent v1.0` schema remains untouched.
- R5.5 Kafka topic routing contracts remain untouched.

---

## 2. Telemetry Architecture

```text
                     FinPulse Transaction
                              │
                              ▼
                    ┌──────────────────┐
                    │ Correlation      │  transaction_id, customer_id,
                    │ Context          │  event_id, model_version, schema_version
                    └────────┬─────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
  Stage Histograms    Bounded Counters    Structured JSON Logs
  (Redis, Feature,    (Decisions, Rate,   (With Sensitive Credential
   Model, Publish)     Errors, Replays)    Scrubbing / Redaction)
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                 FastAPI /metrics Endpoint
                             │
                             ▼
                 Prometheus Scrape Engine (Docker)
                             │
                             ▼
                 Grafana Operational Dashboard
```

---

## 3. Prometheus Metric Registry & Semantics

All metrics are exposed under standard Prometheus text exposition format via the `/metrics` endpoint on the serving service (`finpulse-api:8000/metrics`).

### 3.1 Transaction & Decision Counters

| Metric Name | Type | Labels | Semantics |
| :--- | :--- | :--- | :--- |
| `finpulse_transactions_total` | Counter | `status` (`accepted`, `rejected`, `processed`) | Total transactions ingested into pipeline |
| `finpulse_decisions_total` | Counter | `decision` (`APPROVE`, `REVIEW`, `BLOCK`) | Completed fraud decisions by final outcome |
| `finpulse_fraud_alerts_total` | Counter | *None* | High-risk/hard-block alerts routed to Kafka `fraud-alerts` |
| `finpulse_risk_levels_total` | Counter | `risk_level` (`LOW`, `MEDIUM`, `HIGH`) | Decision distribution by categorical risk grade |
| `finpulse_hard_blocks_total` | Counter | *None* | Critical safety rule hard-block overrides |
| `finpulse_replays_total` | Counter | *None* | Duplicate transaction replays detected by idempotency layer |
| `finpulse_dlq_total` | Counter | *None* | Messages routed to Dead-Letter Queue |
| `finpulse_risk_triggers_total` | Counter | `trigger` (`ml`, `velocity`, `behavioral`, `rules`, `anomaly`) | Active multi-signal compounding triggers |
| `finpulse_matched_rules_total` | Counter | `category` (`velocity`, `amount`, `location_device`, `authentication`, `transaction_context`, `hard_block`) | Matched business rules by bounded category |

### 3.2 Error Counters

| Metric Name | Type | Labels | Semantics |
| :--- | :--- | :--- | :--- |
| `finpulse_errors_total` | Counter | `component`, `error_type` | Subsystem errors categorized by bounded labels |

**Bounded Label Sets**:
- `component`: `kafka`, `redis`, `feature_engine`, `model`, `risk_engine`, `rule_engine`, `serialization`, `publisher`, `api`
- `error_type`: `connection_error`, `timeout`, `validation_error`, `schema_error`, `internal_error`, `not_found`, `runtime_error`

### 3.3 Latency Histograms

All latencies are recorded in milliseconds (`ms`) using high-precision `time.perf_counter()` timers.

| Metric Name | Target Subsystem | Histogram Buckets (ms) |
| :--- | :--- | :--- |
| `finpulse_processing_latency_ms` | End-to-End Pipeline Latency | `(0.25, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0)` |
| `finpulse_model_latency_ms` | CatBoost + Platt + Anomaly + SHAP | `(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)` |
| `finpulse_redis_latency_ms` | Redis Sliding Window Query & Update | `(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)` |
| `finpulse_kafka_publish_latency_ms` | Kafka Delivery & Broker Acknowledgement | `(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)` |
| `finpulse_feature_engine_latency_ms` | 32-Feature Extraction Pipeline | `(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)` |
| `finpulse_risk_engine_latency_ms` | Hybrid Risk Weighted Fusion & Escalation | `(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)` |
| `finpulse_rule_engine_latency_ms` | Deterministic Business Rule Evaluation | `(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)` |
| `finpulse_serialization_latency_ms` | DecisionEvent Canonical UTF-8 JSON Serialization | `(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)` |
| `finpulse_kafka_consume_latency_ms` | Kafka Consumer Polling & Deserialization | `(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)` |

---

## 4. Cardinality Guardrails & Verification

Prometheus metric time-series explosions are strictly prevented:
- **Forbidden in Labels**: `transaction_id`, `customer_id`, `event_id`, timestamps, monetary amounts, raw rule texts, stack traces, free-form error messages.
- **Allowed Labels**: Strictly bounded sets with cardinality $\le 9$ elements (`decision` $\le 3$, `risk_level` $\le 3$, `status` $\le 4$, `component` $\le 9$, `error_type` $\le 7$, `trigger` $\le 5$, `category` $\le 6$).
- **Cardinality Audit**: Automated test `test_prometheus_label_cardinality_guardrails` programmatically inspects all metric families in the registry and guarantees zero dynamic IDs or unbounded strings leak into Prometheus labels.

---

## 5. Transaction Correlation & Sensitive Data Safety

### 5.1 Context Propagation
The [`CorrelationContext`](file:///d:/Fraud-Detection-System/FinPulse/src/monitoring/correlation.py) leverages Python `contextvars` to maintain execution tracing without passing tracing objects across internal interfaces:
- `transaction_id`: Business idempotency key
- `customer_id`: Entity identifier
- `event_id`: Deterministic event UUID
- `model_version`: Authoritative ML release (`finpulse-v3`)
- `feature_schema_version`: Authoritative feature contract (`2.0`)

### 5.2 Sensitive Data Redaction
The [`sanitize_log_payload`](file:///d:/Fraud-Detection-System/FinPulse/src/monitoring/logger.py) engine recursively scrubs:
- `password`, `passwd`, `secret`, `token`, `api_key`, `auth_token`, `authorization`
- `card_number`, `pan`, `cvv`, `cvc`, `pin`, `ssn`, `credit_card`
- Values are replaced with `[REDACTED]` prior to JSON serialization.

---

## 6. Live Prometheus & Docker Integration Validation

Live operational validation was executed using the official Prometheus Docker image (`prom/prometheus:v2.51.0`) scraping the running FastAPI microservice:

```text
[1/5] Launching FinPulse FastAPI serving server on port 8000...
   [OK] FastAPI server ready with model loaded.

[2/5] Starting Prometheus Docker container 'finpulse-prometheus-val'...
   [OK] Prometheus container running.
   [OK] Prometheus ready endpoint returned 200 OK.

[3/5] Generating transactions to populate telemetry...
   Processed tx_val_01: decision=APPROVE, risk_score=8.9
   Processed tx_val_02: decision=APPROVE, risk_score=12.9
   Processed tx_val_03: decision=APPROVE, risk_score=24.3

[4/5] Waiting for Prometheus scrape cycles (3 seconds)...
   Active scrape targets found: 1
   Target 'finpulse-api': health='up', last_error=''

[5/5] Querying Prometheus API for ingested FinPulse metrics...
   finpulse_transactions_total series count: 1
   Total transactions recorded: 3.0
   finpulse_decisions_total breakdown: {'APPROVE': 3.0}
   finpulse_processing_latency_ms count: 3.0

======================================================================
[SUCCESS] PROMETHEUS LIVE INTEGRATION VALIDATED END-TO-END!
======================================================================
```

---

## 7. Replay & Dead-Letter Queue (DLQ) State

- **Replay Tracking**: Implemented via in-memory transaction deduplication in `StreamingFraudWorker`. When an incoming `transaction_id` matches a previously processed transaction, `finpulse_replays_total` increments while preserving deterministic `event_id` and routing.
- **DLQ State**: Configured in `configs/streaming.yaml` (`dead_letter: "transactions-dlq"`). `finpulse_dlq_total` is registered and wired to `StreamingFraudWorker.route_to_dlq()`. Note: In the baseline streaming loop, DLQ routing is invoked upon unrecoverable message parsing or processing failures.

---

## 8. Test Execution & Regression Evidence

- **Dedicated R6.1 Suite**: `tests/test_observability_r6_1.py` (14/14 passed)
- **Full Project Regression Suite**: **239 / 239 passed** (100% green across R1 through R6.1)
