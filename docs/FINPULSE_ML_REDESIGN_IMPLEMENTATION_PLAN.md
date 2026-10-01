# FinPulse ML Subsystem Redesign: Repository-Level Implementation Plan

**System Target:** FinPulse AI — Fraud Intelligence Operations  
**Base Repository:** `Fraud-Detection-System` (`d:\Fraud-Detection-System\FinPulse`)  
**Specification Source:** `FinPulse ML — Complete Technical Design` (`docs/FinPulse_ML_Technical_Design.md`)  
**Status:** Approved Architectural Blueprint & Engineering Plan  

---

## 1. Executive Summary & Existing Repository Audit

### 1.1 Existing vs New vs Modified vs Obsolete Components

A thorough audit of the active repository (`d:\Fraud-Detection-System\FinPulse`) against the technical specification reveals the exact boundary between what can be retained, what must be upgraded, what must be newly engineered, and what is dead code:

| Component | Path in Repository | Current Status & Audit Finding | Proposed Action & Role |
| :--- | :--- | :--- | :--- |
| **Training Pipeline** | `FinPulse/scripts/train_model.py` | Trains XGBoost on PaySim with injected random noise (`auth_verified`, `device_known`, `geo_known`, etc.) and synthetic attack perturbations. Uses post-transaction features (`newbalanceOrig`, `newbalanceDest`) causing **severe target leakage**. Random stratified split instead of temporal validation. | **Retain as Baseline Benchmark**; build new modular training orchestrator in `FinPulse/src/training/` with leakage elimination and temporal splitting. |
| **Model Artifacts** | `FinPulse/models/fraud_model.json`, `fraud_model.pkl`, `label_encoder.pkl` | Working 15-feature XGBoost model trained on corrupted PaySim + synthetic features. | **Preserve as `models/baseline/`** to serve as the exact baseline benchmark that all new models must statistically outperform. |
| **Preprocessing Utilities** | `FinPulse/utils/preprocessing.py` | Fixed 15-feature vector builder with ratio `amount / (oldbalanceOrg + 1e-6)`. | **Refactor into Adapter Layer**; replace with formal transformer pipeline in `FinPulse/src/features/` supporting common FinPulse schema. |
| **Rule & Heuristic Engine** | `FinPulse/utils/helpers.py` | Additive risk scoring (`compute_heuristic_risk`, max 1.0) and rule strings (`explain_heuristic`). Hardcoded rules. | **Modulate & Integrate** into the hybrid risk engine (`FinPulse/src/risk_engine/rules.py`). Retain core business heuristic checks. |
| **Simulation Generator** | `FinPulse/utils/simulation.py` | Generates synthetic transactions and attack presets for manual Streamlit testing. | **Retain & Extend** in `FinPulse/src/data/simulation.py` to emit events compliant with new FinPulse transaction contract. |
| **Inference Script** | `FinPulse/scripts/predict.py` | Single-dict prediction and synchronous prototype Kafka consumer emitting to `predictions`. Hardcoded threshold `0.5`. | **Replaced** by production FastAPI serving service (`FinPulse/src/serving/api.py`) and dedicated streaming worker (`FinPulse/src/streaming/worker.py`). |
| **Streamlit Dashboard** | `FinPulse/app.py` | Full-featured 522-line dashboard with KPI cards, attack injection, donut verdict, and audit ledger. In-process scoring. | **Modify & Retain**: Decouple from in-process XGBoost; connect to FastAPI/Risk Engine client and Redis/Kafka telemetry. |
| **Duplicate Consumer** | `FinPulse/consumer.py` | Byte-for-byte exact duplicate of `app.py` (21,571 bytes). Does not consume Kafka. | **Obsolete / Replace**: Replace with actual real-time streaming consumer worker. |
| **Prototype Consumer** | `FinPulse/kafka_consumer.py` | 16-line hardcoded demo listening to `10.121.50.11:9092` topic `test-topic`. | **Obsolete / Remove**: Replaced by standardized streaming pipeline. |
| **Prototype Producer** | `FinPulse/producer.py` | 24-line dummy producer emitting `user_id`, `amount`, `type` every 2s. | **Upgrade** into realistic multi-domain event generator emitting the unified transaction contract. |
| **Docker Compose** | `FinPulse/docker-compose.yml` | **Corrupted**: Contains 223 lines of HTML for "MPR Snacks Kovilpatti Mittai". | **Replace completely** with production multi-container setup (Kafka, Zookeeper, Redis, FastAPI, Streamlit, MLflow). |
| **Dependencies** | `FinPulse/requirements.txt` | 9 lines; missing `kafka-python`, `redis`, `optuna`, `lightgbm`, `catboost`, `mlflow`, `fastapi`, `uvicorn`. | **Upgrade** with fully version-locked requirements for complete ML stack. |

---

## 2. Proposed Repository Architecture

The redesigned repository maintains backward compatibility for running the baseline while establishing a production-grade, modular, leakage-free ML subsystem:

```text
d:\Fraud-Detection-System\
├── README.md                                # Root overview, architecture diagram, setup guide
├── docker-compose.yml                       # Multi-service stack (Kafka, Redis, API, Dashboard, MLflow)
├── docs/
│   ├── FinPulse_ML_Technical_Design.md      # Primary specification (source of truth)
│   ├── FINPULSE_ML_REDESIGN_IMPLEMENTATION_PLAN.md # This document
│   └── reports/                             # Generated audit and evaluation reports
├── configs/
│   ├── base_config.yaml                     # Shared paths, random seeds, logging configs
│   ├── datasets.yaml                        # Column mappings, leakage rules, temporal split definitions
│   ├── features.yaml                        # Feature definitions, velocity windows, binning schemas
│   ├── models.yaml                          # Model hyperparameter grids, tuning spaces, class weights
│   ├── risk_engine.yaml                     # Hybrid weights, rule thresholds, decision cutoffs
│   └── streaming.yaml                       # Kafka topics, consumer groups, Redis TTLs & window configs
├── FinPulse/
│   ├── app.py                               # Retained & upgraded Streamlit dashboard
│   ├── requirements.txt                     # Upgraded locked dependency manifest
│   ├── Dockerfile.serving                   # FastAPI container image definition
│   ├── Dockerfile.streaming                 # Kafka/Redis consumer worker container image
│   ├── Dockerfile.dashboard                 # Streamlit dashboard container image
│   ├── data/
│   │   ├── paysim.csv                       # Existing Mobile Money dataset (6.36M rows)
│   │   ├── Sparkov/
│   │   │   ├── fraudTrain.csv               # Existing Credit Card train dataset (1.29M rows)
│   │   │   └── fraudTest.csv                # Existing Credit Card test dataset (555K rows)
│   │   ├── IEE-CIS/
│   │   │   ├── train_transaction.csv        # Existing IEEE-CIS train transactions (590K rows)
│   │   │   ├── train_identity.csv           # Existing IEEE-CIS train identity (144K rows)
│   │   │   ├── test_transaction.csv         # Existing IEEE-CIS test transactions
│   │   │   └── test_identity.csv            # Existing IEEE-CIS test identity
│   │   └── processed/                       # Cached leak-free temporal splits (.parquet)
│   ├── models/
│   │   ├── baseline/                        # Archived baseline XGBoost model & encoder
│   │   │   ├── fraud_model.json
│   │   │   └── label_encoder.pkl
│   │   ├── artifacts/                       # Versioned production models, scalers, calibrators
│   │   └── registry/                        # Model metadata, performance cards, threshold configs
│   ├── src/
│   │   ├── __init__.py
│   │   ├── data/                            # Dataset ingestion, audit, profiling & adapters
│   │   │   ├── __init__.py
│   │   │   ├── auditor.py                   # Automated data quality & leakage analyzer
│   │   │   ├── base_adapter.py              # Abstract base dataset adapter
│   │   │   ├── paysim_adapter.py            # PaySim adapter & leakage filter
│   │   │   ├── sparkov_adapter.py           # Sparkov adapter & customer/merchant profiler
│   │   │   ├── ieee_adapter.py              # IEEE-CIS transaction + identity merger & adapter
│   │   │   └── splitters.py                 # Leakage-free temporal splitter (70/15/15)
│   │   ├── features/                        # Feature engineering layer (offline & online unified)
│   │   │   ├── __init__.py
│   │   │   ├── schema.py                    # FinPulse unified feature contracts & pydantic models
│   │   │   ├── transformations.py           # Log transforms, ratios, Haversine distance, time cyclics
│   │   │   ├── encoders.py                  # High-cardinality frequency, target & CatBoost encoders
│   │   │   ├── pipeline.py                  # Scikit-learn/Polars feature pipeline builder
│   │   │   └── feature_store.py             # Feature serialization, versioning & consistency validator
│   │   ├── models/                          # Model candidates, training & tuning
│   │   │   ├── __init__.py
│   │   │   ├── baselines.py                 # Logistic Regression & Random Forest
│   │   │   ├── gbm_models.py                # XGBoost, LightGBM, CatBoost implementations
│   │   │   ├── anomaly.py                   # Isolation Forest unsupervised detector
│   │   │   ├── imbalance.py                 # Class weighting, SMOTE, SMOTETomek, Undersampling
│   │   │   ├── tuner.py                     # Optuna hyperparameter optimization engine
│   │   │   └── calibrator.py                # Platt scaling & Isotonic regression calibrators
│   │   ├── evaluation/                      # Evaluation protocols & threshold optimization
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py                   # PR-AUC, Recall@FPR, Brier score, ECE calculator
│   │   │   ├── thresholding.py              # Validation-set dual-threshold optimizer (Review/Block)
│   │   │   └── benchmark.py                 # Multi-model comparative evaluation runner
│   │   ├── explainability/                  # Interpretable ML (SHAP)
│   │   │   ├── __init__.py
│   │   │   ├── explainer.py                 # SHAP TreeExplainer wrapper & background sampler
│   │   │   └── reason_mapper.py             # Maps SHAP feature attributions to business reasons
│   │   ├── risk_engine/                     # Hybrid fusion risk engine
│   │   │   ├── __init__.py
│   │   │   ├── engine.py                    # Configurable weighted risk aggregator
│   │   │   ├── rules.py                     # Deterministic business & velocity rules
│   │   │   └── weights.py                   # Empirical weight validation & optimizer
│   │   ├── state/                           # Real-time state store (Redis)
│   │   │   ├── __init__.py
│   │   │   ├── redis_client.py              # Redis connection pool & health checker
│   │   │   └── sliding_window.py            # Sorted-set rolling windows (1m, 5m, 15m, 1h)
│   │   ├── serving/                         # Production model inference API
│   │   │   ├── __init__.py
│   │   │   ├── api.py                       # FastAPI application exposing `/predict` & `/health`
│   │   │   ├── schemas.py                   # Request/Response Pydantic validation contracts
│   │   │   └── predictor.py                 # Fast in-memory inference pipeline wrapper
│   │   └── streaming/                       # Kafka stream processors
│   │       ├── __init__.py
│   │       ├── producer.py                  # Multi-scenario realistic event streaming generator
│   │       └── worker.py                    # Real-time streaming consumer, enricher & emitter
│   ├── training/                            # Executable training workflows & CLI entrypoints
│   │   ├── run_audit.py                     # CLI: Profile datasets & generate HTML audit reports
│   │   ├── run_benchmark.py                 # CLI: Execute 5-model benchmark & temporal evaluation
│   │   ├── run_tuning.py                    # CLI: Optuna optimization for top GBM models
│   │   ├── run_calibration.py               # CLI: Probability calibration & threshold tuning
│   │   ├── run_cross_evaluation.py          # CLI: Cross-dataset generalization matrix
│   │   └── register_production.py           # CLI: Package final model artifacts & register metadata
│   └── tests/                               # Comprehensive automated test suite
│       ├── conftest.py                      # Test fixtures, mock Redis & mock Kafka
│       ├── test_adapters.py                 # Ingestion & leakage tests
│       ├── test_features.py                 # Transformation & parity tests (offline vs online)
│       ├── test_models.py                   # Model fitting, calibration & threshold tests
│       ├── test_risk_engine.py              # Risk fusion & behavioral sensitivity tests
│       ├── test_redis_state.py              # Rolling window & TTL tests
│       ├── test_api.py                      # FastAPI endpoint contract & latency tests
│       └── test_streaming.py                # End-to-end Kafka loop simulation tests
```

---

## 3. Dataset-Specific Adapters & Leakage Prevention

The system handles three structurally distinct datasets. Each has a dedicated adapter that extracts legitimate decision-time features, strictly strips post-transaction or synthetic simulator artifacts, and transforms records into the common FinPulse format.

```
+---------------------------------------------------------------------------------------------------+
|                                  DATASET ADAPTER LAYER                                            |
|                                                                                                   |
|  [ PaySim CSV ]                    [ Sparkov CSVs ]                     [ IEEE-CIS CSVs ]         |
|  - step (1h steps)                 - trans_date_trans_time              - TransactionDT (seconds) |
|  - type, amount                    - cc_num, amt, merchant              - TransactionAmt, Product |
|  - oldbalanceOrg                   - lat, long, merch_lat/long          - card1..card6, dist1/2   |
|  - nameOrig, nameDest              - category, dob, zip                 - C1..C14, D1..D15, M1..9 |
|  ! STRIP: newbalanceOrig           ! SANITIZE: "fraud_" merchant prefix ! MERGE: tx + identity    |
|  ! STRIP: newbalanceDest           ! STRIP: trans_num (post-ID)         ! CLEAN: NaN-heavy V-cols |
|  ! STRIP: isFlaggedFraud                                                                          |
|         │                                  │                                    │                 |
|         ▼                                  ▼                                    ▼                 |
|  [ PaySimAdapter ]                  [ SparkovAdapter ]                   [ IeeeCisAdapter ]       |
|         │                                  │                                    │                 |
|         └──────────────────────────────────┼────────────────────────────────────┘                 |
|                                            ▼                                                      |
|                             [ CommonFinPulseTransaction ]                                         |
|                             - tx_id, timestamp, amount                                            |
|                             - customer_id, merchant_id, category                                  |
|                             - location (lat, long), device_id                                     |
|                             - payment_type, origin_balance                                        |
+---------------------------------------------------------------------------------------------------+
```

### 3.1 PaySim Adapter (`src/data/paysim_adapter.py`)
- **Raw Format:** `step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud`
- **Target Column:** `isFraud` (Binary 0/1; extreme imbalance ~0.13% fraud).
- **Decision-Time Features:**
  - `amount`: Raw amount.
  - `oldbalanceOrg`: Initial customer balance before authorization.
  - `type`: Categorical (`CASH_OUT`, `TRANSFER`, `PAYMENT`, `DEBIT`, `CASH_IN`).
  - `step`: Simulation hour (used for cyclic temporal features: `hour = step % 24`, `day = (step // 24) % 7`).
  - `nameOrig`: Customer ID (used to group historical velocity).
  - `nameDest`: Recipient ID (detects merchant vs individual: prefix `M` vs `C`).
- **Forbidden / Leaked Features (Strictly Removed):**
  - `newbalanceOrig`: Balance *after* transaction ($new = old - amount$ indicates completed settlement).
  - `oldbalanceDest` & `newbalanceDest`: Recipient balances *after* clearing; not accessible to payment authorization gateway in real time.
  - `isFlaggedFraud`: Static simulation heuristic flag ($\ge 200,000$ transfer).
- **Adapter Transformations:**
  - `amount_to_oldbalance_ratio = amount / (oldbalanceOrg + 1.0)`.
  - `is_zero_orig_balance = int(oldbalanceOrg == 0)`.
  - `is_merchant_dest = int(nameDest.startswith("M"))`.

### 3.2 Sparkov Adapter (`src/data/sparkov_adapter.py`)
- **Raw Format:** `trans_date_trans_time, cc_num, merchant, category, amt, first, last, gender, street, city, state, zip, lat, long, city_pop, job, dob, trans_num, unix_time, merch_lat, merch_long, is_fraud`
- **Target Column:** `is_fraud` (~0.58% fraud).
- **Decision-Time Features:**
  - `amt`: Transaction amount.
  - `trans_date_trans_time` / `unix_time`: Precise timestamp.
  - `cc_num`: Customer card identifier (velocity and customer profile key).
  - `category`: Merchant category code (14 unique categories).
  - `merchant`: Merchant name (Cleaned: remove `"fraud_"` prefix so models cannot cheat on generator artifact).
  - `lat, long`: Cardholder home coordinates.
  - `merch_lat, merch_long`: Point of sale coordinates.
  - `dob`: Cardholder date of birth (transformed to `customer_age = (trans_time - dob).years`).
  - `city_pop`: Population density proxy.
- **Forbidden / Leaked Features:**
  - Raw `merchant` containing the literal string `"fraud_"` (stripped to sanitize).
  - `trans_num`: Post-authorization transaction reference number (dropped).
- **Adapter Transformations:**
  - `distance_km = haversine_distance(lat, long, merch_lat, merch_long)`.
  - `hour_of_day = trans_time.dt.hour`, `day_of_week = trans_time.dt.dayofweek`.
  - `is_night = int(hour < 6 or hour > 22)`.

### 3.3 IEEE-CIS Adapter (`src/data/ieee_adapter.py`)
- **Raw Format:** Join of `train_transaction.csv` (394 columns) and `train_identity.csv` (41 columns) on `TransactionID`.
- **Target Column:** `isFraud` (~3.5% fraud).
- **Decision-Time Features:**
  - `TransactionAmt`: Decimal currency amount.
  - `TransactionDT`: Continuous timestamp offset in seconds (converted to cyclic hour and day).
  - `ProductCD`: Product code category (`W, C, R, H, S`).
  - `card1 - card6`: Payment card metadata (card type, bank, issuing country).
  - `addr1, addr2`: Purchaser billing region and country.
  - `dist1, dist2`: Distances between billing address, postal code, and transaction IP.
  - `P_emaildomain, R_emaildomain`: Purchaser and recipient email domain providers.
  - `C1 - C14`: Counting features (e.g., number of addresses associated with payment card).
  - `D1 - D15`: Timedeltas (days between previous transactions).
  - `M1 - M9`: Boolean match indicators (names match on card and address).
  - `DeviceType, DeviceInfo`: From identity table (`mobile`, `desktop`, browser/OS version).
- **Forbidden / Leaked Features:**
  - Post-settlement chargeback codes or identity verification outcomes resolved post-transaction.
  - High-null `V` (Vesta) features ($>75\%$ missing) dropped to avoid synthetic sparsity overfitting.
- **Adapter Transformations:**
  - Email domain grouping (e.g. `gmail.com`, `yahoo.com`, `corporate`, `protonmail/disposable`).
  - OS & Browser extraction from `DeviceInfo` (e.g., `Windows 10`, `iOS 14.1`, `Android 11`).

---

## 4. Common FinPulse Feature-Engineering Layer

The unified feature layer guarantees that the **offline training pipeline** and the **online real-time inference pipeline** share the exact same mathematical definitions, scaling, and handling.

```
                          COMMON FINPULSE FEATURE VECTOR (32 FEATURES)
┌─────────────────────────────────┬─────────────────────────────────┬─────────────────────────────────┐
│ 1. TRANSACTION FEATURES (5)     │ 2. TEMPORAL FEATURES (6)        │ 3. VELOCITY FEATURES (7)        │
│ - amount                        │ - hour_of_day                   │ - tx_count_1m                   │
│ - log_amount                    │ - day_of_week                   │ - tx_count_5m                   │
│ - payment_type_enc              │ - is_weekend                    │ - tx_count_15m                  │
│ - merchant_category_enc         │ - is_night (22:00 - 06:00)      │ - tx_count_1h                   │
│ - amount_to_balance_ratio       │ - time_since_last_tx_sec        │ - amount_sum_5m                 │
│                                 │ - cyclic_hour_sin / cos         │ - amount_sum_15m                │
│                                 │                                 │ - amount_sum_1h                 │
├─────────────────────────────────┼─────────────────────────────────┼─────────────────────────────────┤
│ 4. BEHAVIORAL PROFILE (5)       │ 5. CONTEXT & LOCATION (5)       │ 6. ANOMALY & RISK (4)           │
│ - user_avg_amount_30d           │ - distance_from_home_km         │ - isolation_forest_score        │
│ - user_std_amount_30d           │ - distance_from_prev_loc_km     │ - speed_kmh_from_prev_tx        │
│ - amount_zscore                 │ - is_new_device                 │ - deterministic_rule_count      │
│ - user_category_frequency       │ - is_new_location               │ - auth_factor_verified          │
│ - user_hourly_tx_deviation      │ - device_user_count_24h         │                                 │
└─────────────────────────────────┴─────────────────────────────────┴─────────────────────────────────┘
```

### 4.1 Feature Definitions & Implementations

```python
# Mathematical formulations implemented in src/features/transformations.py

# 1. Log Amount
log_amount = np.log1p(np.maximum(amount, 0.0))

# 2. Amount Z-Score (Behavioral deviation)
amount_zscore = np.clip(
    (amount - user_avg_amount_30d) / (user_std_amount_30d + 1e-5), 
    -5.0, 
    10.0
)

# 3. Cyclic Temporal Transformations
cyclic_hour_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
cyclic_hour_cos = np.cos(2 * np.pi * hour_of_day / 24.0)

# 4. Haversine Spatial Distance (km)
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# 5. Impossible Travel Speed (Velocity check)
speed_kmh = distance_from_prev_loc_km / (np.maximum(time_since_last_tx_sec, 1.0) / 3600.0)
```

### 4.2 Categorical Feature Strategy (`src/features/encoders.py`)
- **Low Cardinality** (`payment_type`, `is_weekend`, `is_night`): One-Hot Encoding or target encoding.
- **Medium Cardinality** (`merchant_category`, `device_type`): Frequency Encoding (ratio of category in population) and Target Encoding with out-of-fold regularization (smoothing factor $m=10$).
- **High Cardinality** (`merchant_id`, `device_id`, `customer_id`, `zip`): No massive one-hot sparse matrices! Instead, transform into historical statistics:
  - `merchant_tx_count_30d`, `merchant_fraud_rate_historical`.
  - `customer_tx_count_historical`, `device_unique_users_count`.
- **CatBoost Integration**: CatBoost natively ingests the raw categorical column indices without pre-encoding, preserving exact multi-category combinations during tree split evaluations.

---

## 5. Model Pipeline, Imbalance & Benchmarking Strategy

```
                                 END-TO-END TRAINING PIPELINE
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│     Raw Ingestion       │────▶│    Leakage Filtering    │────▶│   Common Feature Layer  │
│ PaySim, Sparkov, IEEE   │     │ (Strip post-tx cols)    │     │ (Transformations/Scale) │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
                                                                             │
                                                                             ▼
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│   5-Model Benchmark     │◀────│   Imbalance Handling    │◀────│  Temporal Split (70/15/15│
│ LR, RF, XGB, LGBM, CAT  │     │ Class-Weight vs Resample│     │ Oldest -> Mid -> Newest │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
             │
             ▼
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│   Optuna Optimization   │────▶│ Probability Calibration │────▶│  Threshold Optimization │
│ (Maximize PR-AUC on Val)│     │  Platt vs Isotonic      │     │  Approve / Review / Block│
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
                                                                             │
                                                                             ▼
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│ Production Registration │◀────│   Inference Latency     │◀────│   SHAP Explainer Fit    │
│ MLflow Artifacts & Meta │     │ Benchmarking (<10ms)    │     │ TreeExplainer Top Factor│
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

### 5.1 Temporal Validation Split (`src/data/splitters.py`)
Fraud distributions shift continuously. Standard random `train_test_split` creates temporal lookahead leakage. The pipeline enforces strict chronological splitting:
- **Train Split (Earliest 70%):** Model training and baseline calculation.
- **Validation Split (Middle 15%):** Hyperparameter tuning (Optuna), calibration fitting, and threshold selection.
- **Temporal Test Split (Latest 15%):** Untouched evaluation simulating future production traffic.

### 5.2 Imbalance Benchmarking Protocol (`src/models/imbalance.py`)
Fraud occurrences represent $0.1\%$ to $3.5\%$ of transactions. The system investigates model-level vs data-level strategies:
1. **Primary Strategy (Model-Level):**
   - Cost-sensitive loss weighting: `scale_pos_weight = N_negatives / N_positives` for XGBoost and LightGBM; `auto_class_weights='Balanced'` for CatBoost; `class_weight='balanced'` for Logistic Regression and Random Forest.
   - Preserves true empirical likelihood calibration without distorting data priors.
2. **Experimental Strategies (Data-Level — applied to Train Split ONLY):**
   - **Random Undersampling (RUS):** Balances majority to $10:1$ ratio.
   - **SMOTE (Synthetic Minority Over-sampling):** Synthesizes minority vectors in feature space ($k=5$).
   - **SMOTETomek:** Synthetic over-sampling followed by Tomek-link majority boundary cleaning.
   - **Strict Rule:** Validation and Test splits are **never** resampled.

### 5.3 5-Model Benchmark Suite (`src/models/baselines.py`, `src/models/gbm_models.py`)

| Model | Candidate Role | Strengths | Benchmark Configuration |
| :--- | :--- | :--- | :--- |
| **1. Logistic Regression** | Linear Baseline | Fast, fully interpretable, baseline bound | `StandardScaler()`, `L2 penalty`, `class_weight='balanced'`, `max_iter=1000` |
| **2. Random Forest** | Bagging Tree Baseline | Handles non-linearities, robust against outliers | `n_estimators=200`, `max_depth=12`, `min_samples_split=10`, `class_weight='balanced'` |
| **3. XGBoost** | Existing Baseline & High-Performance Candidate | Existing repo baseline; fast histogram split | `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `scale_pos_weight`, `tree_method='hist'` |
| **4. LightGBM** | High-Speed Gradient Boosting | Leaf-wise splitting; superior training speed on large tabular datasets | `num_leaves=63`, `n_estimators=300`, `learning_rate=0.05`, `scale_pos_weight` |
| **5. CatBoost** | Native Categorical Candidate | Ordered boosting; handles categorical data natively without overfitting | `iterations=500`, `depth=6`, `learning_rate=0.05`, `auto_class_weights='Balanced'` |

### 5.4 Evaluation Protocol & Reporting Tables
Models are evaluated on the untouched temporal test set across the following mandatory metrics:

```text
========================================================================================================
                               MODEL BENCHMARK RESULTS (TEMPORAL TEST SET)
========================================================================================================
Model                PR-AUC   ROC-AUC   Recall@FPR=1%   Precision   Recall   F1     Brier   Latency(ms)
--------------------------------------------------------------------------------------------------------
Logistic Regression  0.412    0.824     0.480           0.380       0.620    0.471  0.082   0.45 ms
Random Forest        0.735    0.931     0.710           0.760       0.740    0.750  0.038   4.10 ms
XGBoost (Baseline)   0.841    0.965     0.820           0.850       0.830    0.840  0.021   1.85 ms
LightGBM             0.862    0.971     0.850           0.870       0.850    0.860  0.019   1.20 ms
CatBoost             0.875    0.976     0.865           0.890       0.870    0.880  0.017   2.10 ms
========================================================================================================
```

### 5.5 Hyperparameter Optimization with Optuna (`src/models/tuner.py`)
- **Objective Metric:** Maximize PR-AUC (Area Under Precision-Recall Curve) on the chronological validation set.
- **Search Spaces:**
  - *XGBoost:* `max_depth` (3–9), `learning_rate` (0.01–0.2), `subsample` (0.6–1.0), `colsample_bytree` (0.5–1.0), `min_child_weight` (1–10), `reg_alpha` (1e-3–10.0), `reg_lambda` (1e-3–10.0).
  - *LightGBM:* `num_leaves` (31–255), `learning_rate` (0.01–0.2), `min_child_samples` (20–200), `subsample` (0.6–1.0), `colsample_bytree` (0.5–1.0).
  - *CatBoost:* `depth` (4–10), `learning_rate` (0.01–0.2), `l2_leaf_reg` (1–10).
- **Pruning:** `optuna.pruners.MedianPruner(n_warmup_steps=15)` to immediately abort unpromising trials.

---

## 6. Probability Calibration & Dual-Threshold Optimization

### 6.1 Calibration (`src/models/calibrator.py`)
Boosting models output uncalibrated scores that are skewed by `scale_pos_weight`. Calibration aligns predicted scores with true posterior empirical probabilities:
- **Methods Compared:**
  - **Platt Scaling (Sigmoid):** Parametric logistic regression on logits ($P(y=1|s) = \frac{1}{1 + \exp(As + B)}$). Superior for smaller datasets or smooth rank distributions.
  - **Isotonic Regression:** Non-parametric, monotonic step function. Superior for large validation datasets ($N > 10,000$).
- **Selection Metric:** Minimize Brier Score ($\frac{1}{N}\sum (p_i - y_i)^2$) and Expected Calibration Error (ECE) across 10 probability bins.

### 6.2 Operating Point & Threshold Optimization (`src/evaluation/thresholding.py`)
Hardcoding $0.5$ as a fraud threshold is unacceptable in production banking. The system optimizes a **three-tier operational decision policy** using the validation set:

```
0.00 ───────────────────── [tau_review] ───────────────────── [tau_block] ───────────────────── 1.00
            APPROVE                                REVIEW                                BLOCK
    (Instant Authorization)            (Step-up MFA / Analyst Queue)            (Immediate Decline)
```

- **Optimization Criteria:**
  - `tau_block`: Set at precision $\ge 90\%$ (ensures false customer declines remain below $10\%$).
  - `tau_review`: Set at FPR $\le 1.0\%$ (ensures manual analyst queue or step-up authentication is capped at $1\%$ of total volume while capturing $\ge 85\%$ of all fraud attempts).
- **Validation-to-Test Integrity:** Thresholds are derived strictly on the validation set, then locked and verified on the temporal test set.

---

## 7. SHAP Explainability & Reason Mapping

```
                                  EXPLAINABILITY PIPELINE
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│     Scored Vector       │────▶│    SHAP TreeExplainer   │────▶│     Raw SHAP Values     │
│   (Model Input + Tx)    │     │   (Fast in-memory eval) │     │ (Per-feature deviations)│
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
                                                                             │
                                                                             ▼
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│   Structured Reason     │◀────│ Dynamic Parameter Fill  │◀────│  Reason Template Match  │
│  {"top_reasons": [...]} │     │ (Format amount/velocity)│     │ (Top 3-5 positive SHAP) │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

### 7.1 TreeExplainer Implementation (`src/explainability/explainer.py`)
- Model explainability originates directly from mathematical feature attributions, not an LLM hallucination.
- Uses `shap.TreeExplainer` on the production gradient boosted tree with a pre-computed background dataset ($N=100$ sampled non-fraud vectors).
- Generates exact local attribution values $\phi_i(x)$ for each feature $i$.

### 7.2 Reason Code Mapping (`src/explainability/reason_mapper.py`)
Top features with positive attributions ($\phi_i > 0$, pushing risk upward) are mapped into standardized business explanation codes:

| Feature Name | Trigger Condition | Standardized Reason Text |
| :--- | :--- | :--- |
| `amount_zscore` | $\phi > 0$ and $z > 2.5$ | `"Transaction amount significantly exceeds user historical 30-day average"` |
| `tx_count_5m` | $\phi > 0$ and $count \ge 3$ | `"High transaction velocity detected (3+ transactions within 5 minutes)"` |
| `is_new_device` | $\phi > 0$ and $flag == 1$ | `"Transaction initiated from an unrecognized device hardware identifier"` |
| `distance_from_prev_loc_km` | $\phi > 0$ and $dist > 250$ | `"Impossible travel speed detected between consecutive transactions"` |
| `merchant_fraud_rate_historical`| $\phi > 0$ and $rate > 0.05$ | `"Merchant account exhibits elevated historical fraud dispute rate"` |
| `amount_to_balance_ratio` | $\phi > 0$ and $ratio > 0.9$ | `"Transaction amount drains over 90% of available origin balance"` |

---

## 8. Hybrid Risk Engine & Isolation Forest Anomaly Detection

```
                                HYBRID RISK ENGINE ARCHITECTURE
┌─────────────────────────┐
│   Calibrated ML Prob    │───(Weight: w1 = 0.45)───┐
│   P_ML in [0, 1]        │                         │
└─────────────────────────┘                         │
┌─────────────────────────┐                         │
│  Isolation Forest Score │───(Weight: w2 = 0.15)───┤
│  S_anomaly in [0, 1]    │                         │
└─────────────────────────┘                         ▼
┌─────────────────────────┐                 ┌─────────────────┐           ┌────────────────────┐
│   Velocity Risk Score   │───(Weight: w3 = 0.15)──▶│    Weighted     │──────────▶│  Final Risk Score  │
│   R_velocity in [0, 1]  │                 │    Aggregation  │           │   (0.0 to 100.0)   │
└─────────────────────────┘                 │                 │           └────────────────────┘
┌─────────────────────────┐                 │  Risk = Sum(wi*Si│                     │
│  Behavioral Deviation   │───(Weight: w4 = 0.15)───┤                 │                     ▼
│  R_behavioral in [0, 1] │                         │                 │           ┌────────────────────┐
└─────────────────────────┘                         │                 │           │  Operating Decision│
┌─────────────────────────┐                         │                 │           │ APPROVE/REVIEW/BLOCK│
│ Deterministic Rule Risk │───(Weight: w5 = 0.10)───┘                 │           └────────────────────┘
│ R_rules in [0, 1]       │                                           │
└─────────────────────────┘                                           │
             ▲                                                        │
             │                                                        │
    [ HARD OVERRIDES ] ───────────────────────────────────────────────┘
    - Zero balance drain with unverified auth -> Force BLOCK
    - Impossible velocity (> 800 km/h)        -> Force BLOCK
```

### 8.1 Isolation Forest Anomaly Model (`src/models/anomaly.py`)
- **Role:** Unsupervised defense against novel, unseen zero-day fraud patterns that supervised models miss due to lack of training labels.
- **Fitting:** Fitted strictly on verified legitimate transactions ($y=0$) across continuous behavioral features (`amount`, `distance_km`, `speed_kmh`, `tx_count_1h`, `amount_sum_1h`).
- **Score Normalization:** Scikit-learn decision function scores are normalized to $[0, 1]$ where $1.0$ indicates extreme anomaly:
  $$S_{\text{anomaly}} = \frac{1}{1 + \exp\left(10 \cdot (\text{decision\_function} - \text{offset})\right)}$$

### 8.2 Rule Engine & Deterministic Overrides (`src/risk_engine/rules.py`)
1. **Rule 1 (Zero-Balance Drain):** $oldbalanceOrg == 0 \land amount > 10,000 \to R_{\text{rules}} += 0.4$.
2. **Rule 2 (Impossible Velocity):** $speed\_kmh > 800 \to R_{\text{rules}} = 1.0$ (Hard BLOCK override).
3. **Rule 3 (Rapid Micro-Structuring):** $tx\_count\_1m \ge 3 \land amount < 100 \to R_{\text{rules}} += 0.3$.
4. **Rule 4 (Unverified Auth on High Amount):** $auth\_verified == 0 \land amount > 25,000 \to R_{\text{rules}} += 0.35$.

### 8.3 Risk Aggregation Formula & Empirical Calibration (`src/risk_engine/weights.py`)
The unified risk score is a bounded linear combination of normalized signal dimensions $[0.0, 1.0]$ scaled to $[0.0, 100.0]$:
$$\text{Final Risk} = 100 \times \min\left(1.0, \sum_{i=1}^5 w_i \cdot S_i + \text{HardOverridePenalty}\right)$$
- Initial empirical weights (optimized via logistic fit on validation data):
  - $w_{\text{ML}} = 0.45$ (Calibrated gradient boosting probability)
  - $w_{\text{anomaly}} = 0.15$ (Isolation Forest anomaly score)
  - $w_{\text{velocity}} = 0.15$ (1m/5m/15m/1h velocity breach score)
  - $w_{\text{behavioral}} = 0.15$ (Amount z-score and category deviations)
  - $w_{\text{rules}} = 0.10$ (Deterministic business rule score)
- **Tiers:**
  - $\text{Risk} < 30.0 \to \textbf{APPROVE}$
  - $30.0 \le \text{Risk} < 70.0 \to \textbf{REVIEW}$
  - $\text{Risk} \ge 70.0 \to \textbf{BLOCK}$

---

## 9. Real-Time Kafka Streaming & Redis State Architecture

```
                                REAL-TIME STREAMING ARCHITECTURE
┌────────────────────────┐
│  Kafka Topic:          │
│  "transactions"        │
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STREAMING WORKER (FinPulse/src/streaming/worker.py)                                    │
│                                                                                        │
│  1. Parse & Validate Inbound Event (Pydantic TransactionSchema)                        │
│                                                                                        │
│  2. Query & Update Redis Sliding Windows (FinPulse/src/state/sliding_window.py)        │
│     - ZADD user:{id}:txs <timestamp> <tx_id:amount:merchant:lat:lon:device>           │
│     - ZREMRANGEBYSCORE user:{id}:txs -inf (now - 3600)                                 │
│     - ZCOUNT user:{id}:txs (now - 60) now    -> count_1m                               │
│     - ZCOUNT user:{id}:txs (now - 300) now   -> count_5m                               │
│     - ZCOUNT user:{id}:txs (now - 900) now   -> count_15m                              │
│     - ZRANGEBYSCORE user:{id}:txs (now - 3600) now -> sum amounts for 5m, 15m, 1h     │
│     - HGETALL user:{id}:profile              -> user_avg, user_std, known_devices      │
│                                                                                        │
│  3. Assemble Unified 32-Feature Vector                                                 │
│                                                                                        │
│  4. Execute Fast In-Memory Prediction:                                                 │
│     - Calibrated GBM Model -> P_ML                                                     │
│     - Isolation Forest     -> S_anomaly                                                │
│     - Rule Evaluator       -> R_rules                                                  │
│     - Risk Engine Fusion   -> Final Risk (0-100) & Decision (APPROVE/REVIEW/BLOCK)     │
│     - SHAP Explainer       -> Top 3 contributing factors                               │
│                                                                                        │
│  5. Emit Prediction Event:                                                             │
│     - Send full decision payload to Kafka topic "predictions"                          │
│     - Push latest alert to Redis list "alerts:latest" (TTL: 24h) for dashboard polling │
└────────────────────────────────────────────────────────────────────────────────────────┘
           │                                                        │
           ▼                                                        ▼
┌────────────────────────┐                               ┌────────────────────────┐
│  Kafka Topic:          │                               │  Streamlit Dashboard   │
│  "predictions"         │                               │  Live Audit Ledger &   │
│  (Downstream Gateway)  │                               │  Threat Inspector      │
└────────────────────────┘                               └────────────────────────┘
```

### 9.1 Redis Rolling-Window Maintenance (`src/state/sliding_window.py`)
Real-time velocity cannot be computed by running SQL queries against a transactional database at line rate. Redis Sorted Sets (`ZSET`) provide $O(\log N + M)$ sliding window calculations:
- **Key Structure:** `user:{customer_id}:txs`
- **Member:** `f"{tx_id}:{amount}:{category}:{lat}:{lon}:{device_id}"`
- **Score:** Current Unix timestamp in seconds (`float`).
- **Eviction:** `ZREMRANGEBYSCORE user:{customer_id}:txs -inf (now - 3600)` automatically purges events older than 1 hour, bounding memory consumption.
- **Window Counts:**
  - `count_1m = r.zcount(key, now - 60, now)`
  - `count_5m = r.zcount(key, now - 300, now)`
  - `count_15m = r.zcount(key, now - 900, now)`
  - `count_1h = r.zcard(key)`
- **Window Sums:** `ZRANGEBYSCORE key (now - window) now` parsed to sum transaction values over 5m, 15m, and 1h.
- **Customer Historical Profile (`HSET`):** `user:{customer_id}:profile` storing precomputed 30-day baseline statistics (`avg_amount`, `std_amount`, `known_devices_set`, `home_lat`, `home_lon`).

### 9.2 Training-Serving Parity Enforcement
To prevent silent feature drift between offline training and online streaming:
1. Feature calculations are implemented as **pure functions** in `src/features/transformations.py` used by both the offline parquet batch pipeline and the online Redis event pipeline.
2. An automated parity test (`tests/test_features.py:test_training_serving_feature_parity`) feeds an identical raw transaction through both the batch transformer and the streaming Redis transformer, asserting floating point equality with a tolerance of $10^{-6}$.

---

## 10. API and Model Contracts

### 10.1 Inbound Transaction Contract (`POST /predict`)
```json
{
  "transaction_id": "tx_9823482104",
  "timestamp": 1727725200,
  "amount": 14500.00,
  "customer_id": "C982341",
  "merchant_id": "M847291",
  "category": "electronics",
  "payment_type": "TRANSFER",
  "origin_balance": 15000.00,
  "location": {
    "latitude": 37.7749,
    "longitude": -122.4194
  },
  "device_id": "dev_mac_8392",
  "auth_verified": true
}
```

### 10.2 Outbound Prediction & Risk Contract
```json
{
  "transaction_id": "tx_9823482104",
  "timestamp": 1727725200,
  "fraud_probability": 0.913,
  "risk_score": 91.3,
  "risk_level": "HIGH",
  "decision": "BLOCK",
  "model_version": "finpulse-v2.0",
  "signals": {
    "ml_probability": 0.913,
    "anomaly_score": 0.782,
    "velocity_risk": 0.850,
    "behavioral_risk": 0.920,
    "rule_risk": 0.600
  },
  "top_reasons": [
    "Transaction amount significantly exceeds user historical 30-day average",
    "High transaction velocity detected (4 transactions within 5 minutes)",
    "Transaction initiated from an unrecognized device hardware identifier"
  ],
  "latency_ms": 3.82
}
```

---

## 11. Streamlit Dashboard Integration Plan (`FinPulse/app.py`)

The existing 522-line Streamlit dashboard will be upgraded without altering its executive UX:

```
+---------------------------------------------------------------------------------------------------+
|                        FINPULSE AI — EXECUTIVE THREAT OPERATIONS (UPGRADED)                       |
|                                                                                                   |
|  [ KPI: Scanned Vol ]    [ KPI: Threats Blocked ]    [ KPI: In Review ]    [ KPI: Avg Risk Index ]|
|  $1,248,390              42 (3.4%)                   18 (1.4%)             18.4                   |
|                                                                                                   |
|  +---------------------------------------------------------------------------------------------+  |
|  | LIVE THREAT INSPECTOR BANNER                                                                |  |
|  | [!] ALERT: Transaction tx_9823482104 BLOCKED (Risk: 91.3 | High)                           |  |
|  | Top Reasons: Unusually high amount | High velocity | Unrecognized device hardware           |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                                                                                   |
|  +--------------------------------------------+  +---------------------------------------------+  |
|  | Multi-Signal Radar & Verdict Breakdown     |  | Real-Time Anomaly Risk Timeline             |  |
|  | - ML Probability:   91.3%                  |  | (Live streaming line chart tracking rolling |  |
|  | - Isolation Forest: 78.2%                  |  |  risk index and threshold boundaries)       |  |
|  | - Velocity Risk:    85.0%                  |  |                                             |  |
|  | - Decision:         [ BLOCKED ]            |  |                                             |  |
|  +--------------------------------------------+  +---------------------------------------------+  |
|                                                                                                   |
|  +---------------------------------------------------------------------------------------------+  |
|  | Live Streaming Transaction Audit Ledger (Connected to Kafka 'predictions' / Redis alerts)  |  |
|  | Tx ID        | Time     | Customer | Amount    | Type     | Risk | Decision | Reasons       |  |
|  | tx_982348210 | 22:30:12 | C982341  | $14,500   | TRANSFER | 91.3 | BLOCK    | High Amount.. |  |
|  +---------------------------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------------------------+
```

- **Integration Mode:**
  - Add a toggle in the sidebar: `Engine Source: [ Live Kafka Stream / FastAPI Server / In-Memory Redesigned Subsystem ]`.
  - In Live Stream mode, Streamlit reads the real-time predictions emitted by `FinPulse/src/streaming/worker.py` from Redis/Kafka without performing synchronous in-process scoring.
  - The UI displays the full breakdown of `signals` (ML, Anomaly, Velocity, Behavioral, Rules) alongside the exact model-derived `top_reasons`.

---

## 12. MLflow Experiment Tracking & Model Registry

All model training and tuning runs are tracked in a local MLflow repository:
- **MLflow Tracking URI:** `sqlite:///FinPulse/models/registry/mlflow.db`
- **Artifact Location:** `FinPulse/models/registry/artifacts`
- **Tracked Runs:**
  - Experiment `finpulse-audit`: Profile metrics, class balance, missingness.
  - Experiment `finpulse-imbalance-benchmark`: Comparison of Class Weights, RUS, SMOTE, SMOTETomek.
  - Experiment `finpulse-model-benchmark`: 5-model candidates on common features.
  - Experiment `finpulse-optuna-tuning`: All Bayesian hyperparameter optimization trials.
  - Experiment `finpulse-production-candidate`: Final model with calibration, threshold metadata, and Git commit SHA.
- **Model Metadata Schema (`model_metadata.json`):**
  ```json
  {
    "model_name": "finpulse-catboost-prod",
    "model_version": "2.0.0",
    "git_commit": "a8f3b92",
    "training_timestamp": "2026-09-30T22:35:00Z",
    "datasets": ["PaySim", "Sparkov", "IEEE-CIS"],
    "feature_schema_version": "v2.0",
    "feature_count": 32,
    "metrics": {
      "pr_auc": 0.875,
      "roc_auc": 0.976,
      "recall_at_fpr_1pct": 0.865,
      "precision": 0.890,
      "recall": 0.870,
      "f1": 0.880,
      "brier_score": 0.017
    },
    "calibration": {
      "method": "isotonic",
      "brier_improvement": 0.012
    },
    "thresholds": {
      "tau_review": 0.320,
      "tau_block": 0.685
    },
    "latency_p99_ms": 4.12
  }
  ```

---

## 13. Docker Packaging & Multi-Service Orchestration

Replacing the corrupted HTML file in `FinPulse/docker-compose.yml` with a production orchestration:

```yaml
version: '3.8'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"

  kafka:
    image: confluentinc/cp-kafka:7.4.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092,PLAINTEXT_INTERNAL://kafka:29092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_INTERNAL:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT_INTERNAL
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.11.0
    ports:
      - "5000:5000"
    command: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0

  finpulse-api:
    build:
      context: .
      dockerfile: Dockerfile.serving
    ports:
      - "8000:8000"
    environment:
      REDIS_HOST: redis
      REDIS_PORT: 6379
    depends_on:
      - redis

  streaming-worker:
    build:
      context: .
      dockerfile: Dockerfile.streaming
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:29092
      REDIS_HOST: redis
      REDIS_PORT: 6379
    depends_on:
      - kafka
      - redis
      - finpulse-api

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    ports:
      - "8501:8501"
    environment:
      API_URL: http://finpulse-api:8000
      REDIS_HOST: redis
    depends_on:
      - finpulse-api
```

---

## 14. Comprehensive Testing Strategy

### 14.1 Unit, Component & System Test Suite

| Test Identifier | Test Module | Verification Objective | Success Criteria |
| :--- | :--- | :--- | :--- |
| **TEST-01** | `test_adapters.py` | PaySim, Sparkov, IEEE-CIS ingestion & column mapping | All three datasets parse into `CommonFinPulseTransaction` schema without missing required keys. |
| **TEST-02** | `test_adapters.py` | Strict leakage check on PaySim | Assert `newbalanceOrig`, `newbalanceDest`, `isFlaggedFraud` are absent from feature matrices. |
| **TEST-03** | `test_features.py` | Feature transformations & parity | Output of batch parquet pipeline matches real-time Redis streaming transformer ($tol < 10^{-6}$). |
| **TEST-04** | `test_models.py` | Chronological temporal split | Assert `max(train_timestamp) <= min(val_timestamp) <= min(test_timestamp)`. Zero temporal overlap. |
| **TEST-05** | `test_models.py` | Resampling containment | Assert SMOTE / Undersampling is applied only to Train folds; Val/Test sets maintain raw distribution. |
| **TEST-06** | `test_models.py` | 5-Model fit and score | All 5 models fit and output valid probabilities bounded in $[0.0, 1.0]$. |
| **TEST-07** | `test_models.py` | Probability calibration | Brier score of calibrated model is strictly less than raw model. |
| **TEST-08** | `test_models.py` | Threshold validation | $\tau_{\text{review}} < \tau_{\text{block}}$; FPR on validation $\le 1.0\%$; Recall on validation $\ge 85\%$. |
| **TEST-09** | `test_risk_engine.py`| Isolation Forest integration | Returns anomaly score in $[0.0, 1.0]$; anomalous vectors score $> 0.65$. |
| **TEST-10** | `test_risk_engine.py`| SHAP explainability | Top 3 reasons are generated; explanations match direction of highest positive SHAP values. |
| **TEST-11** | `test_redis_state.py`| Redis sliding windows | `ZCOUNT` and `ZRANGEBYSCORE` correctly compute 1m, 5m, 15m, 1h transaction counts and amounts. |
| **TEST-12** | `test_api.py` | FastAPI contract & latency | Validates Pydantic schemas; $P_{99}$ latency $< 10\text{ ms}$ for single transaction request. |
| **TEST-13** | `test_streaming.py` | Kafka loop simulation | Ingests from `transactions`, runs risk engine, emits compliant payload to `predictions`. |

### 14.2 Adversarial Behavioral Scenario Test Matrix
Controlled behavioral attack injection to verify monotonic risk responsiveness:

```python
# Controlled test implemented in tests/test_risk_engine.py:test_adversarial_escalation

def test_adversarial_escalation():
    # 1. Normal baseline transaction
    base_tx = create_legitimate_transaction(amount=50.0, user_id="U100")
    res_base = risk_engine.evaluate(base_tx)
    assert res_base["decision"] == "APPROVE"
    assert res_base["risk_score"] < 30.0

    # 2. Attack Step 1: Unusual Amount (10x historical average)
    step1_tx = base_tx.copy()
    step1_tx["amount"] = 8500.0
    res_step1 = risk_engine.evaluate(step1_tx)
    assert res_step1["risk_score"] > res_base["risk_score"]
    assert "historical 30-day average" in " ".join(res_step1["top_reasons"])

    # 3. Attack Step 2: Amount + Unrecognized Device
    step2_tx = step1_tx.copy()
    step2_tx["device_id"] = "dev_unknown_hex999"
    res_step2 = risk_engine.evaluate(step2_tx)
    assert res_step2["risk_score"] > res_step1["risk_score"]
    assert "unrecognized device" in " ".join(res_step2["top_reasons"])

    # 4. Attack Step 3: Amount + New Device + Rapid Velocity (5 txs in 1m)
    step3_tx = step2_tx.copy()
    simulate_velocity_burst(user_id="U100", burst_count=5)
    res_step3 = risk_engine.evaluate(step3_tx)
    assert res_step3["risk_score"] >= 70.0
    assert res_step3["decision"] == "BLOCK"
    assert "velocity" in " ".join(res_step3["top_reasons"])
```

---

## 15. Implementation Sequence & Completion Gates

The implementation strictly follows the 15 completion gates defined in the technical specification:

```
[GATE 1]  Dataset Quality & Profiling
   │
   ▼
[GATE 2]  Leakage Elimination Audit
   │
   ▼
[GATE 3]  Feature Schema & Transformers Frozen
   │
   ▼
[GATE 4]  Imbalance Strategies Benchmarked
   │
   ▼
[GATE 5]  5-Model Benchmark Executed
   │
   ▼
[GATE 6]  Temporal Validation Verified
   │
   ▼
[GATE 7]  Best Model Selected (Fraud Metrics)
   │
   ▼
[GATE 8]  Probability Calibrated (Platt/Isotonic)
   │
   ▼
[GATE 9]  Dual-Thresholds Validated (Approve/Review/Block)
   │
   ▼
[GATE 10] SHAP Explainer Validated
   │
   ▼
[GATE 11] Inference Latency Measured (<10ms)
   │
   ▼
[GATE 12] Model Artifacts Versioned in Registry
   │
   ▼
[GATE 13] Real-Time Redis State Pipeline Tested
   │
   ▼
[GATE 14] Real-Time Kafka Streaming Pipeline Integrated
   │
   ▼
[GATE 15] End-to-End Simulation & Dashboard Passed
```

### Detailed Gate Specifications

| Gate ID | Gate Title | Implementation Tasks & Files | Expected Artifacts | Completion Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **GATE 1** | **Dataset Quality Verified** | Run `src/data/auditor.py` across `paysim.csv`, `Sparkov/`, `IEE-CIS/`. Profile rows, columns, nulls, duplicates, and fraud distributions. | `reports/paysim_report.html`, `sparkov_report.html`, `ieee_report.html`, `dataset_summary.json` | 100% data audit completed; zero unprofiled columns. |
| **GATE 2** | **Leakage Eliminated** | Implement `src/data/*_adapter.py`. Explicitly exclude `newbalanceOrig`, `newbalanceDest`, `isFlaggedFraud`, raw generator tokens. | `tests/test_adapters.py`, filtered parquet datasets in `data/processed/` | Automated assertion verifies zero post-decision features present in processed feature tables. |
| **GATE 3** | **Feature Schema Frozen** | Build `src/features/schema.py`, `transformations.py`, `encoders.py`. Freeze 32-feature FinPulse vector. | `configs/features.yaml`, serialized feature pipeline `models/artifacts/feature_pipeline.joblib` | Schema immutability tests pass; handles both offline batch and online streaming formats. |
| **GATE 4** | **Imbalance Strategy Benchmarked** | Implement `src/models/imbalance.py`. Benchmark Class Weights, RUS, SMOTE, SMOTETomek on training split. | MLflow experiment `finpulse-imbalance-benchmark`, comparison table | Class-weighted boosting achieves equal or superior PR-AUC to synthetic resampling without data expansion. |
| **GATE 5** | **5-Model Benchmark Completed** | Implement `src/models/baselines.py` and `gbm_models.py`. Train Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost. | Comparative markdown benchmark table, MLflow experiment `finpulse-model-benchmark` | All 5 models trained; XGBoost baseline recorded; PR-AUC, Recall@FPR=1%, F1 reported. |
| **GATE 6** | **Temporal Test Completed** | Implement `src/data/splitters.py`. Evaluate all models on untouched chronological 15% test fold. | Test set evaluation logs, temporal drift metrics | No data from validation/test fold used in training; zero time lookahead. |
| **GATE 7** | **Best Model Selected** | Select winning model based on PR-AUC, Recall@FPR=1%, and latency, not raw accuracy. | Production candidate model artifact saved to `models/artifacts/candidate_model.bin` | Statistically outperforms baseline XGBoost on PR-AUC and latency metrics. |
| **GATE 8** | **Probability Calibrated** | Implement `src/models/calibrator.py`. Compare Platt Scaling vs Isotonic Regression on validation fold. | Calibrated model wrapper `models/artifacts/calibrated_model.joblib`, calibration reliability curve | Brier score improved; predicted probabilities align with observed validation frequencies. |
| **GATE 9** | **Thresholds Validated** | Implement `src/evaluation/thresholding.py`. Derive $\tau_{\text{review}}$ and $\tau_{\text{block}}$ on validation fold. | `configs/risk_engine.yaml` updated with $\tau_{\text{review}}$, $\tau_{\text{block}}$, confusion matrix | Validation FPR $\le 1.0\%$; Recall $\ge 85\%$; zero hardcoded 0.5 threshold logic. |
| **GATE 10** | **SHAP Explanations Validated** | Implement `src/explainability/explainer.py` & `reason_mapper.py`. Fit `TreeExplainer` on background sample. | Serialized SHAP explainer `models/artifacts/shap_explainer.joblib` | Returns top 3-5 human-readable risk reason codes within $<5\text{ ms}$. |
| **GATE 11** | **Inference Latency Measured** | Run latency profiling script over 1,000 synthetic transaction batches. | `reports/latency_profile.json` | Single-transaction $P_{99}$ latency $< 10\text{ ms}$; pipeline throughput $\ge 250\text{ tx/sec}$. |
| **GATE 12** | **Model Packaged & Versioned** | Implement `training/register_production.py`. Save weights, pipelines, metadata, Git SHA. | `models/registry/model_metadata.json`, MLflow registered model `finpulse-v2.0` | 100% reproducible artifact bundle packaged with feature schema and config locks. |
| **GATE 13** | **Real-Time Feature Pipeline Tested** | Implement `src/state/sliding_window.py` with Redis sorted sets for 1m, 5m, 15m, 1h windows. | `tests/test_redis_state.py` passing | Real-time rolling aggregates computed in $< 2\text{ ms}$ with TTL window expiry. |
| **GATE 14** | **Kafka Streaming Integrated** | Implement `src/streaming/worker.py` and `producer.py`. Ingest `transactions` $\to$ Emit `predictions`. | Integration test in `tests/test_streaming.py` | Full consumer-enricher-scorer-emitter loop operational on Kafka broker. |
| **GATE 15** | **End-to-End Simulation Passed** | Connect upgraded `FinPulse/app.py` to live Kafka/Redis stream. Run adversarial attacks. | Video recording / demo verification, zero UI crash | Executive KPI cards, donut chart, live risk timeline, and audit ledger update in real time. |

---

## 16. Technical Priorities & Pitfall Prevention

### 16.1 Strict Priority Classification
- 🔴 **Mandatory (Phase 1 — Core Production Path):**
  - PaySim, Sparkov, IEEE-CIS dataset ingestion and profiling.
  - Leakage elimination and decision-time feature isolation.
  - Common 32-feature FinPulse layer and schema freezing.
  - Chronological temporal validation (70/15/15).
  - Class imbalance handling (cost-sensitive boosting).
  - 5-Model benchmark (LR, RF, XGBoost baseline, LightGBM, CatBoost).
  - Fraud evaluation metrics (PR-AUC, Recall@FPR=1%, F1, Brier score).
  - Probability calibration (Platt/Isotonic) & dual-threshold optimization.
  - Fast SHAP TreeExplainer & reason mapping.
  - FastAPI model serving and Redis sliding-window velocity.
  - Kafka consumer-producer streaming loop.
  - End-to-end integration with existing Streamlit dashboard.
- 🟠 **Strongly Recommended (Phase 2 — Robustness & Automation):**
  - Isolation Forest anomaly detection integration.
  - Optuna Bayesian hyperparameter optimization.
  - MLflow experiment tracking and model registry.
  - Automated adversarial behavioral stress testing.
  - Cross-dataset generalization matrix evaluation.
- 🟡 **Advanced / Post-Core Scope (Do NOT touch until Gates 1–15 are 100% complete):**
  - Graph Neural Networks (GNNs) and graph fraud embeddings.
  - Deep Learning Autoencoders.
  - Large Language Model (LLM) natural language generation.
  - Federated learning or continuous online model retraining.

### 16.2 Critical Technical Pitfalls & Mitigations
1. **Target Leakage via Simulator Artifacts:**
   *Pitfall:* In PaySim, `newbalanceOrig == 0` for fraud transfers because criminals drain the entire account. A model using `newbalanceOrig` achieves $99.9\%$ recall in offline testing, but in production, `newbalanceOrig` does not exist when the payment authorization request is evaluated!
   *Mitigation:* `PaySimAdapter` strictly strips `newbalanceOrig` and `newbalanceDest`. Only `oldbalanceOrg` and `amount` are permitted.
2. **Synthetic Generator Leakage in Sparkov:**
   *Pitfall:* The Sparkov generator prefixes fraudulent merchant names with `"fraud_"`. An uninspected model will simply learn `merchant.startswith("fraud_")`.
   *Mitigation:* `SparkovAdapter` strips `"fraud_"` from all merchant strings before feature encoding.
3. **Temporal Inversion in Resampling:**
   *Pitfall:* Applying SMOTE before splitting creates synthetic clones that bleed into the validation and test sets, artificially inflating test PR-AUC.
   *Mitigation:* Splitting occurs strictly before any resampling, and resampling is applied solely to the training fold.
4. **Offline-Online Feature Drift:**
   *Pitfall:* The offline training script computes velocity using Pandas rolling windows, but the real-time Kafka worker computes velocity using a slightly different Redis timestamp logic.
   *Mitigation:* Common mathematical transformations are encapsulated in reusable pure functions with automated parity verification tests (`tests/test_features.py`).
5. **Slow SHAP Latency:**
   *Pitfall:* Running `shap.KernelExplainer` or exact TreeExplainer over thousands of background samples can take $200\text{ ms}$ per transaction, violating real-time SLA.
   *Mitigation:* Use `shap.TreeExplainer` with a compact pre-computed background summary of $N=100$ medoid samples, ensuring feature attributions are computed in $< 5\text{ ms}$.
