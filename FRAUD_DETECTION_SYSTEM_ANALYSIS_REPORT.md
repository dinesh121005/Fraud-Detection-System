# Comprehensive Technical Analysis: FinPulse Fraud Detection System

**Repository:** `Fraud-Detection-System` / `FinPulse`  
**Analysis Date:** 2026-09-29  
**Audit Scope:** Source Code, Configuration, Models, Streaming Scripts, Data Pipelines, and Architecture  

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Current Architecture](#2-current-architecture)
3. [End-to-End Workflow](#3-end-to-end-workflow)
4. [Fraud Detection Logic](#4-fraud-detection-logic)
5. [Models and Rules Used](#5-models-and-rules-used)
6. [Actual Outputs and Outcomes](#6-actual-outputs-and-outcomes)
7. [Kafka and Real-Time Processing](#7-kafka-and-real-time-processing)
8. [Dashboard and Monitoring](#8-dashboard-and-monitoring)
9. [How It Can Help in a Real-Time Fraud System](#9-how-it-can-help-in-a-real-time-fraud-system)
10. [What Is Actually Implemented vs Simulated or Missing](#10-what-is-actually-implemented-vs-simulated-or-missing)
11. [Technical Limitations](#11-technical-limitations)
12. [Example Real-Time Transaction Flow](#12-example-real-time-transaction-flow)
13. [Current System vs Production-Ready Fraud Platform](#13-current-system-vs-production-ready-fraud-platform)
14. [Overall Technical Summary](#14-overall-technical-summary)

---

## 1. Project Overview

The project in this repository is titled **"FinPulse AI — Fraud Intelligence Operations"** (residing within the [`FinPulse/`](file:///d:/Fraud-Detection-System/FinPulse) directory). 

The repository implements a **Streamlit-driven proof-of-concept (PoC) fraud scoring system** combining a pre-trained **XGBoost gradient-boosted classification model** with an **additive heuristic rule engine**. The system is built to simulate financial transactions, evaluate them in real time across machine learning and heuristic dimensions, and present threat metrics through an interactive executive dashboard.

### Repository File Structure
```
Fraud-Detection-System/
├── README.md                      # Empty placeholder (1 line: "# Fraud-Detection-System")
├── .gitignore                     # Ignores venvs, cache, kafka binaries, *.csv, data/
└── FinPulse/
    ├── app.py                     # Primary Streamlit dashboard application (522 lines)
    ├── consumer.py                # Byte-for-byte duplicate of app.py (not a Kafka consumer)
    ├── docker-compose.yml         # Malformed: Contains HTML for a snack shop ("MPR Snacks")
    ├── requirements.txt           # Python dependencies (missing kafka-python)
    ├── producer.py                # Standalone dummy Kafka producer (24 lines)
    ├── kafka_consumer.py          # Standalone demo consumer for LAN IP '10.121.50.11' (16 lines)
    ├── models/
    │   ├── fraud_model.json       # XGBoost binary classification model (3.8 MB)
    │   ├── fraud_model.pkl        # Pickled XGBoost model (753 KB)
    │   └── label_encoder.pkl      # Scikit-learn LabelEncoder (5 transaction types)
    ├── scripts/
    │   ├── train_model.py         # Training pipeline on PaySim + synthetic attack augmentation
    │   └── predict.py             # CLI inference and prototype Kafka consumer-producer loop
    └── utils/
        ├── preprocessing.py       # Feature alignment, filling, and ratio engineering
        ├── helpers.py             # Rule-based heuristic scoring and explanation engine
        └── simulation.py          # Synthetic transaction and attack generator
```

---

## 2. Current Architecture

The actual architecture implemented in the codebase is a **monolithic local simulation running in-process within Streamlit**. External Kafka scripts exist as disjointed prototypes and do not form an active, end-to-end streaming data pipeline.

```
+---------------------------------------------------------------------------------------------------+
|                                 STREAMLIT APPLICATION (app.py)                                    |
|                                                                                                   |
|  +--------------------------------+       +----------------------------------------------------+  |
|  | Simulation / Input Generator   | ----> | In-Memory Preprocessing & Scoring (Synchronous)    |  |
|  | - Preset attack scenarios      |       | 1. Label Encoding (le.transform)                   |  |
|  | - Manual parameter sliders    |       | 2. Feature Engineering (utils/preprocessing.py)    |  |
|  | - Continuous batch simulator   |       | 3. XGBoost Inference (model.predict_proba)         |  |
|  +--------------------------------+       | 4. Heuristic Rule Engine (compute_heuristic_risk)  |  |
|                                           | 5. Fusion Decision Logic (OR thresholding)         |  |
|                                           +----------------------------------------------------+  |
|                                                                     |                             |
|                                                                     v                             |
|  +---------------------------------------------------------------------------------------------+  |
|  | Presentation & Session State (st.session_state.transactions_history)                        |  |
|  | - Executive KPI Cards (Scanned Volume, Threats Blocked, Clean Volume, Avg Risk Index)       |  |
|  | - Threat Inspector Banner (Triggered Risk Indicators)                                       |  |
|  | - Visualizations: Donut Verdict Chart, Bar Volume Chart, Real-Time Anomaly Timeline        |  |
|  | - Live Streaming Transaction Audit Ledger (Interactive Styled DataFrame)                   |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                      |                                                            |
|                                      v (Optional / Best Effort Fire-and-Forget)                   |
|                        KafkaProducer.send("transactions", tx)                                     |
+--------------------------------------|------------------------------------------------------------+
                                       |
                                       v (If Kafka broker on localhost:9092 is active)
                 +---------------------------------------------+
                 | Kafka Topic: "transactions"                 |
                 +---------------------------------------------+
                                       |
               +-----------------------+-----------------------+
               | (Disconnected / Unused)                       | (CLI Mode Only)
               v                                               v
    [kafka_consumer.py]                            [scripts/predict.py --kafka]
    - Listens to: 'test-topic'                     - Listens to: 'transactions'
    - Hardcoded IP: '10.121.50.11:9092'            - Runs predict_fraud()
    - Action: print(message.value)                 - Emits to: 'predictions'
    - No ML / No Fraud Logic                       - (Nothing consumes 'predictions')
```

---

## 3. End-to-End Workflow

Tracing an incoming transaction through [`FinPulse/app.py`](file:///d:/Fraud-Detection-System/FinPulse/app.py#L243-L370):

1. **Transaction Ingestion / Generation**:
   - The user inputs parameters manually, selects a preset (e.g., *"Attack: Large Amount Exceeding Balance"*, *"Attack: Cash-Out from Zero-Balance Account"*, *"Legit: Small Payment"*), or enables the batch stream generator via [`generate_transaction()`](file:///d:/Fraud-Detection-System/FinPulse/utils/simulation.py#L10-L55).
2. **Type Encoding**:
   - The categorical field `type` (`TRANSFER`, `CASH_OUT`, `PAYMENT`, `DEBIT`, `CASH_IN`) is mapped to an integer using the fitted [`label_encoder.pkl`](file:///d:/Fraud-Detection-System/FinPulse/models/label_encoder.pkl) ([`app.py:L247-L250`](file:///d:/Fraud-Detection-System/FinPulse/app.py#L247-L250)).
3. **Feature Preprocessing & Enrichment**:
   - Executed by [`preprocess_transaction(tx)`](file:///d:/Fraud-Detection-System/FinPulse/utils/preprocessing.py#L4-L34):
     - Calculates balance ratio: $\text{amount\_balance\_ratio} = \frac{\text{amount}}{\text{oldbalanceOrg} + 10^{-6}}$.
     - Computes indicator flags: `unusual_device = (device_known == 0)`, `unusual_location = (geo_known == 0)`, `initiated_by_third = (initiated_by == "third_party")`.
     - Fills missing values with `0` and sorts the vector into the 15-feature sequence expected by the XGBoost model.
4. **Machine Learning Inference**:
   - The preprocessed DataFrame is evaluated by `model.predict(X)` and `model.predict_proba(X)` using the pre-loaded XGBoost model ([`app.py:L256-L260`](file:///d:/Fraud-Detection-System/FinPulse/app.py#L256-L260)).
   - Produces raw binary prediction `isFraud` $\in \{0, 1\}$ and confidence score `Fraud_Prob` $\in [0.0, 1.0]$.
5. **Rule-Based Heuristic Evaluation**:
   - Executed via [`compute_heuristic_risk(tx)`](file:///d:/Fraud-Detection-System/FinPulse/utils/helpers.py#L10-L56) and [`explain_heuristic(tx)`](file:///d:/Fraud-Detection-System/FinPulse/utils/helpers.py#L58-L88).
   - Produces a floating-point score `heuristic_risk` $\in [0.0, 1.0]$ and a list of human-readable trigger strings `reasons`.
6. **Decision Fusion**:
   - Evaluates:
     $$\text{isFraudFinal} = \begin{cases} 1 & \text{if } \text{isFraud} = 1 \text{ or } \text{heuristic\_risk} > 0.5 \\ 0 & \text{otherwise} \end{cases}$$
   - Generates status: `"🚨 BLOCKED"` if $\text{isFraudFinal} = 1$, else `"✅ APPROVED"` ([`app.py:L271-L272`](file:///d:/Fraud-Detection-System/FinPulse/app.py#L271-L272)).
7. **Kafka Broadcast (Side Effect)**:
   - If a Kafka broker is reachable at `localhost:9092`, the fully processed transaction dictionary (including prediction, probabilities, heuristic scores, and verdicts) is published to the `transactions` topic ([`app.py:L347-L351`](file:///d:/Fraud-Detection-System/FinPulse/app.py#L347-L351)). If unreachable, the exception is caught silently.
8. **UI & State Update**:
   - The transaction is appended to `st.session_state.transactions_history`. All Streamlit metrics, summary charts, and ledger tables re-render synchronously.

---

## 4. Fraud Detection Logic

The system utilizes a **dual-track detection mechanism**: an XGBoost supervised model operating alongside an additive rule heuristic.

```
                       +-----------------------------+
                       |    Incoming Transaction     |
                       +-----------------------------+
                                      |
                      +---------------+---------------+
                      |                               |
                      v                               v
         +--------------------------+    +--------------------------+
         |     XGBoost Classifier   |    |  Heuristic Rule Engine   |
         |  (models/fraud_model.*)  |    |    (utils/helpers.py)    |
         +--------------------------+    +--------------------------+
                      |                               |
                      | Fraud_Prob >= 0.5             | heuristic_risk > 0.5
                      v                               v
              isFraud (0 or 1)               heuristic_score (0.0 - 1.0)
                      \                              /
                       \                            /
                        v                          v
                     +--------------------------------+
                     |    Decision Fusion Rule        |
                     |  isFraud == 1 OR risk > 0.5    |
                     +--------------------------------+
                                     |
                         +-----------+-----------+
                         |                       |
                         v                       v
                  [🚨 BLOCKED]            [✅ APPROVED]
```

### Feature Vector (15 Features)
The XGBoost model expects the following 15 ordered features ([`utils/preprocessing.py:L18-L24`](file:///d:/Fraud-Detection-System/FinPulse/utils/preprocessing.py#L18-L24)):
1. `step`: Integer timeline step / tick.
2. `type`: Integer-encoded transaction type (`0: CASH_IN`, `1: CASH_OUT`, `2: DEBIT`, `3: PAYMENT`, `4: TRANSFER`).
3. `amount`: Transaction amount.
4. `oldbalanceOrg`: Sender initial balance.
5. `newbalanceOrig`: Sender updated balance.
6. `oldbalanceDest`: Receiver initial balance.
7. `newbalanceDest`: Receiver updated balance.
8. `amount_balance_ratio`: Derived ratio: $\text{amount} / (\text{oldbalanceOrg} + 10^{-6})$.
9. `unusual_device`: Binary flag ($1$ if `device_known == 0`, else $0$).
10. `unusual_location`: Binary flag ($1$ if `geo_known == 0`, else $0$).
11. `sender_hourly_tx_count`: Velocity counter for sender in current hour.
12. `sender_daily_tx_count`: Velocity counter for sender in current day.
13. `receiver_risk_score`: Historical risk score of recipient ($0.0$ to $1.0$).
14. `initiated_by_third`: Binary flag ($1$ if `initiated_by == "third_party"`, else $0$).
15. `auth_verified`: Binary flag ($1$ if 2FA/biometrics verified, else $0$).

### Training Pipeline ([`scripts/train_model.py`](file:///d:/Fraud-Detection-System/FinPulse/scripts/train_model.py))
- **Baseline Dataset**: Kaggle PaySim mobile money dataset (`onlinefraud.csv`).
- **Data Augmentation**: Because PaySim lacks behavioral context, the script injects synthetic features:
  - `auth_verified`: Bernoulli ($p=0.95$).
  - `device_known`, `geo_known`: Bernoulli ($p=0.90$).
  - `initiated_by`: Categorical ($98\%$ sender, $2\%$ third-party).
  - `receiver_risk_score`: Uniform random in $[0, 1]$.
  - `sender_hourly_tx_count`, `sender_daily_tx_count`: Poisson distributions ($\lambda=1, \lambda=2$).
- **Synthetic Attack Injection**: 2% of the dataset is forcibly injected with fraudulent patterns:
  - `auth_verified = 0`, `initiated_by = "third_party"`, `device_known = 0`, `geo_known = 0`.
  - `amount = oldbalanceOrg + uniform(1, 5000)`.
  - `receiver_risk_score` boosted by $0.2 - 0.8$.
  - `isFraud = 1`.
- **Model Training**: `XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42)` fitted on an 80/20 stratified split.

---

## 5. Models and Rules Used

### 1. Machine Learning Model
- **Algorithm**: XGBoost (`XGBClassifier`) with 200 estimators.
- **Model Files**: 
  - [`models/fraud_model.json`](file:///d:/Fraud-Detection-System/FinPulse/models/fraud_model.json) (3.8 MB, standard JSON format).
  - [`models/fraud_model.pkl`](file:///d:/Fraud-Detection-System/FinPulse/models/fraud_model.pkl) (753 KB, Joblib pickle).
  - [`models/label_encoder.pkl`](file:///d:/Fraud-Detection-System/FinPulse/models/label_encoder.pkl) (Scikit-learn encoder for transaction categories).
- **Default Cutoff Threshold**: $0.50$ (in both [`scripts/predict.py:L45`](file:///d:/Fraud-Detection-System/FinPulse/scripts/predict.py#L45) and [`app.py:L256`](file:///d:/Fraud-Detection-System/FinPulse/app.py#L256)).

### 2. Rule-Based Scoring Engine ([`utils/helpers.py:L10-L56`](file:///d:/Fraud-Detection-System/FinPulse/utils/helpers.py#L10-L56))
The heuristic score starts at `0.0` and accumulates penalties up to a ceiling of `1.0`:

| Rule Condition | Risk Increment | Trigger Explanation Generated |
| :--- | :--- | :--- |
| `amount > oldbalanceOrg` (when `oldbalanceOrg > 0`) | $+ \min(0.4, \text{ratio} \times 0.1)$ | *N/A in explanation list* |
| `oldbalanceOrg == 0` and `amount > 0` | $+ 0.3$ | `"Large amount sent from zero balance account"` |
| `type` in `["CASH_OUT", "TRANSFER"]` | $+ 0.2$ | *N/A in explanation list* |
| `auth_verified == 0` | $+ 0.2$ | `"Auth not verified (OTP/biometric missing)"` |
| `device_known == 0` | $+ 0.1$ | `"Unknown device for sender"` |
| `geo_known == 0` | $+ 0.1$ | `"Unknown geo location for sender"` |
| `sender_hourly_tx_count > 5` | $+ 0.1$ | `"Sender exceeded hourly transaction count"` |
| `sender_daily_tx_count > 20` | $+ 0.1$ | `"Sender exceeded daily transaction count"` |
| `receiver_risk_score > 0.7` | $+ 0.1$ | `"Receiver has high risk score"` |

### 3. Pipeline Inconsistency
- In [`app.py`](file:///d:/Fraud-Detection-System/FinPulse/app.py#L271), the final verdict is an **OR fusion**: `isFraud == 1` OR `heuristic_risk > 0.5`.
- In [`scripts/predict.py`](file:///d:/Fraud-Detection-System/FinPulse/scripts/predict.py#L45), the heuristic engine is **omitted entirely**. Only the XGBoost probability `prob >= 0.5` is evaluated.

---

## 6. Actual Outputs and Outcomes

| Output | Type / Range | Calculation Method / Source | Component | Implementation Status |
| :--- | :--- | :--- | :--- | :--- |
| **`isFraud`** | Binary (`0` or `1`) | `model.predict(X)[0]` or `int(prob >= 0.5)` | [`app.py`](file:///d:/Fraud-Detection-System/FinPulse/app.py#L256), [`predict.py`](file:///d:/Fraud-Detection-System/FinPulse/scripts/predict.py#L45) | **Genuinely Implemented** (runs against trained model) |
| **`Fraud_Prob`** | Float (`0.0000` - `1.0000`) | `model.predict_proba(X)[0][1]` | [`app.py`](file:///d:/Fraud-Detection-System/FinPulse/app.py#L258), [`predict.py`](file:///d:/Fraud-Detection-System/FinPulse/scripts/predict.py#L48) | **Genuinely Implemented** |
| **`heuristic_risk`** | Float (`0.00` - `1.00`) | Additive rule summation capped at 1.0 | [`utils/helpers.py`](file:///d:/Fraud-Detection-System/FinPulse/utils/helpers.py#L10) | **Genuinely Implemented** |
| **`isFraudFinal`** | Binary (`0` or `1`) | `1 if (isFraud == 1 or heuristic_risk > 0.5) else 0` | [`app.py:L271`](file:///d:/Fraud-Detection-System/FinPulse/app.py#L271) | **Genuinely Implemented** |
| **`status`** | Categorical | `"🚨 BLOCKED"` if `isFraudFinal == 1`, else `"✅ APPROVED"` | [`app.py:L272`](file:///d:/Fraud-Detection-System/FinPulse/app.py#L272) | **Genuinely Implemented** |
| **`reasons`** | List of strings | Pattern matching against rules | [`utils/helpers.py:L58`](file:///d:/Fraud-Detection-System/FinPulse/utils/helpers.py#L58) | **Genuinely Implemented** |
| **Threat Banner** | HTML Banner | Color-coded alert box showing verdict, probabilities, and triggers | [`app.py:L406-L422`](file:///d:/Fraud-Detection-System/FinPulse/app.py#L406-L422) | **Genuinely Implemented** |
| **KPI Metrics** | Numerical aggregates | `sum()`, `mean()`, and counts across `st.session_state` | [`app.py:L374-L395`](file:///d:/Fraud-Detection-System/FinPulse/app.py#L374-L395) | **Genuinely Implemented** (ephemeral in memory) |
| **Visual Charts** | Altair chart objects | Donut verdict split, volume bar chart, probability timeline | [`app.py:L436-L482`](file:///d:/Fraud-Detection-System/FinPulse/app.py#L436-L482) | **Genuinely Implemented** |
| **Audit Ledger** | Styled Table | DataFrame styling highlighting fraud rows in red | [`app.py:L489-L521`](file:///d:/Fraud-Detection-System/FinPulse/app.py#L489-L521) | **Genuinely Implemented** |
| **Kafka Event** | JSON string | Serialized transaction dictionary published to Kafka | [`app.py:L349`](file:///d:/Fraud-Detection-System/FinPulse/app.py#L349), [`predict.py:L77`](file:///d:/Fraud-Detection-System/FinPulse/scripts/predict.py#L77) | **Partially Implemented / Demo** (fire-and-forget, unvalidated downstream) |

---

## 7. Kafka and Real-Time Processing

The repository references Kafka across several scripts, but **the Kafka implementation does not constitute a functioning real-time streaming pipeline**.

### Code-Level Findings:
1. **`docker-compose.yml` is Invalid**:
   - The file [`docker-compose.yml`](file:///d:/Fraud-Detection-System/FinPulse/docker-compose.yml) in `FinPulse/` does not configure Zookeeper, Kafka brokers, or schema registries. Instead, it contains **223 lines of static HTML/Tailwind code for a commercial snack website ("MPR Snacks - Authentic Kovilpatti Mittai")**. Running `docker compose up` fails immediately.
2. **Missing Dependencies**:
   - [`requirements.txt`](file:///d:/Fraud-Detection-System/FinPulse/requirements.txt) lists `streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`, `shap`, `lime`. **Neither `kafka-python` nor `confluent-kafka` is listed**.
3. **Payload Incompatibility**:
   - [`producer.py`](file:///d:/Fraud-Detection-System/FinPulse/producer.py#L16-L20) publishes messages with only 3 fields:
     ```json
     {"user_id": 4821, "amount": 1250.50, "type": "TRANSFER"}
     ```
   - However, the preprocessing script [`preprocessing.py`](file:///d:/Fraud-Detection-System/FinPulse/utils/preprocessing.py) and model expect 15 features, including balances and behavioral flags. Consuming this payload in `predict.py` causes missing fields to default to `0`, skewing the features.
4. **Hardcoded IP in `kafka_consumer.py`**:
   - [`kafka_consumer.py`](file:///d:/Fraud-Detection-System/FinPulse/kafka_consumer.py#L5-L10) connects to `'10.121.50.11:9092'` with comment `"# IP of Laptop3"`. It only prints incoming strings to stdout without performing fraud checks.
5. **Inverted Architecture in `app.py`**:
   - The Streamlit application acts as a Kafka *producer*, not a consumer. It scores transactions internally in memory and publishes the completed result to `localhost:9092` purely as an optional side-effect ([`app.py:L346-L351`](file:///d:/Fraud-Detection-System/FinPulse/app.py#L346-L351)). If Kafka is offline, it falls back silently to local memory.
6. **Dead-End Output Topic**:
   - When run with `--kafka`, [`predict.py`](file:///d:/Fraud-Detection-System/FinPulse/scripts/predict.py#L73-L79) consumes `'transactions'` and publishes to `'predictions'`. No downstream service or consumer ever reads from `'predictions'`.

---

## 8. Dashboard and Monitoring

The Streamlit UI in [`FinPulse/app.py`](file:///d:/Fraud-Detection-System/FinPulse/app.py) provides a single-page monitoring console:

### UI Capabilities:
1. **Simulation Sidebar Controls**:
   - **Preset Scenarios**: Dropdown to test known attack patterns (exceeding balance, zero-balance cash-out), legitimate payments, or random generators.
   - **Manual Sliders**: Direct manipulation of transaction attributes (amounts, balances, 2FA status, device status, geolocation status, velocity counters).
   - **Batch Stream Generator**: Triggers automated batches of 1 to 50 transactions with configurable inter-arrival delays.
2. **Executive Metric Cards**:
   - *Total Scanned Volume*: Cumulative volume in dollars and transaction count.
   - *Threats Intercepted*: Dollar volume blocked, count blocked, and percentage fraud rate.
   - *Safe Volume Cleared*: Dollar volume approved and clean count.
   - *Avg Heuristic Risk Index*: Mean heuristic risk score across the session ($0.0 - 1.0$).
3. **Threat Intelligence Inspector (Alert Banner)**:
   - Displays a dynamic HTML alert box for the most recent transaction.
   - Shows the final verdict (`🚨 THREAT INTERCEPTED` vs. `✅ TRANSACTION VERIFIED`), ML probability percentage, heuristic score, and individual warning pills for each triggered heuristic rule.
4. **Analytics Tab**:
   - Donut chart of Approved vs. Blocked counts.
   - Bar chart of transaction volume segmented by transaction category.
   - Dual-line timeline tracking ML probability and Heuristic index across the transaction sequence.
5. **Transaction Ledger Tab**:
   - Interactive table detailing all processed transactions with red styling applied to flagged rows.

*State Management Note:* All metrics and ledger entries are stored in `st.session_state.transactions_history`. Refreshing the browser clears all recorded history.

---

## 9. How It Can Help in a Real-Time Fraud System

While not yet production-ready, the repository contains viable foundational patterns for real-time fraud detection:

1. **Dual-Layered Hybrid Decision Strategy**:
   - Blending an ML classifier (XGBoost) with an explicit, explainable rule engine reflects industry best practices. Static rules guard against high-risk structural anomalies (e.g., zero-balance cash-outs), while ML captures non-linear combinations across multi-dimensional features.
2. **Explainability & Reason Codes**:
   - The [`explain_heuristic()`](file:///d:/Fraud-Detection-System/FinPulse/utils/helpers.py#L58-L88) function illustrates how automated decisions can supply human fraud analysts with actionable reason codes (e.g., *"Auth not verified"*, *"Sender exceeded hourly transaction count"*) alongside risk probabilities.
3. **Interactive What-If Simulation**:
   - The dashboard serves as an effective demonstration console for risk officers, stakeholders, and product teams to test what-if scenarios (e.g., observing how unchecking 2FA or altering transaction velocity impacts the risk score).
4. **Feature Engineering Precedents**:
   - The transformations in [`utils/preprocessing.py`](file:///d:/Fraud-Detection-System/FinPulse/utils/preprocessing.py) (balance ratios, device familiarity indicators) illustrate the baseline features required in an online transaction processing (OLTP) pipeline.

---

## 10. What Is Actually Implemented vs Simulated or Missing

| Capability / Component | Actual Code Status | Details / Evidence from Repository |
| :--- | :--- | :--- |
| **XGBoost Inference** | **Implemented** | Loads JSON/PKL model and produces calibrated class probabilities. |
| **Rule-Based Heuristic Engine** | **Implemented** | Computes additive penalties and returns human-readable reason codes. |
| **Hybrid Decision Fusion** | **Implemented** | Evaluates union of ML and rule thresholds in `app.py`. |
| **Feature Preprocessing** | **Implemented** | Cleans, calculates ratios, handles missing keys, and orders features. |
| **Streamlit Interactive UI** | **Implemented** | Complete executive dashboard with Altair visualizations and styled ledger. |
| **Real Transaction Ingestion** | **Simulated** | No HTTP REST endpoint, webhook, or payment gateway connector exists. Driven via UI sliders and random generation. |
| **Feature State Stores / Aggregations** | **Simulated** | Velocity counters (`sender_hourly_tx_count`, `sender_daily_tx_count`) are manual inputs or random integers; no Redis/database tracks historical velocities. |
| **Kafka Streaming Pipeline** | **Simulated / Broken** | `docker-compose.yml` is HTML; Kafka consumer has a hardcoded LAN IP; producer message schema doesn't match model schema; `app.py` doesn't consume Kafka. |
| **Consumer Service (`consumer.py`)** | **Missing / Duplicate** | File is an identical copy of `app.py`, not an autonomous background worker. |
| **Persistent Storage / Database** | **Missing** | No SQL/NoSQL database integration. All transaction records reside in ephemeral `st.session_state`. |
| **Authentication & Access Control** | **Missing** | No user management, role-based access control (RBAC), or session security. |
| **Model Monitoring & Drift Detection** | **Missing** | No tracking of feature drift, concept drift, ground-truth feedback, or latency metrics. |
| **Automated Testing Suite** | **Missing** | Zero unit tests, integration tests, or end-to-end test cases in the repository. |

---

## 11. Technical Limitations

1. **Ephemeral In-Memory State**:
   - Transaction history is held entirely in Streamlit session state (`st.session_state`). Any page reload, network disconnect, or process restart destroys all historical ledger entries and KPI metrics.
2. **Offline Data Leakage & Synthetic Features**:
   - As implemented in [`scripts/train_model.py:L45-L78`](file:///d:/Fraud-Detection-System/FinPulse/scripts/train_model.py#L45-L78), contextual features (`auth_verified`, `device_known`, `geo_known`, etc.) were artificially injected using random distributions and 2% synthetic attacks. The model has not been evaluated against real-world adversarial attacks or genuine production telemetry.
3. **Hardcoded Machine Paths & Network Addresses**:
   - [`scripts/predict.py:L7`](file:///d:/Fraud-Detection-System/FinPulse/scripts/predict.py#L7) hardcodes `sys.path.append(r"D:\ssn\FinPulse")`.
   - [`kafka_consumer.py:L6`](file:///d:/Fraud-Detection-System/FinPulse/kafka_consumer.py#L6) hardcodes `bootstrap_servers='10.121.50.11:9092'`.
4. **Static Thresholds**:
   - The decision thresholds ($0.5$ for ML probability and $0.5$ for heuristic risk) are hardcoded. Production fraud engines require dynamic thresholds conditioned on customer segments, transaction types, and risk tolerance.
5. **Synchronous Execution Latency**:
   - Running model inference and heuristic evaluation synchronously within the Streamlit UI thread introduces latency bottlenecks under high transaction concurrency.
6. **Corrupted Infrastructure File**:
   - The corrupted [`docker-compose.yml`](file:///d:/Fraud-Detection-System/FinPulse/docker-compose.yml) prevents local containerization or testing of Kafka brokers.

---

## 12. Example Real-Time Transaction Flow

### Scenario A: High-Risk Cash-Out Transaction
1. **Input Payload**:
   - Type: `CASH_OUT`, Amount: `$4,500.00`, Sender Initial Balance: `$0.00`.
   - `auth_verified = 0`, `device_known = 0`, `geo_known = 0`, `receiver_risk_score = 0.90`.
   - `sender_hourly_tx_count = 8`, `sender_daily_tx_count = 25`.
2. **Preprocessing**:
   - $\text{amount\_balance\_ratio} = 4500 / (0 + 10^{-6}) = 4,500,000,000.0$.
   - Flags set: `unusual_device = 1`, `unusual_location = 1`, `initiated_by_third = 1`, `auth_verified = 0`.
3. **Scoring**:
   - **XGBoost Inference**: Evaluates tree ensemble $\to$ `Fraud_Prob = 0.9982` $\to$ `isFraud = 1`.
   - **Heuristic Engine**:
     - Zero balance check: $+0.30$
     - CASH_OUT type: $+0.20$
     - Unverified auth: $+0.20$
     - Unknown device: $+0.10$
     - Unknown geo: $+0.10$
     - Hourly velocity ($8 > 5$): $+0.10$
     - Daily velocity ($25 > 20$): $+0.10$
     - Receiver risk ($0.90 > 0.70$): $+0.10$
     - Raw sum: $1.20 \to$ Capped at `heuristic_risk = 1.00`.
4. **Decision Fusion**:
   - `isFraud == 1` and `heuristic_risk > 0.5` $\to$ `isFraudFinal = 1`.
   - Status: `"🚨 BLOCKED"`.
   - Triggered reasons: 7 distinct alert strings populated.

### Scenario B: Legitimate Low-Value Payment
1. **Input Payload**:
   - Type: `PAYMENT`, Amount: `$25.50`, Sender Initial Balance: `$500.00`.
   - `auth_verified = 1`, `device_known = 1`, `geo_known = 1`, `receiver_risk_score = 0.05`.
   - `sender_hourly_tx_count = 1`, `sender_daily_tx_count = 3`.
2. **Scoring**:
   - **XGBoost Inference**: `Fraud_Prob = 0.0014` $\to$ `isFraud = 0`.
   - **Heuristic Engine**: No penalty conditions met $\to$ `heuristic_risk = 0.00`.
3. **Decision Fusion**:
   - Neither condition met $\to$ `isFraudFinal = 0`.
   - Status: `"✅ APPROVED"`.

---

## 13. Current System vs Production-Ready Fraud Platform

| Feature Area | Current Implementation in Repository | Production-Ready Platform Standard |
| :--- | :--- | :--- |
| **Ingestion Layer** | Manual Streamlit inputs & local random generation functions | Resilient streaming connectors (Kafka, Flink, RabbitMQ) and low-latency REST/gRPC gateways |
| **State & Feature Store** | None; velocity and balances passed manually in input payload | Real-time distributed feature store (e.g., Feast, Redis) computing rolling velocity windows |
| **Streaming Engine** | Disjointed demo scripts (`producer.py`, `kafka_consumer.py`) | Managed event streaming cluster (Apache Kafka / AWS Kinesis) with consumer groups and partitioning |
| **Scoring Service** | Embedded synchronous Python execution inside Streamlit script | Decoupled, containerized microservice (e.g., FastAPI, Triton, BentoML) with sub-50ms p99 latency SLA |
| **Storage / Persistence** | Volatile `st.session_state` DataFrame | ACID transaction datastore (PostgreSQL) paired with analytical lakehouse (ClickHouse/Snowflake) |
| **Decision Rules** | Hardcoded if/else statements in [`utils/helpers.py`](file:///d:/Fraud-Detection-System/FinPulse/utils/helpers.py) | Dynamic Rule Engine (e.g., Drools, JSON rules engine) configurable by risk analysts without code deployment |
| **Alerting / Operations** | Streamlit UI banner and dataframe highlight | Case management system, webhook dispatch, SMS/email alerts, and analyst workflow queues |
| **Security & Compliance** | None (no auth, open access) | OAuth2/mTLS, audit trails, PII encryption at rest/in transit, PCI-DSS compliance |
| **CI/CD & DevOps** | Corrupted `docker-compose.yml` (HTML contents) | Multi-stage Dockerfiles, Helm charts, automated unit/integration test pipelines |

---

## 14. Overall Technical Summary

The repository represents a **working, well-styled local proof-of-concept for fraud scoring demonstration**, rather than a production-grade or deployable real-time fraud detection platform.

### Summary Assessment:
- **What Is Working**: The internal ML inference pipeline (using pre-trained XGBoost models) and the heuristic explanation engine are fully functional when executed within the Streamlit UI. The UI offers clear scenario presets, charts, and instant explainability tags.
- **What Is Simulated / Defective**: 
  - Kafka integration is non-functional for end-to-end streaming.
  - `consumer.py` is an accidental duplicate of `app.py`.
  - `docker-compose.yml` contains unrelated HTML code.
  - Historical context (velocity counters and past device records) are not calculated from a database; they are generated on the fly.
- **Recommended Remediation Path**:
  1. Replace the corrupted `docker-compose.yml` with a standard Zookeeper + Kafka broker definition.
  2. Implement a dedicated scoring microservice (e.g., FastAPI or a genuine Kafka consumer daemon).
  3. Integrate an in-memory cache (e.g., Redis) to compute stateful rolling velocity features (`sender_hourly_tx_count`, `sender_daily_tx_count`).
  4. Attach a persistent relational database (e.g., PostgreSQL) to record transaction ledgers and flagged events.
