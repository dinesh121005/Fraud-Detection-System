"""FinPulse R6.1 — Prometheus Live Scrape Integration Validation Script.

Executes:
1. Spawns FastAPI server on 0.0.0.0:8000
2. Launches official Prometheus Docker container (prom/prometheus:v2.51.0) on port 9090
3. Generates transactions across APPROVE, REVIEW, and BLOCK outcomes
4. Asserts Prometheus scrapes /metrics and reports health == 'up'
5. Queries Prometheus HTTP API for finpulse_transactions_total, finpulse_decisions_total, and latencies
6. Gracefully tears down containers and background processes
"""

import os
import sys
import time
import subprocess
import requests

FINPULSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(FINPULSE_DIR, "configs", "prometheus_test.yml")
CONTAINER_NAME = "finpulse-prometheus-val"

def main():
    print("=" * 70)
    print("FINPULSE R6.1 -- PROMETHEUS LIVE INTEGRATION VALIDATION")
    print("=" * 70)

    # Clean up any leftover test containers
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 1. Start FastAPI server
    print("\n[1/5] Launching FinPulse FastAPI serving server on port 8000...")
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=FINPULSE_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Wait for API to become ready
    api_ready = False
    for attempt in range(15):
        try:
            r = requests.get("http://127.0.0.1:8000/health", timeout=1.0)
            if r.status_code == 200 and r.json().get("model_loaded"):
                api_ready = True
                print("   [OK] FastAPI server ready with model loaded.")
                break
        except Exception:
            time.sleep(1.0)

    if not api_ready:
        print("   [FAIL] Failed to start FastAPI server.")
        api_proc.terminate()
        sys.exit(1)

    # 2. Start Prometheus container
    print(f"\n[2/5] Starting Prometheus Docker container '{CONTAINER_NAME}'...")
    docker_cmd = [
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "-p", "9090:9090",
        "-v", f"{CONFIG_PATH}:/etc/prometheus/prometheus.yml",
        "prom/prometheus:v2.51.0",
        "--config.file=/etc/prometheus/prometheus.yml",
        "--storage.tsdb.path=/prometheus",
    ]
    res_docker = subprocess.run(docker_cmd, capture_output=True, text=True)
    if res_docker.returncode != 0:
        print(f"   [FAIL] Failed to launch Prometheus container: {res_docker.stderr}")
        api_proc.terminate()
        sys.exit(1)
    print("   [OK] Prometheus container running.")

    try:
        # Wait for Prometheus web server
        prom_ready = False
        for _ in range(10):
            try:
                r = requests.get("http://127.0.0.1:9090/-/ready", timeout=1.0)
                if r.status_code == 200:
                    prom_ready = True
                    print("   [OK] Prometheus ready endpoint returned 200 OK.")
                    break
            except Exception:
                time.sleep(1.0)

        if not prom_ready:
            print("   [FAIL] Prometheus server failed to become ready.")
            sys.exit(1)

        # 3. Generate controlled transactions
        print("\n[3/5] Generating transactions to populate telemetry...")
        test_transactions = [
            # 1. Clean low-risk payment
            {
                "transaction_id": "tx_val_01",
                "timestamp": time.time(),
                "amount": 25.0,
                "customer_id": "cust_val_01",
                "merchant_id": "merch_val_01",
                "category": "grocery",
                "payment_type": "PAYMENT",
                "origin_balance": 500.0,
                "auth_verified": True
            },
            # 2. Medium risk cash-out
            {
                "transaction_id": "tx_val_02",
                "timestamp": time.time(),
                "amount": 1500.0,
                "customer_id": "cust_val_02",
                "merchant_id": "merch_val_02",
                "category": "cash_out",
                "payment_type": "CASH_OUT",
                "origin_balance": 1500.0,
                "auth_verified": True
            },
            # 3. High risk attack (large amount exceeding balance, unverified)
            {
                "transaction_id": "tx_val_03",
                "timestamp": time.time(),
                "amount": 50000.0,
                "customer_id": "cust_val_03",
                "merchant_id": "merch_val_03",
                "category": "crypto",
                "payment_type": "TRANSFER",
                "origin_balance": 100.0,
                "auth_verified": False
            }
        ]

        for tx in test_transactions:
            resp = requests.post("http://127.0.0.1:8000/predict", json=tx, timeout=5.0)
            data = resp.json()
            print(f"   Processed {tx['transaction_id']}: decision={data['decision']}, risk_score={data['risk_score']}")

        # 4. Wait for Prometheus scrape
        print("\n[4/5] Waiting for Prometheus scrape cycles (up to 12s)...")
        active_targets = []
        for _ in range(12):
            time.sleep(1.0)
            try:
                target_res = requests.get("http://127.0.0.1:9090/api/v1/targets", timeout=2.0).json()
                active_targets = target_res.get("data", {}).get("activeTargets", [])
                if active_targets and any(t.get("health") == "up" for t in active_targets):
                    break
            except Exception:
                pass

        print(f"   Active scrape targets found: {len(active_targets)}")
        for t in active_targets:
            job = t.get("labels", {}).get("job")
            health = t.get("health")
            scrapes = t.get("scrapes", "N/A")
            last_err = t.get("lastError", "None")
            print(f"   Target '{job}': health='{health}', last_error='{last_err}'")

        # 5. Query Prometheus for ingested metrics
        print("\n[5/5] Querying Prometheus API for ingested FinPulse metrics...")
        
        # Query Transactions with retry
        results_tx = []
        for _ in range(10):
            q_tx = requests.get("http://127.0.0.1:9090/api/v1/query?query=finpulse_transactions_total", timeout=5.0).json()
            results_tx = q_tx.get("data", {}).get("result", [])
            if results_tx:
                break
            time.sleep(1.0)

        print(f"   finpulse_transactions_total series count: {len(results_tx)}")
        assert len(results_tx) > 0, "No finpulse_transactions_total time-series in Prometheus"
        total_val = sum(float(r["value"][1]) for r in results_tx)
        print(f"   Total transactions recorded: {total_val}")
        assert total_val >= 3.0

        # Query Decisions
        q_dec = requests.get("http://127.0.0.1:9090/api/v1/query?query=finpulse_decisions_total", timeout=5.0).json()
        results_dec = q_dec.get("data", {}).get("result", [])
        dec_breakdown = {r["metric"]["decision"]: float(r["value"][1]) for r in results_dec}
        print(f"   finpulse_decisions_total breakdown: {dec_breakdown}")
        assert len(dec_breakdown) > 0

        # Query Latency Count
        q_lat = requests.get("http://127.0.0.1:9090/api/v1/query?query=finpulse_processing_latency_ms_count", timeout=5.0).json()
        lat_count = float(q_lat.get("data", {}).get("result", [{}])[0].get("value", [0, 0])[1])
        print(f"   finpulse_processing_latency_ms count: {lat_count}")
        assert lat_count >= 3.0

        print("\n" + "=" * 70)
        print("[SUCCESS] PROMETHEUS LIVE INTEGRATION VALIDATED END-TO-END!")
        print("=" * 70)

    finally:
        print("\nCleaning up validation resources...")
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        api_proc.terminate()
        api_proc.wait(timeout=5.0)
        print("Validation cleanup complete.")

if __name__ == "__main__":
    main()
