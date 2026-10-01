"""CLI Entrypoint for Kafka Transaction Producer and Simulator."""
import os
import sys
import argparse

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.streaming.config import load_streaming_config
from src.streaming.producer import TransactionProducer, TransactionSimulator

def main():
    parser = argparse.ArgumentParser(description="FinPulse Real-Time Kafka Transaction Simulator & Producer")
    parser.add_argument("--rate", type=float, default=2.0, help="Transactions to emit per second (default: 2.0)")
    parser.add_argument("--count", type=int, default=None, help="Total transactions to emit (default: infinite)")
    parser.add_argument("--fraud-ratio", type=float, default=0.05, help="Ratio of simulated anomalies (default: 0.05)")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (buffers events without network I/O)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic simulation")
    parser.add_argument("--bootstrap-servers", type=str, default=None, help="Override Kafka bootstrap servers")
    args = parser.parse_args()

    config = load_streaming_config()
    if args.bootstrap_servers:
        config.bootstrap_servers = args.bootstrap_servers

    print("=" * 80)
    print("  FINPULSE REAL-TIME KAFKA PRODUCER & TRANSACTION SIMULATOR")
    print("=" * 80)
    print(f"Bootstrap Servers: {config.bootstrap_servers}")
    print(f"Target Topic:      {config.topics.inbound_transactions}")
    print(f"Emission Rate:     {args.rate} tx/sec")
    print(f"Planned Count:     {args.count or 'Infinite (Ctrl+C to stop)'}")
    print(f"Fraud Ratio:       {args.fraud_ratio * 100:.1f}%")
    print(f"Dry Run:           {args.dry_run}")
    print("-" * 80)

    producer = TransactionProducer(config=config, dry_run=args.dry_run)
    simulator = TransactionSimulator(seed=args.seed)

    try:
        total_sent = simulator.run_simulation(
            producer=producer,
            rate_per_sec=args.rate,
            max_events=args.count,
            fraud_ratio=args.fraud_ratio
        )
        print(f"\nSimulation finished. Published: {total_sent} transactions.")
    finally:
        producer.close()

if __name__ == "__main__":
    main()
