"""
FinPulse R7-I — Retraining CLI & Automated Promotion Gate Entry Point.
"""

import os
import sys
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from training.retrain import ModelPromotionGate, run_synthetic_retraining_cycle


def main():
    parser = argparse.ArgumentParser(description="FinPulse R7 Model Retraining & Promotion CLI")
    parser.add_argument("--candidate-version", default="finpulse-v4-candidate", help="Version identifier for new candidate")
    parser.add_argument("--force", action="store_true", help="Force promotion (bypass gate - testing only)")
    args = parser.parse_args()

    print("===================================================================")
    print("      FinPulse R7: Model Retraining & Promotion Gate Execution      ")
    print("===================================================================")

    prod_m, cand_m, report = run_synthetic_retraining_cycle()

    print(f"\nProduction Version: {report.production_version}")
    print(f"Candidate Version:  {report.candidate_version}")
    print(f"Evaluation Timestamp: {report.timestamp}")
    print("-------------------------------------------------------------------")
    print("Gate Checks:")
    for c in report.checks:
        status_sym = "[PASSED]" if c.passed else "[FAILED]"
        print(f"  {status_sym} {c.name}: {c.details} (Rule: {c.threshold_condition})")

    print("-------------------------------------------------------------------")
    print(f"Overall Gate Decision: {report.overall_status}")
    print(f"Summary: {report.summary_reason}")
    print("===================================================================")

    if report.overall_status != "APPROVED" and not args.force:
        sys.exit(1)


if __name__ == "__main__":
    main()
