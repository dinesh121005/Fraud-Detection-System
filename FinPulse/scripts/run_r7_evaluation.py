"""
FinPulse R7 — Master Evaluation & Verification Runner.
Executes E1–E10 suite and generates reports/r7_evaluation_report.json.
"""

import os
import sys
import json
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.evaluation.suite_e1_e10 import FullSystemEvaluationSuite


def main():
    print("===================================================================")
    print("      FinPulse R7: Full Product Workflow & Lifecycle Evaluation    ")
    print("                     Executing Experiments E1–E10                  ")
    print("===================================================================")

    suite = FullSystemEvaluationSuite()
    report = suite.run_all()

    reports_dir = os.path.join(FINPULSE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, "r7_evaluation_report.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nExecution Duration: {report['total_duration_seconds']}s")
    print(f"Overall Status: {report['overall_status']}")
    print(f"Experiments Passed: {report['passed_count']} / {report['experiments_count']}\n")

    print(f"{'ID':<6} {'Experiment Name':<45} {'Duration':<12} {'Status'}")
    print("-" * 75)
    for exp in report["experiments"]:
        status_str = f"[{exp['status']}]"
        print(f"{exp['id']:<6} {exp['name']:<45} {exp['duration_ms']:>8.2f} ms   {status_str}")

    print("\n-------------------------------------------------------------------")
    cf = report["cost_frontier_analysis"]
    print(f"Optimal Decision Frontier Threshold: Risk Score >= {cf['optimal_threshold']}")
    print(f"Minimum Estimated Operational Cost:   ${cf['minimum_cost_dollars']:,.2f}")
    print("===================================================================")
    print(f"Structured report persisted to: {out_path}\n")


if __name__ == "__main__":
    main()
