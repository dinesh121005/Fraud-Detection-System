"""Run Dataset Audit CLI - Phase ML-1 / Gate 1 Deliverable."""
import os
import sys
import json
import yaml

# Add FinPulse root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.data.auditor import DatasetAuditor
from src.data.paysim_adapter import PaySimAdapter
from src.data.sparkov_adapter import SparkovAdapter
from src.data.ieee_adapter import IeeeCisAdapter

def main():
    print("=" * 80)
    print("  FINPULSE ML REDESIGN: PHASE ML-1 DATASET AUDIT (GATE 1)")
    print("=" * 80)

    # Load datasets config
    config_path = os.path.join(FINPULSE_DIR, "configs", "datasets.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)["datasets"]

    auditor = DatasetAuditor(output_dir=os.path.join(FINPULSE_DIR, "reports"))
    summary = {}

    # 1. Audit PaySim
    print("\n[1/3] Auditing PaySim Mobile Money Dataset...")
    paysim_adapter = PaySimAdapter(config["paysim"])
    df_paysim = paysim_adapter.load_raw(sample_size=100000)
    profile_paysim = auditor.audit_dataframe(df_paysim, "PaySim", "isFraud")
    auditor.generate_html_report(profile_paysim, "paysim_report.html")
    summary["paysim"] = profile_paysim
    print(f"  -> Records Sampled: {profile_paysim['total_rows']:,}")
    print(f"  -> Features: {profile_paysim['total_columns']}")
    print(f"  -> Fraud %: {profile_paysim['fraud_distribution']['fraud_percentage']}%")
    print(f"  -> Imbalance Ratio: {profile_paysim['fraud_distribution']['imbalance_ratio']}")

    # 2. Audit Sparkov
    print("\n[2/3] Auditing Sparkov Credit Card Dataset...")
    sparkov_adapter = SparkovAdapter(config["sparkov"])
    df_sparkov = sparkov_adapter.load_raw(split="train", sample_size=100000)
    profile_sparkov = auditor.audit_dataframe(df_sparkov, "Sparkov", "is_fraud")
    auditor.generate_html_report(profile_sparkov, "sparkov_report.html")
    summary["sparkov"] = profile_sparkov
    print(f"  -> Records Sampled: {profile_sparkov['total_rows']:,}")
    print(f"  -> Features: {profile_sparkov['total_columns']}")
    print(f"  -> Fraud %: {profile_sparkov['fraud_distribution']['fraud_percentage']}%")
    print(f"  -> Imbalance Ratio: {profile_sparkov['fraud_distribution']['imbalance_ratio']}")

    # 3. Audit IEEE-CIS
    print("\n[3/3] Auditing IEEE-CIS Transaction & Identity Dataset...")
    ieee_adapter = IeeeCisAdapter(config["ieee_cis"])
    df_ieee = ieee_adapter.load_raw(sample_size=100000)
    profile_ieee = auditor.audit_dataframe(df_ieee, "IEEE-CIS", "isFraud")
    auditor.generate_html_report(profile_ieee, "ieee_report.html")
    summary["ieee_cis"] = profile_ieee
    print(f"  -> Records Sampled: {profile_ieee['total_rows']:,}")
    print(f"  -> Features: {profile_ieee['total_columns']}")
    print(f"  -> Fraud %: {profile_ieee['fraud_distribution']['fraud_percentage']}%")
    print(f"  -> Imbalance Ratio: {profile_ieee['fraud_distribution']['imbalance_ratio']}")

    # Save summary json
    summary_path = os.path.join(FINPULSE_DIR, "reports", "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("  GATE 1 COMPLETION VERIFIED")
    print(f"  Reports saved to: {os.path.join(FINPULSE_DIR, 'reports')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
