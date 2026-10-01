"""Automated Dataset Quality and Imbalance Auditor for FinPulse."""
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any

class DatasetAuditor:
    """Profiles datasets for missing values, duplicates, types, and class imbalance."""

    def __init__(self, output_dir: str = "d:/Fraud-Detection-System/FinPulse/reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def audit_dataframe(self, df: pd.DataFrame, dataset_name: str, target_col: str) -> Dict[str, Any]:
        """Generate a complete statistical health profile of a dataset."""
        n_rows = len(df)
        n_cols = len(df.columns)
        
        # Missing values
        missing_counts = df.isnull().sum()
        missing_pcts = (missing_counts / n_rows * 100).round(2)
        cols_with_missing = {
            col: {"count": int(missing_counts[col]), "percentage": float(missing_pcts[col])}
            for col in df.columns if missing_counts[col] > 0
        }

        # Duplicates
        duplicate_count = int(df.duplicated().sum())
        duplicate_pct = round((duplicate_count / n_rows) * 100, 2)

        # Types
        numerical_cols = list(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(df.select_dtypes(exclude=[np.number]).columns)

        # Target distribution
        if target_col in df.columns:
            target_counts = df[target_col].value_counts().to_dict()
            fraud_count = int(target_counts.get(1, 0))
            normal_count = int(target_counts.get(0, 0))
            fraud_pct = round((fraud_count / n_rows) * 100, 4)
            imbalance_ratio = round(normal_count / max(fraud_count, 1), 2)
        else:
            fraud_count, normal_count, fraud_pct, imbalance_ratio = 0, 0, 0.0, 0.0

        profile = {
            "dataset_name": dataset_name,
            "total_rows": n_rows,
            "total_columns": n_cols,
            "duplicate_rows": duplicate_count,
            "duplicate_percentage": duplicate_pct,
            "numerical_columns_count": len(numerical_cols),
            "categorical_columns_count": len(categorical_cols),
            "columns_with_missing_values_count": len(cols_with_missing),
            "missing_summary": cols_with_missing,
            "fraud_distribution": {
                "normal_transactions": normal_count,
                "fraud_transactions": fraud_count,
                "fraud_percentage": fraud_pct,
                "imbalance_ratio": f"{imbalance_ratio}:1"
            }
        }

        return profile

    def generate_html_report(self, profile: Dict[str, Any], filename: str):
        """Render a clean HTML audit report."""
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>FinPulse Audit: {profile['dataset_name']}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 30px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2); }}
        h1 {{ color: #38bdf8; font-size: 24px; margin-top: 0; }}
        h2 {{ color: #94a3b8; font-size: 18px; margin-top: 0; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 16px; }}
        .metric {{ background: #334155; padding: 16px; border-radius: 8px; border-left: 4px solid #38bdf8; }}
        .metric-title {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; }}
        .metric-val {{ font-size: 22px; font-weight: bold; margin-top: 4px; }}
        .alert {{ border-left-color: #ef4444; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>FinPulse Dataset Audit Report: {profile['dataset_name']}</h1>
        <h2>Quality, Cardinality, and Class Distribution Profile</h2>
        <div class="grid">
            <div class="metric"><div class="metric-title">Total Records</div><div class="metric-val">{profile['total_rows']:,}</div></div>
            <div class="metric"><div class="metric-title">Total Features</div><div class="metric-val">{profile['total_columns']}</div></div>
            <div class="metric"><div class="metric-title">Duplicates</div><div class="metric-val">{profile['duplicate_percentage']}%</div></div>
            <div class="metric"><div class="metric-title">Fraud Percentage</div><div class="metric-val">{profile['fraud_distribution']['fraud_percentage']}%</div></div>
            <div class="metric alert"><div class="metric-title">Imbalance Ratio</div><div class="metric-val">{profile['fraud_distribution']['imbalance_ratio']}</div></div>
        </div>
    </div>
</body>
</html>"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        return filepath
