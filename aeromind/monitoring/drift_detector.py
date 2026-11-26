"""
Sensor Data Drift & Distribution Shift Detector for AeroMind Ops.

Monitors sensor distribution drift between reference (training fleet) and
current (operational / test fleet) telemetry. Uses:
  - Kolmogorov-Smirnov 2-sample tests (KS test)
  - Wasserstein Distance (Earth Mover's Distance)
  - Mean and standard deviation drift ratios

Outputs:
  - JSON audit report: data/processed/drift_report.json
  - Visual HTML Dashboard: docs/reports/data_drift_dashboard.html
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from aeromind.core.constants import SENSOR_COLUMNS
from aeromind.data.ingestion import load_raw_dataset

DEFAULT_P_VALUE_THRESHOLD = 0.05
DEFAULT_DATASET_DRIFT_THRESHOLD = 0.30  # Flag if >30% features drifted


class SensorDriftDetector:
    """
    Statistical distribution drift detector for turbofan telemetry streams.
    """

    def __init__(
        self,
        p_val_threshold: float = DEFAULT_P_VALUE_THRESHOLD,
        dataset_drift_threshold: float = DEFAULT_DATASET_DRIFT_THRESHOLD,
        output_dir: Path = Path("data/processed"),
        reports_dir: Path = Path("docs/reports"),
    ):
        self.p_val_threshold = p_val_threshold
        self.dataset_drift_threshold = dataset_drift_threshold
        self.output_dir = Path(output_dir)
        self.reports_dir = Path(reports_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.reference_data: Optional[pd.DataFrame] = None
        self.features_to_monitor: List[str] = []

    def fit_reference(self, reference_df: pd.DataFrame, features: Optional[List[str]] = None):
        """Sets the baseline reference dataset (typically training fleet)."""
        self.reference_data = reference_df.copy()
        if features:
            self.features_to_monitor = [f for f in features if f in reference_df.columns]
        else:
            # Monitored informative sensors by default
            active = [f"sensor_{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
            self.features_to_monitor = [f for f in active if f in reference_df.columns]

    def detect_drift(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates distribution drift between reference baseline and current batch.
        """
        if self.reference_data is None:
            raise ValueError("Reference dataset not fitted. Call fit_reference first.")

        metrics_by_feature = {}
        drifted_count = 0

        for col in self.features_to_monitor:
            if col not in current_df.columns:
                continue

            ref_vals = self.reference_data[col].dropna().values
            curr_vals = current_df[col].dropna().values

            if len(ref_vals) == 0 or len(curr_vals) == 0:
                continue

            # 1. Kolmogorov-Smirnov Test
            ks_res = ks_2samp(ref_vals, curr_vals)
            p_value = float(ks_res.pvalue)
            stat = float(ks_res.statistic)

            # 2. Wasserstein Distance
            w_dist = float(wasserstein_distance(ref_vals, curr_vals))

            # 3. Descriptive Stats Shift
            ref_mean = float(np.mean(ref_vals))
            curr_mean = float(np.mean(curr_vals))
            mean_shift_pct = float(abs(curr_mean - ref_mean) / (abs(ref_mean) + 1e-8) * 100.0)

            ref_std = float(np.std(ref_vals))
            curr_std = float(np.std(curr_vals))

            is_drifted = p_value < self.p_val_threshold
            if is_drifted:
                drifted_count += 1

            metrics_by_feature[col] = {
                "drift_detected": is_drifted,
                "p_value": round(p_value, 6),
                "ks_statistic": round(stat, 4),
                "wasserstein_distance": round(w_dist, 4),
                "reference_mean": round(ref_mean, 2),
                "current_mean": round(curr_mean, 2),
                "mean_shift_pct": round(mean_shift_pct, 2),
                "reference_std": round(ref_std, 2),
                "current_std": round(curr_std, 2),
            }

        total_features = max(1, len(metrics_by_feature))
        drift_share = round(drifted_count / total_features, 4)
        dataset_drift_detected = drift_share >= self.dataset_drift_threshold

        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset_drift_detected": dataset_drift_detected,
            "drift_share": drift_share,
            "drifted_features_count": drifted_count,
            "total_monitored_features": total_features,
            "p_value_threshold": self.p_val_threshold,
            "alert_status": "CRITICAL_DRIFT_ALERT" if dataset_drift_detected else "NORMAL",
            "recommended_action": (
                "TRIGGER_CONTINUOUS_RETRAINING: Severe data distribution shift detected across sensor fleet."
                if dataset_drift_detected
                else "CONTINUE_NORMAL_OPERATION: Sensor telemetry within baseline statistical bounds."
            ),
            "features": metrics_by_feature,
        }

        # Save JSON report
        json_path = self.output_dir / "drift_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Generate HTML Dashboard
        self._generate_html_dashboard(report)

        return report

    def _generate_html_dashboard(self, report: Dict[str, Any]):
        """Generates a responsive visual HTML dashboard."""
        html_path = self.reports_dir / "data_drift_dashboard.html"

        rows = ""
        for feat, data in report["features"].items():
            status_badge = (
                '<span style="background:#fee2e2;color:#991b1b;padding:3px 8px;border-radius:4px;font-weight:bold;">DRIFTED</span>'
                if data["drift_detected"]
                else '<span style="background:#dcfce7;color:#166534;padding:3px 8px;border-radius:4px;font-weight:bold;">STABLE</span>'
            )
            rows += f"""
            <tr>
              <td><code>{feat}</code></td>
              <td>{status_badge}</td>
              <td>{data['p_value']}</td>
              <td>{data['ks_statistic']}</td>
              <td>{data['wasserstein_distance']}</td>
              <td>{data['reference_mean']} ± {data['reference_std']}</td>
              <td>{data['current_mean']} ± {data['current_std']}</td>
              <td>{data['mean_shift_pct']}%</td>
            </tr>
            """

        alert_color = "#dc2626" if report["dataset_drift_detected"] else "#16a34a"
        alert_title = "DATASET DRIFT DETECTED" if report["dataset_drift_detected"] else "TELEMETRY STABLE"

        html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>AeroMind Ops — Data Drift Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8fafc; margin: 0; padding: 24px; color: #1e293b; }}
    .container {{ max-width: 1100px; margin: 0 auto; background: #fff; padding: 28px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }}
    .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 16px; margin-bottom: 24px; }}
    .badge {{ background: {alert_color}; color: white; padding: 8px 16px; border-radius: 8px; font-weight: bold; font-size: 14px; }}
    .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }}
    .card {{ background: #f1f5f9; padding: 16px; border-radius: 8px; border-left: 4px solid #3b82f6; }}
    .card-title {{ font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: 600; margin-bottom: 4px; }}
    .card-value {{ font-size: 24px; font-weight: bold; color: #0f172a; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #e2e8f0; font-size: 13px; }}
    th {{ background: #f8fafc; color: #475569; font-weight: 600; }}
    tr:hover {{ background: #f8fafc; }}
    .action-box {{ background: #eff6ff; border-left: 4px solid #2563eb; padding: 14px; border-radius: 6px; margin-top: 24px; font-size: 14px; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <h1 style="margin: 0; font-size: 22px;">AeroMind Ops — Fleet Sensor Drift Report</h1>
        <p style="margin: 4px 0 0; color: #64748b; font-size: 13px;">Generated at: {report['timestamp']}</p>
      </div>
      <div class="badge">{alert_title}</div>
    </div>

    <div class="metrics-grid">
      <div class="card">
        <div class="card-title">Drifted Sensors</div>
        <div class="card-value">{report['drifted_features_count']} / {report['total_monitored_features']}</div>
      </div>
      <div class="card">
        <div class="card-title">Drift Ratio</div>
        <div class="card-value">{round(report['drift_share'] * 100, 1)}%</div>
      </div>
      <div class="card">
        <div class="card-title">Threshold</div>
        <div class="card-value">p &lt; {report['p_value_threshold']}</div>
      </div>
      <div class="card">
        <div class="card-title">Status</div>
        <div class="card-value" style="color: {alert_color};">{report['alert_status']}</div>
      </div>
    </div>

    <h2 style="font-size: 16px; margin-top: 0;">Statistical Feature Analysis (Kolmogorov-Smirnov &amp; Wasserstein)</h2>
    <table>
      <thead>
        <tr>
          <th>Sensor Channel</th>
          <th>Drift Status</th>
          <th>p-value</th>
          <th>KS Stat</th>
          <th>Wasserstein Dist</th>
          <th>Reference (Train)</th>
          <th>Current (Test)</th>
          <th>Mean Shift</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>

    <div class="action-box">
      <strong>Recommended Action:</strong> {report['recommended_action']}
    </div>
  </div>
</body>
</html>
"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)


def run_drift_analysis(
    dataset_name: str = "FD001",
    p_val: float = DEFAULT_P_VALUE_THRESHOLD,
) -> Dict[str, Any]:
    """Runs automated drift detection between raw train and test fleets."""
    train_df, test_df, _ = load_raw_dataset(dataset_name)

    detector = SensorDriftDetector(p_val_threshold=p_val)
    detector.fit_reference(train_df)
    report = detector.detect_drift(test_df)

    print("\n" + "=" * 60)
    print("AEROMIND SENSOR DATA DRIFT ANALYSIS")
    print("=" * 60)
    print(f"Dataset:                  {dataset_name}")
    print(f"Dataset Drift Detected:   {report['dataset_drift_detected']}")
    print(f"Drifted Sensors Count:    {report['drifted_features_count']} / {report['total_monitored_features']} ({round(report['drift_share']*100, 1)}%)")
    print(f"Status:                   {report['alert_status']}")
    print(f"Action:                   {report['recommended_action']}")
    print(f"HTML Dashboard:           docs/reports/data_drift_dashboard.html")
    print("=" * 60)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AeroMind Ops Sensor Data Drift Detector")
    parser.add_argument("--dataset", type=str, default="FD001", help="Dataset name (FD001-FD004)")
    parser.add_argument("--pval", type=float, default=DEFAULT_P_VALUE_THRESHOLD, help="p-value threshold (default 0.05)")
    args = parser.parse_args()

    run_drift_analysis(dataset_name=args.dataset, p_val=args.pval)
