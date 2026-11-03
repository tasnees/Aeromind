import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

from aeromind.core.constants import SENSOR_COLUMNS, SETTING_COLUMNS
from aeromind.data.ingestion import load_raw_dataset
from aeromind.data.target import compute_rul_target


def perform_eda(
    dataset_names: List[str] = ["FD001", "FD004"],
    raw_dir: Path = Path("data/raw"),
    output_dir: Path = Path("docs/reports"),
) -> Dict[str, any]:
    """
    Performs exploratory data analysis comparing FD001 and FD004:
    - Identifies invariant vs degrading sensors.
    - Computes sensor correlations with cycle / RUL.
    - Evaluates operating condition distributions.
    - Saves statistical report as JSON and Markdown.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    for name in dataset_names:
        train_df, test_df, rul_df = load_raw_dataset(name, raw_dir=raw_dir)
        train_df = compute_rul_target(train_df, is_test=False)
        
        # Sensor summary statistics
        sensor_stats = {}
        invariant_sensors = []
        informative_sensors = []
        correlations_with_rul = {}

        for sensor in SENSOR_COLUMNS:
            std = float(train_df[sensor].std())
            mean = float(train_df[sensor].mean())
            corr = float(train_df[sensor].corr(train_df["rul"]))
            
            sensor_stats[sensor] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "min": round(float(train_df[sensor].min()), 4),
                "max": round(float(train_df[sensor].max()), 4),
                "rul_correlation": round(corr, 4),
            }

            if std < 1e-3:
                invariant_sensors.append(sensor)
            else:
                informative_sensors.append(sensor)
                correlations_with_rul[sensor] = round(corr, 4)

        # Sort sensors by absolute correlation with RUL
        sorted_sensors = sorted(
            correlations_with_rul.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        # Trajectory lengths
        engine_lifetimes = train_df.groupby("unit_nr")["time_cycles"].max()

        report[name] = {
            "num_engines": int(train_df["unit_nr"].nunique()),
            "total_cycles": len(train_df),
            "min_engine_life": int(engine_lifetimes.min()),
            "max_engine_life": int(engine_lifetimes.max()),
            "mean_engine_life": round(float(engine_lifetimes.mean()), 2),
            "invariant_sensors": invariant_sensors,
            "informative_sensors_count": len(informative_sensors),
            "top_degradation_sensors": sorted_sensors[:6],
            "settings_summary": {
                s: {
                    "unique_values": int(train_df[s].nunique()),
                    "min": round(float(train_df[s].min()), 4),
                    "max": round(float(train_df[s].max()), 4),
                }
                for s in SETTING_COLUMNS
            },
        }

    # Save JSON report
    with open(output_dir / "eda_summary.json", "w") as f:
        json.dump(report, f, indent=2)

    # Save Markdown report
    md_content = "# AeroMind Ops — Exploratory Data Analysis Report\n\n"
    for name, data in report.items():
        md_content += f"## Dataset: {name}\n"
        md_content += f"- **Engines:** {data['num_engines']} trajectories\n"
        md_content += f"- **Cycle Lifespans:** Min: {data['min_engine_life']} | Max: {data['max_engine_life']} | Mean: {data['mean_engine_life']:.1f} cycles\n"
        md_content += f"- **Invariant Sensors ({len(data['invariant_sensors'])}):** {', '.join(data['invariant_sensors']) if data['invariant_sensors'] else 'None'}\n"
        md_content += f"- **Top Informative Sensors (by RUL correlation):**\n"
        for s, corr in data['top_degradation_sensors']:
            direction = "Decreases with wear" if corr > 0 else "Increases with wear"
            md_content += f"  - `{s}`: Corr = {corr:+.4f} ({direction})\n"
        md_content += f"- **Operating Regimes (Settings):**\n"
        for s, s_data in data['settings_summary'].items():
            md_content += f"  - `{s}`: {s_data['unique_values']} distinct operating points (range: [{s_data['min']}, {s_data['max']}])\n"
        md_content += "\n"

    with open(output_dir / "eda_summary.md", "w") as f:
        f.write(md_content)

    return report


if __name__ == "__main__":
    report = perform_eda()
    print("EDA completed successfully.")

