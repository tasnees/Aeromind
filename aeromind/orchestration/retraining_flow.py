"""
Automated Continuous Training (CT) & Orchestration Flow for AeroMind Ops.

Orchestrates the entire predictive maintenance lifecycle:
  1. Ingestion & schema validation
  2. Telemetry drift detection (triggers retraining if drift > threshold)
  3. Feature engineering & sequence windowing
  4. Model training (tracked via MLflow)
  5. Benchmark test evaluation & Model Registry promotion gate
  6. Automated champion deployment / rollback

Compatible with Prefect 2.x/3.x or runs natively as a robust DAG workflow.
"""

import argparse
import functools
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Try importing Prefect; fallback to transparent native decorators if not installed
try:
    from prefect import flow, task  # type: ignore
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False

    def task(fn=None, **kwargs):
        if fn is None:
            return lambda f: f
        return fn

    def flow(fn=None, **kwargs):
        if fn is None:
            return lambda f: f
        return fn


@task(name="Ingest and Validate Fleet Data")
def task_ingest_and_validate(dataset_name: str) -> Dict[str, Any]:
    """Task 1: Ingests raw sensor telemetry and validates monotonic cycles."""
    print(f"\n[Task 1/5] Ingesting and validating {dataset_name} fleet telemetry...")
    from aeromind.data.ingestion import DataIngestionValidator, load_raw_dataset

    train_df, test_df, rul_df = load_raw_dataset(dataset_name)
    val_res = DataIngestionValidator.validate_dataframe(train_df, dataset_name)
    if not val_res["passed"]:
        raise ValueError(f"Data ingestion validation failed: {val_res['errors']}")

    print(f"  ✓ Data validated: {val_res['unique_engines']} engines, {val_res['total_rows']} cycles.")
    return {
        "dataset_name": dataset_name,
        "total_engines": val_res["unique_engines"],
        "total_records": val_res["total_rows"],
        "status": "VALIDATED",
    }


@task(name="Detect Telemetry Data Drift")
def task_detect_data_drift(dataset_name: str) -> Dict[str, Any]:
    """Task 2: Checks sensor distribution drift between reference and incoming fleet."""
    print(f"\n[Task 2/5] Evaluating sensor distribution drift...")
    from aeromind.monitoring.drift_detector import run_drift_analysis

    drift_report = run_drift_analysis(dataset_name=dataset_name)
    drift_detected = drift_report["dataset_drift_detected"]
    print(
        f"  ✓ Drift status: {drift_report['alert_status']} "
        f"({drift_report['drifted_features_count']}/{drift_report['total_monitored_features']} drifted)"
    )
    return {
        "drift_detected": drift_detected,
        "drift_share": drift_report["drift_share"],
        "alert_status": drift_report["alert_status"],
    }


@task(name="Run Feature Engineering & Windowing")
def task_preprocess_pipeline(
    dataset_name: str,
    window_size: int = 30,
    max_rul: int = 125,
) -> Dict[str, Any]:
    """Task 3: Fits scalers and computes piece-wise linear RUL sliding windows."""
    print(f"\n[Task 3/5] Running preprocessing & sliding window pipeline...")
    from aeromind.data.pipeline import run_data_pipeline

    datasets, preprocessor = run_data_pipeline(
        dataset_name=dataset_name,
        window_size=window_size,
        max_rul=max_rul,
    )
    print(
        f"  ✓ Processed: {len(datasets['train'])} train windows, "
        f"{len(datasets['val'])} val windows, {len(datasets['test'])} test windows."
    )
    return {
        "num_features": len(preprocessor.selected_features),
        "train_windows": len(datasets["train"]),
        "val_windows": len(datasets["val"]),
        "test_windows": len(datasets["test"]),
    }


@task(name="Train Candidate Prognostics Model")
def task_train_candidate_model(
    dataset_name: str,
    exp_id: str = "EXP-04",
    epochs: int = 15,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Task 4: Trains candidate model with MLflow experiment tracking."""
    print(f"\n[Task 4/5] Training candidate model ({exp_id}) with MLflow tracking...")
    from aeromind.training.experiments import run_experiment

    results = run_experiment(
        exp_id=exp_id,
        dataset_name=dataset_name,
        epochs=epochs,
        batch_size=batch_size,
    )
    print(f"  ✓ Training completed. Candidate checkpoint: {results['checkpoint']}")
    return results


@task(name="Evaluate Promotion Gate & Deploy")
def task_evaluate_and_promote(
    candidate_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Task 5: Evaluates promotion criteria and elects new Champion if verified."""
    print(f"\n[Task 5/5] Evaluating Model Registry promotion gate...")
    from aeromind.mlops.registry import ModelRegistry

    registry = ModelRegistry()
    version_num = candidate_results["version"]
    gate_result = registry.evaluate_promotion_gate(version_num)

    champion = registry.get_champion_model()
    is_champion = champion is not None and champion["version"] == version_num
    promoted = gate_result.get("promoted", False)
    status_str = "PROMOTED_TO_CHAMPION" if promoted else "REJECTED_BELOW_GATE_THRESHOLD"

    print(f"  ✓ Promotion decision: {status_str}")
    print(f"  ✓ Active Champion: Version {champion['version'] if champion else 'None'} ({champion['architecture'] if champion else 'None'})")

    return {
        "candidate_version": version_num,
        "gate_passed": promoted,
        "is_active_champion": is_champion,
        "details": gate_result,
    }


@flow(name="AeroMind Continuous Training Pipeline")
def continuous_retraining_flow(
    dataset_name: str = "FD001",
    epochs: int = 15,
    batch_size: int = 64,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """
    Main Orchestrated Flow:
    Coordinates ingestion, drift checks, training, evaluation, and deployment.
    """
    start_time = time.time()
    print("=" * 65)
    print("AEROMIND OPS — CONTINUOUS TRAINING & ORCHESTRATION PIPELINE")
    print(f"Engine: {'Prefect Engine' if PREFECT_AVAILABLE else 'Native Directed Acyclic Graph (DAG)'}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%SZ')}")
    print("=" * 65)

    # 1. Ingestion
    ingest_summary = task_ingest_and_validate(dataset_name)

    # 2. Drift Detection
    drift_summary = task_detect_data_drift(dataset_name)

    # Conditional Execution: Retrain if drift detected OR if force_retrain requested
    should_retrain = drift_summary["drift_detected"] or force_retrain
    if not should_retrain:
        print("\n[Orchestrator] No telemetry drift detected and force_retrain=False. Skipping retraining.")
        return {
            "status": "SKIPPED_NO_DRIFT",
            "drift_summary": drift_summary,
            "duration_sec": round(time.time() - start_time, 2),
        }

    print("\n[Orchestrator] Telemetry drift or manual trigger confirmed. Proceeding with retraining...")

    # 3. Preprocessing
    prep_summary = task_preprocess_pipeline(dataset_name)

    # 4. Training
    train_summary = task_train_candidate_model(
        dataset_name=dataset_name,
        exp_id="EXP-04",
        epochs=epochs,
        batch_size=batch_size,
    )

    # 5. Promotion Gate
    deploy_summary = task_evaluate_and_promote(train_summary)

    duration = round(time.time() - start_time, 2)
    pipeline_summary = {
        "pipeline_status": "SUCCESS",
        "dataset_name": dataset_name,
        "duration_seconds": duration,
        "ingestion": ingest_summary,
        "drift": drift_summary,
        "preprocessing": prep_summary,
        "training": {
            "candidate_version": train_summary["version"],
            "test_rmse": train_summary["test_metrics"]["rmse"],
            "test_nasa_score": train_summary["test_metrics"]["nasa_score"],
        },
        "deployment": deploy_summary,
    }

    # Save pipeline execution artifact
    log_dir = Path("docs/reports")
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "latest_pipeline_execution.json", "w", encoding="utf-8") as f:
        json.dump(pipeline_summary, f, indent=2)

    print("\n" + "=" * 65)
    print(f"PIPELINE COMPLETED IN {duration}s")
    print(f"Champion Promoted: {deploy_summary['is_active_champion']}")
    print(f"Report saved: docs/reports/latest_pipeline_execution.json")
    print("=" * 65)
    return pipeline_summary


def run_orchestrated_pipeline(dataset: str = "FD001", epochs: int = 2):
    return continuous_retraining_flow(dataset_name=dataset, epochs=epochs, force_retrain=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AeroMind Continuous Retraining Flow")
    parser.add_argument("--dataset", type=str, default="FD001")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force-retrain", action="store_true", default=True)
    args = parser.parse_args()

    continuous_retraining_flow(
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        force_retrain=args.force_retrain,
    )
