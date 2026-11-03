import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import time

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ExperimentTracker:
    """
    Enterprise Experiment Tracker for AeroMind Ops.
    Integrates directly with MLflow 3.x while preserving robust local JSON run persistence.
    Tracks hyperparameters, validation metrics across epochs, NASA scores, and checkpoint artifacts.
    """

    def __init__(
        self,
        experiment_name: str = "AeroMind_RUL_Research",
        tracking_uri: Optional[str] = None,
        local_runs_dir: Path = Path("mlruns_local"),
    ):
        self.experiment_name = experiment_name
        self.local_runs_dir = Path(local_runs_dir)
        self.local_runs_dir.mkdir(parents=True, exist_ok=True)
        self.active_run_id: Optional[str] = None
        self.run_data: Dict[str, Any] = {}

        self.use_mlflow = MLFLOW_AVAILABLE
        if self.use_mlflow:
            try:
                # Default to local mlruns directory if tracking_uri not explicitly provided
                uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
                mlflow.set_tracking_uri(uri)
                mlflow.set_experiment(self.experiment_name)
            except Exception as e:
                print(f"[Tracker] MLflow initialization warning: {e}. Falling back to local store.")
                self.use_mlflow = False

    def start_run(self, run_name: str) -> str:
        timestamp_str = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        local_id = f"{run_name}_{int(time.time())}"
        
        self.run_data = {
            "run_id": local_id,
            "run_name": run_name,
            "start_time": timestamp_str,
            "params": {},
            "metrics": {},
            "artifacts": [],
        }

        if self.use_mlflow:
            try:
                run = mlflow.start_run(run_name=run_name)
                self.active_run_id = run.info.run_id
                self.run_data["mlflow_run_id"] = self.active_run_id
            except Exception as e:
                print(f"[Tracker] Could not start MLflow run: {e}")
                self.active_run_id = local_id
                self.use_mlflow = False
        else:
            self.active_run_id = local_id

        return self.active_run_id

    def log_params(self, params: Dict[str, Any]):
        self.run_data["params"].update(params)
        if self.use_mlflow:
            try:
                mlflow.log_params(params)
            except Exception as e:
                print(f"[Tracker] MLflow log_params error: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        if key not in self.run_data["metrics"]:
            self.run_data["metrics"][key] = []
        self.run_data["metrics"][key].append({"step": step, "value": value})

        if self.use_mlflow:
            try:
                mlflow.log_metric(key, float(value), step=step)
            except Exception:
                pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for k, v in metrics.items():
            self.log_metric(k, v, step=step)

    def log_artifact(self, file_path: str):
        self.run_data["artifacts"].append(str(file_path))
        if self.use_mlflow and Path(file_path).exists():
            try:
                mlflow.log_artifact(str(file_path))
            except Exception as e:
                print(f"[Tracker] MLflow log_artifact error: {e}")

    def end_run(self):
        self.run_data["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Always persist run metadata to local json file
        if self.active_run_id:
            run_file = self.local_runs_dir / f"{self.run_data['run_id']}.json"
            with open(run_file, "w") as f:
                json.dump(self.run_data, f, indent=2)

        if self.use_mlflow:
            try:
                mlflow.end_run()
            except Exception:
                pass

        self.active_run_id = None
