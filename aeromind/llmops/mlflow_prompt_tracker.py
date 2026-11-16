"""
MLflow Model and Prompt Tracker for AeroMind Ops.

Enforces centralized tracking of:
- Versioned Prompt Templates & System Prompts.
- RAG evaluation runs (Ragas & DeepEval composite metrics).
- Prompt-to-Model lineage linking LLM copilot versions with underlying DL prognostics models.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from aeromind.mlops.tracker import ExperimentTracker
from aeromind.mlops.registry import ModelRegistry

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MLflowPromptTracker:
    """
    Dedicated MLflow tracker for Prompts, RAG Evaluations, and LLMOps artifacts.
    """

    def __init__(
        self,
        experiment_name: str = "AeroMind_LLMOps_Evaluation",
        tracking_uri: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.local_runs_dir = Path("mlruns_local/llmops")
        self.local_runs_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ModelRegistry()

        self.tracker = ExperimentTracker(
            experiment_name=self.experiment_name,
            tracking_uri=tracking_uri,
            local_runs_dir=self.local_runs_dir,
        )

    def log_prompt_version(
        self,
        prompt_name: str,
        version_id: str,
        system_prompt: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Registers a prompt template into MLflow and local registry.
        """
        run_name = f"prompt_{prompt_name}_{version_id}"
        run_id = self.tracker.start_run(run_name=run_name)

        params = {
            "prompt_name": prompt_name,
            "version_id": version_id,
            "description": description,
            "system_prompt_length": len(system_prompt),
        }
        if parameters:
            params.update(parameters)

        self.tracker.log_params(params)

        # Save prompt artifact
        prompt_artifact = {
            "prompt_name": prompt_name,
            "version_id": version_id,
            "system_prompt": system_prompt,
            "description": description,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        artifact_path = self.local_runs_dir / f"{prompt_name}_{version_id}.json"
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(prompt_artifact, f, indent=2)

        self.tracker.log_artifact(str(artifact_path))
        self.tracker.end_run(status="REGISTERED")

        print(f"[MLflow] Prompt '{prompt_name}' ({version_id}) logged to MLflow run {run_id}.")
        return run_id

    def log_evaluation_run(
        self,
        prompt_version: str,
        ragas_metrics: Dict[str, float],
        deepeval_metrics: Dict[str, Any],
        copilot_metrics: Dict[str, float],
        gate_decision: str,
        sample_results: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Logs a full LLMOps evaluation cycle (Ragas + DeepEval + Copilot) to MLflow.
        Links the active Champion DL prognostics model for end-to-end lineage.
        """
        run_name = f"eval_{prompt_version}_{int(time.time())}"
        run_id = self.tracker.start_run(run_name=run_name)

        # 1. Lineage: Link active champion DL model
        champ_model = self.registry.get_champion_model()
        if champ_model:
            self.tracker.log_params({
                "champion_dl_model": champ_model.get("model_name"),
                "champion_dl_version": champ_model.get("version"),
                "champion_dl_arch": champ_model.get("architecture"),
            })

        # 2. Log Parameters
        self.tracker.log_params({
            "prompt_version": prompt_version,
            "gate_decision": gate_decision,
            "eval_frameworks": "Ragas + DeepEval + Langfuse + MLflow",
        })

        # 3. Log Ragas Metrics
        for k, v in ragas_metrics.items():
            if isinstance(v, (int, float)):
                self.tracker.log_metric(k, float(v))

        # 4. Log DeepEval Metrics
        self.tracker.log_metric("deepeval_pass_rate", float(deepeval_metrics.get("pass_rate", 0.0)))
        self.tracker.log_metric("deepeval_hallucination_rate", float(deepeval_metrics.get("hallucination_rate", 0.0)))

        # 5. Log Copilot Composite Metrics
        for k, v in copilot_metrics.items():
            if isinstance(v, (int, float)):
                self.tracker.log_metric(k, float(v))

        # 6. Save and log evaluation report artifact
        report_data = {
            "run_id": run_id,
            "prompt_version": prompt_version,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "gate_decision": gate_decision,
            "ragas": ragas_metrics,
            "deepeval": deepeval_metrics,
            "copilot": copilot_metrics,
            "sample_results_count": len(sample_results) if sample_results else 0,
        }
        report_path = self.local_runs_dir / f"eval_report_{run_id}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        self.tracker.log_artifact(str(report_path))
        self.tracker.end_run(status="COMPLETED" if gate_decision == "PASS" else "FAILED")

        print(f"[MLflow] Evaluation run logged to MLflow under run {run_id} (Gate: {gate_decision}).")
        return run_id
