from pathlib import Path
import time
from typing import Any, Dict, List
import numpy as np
import torch
from fastapi import APIRouter, HTTPException

from aeromind.agent.copilot import MaintenanceCopilotAgent
from aeromind.agent.tools import MaintenanceTools
from aeromind.api.schemas import (
    ChatRequest,
    ChatResponse,
    EngineSummary,
    PredictRequest,
    PredictResponse,
    ReportRequest,
    SystemHealthResponse,
    UncertaintyBounds,
)
from aeromind.core.constants import STAGE_NAMES
from aeromind.mlops.registry import ModelRegistry
from aeromind.models.lstm import LSTMRegressor
from aeromind.models.multitask_transformer import ConditionAwareMultiTaskTransformer

router = APIRouter(prefix="/api/v1")
registry = ModelRegistry()
tools = MaintenanceTools()
copilot = MaintenanceCopilotAgent(tools=tools)

# Cached loaded model
_loaded_model = None
_loaded_model_version = None


def get_active_model():
    global _loaded_model, _loaded_model_version
    champ = registry.get_champion_model()
    if not champ:
        return None, "No registered model"

    version = champ["version"]
    if _loaded_model is not None and _loaded_model_version == version:
        return _loaded_model, champ

    # Load model
    ckpt_path = Path(champ["checkpoint_path"])
    if not ckpt_path.exists():
        return None, f"Checkpoint not found at {ckpt_path}"

    arch = champ["architecture"]
    feat_dim = champ["hyperparameters"].get("feature_dim", 15)

    if arch == "LSTM":
        model = LSTMRegressor(input_dim=feat_dim)
    else:
        model = ConditionAwareMultiTaskTransformer(input_dim=feat_dim)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _loaded_model = model
    _loaded_model_version = version
    return _loaded_model, champ


@router.post("/predict", response_model=PredictResponse)
def predict_rul(request: PredictRequest):
    model, champ_info = get_active_model()
    if model is None:
        raise HTTPException(status_code=503, detail=champ_info)

    seq_arr = np.array(request.sequence, dtype=np.float32)
    if seq_arr.ndim != 2:
        raise HTTPException(
            status_code=400,
            detail=f"Sequence must be 2D of shape (window_size, num_features), got shape {seq_arr.shape}",
        )

    x_tensor = torch.tensor(seq_arr).unsqueeze(0)  # (1, W, D)

    with torch.no_grad():
        if hasattr(model, "stage_head"):
            outputs = model(x_tensor)
            rul_val = float(outputs["rul"].squeeze().item())
            
            if outputs.get("quantiles") is not None:
                q_vals = outputs["quantiles"].squeeze().tolist()
                q10, q50, q90 = float(q_vals[0]), float(q_vals[1]), float(q_vals[2])
            else:
                q10, q50, q90 = max(0.0, rul_val - 7.0), rul_val, rul_val + 7.5

            if outputs.get("stage_logits") is not None:
                stage_idx = int(torch.argmax(outputs["stage_logits"], dim=-1).item())
                stage_name = STAGE_NAMES.get(stage_idx, "Degraded")
            else:
                stage_name = "Healthy" if rul_val > 80 else ("Degraded" if rul_val > 30 else "Critical")
        else:
            rul_val = float(model(x_tensor).squeeze().item())
            q10, q50, q90 = max(0.0, rul_val - 7.0), rul_val, rul_val + 7.5
            stage_name = "Healthy" if rul_val > 80 else ("Degraded" if rul_val > 30 else "Critical")

    rul_clamped = round(max(0.0, min(125.0, rul_val)), 1)
    q10_clamped = round(max(0.0, min(125.0, q10)), 1)
    q50_clamped = round(max(0.0, min(125.0, q50)), 1)
    q90_clamped = round(max(0.0, min(125.0, q90)), 1)

    alert = "GREEN" if stage_name == "Healthy" else ("YELLOW" if stage_name == "Degraded" else "RED")
    action = (
        "Nominal operation. Continue automated telemetry monitoring."
        if stage_name == "Healthy"
        else (
            "Schedule HPC tip clearance inspection within 15 cycles."
            if stage_name == "Degraded"
            else "MANDATORY ACTION: Schedule urgent borescope inspection within 5 cycles."
        )
    )

    return PredictResponse(
        engine_id=request.engine_id,
        model_version=f"v{champ_info['version']}",
        architecture=champ_info["architecture"],
        rul_point_estimate=rul_clamped,
        uncertainty_intervals=UncertaintyBounds(
            q10=q10_clamped,
            q50=q50_clamped,
            q90=q90_clamped,
            confidence_span=round(q90_clamped - q10_clamped, 1),
        ),
        degradation_stage=stage_name,
        alert_level=alert,
        recommended_action=action,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


@router.get("/engines", response_model=List[EngineSummary])
def list_fleet_engines(limit: int = 50):
    if tools.test_df.empty:
        raise HTTPException(status_code=404, detail="Fleet data not loaded.")

    units = sorted(tools.test_df["unit_nr"].unique())[:limit]
    fleet = []
    for u in units:
        health = tools.get_engine_health(u)
        fleet.append(
            EngineSummary(
                engine_id=u,
                cycles_logged=health["operational_cycles_logged"],
                estimated_rul=health["estimated_rul"],
                degradation_stage=health["degradation_stage"],
                alert_level=health["alert_level"],
            )
        )
    return fleet


@router.get("/engines/{engine_id}/health")
def get_engine_health_endpoint(engine_id: int):
    health = tools.get_engine_health(engine_id)
    if "error" in health:
        raise HTTPException(status_code=404, detail=health["error"])
    return health


@router.get("/engines/{engine_id}/history")
def get_engine_history_endpoint(engine_id: int, window: int = 30):
    history = tools.get_sensor_history(engine_id, window=window)
    if "error" in history:
        raise HTTPException(status_code=404, detail=history["error"])
    return history


@router.post("/copilot/chat", response_model=ChatResponse)
def copilot_chat_endpoint(request: ChatRequest):
    result = copilot.run(request.message)
    return ChatResponse(
        response=result["response"],
        engine_id=result.get("engine_id"),
        tool_calls=result.get("tool_calls", []),
        citations=result.get("citations", []),
        prompt_version=result["prompt_version"],
        latency_sec=result["latency_sec"],
        status=result["status"],
    )


@router.post("/reports")
def generate_report_endpoint(request: ReportRequest):
    report = tools.generate_maintenance_report(request.engine_id)
    if "error" in report:
        raise HTTPException(status_code=404, detail=report["error"])
    return report


@router.get("/models")
def list_models_endpoint():
    return registry.models


@router.get("/monitoring/health", response_model=SystemHealthResponse)
def system_health():
    champ = registry.get_champion_model()
    return SystemHealthResponse(
        status="HEALTHY",
        service="AeroMind Ops Enterprise API",
        version="1.0.0",
        active_model=champ["architecture"] if champ else None,
        model_status=champ["status"] if champ else "UNAVAILABLE",
    )

