from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    engine_id: int = Field(..., description="Engine serial / unit identifier")
    sequence: List[List[float]] = Field(
        ...,
        description="Temporal sliding window sequence of shape (window_size, num_features)",
    )
    operating_settings: Optional[List[float]] = Field(
        None, description="Optional 3 operating condition settings"
    )


class UncertaintyBounds(BaseModel):
    q10: float
    q50: float
    q90: float
    confidence_span: float


class PredictResponse(BaseModel):
    engine_id: int
    model_version: str
    architecture: str
    rul_point_estimate: float
    uncertainty_intervals: UncertaintyBounds
    degradation_stage: str
    alert_level: str
    recommended_action: str
    timestamp: str


class EngineSummary(BaseModel):
    engine_id: int
    cycles_logged: int
    estimated_rul: float
    degradation_stage: str
    alert_level: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Maintenance engineer prompt")
    engine_id: Optional[int] = Field(None, description="Optional target engine ID")


class ChatResponse(BaseModel):
    response: str
    engine_id: Optional[int]
    tool_calls: List[Dict[str, Any]]
    citations: List[str]
    prompt_version: str
    latency_sec: float
    status: str


class ReportRequest(BaseModel):
    engine_id: int = Field(..., description="Target engine ID")


class SystemHealthResponse(BaseModel):
    status: str
    service: str
    version: str
    active_model: Optional[str]
    model_status: Optional[str]

