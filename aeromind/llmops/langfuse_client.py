"""
Langfuse Client for AeroMind Ops — Tracing, Prompt Management, and Evaluation Logging.

Enables full observability of the Maintenance Copilot:
- Distributed execution tracing (end-to-end latency, tool calls, guardrails).
- Centralized prompt management with versioned retrieval and caching.
- Evaluation score logging for trace-level auditing.
Supports graceful offline fallback when API keys or network are unavailable.
"""

import os
import time
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

from aeromind.core.config import LLMOpsConfig

try:
    from langfuse import Langfuse  # type: ignore
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False


class LocalTrace:
    """Lightweight in-memory/local tracer used when Langfuse credentials are not configured."""

    def __init__(self, trace_id: str, name: str, metadata: Optional[Dict[str, Any]] = None):
        self.id = trace_id
        self.name = name
        self.metadata = metadata or {}
        self.spans: List[Dict[str, Any]] = []
        self.scores: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.output: Any = None

    def span(
        self,
        name: str,
        input: Any = None,
        input_data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        payload = input if input is not None else input_data
        meta = metadata or {}
        meta.update(kwargs)
        return LocalSpan(self, name, payload, meta)

    def score(self, name: str, value: float, comment: Optional[str] = None):
        self.scores.append({"name": name, "value": value, "comment": comment, "timestamp": time.time()})

    def update(self, output: Any = None, metadata: Optional[Dict[str, Any]] = None):
        self.output = output
        if metadata:
            self.metadata.update(metadata)
        self.end_time = time.time()


class LocalSpan:
    """Local representation of an execution span (e.g. tool invocation or guardrail check)."""

    def __init__(self, trace: LocalTrace, name: str, input_data: Any = None, metadata: Optional[Dict[str, Any]] = None):
        self.trace = trace
        self.name = name
        self.input = input_data
        self.metadata = metadata or {}
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.output: Any = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end(output=str(exc_val) if exc_val else None)

    def end(self, output: Any = None, metadata: Optional[Dict[str, Any]] = None):
        self.end_time = time.time()
        self.output = output
        if metadata:
            self.metadata.update(metadata)
        self.trace.spans.append({
            "name": self.name,
            "input": self.input,
            "output": self.output,
            "duration_sec": round(self.end_time - self.start_time, 4),
            "metadata": self.metadata,
        })


class LangfuseOpsClient:
    """
    Enterprise Langfuse integration for AeroMind Ops.
    Connects to Langfuse Cloud or self-hosted instance, falling back to local audit logging.
    """

    def __init__(self, config: Optional[LLMOpsConfig] = None):
        self.config = config or LLMOpsConfig()
        self.public_key = self.config.langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = self.config.langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = self.config.langfuse_host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        self.client = None
        self.is_connected = False

        if LANGFUSE_AVAILABLE and self.public_key and self.secret_key:
            try:
                self.client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host,
                )
                self.is_connected = True
                print(f"[Langfuse] Connected to Langfuse at {self.host}")
            except Exception as e:
                print(f"[Langfuse] Connection warning: {e}. Falling back to local tracing.")
                self.is_connected = False
        else:
            status_reason = "missing credentials" if LANGFUSE_AVAILABLE else "langfuse library not installed"
            print(f"[Langfuse] Operating in local audit mode ({status_reason}).")

        self.local_traces_dir = Path("mlruns_local/langfuse_traces")
        self.local_traces_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Tracing API
    # -------------------------------------------------------------------------
    def start_trace(
        self,
        name: str,
        user_id: Optional[str] = "maintenance_engineer",
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Any:
        """Starts a root trace for an agent turn or query."""
        trace_id = f"trace_{int(time.time() * 1000)}"
        tags = tags or ["aeromind", "copilot"]

        if self.is_connected and self.client:
            try:
                return self.client.trace(
                    name=name,
                    user_id=user_id,
                    session_id=session_id,
                    metadata=metadata,
                    tags=tags,
                )
            except Exception as e:
                print(f"[Langfuse] Trace creation error: {e}")

        # Local fallback tracer
        return LocalTrace(trace_id=trace_id, name=name, metadata=metadata)

    def log_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
    ):
        """Logs an evaluation score (e.g. faithfulness, citation accuracy) to a trace."""
        if self.is_connected and self.client:
            try:
                self.client.score(
                    trace_id=trace_id,
                    name=name,
                    value=value,
                    comment=comment,
                )
                return
            except Exception as e:
                print(f"[Langfuse] Error logging score to remote: {e}")

        # Save locally
        audit_file = self.local_traces_dir / "evaluation_scores.jsonl"
        with open(audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "trace_id": trace_id,
                "name": name,
                "value": value,
                "comment": comment,
                "timestamp": time.time(),
            }) + "\n")

    # -------------------------------------------------------------------------
    # 2. Prompt Management API
    # -------------------------------------------------------------------------
    def get_prompt(
        self,
        prompt_name: str,
        version: Optional[str] = None,
        fallback_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves a versioned prompt from Langfuse Prompt Management.
        Falls back to local prompt registry if not found or offline.
        """
        if self.is_connected and self.client:
            try:
                lf_prompt = self.client.get_prompt(name=prompt_name, version=version)
                if lf_prompt:
                    return {
                        "prompt": lf_prompt.prompt,
                        "version": lf_prompt.version,
                        "name": prompt_name,
                        "source": "langfuse_remote",
                    }
            except Exception as e:
                print(f"[Langfuse] Prompt retrieval error ({prompt_name}): {e}. Using fallback.")

        # Local fallback
        return {
            "prompt": fallback_prompt or "You are AeroMind Ops Maintenance Assistant.",
            "version": version or "v3.0.0",
            "name": prompt_name,
            "source": "local_registry",
        }

    def register_prompt(
        self,
        prompt_name: str,
        prompt_text: str,
        labels: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Uploads / registers a versioned prompt template to Langfuse."""
        if self.is_connected and self.client:
            try:
                self.client.create_prompt(
                    name=prompt_name,
                    prompt=prompt_text,
                    labels=labels or ["production"],
                    config=config or {},
                )
                print(f"[Langfuse] Prompt '{prompt_name}' successfully registered in Langfuse.")
                return True
            except Exception as e:
                print(f"[Langfuse] Failed to register prompt '{prompt_name}': {e}")
                return False
        return False

    def flush(self):
        """Flushes pending events to Langfuse."""
        if self.is_connected and self.client:
            try:
                self.client.flush()
            except Exception:
                pass
