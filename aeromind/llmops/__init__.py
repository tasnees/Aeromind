"""
AeroMind LLMOps Module — Observability, Tracing, RAG Evaluation, and Testing Suite.

Integrates:
- Langfuse for tracing, prompt management, and evaluation score persistence.
- Ragas for quantitative RAG pipeline assessment (faithfulness, relevancy, precision, recall).
- DeepEval for automated testing of LLM responses and hallucination checks.
- MLflow for centralized prompt tracking and evaluation run metrics.
"""

from aeromind.llmops.deepeval_suite import DeepEvalMaintenanceSuite
from aeromind.llmops.evaluation import LLMEvaluationSuite
from aeromind.llmops.langfuse_client import LangfuseOpsClient
from aeromind.llmops.mlflow_prompt_tracker import MLflowPromptTracker
from aeromind.llmops.prompts import PROMPTS_REGISTRY, get_prompt_version
from aeromind.llmops.ragas_evaluator import RagasEvaluator

__all__ = [
    "LangfuseOpsClient",
    "RagasEvaluator",
    "DeepEvalMaintenanceSuite",
    "MLflowPromptTracker",
    "LLMEvaluationSuite",
    "PROMPTS_REGISTRY",
    "get_prompt_version",
]
