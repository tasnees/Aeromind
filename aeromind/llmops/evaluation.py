"""
Unified LLMOps Evaluation Suite for AeroMind Ops.

Integrates:
- Langfuse: Trace attribution and evaluation score persistence.
- Ragas: Faithfulness, Answer Relevancy, Context Precision, and Context Recall.
- DeepEval: Test cases checking hallucination, numerical consistency, and guardrails.
- MLflow: Comprehensive tracking of prompt versions, metrics, and evaluation artifacts.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from aeromind.core.config import LLMOpsConfig
from aeromind.llmops.deepeval_suite import DeepEvalMaintenanceSuite
from aeromind.llmops.langfuse_client import LangfuseOpsClient
from aeromind.llmops.mlflow_prompt_tracker import MLflowPromptTracker
from aeromind.llmops.ragas_evaluator import RagasEvaluator


class LLMEvaluationSuite:
    """
    Enterprise LLMOps Evaluation Harness for AeroMind Maintenance Assistant.
    Coordinates Ragas, DeepEval, Langfuse, and MLflow under a unified Quality Gate.
    """

    def __init__(
        self,
        benchmark_file: Path = Path("data/processed/llm_benchmark_dataset.json"),
        agent: Optional[Any] = None,
        config: Optional[LLMOpsConfig] = None,
    ):
        self.benchmark_file = Path(benchmark_file)
        self.config = config or LLMOpsConfig()

        if agent is None:
            from aeromind.agent.copilot import MaintenanceCopilotAgent
            self.agent = MaintenanceCopilotAgent(prompt_version="v3_uncertainty_aware")
        else:
            self.agent = agent

        # LLMOps Infrastructure Clients
        self.langfuse = LangfuseOpsClient(config=self.config)
        self.ragas_evaluator = RagasEvaluator()
        self.deepeval_suite = DeepEvalMaintenanceSuite(
            faithfulness_threshold=self.config.min_faithfulness,
            hallucination_threshold=self.config.max_hallucination_rate,
        )
        self.mlflow_tracker = MLflowPromptTracker(
            experiment_name=self.config.mlflow_experiment_name,
            tracking_uri=self.config.mlflow_tracking_uri,
        )

        with open(self.benchmark_file, "r", encoding="utf-8") as f:
            self.benchmark_data: List[Dict[str, Any]] = json.load(f)

    def run_evaluation(
        self,
        max_samples: int = 30,
        log_to_mlflow: bool = True,
        log_to_langfuse: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes a complete evaluation across the benchmark dataset:
        1. Agent execution + Langfuse trace logging.
        2. Ragas RAG pipeline metric evaluation.
        3. DeepEval test case verification.
        4. MLflow experiment tracking and prompt artifact logging.
        5. Quality Gate pass/fail determination.
        """
        samples = self.benchmark_data[:max_samples]
        print(f"\n[INFO] Starting LLMOps evaluation suite over {len(samples)} benchmark samples...")

        agent_results = []
        ragas_samples = []
        deepeval_test_cases = []

        citation_scores = []
        abstention_scores = []
        latencies = []

        for sample in samples:
            query = sample["query"]
            expected_keywords = sample.get("expected_answer_keywords", [])
            should_abstain = sample.get("should_abstain", False)

            # 1. Execute Copilot Agent
            res = self.agent.run(query)
            agent_results.append(res)
            response = res["response"]
            latencies.append(res.get("latency_sec", 0.05))

            # Retrieve documentation chunks retrieved by the agent's tool calls
            retrieved_chunks = []
            for tc in res.get("tool_calls", []):
                if tc.get("tool") == "search_technical_docs":
                    pass

            # Ground truth text proxy
            ground_truth = " ".join(expected_keywords) if expected_keywords else "Standard turbofan maintenance operating procedure."

            contexts = res.get("retrieval_context", [])
            if not contexts:
                contexts = res.get("citations", ["NASA C-MAPSS Technical Reference Manual"])

            # Build Ragas sample
            ragas_samples.append({
                "question": query,
                "contexts": contexts,
                "answer": response,
                "ground_truth": ground_truth,
            })

            # Build DeepEval test case
            deepeval_test_cases.append(
                self.deepeval_suite.build_test_case(
                    query=query,
                    actual_output=response,
                    expected_output=ground_truth,
                    retrieval_context=contexts,
                    telemetry_data=res.get("health_summary", {}),
                )
            )

            # Record Citation & Abstention
            if should_abstain:
                is_abstained = res.get("status") in [
                    "ABSTAINED_OUT_OF_DOMAIN",
                    "ABSTAINED_NO_EVIDENCE",
                    "REJECTED_SECURITY",
                ]
                abstention_scores.append(1.0 if is_abstained else 0.0)
            else:
                cit_score = 1.0 if res.get("citation_verified", False) else 0.0
                citation_scores.append(cit_score)

            # Log score to Langfuse if trace available
            if log_to_langfuse and res.get("trace_id"):
                self.langfuse.log_score(
                    trace_id=res["trace_id"],
                    name="citation_verified",
                    value=1.0 if res.get("citation_verified") else 0.0,
                )

        # 2. Compute Ragas RAG Metrics
        ragas_metrics = self.ragas_evaluator.evaluate_samples(ragas_samples)

        # 3. Compute DeepEval Suite Results
        deepeval_results = self.deepeval_suite.evaluate_test_cases(deepeval_test_cases)

        # 4. Compute Agent Metrics
        mean_citation = float(np.mean(citation_scores)) if citation_scores else 1.0
        mean_abstention = float(np.mean(abstention_scores)) if abstention_scores else 1.0
        mean_latency = float(np.mean(latencies))

        copilot_metrics = {
            "citation_correctness": round(mean_citation, 4),
            "abstention_quality": round(mean_abstention, 4),
            "mean_latency_sec": round(mean_latency, 4),
        }

        # 5. Evaluate Quality Gate Decision
        gate_decision = "PASS"
        failure_reasons = []

        if ragas_metrics["ragas_faithfulness"] < self.config.min_faithfulness:
            gate_decision = "FAIL"
            failure_reasons.append(f"Ragas faithfulness {ragas_metrics['ragas_faithfulness']} < {self.config.min_faithfulness}")

        if deepeval_results["pass_rate"] < 0.85:
            gate_decision = "FAIL"
            failure_reasons.append(f"DeepEval pass rate {deepeval_results['pass_rate']} < 0.85")

        if copilot_metrics["citation_correctness"] < self.config.min_citation_correctness:
            gate_decision = "FAIL"
            failure_reasons.append(f"Citation correctness {copilot_metrics['citation_correctness']} < {self.config.min_citation_correctness}")

        if copilot_metrics["abstention_quality"] < self.config.min_abstention_quality:
            gate_decision = "FAIL"
            failure_reasons.append(f"Abstention quality {copilot_metrics['abstention_quality']} < {self.config.min_abstention_quality}")

        report = {
            "prompt_version": self.agent.prompt_version,
            "evaluated_samples": len(samples),
            "gate_decision": gate_decision,
            "failure_reasons": failure_reasons,
            "ragas_evaluation": ragas_metrics,
            "deepeval_evaluation": {
                "total_cases": deepeval_results["total_test_cases"],
                "passed_cases": deepeval_results["passed_test_cases"],
                "pass_rate": deepeval_results["pass_rate"],
                "hallucination_rate": deepeval_results["hallucination_rate"],
            },
            "copilot_metrics": copilot_metrics,
        }

        # 6. Log Evaluation Run and Prompt Artifacts to MLflow
        if log_to_mlflow:
            run_id = self.mlflow_tracker.log_evaluation_run(
                prompt_version=self.agent.prompt_version,
                ragas_metrics=ragas_metrics,
                deepeval_metrics=deepeval_results,
                copilot_metrics=copilot_metrics,
                gate_decision=gate_decision,
            )
            report["mlflow_run_id"] = run_id

        # Flush any batched Langfuse events
        if log_to_langfuse:
            self.langfuse.flush()

        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AeroMind Unified LLMOps Evaluation Suite")
    parser.add_argument("--samples", type=int, default=25, help="Number of benchmark samples to evaluate")
    parser.add_argument("--prompt-version", type=str, default="v3_uncertainty_aware", help="Prompt version to test")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow run logging")
    args = parser.parse_args()

    from aeromind.agent.copilot import MaintenanceCopilotAgent
    agent = MaintenanceCopilotAgent(prompt_version=args.prompt_version)
    suite = LLMEvaluationSuite(agent=agent)
    result = suite.run_evaluation(max_samples=args.samples, log_to_mlflow=not args.no_mlflow)

    print("\n================== LLMOps Quality Gate Report ==================")
    print(json.dumps(result, indent=2))
    print(f"\nFinal Quality Gate Decision: [{result['gate_decision']}]")
    if result["gate_decision"] == "FAIL":
        print(f"Gate Failures: {result['failure_reasons']}")
        exit(1)
    else:
        print("All quality thresholds met (Ragas, DeepEval, Langfuse, MLflow).")
