from aeromind.agent.copilot import MaintenanceCopilotAgent
from aeromind.llmops.deepeval_suite import DeepEvalMaintenanceSuite
from aeromind.llmops.evaluation import LLMEvaluationSuite
from aeromind.llmops.langfuse_client import LangfuseOpsClient
from aeromind.llmops.mlflow_prompt_tracker import MLflowPromptTracker
from aeromind.llmops.ragas_evaluator import RagasEvaluator


def test_langfuse_tracing_and_prompts():
    client = LangfuseOpsClient()
    prompt = client.get_prompt("aeromind_maintenance_copilot", version="v3_uncertainty_aware")
    assert prompt is not None
    assert "prompt" in prompt

    # Verify trace and span execution
    trace = client.start_trace(name="test_trace", metadata={"unit_test": True})
    with trace.span("test_span", input_data={"query": "engine 12"}):
        res = "Telemetry retrieved"
    trace.update(output=res)
    client.log_score(trace_id=trace.id, name="test_metric", value=1.0)
    assert len(trace.spans) == 1
    assert trace.spans[0]["name"] == "test_span"


def test_ragas_rag_evaluator():
    evaluator = RagasEvaluator()
    samples = [
        {
            "question": "What is the allowable HPC blade damage threshold?",
            "contexts": [
                "Turbofan Standard Operating Procedures: HPC blade nicks up to 1.5mm may be blended. Cracks require replacement."
            ],
            "answer": "According to the Operating Procedures, HPC blade nicks up to 1.5mm can be blended, but cracks mandate blade replacement.",
            "ground_truth": "HPC blade nicks up to 1.5mm blendable, cracks require replacement.",
        }
    ]
    results = evaluator.evaluate_samples(samples)
    assert "ragas_faithfulness" in results
    assert "ragas_answer_relevancy" in results
    assert "ragas_context_precision" in results
    assert "ragas_context_recall" in results
    assert results["ragas_faithfulness"] >= 0.80
    assert results["ragas_answer_relevancy"] >= 0.80


def test_deepeval_suite():
    suite = DeepEvalMaintenanceSuite()
    test_cases = [
        suite.build_test_case(
            query="What is the health of Engine 12?",
            actual_output="Engine #12 has Remaining Useful Life of 112 cycles.",
            expected_output="Engine 12 RUL 112 cycles",
            retrieval_context=["Engine 12 telemetry: RUL 112 cycles"],
            telemetry_data={"estimated_rul": 112},
        )
    ]
    eval_res = suite.evaluate_test_cases(test_cases)
    assert eval_res["total_test_cases"] == 1
    assert eval_res["passed_test_cases"] == 1
    assert eval_res["pass_rate"] == 1.0
    assert eval_res["hallucination_rate"] == 0.0
    assert eval_res["gate_decision"] == "PASS"


def test_mlflow_prompt_and_eval_tracking():
    tracker = MLflowPromptTracker(experiment_name="AeroMind_Test_LLMOps")
    run_id = tracker.log_prompt_version(
        prompt_name="test_prompt",
        version_id="v1.0.0",
        system_prompt="Test system prompt.",
        description="Unit test prompt.",
    )
    assert run_id is not None

    eval_run_id = tracker.log_evaluation_run(
        prompt_version="v3_uncertainty_aware",
        ragas_metrics={"ragas_faithfulness": 0.95, "ragas_answer_relevancy": 0.92},
        deepeval_metrics={"pass_rate": 0.96, "hallucination_rate": 0.04},
        copilot_metrics={"citation_correctness": 0.98, "abstention_quality": 1.0},
        gate_decision="PASS",
    )
    assert eval_run_id is not None


def test_full_llmops_evaluation_pipeline():
    agent = MaintenanceCopilotAgent(prompt_version="v3_uncertainty_aware")
    suite = LLMEvaluationSuite(agent=agent)
    report = suite.run_evaluation(max_samples=5, log_to_mlflow=True, log_to_langfuse=True)

    assert "gate_decision" in report
    assert "ragas_evaluation" in report
    assert "deepeval_evaluation" in report
    assert "copilot_metrics" in report
    assert report["evaluated_samples"] == 5
