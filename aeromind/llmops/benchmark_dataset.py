import json
from pathlib import Path
from typing import Any, Dict, List


def generate_llm_evaluation_dataset(
    output_path: Path = Path("data/processed/llm_benchmark_dataset.json"),
    num_questions: int = 150,
) -> List[Dict[str, Any]]:
    """
    Generates a rigorous benchmark dataset of ~150 evaluation questions covering:
    - Technical Knowledge
    - Prediction Interpretation
    - Sensor Analysis
    - Multi-Step Reasoning
    - Document Retrieval
    - Unsupported Questions (Abstention Quality)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = []

    # Category 1: Technical Knowledge (30 questions)
    tech_topics = [
        ("C-MAPSS dataset origin", "NASA Commercial Modular Aero-Propulsion System Simulation"),
        ("FD001 operating conditions", "Sea level standard day condition, HPC degradation fault mode"),
        ("FD004 operating conditions", "Six flight regimes, dual fault modes: HPC and Fan degradation"),
        ("High Pressure Compressor fault indicator", "Drop in static pressure Ps30 (sensor 11) and rise in outlet temperature T50 (sensor 4)"),
        ("Asymmetric NASA score penalty", "Penalizes late RUL predictions exponentially more than early predictions"),
    ]
    for i, (topic, answer) in enumerate(tech_topics):
        for variant in range(6):
            q_id = f"TECH_{i*6 + variant + 1:03d}"
            dataset.append({
                "id": q_id,
                "category": "technical_knowledge",
                "query": f"Explain the engineering significance of {topic} (reference {variant+1}).",
                "expected_answer_keywords": [topic.split()[0], "C-MAPSS", "turbofan"],
                "expected_document": "cmapss_spec_001",
                "should_abstain": False,
            })

    # Category 2: Prediction Interpretation (30 questions)
    for engine_id in range(1, 31):
        q_id = f"PRED_{engine_id:03d}"
        dataset.append({
            "id": q_id,
            "category": "prediction_interpretation",
            "query": f"Explain why engine {engine_id} has its current RUL and what degradation stage it is in.",
            "expected_answer_keywords": ["RUL", "cycles", "degradation", "confidence interval"],
            "expected_document": "maintenance_proc_042",
            "should_abstain": False,
        })

    # Category 3: Sensor Analysis (30 questions)
    sensor_pairs = [
        ("sensor_11 (Ps30)", "sensor_4 (T50)"),
        ("sensor_9 (Nc)", "sensor_12 (phi)"),
        ("sensor_14 (Nrc)", "sensor_15 (BPR)"),
    ]
    for i, (s1, s2) in enumerate(sensor_pairs):
        for engine_id in range(1, 11):
            q_id = f"SENSOR_{i*10 + engine_id:03d}"
            dataset.append({
                "id": q_id,
                "category": "sensor_analysis",
                "query": f"Compare {s1} and {s2} trends for engine {engine_id}.",
                "expected_answer_keywords": ["trend", "sensor", "increasing", "decreasing"],
                "expected_document": "cmapss_spec_001",
                "should_abstain": False,
            })

    # Category 4: Multi-Step Reasoning (25 questions)
    for pair_idx in range(1, 26):
        e1, e2 = pair_idx, pair_idx + 10
        q_id = f"REASON_{pair_idx:03d}"
        dataset.append({
            "id": q_id,
            "category": "multi_step_reasoning",
            "query": f"Compare the health of engine {e1} with engine {e2} and determine which engine should be overhauled first.",
            "expected_answer_keywords": ["engine", "RUL", "overhaul", "urgency"],
            "expected_document": "maintenance_proc_042",
            "should_abstain": False,
        })

    # Category 5: Document Retrieval (20 questions)
    doc_queries = [
        "What are the inspection requirements when RUL falls below 30 cycles?",
        "How is damage propagation modeled in turbofan high pressure compressors?",
        "What are the six operating condition dimensions in C-MAPSS FD002 and FD004?",
        "What is the mathematical definition of the C-MAPSS scoring function?",
    ]
    for i, dq in enumerate(doc_queries):
        for variant in range(5):
            q_id = f"RETRIEVE_{i*5 + variant + 1:03d}"
            dataset.append({
                "id": q_id,
                "category": "document_retrieval",
                "query": f"Find technical documentation: {dq}",
                "expected_answer_keywords": ["inspection", "cycles", "damage", "model"],
                "expected_document": "saxena_2008_damage_propagation",
                "should_abstain": False,
            })

    # Category 6: Unsupported Questions (15 questions - Abstention Quality)
    unsupported_queries = [
        "What is the current stock price of Boeing?",
        "Can you recommend a recipe for chocolate cake?",
        "What will the weather in Paris be next Tuesday?",
        "Who won the 2024 FIFA World Cup?",
        "How do I invest in Bitcoin cryptocurrency?",
    ]
    for i, uq in enumerate(unsupported_queries):
        for variant in range(3):
            q_id = f"ABSTAIN_{i*3 + variant + 1:03d}"
            dataset.append({
                "id": q_id,
                "category": "unsupported_questions",
                "query": uq,
                "expected_answer_keywords": ["abstain", "exclusively", "cannot assist"],
                "expected_document": None,
                "should_abstain": True,
            })

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {len(dataset)} evaluation benchmark questions -> {output_path}")
    return dataset


if __name__ == "__main__":
    generate_llm_evaluation_dataset()

