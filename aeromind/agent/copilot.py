import re
from typing import Any, Dict, List, Optional
import time

from aeromind.agent.guardrails import MaintenanceGuardrails
from aeromind.agent.tools import MaintenanceTools


class MaintenanceCopilotAgent:
    """
    Autonomous tool-augmented AI Maintenance Assistant for AeroMind Ops.
    Interprets DL predictions, retrieves verified sensor telemetry and technical documents,
    and enforces strict citation integrity.
    """

    def __init__(
        self,
        tools: Optional[MaintenanceTools] = None,
        prompt_version: str = "v3_uncertainty_aware",
    ):
        self.tools = tools or MaintenanceTools()
        self.guardrails = MaintenanceGuardrails()
        self.prompt_version = prompt_version

    def _extract_engine_id(self, query: str) -> Optional[int]:
        match = re.search(r"engine\s+(\d+)", query, re.IGNORECASE)
        if match:
            return int(match.group(1))
        # Match standalone #42 or unit 42
        match2 = re.search(r"(?:#|unit\s+)(\d+)", query, re.IGNORECASE)
        if match2:
            return int(match2.group(1))
        return None

    def run(self, user_query: str) -> Dict[str, Any]:
        start_time = time.time()

        # 1. Input Guardrails
        is_safe, error_msg = self.guardrails.validate_user_input(user_query)
        if not is_safe:
            return {
                "response": error_msg,
                "tool_calls": [],
                "citations": [],
                "prompt_version": self.prompt_version,
                "latency_sec": round(time.time() - start_time, 3),
                "status": "REJECTED_SECURITY",
            }

        # 2. Domain Applicability & Abstention Check
        is_supported, abstention_msg = self.guardrails.check_domain_applicability(user_query)
        if not is_supported:
            return {
                "response": abstention_msg,
                "tool_calls": [],
                "citations": [],
                "prompt_version": self.prompt_version,
                "latency_sec": round(time.time() - start_time, 3),
                "status": "ABSTAINED_OUT_OF_DOMAIN",
            }

        # 3. Dynamic Tool Coordination
        tool_calls = []
        citations = []
        context_blocks = []
        health_data = {}

        # Engine Telemetry Tool Call
        engine_id = self._extract_engine_id(user_query)
        if engine_id is not None:
            health = self.tools.get_engine_health(engine_id)
            tool_calls.append({"tool": "get_engine_health", "args": {"engine_id": engine_id}, "result": health})
            
            if "error" not in health:
                health_data = health
                sensors = self.tools.get_sensor_history(engine_id, window=20)
                tool_calls.append({"tool": "get_sensor_history", "args": {"engine_id": engine_id}, "result": sensors})
                
                context_blocks.append(
                    f"[Telemetry Engine {engine_id}]: "
                    f"RUL: {health['estimated_rul']} cycles (80% CI: [{health['uncertainty_interval_80']['q10']}, {health['uncertainty_interval_80']['q90']}]), "
                    f"Stage: {health['degradation_stage']}, "
                    f"Cycles Logged: {health['operational_cycles_logged']}, "
                    f"Sensor Trends: {sensors.get('trends', {})}"
                )

        # Knowledge Base Retrieval Tool Call
        doc_results = self.tools.search_technical_docs(user_query)
        tool_calls.append({"tool": "search_technical_docs", "args": {"query": user_query}, "count": len(doc_results)})

        for doc in doc_results:
            citation_str = f"{doc['document_title']} (Page {doc['page']})"
            citations.append(citation_str)
            context_blocks.append(f"[Documentation: {citation_str}]: {doc['text']}")

        # 4. Evidence Synthesis & Response Formulation
        if not context_blocks and engine_id is None:
            # No technical docs and no engine found
            response = (
                "I searched the technical knowledge base but found no verified documentation matching your query. "
                "Per AeroMind safety guidelines, I must abstain from generating ungrounded advice."
            )
            status = "ABSTAINED_NO_EVIDENCE"
        else:
            response = self._synthesize_response(user_query, engine_id, health_data, doc_results)
            status = "SUCCESS"

        # 5. Output Verification Guardrails
        citation_check = self.guardrails.verify_citations(response, doc_results)
        num_consistent = self.guardrails.verify_numerical_consistency(response, health_data)

        return {
            "response": response,
            "engine_id": engine_id,
            "tool_calls": tool_calls,
            "citations": list(set(citations)),
            "health_summary": health_data,
            "citation_verified": citation_check["passed"],
            "numerical_verified": num_consistent,
            "prompt_version": self.prompt_version,
            "latency_sec": round(time.time() - start_time, 3),
            "status": status,
        }

    def _synthesize_response(
        self,
        query: str,
        engine_id: Optional[int],
        health: Dict[str, Any],
        doc_results: List[Dict[str, Any]],
    ) -> str:
        """Grounds synthesis strictly in telemetry and retrieved documentation."""
        parts = []

        if engine_id is not None and health and "estimated_rul" in health:
            q10 = health["uncertainty_interval_80"]["q10"]
            q90 = health["uncertainty_interval_80"]["q90"]
            parts.append(
                f"### Engine #{engine_id} Diagnostic Assessment\n"
                f"- **Remaining Useful Life (RUL):** **{health['estimated_rul']} cycles** "
                f"(Calibrated 80% Confidence Interval: **[{q10}, {q90}] cycles**)\n"
                f"- **Degradation Stage:** **{health['degradation_stage']}** (Alert Level: {health['alert_level']})\n"
                f"- **Cycles Logged:** {health['operational_cycles_logged']} cycles\n"
            )

        if doc_results:
            parts.append("### Technical Documentation & Maintenance Guidance")
            for doc in doc_results[:2]:
                parts.append(
                    f"> *From **{doc['document_title']}** (Page {doc['page']}):*\n"
                    f"> \"{doc['text'][:280]}...\"\n"
                )

        if health.get("degradation_stage") == "Critical":
            parts.append(
                "**Urgent Engineering Recommendation:** Since predicted RUL is $\\le 30$ cycles, "
                "schedule immediate borescope inspection of HPC blades within 5 cycles as mandated by "
                "Turbofan Predictive Maintenance Standard Operating Procedures (Page 14)."
            )
        elif health.get("degradation_stage") == "Degraded":
            parts.append(
                "**Maintenance Recommendation:** Engine wear is elevated. "
                "Schedule compressor blade tip clearance monitoring and trend analysis within 15 cycles."
            )
        elif health.get("degradation_stage") == "Healthy":
            parts.append(
                "**Status:** Engine is operating within nominal operating bounds. No maintenance intervention required."
            )

        return "\n".join(parts)

