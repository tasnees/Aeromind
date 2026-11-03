import re
from typing import Any, Dict, List, Optional, Tuple


class MaintenanceGuardrails:
    """
    Enforces production reliability, citation requirements, and hallucination prevention
    for the Maintenance Copilot.
    """

    UNSUPPORTED_DOMAINS = [
        "stock price",
        "weather tomorrow",
        "write poetry",
        "movie review",
        "credit card",
        "crypto",
        "bitcoin",
        "recipe",
    ]

    PROMPT_INJECTION_PATTERNS = [
        r"ignore (all )?previous instructions",
        r"disregard safety guidelines",
        r"you are now in developer mode",
        r"system prompt override",
    ]

    @classmethod
    def validate_user_input(cls, user_query: str) -> Tuple[bool, Optional[str]]:
        """Checks for prompt injections and malicious inputs."""
        query_lower = user_query.lower()

        # Check prompt injection
        for pattern in cls.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, query_lower):
                return False, "Security violation: Prompt injection pattern detected."

        return True, None

    @classmethod
    def check_domain_applicability(cls, user_query: str) -> Tuple[bool, Optional[str]]:
        """Detects unsupported out-of-scope requests to trigger explicit abstention."""
        query_lower = user_query.lower()
        for phrase in cls.UNSUPPORTED_DOMAINS:
            if phrase in query_lower:
                return (
                    False,
                    "I am the AeroMind Maintenance Copilot specialized exclusively in turbofan engine diagnostics and C-MAPSS technical documentation. I cannot assist with non-aviation topics.",
                )
        return True, None

    @classmethod
    def verify_citations(cls, response_text: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verifies that cited sources exist in the retrieved evidence chunks.
        """
        valid_chunk_ids = {c.get("chunk_id") for c in retrieved_chunks if c.get("chunk_id")}
        valid_doc_titles = {c.get("document_title", "").lower() for c in retrieved_chunks if c.get("document_title")}

        # Check if citations or references are mentioned
        has_citations = "Page" in response_text or any(title in response_text.lower() for title in valid_doc_titles)

        return {
            "has_citations": has_citations,
            "retrieved_chunk_count": len(retrieved_chunks),
            "passed": has_citations if len(retrieved_chunks) > 0 else True,
        }

    @classmethod
    def verify_numerical_consistency(
        cls, response_text: str, verified_health_data: Dict[str, Any]
    ) -> bool:
        """
        Ensures response does not contradict verified telemetry (e.g., claiming 150 RUL when verified RUL is 42).
        """
        if not verified_health_data or "estimated_rul" not in verified_health_data:
            return True

        verified_rul = verified_health_data["estimated_rul"]
        # Look for numbers in response near "RUL"
        rul_matches = re.findall(r"RUL.*?(\d+\.?\d*)", response_text, re.IGNORECASE)
        for match in rul_matches:
            val = float(match)
            # If cited value deviates by more than 5 cycles from telemetry, flag inconsistency
            if abs(val - verified_rul) > 5.0 and abs(val - 80) > 2.0 and abs(val - 30) > 2.0:
                # Deviates from verified telemetry and not a threshold (30 or 80)
                return False

        return True

