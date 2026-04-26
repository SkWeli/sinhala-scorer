"""
Agent 4 — Explanation Agent
Uses LLM only for a SHORT explanation paragraph.
Much faster than full scoring — typically 20-40 seconds.
"""

import ollama
from config.settings import OLLAMA_SINHALA_MODEL, OLLAMA_BASE_URL


EXPLANATION_PROMPT = """\
You are a Sri Lankan history teacher. A student answered a question and received this score:

Total Score: {total_score}/20
Criteria results:
{criteria_summary}

Top 2 knowledge base references:
{rag_context}

Write SHORT feedback (max 150 words) covering:
1. One sentence on overall performance
2. Two bullet points on what was done well
3. Two bullet points on what to improve

Write in English. Be encouraging.
"""


class ExplanationAgent:

    def run(self, scoring_result: dict, retrieval_result: dict) -> str:
        # Build a compact criteria summary
        lines = []
        for c in scoring_result.get("criterion_scores", []):
            status = "✅" if c["awarded"] >= c["max"] * 0.7 else "⚠️" if c["awarded"] > 0 else "❌"
            lines.append(f"{status} {c['criterion']}: {c['awarded']}/{c['max']}")
        criteria_summary = "\n".join(lines)

        # Only top 2 passages to keep prompt short
        passages = retrieval_result.get("rag_passages", [])[:2]
        rag_context = "\n".join(
            f"- [{p['source']}]: {p['content'][:150]}"
            for p in passages
        ) if passages else "No context."

        prompt = EXPLANATION_PROMPT.format(
            total_score=scoring_result.get("total_score", 0),
            criteria_summary=criteria_summary,
            rag_context=rag_context,
        )

        try:
            client   = ollama.Client(host=OLLAMA_BASE_URL)
            response = client.generate(
                model=OLLAMA_SINHALA_MODEL,
                prompt=prompt,
                options={
                    "num_predict": 250,   # short output — fast
                    "temperature": 0.7,
                },
            )
            return response.get("response", "Explanation unavailable.")
        except Exception as e:
            return f"Explanation unavailable: {e}"