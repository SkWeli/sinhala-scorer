"""
Agent 4 — Explanation Agent
Generates a student-friendly, evidence-grounded explanation
based on the scoring results and retrieved knowledge base passages.
"""

import json
import ollama

from config.settings import OLLAMA_MODEL, OLLAMA_BASE_URL


EXPLANATION_PROMPT = """\
You are a warm, supportive Sri Lankan history teacher providing
detailed feedback to a student on their exam answer.

SCORING RESULT:
{scoring_json}

RELEVANT KNOWLEDGE BASE EVIDENCE:
{rag_context}

Write feedback using EXACTLY this structure:

**Overall Performance**
<2 sentences summarising the student's performance and total score.>

**What You Did Well ✅**
- <point 1>
- <point 2>
- <point 3 if applicable>

**Areas to Improve 📚**
- <specific gap with a constructive hint>
- <another gap with a hint referencing the knowledge base>
- <another if needed>

**Key Concepts for a Full-Mark Answer 🔑**
- <concept from ontology/knowledge base that was missing or incomplete>
- <another key concept>
- <another key concept>

Keep the tone encouraging and academic. Write in English.
"""


class ExplanationAgent:
    """
    Responsibilities:
      - Format scoring results and RAG context into a feedback prompt
      - Call OLLAMA to generate structured student feedback
      - Return the explanation as a formatted string
    """

    def run(self, scoring_result: dict, retrieval_result: dict) -> str:
        scoring_json = json.dumps(scoring_result, indent=2, ensure_ascii=False)

        # Use top 3 passages as evidence in the explanation
        passages = retrieval_result.get("rag_passages", [])[:3]
        if passages:
            rag_context = "\n".join(
                f"- [{p['source']}]: {p['content'][:250]}"
                for p in passages
            )
        else:
            rag_context = "No context available."

        prompt = EXPLANATION_PROMPT.format(
            scoring_json=scoring_json,
            rag_context=rag_context,
        )

        client   = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.generate(model=OLLAMA_MODEL, prompt=prompt)
        return response.get("response", "Explanation could not be generated.")