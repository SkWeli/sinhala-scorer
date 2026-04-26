"""
Agent 3 — Scoring Agent
Calls the local OLLAMA model with a structured prompt to score
the student answer criterion-by-criterion and return JSON results.
"""

import json
import re
import ollama

from config.settings import OLLAMA_MODEL, OLLAMA_BASE_URL, MAX_MARKS


# ── Prompt Template ────────────────────────────────────────────────────────────
SCORING_PROMPT = """\
You are an expert Sri Lankan history examiner with deep knowledge of the
Colonial period (Portuguese, Dutch, and British rule).

QUESTION:
{question}

MARKING GUIDE (each criterion has a maximum mark allocation):
{marking_guide_text}

RETRIEVED REFERENCE CONTEXT (from the knowledge base):
{rag_context}

ONTOLOGY CONCEPTS (key historical entities and relationships):
{ontology_concepts}

STUDENT ANSWER (written in Sinhala — evaluate meaning, not just keywords):
{student_answer}

TASK:
For each criterion in the marking guide:
  1. Decide how many marks to award (0 to max for that criterion).
  2. Provide a brief justification citing specific evidence from the
     student's answer AND/OR the retrieved context.

Return ONLY valid JSON — no markdown fences, no extra text:
{{
  "criterion_scores": [
    {{
      "criterion": "<exact criterion text>",
      "awarded": <integer>,
      "max": <integer>,
      "reason": "<1-2 sentence justification in English>"
    }}
  ],
  "total_score": <integer 0-20>,
  "overall_comment": "<2-3 sentence overall assessment in English>"
}}
"""


class ScoringAgent:
    """
    Responsibilities:
      - Format the structured scoring prompt
      - Call OLLAMA (local inference) for criterion-level scoring
      - Parse and validate the JSON response
      - Provide a safe fallback if parsing fails
    """

    def run(
        self,
        question: dict,
        student_answer: str,
        retrieval_result: dict,
    ) -> dict:

        # ── Format marking guide ──────────────────────────────────────────────
        mg_lines = [
            f"{i+1}. {c['criterion']} [{c['marks']} marks]"
            for i, c in enumerate(question["marking_guide"])
        ]
        mg_text = "\n".join(mg_lines)

        # ── Format RAG passages ───────────────────────────────────────────────
        passages = retrieval_result.get("rag_passages", [])
        if passages:
            rag_text = "\n---\n".join(
                f"[Source: {p['source']}]\n{p['content']}"
                for p in passages
            )
        else:
            rag_text = "No relevant context retrieved."

        # ── Format ontology concepts ──────────────────────────────────────────
        onto_items = retrieval_result.get("ontology_concepts", [])
        onto_text  = "\n".join(onto_items) if onto_items else "No ontology concepts matched."

        # ── Build prompt ──────────────────────────────────────────────────────
        prompt = SCORING_PROMPT.format(
            question=question["question_en"],
            marking_guide_text=mg_text,
            rag_context=rag_text,
            ontology_concepts=onto_text,
            student_answer=student_answer,
        )

        # ── Call OLLAMA ───────────────────────────────────────────────────────
        client   = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.generate(model=OLLAMA_MODEL, prompt=prompt)
        raw      = response.get("response", "")

        # ── Parse JSON ────────────────────────────────────────────────────────
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())

                # Clamp individual awarded marks to their max
                for cs in result.get("criterion_scores", []):
                    cs["awarded"] = max(0, min(cs["awarded"], cs["max"]))

                # Recompute total from criterion scores
                computed_total = sum(
                    cs["awarded"]
                    for cs in result.get("criterion_scores", [])
                )
                result["total_score"] = min(computed_total, MAX_MARKS)
                return result

            except (json.JSONDecodeError, KeyError):
                pass  # fall through to fallback

        # ── Fallback ──────────────────────────────────────────────────────────
        return {
            "criterion_scores": [
                {
                    "criterion": c["criterion"],
                    "awarded":   0,
                    "max":       c["marks"],
                    "reason":    "Could not parse LLM response.",
                }
                for c in question["marking_guide"]
            ],
            "total_score":     0,
            "overall_comment": (
                f"Scoring failed due to response parsing error. "
                f"Raw output (first 300 chars): {raw[:300]}"
            ),
        }