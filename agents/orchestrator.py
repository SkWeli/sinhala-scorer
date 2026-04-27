"""
Orchestrator Agent — Master coordinator.
Runs all four sub-agents in sequence and returns a unified result dict.

Pipeline:
  1. RetrievalAgent   → RAG passages + ontology concepts
  2. CoverageAgent    → Lightweight criterion coverage flags
  3. ScoringAgent     → Rule-based / coverage-based per-criterion scoring
  4. ExplanationAgent → Lightweight Ollama explanation + fallback feedback
"""

from agents.retrieval_agent import RetrievalAgent
from agents.coverage_agent import CoverageAgent
from agents.scoring_agent import ScoringAgent
from agents.explanation_agent import ExplanationAgent


class OrchestratorAgent:
    def __init__(self):
        self.retrieval = RetrievalAgent()
        self.coverage = CoverageAgent()
        self.scoring = ScoringAgent()
        self.explanation = ExplanationAgent()

    def _normalise_retrieval_result(self, retrieval_result: dict) -> dict:
        """
        Ensures retrieval output always has the expected structure.
        This prevents scoring/explanation agents from crashing if retrieval
        returns slightly different key names.
        """

        if not isinstance(retrieval_result, dict):
            return {
                "rag_passages": [],
                "ontology_concepts": [],
            }

        return {
            "rag_passages": retrieval_result.get("rag_passages", []),
            "ontology_concepts": retrieval_result.get("ontology_concepts", []),
        }

    def score_answer(self, question: dict, student_answer: str) -> dict:
        """
        Full multi-agent scoring pipeline.

        Args:
            question: One entry from QUESTIONS dict.
                      Expected keys: id, text, era, guide
            student_answer: Raw Sinhala text from the student.

        Returns:
            Unified result dict with retrieval, coverage, scoring,
            explanation, final_score, and max_score.
        """

        topic = question.get("era", "")
        question_text = question.get("text", "")
        guide = question.get("guide", {})

        marking_guide_list = [
            {"criterion": criterion, "marks": marks}
            for criterion, marks in guide.items()
        ]

        normalised_question = {
            "id": question.get("id"),
            "text": question_text,
            "era": topic,
            "guide": guide,
            "marking_guide": marking_guide_list,
        }

        try:
            # ── Step 1: Retrieval ─────────────────────────────────────────────
            print("[Orchestrator] Step 1: Retrieval…", flush=True)

            retrieval_result = self.retrieval.run(
                question_text=question_text,
                student_answer=student_answer,
                topic=topic,
            )

            retrieval_result = self._normalise_retrieval_result(retrieval_result)

            print(
                f"[Orchestrator] Retrieved "
                f"{len(retrieval_result['rag_passages'])} RAG passage(s) and "
                f"{len(retrieval_result['ontology_concepts'])} ontology concept(s).",
                flush=True,
            )

            # ── Step 2: Coverage Check ────────────────────────────────────────
            print("[Orchestrator] Step 2: Coverage check…", flush=True)

            coverage_result = self.coverage.run(
                student_answer,
                marking_guide_list,
                topic,
            )

            # ── Step 3: Rule-based Scoring ───────────────────────────────────
            print("[Orchestrator] Step 3: Scoring…", flush=True)

            scoring_result = self.scoring.run(
                normalised_question,
                student_answer,
                retrieval_result,
                coverage_result,
            )

            # ── Step 4: Lightweight Explanation ──────────────────────────────
            print("[Orchestrator] Step 4: Explanation…", flush=True)

            explanation = self.explanation.run(
                scoring_result,
                retrieval_result,
            )

            final_score = scoring_result.get("total_score", 0)

            return {
                "success": True,
                "question_id": question.get("id"),
                "question_text": question_text,
                "era": topic,
                "student_answer": student_answer,
                "retrieval": retrieval_result,
                "coverage": coverage_result,
                "scoring": scoring_result,
                "explanation": explanation,
                "final_score": final_score,
                "max_score": 20,
            }

        except Exception as e:
            print("[Orchestrator] ERROR:", type(e).__name__, e, flush=True)

            return {
                "success": False,
                "question_id": question.get("id"),
                "question_text": question_text,
                "era": topic,
                "student_answer": student_answer,
                "retrieval": {
                    "rag_passages": [],
                    "ontology_concepts": [],
                },
                "coverage": {},
                "scoring": {
                    "total_score": 0,
                    "breakdown": [],
                },
                "explanation": f"Scoring failed due to an internal error: {e}",
                "final_score": 0,
                "max_score": 20,
                "error": str(e),
            }