"""
Orchestrator Agent — Master coordinator.
Runs all four sub-agents in sequence and returns a unified result dict.

Pipeline:
  1. RetrievalAgent   → RAG passages + ontology concepts
  2. CoverageAgent    → Lightweight criterion coverage flags
  3. ScoringAgent     → LLM-based per-criterion scoring
  4. ExplanationAgent → Student-friendly feedback
"""

from agents.retrieval_agent   import RetrievalAgent
from agents.coverage_agent    import CoverageAgent
from agents.scoring_agent     import ScoringAgent
from agents.explanation_agent import ExplanationAgent


class OrchestratorAgent:

    def __init__(self):
        self.retrieval   = RetrievalAgent()
        self.coverage    = CoverageAgent()
        self.scoring     = ScoringAgent()
        self.explanation = ExplanationAgent()

    def score_answer(self, question: dict, student_answer: str) -> dict:
        """
        Full multi-agent scoring pipeline.

        Args:
            question:       One entry from questions.QUESTIONS
            student_answer: Raw Sinhala text from the student

        Returns:
            Unified result dict with retrieval, coverage, scoring,
            explanation, final_score, and max_score.
        """
        topic = question.get("topic", "")

        # ── Step 1: Retrieval ─────────────────────────────────────────────────
        retrieval_result = self.retrieval.run(question["question_en"], topic)

        # ── Step 2: Coverage Check (fast, pre-LLM) ────────────────────────────
        coverage_result = self.coverage.run(
            student_answer,
            question["marking_guide"],
            topic,
        )

        # ── Step 3: LLM Scoring ───────────────────────────────────────────────
        scoring_result = self.scoring.run(
            question,
            student_answer,
            retrieval_result,
        )

        # ── Step 4: Explanation ───────────────────────────────────────────────
        explanation = self.explanation.run(scoring_result, retrieval_result)

        return {
            "question_id":    question["id"],
            "question_en":    question["question_en"],
            "question_si":    question["question_si"],
            "student_answer": student_answer,
            "retrieval":      retrieval_result,
            "coverage":       coverage_result,
            "scoring":        scoring_result,
            "explanation":    explanation,
            "final_score":    scoring_result.get("total_score", 0),
            "max_score":      20,
        }