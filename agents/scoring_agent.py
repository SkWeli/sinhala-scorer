"""
Agent 3 — Scoring Agent (Rule-Based)
Scores each criterion deterministically using:
  - Coverage agent flags
  - RAG passage keyword overlap
  - Ontology concept matching
No LLM call — fast, reliable, explainable.
"""

from config.settings import MAX_MARKS


class ScoringAgent:
    """
    Rule-based criterion scorer.
    Awards marks based on evidence found in:
      1. Student answer keyword overlap with criterion
      2. RAG retrieved passages matching criterion
      3. Ontology concepts matched
    """

    def run(
        self,
        question: dict,
        student_answer: str,
        retrieval_result: dict,
        coverage_result: dict,
    ) -> dict:

        answer_lower   = student_answer.lower()
        rag_passages   = retrieval_result.get("rag_passages", [])
        onto_concepts  = retrieval_result.get("ontology_concepts", [])
        onto_text      = " ".join(onto_concepts).lower()
        rag_text       = " ".join(p["content"] for p in rag_passages).lower()

        criterion_scores = []

        for flag in coverage_result.get("criterion_flags", []):
            criterion  = flag["criterion"]
            max_marks  = flag["max_marks"]
            crit_lower = criterion.lower()

            # Extract meaningful words from criterion (len > 4)
            crit_words = [w for w in crit_lower.split() if len(w) > 4]

            # ── Signal 1: Direct keyword hit in student answer ────────────────
            answer_hits = sum(1 for w in crit_words if w in answer_lower)
            answer_ratio = answer_hits / len(crit_words) if crit_words else 0

            # ── Signal 2: RAG passage supports the criterion ──────────────────
            rag_hits   = sum(1 for w in crit_words if w in rag_text)
            rag_ratio  = rag_hits / len(crit_words) if crit_words else 0

            # ── Signal 3: Ontology concept matched ───────────────────────────
            onto_hit   = any(w in onto_text for w in crit_words)

            # ── Scoring formula ───────────────────────────────────────────────
            # Base: answer coverage drives the score
            # Bonus: RAG + ontology confirm the concept exists
            base_score = answer_ratio * max_marks

            # Bonus up to 20% extra if RAG confirms + ontology matches
            bonus = 0
            if rag_ratio > 0.3:
                bonus += 0.1 * max_marks
            if onto_hit:
                bonus += 0.1 * max_marks

            raw_score = min(base_score + bonus, max_marks)
            awarded   = round(raw_score)
            awarded   = max(0, min(awarded, max_marks))

            # ── Build reason ──────────────────────────────────────────────────
            if awarded == max_marks:
                reason = f"Fully addressed. Answer contains {answer_hits}/{len(crit_words)} key terms."
            elif awarded >= max_marks * 0.6:
                reason = f"Partially addressed. {answer_hits}/{len(crit_words)} key terms found in answer."
            elif awarded > 0:
                reason = f"Weakly addressed. Only {answer_hits}/{len(crit_words)} key terms present."
            else:
                reason = f"Not addressed. None of the expected key terms found in answer."

            if onto_hit:
                reason += " Ontology concept confirmed."
            if rag_ratio > 0.3:
                reason += " Supported by knowledge base."

            criterion_scores.append({
                "criterion": criterion,
                "awarded":   awarded,
                "max":       max_marks,
                "reason":    reason,
            })

        total = min(sum(c["awarded"] for c in criterion_scores), MAX_MARKS)

        return {
            "criterion_scores": criterion_scores,
            "total_score":      total,
            "overall_comment":  "",   # filled by ExplanationAgent
        }