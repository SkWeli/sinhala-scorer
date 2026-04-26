"""
Agent 2 — Coverage Agent
Performs a fast, lightweight keyword-based check to determine which
marking-guide criteria are likely addressed in the student's answer.
This runs BEFORE the expensive LLM call as a pre-screening signal.
"""


class CoverageAgent:
    """
    Uses simple token overlap between the student answer and each criterion
    description to flag whether a criterion has surface-level coverage.
    """

    # Domain keywords per colonial power topic
    _DOMAIN_KEYWORDS = {
        "portuguese": [
            "පෘතුගීසි", "portuguese", "කොත්තේ", "kotte",
            "cinnamon", "කිතුනු", "fort", "fortress", "goa",
        ],
        "dutch": [
            "ලන්දේසි", "dutch", "voc", "roman", "rajakariya",
            "රාජකාරිය", "batavia", "spice",
        ],
        "british": [
            "බ්‍රිතාන්‍ය", "british", "colebrooke", "cameron",
            "plantation", "railway", "වතු", "kandy",
        ],
        "transition": [
            "portuguese", "dutch", "compare", "contrast",
            "policy", "1638", "1658",
        ],
        "general": [
            "colonial", "administration", "reform", "trade",
            "law", "church", "economy", "religion",
        ],
    }

    def run(
        self,
        student_answer: str,
        marking_guide: list,
        topic: str,
    ) -> dict:
        answer_lower = student_answer.lower()

        topic_key = topic.lower()
        topic_kws = (
            self._DOMAIN_KEYWORDS.get(topic_key, [])
            + self._DOMAIN_KEYWORDS["general"]
        )

        criterion_flags = []
        for criterion in marking_guide:
            # Check if any meaningful word from criterion text is in the answer
            crit_words = [
                w for w in criterion["criterion"].lower().split()
                if len(w) > 4
            ]
            matched = any(w in answer_lower for w in crit_words)
            criterion_flags.append({
                "criterion":      criterion["criterion"],
                "max_marks":      criterion["marks"],
                "surface_covered": matched,
            })

        covered_count  = sum(1 for c in criterion_flags if c["surface_covered"])
        coverage_ratio = covered_count / len(criterion_flags) if criterion_flags else 0

        topic_hits = [k for k in topic_kws if k.lower() in answer_lower]

        return {
            "criterion_flags":      criterion_flags,
            "coverage_ratio":       coverage_ratio,
            "topic_keywords_found": topic_hits,
        }