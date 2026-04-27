"""
Agent 4 — Explanation Agent
Uses LLM only for a SHORT Sinhala feedback paragraph.

Purpose:
  - Does NOT score the answer
  - Only explains the already calculated score
  - Produces Sinhala feedback for the student
"""

import ollama
from config.settings import OLLAMA_SINHALA_MODEL, OLLAMA_BASE_URL


EXPLANATION_PROMPT = """\
ඔබ ශ්‍රී ලංකා ඉතිහාස ගුරුවරයෙකි.

සිසුවෙකුගේ පිළිතුරට ලැබුණු ලකුණු පහත පරිදි වේ:

මුළු ලකුණු: {total_score}/20

නිර්ණායක අනුව ලකුණු:
{criteria_summary}

වැරදි/අඩුපාඩු:
{misconceptions}

දැනුම් පදනමේ අදාළ කරුණු:
{rag_context}

කරුණාකර සිංහලෙන් කෙටි, පැහැදිලි feedback එකක් ලියන්න.

අනිවාර්ය නීති:
- සිංහලෙන් පමණක් ලියන්න.
- English වචන භාවිතා නොකරන්න.
- "Would you like", "I can", "Okay" වැනි වාක්‍ය නොලියන්න.
- උපරිම වචන 120ක් පමණක්.
- සිසුවා දිරිමත් කරන ආකාරයෙන් ලියන්න.
- අලුත් ලකුණු ගණනය නොකරන්න.
- ලබා දී ඇති ලකුණු පමණක් පැහැදිලි කරන්න.

ආකෘතිය:
සමස්ත අදහස: ...
හොඳ කරුණු:
- ...
- ...
වැඩි දියුණු කළ යුතු කරුණු:
- ...
- ...
"""


class ExplanationAgent:

    def _build_fallback_feedback(self, scoring_result: dict) -> str:
        """
        Sinhala fallback feedback if Ollama fails or returns empty output.
        """
        total_score = scoring_result.get("total_score", 0)
        criterion_scores = scoring_result.get("criterion_scores", [])
        misconceptions = scoring_result.get("misconceptions", [])

        strong = []
        weak = []

        for c in criterion_scores:
            criterion = c.get("criterion", "")
            awarded = c.get("awarded", 0)
            max_marks = c.get("max", 1)

            ratio = awarded / max_marks if max_marks else 0

            if ratio >= 0.7:
                strong.append(criterion)
            elif ratio < 0.5:
                weak.append(criterion)

        if total_score >= 15:
            overall = "ඔබගේ පිළිතුර ඉතා හොඳ මට්ටමක පවතී."
        elif total_score >= 10:
            overall = "ඔබගේ පිළිතුර මධ්‍යම මට්ටමේ හොඳ උත්සාහයකි."
        elif total_score >= 5:
            overall = "ඔබගේ පිළිතුරේ මූලික කරුණු කිහිපයක් තිබුණත් තවත් විස්තර අවශ්‍ය වේ."
        else:
            overall = "ඔබගේ පිළිතුර තවත් ශක්තිමත් කර ගැනීමට අවශ්‍ය වේ."

        strong_1 = strong[0] if len(strong) > 0 else "ප්‍රශ්නයට අදාළ මූලික කරුණු කිහිපයක් සඳහන් කර ඇත"
        strong_2 = strong[1] if len(strong) > 1 else "ඉතිහාස සිද්ධි පිළිබඳ යම් අවබෝධයක් පෙන්වා ඇත"

        weak_1 = weak[0] if len(weak) > 0 else "තවත් නිශ්චිත කරුණු හා වර්ෂ සඳහන් කිරීම"
        weak_2 = weak[1] if len(weak) > 1 else "කරුණු වඩාත් පැහැදිලිව හා සම්පූර්ණව විස්තර කිරීම"

        feedback = f"""සමස්ත අදහස: {overall} ඔබට {total_score}/20 ලැබී ඇත.

හොඳ කරුණු:
- {strong_1}
- {strong_2}

වැඩි දියුණු කළ යුතු කරුණු:
- {weak_1}
- {weak_2}"""

        if misconceptions:
            feedback += "\n\nනිවැරදි කළ යුතු කරුණ:\n"
            feedback += f"- {misconceptions[0]}"

        return feedback

    def run(self, scoring_result: dict, retrieval_result: dict) -> str:
        # Build compact criteria summary
        lines = []

        for c in scoring_result.get("criterion_scores", []):
            awarded = c.get("awarded", 0)
            max_marks = c.get("max", 1)
            criterion = c.get("criterion", "")

            if awarded >= max_marks * 0.7:
                status = "හොඳයි"
            elif awarded > 0:
                status = "භාගිකයි"
            else:
                status = "අඩුයි"

            lines.append(f"- {criterion}: {awarded}/{max_marks} ({status})")

        criteria_summary = "\n".join(lines)

        # Add misconceptions if available
        misconceptions_list = scoring_result.get("misconceptions", [])

        if misconceptions_list:
            misconceptions = "\n".join(f"- {m}" for m in misconceptions_list)
        else:
            misconceptions = "විශේෂ වැරදි කරුණු හඳුනාගෙන නැත."

        # Only top 2 passages to keep prompt short
        passages = retrieval_result.get("rag_passages", [])[:2]

        if passages:
            rag_context = "\n".join(
                f"- [{p.get('source', 'Unknown')}]: {p.get('content', '')[:250]}"
                for p in passages
            )
        else:
            rag_context = "දැනුම් පදනමෙන් අදාළ කරුණු නොලැබුණි."

        prompt = EXPLANATION_PROMPT.format(
            total_score=scoring_result.get("total_score", 0),
            criteria_summary=criteria_summary,
            misconceptions=misconceptions,
            rag_context=rag_context,
        )

        try:
            client = ollama.Client(host=OLLAMA_BASE_URL)

            response = client.generate(
                model=OLLAMA_SINHALA_MODEL,
                prompt=prompt,
                options={
                    "num_predict": 220,
                    "temperature": 0.3,
                    "top_p": 0.8,
                },
            )

            explanation = response.get("response", "").strip()

            if not explanation:
                return self._build_fallback_feedback(scoring_result)

            # Block unwanted English-style endings
            banned_phrases = [
                "would you like",
                "i can",
                "okay",
                "here’s",
                "here is",
                "feedback for your student",
            ]

            explanation_lower = explanation.lower()

            if any(phrase in explanation_lower for phrase in banned_phrases):
                return self._build_fallback_feedback(scoring_result)

            return explanation

        except Exception:
            return self._build_fallback_feedback(scoring_result)