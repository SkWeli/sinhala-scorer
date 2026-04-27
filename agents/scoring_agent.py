"""
Agent 3 — Scoring Agent (Rule-Based)

Scores each criterion deterministically using:
  - Student answer concept coverage
  - Sinhala/English synonym matching
  - RAG passage support
  - Ontology concept support

No LLM call — fast, reliable, explainable.
"""

import re
from config.settings import MAX_MARKS


class ScoringAgent:
    """
    Rule-based criterion scorer.

    Important design:
    - Marks are mainly awarded from the STUDENT ANSWER.
    - RAG and ontology are used as support evidence only.
    - This prevents giving marks just because the knowledge base contains facts.
    """

    def _normalise(self, text: str) -> str:
        if not text:
            return ""

        text = text.lower()
        text = text.replace("\u200d", "")  # zero-width joiner
        text = text.replace("\u200c", "")  # zero-width non-joiner
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _contains_any(self, text: str, aliases: list[str]) -> bool:
        text = self._normalise(text)

        for alias in aliases:
            alias = self._normalise(alias)
            if alias and alias in text:
                return True

        return False

    def _criterion_groups(self, criterion: str) -> list[list[str]]:
        """
        Converts a marking criterion into concept groups.

        A group is satisfied if the student answer contains ANY alias
        inside that group.

        Example:
            ["කෝට්ටේ", "kotte"] means either Sinhala or English spelling is enough.
        """

        c = self._normalise(criterion)

        # Q1 Portuguese political situation / three kingdoms
        if (
            "රාජධානි 3" in c
            or "රාජධානි තුන" in c
            or ("කෝට්ටේ" in c and "කඳුරට" in c and "යාපනය" in c)
        ):
            return [
                ["රාජධානි", "රාජ්‍ය", "kingdom", "kingdoms", "තුන", "3"],
                ["කෝට්ටේ", "kotte"],
                ["කඳුරට", "උඩරට", "kandy", "kandyan"],
                ["යාපනය", "jaffna", "yapane"],
            ]

        # Portuguese arrival in 1505
        if (
            "ලෝරෙන්සෝ" in c
            or "අල්මේදා" in c
            or "1505" in c
            or "පැමිණීම" in c
        ):
            return [
                ["1505"],
                ["ලෝරෙන්සෝ", "lorenço", "lorenzo", "lourenco"],
                ["අල්මේදා", "almeida"],
                ["පැමිණ", "ආවේ", "ආ", "arrival", "arrived"],
                ["අහඹු", "කුණාටුව", "accidental", "storm"],
            ]

        # European spice/cinnamon trade demand
        if (
            "කුරුඳු" in c
            or "වෙළඳ" in c
            or "කුළු බඩු" in c
            or "කොළු බඩු" in c
            or "ඉල්ලුම" in c
        ):
            return [
                ["කුරුඳු", "cinnamon"],
                ["වෙළඳ", "වෙළඳාම", "trade", "trading"],
                ["ඉල්ලුම", "උනන්දු", "අවධානය", "demand", "interest"],
                ["යුරෝපා", "යුරෝපීය", "europe", "european"],
                ["කුළු බඩු", "කොළු බඩු", "spice", "spices"],
            ]

        # Kotte political structure / 8th Jayasinha or related king
        if (
            "කෝට්ටේ" in c
            and ("8" in c or "ජ්‍යෙෂ්ඨ" in c or "රජු" in c)
        ):
            return [
                ["කෝට්ටේ", "kotte"],
                ["8", "අට", "අටවන", "viii"],
                ["ජ්‍යෙෂ්ඨ", "jayasinha", "jayasimha"],
                ["රජු", "king", "පාලනය"],
                ["ව්‍යූහ", "රාජ්‍ය ව්‍යූහය", "political structure"],
            ]

        # Dutch economic systems
        if "ලන්දේසි" in c or "dutch" in c:
            groups = []

            if "voc" in c or "වොක්" in c:
                groups.append(["voc", "ලන්දේසි පෙරදිග ඉන්දියා සමාගම"])
            if "කුරුඳු" in c or "cinnamon" in c:
                groups.append(["කුරුඳු", "cinnamon", "මහබද්ද", "mahabadde"])
            if "තෝම්බු" in c or "thombu" in c:
                groups.append(["තෝම්බු", "thombu", "ඉඩම් ලේඛන"])
            if "බදු" in c or "tax" in c:
                groups.append(["බදු", "tax", "taxation"])
            if "ඇළ" in c or "canal" in c:
                groups.append(["ඇළ", "canal", "හැමිල්ටන්", "hamilton"])

            if groups:
                return groups

        # British reforms / economy / administration
        if "බ්‍රිතාන්‍ය" in c or "british" in c:
            groups = []

            if "කෝල්බෲක්" in c or "colebrooke" in c:
                groups.append(["කෝල්බෲක්", "colebrooke"])
            if "කැමරන්" in c or "cameron" in c:
                groups.append(["කැමරන්", "cameron"])
            if "වැවිලි" in c or "plantation" in c:
                groups.append(["වැවිලි", "plantation", "තේ", "tea", "කෝපි", "coffee"])
            if "රාජකාරිය" in c:
                groups.append(["රාජකාරිය", "rajkariya", "rajakariya"])
            if "1833" in c:
                groups.append(["1833"])
            if "1815" in c:
                groups.append(["1815", "උඩරට ගිවිසුම", "kandyan convention"])

            if groups:
                return groups

        # Generic fallback for other criteria
        tokens = re.findall(r"[a-zA-Z0-9\u0D80-\u0DFF]+", c)

        stopwords = {
            "හා",
            "සහ",
            "යන",
            "වූ",
            "විය",
            "කරුණු",
            "විස්තර",
            "කරන්න",
            "කිරීම",
            "ලෙස",
            "තුළ",
            "එම",
            "මෙම",
            "the",
            "and",
            "with",
        }

        keywords = []
        for token in tokens:
            if len(token) >= 3 and token not in stopwords:
                keywords.append(token)

        # Remove duplicates while preserving order
        seen = set()
        groups = []

        for keyword in keywords:
            if keyword not in seen:
                groups.append([keyword])
                seen.add(keyword)

        return groups

    def _support_ratio(self, criterion_groups: list[list[str]], text: str) -> float:
        if not criterion_groups:
            return 0.0

        hits = sum(
            1 for group in criterion_groups
            if self._contains_any(text, group)
        )

        return hits / len(criterion_groups)

    def _detect_misconceptions(self, student_answer: str, topic: str) -> list[str]:
        """
        Detects serious factual errors.
        Used only as a small final penalty.
        """

        answer = self._normalise(student_answer)
        topic = self._normalise(topic)

        misconceptions = []

        # In Q1, saying Portuguese introduced Rajakariya is misleading.
        # Rajakariya existed before; British later abolished it.
        if (
            ("පෘතුගීසි" in topic or "portuguese" in topic)
            and "රාජකාරිය" in answer
            and ("හඳුන්වා" in answer or "ඇති කළ" in answer or "introduced" in answer)
        ):
            misconceptions.append(
                "රාජකාරිය ක්‍රමය පෘතුගීසීන් විසින් හඳුන්වා දුන්නා ලෙස දැක්වීම නිවැරදි නොවේ."
            )

        return misconceptions

    def run(
        self,
        question: dict,
        student_answer: str,
        retrieval_result: dict,
        coverage_result: dict,
    ) -> dict:

        answer_text = self._normalise(student_answer)
        topic = question.get("era", "")

        rag_passages = retrieval_result.get("rag_passages", [])
        onto_concepts = retrieval_result.get("ontology_concepts", [])

        rag_text = self._normalise(
            " ".join(p.get("content", "") for p in rag_passages)
        )

        onto_text = self._normalise(" ".join(str(c) for c in onto_concepts))

        criterion_scores = []

        for flag in coverage_result.get("criterion_flags", []):
            criterion = flag.get("criterion", "")
            max_marks = int(flag.get("max_marks", 0))

            concept_groups = self._criterion_groups(criterion)

            if not concept_groups or max_marks <= 0:
                criterion_scores.append({
                    "criterion": criterion,
                    "awarded": 0,
                    "max": max_marks,
                    "reason": "මෙම නිර්ණායකය සඳහා ලකුණු ගණනය කළ නොහැකි විය.",
                })
                continue

            matched_groups = [
                group for group in concept_groups
                if self._contains_any(answer_text, group)
            ]

            answer_hits = len(matched_groups)
            total_concepts = len(concept_groups)
            answer_ratio = answer_hits / total_concepts

            rag_ratio = self._support_ratio(concept_groups, rag_text)
            onto_ratio = self._support_ratio(concept_groups, onto_text)

            # Marks come from the answer itself.
            raw_score = answer_ratio * max_marks

            # Small support bonus only when answer has at least some content.
            # This avoids giving marks only because the KB has the answer.
            if answer_hits > 0 and rag_ratio >= 0.5:
                raw_score += 0.05 * max_marks

            if answer_hits > 0 and onto_ratio >= 0.4:
                raw_score += 0.05 * max_marks

            awarded = round(raw_score)

            # Ensure one correct concept gets at least 1 mark.
            if answer_hits > 0:
                awarded = max(1, awarded)

            # Prevent one tiny keyword from receiving too many marks.
            if answer_hits == 1 and total_concepts >= 4:
                awarded = min(awarded, 2)

            awarded = max(0, min(awarded, max_marks))

            if awarded == max_marks:
                reason = (
                    f"සම්පූර්ණයෙන් ආවරණය කර ඇත. "
                    f"ප්‍රධාන සංකල්ප {answer_hits}/{total_concepts}ක් පිළිතුරේ ඇත."
                )
            elif awarded >= max_marks * 0.6:
                reason = (
                    f"හොඳින්/භාගිකව ආවරණය කර ඇත. "
                    f"ප්‍රධාන සංකල්ප {answer_hits}/{total_concepts}ක් පිළිතුරේ ඇත."
                )
            elif awarded > 0:
                reason = (
                    f"අඩු මට්ටමින් ආවරණය කර ඇත. "
                    f"ප්‍රධාන සංකල්ප {answer_hits}/{total_concepts}ක් පමණක් ඇත."
                )
            else:
                reason = (
                    "ආවරණය කර නොමැත. මෙම නිර්ණායකයට අදාළ ප්‍රධාන කරුණු "
                    "පිළිතුරේ නොමැත."
                )

            if rag_ratio >= 0.5:
                reason += " දැනුම් පදනමෙන් මෙම නිර්ණායකයට සහාය ලැබේ."

            if onto_ratio >= 0.4:
                reason += " Ontology සංකල්පද ගැළපේ."

            criterion_scores.append({
                "criterion": criterion,
                "awarded": awarded,
                "max": max_marks,
                "reason": reason,
                "matched_concepts": answer_hits,
                "total_concepts": total_concepts,
            })

        raw_total = sum(c["awarded"] for c in criterion_scores)

        misconceptions = self._detect_misconceptions(student_answer, topic)
        penalty = min(len(misconceptions), 2)

        total = max(0, min(raw_total - penalty, MAX_MARKS))

        return {
            "criterion_scores": criterion_scores,
            "total_score": total,
            "raw_score_before_penalty": raw_total,
            "penalty": penalty,
            "misconceptions": misconceptions,
            "overall_comment": "",
        }