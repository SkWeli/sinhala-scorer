"""
Agent 2 — Coverage Agent

Performs a fast, lightweight rule-based check to determine which
marking-guide criteria are likely addressed in the student's answer.

This agent does NOT award final marks.
It only gives surface-level coverage signals for the ScoringAgent and UI.
"""


import re


class CoverageAgent:
    """
    Uses Sinhala/English keyword and concept matching to detect whether
    each marking-guide criterion is touched by the student's answer.
    """

    _DOMAIN_KEYWORDS = {
        "portuguese": [
            "පෘතුගීසි",
            "portuguese",
            "කෝට්ටේ",
            "kotte",
            "කුරුඳු",
            "cinnamon",
            "ලෝරෙන්සෝ",
            "අල්මේදා",
            "almeida",
            "1505",
            "කිතුනු",
            "fort",
            "goa",
        ],
        "dutch": [
            "ලන්දේසි",
            "dutch",
            "voc",
            "තෝම්බු",
            "thombu",
            "කුරුඳු",
            "cinnamon",
            "මහබද්ද",
            "rajakariya",
            "රාජකාරිය",
            "batavia",
            "spice",
        ],
        "british": [
            "බ්‍රිතාන්‍ය",
            "british",
            "කෝල්බෲක්",
            "colebrooke",
            "කැමරන්",
            "cameron",
            "වැවිලි",
            "plantation",
            "railway",
            "දුම්රිය",
            "kandy",
            "උඩරට",
        ],
        "transition": [
            "portuguese",
            "dutch",
            "british",
            "compare",
            "contrast",
            "policy",
            "1638",
            "1658",
            "1796",
        ],
        "general": [
            "colonial",
            "යටත්",
            "පාලනය",
            "administration",
            "reform",
            "trade",
            "වෙළඳ",
            "law",
            "නීතිය",
            "church",
            "religion",
            "ආගම",
            "economy",
            "ආර්ථික",
        ],
    }

    def _normalise(self, text: str) -> str:
        if not text:
            return ""

        text = text.lower()
        text = text.replace("\u200d", "")
        text = text.replace("\u200c", "")
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _contains_any(self, text: str, aliases: list[str]) -> bool:
        text = self._normalise(text)

        for alias in aliases:
            alias = self._normalise(alias)
            if alias and alias in text:
                return True

        return False

    def _matched_aliases(self, text: str, aliases: list[str]) -> list[str]:
        text = self._normalise(text)
        found = []

        for alias in aliases:
            alias_norm = self._normalise(alias)
            if alias_norm and alias_norm in text:
                found.append(alias)

        return found

    def _criterion_groups(self, criterion: str) -> list[list[str]]:
        """
        Convert criterion text into concept groups.

        One concept group can contain multiple accepted aliases.
        Example:
            ["කෝට්ටේ", "kotte"]
        """

        c = self._normalise(criterion)

        # Q1 — Political situation / three kingdoms
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

        # Q1 — Lourenço de Almeida arrival
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
                ["පැමිණ", "ආවේ", "arrival", "arrived"],
                ["අහඹු", "කුණාටුව", "accidental", "storm"],
            ]

        # Q1 — European spice/cinnamon trade demand
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

        # Q1 — Kotte state structure / 8th Jayasinha king
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

        # Dutch criteria
        if "ලන්දේසි" in c or "dutch" in c:
            groups = []

            if "voc" in c:
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

        # British criteria
        if "බ්‍රිතාන්‍ය" in c or "british" in c:
            groups = []

            if "කෝල්බෲක්" in c or "colebrooke" in c:
                groups.append(["කෝල්බෲක්", "colebrooke"])

            if "කැමරන්" in c or "cameron" in c:
                groups.append(["කැමරන්", "cameron"])

            if "වැවිලි" in c or "plantation" in c:
                groups.append(["වැවිලි", "plantation", "තේ", "tea", "කෝපි", "coffee"])

            if "රාජකාරිය" in c:
                groups.append(["රාජකාරිය", "rajakariya", "rajkariya"])

            if "1833" in c:
                groups.append(["1833"])

            if "1815" in c:
                groups.append(["1815", "උඩරට ගිවිසුම", "kandyan convention"])

            if groups:
                return groups

        # Generic fallback
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
            "about",
            "explain",
        }

        groups = []
        seen = set()

        for token in tokens:
            if len(token) >= 3 and token not in stopwords and token not in seen:
                groups.append([token])
                seen.add(token)

        return groups

    def run(
        self,
        student_answer: str,
        marking_guide: list,
        topic: str,
    ) -> dict:

        answer_text = self._normalise(student_answer)

        topic_key = self._normalise(topic)

        topic_kws = (
            self._DOMAIN_KEYWORDS.get(topic_key, [])
            + self._DOMAIN_KEYWORDS["general"]
        )

        criterion_flags = []
        all_keywords_found = []

        for criterion_item in marking_guide:
            criterion_text = criterion_item.get("criterion", "")
            max_marks = criterion_item.get("marks", 0)

            concept_groups = self._criterion_groups(criterion_text)

            matched_keywords = []
            matched_group_count = 0

            for group in concept_groups:
                group_matches = self._matched_aliases(answer_text, group)

                if group_matches:
                    matched_group_count += 1
                    matched_keywords.extend(group_matches)

            total_groups = len(concept_groups)
            criterion_coverage_ratio = (
                matched_group_count / total_groups if total_groups else 0
            )

            # Surface coverage should mean the criterion is meaningfully touched.
            # For criteria with many concepts, require at least 2 matched groups.
            if total_groups >= 4:
                surface_covered = matched_group_count >= 2
            else:
                surface_covered = matched_group_count >= 1

            if criterion_coverage_ratio >= 0.75:
                coverage_level = "strong"
            elif criterion_coverage_ratio >= 0.4:
                coverage_level = "partial"
            elif criterion_coverage_ratio > 0:
                coverage_level = "weak"
            else:
                coverage_level = "none"

            criterion_flags.append({
                "criterion": criterion_text,
                "max_marks": max_marks,
                "surface_covered": surface_covered,
                "coverage_level": coverage_level,
                "coverage_ratio": criterion_coverage_ratio,
                "matched_keywords": matched_keywords,
                "matched_concepts": matched_group_count,
                "total_concepts": total_groups,
            })

            all_keywords_found.extend(matched_keywords)

        covered_count = sum(1 for c in criterion_flags if c["surface_covered"])
        coverage_ratio = covered_count / len(criterion_flags) if criterion_flags else 0

        topic_hits = self._matched_aliases(answer_text, topic_kws)

        # Combine topic hits + criterion hits for UI display
        combined_hits = topic_hits + all_keywords_found

        # Remove duplicates while preserving order
        seen = set()
        unique_hits = []

        for keyword in combined_hits:
            key = self._normalise(keyword)
            if key and key not in seen:
                unique_hits.append(keyword)
                seen.add(key)

        return {
            "criterion_flags": criterion_flags,
            "coverage_ratio": coverage_ratio,
            "topic_keywords_found": unique_hits,
        }