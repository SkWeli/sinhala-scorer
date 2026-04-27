"""
Agent 1 — Retrieval Agent
Fetches relevant RAG passages AND matching ontology concepts.

Uses:
  - JSON-based local RAG retrieval
  - Ontology concept matching
  - Fully offline pipeline
"""

from rag.rag_pipeline import retrieve_context
from ontology.ontology_builder import query_ontology_concepts


class RetrievalAgent:
    def run(
        self,
        question_text: str,
        topic: str = "",
        student_answer: str = "",
    ) -> dict:
        """
        Run retrieval for the given question/topic.

        Args:
            question_text: The exam/question text.
            topic: Era/topic such as Portuguese, Dutch, British.
            student_answer: Student's Sinhala answer.
                            Accepted for orchestrator compatibility,
                            but not directly used as the main retrieval query.

        Returns:
            {
                "rag_passages": [...],
                "ontology_concepts": [...]
            }
        """

        # ── Step 1: RAG Retrieval ─────────────────────────────────────────────
        rag_results = []

        try:
            print("[Agent1] Starting RAG retrieval…", flush=True)

            # Important:
            # Use the QUESTION as the retrieval query, not the student answer.
            # Student answers may be incomplete or wrong, so the question/topic
            # is safer for retrieving correct reference evidence.
            rag_results = retrieve_context(
                query=question_text,
                topic=topic,
            )

            print(
                f"[Agent1] RAG retrieved {len(rag_results)} passage(s).",
                flush=True,
            )

        except Exception as e:
            print(f"[Agent1] RAG retrieval failed: {type(e).__name__}: {e}", flush=True)
            rag_results = []

        # ── Step 2: Ontology Query ────────────────────────────────────────────
        ontology_concepts = []

        try:
            print("[Agent1] Querying ontology…", flush=True)

            topic_lower = topic.lower().strip()
            question_lower = question_text.lower().strip()

            keywords = []

            # Basic topic/question keywords
            keywords.extend(topic_lower.split())
            keywords.extend(question_lower.split())

            # English colonial keywords
            keywords.extend([
                "colonial",
                "sri lanka",
                "portuguese",
                "dutch",
                "british",
                "ceylon",
            ])

            # Sinhala colonial keywords
            keywords.extend([
                "ශ්‍රී ලංකාව",
                "පෘතුගීසි",
                "ලන්දේසි",
                "බ්‍රිතාන්‍ය",
                "යටත් විජිත",
                "කෝට්ටේ",
                "කන්දිය",
                "උඩරට",
                "කුරුඳු",
                "වෙළඳාම",
                "පාලනය",
            ])

            # Topic-specific boosts
            if "portuguese" in topic_lower or "පෘතුගීසි" in topic_lower:
                keywords.extend([
                    "Portuguese",
                    "Lourenço de Almeida",
                    "Dharmapala",
                    "Malwana",
                    "Kotte",
                    "පෘතුගීසි",
                    "ලෝරෙන්සෝ ද අල්මේදා",
                    "ධර්මපාල",
                    "මල්වාන",
                    "කෝට්ටේ",
                ])

            if "dutch" in topic_lower or "ලන්දේසි" in topic_lower:
                keywords.extend([
                    "Dutch",
                    "VOC",
                    "Thombu",
                    "Cinnamon",
                    "Mahabadde",
                    "ලන්දේසි",
                    "තෝම්බු",
                    "කුරුඳු",
                    "මහබද්ද",
                ])

            if "british" in topic_lower or "බ්‍රිතාන්‍ය" in topic_lower:
                keywords.extend([
                    "British",
                    "Colebrooke",
                    "Cameron",
                    "Plantation",
                    "Kandyan Convention",
                    "බ්‍රිතාන්‍ය",
                    "කෝල්බෲක්",
                    "කැමරන්",
                    "වැවිලි",
                    "උඩරට ගිවිසුම",
                ])

            # Remove duplicates while preserving order
            seen = set()
            cleaned_keywords = []

            for kw in keywords:
                kw = kw.strip()
                if kw and kw not in seen:
                    cleaned_keywords.append(kw)
                    seen.add(kw)

            ontology_concepts = query_ontology_concepts(cleaned_keywords)

            print(
                f"[Agent1] Ontology returned {len(ontology_concepts)} concept(s).",
                flush=True,
            )

        except Exception as e:
            print(f"[Agent1] Ontology query failed: {type(e).__name__}: {e}", flush=True)
            ontology_concepts = []

        return {
            "rag_passages": rag_results,
            "ontology_concepts": ontology_concepts,
        }