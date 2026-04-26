"""
Agent 1 — Retrieval Agent
Fetches relevant RAG passages AND matching ontology concepts.
"""

from rag.rag_pipeline import retrieve_context
from ontology.ontology_builder import query_ontology_concepts


class RetrievalAgent:

    def run(self, question_text: str, topic: str) -> dict:

        # ── RAG Retrieval ─────────────────────────────────────────────────────
        rag_results = []
        try:
            print("[Agent1] Starting RAG retrieval…", flush=True)
            rag_results = retrieve_context(question_text, topic)
            print(f"[Agent1] RAG retrieved {len(rag_results)} passages.", flush=True)
        except Exception as e:
            print(f"[Agent1] RAG retrieval failed: {e}", flush=True)
            rag_results = []

        # ── Ontology Query ────────────────────────────────────────────────────
        ontology_concepts = []
        try:
            print("[Agent1] Querying ontology…", flush=True)
            keywords = topic.lower().split() + [
                "colonial", "sri lanka", "portuguese", "dutch", "british"
            ]
            ontology_concepts = query_ontology_concepts(keywords)
            print(f"[Agent1] Ontology returned {len(ontology_concepts)} concepts.", flush=True)
        except Exception as e:
            print(f"[Agent1] Ontology query failed: {e}", flush=True)
            ontology_concepts = []

        return {
            "rag_passages":      rag_results,
            "ontology_concepts": ontology_concepts,
        }