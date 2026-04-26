"""
Agent 1 — Retrieval Agent
Fetches relevant RAG passages AND matching ontology concepts
for a given question + topic.
"""

from rag.rag_pipeline import retrieve_context
from ontology.ontology_builder import query_ontology_concepts


class RetrievalAgent:
    """
    Responsibilities:
      - Query ChromaDB with MMR retrieval for relevant passages
      - Query the OWL ontology for matching entities/classes
      - Return combined context for downstream agents
    """

    def run(self, question_text: str, topic: str) -> dict:
        # RAG: retrieve top-K passages from the local vector store
        rag_results = retrieve_context(question_text, topic)

        # Ontology: build keyword list and fetch matching concepts
        keywords = topic.lower().split() + ["colonial", "sri lanka",
                                             "portuguese", "dutch", "british"]
        ontology_concepts = query_ontology_concepts(keywords)

        return {
            "rag_passages":      rag_results,
            "ontology_concepts": ontology_concepts,
        }