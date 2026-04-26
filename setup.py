"""
build the ontology + index the knowledge base into ChromaDB.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ontology.ontology_builder import build_ontology
from rag.rag_pipeline import build_vector_store

if __name__ == "__main__":
    print("=" * 50)
    print("Step 1/2 — Building Colonial Sri Lanka Ontology…")
    build_ontology()

    print("\nStep 2/2 — Indexing knowledge base into ChromaDB…")
    build_vector_store(force_rebuild=True)

    print("\n✅ Setup complete. Run: streamlit run app.py")