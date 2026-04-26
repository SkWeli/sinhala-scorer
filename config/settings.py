"""Central configuration for the Sinhala Scorer system."""
import os

# OLLAMA
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "Tharusha_Dilhara_Jayadeera/singemma"            # change to any locally pulled model
OLLAMA_EMBED_MODEL = "nomic-embed-text"  # local embedding model

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")
VECTOR_STORE_DIR   = os.path.join(BASE_DIR, "vector_store", "chroma_db")
ONTOLOGY_PATH      = os.path.join(BASE_DIR, "ontology", "colonial_history.owl")

# RAG
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 100
TOP_K_RETRIEVAL = 5

# Scoring
MAX_MARKS = 20