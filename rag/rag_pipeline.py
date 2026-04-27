"""
RAG Pipeline — fully offline, no ChromaDB, no SQLite, no torch dependency.

Uses:
  - raw UTF-8 Sinhala text files
  - langchain-text-splitters
  - Ollama embeddings via /api/embed
  - local JSON vector store
  - cosine similarity retrieval
"""

import os
import json
import math
import httpx

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import (
    KNOWLEDGE_BASE_DIR,
    VECTOR_STORE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    TOP_K_RETRIEVAL,
)


VECTOR_STORE_FILE = os.path.join(VECTOR_STORE_DIR, "colonial_kb.json")


def _load_txt(fpath: str, source: str) -> list[Document]:
    """
    Read a UTF-8 Sinhala text file and return it as a LangChain Document.
    """
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    return [
        Document(
            page_content=content,
            metadata={"source": source},
        )
    ]


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed one or more texts using Ollama's /api/embed endpoint.
    """
    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={
                "model": OLLAMA_EMBED_MODEL,
                "input": texts,
                "keep_alive": "10m",
            },
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

        response.raise_for_status()
        data = response.json()

        if "embeddings" not in data:
            print("[RAG] Unexpected Ollama response:", data, flush=True)
            raise KeyError("Missing 'embeddings' key in Ollama response.")

        embeddings = data["embeddings"]

        if len(embeddings) != len(texts):
            raise ValueError(
                f"Embedding count mismatch: got {len(embeddings)}, expected {len(texts)}"
            )

        return embeddings

    except Exception as e:
        print("[RAG] Embedding failed.", flush=True)
        print(type(e).__name__, e, flush=True)
        raise


def _embed_query(text: str) -> list[float]:
    """
    Embed a single query.
    """
    return _embed_texts([text])[0]


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def build_vector_store(force_rebuild: bool = False) -> None:
    """
    Build a local JSON vector store from Sinhala knowledge base files.
    """
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    if os.path.exists(VECTOR_STORE_FILE) and not force_rebuild:
        print("[RAG] JSON vector store already exists — skipping rebuild.", flush=True)
        return

    docs: list[Document] = []

    files_to_load = [
        ("portuguese.txt", "Portuguese"),
        ("dutch.txt", "Dutch"),
        ("british.txt", "British"),
    ]

    for fname, source in files_to_load:
        fpath = os.path.join(KNOWLEDGE_BASE_DIR, fname)

        if not os.path.exists(fpath):
            print(f"[RAG] WARNING: {fpath} not found — skipping.", flush=True)
            continue

        loaded_docs = _load_txt(fpath, source)
        docs.extend(loaded_docs)

        print(
            f"[RAG] Loaded: {fname} ({len(loaded_docs[0].page_content)} chars)",
            flush=True,
        )

    if not docs:
        raise FileNotFoundError(
            "No knowledge base files found. Please add portuguese.txt, dutch.txt, british.txt."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(docs)

    print(
        f"[RAG] Split into {len(chunks)} chunks from {len(docs)} file(s).",
        flush=True,
    )

    store = []

    BATCH_SIZE = 5

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [chunk.page_content for chunk in batch]

        print(
            f"[RAG] Embedding chunks {i + 1}–{i + len(batch)}...",
            flush=True,
        )

        vectors = _embed_texts(texts)

        for j, chunk in enumerate(batch):
            store.append(
                {
                    "id": f"chunk_{i + j}",
                    "content": chunk.page_content,
                    "source": chunk.metadata.get("source", "Unknown"),
                    "embedding": vectors[j],
                }
            )

        print(
            f"[RAG] Stored chunks {i + 1}–{i + len(batch)} in memory.",
            flush=True,
        )

    with open(VECTOR_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)

    print(f"[RAG] ✅ JSON vector store ready → {VECTOR_STORE_FILE}", flush=True)
    print(f"[RAG] Total chunks stored: {len(store)}", flush=True)


def retrieve_context(query: str, topic: str = "") -> list[dict]:
    """
    Retrieve top-k relevant chunks using hybrid scoring:
      - embedding cosine similarity
      - keyword overlap boost
      - topic/source boost
    """
    if not os.path.exists(VECTOR_STORE_FILE):
        raise FileNotFoundError(
            "Vector store not found. Run `python rebuild_rag.py` first."
        )

    full_query = f"{topic} {query}".strip()

    if not full_query:
        return []

    query_vector = _embed_query(full_query)

    with open(VECTOR_STORE_FILE, "r", encoding="utf-8") as f:
        store = json.load(f)

    # Simple Sinhala/English keyword extraction
    query_terms = [
        term.strip().lower()
        for term in full_query.replace("?", " ").replace(",", " ").split()
        if len(term.strip()) >= 3
    ]

    scored_results = []

    for item in store:
        content = item["content"]
        content_lower = content.lower()
        source_lower = item["source"].lower()

        embedding_score = _cosine_similarity(query_vector, item["embedding"])

        # Keyword overlap score
        matched_terms = [
            term for term in query_terms
            if term in content_lower
        ]

        keyword_score = len(matched_terms) / max(len(query_terms), 1)

        # Topic/source boost
        source_boost = 0.0

        if "පෘතුගීසි" in full_query or "portuguese" in full_query.lower():
            if source_lower == "portuguese":
                source_boost += 0.08

        if "ලන්දේසි" in full_query or "dutch" in full_query.lower():
            if source_lower == "dutch":
                source_boost += 0.08

        if "බ්‍රිතාන්‍ය" in full_query or "british" in full_query.lower():
            if source_lower == "british":
                source_boost += 0.08

        # Final hybrid score
        final_score = (
            0.70 * embedding_score
            + 0.25 * keyword_score
            + source_boost
        )

        scored_results.append(
            {
                "content": item["content"],
                "source": item["source"],
                "score": final_score,
                "embedding_score": embedding_score,
                "keyword_score": keyword_score,
                "matched_terms": matched_terms,
            }
        )

    scored_results.sort(key=lambda x: x["score"], reverse=True)

    return scored_results[:TOP_K_RETRIEVAL]