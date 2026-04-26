"""
RAG Pipeline — fully offline, no torch/sentence-transformers dependency.
Uses raw file reading + langchain-text-splitters + chromadb + ollama embeddings.
"""

import os
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from config.settings import (
    KNOWLEDGE_BASE_DIR,
    VECTOR_STORE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    TOP_K_RETRIEVAL,
)


def _get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def _get_chroma_client() -> chromadb.PersistentClient:
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    return chromadb.PersistentClient(
        path=VECTOR_STORE_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _load_txt(fpath: str, source: str) -> list[Document]:
    """Read a plain text file and return as a LangChain Document."""
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": source})]


def build_vector_store(force_rebuild: bool = False) -> None:
    """
    Load .txt knowledge base files, chunk, embed via Ollama,
    and persist into ChromaDB — no torch, no ONNX, no sentence-transformers.
    """
    client = _get_chroma_client()

    existing = [c.name for c in client.list_collections()]

    if "colonial_kb" in existing and not force_rebuild:
        print("[RAG] Collection 'colonial_kb' already exists — skipping rebuild.")
        return

    if "colonial_kb" in existing:
        client.delete_collection("colonial_kb")

    # Load documents manually (no langchain-community loader needed)
    docs = []
    for fname, source in [
        ("portuguese.txt", "Portuguese"),
        ("dutch.txt",      "Dutch"),
        ("british.txt",    "British"),
    ]:
        fpath = os.path.join(KNOWLEDGE_BASE_DIR, fname)
        if not os.path.exists(fpath):
            print(f"[RAG] WARNING: {fpath} not found — skipping.")
            continue
        docs.extend(_load_txt(fpath, source))
        print(f"[RAG] Loaded: {fname}")

    if not docs:
        raise FileNotFoundError(
            "No knowledge base files found in knowledge_base/. "
            "Please add portuguese.txt, dutch.txt, british.txt."
        )

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"[RAG] Split into {len(chunks)} chunks from {len(docs)} file(s).")

    # Create collection (no default EF — we supply embeddings manually)
    embeddings = _get_embeddings()
    collection = client.create_collection(
        name="colonial_kb",
        metadata={"hnsw:space": "cosine"},
    )

    # Embed and insert in batches
    BATCH = 40
    for i in range(0, len(chunks), BATCH):
        batch     = chunks[i : i + BATCH]
        texts     = [c.page_content for c in batch]
        metadatas = [c.metadata     for c in batch]
        ids       = [f"chunk_{i + j}" for j in range(len(batch))]
        vectors   = embeddings.embed_documents(texts)

        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"[RAG]   Stored chunks {i + 1}–{i + len(batch)}")

    print(f"[RAG] ✅ Vector store ready → {VECTOR_STORE_DIR}")


def retrieve_context(query: str, topic: str = "") -> list[dict]:
    """Retrieve top-K relevant passages using cosine similarity."""
    client     = _get_chroma_client()
    embeddings = _get_embeddings()

    collection = client.get_collection("colonial_kb")
    full_query = f"{topic} {query}".strip()
    query_vec  = embeddings.embed_query(full_query)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=TOP_K_RETRIEVAL,
        include=["documents", "metadatas"],
    )

    return [
        {
            "content": doc,
            "source":  meta.get("source", "Unknown"),
        }
        for doc, meta in zip(
            results["documents"][0],
            results["metadatas"][0],
        )
    ]