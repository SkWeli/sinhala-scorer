"""
RAG Pipeline — fully offline.
Uses ChromaDB + nomic-embed-text (via Ollama) for embedding and retrieval.
Indexes portuguese.txt, dutch.txt, british.txt from knowledge_base/.
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
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


def build_vector_store(force_rebuild: bool = False) -> Chroma:
    """
    Load knowledge-base text files, chunk them, embed with nomic-embed-text,
    and persist the resulting ChromaDB to disk.
    """
    if os.path.exists(VECTOR_STORE_DIR) and not force_rebuild:
        print("[RAG] Vector store already exists — loading existing index.")
        return load_vector_store()

    docs = []
    for fname in ["portuguese.txt", "dutch.txt", "british.txt"]:
        fpath = os.path.join(KNOWLEDGE_BASE_DIR, fname)
        if not os.path.exists(fpath):
            print(f"[RAG] WARNING: {fpath} not found — skipping.")
            continue
        loader = TextLoader(fpath, encoding="utf-8")
        loaded = loader.load()
        source_label = fname.replace(".txt", "").capitalize()
        for doc in loaded:
            doc.metadata["source"] = source_label
        docs.extend(loaded)

    if not docs:
        raise FileNotFoundError(
            "No knowledge base files found in knowledge_base/. "
            "Please add portuguese.txt, dutch.txt, british.txt."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"[RAG] Indexing {len(chunks)} chunks from {len(docs)} document(s)…")

    embeddings = _get_embeddings()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR,
    )
    print(f"[RAG] Vector store persisted → {VECTOR_STORE_DIR}")
    return vectordb


def load_vector_store() -> Chroma:
    """Load an already-built ChromaDB from disk."""
    return Chroma(
        persist_directory=VECTOR_STORE_DIR,
        embedding_function=_get_embeddings(),
    )


def retrieve_context(query: str, topic: str = "") -> list[dict]:
    """
    Retrieve the top-K most relevant passages using MMR search.
    Returns a list of {content, source} dicts.
    """
    vectordb = load_vector_store()
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K_RETRIEVAL, "fetch_k": 15},
    )
    full_query = f"{topic} {query}".strip()
    docs = retriever.invoke(full_query)
    return [
        {
            "content": d.page_content,
            "source":  d.metadata.get("source", "Unknown"),
        }
        for d in docs
    ]