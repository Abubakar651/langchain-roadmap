"""
Vector Store Manager
--------------------
Creates, updates, persists, and queries a FAISS vector store.
Uses HuggingFace sentence-transformers for embeddings (no API key needed).
"""

import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

# Path where the FAISS index is saved between sessions
INDEX_DIR = Path(__file__).parent / "faiss_index"

# Embedding model — downloaded once, cached in ~/.cache/huggingface/
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFaceEmbeddings instance."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # cosine-friendly
    )


def build_vector_store(documents: list[Document]) -> FAISS:
    """
    Create a new FAISS index from a list of Documents and save it.

    Args:
        documents: Chunked Document objects (from pdf_processor)
    Returns:
        FAISS vector store
    """
    embeddings = get_embeddings()
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(str(INDEX_DIR))
    print(f"FAISS index saved to '{INDEX_DIR}/' ({len(documents)} vectors)")
    return db


def load_vector_store() -> FAISS | None:
    """
    Load a previously saved FAISS index from disk.

    Returns None if no index exists yet.
    """
    index_file = INDEX_DIR / "index.faiss"
    if not index_file.exists():
        return None

    embeddings = get_embeddings()
    db = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return db


def add_documents(db: FAISS, documents: list[Document]) -> FAISS:
    """Add new documents to an existing FAISS store and save."""
    db.add_documents(documents)
    db.save_local(str(INDEX_DIR))
    return db


def get_retriever(db: FAISS, k: int = 4) -> VectorStoreRetriever:
    """
    Return a retriever that fetches the top-k most relevant chunks.

    Uses MMR (Maximal Marginal Relevance) to balance relevance AND diversity,
    so you don't get k near-identical chunks.
    """
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3},
    )


def clear_index() -> None:
    """Delete the persisted FAISS index."""
    import shutil
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
        print(f"Deleted FAISS index at '{INDEX_DIR}/'")


def index_exists() -> bool:
    """Check if a FAISS index exists on disk."""
    return (INDEX_DIR / "index.faiss").exists()
