"""
=====================================================================
PHASE 4 - CONCEPT 3: FAISS Vector Store Operations
=====================================================================

FAISS (Facebook AI Similarity Search)
---------------------------------------
- Developed by Meta AI Research
- In-memory or on-disk vector index
- Supports billions of vectors with millisecond search
- No server required — pure Python library

FAISS IN LANGCHAIN
-------------------
LangChain wraps FAISS with a simple API:
  - FAISS.from_documents()    → create from docs
  - FAISS.from_texts()        → create from plain strings
  - .similarity_search()      → top-k by semantic similarity
  - .save_local() / .load_local() → persist to disk
  - .add_documents()          → add more docs later
  - .as_retriever()           → plug into a chain
"""

import os
import shutil
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)
print("Ready.\n")

# ──────────────────────────────────────────────────────────────────
# STEP 1: Create a FAISS store from documents with metadata
# ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Create FAISS store with metadata")
print("=" * 60)

documents = [
    Document(
        page_content="Python is a high-level, interpreted programming language known for its readability.",
        metadata={"source": "python_docs", "topic": "programming", "page": 1},
    ),
    Document(
        page_content="LangChain provides tools for chaining LLM calls and building agents.",
        metadata={"source": "langchain_docs", "topic": "AI", "page": 1},
    ),
    Document(
        page_content="FAISS allows efficient similarity search over large collections of vectors.",
        metadata={"source": "faiss_docs", "topic": "vector_db", "page": 1},
    ),
    Document(
        page_content="Transformers are neural network architectures that use self-attention mechanisms.",
        metadata={"source": "ml_textbook", "topic": "deep_learning", "page": 42},
    ),
    Document(
        page_content="RAG combines a retrieval system with a language model for knowledge-grounded answers.",
        metadata={"source": "rag_paper", "topic": "AI", "page": 3},
    ),
    Document(
        page_content="Embeddings map text to dense vectors in a continuous semantic space.",
        metadata={"source": "nlp_textbook", "topic": "NLP", "page": 15},
    ),
    Document(
        page_content="Streamlit lets you build interactive web apps with pure Python.",
        metadata={"source": "streamlit_docs", "topic": "web", "page": 1},
    ),
]

db = FAISS.from_documents(documents, embeddings)
print(f"Created FAISS store with {len(documents)} documents.\n")

# ──────────────────────────────────────────────────────────────────
# STEP 2: Basic similarity search
# ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2: Basic similarity search")
print("=" * 60)

query = "How do neural networks process language?"
results = db.similarity_search(query, k=3)

print(f"Query: '{query}'\n")
for i, doc in enumerate(results, 1):
    print(f"Result #{i}")
    print(f"  Content : {doc.page_content}")
    print(f"  Metadata: {doc.metadata}\n")

# ──────────────────────────────────────────────────────────────────
# STEP 3: Similarity search WITH scores
# ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Search with similarity scores")
print("=" * 60)

query2 = "What tools help build LLM applications?"
results_with_scores = db.similarity_search_with_score(query2, k=4)

print(f"Query: '{query2}'\n")
for doc, score in results_with_scores:
    print(f"  L2 dist: {score:.4f}  |  {doc.page_content[:60]}...")

# ──────────────────────────────────────────────────────────────────
# STEP 4: Adding new documents to an existing store
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Add new documents to existing store")
print("=" * 60)

new_docs = [
    Document(
        page_content="Groq is a hardware company that offers ultra-fast LLM inference via API.",
        metadata={"source": "groq_docs", "topic": "AI", "page": 1},
    ),
    Document(
        page_content="Vector quantisation reduces memory by compressing high-dimensional vectors.",
        metadata={"source": "faiss_docs", "topic": "vector_db", "page": 8},
    ),
]

db.add_documents(new_docs)
print(f"Added {len(new_docs)} new documents. Total now: {len(documents) + len(new_docs)}\n")

# ──────────────────────────────────────────────────────────────────
# STEP 5: Save and reload the index from disk
# ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 5: Persist to disk and reload")
print("=" * 60)

save_path = "/tmp/faiss_demo_index"
db.save_local(save_path)
print(f"Saved FAISS index to '{save_path}/'")

# Reload
db_loaded = FAISS.load_local(
    save_path,
    embeddings,
    allow_dangerous_deserialization=True,   # required by LangChain
)
print("Reloaded FAISS index from disk.")

# Verify it works
q = "Tell me about Groq inference"
r = db_loaded.similarity_search(q, k=1)[0]
print(f"\nPost-reload search for '{q}':")
print(f"  → {r.page_content}\n")

# Clean up temp files
shutil.rmtree(save_path, ignore_errors=True)

# ──────────────────────────────────────────────────────────────────
# STEP 6: Use as a LangChain Retriever
# ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 6: FAISS as a LangChain Retriever")
print("=" * 60)

retriever = db.as_retriever(
    search_type="similarity",   # or "mmr" for diversity
    search_kwargs={"k": 3},
)

retrieved = retriever.invoke("How does semantic search work?")
print("Retrieved docs for 'How does semantic search work?':\n")
for doc in retrieved:
    print(f"  - {doc.page_content[:70]}...")

print("""
SUMMARY
-------
FAISS operations:
  from_documents()  → build index
  add_documents()   → extend index
  similarity_search()        → top-k docs
  similarity_search_with_score() → docs + distances
  save_local() / load_local() → persistence
  as_retriever()    → plug into RAG chains
""")
