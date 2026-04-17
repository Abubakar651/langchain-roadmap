"""
=====================================================================
PHASE 4 - CONCEPT 2: Embeddings Deep Dive
=====================================================================

WHAT ARE EMBEDDINGS?
---------------------
Embeddings are dense numerical representations of text (or images,
audio, etc.) in a continuous vector space.

Key property: semantically similar text → similar vectors → small distance

EMBEDDING MODELS
-----------------
┌──────────────────────────────┬─────────┬─────────────────────────┐
│ Model                        │ Dims    │ Notes                   │
├──────────────────────────────┼─────────┼─────────────────────────┤
│ all-MiniLM-L6-v2             │  384    │ Fast, great quality     │
│ all-mpnet-base-v2            │  768    │ Better quality, slower  │
│ text-embedding-3-small       │ 1536    │ OpenAI, paid            │
│ text-embedding-3-large       │ 3072    │ OpenAI, best quality    │
└──────────────────────────────┴─────────┴─────────────────────────┘

We use all-MiniLM-L6-v2 — free, fast, and good enough for most tasks.
"""

import math
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)
print("Ready.\n")


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two vectors (range: -1 to 1)."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2)


# ──────────────────────────────────────────────────────────────────
# DEMO 1: Visualise similarity between sentence pairs
# ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("DEMO 1: Cosine similarity between sentence pairs")
print("=" * 60)

sentence_pairs = [
    ("I love programming in Python",   "Python is my favourite coding language"),   # very similar
    ("The cat sat on the mat",          "A dog is sleeping on the rug"),             # somewhat similar
    ("Machine learning is fascinating","The stock market crashed today"),            # unrelated
    ("How do I install pip?",          "pip install command tutorial"),              # similar topic
    ("Beautiful sunset over the ocean", "Quantum entanglement experiments"),         # unrelated
]

for s1, s2 in sentence_pairs:
    v1 = embeddings.embed_query(s1)
    v2 = embeddings.embed_query(s2)
    sim = cosine_similarity(v1, v2)
    bar = "█" * int(sim * 30)
    print(f"\nS1: {s1}")
    print(f"S2: {s2}")
    print(f"   Similarity: {sim:.4f}  {bar}")

# ──────────────────────────────────────────────────────────────────
# DEMO 2: Embedding multiple documents at once
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DEMO 2: Batch embedding documents")
print("=" * 60)

docs = [
    "LangChain is a framework for building LLM applications.",
    "FAISS enables fast vector similarity search.",
    "RAG combines retrieval with text generation.",
]

vectors = embeddings.embed_documents(docs)

for doc, vec in zip(docs, vectors):
    print(f"\nDoc   : {doc}")
    print(f"Shape : {len(vec)} dimensions")
    print(f"Sample: {[round(x, 4) for x in vec[:5]]}...")

# ──────────────────────────────────────────────────────────────────
# DEMO 3: Odd-one-out — embeddings understand context
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DEMO 3: Find the odd-one-out")
print("=" * 60)

candidates = [
    "Python is a programming language",
    "JavaScript runs in the browser",
    "Rust is known for memory safety",
    "Paris is the capital of France",   # <-- odd one out
    "Go is designed for concurrency",
]

query = "What programming language should I learn?"
q_vec = embeddings.embed_query(query)

scores = []
for text in candidates:
    c_vec = embeddings.embed_query(text)
    scores.append((cosine_similarity(q_vec, c_vec), text))

scores.sort(reverse=True)

print(f"\nQuery: '{query}'\n")
for rank, (score, text) in enumerate(scores, 1):
    marker = " ← odd one out" if "Paris" in text else ""
    print(f"  #{rank}  {score:.4f}  {text}{marker}")

print("""
TAKEAWAY: The Paris sentence scores lowest because it's semantically
unrelated to programming — even without explicit keyword matching.
""")
