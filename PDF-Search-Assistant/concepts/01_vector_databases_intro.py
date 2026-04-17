"""
=====================================================================
PHASE 4 - CONCEPT 1: Introduction to Vector Databases
=====================================================================

WHAT IS A VECTOR DATABASE?
---------------------------
A vector database stores data as high-dimensional numerical vectors
(embeddings) and allows you to search by *semantic similarity*
instead of exact keyword matches.

Normal DB:  "cat" == "cat"   → True / False (exact match)
Vector DB:  "cat" ≈ "feline" → 0.92 similarity score  (semantic match)

HOW DOES IT WORK?
-----------------
1. Text  →  Embedding Model  →  Vector (e.g. [0.12, -0.87, 0.44, ...])
2. Vectors are stored in the DB with an index for fast retrieval
3. At query time: query → vector → find nearest neighbours (ANN search)

WHY USE IT WITH LLMs?
---------------------
LLMs have a limited context window (can't read 1000 pages at once).
Solution: store the 1000 pages as vectors, retrieve only the relevant
chunks, then feed those chunks to the LLM → RAG pattern.

POPULAR VECTOR DATABASES
-------------------------
┌─────────────┬──────────────┬──────────────────────────────────────┐
│ Database    │ Type         │ Best For                             │
├─────────────┼──────────────┼──────────────────────────────────────┤
│ FAISS       │ Library      │ Local, fast, no server needed        │
│ ChromaDB    │ Local/Cloud  │ Easy to use, built for LangChain     │
│ Pinecone    │ Cloud SaaS   │ Production, fully managed            │
│ Weaviate    │ Open-source  │ Rich schema + vector search          │
│ Qdrant      │ Open-source  │ High performance, Rust-based         │
└─────────────┴──────────────┴──────────────────────────────────────┘

We use FAISS in this course: it's free, fast, and runs 100% locally.
"""

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

# ──────────────────────────────────────────────────────────────────
# STEP 1: Create an embedding model
# ──────────────────────────────────────────────────────────────────
# "all-MiniLM-L6-v2" is a lightweight, fast sentence-transformer model.
# It converts any text into a 384-dimensional vector.
print("Loading embedding model (downloads ~90MB on first run)...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)
print("Embedding model loaded.\n")

# ──────────────────────────────────────────────────────────────────
# STEP 2: See what an embedding looks like
# ──────────────────────────────────────────────────────────────────
sample_text = "Artificial intelligence is transforming the world."
vector = embeddings.embed_query(sample_text)

print(f"Text   : '{sample_text}'")
print(f"Vector : {vector[:6]}...  (showing first 6 of {len(vector)} dimensions)\n")

# ──────────────────────────────────────────────────────────────────
# STEP 3: Build a tiny in-memory FAISS vector store
# ──────────────────────────────────────────────────────────────────
documents = [
    Document(page_content="Python is a popular programming language for data science."),
    Document(page_content="LangChain helps developers build LLM-powered applications."),
    Document(page_content="FAISS is a library for efficient similarity search."),
    Document(page_content="Machine learning models learn patterns from data."),
    Document(page_content="Vector databases store embeddings for semantic search."),
    Document(page_content="The Eiffel Tower is located in Paris, France."),
    Document(page_content="Neural networks are inspired by the human brain."),
]

print("Building FAISS vector store from documents...")
vector_store = FAISS.from_documents(documents, embeddings)
print(f"Vector store created with {len(documents)} documents.\n")

# ──────────────────────────────────────────────────────────────────
# STEP 4: Semantic similarity search
# ──────────────────────────────────────────────────────────────────
query = "How do I search for similar text?"
print(f"Query: '{query}'")
print("-" * 50)

results = vector_store.similarity_search_with_score(query, k=3)

for i, (doc, score) in enumerate(results, 1):
    # FAISS uses L2 distance: lower score = more similar
    print(f"#{i}  Score (L2 distance): {score:.4f}")
    print(f"    Text: {doc.page_content}\n")

# ──────────────────────────────────────────────────────────────────
# KEY OBSERVATIONS
# ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. Embeddings convert text → numbers that capture *meaning*.
2. FAISS indexes those vectors for lightning-fast nearest-neighbor search.
3. 'similar text' query correctly finds FAISS/vector/search docs —
   even though the exact words don't appear in the query!
4. Lower L2 distance = more semantically similar.
""")
