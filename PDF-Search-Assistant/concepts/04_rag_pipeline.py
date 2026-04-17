"""
=====================================================================
PHASE 4 - CONCEPT 4: RAG (Retrieval-Augmented Generation) Pipeline
=====================================================================

WHAT IS RAG?
------------
RAG = Retrieval-Augmented Generation

The LLM doesn't "know" everything. RAG lets it answer questions using
your own documents by:
  1. INDEXING  : chunk documents → embed → store in vector DB
  2. RETRIEVAL : embed query → find top-k similar chunks
  3. GENERATION: feed chunks as context to LLM → grounded answer

WITHOUT RAG:  LLM relies only on training data (may hallucinate)
WITH RAG:     LLM grounds answers in YOUR documents (factual, current)

RAG PIPELINE DIAGRAM
---------------------

Documents → [Chunker] → [Embedder] → [FAISS Index]
                                           │
User Query → [Embedder] → similarity search┘
                                ↓
              Top-k relevant chunks (context)
                                ↓
             [Prompt = context + question] → [LLM] → Answer

TEXT SPLITTING STRATEGIES
--------------------------
Splitting is critical — wrong chunk size degrades quality.

  RecursiveCharacterTextSplitter  ← recommended default
    - Splits by paragraphs → sentences → words
    - chunk_size    : max chars per chunk (500–1500 typical)
    - chunk_overlap : chars shared between adjacent chunks (10-20%)
    - Overlap prevents context from being cut off at boundaries
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ──────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE — our "documents" (simulates a PDF / website)
# ──────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = """
LangChain is an open-source framework created by Harrison Chase in 2022.
It helps developers build applications powered by large language models (LLMs).
Key components include: chains, agents, memory, document loaders, and vector stores.
LangChain supports models from OpenAI, Anthropic, Groq, Hugging Face, and more.

Groq is an AI infrastructure company that provides ultra-fast LLM inference.
The Groq Language Processing Unit (LPU) achieves speeds over 750 tokens per second.
Popular models available on Groq include LLaMA 3.3 70B, Mixtral 8x7B, and Gemma.
The Groq API is compatible with the OpenAI API format.

FAISS (Facebook AI Similarity Search) is an open-source library developed by Meta.
It is written in C++ with Python bindings for fast vector similarity search.
FAISS supports flat (exact) and approximate nearest-neighbor indices.
It can search billions of vectors in milliseconds on a single machine.

RAG (Retrieval-Augmented Generation) was introduced in a 2020 paper by Facebook AI.
It combines dense passage retrieval (DPR) with seq2seq generation models.
RAG helps LLMs produce factually grounded answers by retrieving relevant context.
Modern RAG systems use vector databases for the retrieval component.

Embeddings are numerical representations of text in a high-dimensional space.
The sentence-transformers library provides free, high-quality embedding models.
The 'all-MiniLM-L6-v2' model produces 384-dimensional embeddings.
Cosine similarity is commonly used to measure the distance between embeddings.

Vector databases are optimised for storing and querying embedding vectors.
Popular vector databases include FAISS, Pinecone, ChromaDB, Weaviate, and Qdrant.
They use Approximate Nearest Neighbour (ANN) algorithms like HNSW and IVF.
Vector databases enable semantic search beyond simple keyword matching.
"""

# ──────────────────────────────────────────────────────────────────
# STEP 1: Chunk the knowledge base
# ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Split text into chunks")
print("=" * 60)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],
)

chunks = splitter.create_documents([KNOWLEDGE_BASE])

print(f"Original text : {len(KNOWLEDGE_BASE)} characters")
print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n  Chunk {i+1} ({len(chunk.page_content)} chars):\n  {chunk.page_content[:100]}...")

# ──────────────────────────────────────────────────────────────────
# STEP 2: Embed chunks → FAISS index
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Embed chunks and build FAISS index")
print("=" * 60)

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

vectorstore = FAISS.from_documents(chunks, embeddings)
print(f"FAISS index built with {len(chunks)} vectors.\n")

# ──────────────────────────────────────────────────────────────────
# STEP 3: Build the RAG chain using LCEL
# ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Build RAG chain with LangChain Expression Language (LCEL)")
print("=" * 60)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Prompt template — instructs the LLM to answer ONLY from context
rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:""")


def format_docs(docs: list[Document]) -> str:
    """Join retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


# LCEL chain:  question → retrieve → prompt → LLM → string
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# ──────────────────────────────────────────────────────────────────
# STEP 4: Ask questions!
# ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 4: Ask questions against our knowledge base")
print("=" * 60)

questions = [
    "What is LangChain and who created it?",
    "How fast is Groq's inference in tokens per second?",
    "What embedding model produces 384-dimensional vectors?",
    "What is HNSW?",    # Not in KB — should say "I don't have that information"
]

for question in questions:
    print(f"\nQ: {question}")
    # Show which chunks were retrieved
    retrieved_chunks = retriever.invoke(question)
    print(f"   [Retrieved {len(retrieved_chunks)} chunks]")
    answer = rag_chain.invoke(question)
    print(f"A: {answer}")

print("""
SUMMARY
-------
RAG Pipeline:
  1. Chunk  → RecursiveCharacterTextSplitter
  2. Embed  → HuggingFaceEmbeddings (all-MiniLM-L6-v2)
  3. Index  → FAISS.from_documents()
  4. Retrieve → vectorstore.as_retriever()
  5. Generate → ChatGroq with context-grounded prompt
  6. Chain  → LCEL pipe operator (|)
""")
