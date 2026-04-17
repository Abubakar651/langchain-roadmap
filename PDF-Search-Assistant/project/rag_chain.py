"""
RAG Chain
---------
Builds the LangChain LCEL pipeline that:
  1. Retrieves relevant chunks from FAISS
  2. Formats them into a prompt
  3. Sends to Groq LLM
  4. Returns a grounded answer + the source chunks used
"""

from pathlib import Path
from dotenv import load_dotenv

# Load .env from this file's directory (works regardless of cwd)
load_dotenv(Path(__file__).parent / ".env")

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from vector_store import get_retriever

# ──────────────────────────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful AI assistant that answers questions based on the provided document excerpts.

Instructions:
- Answer ONLY using information from the context below.
- Be concise and direct.
- If the context contains partial information, share what you know.
- If the answer is not in the context, say: "I couldn't find relevant information in the uploaded documents."
- Always mention which document/page the information came from when available.

Context from documents:
{context}

Question: {question}

Answer:""")


def format_docs_with_sources(docs: list[Document]) -> str:
    """Format retrieved chunks, showing filename and page number."""
    parts = []
    for i, doc in enumerate(docs, 1):
        filename = doc.metadata.get("filename", "Unknown")
        page = doc.metadata.get("page", "?")
        parts.append(
            f"[Source {i}: {filename}, page {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def build_rag_chain(db: FAISS, k: int = 4):
    """
    Build and return the RAG chain + retriever.

    Returns:
        (chain, retriever) tuple
        - chain    : invokeable with a question string → answer string
        - retriever: invokeable with a question string → list of Documents
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    retriever = get_retriever(db, k=k)

    # Parallel branch: retrieve for context AND pass question through
    setup = RunnableParallel(
        context=retriever | format_docs_with_sources,
        question=RunnablePassthrough(),
    )

    chain = setup | RAG_PROMPT | llm | StrOutputParser()

    return chain, retriever


def ask(chain, retriever, question: str) -> tuple[str, list[Document]]:
    """
    Ask a question and return (answer, source_docs).

    Args:
        chain     : RAG chain from build_rag_chain()
        retriever : retriever from build_rag_chain()
        question  : user's question string
    Returns:
        (answer_text, list_of_source_documents)
    """
    answer = chain.invoke(question)
    sources = retriever.invoke(question)
    return answer, sources
