"""
AI-Powered PDF Search Assistant
--------------------------------
Streamlit app that lets you:
  1. Upload one or more PDF files
  2. Ask natural-language questions about their content
  3. Get grounded answers with source references

Run:
    cd phase4/project
    streamlit run app.py
"""

import sys
import os
import warnings
import logging
import tempfile

# Suppress noisy but harmless warnings before any imports
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*__path__.*")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st

# Allow imports from the project directory
sys.path.insert(0, os.path.dirname(__file__))

from pdf_processor import process_pdfs
from vector_store import (
    build_vector_store,
    load_vector_store,
    add_documents,
    clear_index,
    index_exists,
)
from rag_chain import build_rag_chain, ask


# ──────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Search Assistant",
    page_icon="📄",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {"role": ..., "content": ...}

if "db" not in st.session_state:
    st.session_state.db = None

if "chain" not in st.session_state:
    st.session_state.chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []




# ──────────────────────────────────────────────────────────────────
# Sidebar — document upload & management
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Document Management")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDFs to build the knowledge base.",
    )

    col1, col2 = st.columns(2)

    with col1:
        index_btn = st.button("Index PDFs", type="primary", use_container_width=True)

    with col2:
        clear_btn = st.button("Clear Index", type="secondary", use_container_width=True)

    st.divider()

    # Retrieval settings
    st.subheader("Retrieval Settings")
    k_chunks = st.slider(
        "Chunks to retrieve (k)",
        min_value=1, max_value=10, value=4,
        help="More chunks = more context, but slower and may confuse LLM.",
    )
    chunk_size = st.slider(
        "Chunk size (characters)",
        min_value=200, max_value=2000, value=1000, step=100,
    )
    chunk_overlap = st.slider(
        "Chunk overlap (characters)",
        min_value=0, max_value=400, value=150, step=50,
    )

    st.divider()

    # Status panel
    st.subheader("Status")
    if index_exists() or st.session_state.db is not None:
        st.success("Index is ready")
        if st.session_state.indexed_files:
            for f in st.session_state.indexed_files:
                st.caption(f"  PDF  {f}")
    else:
        st.warning("No index loaded — upload PDFs first")

# ──────────────────────────────────────────────────────────────────
# Index button logic
# ──────────────────────────────────────────────────────────────────
if index_btn:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF file first.")
    else:
        with st.spinner("Processing PDFs and building vector index..."):
            # Save uploaded files to temp paths so PyPDFLoader can read them
            temp_paths = []
            for uf in uploaded_files:
                suffix = f"_{uf.name}"
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp:
                    tmp.write(uf.read())
                    temp_paths.append(tmp.name)

            # Chunk PDFs
            st.sidebar.write("Chunking PDFs...")
            chunks = process_pdfs(temp_paths, chunk_size, chunk_overlap)

            if not chunks:
                st.sidebar.error("No text could be extracted from the PDFs.")
            else:
                # Check if we should extend or replace the existing index
                existing_db = load_vector_store()
                if existing_db is not None:
                    st.sidebar.write("Extending existing index...")
                    db = add_documents(existing_db, chunks)
                else:
                    st.sidebar.write("Building new index...")
                    db = build_vector_store(chunks)

                # Build RAG chain
                chain, retriever = build_rag_chain(db, k=k_chunks)

                st.session_state.db = db
                st.session_state.chain = chain
                st.session_state.retriever = retriever
                st.session_state.indexed_files = [uf.name for uf in uploaded_files]

                # Clean up temp files
                for p in temp_paths:
                    os.unlink(p)

                st.sidebar.success(
                    f"Indexed {len(chunks)} chunks from {len(uploaded_files)} PDF(s)!"
                )
                st.rerun()

# ──────────────────────────────────────────────────────────────────
# Clear button logic
# ──────────────────────────────────────────────────────────────────
if clear_btn:
    clear_index()
    st.session_state.db = None
    st.session_state.chain = None
    st.session_state.retriever = None
    st.session_state.indexed_files = []
    st.session_state.chat_history = []
    st.sidebar.success("Index cleared.")
    st.rerun()

# ──────────────────────────────────────────────────────────────────
# Auto-load existing index if not yet loaded in this session
# ──────────────────────────────────────────────────────────────────
if st.session_state.db is None and index_exists():
    with st.spinner("Loading existing index..."):
        db = load_vector_store()
    if db is not None:
        chain, retriever = build_rag_chain(db, k=k_chunks)
        st.session_state.db = db
        st.session_state.chain = chain
        st.session_state.retriever = retriever

# ──────────────────────────────────────────────────────────────────
# Main area — chat interface
# ──────────────────────────────────────────────────────────────────
st.title("PDF Search Assistant")
st.caption("Upload PDFs in the sidebar, then ask questions about their content.")

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("View source chunks"):
                for i, doc in enumerate(msg["sources"], 1):
                    filename = doc.metadata.get("filename", "Unknown")
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Source {i}** — `{filename}`, page {page}")
                    st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Check if index is ready
    if st.session_state.chain is None:
        response = "Please upload and index some PDF files first (use the sidebar)."
        sources = []
    else:
        with st.spinner("Searching documents and generating answer..."):
            response, sources = ask(
                st.session_state.chain,
                st.session_state.retriever,
                prompt,
            )

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
        if sources:
            with st.expander("View source chunks"):
                for i, doc in enumerate(sources, 1):
                    filename = doc.metadata.get("filename", "Unknown")
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Source {i}** — `{filename}`, page {page}")
                    st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
    })
