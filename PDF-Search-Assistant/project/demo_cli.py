"""
CLI Demo — test the RAG pipeline without Streamlit.

Usage:
    cd phase4/project
    python demo_cli.py path/to/your.pdf
    python demo_cli.py ../sample_docs/ai_overview.pdf
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from pdf_processor import process_pdfs
from vector_store import build_vector_store
from rag_chain import build_rag_chain, ask


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo_cli.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    print(f"\nProcessing: {pdf_path}")
    print("-" * 50)

    # Process PDF
    chunks = process_pdfs([pdf_path], chunk_size=1000, chunk_overlap=150)
    print(f"Total chunks created: {len(chunks)}\n")

    # Build vector store
    print("Building FAISS index...")
    db = build_vector_store(chunks)

    # Build RAG chain
    chain, retriever = build_rag_chain(db, k=4)

    print("\nRAG pipeline ready! Type your questions (or 'quit' to exit).\n")
    print("=" * 60)

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        print("\nSearching documents...")
        answer, sources = ask(chain, retriever, question)

        print(f"\nAnswer:\n{answer}")
        print(f"\nSources used:")
        for i, doc in enumerate(sources, 1):
            filename = doc.metadata.get("filename", "Unknown")
            page = doc.metadata.get("page", "?")
            preview = doc.page_content[:100].replace("\n", " ")
            print(f"  [{i}] {filename} p.{page} — {preview}...")


if __name__ == "__main__":
    main()
