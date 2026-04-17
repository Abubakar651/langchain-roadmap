"""
PDF Processor
-------------
Loads PDF files, splits them into overlapping chunks, and returns
LangChain Document objects ready for embedding.
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    """Load all pages from a PDF file."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    # Attach the filename to each page's metadata
    filename = Path(file_path).name
    for page in pages:
        page.metadata["filename"] = filename
    return pages


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """
    Split documents into smaller overlapping chunks.

    Args:
        documents   : List of Document objects (e.g. from PyPDFLoader)
        chunk_size  : Maximum number of characters per chunk
        chunk_overlap: Number of characters shared between adjacent chunks
    Returns:
        List of chunked Document objects with preserved metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # Number each chunk sequentially per source file
    counts: dict[str, int] = {}
    for chunk in chunks:
        filename = chunk.metadata.get("filename", "unknown")
        counts[filename] = counts.get(filename, 0) + 1
        chunk.metadata["chunk_index"] = counts[filename]

    return chunks


def process_pdfs(
    file_paths: list[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """
    Load and chunk multiple PDF files.

    Returns merged list of all chunks across all PDFs.
    """
    all_chunks: list[Document] = []

    for path in file_paths:
        try:
            pages = load_pdf(path)
            chunks = split_documents(pages, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
            filename = Path(path).name
            print(f"  Loaded '{filename}': {len(pages)} pages → {len(chunks)} chunks")
        except Exception as e:
            print(f"  ERROR loading '{path}': {e}")

    return all_chunks
