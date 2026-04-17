# PDF Search Assistant

An AI-powered document search application that lets you upload PDFs and ask natural language questions — returning precise, source-cited answers grounded in your documents.

Built with **LangChain**, **FAISS**, **HuggingFace Embeddings**, **Groq LLM**, and **Streamlit**.

---

## Features

- Upload one or more PDF documents via a clean web interface
- Semantic search using vector embeddings — finds meaning, not just keywords
- Answers grounded strictly in the uploaded documents (no hallucinations)
- Source citations with page-level traceability for every response
- Persistent FAISS index — no re-indexing needed between sessions
- Configurable retrieval settings (chunk size, overlap, top-k)

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **LangChain** | Orchestration — chains, retrievers, prompt templates |
| **FAISS** | Local vector store for fast similarity search |
| **HuggingFace** | Embedding model (`all-MiniLM-L6-v2`, 384 dimensions) |
| **Groq API** | LLM inference (`llama-3.3-70b-versatile`) |
| **PyPDF** | PDF text extraction |
| **Streamlit** | Web UI |

---

## Project Structure

```
PDF-Search-Assistant/
├── .gitignore
├── requirements.txt
├── README.md
└── project/
    ├── app.py              # Streamlit web app (entry point)
    ├── pdf_processor.py    # PDF loading and text chunking
    ├── vector_store.py     # FAISS index management
    ├── rag_chain.py        # LangChain RAG chain
    └── demo_cli.py         # CLI interface for terminal testing
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Abubakar651/PDF-Search-Assistant.git
cd PDF-Search-Assistant
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file inside the `project/` directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

---

## Running the App

```bash
cd project
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. **Upload PDFs** — Use the sidebar to select one or more PDF files
2. **Index** — Click "Index PDFs" and wait for the confirmation message
3. **Ask questions** — Type any natural language question in the chat input
4. **View sources** — Expand "View source chunks" under any answer to see the exact passage and page used

---

## How It Works

```
PDF Input
   │
   ▼
[PyPDFLoader]                     Extract text page by page
   │
   ▼
[RecursiveCharacterTextSplitter]  Split into overlapping chunks
   │
   ▼
[HuggingFaceEmbeddings]           Convert chunks to 384-dim vectors
   │
   ▼
[FAISS Index]                     Store and persist vectors locally
   │
   ▼  ◄── User query
[MMR Retrieval]                   Fetch top-k relevant chunks
   │
   ▼
[Groq LLM]                        Generate grounded answer
   │
   ▼
Answer + Source Citations
```

---

## CLI Mode

Test the pipeline from the terminal without a browser:

```bash
cd project
python demo_cli.py path/to/your.pdf
```

Type questions interactively. Enter `quit` to exit.

---

## Environment

- Python 3.9+
- HuggingFace model (`all-MiniLM-L6-v2`) downloads automatically on first run (~90 MB, cached at `~/.cache/huggingface/`)
- FAISS index is persisted locally at `project/faiss_index/`
