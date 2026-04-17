# AI Research Assistant

> **A fully functional CLI research assistant built with LangChain, Groq LLaMA 3.3, and ReAct agents.**

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3.14-green?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-orange?style=flat-square)
![SQLite](https://img.shields.io/badge/Database-SQLite-lightblue?style=flat-square&logo=sqlite)

---

## What This Does

A production-like AI assistant that answers your questions using **live web search**, **Wikipedia**, a **calculator**, and **real-time date/time** — all decided automatically by a ReAct agent.

```
You:  What is the current petrol price in Pakistan?

  🤔 Researching...

  Thought:     I need to search for the latest petrol price in Pakistan.
  Action:      web_search("current petrol price Pakistan 2026")
  Observation: The current petrol price is Rs. 366/liter — OGRA, April 2026
  Thought:     I have enough information.
  Final Answer: The current petrol price in Pakistan is Rs. 366 per liter.
```

---

## Project Structure

```
phase3/
│
├── app.py                  ← Main CLI app (run this)
├── agents.py               ← ReAct, Zero-shot & Self-ask agent demos
├── memory.py               ← Buffer, Entity & Vector memory demos
├── tools.py                ← Web search, Wikipedia, Calculator, Datetime
├── requirements.txt        ← All dependencies
├── .env.example            ← API key template
│
└── notebooks/
    ├── 01_agents.ipynb     ← Learn: What are agents? ReAct pattern
    ├── 02_tools.ipynb      ← Learn: Tools, APIs, DB integration
    └── 03_memory.ipynb     ← Learn: Memory types & implementation
```

---

## Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/Abubakar651/langchain-roadmap.git
cd langchain-roadmap/phase3
pip install -r requirements.txt
```

### 2. Set up your API key

```bash
cp .env.example .env
```

Then open `.env` and add your free Groq API key from [console.groq.com](https://console.groq.com):

```env
GROQ_API_KEY=your_key_here
```

### 3. Run the assistant

```bash
python app.py
```

---

## App Commands

| Command | Description |
|:--------|:------------|
| `Any question` | Ask the agent anything |
| `/history` | View last 10 Q&A pairs stored in SQLite |
| `/memory` | Show current conversation context |
| `/clear` | Clear memory and start fresh |
| `/verbose` | Toggle agent reasoning trace on/off |
| `/help` | Show all commands |
| `/quit` | Exit the assistant |

---

## Tools Available to the Agent

| Tool | Source | When the agent uses it |
|:-----|:-------|:----------------------|
| `web_search` | DuckDuckGo — free, no key | Current events, prices, news |
| `wikipedia` | Wikipedia API — free, no key | Factual & encyclopedic questions |
| `calculator` | Custom Python | Any math expression |
| `current_datetime` | Custom Python | Date/time awareness |

---

## How It Works

```
                    ┌─────────────────────────────┐
   User Question ──►│  ConversationBufferMemory   │
                    │  (injects chat history)      │
                    └────────────┬────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────┐
                    │   ReAct Agent               │
                    │   LLaMA 3.3 70B via Groq    │
                    │                             │
                    │  Thought → Action → Observe │
                    │  Thought → Action → Observe │
                    │  ...                        │
                    │  Final Answer               │
                    └────────────┬────────────────┘
                                 │
                    ┌────────────▼────────────────┐
                    │  Save to SQLite             │
                    │  Save to Memory Buffer      │
                    └─────────────────────────────┘
```

---

## Memory Types Demonstrated

| Type | How it works | Best for |
|:-----|:-------------|:---------|
| `ConversationBufferMemory` | Stores every message verbatim | Short sessions, chatbots |
| `ConversationEntityMemory` | Extracts named entities via LLM | Personal assistants, CRM bots |
| `VectorStoreRetrieverMemory` | Semantic search over past messages | Long-term memory, large history |

---

## Run Individual Modules

```bash
# Test all 4 tools independently
python tools.py

# Demo all 3 memory types
python memory.py

# Demo ReAct agent with tool use
python agents.py
```

---

## Topics Covered

<details>
<summary><strong>LangChain Agents</strong></summary>

- What are Agents and how they differ from plain LLM chains
- Agent types: ReAct, Zero-shot, Self-ask (conceptual + runnable code)
- Building agents with `create_react_agent` + `AgentExecutor`
- Tool description engineering — the key to agent performance

</details>

<details>
<summary><strong>Integrating External Tools</strong></summary>

- DuckDuckGo Search via `ddgs` (no API key needed)
- Wikipedia integration with `WikipediaAPIWrapper`
- Custom Python tools (calculator, datetime)
- SQLite database integration with SQLAlchemy
- REST API integration — CoinGecko example in `02_tools.ipynb`

</details>

<details>
<summary><strong>LangChain Memory</strong></summary>

- `ConversationBufferMemory` — full verbatim conversation history
- `ConversationEntityMemory` — structured entity extraction
- `VectorStoreRetrieverMemory` — semantic long-term memory with ChromaDB
- How memory integrates with ReAct agents in a real app

</details>

---

## Tech Stack

| Package | Purpose |
|:--------|:--------|
| `langchain` + `langchain-community` | Agent framework |
| `langchain-groq` | Groq LLM integration |
| `groq` | LLaMA 3.3 70B inference |
| `ddgs` | DuckDuckGo web search |
| `wikipedia` | Wikipedia API |
| `chromadb` + `langchain-chroma` | Vector store for memory |
| `sentence-transformers` | Free local embeddings |
| `sqlalchemy` | SQLite ORM for history |
| `python-dotenv` | Environment variable management |

---

## Learning Path

Start here if you're learning:

```
01_agents.ipynb  →  02_tools.ipynb  →  03_memory.ipynb  →  app.py
   (concepts)         (tools)            (memory)          (full app)
```

---

## Related Projects

- [PDF Search Assistant](https://github.com/Abubakar651/PDF-Search-Assistant) — RAG-powered PDF Q&A with FAISS and Streamlit
