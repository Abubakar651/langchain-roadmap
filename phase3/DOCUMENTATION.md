# Complete Technical Documentation
## Phase 3 — AI Research Assistant with LangChain

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Data Flow](#2-architecture--data-flow)
3. [Core Concepts](#3-core-concepts)
   - 3.1 [What is LangChain?](#31-what-is-langchain)
   - 3.2 [What is a Large Language Model (LLM)?](#32-what-is-a-large-language-model-llm)
   - 3.3 [What is Groq?](#33-what-is-groq)
   - 3.4 [What is a ReAct Agent?](#34-what-is-a-react-agent)
   - 3.5 [What are Tools?](#35-what-are-tools)
   - 3.6 [What is Memory?](#36-what-is-memory)
4. [File-by-File Breakdown](#4-file-by-file-breakdown)
   - 4.1 [tools.py](#41-toolspy)
   - 4.2 [memory.py](#42-memorypy)
   - 4.3 [agents.py](#43-agentspy)
   - 4.4 [app.py](#44-apppy)
5. [Agents — Deep Dive](#5-agents--deep-dive)
6. [Tools — Deep Dive](#6-tools--deep-dive)
7. [Memory — Deep Dive](#7-memory--deep-dive)
8. [Database Layer — SQLAlchemy + SQLite](#8-database-layer--sqlalchemy--sqlite)
9. [Environment & Configuration](#9-environment--configuration)
10. [Dependencies Explained](#10-dependencies-explained)
11. [Common Interview Questions & Answers](#11-common-interview-questions--answers)

---

## 1. Project Overview

This project is a **CLI-based AI Research Assistant** that can answer questions using real-world data. It is not a simple chatbot — it is an **autonomous agent** that:

- **Decides** which tool to use based on your question
- **Fetches** live data from the web, Wikipedia, or computes math
- **Remembers** the conversation so you can ask follow-up questions
- **Stores** every Q&A in a SQLite database for history

### What makes it different from a plain chatbot?

| Feature | Plain ChatBot | This Assistant |
|:--------|:-------------|:---------------|
| Web access | No — uses only training data | Yes — live DuckDuckGo search |
| Math accuracy | Approximates | Exact — uses Python `eval()` |
| Date awareness | Knows training cutoff only | Knows real current date/time |
| Memory | Forgets after each message | Remembers full conversation |
| History | Lost after session | Persisted in SQLite database |
| Reasoning | Direct answer | Shows Thought → Action → Answer |

### Tech Stack at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│                        USER (CLI)                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   app.py  (Orchestrator)                     │
│   ConversationBufferMemory + AgentExecutor + SQLAlchemy      │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              ReAct Agent  (agents.py)                        │
│         LLaMA 3.3 70B via Groq API                          │
└────┬──────────────┬──────────────┬────────────┬─────────────┘
     │              │              │            │
┌────▼────┐  ┌──────▼────┐  ┌─────▼────┐ ┌────▼──────┐
│web_     │  │wikipedia  │  │calculator│ │current_   │
│search   │  │           │  │          │ │datetime   │
│(ddgs)   │  │(Wikipedia │  │(Python   │ │(Python    │
│         │  │ API)      │  │ math)    │ │ datetime) │
└─────────┘  └───────────┘  └──────────┘ └───────────┘
```

---

## 2. Architecture & Data Flow

### Step-by-step flow when you ask a question

```
Step 1: User types a question
        "What is the petrol price in Pakistan?"

Step 2: app.py calls ResearchAssistant.ask(question)

Step 3: ConversationBufferMemory loads previous chat history
        Previous turns are injected into the question context

Step 4: AgentExecutor.invoke({"input": context_question})
        Sends the enriched prompt to the LLM

Step 5: LLM (LLaMA 3.3 70B) runs the ReAct loop:
        ┌──────────────────────────────────────────┐
        │ Thought: I need to search for this       │
        │ Action: web_search                        │
        │ Action Input: "petrol price Pakistan"     │
        │ Observation: Rs. 366/liter (OGRA 2026)   │
        │ Thought: I have enough info               │
        │ Final Answer: Petrol price is Rs. 366    │
        └──────────────────────────────────────────┘

Step 6: Answer is returned to app.py

Step 7: Answer is saved to ConversationBufferMemory
        (for future context in this session)

Step 8: Answer is saved to SQLite database
        (persisted permanently for history)

Step 9: Answer is printed to the user
```

---

## 3. Core Concepts

### 3.1 What is LangChain?

LangChain is a **Python framework** that makes it easy to build applications powered by Large Language Models (LLMs). It provides pre-built components for:

- Connecting to LLMs (OpenAI, Groq, Anthropic, etc.)
- Creating agents that can use tools
- Managing conversation memory
- Building chains of LLM calls
- Integrating vector databases for semantic search

**Why use LangChain instead of calling the API directly?**

Without LangChain, you would need to manually:
- Format the ReAct prompt
- Parse the LLM's output to extract the tool name and input
- Call the tool
- Feed the result back to the LLM
- Handle errors and retries
- Manage conversation history

LangChain handles all of this automatically.

**Example — Without LangChain vs With LangChain:**

```python
# WITHOUT LangChain — you handle everything manually
import groq
client = groq.Client()
history = []
while True:
    user_input = input("You: ")
    history.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=history
    )
    answer = response.choices[0].message.content
    history.append({"role": "assistant", "content": answer})
    print("AI:", answer)
    # No tools, no memory management, no agent reasoning...

# WITH LangChain — framework handles all complexity
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)
result = agent_executor.invoke({"input": "What is the petrol price?"})
# Agent automatically chooses web_search, fetches data, formats answer
```

---

### 3.2 What is a Large Language Model (LLM)?

An LLM is a neural network trained on massive amounts of text that can understand and generate human language. In this project we use **LLaMA 3.3 70B** — Meta's open-source model with 70 billion parameters.

**Key properties:**
- **Stateless** — each API call is independent; the model has no memory by default
- **Token-based** — processes text as chunks called tokens (roughly 1 token ≈ 0.75 words)
- **Temperature** — controls randomness. `0.0` = deterministic, `1.0` = very creative

In this project, temperature is set to `0.3` — low enough for factual accuracy but slight flexibility.

```python
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,          # near-deterministic for factual answers
    groq_api_key=GROQ_API_KEY,
)
```

---

### 3.3 What is Groq?

Groq is a cloud inference provider that runs open-source models like LLaMA on custom LPU (Language Processing Unit) hardware — making them **extremely fast** (typically 400–800 tokens/second vs ~50 for standard GPUs).

**Why Groq instead of OpenAI?**
- Free tier available at [console.groq.com](https://console.groq.com)
- Much faster inference speed
- Runs open-source models (LLaMA, Mixtral, Gemma)
- No per-token billing on free tier (rate limits apply)

**Model used:** `llama-3.3-70b-versatile`
- 70 billion parameters
- 128K context window
- Excellent reasoning and tool-use capability

---

### 3.4 What is a ReAct Agent?

**ReAct = Reasoning + Acting**

ReAct is a prompting strategy where the LLM alternates between:
1. **Thought** — reasoning about what to do next
2. **Action** — calling a tool
3. **Observation** — reading the tool's output
4. Repeating until it has a **Final Answer**

This was introduced in the paper *"ReAct: Synergizing Reasoning and Acting in Language Models"* (Yao et al., 2022).

**Why ReAct over a plain LLM call?**

A plain LLM would hallucinate an answer. ReAct forces the model to:
- Ground its answer in real data (from tools)
- Show its reasoning (transparent and debuggable)
- Handle multi-step problems naturally

**Full ReAct trace example:**

```
Question: Who is the current prime minister of Pakistan and how old are they?

Thought: I need to find who the current prime minister of Pakistan is.
Action: web_search
Action Input: "current prime minister of Pakistan 2026"
Observation: Shehbaz Sharif is the Prime Minister of Pakistan as of 2024.

Thought: Now I need his age or date of birth.
Action: web_search
Action Input: "Shehbaz Sharif date of birth age"
Observation: Shehbaz Sharif was born on September 23, 1951.

Thought: I can calculate his age. Current year is 2026. 2026 - 1951 = 74 years old.
Action: calculator
Action Input: 2026 - 1951
Observation: Result: 75

Thought: I now have all the information needed.
Final Answer: Shehbaz Sharif is the current Prime Minister of Pakistan.
He was born on September 23, 1951, making him 74-75 years old.
```

**Key parameters in this project:**

```python
AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,  # if LLM formats output wrong, recover gracefully
    max_iterations=10,           # stop after 10 Thought/Action cycles (prevent loops)
    max_execution_time=60,       # hard timeout: 60 seconds maximum
)
```

---

### 3.5 What are Tools?

A **Tool** is a Python function that the agent can call. Each tool has:
- **name** — the exact string the LLM uses to invoke it
- **func** — the Python function that runs
- **description** — natural language explanation the LLM reads to decide when to use it

```python
Tool(
    name="calculator",           # LLM writes "Action: calculator"
    func=_calculate,             # Python function that gets called
    description=(
        "Evaluates a mathematical expression and returns the result. "
        "Use this for any math calculation."
    )
)
```

**Critical insight:** The `description` IS the agent's instruction manual. The LLM has never seen your code — it only reads the description and decides whether to use the tool. Vague descriptions = wrong tool choices.

---

### 3.6 What is Memory?

LLMs are **stateless** — each API call is completely independent. Without memory:

```
You: My name is Bakar.
AI:  Nice to meet you, Bakar!
You: What is my name?
AI:  I don't know your name.  ← Forgot!
```

Memory solves this by saving conversation history and injecting it into every new prompt.

```
You: My name is Bakar.
AI:  Nice to meet you, Bakar!
You: What is my name?
[Memory injects: "User said: My name is Bakar. AI said: Nice to meet you!"]
AI:  Your name is Bakar.  ← Remembered!
```

Three memory types are demonstrated in this project (covered in detail in Section 7).

---

## 4. File-by-File Breakdown

### 4.1 `tools.py`

**Purpose:** Defines all 4 tools the agent can use. Can also be run standalone to test tools.

**What it exports:** `get_all_tools()`, `get_search_tool()`, `get_wikipedia_tool()`, `get_calculator_tool()`, `get_datetime_tool()`

**How to test it:**
```bash
python tools.py
```

---

### 4.2 `memory.py`

**Purpose:** Demonstrates all 3 LangChain memory types with runnable examples.

**Classes:**
- `BufferMemoryDemo` — shows ConversationBufferMemory
- `EntityMemoryDemo` — shows ConversationEntityMemory (requires API key)
- `VectorMemoryDemo` — shows VectorStoreRetrieverMemory with ChromaDB

**How to test it:**
```bash
python memory.py
```

---

### 4.3 `agents.py`

**Purpose:** Demonstrates how to build a ReAct agent. Also explains Zero-shot and Self-ask agents conceptually.

**Key function:** `create_react_research_agent()` — builds and returns a ready-to-use AgentExecutor.

**How to test it:**
```bash
python agents.py
# Runs the agent on: "What is the square root of 1764 and today's date?"
```

---

### 4.4 `app.py`

**Purpose:** The main application. Integrates everything — agent, memory, database, and CLI interface.

**Key classes and functions:**

| Component | Type | Purpose |
|:----------|:-----|:--------|
| `ResearchEntry` | SQLAlchemy Model | Defines the database table schema |
| `save_to_db()` | Function | Writes a Q&A pair to SQLite |
| `get_history()` | Function | Reads past Q&A pairs from SQLite |
| `build_agent()` | Function | Creates the ReAct agent + tools |
| `ResearchAssistant` | Class | Wraps the agent with memory management |
| `main()` | Function | Runs the interactive CLI loop |

---

## 5. Agents — Deep Dive

### How `create_react_agent` works internally

```python
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent

# Step 1: Pull the ReAct prompt template from LangChain Hub
prompt = hub.pull("hwchase17/react")
```

The prompt downloaded from LangChain Hub looks like this (simplified):

```
Answer the following questions as best you can.
You have access to the following tools:

{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
```

The `{tools}` placeholder gets filled with each tool's name and description. The `{agent_scratchpad}` is where previous Thought/Action/Observation turns accumulate during a single question.

```python
# Step 2: Create the agent (brain only — no execution loop)
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# Step 3: Wrap in AgentExecutor (adds the execution loop)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    max_execution_time=60,
)
```

**Agent vs AgentExecutor:**
- `agent` — the LLM + prompt + output parser. Only generates ONE Thought/Action response.
- `AgentExecutor` — the loop that keeps calling the agent until it says "Final Answer".

### Agent Types Explained

#### ReAct Agent (used in this project)
- Loops through Thought → Action → Observation
- Transparent reasoning
- Best for: multi-step problems, tool use

#### Zero-Shot Agent
- Makes decisions based purely on tool descriptions
- No examples are provided in the prompt
- "Zero-shot" = zero demonstrations
- **Note:** `create_react_agent` IS a zero-shot agent — it provides no examples, only descriptions

#### Self-Ask Agent
- Breaks complex questions into sub-questions
- Requires a tool called exactly `"Intermediate Answer"`
- Best for: compositional reasoning (A depends on B depends on C)

```
Question: "Who was president of the US when the Eiffel Tower was built?"

Are follow up questions needed? Yes.
Follow up: When was the Eiffel Tower built?
Intermediate answer: 1889.
Follow up: Who was US president in 1889?
Intermediate answer: Grover Cleveland.
Final answer: Grover Cleveland.
```

---

## 6. Tools — Deep Dive

### Tool 1: Web Search

```python
from ddgs import DDGS

def _web_search(query: str) -> str:
    try:
        results = list(DDGS().text(query, max_results=5))
        if not results:
            return "No results found for that query."
        return "\n\n".join(
            f"{r.get('title', '')}\n{r.get('body', '')}"
            for r in results
        )
    except Exception as exc:
        return f"Search error: {exc}"
```

**How it works:**
1. `DDGS()` creates a DuckDuckGo search session
2. `.text(query, max_results=5)` fetches top 5 search results
3. Each result has `title`, `body`, and `href` fields
4. We join them into a single string the LLM can read

**Why `ddgs` and not `duckduckgo-search`?**
The `duckduckgo-search` package was renamed to `ddgs`. The old package had a critical `UnboundLocalError` bug in version 6.3.x across all backends (API, HTML, lite) due to DuckDuckGo changing their API. The new `ddgs` package works correctly.

---

### Tool 2: Wikipedia

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
wiki = WikipediaQueryRun(api_wrapper=wrapper)
```

**How it works:**
1. `WikipediaAPIWrapper` wraps the `wikipedia` Python package
2. `top_k_results=2` — fetches summaries from the top 2 matching articles
3. `doc_content_chars_max=1000` — truncates each article to 1000 characters
4. `WikipediaQueryRun` makes it compatible with LangChain's Tool interface

**When to use Wikipedia vs Web Search:**
- Wikipedia: deep factual knowledge (history, science, biographies)
- Web Search: current events, prices, news, recent data

---

### Tool 3: Calculator

```python
def _calculate(expression: str) -> str:
    try:
        safe_namespace = {
            "sqrt": math.sqrt,
            "log": math.log,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
            "round": round,
        }
        result = eval(expression, {"__builtins__": {}}, safe_namespace)
        return f"Result: {result}"
    except Exception as exc:
        return f"Calculator error: {exc}"
```

**How it works:**
Python's `eval()` executes a string as Python code. This is normally dangerous because a user could pass `__import__('os').system('rm -rf /')`.

We prevent this with two arguments to `eval()`:
- `{"__builtins__": {}}` — removes ALL built-in functions (`open`, `exec`, `__import__`, etc.)
- `safe_namespace` — provides ONLY the math functions we want

```python
# Example calls the agent makes:
_calculate("sqrt(1764)")        # → "Result: 42.0"
_calculate("340 * 0.15")        # → "Result: 51.0"
_calculate("2 ** 10")           # → "Result: 1024"
_calculate("sin(pi / 2)")       # → "Result: 1.0"
```

---

### Tool 4: Current Datetime

```python
def _get_datetime(_: str = "") -> str:
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y %I:%M %p')}"
```

**Why does the LLM need this?**
LLMs have a training cutoff date. Without this tool, the model would guess the current date or use its training cutoff. This tool gives it the real date/time.

**The `_: str = ""` parameter:**
LangChain requires all tool functions to accept exactly one string argument (the `Action Input`). Since this tool doesn't need any input, we use `_` as a throwaway parameter name and give it a default value of `""`.

```python
# Agent call:
Action: current_datetime
Action Input: (empty or anything)
Observation: Current date and time: Thursday, April 16, 2026 06:30 PM
```

---

## 7. Memory — Deep Dive

### Memory Type 1: ConversationBufferMemory

**How it works:** Stores EVERY message verbatim in a Python list. On each new question, the entire list is injected into the prompt.

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",  # key used to inject into prompt
    return_messages=False,       # False = plain text, True = Message objects
    human_prefix="User",
    ai_prefix="Assistant",
)

# Saving a turn
memory.save_context(
    {"input": "My name is Bakar"},
    {"output": "Nice to meet you, Bakar!"}
)

# Loading history (returns a dict)
history = memory.load_memory_variables({})
# → {"chat_history": "User: My name is Bakar\nAssistant: Nice to meet you, Bakar!"}
```

**What gets injected into the prompt:**
```
Previous conversation:
User: My name is Bakar.
Assistant: Nice to meet you, Bakar!
User: I am learning LangChain.
Assistant: That's great! LangChain is powerful.

Current question: What did I say my name was?
```

**Pros and Cons:**

| Pros | Cons |
|:-----|:-----|
| Simple to implement | Grows indefinitely |
| Perfectly accurate | Gets expensive as conversation grows |
| No LLM call needed | Can exceed context window on long chats |

**Used in app.py because:** The research assistant is designed for focused sessions, not infinite conversations. Buffer memory is the most reliable option.

---

### Memory Type 2: ConversationEntityMemory

**How it works:** Instead of storing the raw conversation, it calls the LLM after each turn to extract and update a structured **entity store**.

```python
from langchain.memory import ConversationEntityMemory

memory = ConversationEntityMemory(llm=llm)

memory.save_context(
    {"input": "Elon Musk founded SpaceX in 2002."},
    {"output": "Yes, SpaceX revolutionised rockets."}
)

# Entity store gets built:
# {
#   "Elon Musk": "Founded SpaceX in 2002.",
#   "SpaceX":    "Rocket company founded by Elon Musk in 2002."
# }
```

**How entity extraction works internally:**
After `save_context()`, the memory makes an LLM call with a prompt like:

```
From the conversation below, extract any named entities (people, places,
organizations, products) and summarize what we know about each.Conversation: "Elon Musk founded SpaceX in 2002."
```

**Pros and Cons:**

| Pros | Cons |
|:-----|:-----|
| Lightweight — only stores key facts | Requires LLM call on every save |
| Structured output | Entity extraction can miss things |
| Great for tracking people/places | Costs API tokens |

---

### Memory Type 3: VectorStoreRetrieverMemory

**How it works:** Every conversation turn is converted into a **vector (embedding)** and stored in a vector database (ChromaDB). When a new question comes in, the system finds the most **semantically similar** past turns and injects only those.

```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Create embeddings model (converts text → vector of numbers)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Create vector store
vectorstore = Chroma(
    embedding_function=embeddings,
    collection_name="memory_demo",
)

# 3. Create memory with semantic retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
memory = VectorStoreRetrieverMemory(retriever=retriever)
```

**What is an embedding?**
An embedding is a list of floating-point numbers that represents the meaning of text. Similar sentences have similar vectors.

```
"What is LangChain?"     → [0.12, -0.45, 0.87, ...]  (384 numbers)
"Tell me about LangChain" → [0.11, -0.44, 0.86, ...]  (very similar!)
"What is the weather?"    → [0.91,  0.23, -0.12, ...] (very different)
```

**Semantic retrieval example:**

```
Past stored turns:
  [v1] "What is LangChain?" → "LangChain is a framework for LLM apps."
  [v2] "What is ChromaDB?"  → "ChromaDB is a vector database."
  [v3] "Tell me about agents." → "Agents use LLMs to decide tool use."
  [v4] "Who is Bakar?"     → "Bakar is a developer learning LangChain."

New question: "Tell me about AI frameworks"
→ Semantic search finds: [v1] and [v3] (most similar)
→ Injects only those two turns into the prompt (NOT v2 and v4)
```

**Pros and Cons:**

| Pros | Cons |
|:-----|:-----|
| Scales to thousands of turns | Requires embedding model setup |
| Only injects relevant history | Not perfectly chronological |
| No context window overflow | Semantic similarity can miss things |

**Model used:** `sentence-transformers/all-MiniLM-L6-v2`
- 22M parameters, runs locally, no API key needed
- Produces 384-dimensional vectors
- Fast and accurate for English text similarity

---

## 8. Database Layer — SQLAlchemy + SQLite

### Why store to a database?

`ConversationBufferMemory` lives **in RAM** — it is lost when the program exits. SQLite gives **permanent storage** so you can review past research sessions with `/history`.

### The ORM Model

```python
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session

class Base(DeclarativeBase):
    pass

class ResearchEntry(Base):
    __tablename__ = "research_history"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), nullable=False)
    question   = Column(Text, nullable=False)
    answer     = Column(Text, nullable=False)
    tools_used = Column(String(200), default="")
    timestamp  = Column(DateTime, default=lambda: datetime.now(timezone.utc))
```

**What is an ORM?**
ORM (Object-Relational Mapper) lets you work with database tables as Python classes. Instead of writing SQL:
```sql
INSERT INTO research_history (session_id, question, answer) VALUES (?, ?, ?);
```
You write Python:
```python
entry = ResearchEntry(session_id="abc", question="...", answer="...")
db_session.add(entry)
db_session.commit()
```

### Creating the database and table

```python
engine = create_engine("sqlite:///research_history.db", echo=False)
Base.metadata.create_all(engine)
# Creates the file research_history.db and the table if they don't exist
```

- `sqlite:///research_history.db` — SQLite file in the current directory
- `echo=False` — don't print SQL statements to the console
- `create_all()` — reads all ORM models and creates their tables

### Saving a Q&A pair

```python
def save_to_db(session_id: str, question: str, answer: str) -> None:
    with Session(engine) as db_session:
        entry = ResearchEntry(
            session_id=session_id,
            question=question,
            answer=answer,
        )
        db_session.add(entry)
        db_session.commit()
```

The `with Session(engine) as db_session:` block automatically closes the connection when done, even if an error occurs.

### Reading history

```python
def get_history(session_id: str, limit: int = 10) -> list[ResearchEntry]:
    with Session(engine) as db_session:
        return (
            db_session.query(ResearchEntry)
            .filter_by(session_id=session_id)
            .order_by(ResearchEntry.timestamp.desc())
            .limit(limit)
            .all()
        )
```

This translates to:
```sql
SELECT * FROM research_history
WHERE session_id = ?
ORDER BY timestamp DESC
LIMIT 10;
```

### Session ID

Each run of the app generates a unique session ID:

```python
session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# Example: "session_20260416_183045"
```

This allows the history to be filtered per-session. Future versions could support user-specific sessions.

---

## 9. Environment & Configuration

### The `.env` file

Never hardcode API keys in source code. Use environment variables:

```env
# .env  (never commit this file!)
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxx

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=lsv2_xxxxxxxxxxxxxxxxxxxxx
LANGSMITH_PROJECT=phase3-research-assistant
```

### How `python-dotenv` works

```python
from dotenv import load_dotenv
import os

load_dotenv()  # reads .env file and sets environment variables

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

`load_dotenv()` reads the `.env` file and calls `os.environ[KEY] = VALUE` for each line. The variables are then accessible with `os.getenv()`.

### Validation at startup

```python
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    print("❌ GROQ_API_KEY is not set!")
    sys.exit(1)
```

The app exits immediately with a clear error if the key is missing or still has the placeholder value. This prevents confusing errors later.

### `.gitignore` entries

```gitignore
.env                 # Never commit API keys
venv/                # Virtual environment (huge, rebuild from requirements.txt)
__pycache__/         # Python bytecode cache
*.db                 # SQLite database (runtime data)
.ipynb_checkpoints/  # Jupyter auto-saves
.claude/             # IDE-specific config
```

---

## 10. Dependencies Explained

| Package | Version | Purpose |
|:--------|:--------|:--------|
| `langchain` | 0.3.14 | Core framework — agents, memory, chains |
| `langchain-core` | 0.3.29 | Base interfaces and primitives |
| `langchain-community` | 0.3.14 | Third-party integrations (Wikipedia, DDG, Chroma) |
| `langchain-groq` | 0.2.3 | Groq LLM adapter for LangChain |
| `groq` | 0.13.0 | Official Groq Python client |
| `ddgs` | latest | DuckDuckGo search (successor to `duckduckgo-search`) |
| `wikipedia` | 1.4.0 | Wikipedia article fetching |
| `chromadb` | 0.5.23 | Vector database for semantic memory |
| `langchain-chroma` | 0.1.4 | LangChain adapter for ChromaDB |
| `sentence-transformers` | 3.3.1 | Local embedding model (no API key) |
| `sqlalchemy` | 2.0.36 | ORM for SQLite database |
| `python-dotenv` | 1.0.1 | Load `.env` files |
| `tiktoken` | 0.8.0 | Token counting for OpenAI-compatible models |

---

## 11. Common Interview Questions & Answers

---

**Q: What is a ReAct agent and why did you use it?**

A ReAct agent uses the "Reasoning + Acting" pattern where the LLM alternates between Thought (reasoning), Action (calling a tool), and Observation (reading the result), repeating until it reaches a Final Answer. I used it because it handles multi-step problems naturally, grounds answers in real data (not hallucination), and makes the reasoning transparent and debuggable. Unlike a plain LLM call, the agent can decide whether to search the web, compute math, or look up Wikipedia — choosing the right tool for each question.

---

**Q: What is the difference between an Agent and a Chain in LangChain?**

A **Chain** is a fixed, predetermined sequence of steps. Example: always call the LLM, then always format the output. The flow is hardcoded.

An **Agent** is dynamic. It uses the LLM itself to decide which steps to take and in what order. The flow depends on the question. An agent can call one tool, three tools, or no tools — whatever the question requires.

---

**Q: How does memory work in this project?**

The project uses `ConversationBufferMemory`. Before sending a question to the agent, the app calls `memory.load_memory_variables({})` to get the full conversation history as a text string. This history is prepended to the current question so the agent has full context. After the agent answers, `memory.save_context()` adds the new Q&A pair to the buffer. Separately, every Q&A pair is also saved to a SQLite database using SQLAlchemy for permanent storage across sessions.

---

**Q: What is the difference between ConversationBufferMemory, ConversationEntityMemory, and VectorStoreRetrieverMemory?**

- **BufferMemory**: Stores every message verbatim. Simple but grows indefinitely. No LLM call needed. Best for short sessions.
- **EntityMemory**: After each turn, calls the LLM to extract and update a structured store of named entities (people, places, things). More compact but costs extra API tokens.
- **VectorStoreRetrieverMemory**: Converts each turn into a vector embedding and stores it in ChromaDB. On each new question, semantically similar past turns are retrieved. Scales to thousands of turns without context overflow. Best for long-running assistants.

---

**Q: Why is the tool `description` so important?**

The LLM cannot see your Python code. The only information it has about each tool is its `description` string. When the agent receives a question, it reads all tool descriptions and decides which to use purely based on that text. A vague description like `"searches stuff"` will cause the agent to use the tool incorrectly or ignore it entirely. A precise description like `"Searches the web for current events, news, and recent information. Input: a search query string."` helps the agent make the right choice every time.

---

**Q: How did you make the calculator secure?**

I used Python's `eval()` with two safety measures:
1. `{"__builtins__": {}}` — removes all built-in functions, preventing access to `open()`, `exec()`, `__import__()`, and other dangerous functions
2. A `safe_namespace` dictionary — provides only the math functions explicitly allowed (`sqrt`, `sin`, `cos`, etc.)

This means even if a malicious input like `__import__('os').system('rm -rf /')` is passed, it will fail because `__import__` is not in the namespace.

---

**Q: What is `AgentExecutor` and what does `max_iterations` do?**

`AgentExecutor` is the loop that runs the agent. It calls the agent (LLM), executes the chosen tool, feeds the result back to the agent, and repeats until the agent outputs "Final Answer".

`max_iterations=10` is a safety limit. Without it, a confused agent could loop forever (Thought → Action → Observation → Thought → Action...). After 10 iterations it stops and returns whatever it has. `max_execution_time=60` adds a hard time limit of 60 seconds.

---

**Q: What is the difference between `verbose=True` and `verbose=False`?**

When `verbose=True`, the AgentExecutor prints every Thought, Action, Action Input, and Observation to the console in real time. This is useful for debugging but noisy in production. When `verbose=False`, only the Final Answer is returned. In this app, verbose mode can be toggled at runtime with the `/verbose` command.

---

**Q: Why do you use Groq instead of OpenAI?**

Groq provides a free tier with no per-token billing (within rate limits). It runs LLaMA 3.3 70B on custom LPU hardware, achieving 400–800 tokens/second — 8-10x faster than a typical GPU-based API. LLaMA 3.3 70B is an open-source model from Meta that is competitive with GPT-4 on many benchmarks. For a learning project, Groq gives production-quality results at zero cost.

---

**Q: What is SQLAlchemy and why use it instead of raw SQL?**

SQLAlchemy is a Python ORM (Object-Relational Mapper). It lets you work with database records as Python objects instead of writing SQL strings. Benefits:
1. **Type safety** — Python catches errors before they reach the database
2. **Portability** — switching from SQLite to PostgreSQL requires only changing the connection string
3. **Security** — automatically parameterizes queries, preventing SQL injection
4. **Readability** — Python code is easier to read and maintain than embedded SQL strings

---

**Q: What is a vector database and why use ChromaDB?**

A vector database stores high-dimensional vectors (embeddings) and allows fast similarity search — finding which stored vectors are closest to a query vector. ChromaDB is an open-source, easy-to-set-up vector database that can run in-memory (no disk) or persisted to disk.

In this project, ChromaDB stores conversation embeddings for `VectorStoreRetrieverMemory`. When a new question comes in, the top-k most semantically similar past turns are retrieved and injected as context.

---

**Q: What is `handle_parsing_errors=True` in AgentExecutor?**

Sometimes the LLM generates output that doesn't match the expected ReAct format (e.g., it writes "action:" in lowercase instead of "Action:"). Without `handle_parsing_errors=True`, this crashes the agent. With it, the AgentExecutor catches the parsing error and sends it back to the LLM as an observation, asking it to fix its formatting. This makes the agent much more robust.

---

**Q: What does `hub.pull("hwchase17/react")` do?**

It downloads a pre-built prompt template from LangChain Hub (a community repository of prompts). The `hwchase17/react` prompt is the standard ReAct prompt created by Harrison Chase (LangChain's creator). It instructs the LLM to follow the Thought/Action/Action Input/Observation/Final Answer format. Using the Hub prompt ensures we use the battle-tested, community-verified version rather than writing our own from scratch.

---

**Q: How does the session ID work and what is it used for?**

```python
session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# Example: "session_20260416_183045"
```

Each time `app.py` runs, it generates a unique session ID from the current timestamp. This ID is stored with every database entry, allowing queries to filter history by session:

```python
db_session.query(ResearchEntry).filter_by(session_id=session_id)
```

The `/history` command shows only the current session's Q&A pairs, not all historical data.

---

**Q: What is the `return_messages` parameter in ConversationBufferMemory?**

- `return_messages=True` — returns a list of `HumanMessage` and `AIMessage` objects
- `return_messages=False` — returns a plain text string

In `app.py`, `return_messages=False` is used because the plain text format integrates more cleanly into the ReAct agent's prompt. The agent prompt is a single string, so injecting formatted text (not Message objects) is simpler and more reliable.

In `memory.py`'s demo, `return_messages=True` is used so we can iterate over the messages and print their type (`human` or `ai`).

---

**Q: What happens when DuckDuckGo rate-limits the search?**

The `_web_search()` function wraps the DDGS call in a try/except block:

```python
try:
    results = list(DDGS().text(query, max_results=5))
    ...
except Exception as exc:
    return f"Search error: {exc}"
```

When an error occurs, the function returns a string starting with `"Search error:"`. The agent reads this as an observation, understands the search failed, and either tries a different query or falls back to Wikipedia or its own knowledge.

---

**Q: How does ConversationBufferMemory integrate with the ReAct agent?**

The agent itself does not natively support memory. Memory is integrated manually in `ResearchAssistant._build_context_input()`:

```python
def _build_context_input(self, question: str) -> str:
    history = self.memory.load_memory_variables({}).get("chat_history", "")
    if history:
        return (
            f"Previous conversation:\n{history}\n\n"
            f"Current question: {question}"
        )
    return question
```

The conversation history is prepended to the question before it's sent to the agent. The agent sees both the history and the new question as a single input string, which is how it maintains context awareness.

---

*End of Documentation*
