"""
memory.py – Phase 3: LangChain Advanced Concepts
=================================================
Demonstrates three memory types used in LangChain:

  1. ConversationBufferMemory   – stores the full conversation verbatim
  2. ConversationEntityMemory   – tracks named entities (people, places, things)
  3. VectorStoreRetrieverMemory – semantic (embedding-based) long-term memory

Each section includes a runnable demo showing how the memory works.
"""

import os

from dotenv import load_dotenv
from langchain.memory import (
    ConversationBufferMemory,
    ConversationEntityMemory,
    VectorStoreRetrieverMemory,
)
from langchain_groq import ChatGroq

load_dotenv()


# ─── Shared LLM ──────────────────────────────────────────────────────────────
def _get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. ConversationBufferMemory
# ─────────────────────────────────────────────────────────────────────────────
class BufferMemoryDemo:
    """
    ConversationBufferMemory stores EVERY message in the conversation.

    ✅ Pros: Simple, accurate, full context
    ❌ Cons: Gets very long for extended conversations
    Best for: Short chatbot sessions, simple Q&A bots
    """

    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,   # returns HumanMessage / AIMessage objects
        )

    def save(self, human: str, ai: str):
        """Manually save a turn to memory."""
        self.memory.save_context({"input": human}, {"output": ai})

    def load(self) -> dict:
        """Load the full conversation history."""
        return self.memory.load_memory_variables({})

    def clear(self):
        """Reset memory for a new session."""
        self.memory.clear()

    def demo(self):
        print("\n" + "=" * 60)
        print("  DEMO: ConversationBufferMemory")
        print("=" * 60)

        turns = [
            ("My name is Bakar.", "Nice to meet you, Bakar!"),
            ("I am learning LangChain.", "That's great! LangChain is powerful."),
            ("What did I just say?", "You said you are learning LangChain."),
        ]

        for human, ai in turns:
            self.save(human, ai)

        history = self.load()
        print(f"\nStored {len(history['chat_history'])} messages:")
        for msg in history["chat_history"]:
            role = "Human" if msg.type == "human" else "AI"
            print(f"  [{role}] {msg.content}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ConversationEntityMemory
# ─────────────────────────────────────────────────────────────────────────────
class EntityMemoryDemo:
    """
    ConversationEntityMemory extracts and tracks named entities from the
    conversation (people, places, organisations, topics, etc.).

    ✅ Pros: Lightweight, structured, great for relationship tracking
    ❌ Cons: Requires an LLM call to extract entities
    Best for: Personal assistants, CRM bots, contextual chatbots
    """

    def __init__(self):
        from langchain_core.caches import BaseCache
        from langchain_core.callbacks.base import Callbacks
        ConversationEntityMemory.model_rebuild(
            _types_namespace={"BaseCache": BaseCache, "Callbacks": Callbacks}
        )
        self.memory = ConversationEntityMemory(
            llm=_get_llm(),
            return_messages=True,
        )

    def save(self, human: str, ai: str):
        self.memory.save_context({"input": human}, {"output": ai})

    def load(self) -> dict:
        return self.memory.load_memory_variables({"input": ""})

    def demo(self):
        print("\n" + "=" * 60)
        print("  DEMO: ConversationEntityMemory")
        print("=" * 60)

        turns = [
            ("Elon Musk founded SpaceX in 2002.", "Yes, SpaceX revolutionised rockets."),
            ("SpaceX is based in Hawthorne, California.", "Correct!"),
        ]

        for human, ai in turns:
            self.save(human, ai)
            print(f"  Saved: '{human}'")

        data = self.load()
        print("\nExtracted entity store:")
        for entity, info in data.get("entities", {}).items():
            print(f"  {entity}: {info}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. VectorStoreRetrieverMemory  (Semantic / Long-term Memory)
# ─────────────────────────────────────────────────────────────────────────────
class VectorMemoryDemo:
    """
    VectorStoreRetrieverMemory embeds past conversation turns as vectors and
    retrieves the MOST SEMANTICALLY SIMILAR past turns at query time.

    ✅ Pros: Handles long conversation history efficiently; semantic retrieval
    ❌ Cons: More setup required; not perfectly chronological
    Best for: Long-running assistants, document Q&A, knowledge bases
    """

    def __init__(self):
        # Use in-memory Chroma (no persist directory needed for demo)
        from langchain_chroma import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = Chroma(
            embedding_function=embeddings,
            collection_name="memory_demo",
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        self.memory = VectorStoreRetrieverMemory(retriever=retriever)

    def save(self, human: str, ai: str):
        self.memory.save_context({"input": human}, {"output": ai})

    def retrieve(self, query: str) -> dict:
        return self.memory.load_memory_variables({"prompt": query})

    def demo(self):
        print("\n" + "=" * 60)
        print("  DEMO: VectorStoreRetrieverMemory")
        print("=" * 60)

        facts = [
            ("What is LangChain?", "LangChain is a framework for building LLM apps."),
            ("What is a vector store?", "A vector store indexes embeddings for semantic search."),
            ("Tell me about agents.", "Agents use LLMs to decide which tools to call."),
            ("What is Groq?", "Groq is an ultra-fast LLM inference provider."),
        ]

        for human, ai in facts:
            self.save(human, ai)

        query = "Tell me about LLM frameworks"
        print(f"\nQuery: '{query}'")
        result = self.retrieve(query)
        print("Semantically similar past conversation:")
        print(result.get("history", "No results"))


# ─── Run all demos ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🧠 LangChain Memory Types – Phase 3 Demo")

    # 1. Buffer Memory (no LLM needed)
    BufferMemoryDemo().demo()

    # 2. Entity Memory (requires Groq API key)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and groq_key != "your_groq_api_key_here":
        EntityMemoryDemo().demo()
    else:
        print("\n[EntityMemory] Skipped — set GROQ_API_KEY in .env to run this demo.")

    # 3. Vector Memory
    VectorMemoryDemo().demo()
