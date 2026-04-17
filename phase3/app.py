"""
app.py – Phase 3: AI Research Assistant (Main Application)
===========================================================
A production-like CLI research assistant that demonstrates:

  • ReAct Agent with 4 tools (web search, Wikipedia, calculator, datetime)
  • ConversationBufferMemory for context-aware multi-turn conversations
  • SQLite-backed Q&A history via SQLAlchemy
  • Graceful error handling and session management

Usage:
  python app.py

Commands inside the app:
  /history   – View stored Q&A history from this session
  /memory    – Show current conversation memory (last N turns)
  /clear     – Clear conversation memory (start fresh)
  /quit      – Exit the assistant
"""

import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session

load_dotenv()

# ─── Validate environment ─────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    print(
        "\n❌  GROQ_API_KEY is not set!\n"
        "    1. Copy .env.example to .env\n"
        "    2. Add your key from https://console.groq.com\n"
        "    3. Run: python app.py\n"
    )
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE SETUP (SQLite via SQLAlchemy)
# Stores the Q&A history so you can review past research sessions.
# ─────────────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


class ResearchEntry(Base):
    """ORM model for a single Q&A exchange."""

    __tablename__ = "research_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    tools_used = Column(String(200), default="")          # which tools the agent called
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


DB_PATH = "research_history.db"
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
Base.metadata.create_all(engine)


def save_to_db(session_id: str, question: str, answer: str, tools_used: str = "") -> None:
    """Persist a Q&A pair to SQLite."""
    with Session(engine) as db_session:
        entry = ResearchEntry(
            session_id=session_id,
            question=question,
            answer=answer,
            tools_used=tools_used,
        )
        db_session.add(entry)
        db_session.commit()


def get_history(session_id: str, limit: int = 10) -> list[ResearchEntry]:
    """Retrieve the most recent entries for a session."""
    with Session(engine) as db_session:
        return (
            db_session.query(ResearchEntry)
            .filter_by(session_id=session_id)
            .order_by(ResearchEntry.timestamp.desc())
            .limit(limit)
            .all()
        )


# ─────────────────────────────────────────────────────────────────────────────
# AGENT SETUP
# ─────────────────────────────────────────────────────────────────────────────
def build_agent():
    """Build the ReAct research agent with all tools."""
    # Import here to avoid circular imports
    from langchain import hub
    from langchain.agents import AgentExecutor, create_react_agent
    from tools import get_all_tools

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        groq_api_key=GROQ_API_KEY,
    )
    tools = get_all_tools()
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,              # Set True to see full Thought/Action trace
        handle_parsing_errors=True,
        max_iterations=10,
        max_execution_time=60,
    ), tools


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY-AWARE CONVERSATION WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
class ResearchAssistant:
    """
    Wraps the ReAct agent with ConversationBufferMemory.

    How it works:
    ─────────────
    1. User sends a question.
    2. We inject the conversation history into the agent's system context.
    3. The agent reasons, calls tools, and returns an answer.
    4. The answer is saved to both memory (for context) and SQLite (for history).
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False,   # plain text format for agent prompts
            human_prefix="User",
            ai_prefix="Assistant",
        )

        print("  🔧 Loading tools and building agent...")
        self.agent_executor, self.tools = build_agent()
        self.tool_names = [t.name for t in self.tools]

    def _build_context_input(self, question: str) -> str:
        """Inject memory into the question so the agent has context."""
        history = self.memory.load_memory_variables({}).get("chat_history", "")
        if history:
            return (
                f"Previous conversation:\n{history}\n\n"
                f"Current question: {question}"
            )
        return question

    def ask(self, question: str) -> str:
        """
        Process a user question through the ReAct agent.
        Returns the agent's final answer.
        """
        context_input = self._build_context_input(question)

        result = self.agent_executor.invoke({"input": context_input})
        answer = result.get("output", "I could not generate an answer.")

        # Save to in-memory conversation buffer
        self.memory.save_context(
            {"input": question},
            {"output": answer},
        )

        # Persist to SQLite
        save_to_db(
            session_id=self.session_id,
            question=question,
            answer=answer,
        )

        return answer

    def show_memory(self):
        """Display the current in-memory conversation history."""
        history = self.memory.load_memory_variables({}).get("chat_history", "")
        if not history:
            print("  (No conversation history yet)")
        else:
            print("\n  📝 Conversation Memory:")
            print("  " + "─" * 50)
            print(history)

    def clear_memory(self):
        """Clear the in-memory conversation buffer (start fresh)."""
        self.memory.clear()
        print("  ✅ Memory cleared. Starting a fresh conversation.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════════╗
║        🔬  AI Research Assistant  |  Phase 3 Project         ║
║   Powered by: Groq (LLaMA 3 70B) + LangChain ReAct Agent    ║
╠══════════════════════════════════════════════════════════════╣
║  Tools:  🌐 Web Search  │  📖 Wikipedia  │  🧮 Calculator    ║
║          🕐 Datetime                                          ║
║  Memory: ConversationBufferMemory  │  SQLite History          ║
╠══════════════════════════════════════════════════════════════╣
║  Commands:  /history  /memory  /clear  /verbose  /quit       ║
╚══════════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Available commands:
  /history  – Show last 10 Q&A pairs stored in SQLite
  /memory   – Show current in-memory conversation context
  /clear    – Clear conversation memory (keeps SQLite history)
  /verbose  – Toggle agent reasoning trace on/off
  /quit     – Exit the assistant
  /help     – Show this help message
"""


def print_history(assistant: ResearchAssistant):
    entries = get_history(assistant.session_id)
    if not entries:
        print("  (No history in this session yet)")
        return
    print(f"\n  📚 Last {len(entries)} Q&A pairs (most recent first):")
    print("  " + "─" * 56)
    for i, e in enumerate(entries, 1):
        ts = e.timestamp.strftime("%H:%M:%S")
        print(f"  [{i}] [{ts}] Q: {e.question[:70]}")
        print(f"       A: {e.answer[:100]}{'...' if len(e.answer) > 100 else ''}")
    print()


def main():
    print(BANNER)

    # Generate a unique session ID for this run
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"  Session ID: {session_id}")
    print(f"  History DB: {DB_PATH}\n")

    print("  Initialising Research Assistant...")
    assistant = ResearchAssistant(session_id=session_id)
    print("  ✅ Ready! Ask me anything.\n")

    verbose_mode = False  # toggleable

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye! 👋\n")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("/quit", "/exit", "/q"):
            print("\n  Goodbye! 👋\n")
            break
        elif cmd == "/history":
            print_history(assistant)
        elif cmd == "/memory":
            assistant.show_memory()
        elif cmd == "/clear":
            assistant.clear_memory()
        elif cmd == "/verbose":
            verbose_mode = not verbose_mode
            assistant.agent_executor.verbose = verbose_mode
            state = "ON 🔍" if verbose_mode else "OFF"
            print(f"  Verbose mode: {state}")
        elif cmd in ("/help", "/?"):
            print(HELP_TEXT)
        else:
            # ── Ask the agent ──────────────────────────────────────────────
            print("\n  🤔 Researching...\n")
            try:
                answer = assistant.ask(user_input)
                print(f"\nAssistant: {answer}\n")
                print("  " + "─" * 56)
            except Exception as exc:
                print(f"\n  ❌ Error: {exc}")
                print("  Please try again or type /quit to exit.\n")


if __name__ == "__main__":
    main()
