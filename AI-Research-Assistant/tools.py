"""
tools.py – Phase 3: LangChain Advanced Concepts
================================================
Demonstrates:
  • DuckDuckGo web search tool (no API key needed)
  • Wikipedia lookup tool
  • Custom calculator tool
  • Current datetime tool

These tools are handed to the ReAct agent so it can decide
WHEN and HOW to use them based on the user's question.
"""

import math
from datetime import datetime

from ddgs import DDGS
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()


# ─── 1. DuckDuckGo Web Search ────────────────────────────────────────────────
# Free, no API key required. Uses the ddgs package (renamed from duckduckgo-search).
def _web_search(query: str) -> str:
    """Perform a DuckDuckGo search and return formatted results."""
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


def get_search_tool() -> Tool:
    """
    LangChain Tool wrapping DuckDuckGo search via the ddgs package.
    The agent uses this when it needs up-to-date information from the web.
    """
    return Tool(
        name="web_search",
        func=_web_search,
        description=(
            "Searches the web using DuckDuckGo and returns relevant results. "
            "Use this when you need current/recent information or facts that "
            "are not in your training data. Input should be a search query string."
        ),
    )


# ─── 2. Wikipedia Tool ───────────────────────────────────────────────────────
# Great for factual, encyclopedic knowledge. Returns article summaries.
def get_wikipedia_tool() -> Tool:
    """
    LangChain Tool wrapping Wikipedia search.
    The agent uses this for in-depth factual lookups (history, science, etc.)
    """
    wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
    wiki = WikipediaQueryRun(api_wrapper=wrapper)
    return Tool(
        name="wikipedia",
        func=wiki.run,
        description=(
            "Looks up a topic on Wikipedia and returns a summary. "
            "Use this for factual/encyclopedic questions about people, places, "
            "concepts, history, or science. Input should be a topic name."
        ),
    )


# ─── 3. Calculator Tool ──────────────────────────────────────────────────────
# A simple but powerful example of a custom Python tool.
def _calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Supports: +, -, *, /, **, sqrt(), log(), sin(), cos(), pi, e
    """
    try:
        # Provide a safe math namespace
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
        result = eval(expression, {"__builtins__": {}}, safe_namespace)  # noqa: S307
        return f"Result: {result}"
    except Exception as exc:
        return f"Calculator error: {exc}"


def get_calculator_tool() -> Tool:
    """
    Custom calculator tool. The agent uses this for math questions.
    """
    return Tool(
        name="calculator",
        func=_calculate,
        description=(
            "Evaluates a mathematical expression and returns the result. "
            "Supports: +, -, *, /, **, sqrt(), log(), sin(), cos(), pi, e. "
            "Input should be a valid math expression string, e.g. 'sqrt(144) + 5'."
        ),
    )


# ─── 4. Current Datetime Tool ────────────────────────────────────────────────
def _get_datetime(_: str = "") -> str:
    """Return the current date and time."""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y %I:%M %p')}"


def get_datetime_tool() -> Tool:
    """
    Tool that tells the agent the current date/time.
    """
    return Tool(
        name="current_datetime",
        func=_get_datetime,
        description=(
            "Returns the current date and time. "
            "Use this when the user asks about today's date, time, or "
            "when you need to know the current date for context."
        ),
    )


# ─── Bundle all tools ────────────────────────────────────────────────────────
def get_all_tools() -> list[Tool]:
    """Return the complete list of tools available to the agent."""
    return [
        get_search_tool(),
        get_wikipedia_tool(),
        get_calculator_tool(),
        get_datetime_tool(),
    ]


# ─── Direct test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Testing Tools ===\n")

    calc = get_calculator_tool()
    print("Calculator:", calc.run("sqrt(256) + pi"))

    dt = get_datetime_tool()
    print("DateTime:  ", dt.run(""))

    wiki = get_wikipedia_tool()
    print("\nWikipedia (LangChain):\n", wiki.run("LangChain AI framework")[:300], "...")
