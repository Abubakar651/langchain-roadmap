"""
agents.py – Phase 3: LangChain Advanced Concepts
=================================================
Demonstrates three agent types available in LangChain:

  1. ReAct Agent       – Reason + Act (the most capable, recommended)
  2. Zero-shot Agent   – Acts directly without examples
  3. Self-ask Agent    – Breaks questions into sub-questions

Focus: ReAct is used in the main app because it:
  • Shows its reasoning step-by-step (Thought → Action → Observation)
  • Handles multi-step problems naturally
  • Works reliably with modern chat models
"""

import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq

from tools import get_all_tools

load_dotenv()


# ─── Shared LLM (Groq) ───────────────────────────────────────────────────────
def _get_llm(temperature: float = 0.3) -> ChatGroq:
    """
    Create a Groq ChatLLM.
    Model: llama-3.3-70b-versatile  — excellent for reasoning/tool use tasks.
    """
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# AGENT TYPE 1: ReAct Agent
# ─────────────────────────────────────────────────────────────────────────────
"""
How ReAct Works (Reason + Act):
──────────────────────────────
The agent loops through this cycle until it has a Final Answer:

  Thought:      "I need to find information about X"
  Action:       web_search
  Action Input: "X latest news"
  Observation:  [tool result returned]
  Thought:      "Now I have enough info to answer"
  Final Answer: "..."

This makes the agent's reasoning transparent and debuggable.
"""


def create_react_research_agent(verbose: bool = True) -> AgentExecutor:
    """
    Build a ReAct agent equipped with all tools.
    This is the agent used by the main app.

    Args:
        verbose: If True, prints Thought/Action/Observation trace.

    Returns:
        AgentExecutor ready to `invoke({"input": "..."})`.
    """
    llm = _get_llm()
    tools = get_all_tools()

    # Pull the standard ReAct prompt from LangChain Hub
    # This prompt instructs the model to follow the Thought/Action format
    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,   # gracefully recover from LLM format errors
        max_iterations=8,             # prevent infinite loops
        max_execution_time=60,        # seconds timeout
    )


# ─────────────────────────────────────────────────────────────────────────────
# AGENT TYPE 2: Zero-shot Agent  (conceptual demo)
# ─────────────────────────────────────────────────────────────────────────────
"""
Zero-shot Agent:
────────────────
• Makes decisions based purely on the tool descriptions — no examples provided.
• The LLM reads each tool's description and decides which to call.
• "Zero-shot" = zero demonstrations/examples in the prompt.
• In practice, create_react_agent IS a zero-shot agent — it uses no few-shot
  examples, only the tool descriptions.

Key concept: The quality of the tool 'description' parameter is crucial.
A well-written description = the agent uses the tool correctly.
A vague description = the agent misuses or ignores the tool.
"""


def zero_shot_agent_concept():
    """
    Explains the zero-shot concept with a simple example.
    Note: In modern LangChain, all ReAct agents are zero-shot by default.
    """
    print("""
Zero-Shot Agent Decision Logic:
  Tools available: [web_search, wikipedia, calculator, current_datetime]

  User: "What is 15% of 340?"
  Agent reads tool descriptions...
  → Selects: calculator  (description mentions 'math expressions')
  → Action Input: "340 * 0.15"
  → Final Answer: "51.0"

  Zero-shot = no example was needed. The agent decided from the description alone.
""")


# ─────────────────────────────────────────────────────────────────────────────
# AGENT TYPE 3: Self-Ask Agent (conceptual demo)
# ─────────────────────────────────────────────────────────────────────────────
"""
Self-Ask Agent:
───────────────
• Designed for compositional questions that require multiple sub-answers.
• The agent explicitly asks itself follow-up questions:

  Question: "Who was the president of the US when the Eiffel Tower was built?"

  Are follow up questions needed? Yes.
  Follow up: When was the Eiffel Tower built?
  Intermediate answer: 1889.
  Follow up: Who was the US president in 1889?
  Intermediate answer: Grover Cleveland.
  Final answer: Grover Cleveland.

• Requires a tool named exactly "Intermediate Answer".
• Best for: multi-hop factual reasoning.
"""


def self_ask_agent_concept():
    """Print the Self-Ask reasoning trace pattern."""
    print("""
Self-Ask Agent Reasoning Pattern:
  Question: "Who invented the technology used in the first iPhone?"

  Are follow up questions needed? Yes.
  Follow up: What key technology was in the first iPhone?
  Intermediate answer: Capacitive touchscreen, developed by Apple.
  Follow up: Who developed capacitive touchscreen technology for Apple?
  Intermediate answer: The work was pioneered by Wayne Westerman and John Elias.
  Final answer: Wayne Westerman and John Elias developed the touchscreen tech in the iPhone.
""")


# ─── Demo: Run the ReAct Agent ────────────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 LangChain Agent Types – Phase 3 Demo\n")

    # Explain agent types conceptually
    zero_shot_agent_concept()
    self_ask_agent_concept()

    # Run the real ReAct agent
    agent = create_react_research_agent(verbose=True)
    query = "What is the square root of 1764, and what is today's date?"
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print("=" * 60)
    result = agent.invoke({"input": query})
    print("\nFinal Answer:", result["output"])
