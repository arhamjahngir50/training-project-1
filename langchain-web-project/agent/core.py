"""
Agent core: builds and runs the LangChain ReAct agent with all tools.
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferWindowMemory

from .tools_system import SYSTEM_TOOLS
from .tools_extended import EXTENDED_TOOLS
from .rag import RAG_TOOLS

load_dotenv()

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are ARIA (Autonomous Reasoning & Interaction Agent), an advanced Linux desktop AI assistant.
You help users manage their system, files, processes, and documents using a rich set of tools.

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

Guidelines:
- Be concise and direct in your responses
- For destructive operations (delete, kill, clear), confirm before acting unless the user explicitly said to proceed
- When using RAG tools, always cite which document the information came from
- Format file sizes, dates, and system metrics in a human-friendly way
- If a command fails, explain why and suggest alternatives

Previous conversation:
{chat_history}

Question: {input}
{agent_scratchpad}"""


# ── Agent Factory ─────────────────────────────────────────────────────────────

ALL_TOOLS = SYSTEM_TOOLS + EXTENDED_TOOLS + RAG_TOOLS


def build_agent(temperature: float = 0.2, verbose: bool = False) -> AgentExecutor:
    """Build and return a fully configured AgentExecutor."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set in .env")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=temperature,
        convert_system_message_to_human=True,
    )

    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=10,
        return_messages=False,
    )

    agent = create_react_agent(llm=llm, tools=ALL_TOOLS, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        memory=memory,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=10,
        max_execution_time=120,
    )


def get_tool_manifest() -> list[dict]:
    """Return metadata about all available tools for the UI."""
    categories = {
        "System": SYSTEM_TOOLS,
        "Extended": EXTENDED_TOOLS,
        "RAG / Knowledge": RAG_TOOLS,
    }
    manifest = []
    for cat, tools in categories.items():
        for t in tools:
            manifest.append({
                "category": cat,
                "name": t.name,
                "description": t.description,
            })
    return manifest