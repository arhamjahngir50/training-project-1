"""
LangGraph ReAct agent.
Tools are discovered dynamically from the MCP server — nothing is hardcoded.
"""
from dotenv import load_dotenv
load_dotenv()

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from agent.mcp_client import MCPClient

# --- Connect to MCP server and discover tools once at startup ---
client = MCPClient()
mcp_tools = client.list_tools()

print(f"✅ Discovered {len(mcp_tools)} tools from MCP server:")
for t in mcp_tools:
    print(f"   • {t['name']}: {t['description']}")


def get_gemini_tools(mcp_tools: list[dict]) -> list[dict]:
    """Convert MCP tool schema → Gemini function calling format."""
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["inputSchema"],
        }
        for t in mcp_tools
    ]


gemini_tools = get_gemini_tools(mcp_tools)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_with_tools = llm.bind_tools(gemini_tools)


# --- Graph nodes ---

def agent_node(state: MessagesState):
    """Gemini decides what to do — reply or call a tool."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def tool_node(state: MessagesState):
    """
    Execute tool calls requested by Gemini.
    Calls MCP server directly — no @tool wrappers, no hardcoded logic.
    Works automatically for any tool the MCP server exposes.
    """
    last_message: AIMessage = state["messages"][-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]

        print(f"🔧 Calling MCP tool: {name} with args: {args}")

        raw_result = client.call_tool(name, args)
        result_data = json.loads(raw_result)

        tool_messages.append(
            ToolMessage(
                content=json.dumps(result_data),
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": tool_messages}


def should_continue(state: MessagesState) -> str:
    """If Gemini requested tool calls → tools node, else → end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# --- Build the graph ---
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

agent = graph.compile()


def run_agent(query: str) -> str:
    result = agent.invoke({"messages": [("user", query)]})
    return result["messages"][-1].content