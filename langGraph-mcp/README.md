# LangGraph MCP Agent

A dynamic ReAct agent built with LangGraph that discovers and uses tools from an MCP (Model Context Protocol) server at runtime. No tools are hardcoded — everything is fetched dynamically from the MCP server.

## Overview

This project demonstrates a **fully dynamic tool discovery** approach:
- The agent connects to an MCP server on startup
- It fetches all available tools automatically
- Gemini decides which tools to call and when
- Tool execution happens via MCP JSON-RPC calls
- Works with any MCP-compatible server without code changes

## Features

- **Dynamic Tool Discovery**: No hardcoded tool definitions
- **LangGraph ReAct Pattern**: Reasoning → Action → Observation loop
- **MCP Integration**: Uses Model Context Protocol for tool communication
- **Gemini LLM**: Google Gemini 2.5 Flash for reasoning and tool calling
- **Sample Queries**: Math, weather, summarization examples

## Project Structure

```
langGraph-mcp/
├── main.py              # Entry point with sample queries
├── requirements.txt     # Python dependencies
├── agent/
│   ├── __init__.py
│   ├── graph.py         # LangGraph agent setup
│   └── mcp_client.py    # MCP HTTP client
└── README.md           # This file
```

## Prerequisites

1. **MCP Server**: You need an MCP server running on `http://localhost:8000`
   - The server must expose tools via the MCP protocol
   - Example: A server with math, weather, and text processing tools

2. **Google Gemini API Key**: Get one from [aistudio.google.com](https://aistudio.google.com)

## Setup

### 1. Install Dependencies

```bash
cd langGraph-mcp
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Start MCP Server

Make sure your MCP server is running on `http://localhost:8000`. For example:

```bash
# If you have an MCP server implementation
cd your-mcp-server
python server.py  # or however you start it
```

The MCP server should expose tools like:
- Math calculation tools
- Weather lookup tools
- Text processing tools

## Running the Agent

### Run Sample Queries

```bash
python main.py
```

This will run the agent on three sample queries:
1. Math calculation: `(123 * 456) + 789`
2. Weather lookup: Lahore and Islamabad
3. Text summarization

### Custom Query

To run a single custom query, modify `main.py`:

```python
from agent.graph import run_agent

result = run_agent("Your custom question here?")
print(result)
```

## How It Works

### 1. Tool Discovery
On startup, the agent connects to the MCP server and fetches all available tools:

```python
client = MCPClient()
mcp_tools = client.list_tools()
```

### 2. Gemini Tool Binding
Tools are converted to Gemini's function calling format:

```python
gemini_tools = get_gemini_tools(mcp_tools)
llm_with_tools = llm.bind_tools(gemini_tools)
```

### 3. ReAct Loop
The LangGraph handles the reasoning loop:
- **Agent Node**: Gemini decides what to do
- **Tool Node**: Executes MCP tool calls
- **Conditional Edges**: Continues if tools were called

### 4. Dynamic Execution
When Gemini requests a tool call, it's executed via MCP:

```python
raw_result = client.call_tool(name, args)
```

## Sample Output

```
🟡 Query: What is (123 * 456) + 789?
🔧 Calling MCP tool: calculate with args: {'expression': '(123 * 456) + 789'}
🟢 Answer: The result of (123 * 456) + 789 is 57027.

🟡 Query: What's the weather like in Lahore and Islamabad?
🔧 Calling MCP tool: get_weather with args: {'city': 'Lahore'}
🔧 Calling MCP tool: get_weather with args: {'city': 'Islamabad'}
🟢 Answer: In Lahore, it's currently 28°C with clear skies. In Islamabad, it's 26°C with partly cloudy conditions.

🟡 Query: Summarize this: Artificial intelligence is transforming...
🔧 Calling MCP tool: summarize_text with args: {'text': 'Artificial intelligence...'}
🟢 Answer: AI is revolutionizing industries from healthcare to finance through automation, accelerating the pace of change in work, life, and human interaction.
```

## MCP Server Requirements

Your MCP server must:
- Run on `http://localhost:8000`
- Implement MCP JSON-RPC endpoints:
  - `tools/list` - Return tool definitions
  - `tools/call` - Execute tool calls
- Return results in MCP content format

## Dependencies

- `langgraph>=0.2.0` - Graph-based agent framework
- `langchain>=0.3.0` - LLM integration
- `langchain-google-genai>=2.0.0` - Gemini LLM
- `python-dotenv` - Environment variables
- `fastapi` - (for MCP server, if needed)
- `uvicorn` - (for MCP server, if needed)
- `httpx` - HTTP client for MCP calls
- `sse-starlette` - (for MCP server streaming)

## Troubleshooting

### MCP Server Connection
- Ensure MCP server is running on `http://localhost:8000`
- Check server logs for connection errors
- Verify MCP endpoints are implemented correctly

### Tool Discovery Issues
- Run the agent and check the startup messages
- It should print: `✅ Discovered X tools from MCP server:`
- If no tools are found, check your MCP server implementation

### Gemini API Errors
- Verify `GOOGLE_API_KEY` is set correctly
- Check API quota and billing status
- Ensure internet connection for API calls

### Common Errors
- `httpx.ConnectError`: MCP server not running
- `RuntimeError: MCP error`: Check MCP server logs
- `Tool not found`: Tool name mismatch between discovery and calling

## Extending

### Add More Sample Queries
Edit `QUERIES` in `main.py`:

```python
QUERIES = [
    "Your new query here",
    # ...
]
```

### Change MCP Server URL
Modify `MCP_SERVER_URL` in `mcp_client.py`:

```python
MCP_SERVER_URL = "http://your-server:port"
```

### Use Different LLM
Replace Gemini in `graph.py`:

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")
```

## License

MIT