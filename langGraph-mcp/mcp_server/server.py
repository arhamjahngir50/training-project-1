"""
MCP Server — HTTP + SSE transport using FastAPI.
Endpoints:
  POST /rpc    → JSON-RPC 2.0 (tools/list, tools/call)
  GET  /sse    → SSE stream (keep-alive + server push)
  GET  /health → health check
"""
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
import uvicorn

from mcp_server.tools import calculate, get_weather, summarize_text

app = FastAPI(title="MCP Server")

TOOLS = {
    "calculator": {
        "description": "Evaluate a mathematical expression",
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate e.g. '2 + 2 * 10'"
                }
            },
            "required": ["expression"],
        },
    },
    "weather": {
        "description": "Get current weather for a city",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name e.g. 'Lahore'"
                }
            },
            "required": ["city"],
        },
    },
    "summarizer": {
        "description": "Summarize a block of text into fewer words",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to summarize"
                },
                "max_words": {
                    "type": "integer",
                    "description": "Maximum words in the summary",
                    "default": 30
                },
            },
            "required": ["text"],
        },
    },
}


def handle_rpc(request: dict) -> dict:
    method = request.get("method")
    req_id = request.get("id")
    params = request.get("params", {})

    # tools/list
    if method == "tools/list":
        tools_list = [{"name": name, **meta} for name, meta in TOOLS.items()]
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools_list}}

    # tools/call
    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name == "calculator":
            result = calculate(arguments.get("expression", ""))
        elif tool_name == "weather":
            result = get_weather(arguments.get("city", ""))
        elif tool_name == "summarizer":
            result = summarize_text(
                arguments.get("text", ""),
                arguments.get("max_words", 30),
            )
        else:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": json.dumps(result)}]
            },
        }

    return {
        "jsonrpc": "2.0", "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


@app.post("/rpc")
async def rpc_endpoint(request: Request):
    """Main JSON-RPC 2.0 endpoint."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"jsonrpc": "2.0", "id": None,
             "error": {"code": -32700, "message": "Parse error"}},
            status_code=400,
        )
    response = handle_rpc(body)
    return JSONResponse(response)


@app.get("/sse")
async def sse_endpoint(request: Request):
    """SSE stream — pushes keep-alive pings and can push tool events."""
    async def event_generator():
        yield {"event": "connected", "data": json.dumps({"status": "MCP server ready"})}
        while True:
            if await request.is_disconnected():
                break
            await asyncio.sleep(15)
            yield {"event": "ping", "data": json.dumps({"status": "alive"})}

    return EventSourceResponse(event_generator())


@app.get("/health")
async def health():
    return {"status": "ok", "transport": "http+sse"}


if __name__ == "__main__":
    uvicorn.run("mcp_server.server:app", host="0.0.0.0", port=8000, reload=False)