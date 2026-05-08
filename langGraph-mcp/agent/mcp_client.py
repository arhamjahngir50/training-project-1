"""
MCP HTTP client.
Discovers tools dynamically from the server and calls them over JSON-RPC.
No hardcoded tool definitions anywhere.
"""
import httpx
from typing import Any

MCP_SERVER_URL = "http://localhost:8000"


class MCPClient:
    def __init__(self, base_url: str = MCP_SERVER_URL):
        self.base_url = base_url
        self._req_id = 0
        self._http = httpx.Client(timeout=30)

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _send(self, method: str, params: dict = {}) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params,
        }
        response = self._http.post(f"{self.base_url}/rpc", json=payload)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")
        return data["result"]

    def list_tools(self) -> list[dict]:
        """Fetch all tool definitions from MCP server at runtime."""
        return self._send("tools/list")["tools"]

    def call_tool(self, name: str, arguments: dict) -> str:
        """Call any tool by name — works for whatever the server exposes."""
        result = self._send("tools/call", {"name": name, "arguments": arguments})
        return " ".join(
            block["text"]
            for block in result.get("content", [])
            if block["type"] == "text"
        )

    def close(self):
        self._http.close()