"""
Lightweight agent runner that connects to MCP servers and dispatches
tool calls from an OpenAI-compatible LLM (vLLM, OpenAI, etc.).

The runner:
1. Connects to one or more MCP servers
2. Discovers available tools and converts them to OpenAI tool schemas
3. Sends the conversation + tools to the LLM
4. Dispatches any tool_calls back to the appropriate MCP server
5. Loops until the LLM produces a final text response or max iterations

Usage:
    runner = AgentRunner(
        llm_base_url="http://localhost:8001/v1",
        model="wizard-vicuna-7b-awq",
    )
    runner.add_mcp_server("memory", memory_mcp_server)
    runner.add_mcp_server("postgres", postgres_mcp_server)
    result = await runner.run("List all tables in the database")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastmcp import Client

logger = logging.getLogger("agent-runner")


@dataclass
class ToolRef:
    """Maps a tool name to its MCP server client."""
    server_name: str
    client: Client
    schema: dict[str, Any]


@dataclass
class AgentResult:
    """Result of an agent run."""
    final_response: str
    messages: list[dict[str, Any]]
    tool_calls_made: int
    iterations: int


class AgentRunner:
    """Connects MCP tool servers to an OpenAI-compatible LLM."""

    def __init__(
        self,
        llm_base_url: str = "http://localhost:8001/v1",
        model: str = "wizard-vicuna-7b-awq",
        max_iterations: int = 10,
        temperature: float = 0.1,
        system_prompt: str | None = None,
    ):
        self.llm_base_url = llm_base_url.rstrip("/")
        self.model = model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.system_prompt = system_prompt or (
            "You are Wintermute, an AI agent with access to tools. "
            "Use the available tools to accomplish the user's request. "
            "Think step by step. When you have a final answer, respond directly."
        )
        self._mcp_servers: dict[str, Any] = {}
        self._clients: dict[str, Client] = {}
        self._tools: dict[str, ToolRef] = {}
        self._initialized = False

    def add_mcp_server(self, name: str, server) -> None:
        """Register an MCP server (FastMCP instance or URL string)."""
        self._mcp_servers[name] = server
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Connect to all MCP servers and discover tools."""
        if self._initialized:
            return

        try:
            for name, server in self._mcp_servers.items():
                client = Client(server)
                await client.__aenter__()
                self._clients[name] = client

                tools = await client.list_tools()
                for tool in tools:
                    openai_schema = _mcp_tool_to_openai(tool)
                    self._tools[tool.name] = ToolRef(
                        server_name=name,
                        client=client,
                        schema=openai_schema,
                    )
                    logger.debug("Registered tool %s from server %s", tool.name, name)
        except Exception:
            await self.cleanup()
            raise

        logger.info(
            "Initialized with %d tools from %d servers",
            len(self._tools), len(self._mcp_servers),
        )
        self._initialized = True

    async def cleanup(self) -> None:
        """Disconnect from all MCP servers."""
        for client in self._clients.values():
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass
        self._clients.clear()
        self._tools.clear()
        self._initialized = False

    async def __aenter__(self):
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        return False

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Return all registered tools in OpenAI function-calling format."""
        return [ref.schema for ref in self._tools.values()]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch a tool call to the appropriate MCP server."""
        if name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {name}"})

        ref = self._tools[name]
        try:
            result = await ref.client.call_tool(name, arguments)
            if result.content:
                return result.content[0].text
            return json.dumps({"result": "ok"})
        except Exception as e:
            logger.error("Tool %s failed: %s", name, e)
            return json.dumps({"error": str(e)})

    async def _chat_completion(
        self, messages: list[dict], tools: list[dict]
    ) -> dict:
        """Call the LLM's chat/completions endpoint."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=120) as http:
            resp = await http.post(
                f"{self.llm_base_url}/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    async def run(
        self,
        user_message: str,
        context_messages: list[dict[str, Any]] | None = None,
    ) -> AgentResult:
        """Run the agent loop until a final response or max iterations.

        Args:
            user_message: The user's request.
            context_messages: Optional prior conversation messages.

        Returns:
            AgentResult with final response and history.
        """
        await self._ensure_initialized()

        messages = [{"role": "system", "content": self.system_prompt}]
        if context_messages:
            messages.extend(context_messages)
        messages.append({"role": "user", "content": user_message})

        tools = self.get_openai_tools()
        total_tool_calls = 0

        for iteration in range(1, self.max_iterations + 1):
            logger.info("Iteration %d/%d", iteration, self.max_iterations)

            completion = await self._chat_completion(messages, tools)
            choice = completion["choices"][0]
            msg = choice["message"]

            messages.append(msg)

            if choice.get("finish_reason") == "tool_calls" or msg.get("tool_calls"):
                tool_calls = msg["tool_calls"]
                for tc in tool_calls:
                    fn = tc["function"]
                    tool_name = fn["name"]
                    try:
                        tool_args = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        tool_args = {}

                    logger.info("Calling tool: %s(%s)", tool_name, json.dumps(tool_args)[:200])
                    result_text = await self.call_tool(tool_name, tool_args)
                    total_tool_calls += 1

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_text,
                    })
            else:
                final = msg.get("content", "")
                return AgentResult(
                    final_response=final,
                    messages=messages,
                    tool_calls_made=total_tool_calls,
                    iterations=iteration,
                )

        last_msg = messages[-1]
        return AgentResult(
            final_response=last_msg.get("content", "[max iterations reached]"),
            messages=messages,
            tool_calls_made=total_tool_calls,
            iterations=self.max_iterations,
        )


def _mcp_tool_to_openai(tool) -> dict[str, Any]:
    """Convert an MCP tool descriptor to OpenAI function-calling format."""
    params = {}
    if tool.inputSchema:
        params = dict(tool.inputSchema)
        params.pop("title", None)

    if "type" not in params:
        params["type"] = "object"
    if "properties" not in params:
        params["properties"] = {}

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": params,
        },
    }
