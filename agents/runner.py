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
import os
from dataclasses import dataclass
from typing import Any

import httpx
from fastmcp import Client

logger = logging.getLogger("agent-runner")

# Configurable LLM HTTP timeout (seconds). Default 120s preserves prior behavior.
_AGENT_LLM_TIMEOUT = float(os.getenv("AGENT_LLM_TIMEOUT", "120"))


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

    @staticmethod
    def _load_defaults() -> dict[str, Any]:
        try:
            from shared.config_loader import load_agents_config, load_vllm_config

            agents_cfg = load_agents_config()
            try:
                _, model = load_vllm_config()
            except Exception:
                model = agents_cfg.get("default_model", "wizard-vicuna-7b-awq")
            port = agents_cfg.get("default_llm_port", 8001)
            return {
                "llm_base_url": os.getenv("AGENT_LLM_URL", f"http://localhost:{port}/v1"),
                "model": model,
                "max_iterations": agents_cfg.get("max_iterations", 10),
                "temperature": agents_cfg.get("temperature", 0.1),
            }
        except Exception:
            return {
                "llm_base_url": "http://localhost:8001/v1",
                "model": "wizard-vicuna-7b-awq",
                "max_iterations": 10,
                "temperature": 0.1,
            }

    def __init__(
        self,
        llm_base_url: str | None = None,
        model: str | None = None,
        max_iterations: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ):
        defaults = self._load_defaults()
        self.llm_base_url = (llm_base_url or defaults["llm_base_url"]).rstrip("/")
        self.model = model or defaults["model"]
        self.max_iterations = (
            max_iterations if max_iterations is not None else defaults["max_iterations"]
        )
        self.temperature = temperature if temperature is not None else defaults["temperature"]
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
                    if tool.name in self._tools:
                        existing = self._tools[tool.name].server_name
                        logger.warning(
                            "Tool name collision: '%s' from server '%s' overrides '%s'",
                            tool.name,
                            name,
                            existing,
                        )
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
            len(self._tools),
            len(self._mcp_servers),
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
                return "\n".join(part.text for part in result.content if hasattr(part, "text"))
            return json.dumps({"result": "ok"})
        except Exception as e:
            logger.error("Tool %s failed: %s", name, e)
            return json.dumps({"error": str(e)})

    async def _chat_completion(self, messages: list[dict], tools: list[dict]) -> dict:
        """Call the LLM's chat/completions endpoint."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=_AGENT_LLM_TIMEOUT) as http:
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
            choices = completion.get("choices") or []
            if not choices:
                logger.error("LLM returned empty choices: %s", completion)
                break
            choice = choices[0]
            msg = choice.get("message") or {}

            messages.append(msg)

            if choice.get("finish_reason") == "tool_calls" or msg.get("tool_calls"):
                tool_calls = msg.get("tool_calls") or []
                for tc in tool_calls:
                    fn = tc["function"]
                    tool_name = fn["name"]
                    try:
                        tool_args = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        logger.warning(
                            "Malformed tool-call JSON for %s: %r — skipping call",
                            tool_name,
                            fn["arguments"][:200],
                        )
                        continue

                    logger.info("Calling tool: %s(%s)", tool_name, json.dumps(tool_args)[:200])
                    result_text = await self.call_tool(tool_name, tool_args)
                    total_tool_calls += 1

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result_text,
                        }
                    )
            else:
                final = msg.get("content", "")
                return AgentResult(
                    final_response=final,
                    messages=messages,
                    tool_calls_made=total_tool_calls,
                    iterations=iteration,
                )

        # Max iterations reached — return last non-tool-call content if available
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content") and not msg.get("tool_calls"):
                return AgentResult(
                    final_response=msg["content"],
                    messages=messages,
                    tool_calls_made=total_tool_calls,
                    iterations=self.max_iterations,
                )
        return AgentResult(
            final_response="[max iterations reached]",
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
