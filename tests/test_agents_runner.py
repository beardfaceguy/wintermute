"""Tests for agents/runner.py — AgentRunner, tool dispatch, and LLM loop."""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.runner import AgentResult, AgentRunner, ToolRef, _mcp_tool_to_openai

# ── Dataclass construction ───────────────────────────────────────────────────


def test_toolref_construction():
    """ToolRef should store server_name, client, and schema."""
    client = MagicMock()
    ref = ToolRef(server_name="mem", client=client, schema={"type": "function"})
    assert ref.server_name == "mem"
    assert ref.client is client
    assert ref.schema == {"type": "function"}


def test_agentresult_construction():
    """AgentResult should store response, messages, counts, and iterations."""
    result = AgentResult(
        final_response="done",
        messages=[{"role": "user", "content": "hi"}],
        tool_calls_made=3,
        iterations=2,
    )
    assert result.final_response == "done"
    assert len(result.messages) == 1
    assert result.tool_calls_made == 3
    assert result.iterations == 2


# ── _mcp_tool_to_openai ─────────────────────────────────────────────────────


def test_mcp_tool_to_openai_full_schema():
    """Converts a fully-populated MCP tool descriptor to OpenAI format."""
    tool = MagicMock()
    tool.name = "search"
    tool.description = "Search memory"
    tool.inputSchema = {
        "type": "object",
        "title": "SearchInput",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    result = _mcp_tool_to_openai(tool)

    assert result["type"] == "function"
    assert result["function"]["name"] == "search"
    assert result["function"]["description"] == "Search memory"
    params = result["function"]["parameters"]
    assert params["type"] == "object"
    assert "query" in params["properties"]
    assert "title" not in params  # title should be stripped


def test_mcp_tool_to_openai_missing_schema():
    """Tool with no inputSchema should get default empty object schema."""
    tool = MagicMock()
    tool.name = "ping"
    tool.description = None
    tool.inputSchema = None

    result = _mcp_tool_to_openai(tool)

    assert result["function"]["name"] == "ping"
    assert result["function"]["description"] == ""
    assert result["function"]["parameters"]["type"] == "object"
    assert result["function"]["parameters"]["properties"] == {}


def test_mcp_tool_to_openai_partial_schema():
    """Tool with inputSchema missing 'type' and 'properties' gets defaults."""
    tool = MagicMock()
    tool.name = "status"
    tool.description = "Check status"
    tool.inputSchema = {"title": "StatusInput"}

    result = _mcp_tool_to_openai(tool)

    params = result["function"]["parameters"]
    assert params["type"] == "object"
    assert params["properties"] == {}
    assert "title" not in params


# ── AgentRunner.__init__ ─────────────────────────────────────────────────────


def test_runner_init_defaults(monkeypatch):
    """AgentRunner should have sensible defaults from config."""
    monkeypatch.setattr(
        AgentRunner,
        "_load_defaults",
        staticmethod(
            lambda: {
                "llm_base_url": "http://localhost:8001/v1",
                "model": "wizard-vicuna-7b-awq",
                "max_iterations": 10,
                "temperature": 0.1,
            }
        ),
    )
    runner = AgentRunner()
    assert runner.llm_base_url == "http://localhost:8001/v1"
    assert runner.model == "wizard-vicuna-7b-awq"
    assert runner.max_iterations == 10
    assert runner.temperature == 0.1
    assert "Wintermute" in runner.system_prompt
    assert runner._initialized is False


def test_runner_init_custom():
    """Custom parameters should override defaults."""
    runner = AgentRunner(
        llm_base_url="http://myserver:9000/v1/",
        model="custom-model",
        max_iterations=5,
        temperature=0.7,
        system_prompt="You are a test bot.",
    )
    assert runner.llm_base_url == "http://myserver:9000/v1"  # trailing slash stripped
    assert runner.model == "custom-model"
    assert runner.max_iterations == 5
    assert runner.temperature == 0.7
    assert runner.system_prompt == "You are a test bot."


# ── add_mcp_server ───────────────────────────────────────────────────────────


def test_add_mcp_server_registers_and_resets_init():
    """Adding a server should store it and reset _initialized flag."""
    runner = AgentRunner()
    runner._initialized = True

    server = MagicMock()
    runner.add_mcp_server("memory", server)

    assert "memory" in runner._mcp_servers
    assert runner._mcp_servers["memory"] is server
    assert runner._initialized is False


def test_add_mcp_server_multiple():
    """Multiple servers can be registered."""
    runner = AgentRunner()
    runner.add_mcp_server("a", MagicMock())
    runner.add_mcp_server("b", MagicMock())

    assert len(runner._mcp_servers) == 2


# ── get_openai_tools ─────────────────────────────────────────────────────────


def test_get_openai_tools_returns_registered_schemas():
    """get_openai_tools should return schema dicts from all registered ToolRefs."""
    runner = AgentRunner()
    schema_a = {"type": "function", "function": {"name": "a"}}
    schema_b = {"type": "function", "function": {"name": "b"}}
    runner._tools = {
        "a": ToolRef(server_name="s1", client=MagicMock(), schema=schema_a),
        "b": ToolRef(server_name="s2", client=MagicMock(), schema=schema_b),
    }

    tools = runner.get_openai_tools()
    assert len(tools) == 2
    assert schema_a in tools
    assert schema_b in tools


def test_get_openai_tools_empty_when_no_tools():
    """Should return empty list when no tools registered."""
    runner = AgentRunner()
    assert runner.get_openai_tools() == []


# ── call_tool ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_tool_unknown_returns_error():
    """Unknown tool name should return JSON error without crashing."""
    runner = AgentRunner()
    result = await runner.call_tool("nonexistent", {})
    parsed = json.loads(result)
    assert "error" in parsed
    assert "Unknown tool" in parsed["error"]


@pytest.mark.asyncio
async def test_call_tool_dispatches_to_mcp_client():
    """call_tool should delegate to the MCP client and return the text result."""
    runner = AgentRunner()
    mock_client = AsyncMock()
    content_item = MagicMock()
    content_item.text = '{"data": "hello"}'
    mock_client.call_tool.return_value = MagicMock(content=[content_item])

    runner._tools["greet"] = ToolRef(server_name="test", client=mock_client, schema={})

    result = await runner.call_tool("greet", {"name": "world"})
    assert result == '{"data": "hello"}'
    mock_client.call_tool.assert_awaited_once_with("greet", {"name": "world"})


@pytest.mark.asyncio
async def test_call_tool_empty_content_returns_ok():
    """When MCP returns no content items, call_tool should return ok JSON."""
    runner = AgentRunner()
    mock_client = AsyncMock()
    mock_client.call_tool.return_value = MagicMock(content=[])

    runner._tools["noop"] = ToolRef(server_name="test", client=mock_client, schema={})

    result = await runner.call_tool("noop", {})
    parsed = json.loads(result)
    assert parsed["result"] == "ok"


@pytest.mark.asyncio
async def test_call_tool_handles_exception():
    """Exceptions from MCP client should be caught and returned as JSON error."""
    runner = AgentRunner()
    mock_client = AsyncMock()
    mock_client.call_tool.side_effect = ConnectionError("server down")

    runner._tools["broken"] = ToolRef(server_name="test", client=mock_client, schema={})

    result = await runner.call_tool("broken", {"x": 1})
    parsed = json.loads(result)
    assert "error" in parsed
    assert "server down" in parsed["error"]


# ── run ──────────────────────────────────────────────────────────────────────


def _make_chat_response(content=None, tool_calls=None, finish_reason="stop"):
    """Helper to build a mock chat/completions response."""
    msg = {}
    if content is not None:
        msg["content"] = content
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [
            {
                "message": msg,
                "finish_reason": finish_reason,
            }
        ]
    }


@pytest.mark.asyncio
async def test_run_builds_correct_message_structure():
    """run() should build [system, ...context, user] message list."""
    runner = AgentRunner(system_prompt="Be helpful.")
    runner._initialized = True

    captured_messages = []

    async def fake_chat(messages, tools):
        captured_messages.extend(messages)
        return _make_chat_response(content="Final answer")

    runner._chat_completion = fake_chat
    runner._tools = {}

    await runner.run(
        "What is 2+2?",
        context_messages=[{"role": "assistant", "content": "Prior context"}],
    )

    assert captured_messages[0] == {"role": "system", "content": "Be helpful."}
    assert captured_messages[1] == {"role": "assistant", "content": "Prior context"}
    assert captured_messages[2] == {"role": "user", "content": "What is 2+2?"}


@pytest.mark.asyncio
async def test_run_returns_final_response_no_tools():
    """When the LLM responds without tool calls, run should return immediately."""
    runner = AgentRunner()
    runner._initialized = True

    async def fake_chat(messages, tools):
        return _make_chat_response(content="The answer is 4.")

    runner._chat_completion = fake_chat
    runner._tools = {}

    result = await runner.run("What is 2+2?")

    assert result.final_response == "The answer is 4."
    assert result.tool_calls_made == 0
    assert result.iterations == 1


@pytest.mark.asyncio
async def test_run_handles_tool_call_loop():
    """run() should dispatch tool calls and loop back to LLM."""
    runner = AgentRunner()
    runner._initialized = True

    call_count = [0]

    async def fake_chat(messages, tools):
        call_count[0] += 1
        if call_count[0] == 1:
            return _make_chat_response(
                tool_calls=[
                    {
                        "id": "tc_1",
                        "function": {"name": "add", "arguments": '{"a": 2, "b": 2}'},
                    }
                ],
                finish_reason="tool_calls",
            )
        return _make_chat_response(content="The sum is 4.")

    mock_client = AsyncMock()
    content_item = MagicMock()
    content_item.text = '{"result": 4}'
    mock_client.call_tool.return_value = MagicMock(content=[content_item])

    runner._tools = {
        "add": ToolRef(server_name="math", client=mock_client, schema={}),
    }
    runner._chat_completion = fake_chat

    result = await runner.run("Add 2+2")

    assert result.final_response == "The sum is 4."
    assert result.tool_calls_made == 1
    assert result.iterations == 2


@pytest.mark.asyncio
async def test_run_respects_max_iterations():
    """run() should stop after max_iterations even if LLM keeps calling tools."""
    runner = AgentRunner(max_iterations=2)
    runner._initialized = True

    async def always_tool_call(messages, tools):
        return _make_chat_response(
            tool_calls=[
                {
                    "id": "tc_loop",
                    "function": {"name": "spin", "arguments": "{}"},
                }
            ],
            finish_reason="tool_calls",
        )

    mock_client = AsyncMock()
    content_item = MagicMock()
    content_item.text = "ok"
    mock_client.call_tool.return_value = MagicMock(content=[content_item])

    runner._tools = {
        "spin": ToolRef(server_name="s", client=mock_client, schema={}),
    }
    runner._chat_completion = always_tool_call

    result = await runner.run("Loop forever")

    assert result.iterations == 2
    assert result.tool_calls_made == 2


@pytest.mark.asyncio
async def test_run_handles_malformed_tool_arguments():
    """run() should handle non-JSON tool arguments gracefully (empty dict fallback)."""
    runner = AgentRunner(max_iterations=1)
    runner._initialized = True

    call_count = [0]

    async def fake_chat(messages, tools):
        call_count[0] += 1
        if call_count[0] == 1:
            return _make_chat_response(
                tool_calls=[
                    {
                        "id": "tc_bad",
                        "function": {"name": "oops", "arguments": "NOT VALID JSON"},
                    }
                ],
                finish_reason="tool_calls",
            )
        return _make_chat_response(content="Recovered.")

    mock_client = AsyncMock()
    content_item = MagicMock()
    content_item.text = "ok"
    mock_client.call_tool.return_value = MagicMock(content=[content_item])

    runner._tools = {
        "oops": ToolRef(server_name="s", client=mock_client, schema={}),
    }
    runner._chat_completion = fake_chat

    # Should not raise — malformed args become {}
    await runner.run("Test bad args")
    mock_client.call_tool.assert_awaited_with("oops", {})


# ── cleanup ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cleanup_clears_state():
    """cleanup() should disconnect clients and reset all internal state."""
    runner = AgentRunner()
    mock_client = AsyncMock()
    runner._clients = {"s1": mock_client}
    runner._tools = {"t1": ToolRef(server_name="s1", client=mock_client, schema={})}
    runner._initialized = True

    await runner.cleanup()

    assert len(runner._clients) == 0
    assert len(runner._tools) == 0
    assert runner._initialized is False
    mock_client.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_tolerates_exit_exceptions():
    """cleanup() should not raise even if a client's __aexit__ fails."""
    runner = AgentRunner()
    mock_client = AsyncMock()
    mock_client.__aexit__.side_effect = RuntimeError("cleanup boom")
    runner._clients = {"bad": mock_client}
    runner._initialized = True

    await runner.cleanup()  # should not raise

    assert len(runner._clients) == 0
    assert runner._initialized is False


# ── CLA-262: partial init cleanup ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_partial_init_cleans_up_opened_clients():
    """CLA-262: If _ensure_initialized fails mid-loop, already-opened clients must be cleaned up."""
    runner = AgentRunner()

    good_server = MagicMock()
    bad_server = MagicMock()
    runner.add_mcp_server("good", good_server)
    runner.add_mcp_server("bad", bad_server)

    good_client = AsyncMock()
    good_tool = MagicMock()
    good_tool.name = "tool_a"
    good_tool.description = "A tool"
    good_tool.inputSchema = {"type": "object", "properties": {}}
    good_client.list_tools.return_value = [good_tool]

    bad_client = AsyncMock()
    bad_client.__aenter__.side_effect = ConnectionError("server unreachable")

    call_count = 0

    def make_client(server):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return good_client
        return bad_client

    with patch("agents.runner.Client", side_effect=make_client):
        with pytest.raises(ConnectionError, match="server unreachable"):
            await runner._ensure_initialized()

    assert runner._initialized is False
    assert len(runner._clients) == 0, "Partially-opened clients should have been cleaned up"
    assert len(runner._tools) == 0, "Tools from partial init should have been cleaned up"
    good_client.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_partial_init_no_leak_on_tool_discovery_failure():
    """CLA-262: If tool discovery fails after client connect, the client must still be cleaned up."""
    runner = AgentRunner()
    runner.add_mcp_server("flaky", MagicMock())

    flaky_client = AsyncMock()
    flaky_client.list_tools.side_effect = RuntimeError("tool listing failed")

    with patch("agents.runner.Client", return_value=flaky_client):
        with pytest.raises(RuntimeError, match="tool listing failed"):
            await runner._ensure_initialized()

    assert runner._initialized is False
    assert len(runner._clients) == 0
    flaky_client.__aexit__.assert_awaited_once()
