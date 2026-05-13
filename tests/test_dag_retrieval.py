"""
Tests for mcp_memory/dag_retrieval.py — DAG-structured retrieval pipeline.

All LLM calls and memory_search calls are mocked.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_memory.dag_retrieval import (
    _route_query,
    _topological_sort,
    _parse_json_response,
    _decompose_query,
    _extract_edges,
    dag_search,
    DAGSearchResult,
)


# ---------------------------------------------------------------------------
# Complexity router tests
# ---------------------------------------------------------------------------


class TestRouteQuery:
    def test_short_query_routes_direct(self):
        assert _route_query("What is Wintermute?") == "direct"

    def test_simple_question_routes_direct(self):
        assert _route_query("List all memory entries") == "direct"

    def test_multi_hop_signal_routes_dag(self):
        assert _route_query(
            "What strategy did we use for the task that caused the OOM error?"
        ) == "dag"

    def test_conjunction_routes_dag(self):
        assert _route_query(
            "What happened before the model training and then after deployment?"
        ) == "dag"

    def test_relationship_routes_dag(self):
        assert _route_query(
            "What is the relationship between the Freud auditor and memory trust scores?"
        ) == "dag"

    def test_long_query_routes_dag(self):
        long_q = "Tell me about " + " ".join(["something"] * 30)
        assert _route_query(long_q) == "dag"

    def test_multiple_questions_routes_dag(self):
        assert _route_query("Who built it? And when was it deployed?") == "dag"


# ---------------------------------------------------------------------------
# Topological sort tests
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    def test_no_edges_returns_natural_order(self):
        result = _topological_sort(3, [])
        assert set(result) == {0, 1, 2}
        assert len(result) == 3

    def test_linear_chain(self):
        # 2 depends on 1, 1 depends on 0
        edges = [(2, 1), (1, 0)]
        result = _topological_sort(3, edges)
        assert result.index(0) < result.index(1)
        assert result.index(1) < result.index(2)

    def test_diamond_dependency(self):
        # 3 depends on 1 and 2; 1 depends on 0; 2 depends on 0
        edges = [(3, 1), (3, 2), (1, 0), (2, 0)]
        result = _topological_sort(4, edges)
        assert result.index(0) < result.index(1)
        assert result.index(0) < result.index(2)
        assert result.index(1) < result.index(3)
        assert result.index(2) < result.index(3)

    def test_cycle_graceful_degradation(self):
        # Cycle: 0 -> 1 -> 0
        edges = [(0, 1), (1, 0)]
        result = _topological_sort(2, edges)
        assert set(result) == {0, 1}

    def test_single_node(self):
        result = _topological_sort(1, [])
        assert result == [0]


# ---------------------------------------------------------------------------
# JSON parsing tests
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    def test_clean_json(self):
        assert _parse_json_response('{"key": "value"}') == {"key": "value"}

    def test_markdown_fenced(self):
        text = '```json\n{"subqueries": ["a", "b"]}\n```'
        assert _parse_json_response(text) == {"subqueries": ["a", "b"]}

    def test_json_embedded_in_text(self):
        text = 'Here is the result: {"can_answer": true} hope that helps'
        assert _parse_json_response(text) == {"can_answer": True}

    def test_invalid_returns_none(self):
        assert _parse_json_response("not json at all") is None

    def test_empty_string(self):
        assert _parse_json_response("") is None


# ---------------------------------------------------------------------------
# Decompose query tests (mocked LLM)
# ---------------------------------------------------------------------------


class TestDecomposeQuery:
    @pytest.mark.asyncio
    async def test_successful_decomposition(self):
        mock_response = '{"subqueries": ["What is X?", "How does X relate to Y?"]}'
        with patch("mcp_memory.dag_retrieval._llm_call", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await _decompose_query("complex query", "http://fake/v1", "model")
            assert result == ["What is X?", "How does X relate to Y?"]

    @pytest.mark.asyncio
    async def test_fallback_on_parse_failure(self):
        with patch("mcp_memory.dag_retrieval._llm_call", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "I don't understand"
            result = await _decompose_query("my query", "http://fake/v1", "model")
            assert result == ["my query"]


# ---------------------------------------------------------------------------
# Extract edges tests (mocked LLM)
# ---------------------------------------------------------------------------


class TestExtractEdges:
    @pytest.mark.asyncio
    async def test_valid_edges(self):
        mock_response = '{"dependency_pairs": [[1, 0]]}'
        with patch("mcp_memory.dag_retrieval._llm_call", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await _extract_edges(
                ["What is X?", "What does X lead to?"], "original q", "http://fake/v1", "model"
            )
            assert result == [(1, 0)]

    @pytest.mark.asyncio
    async def test_invalid_indices_filtered(self):
        mock_response = '{"dependency_pairs": [[5, 0], [1, 0]]}'
        with patch("mcp_memory.dag_retrieval._llm_call", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await _extract_edges(
                ["A", "B"], "q", "http://fake/v1", "model"
            )
            assert result == [(1, 0)]

    @pytest.mark.asyncio
    async def test_single_subquery_returns_empty(self):
        result = await _extract_edges(["only one"], "q", "http://fake/v1", "model")
        assert result == []


# ---------------------------------------------------------------------------
# End-to-end dag_search tests (mocked)
# ---------------------------------------------------------------------------


class TestDagSearch:
    @pytest.mark.asyncio
    async def test_direct_route_skips_llm(self):
        fake_results = [{"id": "1", "text": "hello", "similarity": 0.9}]
        with patch("mcp_memory.dag_retrieval._memory_search", return_value=fake_results):
            result = await dag_search("What is X?")
            assert result.routed_as == "direct"
            assert result.rounds == 1
            assert len(result.subquery_results) == 1
            assert result.dag_edges == []

    @pytest.mark.asyncio
    async def test_dag_route_executes_pipeline(self):
        fake_results = [{"id": "1", "text": "fact A", "similarity": 0.8}]

        with patch("mcp_memory.dag_retrieval._memory_search", return_value=fake_results), \
             patch("mcp_memory.dag_retrieval._llm_call", new_callable=AsyncMock) as mock_llm:

            # Sequence: decompose, extract_edges, summarize, can_answer, summarize, can_answer
            mock_llm.side_effect = [
                '{"subqueries": ["What is A?", "How does A affect B?"]}',
                '{"dependency_pairs": [[1, 0]]}',
                "Summary of A",
                '{"can_answer": false}',
                "Summary of A and B relationship",
                '{"can_answer": true}',
            ]

            result = await dag_search(
                "What is the relationship between A and B that caused the issue?",
                force_dag=True,
            )
            assert result.routed_as == "dag"
            assert result.rounds == 2
            assert result.dag_edges == [(1, 0)]
            assert len(result.subquery_results) == 2

    @pytest.mark.asyncio
    async def test_force_dag_overrides_router(self):
        fake_results = [{"id": "1", "text": "short", "similarity": 0.9}]

        with patch("mcp_memory.dag_retrieval._memory_search", return_value=fake_results), \
             patch("mcp_memory.dag_retrieval._llm_call", new_callable=AsyncMock) as mock_llm:

            mock_llm.side_effect = [
                '{"subqueries": ["What is X?"]}',
                '{"dependency_pairs": []}',
                "Summary",
            ]

            result = await dag_search("What is X?", force_dag=True)
            assert result.routed_as == "dag"
