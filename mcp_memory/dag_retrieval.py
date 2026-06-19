"""
DAG-structured retrieval for multi-hop queries over Wintermute's strategic memory.

Inspired by LogicRAG (AAAI 2026) and Plan*RAG (ICLR 2025). Decomposes complex
queries into subproblems, constructs a dependency DAG, and executes retrieval
in topological order — all at inference time, no pre-built graph required.

Simple single-hop queries bypass the DAG and go straight to vector search.

Usage:
    from mcp_memory.dag_retrieval import dag_search

    result = await dag_search("What strategy did we use for the task that caused the OOM?")
    print(result.answer_summary)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import httpx

logger = logging.getLogger("dag-retrieval")

# ---------------------------------------------------------------------------
# Configuration — reads from shared_api_config.json with env var overrides
# ---------------------------------------------------------------------------


def _load_dag_config() -> dict:
    """Load DAG retrieval config from shared config file."""
    try:
        import json
        from pathlib import Path

        config_path = Path(__file__).resolve().parent.parent / "config" / "shared_api_config.json"
        with open(config_path) as f:
            return json.load(f).get("dag_retrieval", {})
    except Exception:
        return {}


_dag_cfg = _load_dag_config()
_LLM_TIMEOUT = int(os.getenv("DAG_LLM_TIMEOUT", str(_dag_cfg.get("llm_timeout", 30))))
_DAG_MAX_TOKENS = int(os.getenv("DAG_MAX_TOKENS", str(_dag_cfg.get("max_tokens", 512))))
_DAG_TEMPERATURE = float(os.getenv("DAG_TEMPERATURE", str(_dag_cfg.get("temperature", 0.1))))


def _get_llm_config() -> tuple[str, str]:
    """Resolve vLLM base URL and model from shared config or env vars."""
    base_url = os.getenv("DAG_LLM_BASE_URL")
    model = os.getenv("DAG_LLM_MODEL")

    if base_url and model:
        return base_url, model

    try:
        from shared.config_loader import load_vllm_config

        url, cfg_model = load_vllm_config()
        base_url = base_url or url.rsplit("/", 1)[0].replace("/completions", "")
        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        model = model or cfg_model
    except Exception:
        base_url = base_url or "http://localhost:8010/v1"
        model = model or "wintermute-mistral-7b-sft"

    return base_url, model


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SubqueryResult:
    subquery: str
    results: list[dict[str, Any]]
    summary: str


@dataclass
class DAGSearchResult:
    answer_summary: str
    subquery_results: list[SubqueryResult]
    dag_edges: list[tuple[int, int]]
    rounds: int
    routed_as: str  # "direct" | "dag"


# ---------------------------------------------------------------------------
# Complexity router
# ---------------------------------------------------------------------------

_MULTI_HOP_SIGNALS = re.compile(
    r"\b(and then|after|before|because|which|that|who also|"
    r"compared to|relationship between|how did .+ affect|"
    r"what .+ led to|connection between)\b",
    re.IGNORECASE,
)


def _route_query(query: str) -> str:
    """Classify query as 'direct' (single-hop) or 'dag' (multi-hop)."""
    if query.count("?") > 1:
        return "dag"
    if _MULTI_HOP_SIGNALS.search(query):
        return "dag"
    words = query.split()
    if len(words) <= 8:
        return "direct"
    if len(words) > 25:
        return "dag"
    return "direct"


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


async def _llm_call(prompt: str, base_url: str, model: str) -> str:
    """LLM completion call. Tries chat/completions first, falls back to completions."""
    async with httpx.AsyncClient(timeout=_LLM_TIMEOUT) as client:
        # Try chat completions first (works with Mistral, OpenAI, etc.)
        chat_payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": _DAG_TEMPERATURE,
            "max_tokens": _DAG_MAX_TOKENS,
        }
        resp = await client.post(
            f"{base_url}/chat/completions",
            json=chat_payload,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

        # Fall back to plain completions (GPT-2 style models)
        completions_payload = {
            "model": model,
            "prompt": prompt,
            "temperature": _DAG_TEMPERATURE,
            "max_tokens": _DAG_MAX_TOKENS,
        }
        resp = await client.post(
            f"{base_url}/completions",
            json=completions_payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["text"].strip()


def _parse_json_response(text: str) -> dict | None:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


async def _decompose_query(query: str, base_url: str, model: str) -> list[str]:
    """LLM decomposes a complex query into atomic subqueries."""
    prompt = f"""Decompose this question into the minimum set of independent sub-questions that must be answered to fully address it. Each sub-question should target a single fact or relationship.

Question: {query}

Respond as a JSON object:
{{"subqueries": ["sub-question 1", "sub-question 2", ...]}}"""

    response = await _llm_call(prompt, base_url, model)
    parsed = _parse_json_response(response)

    if parsed and "subqueries" in parsed:
        subqueries = parsed["subqueries"]
        if isinstance(subqueries, list) and len(subqueries) > 0:
            return subqueries

    # Fallback: treat the original query as a single subquery
    logger.warning("Decomposition failed, falling back to original query")
    return [query]


async def _extract_edges(
    subqueries: list[str], query: str, base_url: str, model: str
) -> list[tuple[int, int]]:
    """LLM determines dependency pairs between subqueries.

    Returns list of (dependent_idx, dependency_idx) meaning
    subqueries[dependent_idx] depends on the answer to subqueries[dependency_idx].
    """
    if len(subqueries) <= 1:
        return []

    numbered = "\n".join(f"  {i}: {sq}" for i, sq in enumerate(subqueries))
    prompt = f"""Given this question and its sub-questions, identify which sub-questions depend on answers from other sub-questions.

Original question: {query}

Sub-questions:
{numbered}

A dependency (A, B) means sub-question A requires the answer to sub-question B first.

Respond as a JSON object:
{{"dependency_pairs": [[dependent_idx, dependency_idx], ...]}}

If no dependencies exist (all sub-questions are independent), return:
{{"dependency_pairs": []}}"""

    response = await _llm_call(prompt, base_url, model)
    parsed = _parse_json_response(response)

    if parsed and "dependency_pairs" in parsed:
        pairs = parsed["dependency_pairs"]
        # Validate indices
        valid = []
        for pair in pairs:
            if (
                isinstance(pair, list | tuple)
                and len(pair) == 2
                and all(isinstance(x, int) for x in pair)
                and 0 <= pair[0] < len(subqueries)
                and 0 <= pair[1] < len(subqueries)
                and pair[0] != pair[1]
            ):
                valid.append((pair[0], pair[1]))
        return valid

    return []


def _topological_sort(num_nodes: int, edges: list[tuple[int, int]]) -> list[int]:
    """Kahn's algorithm. Returns node indices in dependency order.

    If the graph has a cycle, returns all nodes in arbitrary order
    (graceful degradation).
    """
    in_degree = [0] * num_nodes
    adjacency: dict[int, list[int]] = {i: [] for i in range(num_nodes)}

    for dependent, dependency in edges:
        adjacency[dependency].append(dependent)
        in_degree[dependent] += 1

    queue = [i for i in range(num_nodes) if in_degree[i] == 0]
    order = []

    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If cycle detected, append remaining nodes
    if len(order) < num_nodes:
        logger.warning("Cycle detected in dependency DAG, appending remaining nodes")
        remaining = [i for i in range(num_nodes) if i not in order]
        order.extend(remaining)

    return order


async def _summarize_context(
    query: str,
    new_results: list[dict[str, Any]],
    current_summary: str,
    base_url: str,
    model: str,
) -> str:
    """Rolling summary: integrate new retrieval results into existing context."""
    context_texts = [r.get("text", "") for r in new_results if r.get("text")]
    if not context_texts:
        return current_summary

    context_block = "\n---\n".join(context_texts)

    if not current_summary:
        prompt = f"""Summarize the following information as it relates to answering this question. Preserve specific details, names, and facts.

Question: {query}

Retrieved information:
{context_block}

Concise summary:"""
    else:
        prompt = f"""Integrate new information into the existing summary. Remove redundancies, preserve all relevant facts.

Question: {query}

Current summary:
{current_summary}

New information:
{context_block}

Updated summary:"""

    return await _llm_call(prompt, base_url, model)


async def _check_can_answer(query: str, summary: str, base_url: str, model: str) -> bool:
    """Early termination check: can the query be answered with current info?"""
    prompt = f"""Given this question and the information gathered so far, can the question be fully answered?

Question: {query}

Information gathered:
{summary}

Respond with ONLY a JSON object:
{{"can_answer": true}} or {{"can_answer": false}}"""

    response = await _llm_call(prompt, base_url, model)
    parsed = _parse_json_response(response)

    if parsed and "can_answer" in parsed:
        return bool(parsed["can_answer"])
    return False


# ---------------------------------------------------------------------------
# Memory search bridge
# ---------------------------------------------------------------------------


def _memory_search(
    query: str,
    limit: int = 5,
    zone: str | None = None,
    min_trust: float | None = None,
) -> list[dict[str, Any]]:
    """Call the existing memory_search function from mcp_memory."""
    try:
        from mcp_memory.server import memory_search
    except ImportError:
        from server import memory_search

    return memory_search(query=query, limit=limit, zone=zone, min_trust=min_trust)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def dag_search(
    query: str,
    limit_per_hop: int = 5,
    max_hops: int = 4,
    zone: str | None = None,
    min_trust: float | None = None,
    force_dag: bool = False,
) -> DAGSearchResult:
    """High-level retrieval: routes simple queries to direct vector search,
    decomposes complex queries into a DAG, executes in topological order.

    Args:
        query: Natural-language search query.
        limit_per_hop: Max results per subquery retrieval.
        max_hops: Maximum number of DAG nodes to execute.
        zone: Filter by memory zone ('live' or 'cold').
        min_trust: Minimum trust_score filter.
        force_dag: Bypass router and always use DAG pipeline.

    Returns:
        DAGSearchResult with aggregated context and per-node transparency.
    """
    route = "dag" if force_dag else _route_query(query)

    # Direct path: single vector search
    if route == "direct":
        results = _memory_search(query, limit=limit_per_hop, zone=zone, min_trust=min_trust)
        summary = "\n".join(r.get("text", "") for r in results[:3])
        return DAGSearchResult(
            answer_summary=summary,
            subquery_results=[SubqueryResult(subquery=query, results=results, summary=summary)],
            dag_edges=[],
            rounds=1,
            routed_as="direct",
        )

    # DAG path
    base_url, model = _get_llm_config()
    logger.info("DAG retrieval for: %s (llm=%s)", query[:80], model)

    # Step 1: Decompose
    subqueries = await _decompose_query(query, base_url, model)
    logger.info("Decomposed into %d subqueries", len(subqueries))

    # Step 2: Extract dependency edges
    edges = await _extract_edges(subqueries, query, base_url, model)
    logger.info("Dependency edges: %s", edges)

    # Step 3: Topological sort
    execution_order = _topological_sort(len(subqueries), edges)

    # Step 4: Execute in order
    subquery_results: list[SubqueryResult] = []
    rolling_summary = ""
    rounds = 0

    for idx in execution_order[:max_hops]:
        rounds += 1
        sq = subqueries[idx]

        results = _memory_search(sq, limit=limit_per_hop, zone=zone, min_trust=min_trust)
        rolling_summary = await _summarize_context(query, results, rolling_summary, base_url, model)
        subquery_results.append(
            SubqueryResult(
                subquery=sq,
                results=results,
                summary=rolling_summary,
            )
        )

        # Early termination
        if rounds < len(execution_order) and await _check_can_answer(
            query, rolling_summary, base_url, model
        ):
            logger.info("Early termination at round %d/%d", rounds, len(execution_order))
            break

    return DAGSearchResult(
        answer_summary=rolling_summary,
        subquery_results=subquery_results,
        dag_edges=edges,
        rounds=rounds,
        routed_as="dag",
    )


def dag_search_sync(
    query: str,
    limit_per_hop: int = 5,
    max_hops: int = 4,
    zone: str | None = None,
    min_trust: float | None = None,
    force_dag: bool = False,
) -> dict[str, Any]:
    """Synchronous wrapper for dag_search. Returns a plain dict."""
    import asyncio

    result = asyncio.run(dag_search(query, limit_per_hop, max_hops, zone, min_trust, force_dag))
    return asdict(result)
