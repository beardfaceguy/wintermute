"""
Test-driven SQL generation agent.

Loads YAML test cases, uses MCP tools (wintermute-postgres for SQL execution,
wintermute-memory for strategy storage) to:
1. Explore the database schema
2. Generate SQL to answer natural-language questions
3. Validate results against expected outcomes
4. Store successful strategies in memory for future recall

Can run against a live vLLM endpoint or in "manual" mode where
the SQL is provided by the caller (useful for testing the harness itself).

Usage:
    python agents/sql_agent.py                                  # manual mode (no LLM)
    python agents/sql_agent.py --llm                            # LLM mode (uses shared config)
    python agents/sql_agent.py --llm http://gaming-pc-linux:8001/v1  # explicit endpoint
    python agents/sql_agent.py --llm --model mistral-7b         # override model name
    python agents/sql_agent.py --llm --max-retries 5            # more retry attempts
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from fastmcp import Client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_memory.server import mcp as memory_mcp  # noqa: E402
from mcp_servers.mcp_postgres.server import mcp as postgres_mcp  # noqa: E402
from shared.config_loader import load_vllm_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sql-agent")


# ---------------------------------------------------------------------------
# Test case model
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    id: str
    question: str
    expected: dict[str, Any]
    hints: list[str]

    @classmethod
    def from_dict(cls, d: dict) -> "TestCase":
        return cls(
            id=d["id"],
            question=d["question"],
            expected=d.get("expected", {}),
            hints=d.get("hints", []),
        )


@dataclass
class TestResult:
    test_id: str
    passed: bool
    sql_used: str
    result_data: dict[str, Any] | None
    error: str | None
    attempts: int
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Pre-flight LLM check
# ---------------------------------------------------------------------------

async def preflight_check(llm_base_url: str, model: str | None) -> tuple[bool, str, str]:
    """Validate LLM connectivity and discover models.

    Returns (ok, resolved_model, status_message).
    If a model is specified, verifies it's available.
    If model is None, picks the first available model.
    """
    import httpx

    models_url = f"{llm_base_url}/models"
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(models_url)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        return False, "", f"Cannot connect to LLM at {llm_base_url} — is vLLM running?"
    except httpx.TimeoutException:
        return False, "", f"LLM at {llm_base_url} timed out"
    except Exception as e:
        return False, "", f"LLM preflight failed: {e}"

    available = [m["id"] for m in data.get("data", [])]
    if not available:
        return False, "", "LLM is reachable but reports no loaded models"

    if model:
        if model in available:
            return True, model, f"Model '{model}' confirmed (available: {available})"
        return False, "", f"Model '{model}' not found. Available: {available}"

    resolved = available[0]
    return True, resolved, f"Auto-selected model '{resolved}' (available: {available})"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_result(result_data: dict, expected: dict) -> tuple[bool, str]:
    """Validate query results against the expected specification."""
    if "error" in result_data:
        return False, f"Query error: {result_data['error']}"

    check_type = expected.get("type", "row_count")
    rows = result_data.get("rows", [])
    columns = result_data.get("columns", [])

    if check_type == "row_count":
        min_rows = expected.get("min", 1)
        if len(rows) >= min_rows:
            return True, f"Got {len(rows)} rows (min {min_rows})"
        return False, f"Expected at least {min_rows} rows, got {len(rows)}"

    elif check_type == "columns_present":
        required = set(expected.get("columns", []))
        actual = set(columns)
        missing = required - actual
        if not missing:
            return True, f"All required columns present: {required}"
        return False, f"Missing columns: {missing}. Got: {actual}"

    elif check_type == "column_value":
        col = expected.get("column", "")
        val = expected.get("value")
        for row in rows:
            if col in row and row[col] != val:
                return False, f"Row has {col}={row[col]}, expected {val}"
        if not rows:
            return False, "No rows returned"
        return True, f"All rows have {col}={val}"

    return False, f"Unknown check type: {check_type}"


# ---------------------------------------------------------------------------
# SQL Agent (manual mode — no LLM, generates SQL via heuristics/hints)
# ---------------------------------------------------------------------------

async def run_test_case_manual(
    pg_client: Client,
    mem_client: Client,
    case: TestCase,
) -> TestResult:
    """Run a test case in manual mode: use the hints to construct SQL."""
    t0 = time.monotonic()
    logger.info("Running test: %s — %s", case.id, case.question)

    schema_result = await pg_client.call_tool("sql_describe_table", {
        "table_name": "memory_entries",
    })
    schema_result.content[0].text
    logger.info("Schema loaded for memory_entries")

    sql = _generate_sql_from_hints(case)
    if not sql:
        return TestResult(
            test_id=case.id, passed=False, sql_used="",
            result_data=None, error="Could not generate SQL from hints",
            attempts=0, elapsed_s=time.monotonic() - t0,
        )

    logger.info("Generated SQL: %s", sql)

    query_result = await pg_client.call_tool("sql_query", {"sql": sql})
    result_data = json.loads(query_result.content[0].text)

    passed, reason = validate_result(result_data, case.expected)
    logger.info("Test %s: %s — %s", case.id, "PASS" if passed else "FAIL", reason)

    strategy_text = (
        f"SQL strategy for '{case.question}': {sql}\n"
        f"Result: {'PASS' if passed else 'FAIL'} — {reason}"
    )
    await mem_client.call_tool("memory_add", {
        "text": strategy_text,
        "tags": {
            "type": "sql_strategy",
            "test_id": case.id,
            "passed": str(passed),
        },
    })

    return TestResult(
        test_id=case.id, passed=passed, sql_used=sql,
        result_data=result_data, error=None if passed else reason,
        attempts=1, elapsed_s=time.monotonic() - t0,
    )


# ---------------------------------------------------------------------------
# SQL Agent (LLM mode — uses vLLM for query generation)
# ---------------------------------------------------------------------------

async def run_test_case_llm(
    pg_client: Client,
    mem_client: Client,
    case: TestCase,
    llm_base_url: str,
    model: str,
    max_retries: int = 3,
) -> TestResult:
    """Run a test case using the LLM to generate SQL."""
    import httpx

    t0 = time.monotonic()
    logger.info("Running test (LLM): %s — %s", case.id, case.question)

    schema_result = await pg_client.call_tool("sql_describe_table", {
        "table_name": "memory_entries",
    })
    schema_info = schema_result.content[0].text

    prior = await mem_client.call_tool("memory_search", {
        "query": case.question,
        "limit": 3,
        "zone": "cold",
    })
    prior_strategies = prior.content[0].text if prior.content else "[]"

    sql = ""
    result_data: dict[str, Any] = {}
    reason = "no attempts made"

    for attempt in range(1, max_retries + 1):
        prompt = _build_llm_prompt(case, schema_info, prior_strategies, attempt)

        try:
            async with httpx.AsyncClient(timeout=60) as http:
                resp = await http.post(
                    f"{llm_base_url}/chat/completions",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": (
                                "You are a SQL expert. Generate a PostgreSQL query to "
                                "answer the user's question. Return ONLY the raw SQL query. "
                                "Do not include markdown formatting, code fences, or explanations. "
                                "Do not include a trailing semicolon."
                            )},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 500,
                    },
                )
                resp.raise_for_status()
                raw_output = resp.json()["choices"][0]["message"]["content"].strip()
        except httpx.ConnectError:
            reason = f"LLM connection failed on attempt {attempt}"
            logger.error(reason)
            continue
        except httpx.TimeoutException:
            reason = f"LLM timed out on attempt {attempt}"
            logger.error(reason)
            continue
        except httpx.HTTPStatusError as e:
            reason = f"LLM returned {e.response.status_code} on attempt {attempt}"
            logger.error(reason)
            continue
        except (KeyError, IndexError) as e:
            reason = f"Unexpected LLM response structure on attempt {attempt}: {e}"
            logger.error(reason)
            continue

        sql = _clean_sql(raw_output)
        if not sql:
            reason = f"Could not extract SQL from LLM output: {raw_output[:200]}"
            logger.warning("Attempt %d: %s", attempt, reason)
            prior_strategies += f"\n\nFailed attempt: LLM returned non-SQL output"
            continue

        logger.info("Attempt %d SQL: %s", attempt, sql)

        try:
            query_result = await pg_client.call_tool("sql_query", {"sql": sql})
            result_data = json.loads(query_result.content[0].text)
        except Exception as e:
            reason = f"Query execution failed: {e}"
            logger.warning("Attempt %d: %s", attempt, reason)
            prior_strategies += f"\n\nFailed attempt: {sql}\nError: {reason}"
            continue

        passed, reason = validate_result(result_data, case.expected)

        if passed:
            elapsed = time.monotonic() - t0
            logger.info("Test %s: PASS (attempt %d, %.1fs) — %s", case.id, attempt, elapsed, reason)
            strategy_text = (
                f"SQL strategy for '{case.question}': {sql}\n"
                f"Result: PASS — {reason}\n"
                f"Attempts needed: {attempt}"
            )
            await mem_client.call_tool("memory_add", {
                "text": strategy_text,
                "tags": {
                    "type": "sql_strategy",
                    "test_id": case.id,
                    "passed": "true",
                    "model": model,
                    "attempts": str(attempt),
                },
            })
            return TestResult(
                test_id=case.id, passed=True, sql_used=sql,
                result_data=result_data, error=None, attempts=attempt,
                elapsed_s=elapsed,
            )

        logger.info("Test %s: FAIL (attempt %d) — %s", case.id, attempt, reason)
        prior_strategies += f"\n\nFailed attempt: {sql}\nReason: {reason}"

    elapsed = time.monotonic() - t0
    await mem_client.call_tool("memory_add", {
        "text": (
            f"SQL strategy FAILED for '{case.question}' after {max_retries} attempts.\n"
            f"Last SQL: {sql}\nLast error: {reason}"
        ),
        "tags": {
            "type": "sql_strategy",
            "test_id": case.id,
            "passed": "false",
            "model": model,
        },
    })

    return TestResult(
        test_id=case.id, passed=False, sql_used=sql,
        result_data=result_data, error=reason, attempts=max_retries,
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HINT_SQL_MAP = {
    "count_all": "SELECT COUNT(*) as count FROM memory_entries",
    "count_by_zone": "SELECT zone, COUNT(*) as count FROM memory_entries GROUP BY zone",
    "find_flagged": "SELECT * FROM memory_entries WHERE audit_flagged = true",
    "high_trust": "SELECT * FROM memory_entries WHERE trust_score > 0.5 ORDER BY trust_score DESC",
    "tag_filter": "SELECT * FROM memory_entries WHERE tags->>'domain' = 'sql'",
    "recent_entries": "SELECT text, zone, created_at FROM memory_entries ORDER BY created_at DESC LIMIT 3",
}


def _generate_sql_from_hints(case: TestCase) -> str | None:
    """Use known test case IDs to provide reference SQL for manual mode."""
    return _HINT_SQL_MAP.get(case.id)


def _build_llm_prompt(
    case: TestCase,
    schema_info: str,
    prior_strategies: str,
    attempt: int,
) -> str:
    parts = [
        f"Database schema:\n{schema_info}\n",
        f"Question: {case.question}\n",
    ]
    if case.hints:
        parts.append(f"Hints: {'; '.join(case.hints)}\n")
    if prior_strategies and prior_strategies != "[]":
        parts.append(f"Prior strategies for reference:\n{prior_strategies}\n")
    if attempt > 1:
        parts.append(f"This is attempt {attempt}. Previous attempts failed. Try a different approach.\n")
    parts.append("Generate a PostgreSQL SELECT query:")
    return "\n".join(parts)


_SQL_FENCE_RE = re.compile(
    r"```(?:sql|postgresql|pgsql)?\s*\n?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_SELECT_RE = re.compile(
    r"(SELECT\s.+?)(?:;|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def _clean_sql(raw: str) -> str:
    """Extract a clean SQL query from LLM output.

    Handles:
    - Markdown code fences (```sql ... ```)
    - Inline backtick wrapping
    - Explanatory text before/after the query
    - Trailing semicolons
    - Multiple statements (takes the first SELECT)
    """
    raw = raw.strip()

    fence_match = _SQL_FENCE_RE.search(raw)
    if fence_match:
        raw = fence_match.group(1).strip()

    if raw.startswith("`") and raw.endswith("`"):
        raw = raw.strip("`").strip()

    select_match = _SELECT_RE.search(raw)
    if select_match:
        raw = select_match.group(1).strip()

    if raw.endswith(";"):
        raw = raw[:-1].strip()

    if not raw.upper().startswith("SELECT"):
        return ""

    return raw


# ---------------------------------------------------------------------------
# Results reporting
# ---------------------------------------------------------------------------

def _print_summary(
    results: list[TestResult],
    mode: str,
    model: str | None,
    llm_url: str | None,
    total_elapsed: float,
) -> dict[str, Any]:
    """Print and return a structured test results summary."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Mode:    {mode}")
    if model:
        print(f"  Model:   {model}")
    if llm_url:
        print(f"  LLM URL: {llm_url}")
    print(f"  Total:   {total_elapsed:.1f}s")
    print("-" * 60)

    test_details = []
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.test_id:20s}  (attempts: {r.attempts}, {r.elapsed_s:.1f}s)")
        if not r.passed and r.error:
            print(f"         Error: {r.error}")
        test_details.append({
            "test_id": r.test_id,
            "passed": r.passed,
            "attempts": r.attempts,
            "elapsed_s": round(r.elapsed_s, 2),
            "sql": r.sql_used,
            "error": r.error,
        })

    print("-" * 60)
    print(f"  {passed}/{total} tests passed")
    print("=" * 60)

    return {
        "mode": mode,
        "model": model,
        "llm_url": llm_url,
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 2) if total else 0,
        "total_elapsed_s": round(total_elapsed, 2),
        "tests": test_details,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> tuple[str | None, str | None, int]:
    """Parse CLI arguments. Returns (llm_url, model, max_retries).

    Only resolves the shared vLLM config (which requires VLLM_HOST)
    when --llm is used without an explicit URL.
    """
    llm_url: str | None = None
    model: str | None = None
    max_retries = 3
    use_default_llm = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--llm":
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                llm_url = args[i + 1]
                i += 2
            else:
                use_default_llm = True
                i += 1
        elif args[i] == "--model":
            if i + 1 < len(args):
                model = args[i + 1]
                i += 2
            else:
                i += 1
        elif args[i] == "--max-retries":
            if i + 1 < len(args):
                max_retries = int(args[i + 1])
                i += 2
            else:
                i += 1
        else:
            i += 1

    if use_default_llm:
        vllm_url_default, model_default = load_vllm_config()
        llm_url = vllm_url_default.rsplit("/", 1)[0]  # strip /completions → /v1
        if not model:
            model = model_default

    if llm_url and not model:
        model = "/model"

    return llm_url, model, max_retries


async def main():
    test_file = Path(__file__).parent / "test_cases" / "memory_queries.yaml"
    with open(test_file) as f:
        cases_raw = yaml.safe_load(f)
    cases = [TestCase.from_dict(c) for c in cases_raw]

    llm_url, model, max_retries = _parse_args()
    mode = "manual"

    if llm_url:
        mode = "llm"
        print(f"\nPre-flight check: {llm_url}")
        ok, resolved_model, status_msg = await preflight_check(llm_url, model)
        print(f"  {status_msg}")
        if not ok:
            print("\nAborting — LLM is not available. Run in manual mode or start vLLM.")
            sys.exit(1)
        model = resolved_model
        print(f"  Using model: {model}")
        print(f"  Max retries per test: {max_retries}\n")

    t_total = time.monotonic()

    async with Client(postgres_mcp) as pg_client, Client(memory_mcp) as mem_client:
        results: list[TestResult] = []

        for case in cases:
            if llm_url:
                result = await run_test_case_llm(
                    pg_client, mem_client, case, llm_url, model,
                    max_retries=max_retries,
                )
            else:
                result = await run_test_case_manual(pg_client, mem_client, case)
            results.append(result)

        total_elapsed = time.monotonic() - t_total
        summary = _print_summary(results, mode, model, llm_url, total_elapsed)

        strategies = await mem_client.call_tool("memory_search", {
            "query": "SQL strategy",
            "limit": 10,
        })
        strat_data = json.loads(strategies.content[0].text)
        print(f"\n  {len(strat_data)} strategies stored in memory")

        results_file = Path(__file__).parent / "test_cases" / "last_run.json"
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Results written to {results_file}\n")


if __name__ == "__main__":
    asyncio.run(main())
