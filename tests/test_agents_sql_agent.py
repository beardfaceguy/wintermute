"""
Tests for agents/sql_agent.py — batch resilience and validation logic.

The sql_agent test harness runs YAML test cases in a loop. These tests verify
that one failing case doesn't abort the rest of the batch.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.sql_agent import TestCase, TestResult, validate_result


# ── validate_result ──────────────────────────────────────────────────────────


def test_validate_row_count_pass():
    """row_count check passes when enough rows are returned."""
    result_data = {"rows": [{"id": 1}, {"id": 2}], "columns": ["id"]}
    passed, reason = validate_result(result_data, {"type": "row_count", "min": 1})
    assert passed


def test_validate_row_count_fail():
    """row_count check fails when too few rows."""
    result_data = {"rows": [], "columns": ["id"]}
    passed, reason = validate_result(result_data, {"type": "row_count", "min": 1})
    assert not passed


def test_validate_columns_present_pass():
    result_data = {"rows": [{}], "columns": ["id", "text", "zone"]}
    passed, _ = validate_result(result_data, {"type": "columns_present", "columns": ["id", "zone"]})
    assert passed


def test_validate_columns_present_fail():
    result_data = {"rows": [{}], "columns": ["id"]}
    passed, reason = validate_result(result_data, {"type": "columns_present", "columns": ["id", "zone"]})
    assert not passed
    assert "zone" in reason


def test_validate_handles_error_in_result():
    result_data = {"error": "relation does not exist"}
    passed, reason = validate_result(result_data, {"type": "row_count"})
    assert not passed
    assert "error" in reason.lower()


def test_validate_unknown_check_type():
    result_data = {"rows": [{"x": 1}], "columns": ["x"]}
    passed, reason = validate_result(result_data, {"type": "nonexistent_check"})
    assert not passed
    assert "Unknown" in reason


# ── run_test_case_manual resilience ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_manual_case_handles_schema_fetch_failure():
    """run_test_case_manual should return a failed result when schema fetch fails."""
    from agents.sql_agent import run_test_case_manual

    pg_client = AsyncMock()
    mem_client = AsyncMock()

    pg_client.call_tool.side_effect = ConnectionError("DB unreachable")

    case = TestCase(
        id="test-001",
        question="How many entries?",
        expected={"type": "row_count", "min": 1},
        hints=["SELECT COUNT(*) FROM memory_entries"],
    )

    result = await run_test_case_manual(pg_client, mem_client, case)
    assert isinstance(result, TestResult)
    assert not result.passed
    assert "DB unreachable" in result.error


@pytest.mark.asyncio
async def test_manual_case_should_return_error_not_raise():
    """After fix: run_test_case_manual should return a failed TestResult, not raise."""
    from agents.sql_agent import run_test_case_manual

    pg_client = AsyncMock()
    mem_client = AsyncMock()

    pg_client.call_tool.side_effect = ConnectionError("DB unreachable")

    case = TestCase(
        id="test-001",
        question="How many entries?",
        expected={"type": "row_count", "min": 1},
        hints=["SELECT COUNT(*) FROM memory_entries"],
    )

    result = await run_test_case_manual(pg_client, mem_client, case)
    assert isinstance(result, TestResult)
    assert not result.passed
    assert result.error is not None
    assert "DB unreachable" in result.error
