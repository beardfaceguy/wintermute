"""
Tests for mcp_servers/mcp_postgres/server.py — read-only SQL MCP server.

All database access is mocked; no real PostgreSQL connection is required.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# _is_read_only
# ---------------------------------------------------------------------------


class TestIsReadOnly:
    """Validation of the SQL read-only gate."""

    @pytest.fixture(autouse=True)
    def _import_server(self):
        with patch("psycopg2.connect"):
            from mcp_servers.mcp_postgres.server import _is_read_only
            self._is_read_only = _is_read_only

    @pytest.mark.parametrize("sql", [
        "SELECT * FROM users",
        "select count(*) from entries",
        "WITH cte AS (SELECT 1) SELECT * FROM cte",
        "EXPLAIN SELECT 1",
        "explain analyze select 1",
    ])
    def test_accepts_read_statements(self, sql):
        """SELECT, WITH, and EXPLAIN statements are allowed."""
        assert self._is_read_only(sql) is True

    @pytest.mark.parametrize("keyword", [
        "INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE",
        "ALTER", "CREATE", "GRANT", "REVOKE", "COPY",
    ])
    def test_rejects_dangerous_keywords(self, keyword):
        """All mutating keywords are blocked."""
        assert self._is_read_only(f"{keyword} INTO foo VALUES (1)") is False

    @pytest.mark.parametrize("keyword", [
        "insert", "update", "Delete", "dRoP", "TRUNCATE",
    ])
    def test_case_insensitive_rejection(self, keyword):
        """Dangerous keywords are rejected regardless of case."""
        assert self._is_read_only(f"{keyword} something") is False

    def test_empty_string_allowed(self):
        """Empty or whitespace-only strings pass (first_token is empty)."""
        assert self._is_read_only("") is True
        assert self._is_read_only("   ") is True

    def test_whitespace_prefix_select(self):
        """Leading whitespace before SELECT is fine."""
        assert self._is_read_only("   SELECT 1") is True

    def test_rejects_with_delete_cte(self):
        """WITH ... DELETE is a CTE-based mutation — must be blocked."""
        assert self._is_read_only(
            "WITH deleted AS (DELETE FROM users RETURNING *) SELECT * FROM deleted"
        ) is False

    def test_rejects_select_into(self):
        """SELECT INTO creates a new table in PostgreSQL — must be blocked."""
        assert self._is_read_only(
            "SELECT * INTO new_table FROM users"
        ) is False

    def test_rejects_semicolon_chained_mutation(self):
        """Multi-statement payloads with a trailing mutating statement must be blocked."""
        assert self._is_read_only(
            "SELECT 1; DELETE FROM users"
        ) is False


# ---------------------------------------------------------------------------
# sql_query
# ---------------------------------------------------------------------------


class TestSqlQuery:
    """Tests for the sql_query tool function."""

    @pytest.fixture(autouse=True)
    def _import_and_patch(self):
        with patch("psycopg2.connect"):
            import mcp_servers.mcp_postgres.server as mod
            self.mod = mod

    def test_rejects_non_read_only_sql(self):
        """Non-read-only SQL returns an error dict with rejected_sql."""
        result = self.mod.sql_query("DROP TABLE users")
        assert "error" in result
        assert "rejected_sql" in result
        assert result["rejected_sql"] == "DROP TABLE users"

    def test_clamps_max_rows_low(self):
        """max_rows below 1 is clamped to 1."""
        with patch.object(self.mod, "_run_query", return_value=(["id"], [{"id": 1}])):
            result = self.mod.sql_query("SELECT 1", max_rows=-5)
            self.mod._run_query.assert_called_once_with("SELECT 1", max_rows=1)
            assert result["row_count"] == 1

    def test_clamps_max_rows_high(self):
        """max_rows above 1000 is clamped to 1000."""
        with patch.object(self.mod, "_run_query", return_value=([], [])):
            self.mod.sql_query("SELECT 1", max_rows=5000)
            self.mod._run_query.assert_called_once_with("SELECT 1", max_rows=1000)

    def test_returns_proper_structure_on_success(self):
        """Successful query returns columns, rows, row_count, and truncated."""
        fake_cols = ["id", "name"]
        fake_rows = [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
        with patch.object(self.mod, "_run_query", return_value=(fake_cols, fake_rows)):
            result = self.mod.sql_query("SELECT * FROM users", max_rows=200)
            assert result["columns"] == fake_cols
            assert result["rows"] == fake_rows
            assert result["row_count"] == 2
            assert result["truncated"] is False

    def test_truncated_flag_when_rows_equal_max(self):
        """truncated is True when row count equals max_rows."""
        fake_rows = [{"id": i} for i in range(5)]
        with patch.object(self.mod, "_run_query", return_value=(["id"], fake_rows)):
            result = self.mod.sql_query("SELECT id FROM t", max_rows=5)
            assert result["truncated"] is True

    def test_handles_exception_from_run_query(self):
        """Exceptions from _run_query are caught and returned as error dicts."""
        with patch.object(self.mod, "_run_query", side_effect=RuntimeError("connection lost")):
            result = self.mod.sql_query("SELECT 1")
            assert "error" in result
            assert "connection lost" in result["error"]
            assert result["sql"] == "SELECT 1"


# ---------------------------------------------------------------------------
# sql_list_tables
# ---------------------------------------------------------------------------


class TestSqlListTables:
    """Tests for the sql_list_tables tool."""

    @pytest.fixture(autouse=True)
    def _import_and_patch(self):
        with patch("psycopg2.connect"):
            import mcp_servers.mcp_postgres.server as mod
            self.mod = mod

    def test_calls_information_schema(self):
        """sql_list_tables queries information_schema.tables correctly."""
        fake_rows = [{"table_schema": "public", "table_name": "users", "table_type": "BASE TABLE"}]
        with patch.object(self.mod, "_run_query", return_value=([], fake_rows)) as mock_rq:
            result = self.mod.sql_list_tables()
            assert result == fake_rows
            called_sql = mock_rq.call_args[0][0]
            assert "information_schema.tables" in called_sql
            assert "pg_catalog" in called_sql


# ---------------------------------------------------------------------------
# sql_describe_table
# ---------------------------------------------------------------------------


class TestSqlDescribeTable:
    """Tests for the sql_describe_table tool."""

    @pytest.fixture(autouse=True)
    def _import_and_patch(self):
        with patch("psycopg2.connect"):
            import mcp_servers.mcp_postgres.server as mod
            self.mod = mod

    def test_returns_proper_structure(self):
        """sql_describe_table returns schema, table, row_count, columns, pk, fk."""
        col_rows = [{"column_name": "id", "data_type": "uuid", "udt_name": "uuid",
                      "is_nullable": "NO", "column_default": None, "character_maximum_length": None}]
        pk_rows = [{"column_name": "id"}]
        fk_rows = []
        count_rows = [{"count": 42}]

        call_count = [0]
        def fake_run_query(sql, params=None, max_rows=200):
            call_count[0] += 1
            if call_count[0] == 1:
                return ([], col_rows)
            elif call_count[0] == 2:
                return ([], pk_rows)
            elif call_count[0] == 3:
                return ([], fk_rows)
            else:
                return ([], count_rows)

        with patch.object(self.mod, "_run_query", side_effect=fake_run_query):
            result = self.mod.sql_describe_table("users", schema="public")
            assert result["schema"] == "public"
            assert result["table"] == "users"
            assert result["row_count"] == 42
            assert result["columns"] == col_rows
            assert result["primary_key"] == ["id"]
            assert result["foreign_keys"] == []


# ---------------------------------------------------------------------------
# sql_sample_rows
# ---------------------------------------------------------------------------


class TestSqlSampleRows:
    """Tests for the sql_sample_rows tool."""

    @pytest.fixture(autouse=True)
    def _import_and_patch(self):
        with patch("psycopg2.connect"):
            import mcp_servers.mcp_postgres.server as mod
            self.mod = mod

    def test_clamps_limit_low(self):
        """Limit below 1 is clamped to 1."""
        with patch.object(self.mod, "_run_query", return_value=(["id"], [{"id": 1}])):
            result = self.mod.sql_sample_rows("users", limit=0)
            assert result["row_count"] == 1

    def test_clamps_limit_high(self):
        """Limit above 20 is clamped to 20."""
        with patch.object(self.mod, "_run_query", return_value=([], [])) as mock_rq:
            self.mod.sql_sample_rows("users", limit=999)
            called_sql = mock_rq.call_args[0][0]
            assert "LIMIT 20" in called_sql

    def test_returns_proper_structure(self):
        """Successful call returns columns, rows, and row_count."""
        with patch.object(self.mod, "_run_query", return_value=(["id"], [{"id": 1}])):
            result = self.mod.sql_sample_rows("users", limit=5)
            assert "columns" in result
            assert "rows" in result
            assert "row_count" in result

    def test_handles_exception(self):
        """Database errors are caught and returned as error dicts."""
        with patch.object(self.mod, "_run_query", side_effect=RuntimeError("no such table")):
            result = self.mod.sql_sample_rows("nonexistent")
            assert "error" in result
            assert "no such table" in result["error"]


# ---------------------------------------------------------------------------
# sql_explain
# ---------------------------------------------------------------------------


class TestSqlExplain:
    """Tests for the sql_explain tool."""

    @pytest.fixture(autouse=True)
    def _import_and_patch(self):
        with patch("psycopg2.connect"):
            import mcp_servers.mcp_postgres.server as mod
            self.mod = mod

    def test_rejects_non_read_only_sql(self):
        """Mutating SQL is blocked before running EXPLAIN."""
        result = self.mod.sql_explain("DELETE FROM users")
        assert "error" in result
        assert "read-only" in result["error"].lower()

    def test_returns_plan_on_success(self):
        """Valid explain returns the execution plan."""
        fake_plan = [{"Node Type": "Seq Scan"}]
        with patch.object(self.mod, "_run_query", return_value=([], [{"QUERY PLAN": fake_plan}])):
            result = self.mod.sql_explain("SELECT 1")
            assert result["plan"] == fake_plan

    def test_returns_plan_none_when_empty(self):
        """Empty result set returns plan=None."""
        with patch.object(self.mod, "_run_query", return_value=([], [])):
            result = self.mod.sql_explain("SELECT 1")
            assert result["plan"] is None

    def test_handles_exception(self):
        """Exceptions from _run_query produce error dicts."""
        with patch.object(self.mod, "_run_query", side_effect=RuntimeError("timeout")):
            result = self.mod.sql_explain("SELECT 1")
            assert "error" in result
            assert "timeout" in result["error"]
