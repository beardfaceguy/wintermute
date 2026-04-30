"""
Wintermute mcp-postgres — MCP server for read-only SQL access to PostgreSQL.

Provides schema introspection and query execution tools so agents can
explore the Wintermute database, generate SQL, and validate results.

Run:
    python mcp_servers/mcp_postgres/server.py                    # stdio (default)
    python mcp_servers/mcp_postgres/server.py --transport http   # HTTP

Cursor config (.cursor/mcp.json):
    {
      "mcpServers": {
        "wintermute-postgres": {
          "command": "python",
          "args": ["mcp_servers/mcp_postgres/server.py"]
        }
      }
    }
"""

from __future__ import annotations

import json
import logging
import os
import sys
from textwrap import dedent

from fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-postgres")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://wintermute:wintermute@localhost:5432/wintermute",
)

# ---------------------------------------------------------------------------
# DB helpers — use psycopg2 directly for lightweight, read-only access
# ---------------------------------------------------------------------------

import psycopg2  # noqa: E402
import psycopg2.extras  # noqa: E402

_DANGEROUS_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER",
    "CREATE", "GRANT", "REVOKE", "COPY",
}


def _is_read_only(sql: str) -> bool:
    """Reject obviously mutating statements before they hit the DB."""
    first_token = sql.strip().split()[0].upper() if sql.strip() else ""
    return first_token not in _DANGEROUS_KEYWORDS


def _get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    conn.set_session(readonly=True, autocommit=True)
    return conn


def _run_query(sql: str, params: tuple | None = None, max_rows: int = 200):
    """Execute a read-only query and return (columns, rows)."""
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            if cur.description is None:
                return [], []
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchmany(max_rows)
            return columns, [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="wintermute-postgres",
    instructions=(
        "Read-only SQL access to the Wintermute PostgreSQL database. "
        "Use sql_list_tables and sql_describe_table to explore the schema, "
        "then sql_query to run SELECT queries. All queries are enforced read-only."
    ),
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool
def sql_list_tables() -> list[dict[str, str]]:
    """List all user-created tables in the database.

    Returns:
        List of tables with schema, name, and type.
    """
    sql = dedent("""\
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name
    """)
    _, rows = _run_query(sql)
    return rows


@mcp.tool
def sql_describe_table(table_name: str, schema: str = "public") -> dict:
    """Describe a table's columns, types, nullability, and constraints.

    Args:
        table_name: Name of the table to describe.
        schema: Schema name (default: 'public').

    Returns:
        Table description with columns and primary key info.
    """
    col_sql = dedent("""\
        SELECT
            column_name,
            data_type,
            udt_name,
            is_nullable,
            column_default,
            character_maximum_length
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """)
    _, columns = _run_query(col_sql, (schema, table_name))

    pk_sql = dedent("""\
        SELECT kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        WHERE tc.constraint_type = 'PRIMARY KEY'
            AND tc.table_schema = %s
            AND tc.table_name = %s
        ORDER BY kcu.ordinal_position
    """)
    _, pk_rows = _run_query(pk_sql, (schema, table_name))
    pk_columns = [r["column_name"] for r in pk_rows]

    fk_sql = dedent("""\
        SELECT
            kcu.column_name,
            ccu.table_name AS foreign_table,
            ccu.column_name AS foreign_column
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage ccu
            ON tc.constraint_name = ccu.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_schema = %s
            AND tc.table_name = %s
    """)
    _, fk_rows = _run_query(fk_sql, (schema, table_name))

    row_count_sql = f"SELECT COUNT(*) as count FROM {schema}.{table_name}"
    _, count_rows = _run_query(row_count_sql)
    row_count = count_rows[0]["count"] if count_rows else "unknown"

    return {
        "schema": schema,
        "table": table_name,
        "row_count": row_count,
        "columns": columns,
        "primary_key": pk_columns,
        "foreign_keys": fk_rows,
    }


@mcp.tool
def sql_query(
    sql: str,
    max_rows: int = 200,
) -> dict:
    """Execute a read-only SQL query and return results.

    Only SELECT, WITH, and EXPLAIN statements are allowed.
    Results are capped at max_rows.

    Args:
        sql: The SQL query to execute (read-only).
        max_rows: Maximum number of rows to return (default 200, max 1000).

    Returns:
        Query results with columns and rows.
    """
    if not _is_read_only(sql):
        return {
            "error": "Only read-only queries are allowed (SELECT, WITH, EXPLAIN).",
            "rejected_sql": sql,
        }

    max_rows = min(max(max_rows, 1), 1000)
    try:
        columns, rows = _run_query(sql, max_rows=max_rows)
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": len(rows) == max_rows,
        }
    except Exception as e:
        return {
            "error": str(e),
            "sql": sql,
        }


@mcp.tool
def sql_explain(sql: str) -> dict:
    """Run EXPLAIN ANALYZE on a query and return the execution plan.

    Args:
        sql: The SQL query to explain (must be read-only).

    Returns:
        The query execution plan.
    """
    if not _is_read_only(sql):
        return {"error": "Only read-only queries can be explained."}

    explain_sql = f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}"
    try:
        columns, rows = _run_query(explain_sql, max_rows=1)
        if rows:
            plan_key = "QUERY PLAN" if "QUERY PLAN" in rows[0] else list(rows[0].keys())[0]
            return {"plan": rows[0][plan_key]}
        return {"plan": None}
    except Exception as e:
        return {"error": str(e), "sql": sql}


@mcp.tool
def sql_sample_rows(
    table_name: str,
    limit: int = 5,
    schema: str = "public",
) -> dict:
    """Return a few sample rows from a table for quick inspection.

    Args:
        table_name: Table to sample from.
        limit: Number of rows (default 5, max 20).
        schema: Schema name (default: 'public').

    Returns:
        Sample rows from the table.
    """
    limit = min(max(limit, 1), 20)
    sql = f"SELECT * FROM {schema}.{table_name} LIMIT {limit}"
    try:
        columns, rows = _run_query(sql, max_rows=limit)
        return {"columns": columns, "rows": rows, "row_count": len(rows)}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource("postgres://schema-summary")
def schema_summary() -> str:
    """High-level summary of all tables, their row counts, and column counts."""
    sql = dedent("""\
        SELECT
            t.table_schema,
            t.table_name,
            COUNT(c.column_name) as column_count
        FROM information_schema.tables t
        LEFT JOIN information_schema.columns c
            ON t.table_schema = c.table_schema
            AND t.table_name = c.table_name
        WHERE t.table_schema NOT IN ('pg_catalog', 'information_schema')
        GROUP BY t.table_schema, t.table_name
        ORDER BY t.table_schema, t.table_name
    """)
    try:
        _, rows = _run_query(sql)
        return json.dumps(rows, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transport = "stdio"
    port = int(os.getenv("MCP_POSTGRES_PORT", "8003"))

    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        if idx + 1 < len(sys.argv):
            transport = sys.argv[idx + 1]

    if transport == "http":
        mcp.run(transport="http", port=port)
    else:
        mcp.run()
