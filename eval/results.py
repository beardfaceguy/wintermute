"""
Result storage for the eval harness.

Results are stored in eval/results/<run_id>.json (human-readable) and
also indexed in eval/results/runs.db (SQLite) for fast querying and
delta comparisons.

Schema
------
runs(run_id, model_id, suite, started_at, finished_at)
scores(id, run_id, benchmark, suite, metric, score, details_json)
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

RESULTS_DIR = Path(__file__).parent / "results"
DB_PATH = RESULTS_DIR / "runs.db"


def _ensure_db():
    RESULTS_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id      TEXT PRIMARY KEY,
            model_id    TEXT NOT NULL,
            suite       TEXT NOT NULL,
            started_at  TEXT NOT NULL,
            finished_at TEXT
        );
        CREATE TABLE IF NOT EXISTS scores (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id       TEXT NOT NULL,
            benchmark    TEXT NOT NULL,
            suite        TEXT NOT NULL,
            metric       TEXT NOT NULL,
            score        REAL NOT NULL,
            details_json TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );
    """)
    con.commit()
    con.close()


@dataclass
class BenchmarkResult:
    benchmark: str
    suite: str
    metric: str
    score: float
    details: dict = field(default_factory=dict)


@dataclass
class RunRecord:
    run_id: str
    model_id: str
    suite: str
    started_at: str
    results: list[BenchmarkResult] = field(default_factory=list)
    finished_at: Optional[str] = None

    def add(self, result: BenchmarkResult):
        self.results.append(result)

    def finish(self):
        self.finished_at = datetime.now(timezone.utc).isoformat()


def new_run(model_id: str, suite: str) -> RunRecord:
    _ensure_db()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]
    started_at = datetime.now(timezone.utc).isoformat()
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, NULL)",
        (run_id, model_id, suite, started_at),
    )
    con.commit()
    con.close()
    return RunRecord(run_id=run_id, model_id=model_id, suite=suite, started_at=started_at)


def save_run(record: RunRecord):
    _ensure_db()
    record.finish()
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "UPDATE runs SET finished_at=? WHERE run_id=?",
        (record.finished_at, record.run_id),
    )
    for r in record.results:
        con.execute(
            "INSERT INTO scores (run_id, benchmark, suite, metric, score, details_json) VALUES (?,?,?,?,?,?)",
            (record.run_id, r.benchmark, r.suite, r.metric, r.score, json.dumps(r.details)),
        )
    con.commit()
    con.close()

    # Also write human-readable JSON
    out_path = RESULTS_DIR / f"{record.run_id}.json"
    with open(out_path, "w") as f:
        data = asdict(record)
        json.dump(data, f, indent=2)
    return out_path


def list_runs() -> list[dict]:
    _ensure_db()
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT run_id, model_id, suite, started_at, finished_at FROM runs ORDER BY started_at DESC"
    ).fetchall()
    con.close()
    return [
        dict(zip(["run_id", "model_id", "suite", "started_at", "finished_at"], r))
        for r in rows
    ]


def load_run(run_id: str) -> dict:
    path = RESULTS_DIR / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No result file for run {run_id}")
    with open(path) as f:
        return json.load(f)
