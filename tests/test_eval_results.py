"""
Tests for eval/results.py — run record CRUD with a temp SQLite database.
"""

import json

import pytest

from eval.results import BenchmarkResult, RunRecord, list_runs, load_run, new_run, save_run


@pytest.fixture
def tmp_results_dir(tmp_path, monkeypatch):
    """Redirect RESULTS_DIR to a temp directory for each test."""
    monkeypatch.setattr("eval.results.RESULTS_DIR", tmp_path)
    monkeypatch.setattr("eval.results.DB_PATH", tmp_path / "runs.db")
    return tmp_path


class TestBenchmarkResult:
    def test_creation(self):
        r = BenchmarkResult(benchmark="mmlu", suite="intelligence", metric="accuracy", score=0.58)
        assert r.benchmark == "mmlu"
        assert r.score == 0.58
        assert r.details == {}

    def test_with_details(self):
        r = BenchmarkResult(
            "gpqa", "intelligence", "accuracy", 0.40, details={"correct": 80, "total": 198}
        )
        assert r.details["correct"] == 80


class TestRunRecord:
    def test_add_result(self):
        rec = RunRecord(
            run_id="test_001",
            model_id="mistral:7b",
            suite="intelligence",
            started_at="2026-01-01T00:00:00Z",
        )
        rec.add(BenchmarkResult("mmlu", "intelligence", "accuracy", 0.58))
        assert len(rec.results) == 1
        assert rec.results[0].score == 0.58

    def test_finish_sets_timestamp(self):
        rec = RunRecord(
            run_id="test_002",
            model_id="mistral:7b",
            suite="intelligence",
            started_at="2026-01-01T00:00:00Z",
        )
        assert rec.finished_at is None
        rec.finish()
        assert rec.finished_at is not None


class TestNewRunAndSaveRun:
    def test_new_run_creates_db_record(self, tmp_results_dir):
        rec = new_run(model_id="mistral:7b", suite="intelligence")
        assert rec.model_id == "mistral:7b"
        assert rec.suite == "intelligence"
        assert rec.run_id
        assert (tmp_results_dir / "runs.db").exists()

    def test_save_run_writes_json(self, tmp_results_dir):
        rec = new_run(model_id="gpt-4o", suite="intelligence")
        rec.add(BenchmarkResult("mmlu", "intelligence", "accuracy", 0.856))
        out_path = save_run(rec)
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["model_id"] == "gpt-4o"
        assert data["results"][0]["score"] == 0.856

    def test_load_run_roundtrip(self, tmp_results_dir):
        rec = new_run(model_id="claude-sonnet-4-6", suite="intelligence")
        rec.add(BenchmarkResult("gpqa_diamond", "intelligence", "accuracy", 0.404))
        save_run(rec)
        loaded = load_run(rec.run_id)
        assert loaded["model_id"] == "claude-sonnet-4-6"
        assert loaded["results"][0]["benchmark"] == "gpqa_diamond"

    def test_list_runs_returns_records(self, tmp_results_dir):
        rec1 = new_run(model_id="mistral:7b", suite="intelligence")
        save_run(rec1)
        rec2 = new_run(model_id="gpt-4o", suite="coding")
        save_run(rec2)
        runs = list_runs()
        model_ids = [r["model_id"] for r in runs]
        assert "mistral:7b" in model_ids
        assert "gpt-4o" in model_ids

    def test_load_run_missing_raises(self, tmp_results_dir):
        with pytest.raises(FileNotFoundError):
            load_run("nonexistent_run_id")
