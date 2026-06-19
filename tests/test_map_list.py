"""
Tests for mcp_servers/map_list.py — directory listing Flask endpoint.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_servers.map_list import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_list_missing_path_returns_400(client):
    resp = client.post("/list", json={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_list_none_path_returns_400(client):
    resp = client.post("/list", json={"test": None})
    assert resp.status_code == 400


def test_list_nonexistent_path_returns_404(client):
    resp = client.post("/list", json={"test": "/nonexistent/path/abc123"})
    assert resp.status_code == 404


@patch("mcp_servers.map_list.os.path.exists", return_value=True)
@patch("mcp_servers.map_list.subprocess.check_output", return_value="file1.txt\nfile2.txt\n")
def test_list_normal_directory(mock_subproc, mock_exists, client):
    resp = client.post("/list", json={"test": "/tmp"})
    data = resp.get_json()
    assert resp.status_code == 200
    assert "file1.txt" in data["contents"]
    assert "file2.txt" in data["contents"]


@patch("mcp_servers.map_list.os.path.exists", return_value=True)
@patch("mcp_servers.map_list.subprocess.check_output", return_value="")
def test_list_empty_directory_returns_empty_list(mock_subproc, mock_exists, client):
    """Empty directory should return [] not [''] — splitting empty string on newline produces ['']."""
    resp = client.post("/list", json={"test": "/tmp/empty"})
    data = resp.get_json()
    assert resp.status_code == 200
    assert data["contents"] == [], f"Expected empty list, got {data['contents']}"


@patch("mcp_servers.map_list.os.path.exists", return_value=True)
@patch("mcp_servers.map_list.subprocess.check_output", return_value="\n")
def test_list_newline_only_returns_empty_list(mock_subproc, mock_exists, client):
    """A single trailing newline (common from ls) should not produce ghost entries."""
    resp = client.post("/list", json={"test": "/tmp"})
    data = resp.get_json()
    assert "" not in data["contents"], f"Ghost empty-string entry found: {data['contents']}"
