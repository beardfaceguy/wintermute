"""Tests for shared/setup_path.py — sys.path extension utility."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shared.setup_path as sp


def test_extend_path_adds_wintermute_root():
    """extend_path should insert the wintermute repo root into sys.path."""
    expected = str(Path(sp.__file__).resolve().parents[1])

    original_path = sys.path.copy()
    filtered = [p for p in sys.path if p != expected]
    try:
        sys.path[:] = filtered
        sp.extend_path()
        assert expected in sys.path
    finally:
        sys.path[:] = original_path


def test_extend_path_is_idempotent():
    """Calling extend_path twice should not produce duplicate entries."""
    expected = str(Path(sp.__file__).resolve().parents[1])

    original_path = sys.path.copy()
    filtered = [p for p in sys.path if p != expected]
    try:
        sys.path[:] = filtered
        sp.extend_path()
        count_before = sys.path.count(expected)
        sp.extend_path()
        count_after = sys.path.count(expected)
        assert count_before == count_after == 1
    finally:
        sys.path[:] = original_path


def test_extend_path_inserts_at_front():
    """The repo root should be inserted at position 0 for import priority."""
    expected = str(Path(sp.__file__).resolve().parents[1])

    original_path = sys.path.copy()
    filtered = [p for p in sys.path if p != expected]
    try:
        sys.path[:] = filtered
        sp.extend_path()
        assert sys.path[0] == expected
    finally:
        sys.path[:] = original_path
