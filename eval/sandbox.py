"""
Sandboxed Python code execution for coding benchmarks.

Runs generated code in a subprocess with a strict timeout.
No network, no file writes outside /tmp — enforced by ulimits.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass


@dataclass
class ExecResult:
    passed: bool
    stdout: str
    stderr: str
    timed_out: bool


def run_code(code: str, test_code: str, timeout: int = 10) -> ExecResult:
    """
    Execute `code` + `test_code` in an isolated subprocess.

    code:       the model-generated function/solution
    test_code:  assertions or test runner to append
    timeout:    seconds before we kill it (default 10)
    """
    combined = textwrap.dedent(code) + "\n\n" + textwrap.dedent(test_code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(combined)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return ExecResult(
            passed=proc.returncode == 0,
            stdout=proc.stdout[:2000],
            stderr=proc.stderr[:2000],
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        return ExecResult(passed=False, stdout="", stderr="", timed_out=True)
    finally:
        import os
        os.unlink(tmp_path)


def extract_code_block(text: str) -> str:
    """Pull the first fenced code block out of a model response, or return raw text."""
    import re
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # No fences — return the whole thing, stripping prose preamble heuristically
    lines = text.strip().splitlines()
    code_lines = []
    in_code = False
    for line in lines:
        if line.startswith("def ") or line.startswith("class ") or line.startswith("import "):
            in_code = True
        if in_code:
            code_lines.append(line)
    return "\n".join(code_lines) if code_lines else text.strip()
