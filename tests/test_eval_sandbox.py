"""
Tests for eval/sandbox.py — code extraction and sandboxed execution.
"""

from eval.sandbox import extract_code_block, run_code


class TestExtractCodeBlock:
    def test_fenced_python_block(self):
        text = "Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```"
        result = extract_code_block(text)
        assert "def add" in result

    def test_fenced_block_no_language(self):
        text = "```\nx = 1 + 1\n```"
        result = extract_code_block(text)
        assert "x = 1" in result

    def test_no_fence_falls_back_to_def(self):
        text = "Sure! Here you go:\ndef multiply(a, b):\n    return a * b"
        result = extract_code_block(text)
        assert "def multiply" in result

    def test_plain_code_no_prose(self):
        text = "def square(x):\n    return x * x"
        result = extract_code_block(text)
        assert "def square" in result

    def test_empty_string(self):
        result = extract_code_block("")
        assert result == ""


class TestRunCode:
    def test_passing_code(self):
        code = "x = 2 + 2"
        test = "assert x == 4"
        result = run_code(code, test)
        assert result.passed is True
        assert result.timed_out is False

    def test_failing_code(self):
        code = "x = 1"
        test = "assert x == 99"
        result = run_code(code, test)
        assert result.passed is False

    def test_syntax_error(self):
        code = "def broken(:"
        test = "pass"
        result = run_code(code, test)
        assert result.passed is False

    def test_timeout(self):
        code = "while True: pass"
        test = "pass"
        result = run_code(code, test, timeout=2)
        assert result.passed is False
        assert result.timed_out is True

    def test_function_and_assertion(self):
        code = "def add(a, b):\n    return a + b"
        test = "assert add(2, 3) == 5\nassert add(-1, 1) == 0"
        result = run_code(code, test)
        assert result.passed is True

    def test_import_allowed(self):
        code = "import math\nresult = math.sqrt(16)"
        test = "assert abs(result - 4.0) < 1e-9"
        result = run_code(code, test)
        assert result.passed is True
