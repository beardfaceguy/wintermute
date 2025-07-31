#!/usr/bin/env python3
"""
Test runner for thVoice unit tests.
"""
import subprocess
import sys
from pathlib import Path


def run_tests(test_pattern: str = "tests/") -> int:
    """Run tests with the specified pattern."""
    cmd = ["python", "-m", "pytest", test_pattern, "-v"]
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def run_unit_tests() -> int:
    """Run unit tests only."""
    return run_tests("tests/test_*.py")


def run_integration_tests() -> int:
    """Run integration tests only."""
    return run_tests("tests/test_*_integration.py")


def run_specific_test(test_file: str) -> int:
    """Run a specific test file."""
    return run_tests(f"tests/{test_file}")


def main() -> int:
    """Main function to run tests based on command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python test_runner.py [unit|integration|all|specific <file>]")
        return 1

    command = sys.argv[1]

    if command == "unit":
        return run_unit_tests()
    elif command == "integration":
        return run_integration_tests()
    elif command == "all":
        return run_tests()
    elif command == "specific" and len(sys.argv) > 2:
        return run_specific_test(sys.argv[2])
    else:
        print("Unknown command. Use: unit, integration, all, or specific <file>")
        return 1


if __name__ == "__main__":
    sys.exit(main())
