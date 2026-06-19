#!/usr/bin/env python3
"""
Test runner for talkingHead backend tests.
"""

import subprocess
import sys
from pathlib import Path


def run_tests(test_pattern: str = "tests/", verbose: bool = True) -> int:
    """Run pytest with the given pattern."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_pattern,
        "-v" if verbose else "",
        "--tb=short",
        "--disable-warnings",
    ]

    # Remove empty strings
    cmd = [arg for arg in cmd if arg]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def run_unit_tests() -> int:
    """Run all unit tests."""
    print("=" * 60)
    print("Running Unit Tests")
    print("=" * 60)
    return run_tests("tests/")


def run_integration_tests() -> int:
    """Run integration tests."""
    print("=" * 60)
    print("Running Integration Tests")
    print("=" * 60)
    return run_tests("tests/test_integration_*.py")


def run_specific_test(test_name: str) -> int:
    """Run a specific test."""
    print("=" * 60)
    print(f"Running Test: {test_name}")
    print("=" * 60)
    return run_tests(test_name)


def main() -> None:
    """Main test runner."""
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "unit":
            exit_code = run_unit_tests()
        elif command == "integration":
            exit_code = run_integration_tests()
        elif command == "all":
            exit_code = run_unit_tests()
            if exit_code == 0:
                exit_code = run_integration_tests()
        elif command == "specific":
            if len(sys.argv) > 2:
                test_name = sys.argv[2]
                exit_code = run_specific_test(test_name)
            else:
                print("Error: Please provide a test name")
                exit_code = 1
        else:
            print(f"Unknown command: {command}")
            print("Available commands: unit, integration, all, specific")
            exit_code = 1
    else:
        # Default: run all unit tests
        exit_code = run_unit_tests()

    print("=" * 60)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    print("=" * 60)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
