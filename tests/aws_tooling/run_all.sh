#!/usr/bin/env bash
# Run all aws_tooling unit tests. Returns rc=0 if all pass, rc=1 otherwise.

set -uo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ordered list of test files. Add new ones here.
TESTS=(
  "${THIS_DIR}/test_aws_instance_watcher.sh"
  "${THIS_DIR}/test_aws_lifecycle.sh"
  "${THIS_DIR}/test_remote_training_probe_paths.sh"
)

declare -i total_files=0 ok_files=0 fail_files=0

echo "============================================================"
echo "aws_tooling test suite"
echo "============================================================"
echo

for f in "${TESTS[@]}"; do
  total_files=$((total_files + 1))
  echo "------------------------------------------------------------"
  echo "  $(basename "${f}")"
  echo "------------------------------------------------------------"
  if bash "${f}"; then
    ok_files=$((ok_files + 1))
  else
    fail_files=$((fail_files + 1))
  fi
  echo
done

echo "============================================================"
echo "Suite: ${ok_files}/${total_files} files passed"
echo "============================================================"

exit "$(( fail_files == 0 ? 0 : 1 ))"
