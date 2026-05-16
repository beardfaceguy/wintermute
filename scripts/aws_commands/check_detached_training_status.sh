#!/usr/bin/env bash
# Synonym for check_detached_titan_status.sh (generic detached training probe).
# Prefer this name for new pipelines; the titan-named script is the canonical entry.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/check_detached_titan_status.sh" "$@"
