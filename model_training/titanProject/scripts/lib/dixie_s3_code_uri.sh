#!/usr/bin/env bash
# Dixie SFT — validate S3 code URI before `aws s3 sync --delete`.
#
# A mistaken shared prefix (e.g. titan/code/latest) combined with --delete can
# wipe objects that belong to other runs. Per-run prefixes look like
# .../dixie_YYYYMMDDHHMMSS.
#
# Override (team convention only):
#   DIXIE_ALLOW_NONSTANDARD_S3_CODE_URI=1
#
# shellcheck shell=bash

dixie_validate_s3_code_uri_for_delete_sync() {
  local uri="${1:?}"
  if [[ "${DIXIE_ALLOW_NONSTANDARD_S3_CODE_URI:-}" == "1" ]]; then
    return 0
  fi
  # Require /dixie_<14-digit UTC ts> as its own path segment (not a suffix like my_dixie_...).
  if [[ ! "${uri}" =~ (^|/)dixie_[0-9]{14}(/|$) ]]; then
    echo "[launch] FATAL: S3_CODE_URI must include per-run segment dixie_YYYYMMDDHHMMSS to limit --delete blast radius." >&2
    echo "[launch] FATAL: uri=${uri}" >&2
    echo "[launch] FATAL: set DIXIE_ALLOW_NONSTANDARD_S3_CODE_URI=1 only if you accept shared-prefix risk." >&2
    return 1
  fi
  return 0
}
