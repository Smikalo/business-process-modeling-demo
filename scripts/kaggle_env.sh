#!/usr/bin/env bash
# Source this file (not execute) to load Kaggle credentials from the repo-root
# `.env` into the current shell. Supports both the new bearer-token format
# (`KAGGLE_API_TOKEN=KGAT_...`, preferred) and the legacy
# `KAGGLE_USERNAME` + `KAGGLE_KEY` pair.
#
# Usage:
#   source scripts/kaggle_env.sh
#   kaggle datasets list --mine
#
# Nothing here costs money: Kaggle datasets, kernels, and GPU quotas are
# entirely free on the standard account.  There is no paid tier to opt into.

_repo_root() {
  # Works in BOTH bash and zsh: bash uses BASH_SOURCE, zsh uses $0
  # (when sourcing via `source` or `.`).
  local src dir
  if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
    src="${BASH_SOURCE[0]}"
  else
    src="$0"
  fi
  dir="$(cd "$(dirname "$src")/.." && pwd)"
  printf '%s' "$dir"
}

_load_dotenv() {
  # Fallback: if _repo_root mis-resolved (e.g. zsh quirks with $0
  # when sourced from non-script context), try $PWD/.env first.
  local envfile
  if [[ -f "$PWD/.env" ]]; then
    envfile="$PWD/.env"
  else
    envfile="$(_repo_root)/.env"
  fi
  if [[ ! -f "$envfile" ]]; then
    echo "kaggle_env: no .env at $envfile (try cd to repo root first)" >&2
    return 1
  fi
  set -a
  # shellcheck disable=SC1090
  . "$envfile"
  set +a
}

_load_dotenv || return 1

if [[ -z "${KAGGLE_API_TOKEN:-}" && -z "${KAGGLE_KEY:-}" ]]; then
  echo "kaggle_env: neither KAGGLE_API_TOKEN nor KAGGLE_KEY is set in .env" >&2
  return 1
fi

# Kaggle CLI 2.x honours KAGGLE_API_TOKEN natively; nothing else to do.
# Echoing username (not the token) so the caller has a visual sanity check.
if command -v kaggle >/dev/null 2>&1; then
  kaggle config view 2>/dev/null | grep -E "^- (username|auth_method):" || true
fi
