#!/usr/bin/env bash
# Poll a Kaggle kernel until it finishes and download its /kaggle/working
# artefacts into `output/gpu/`.
#
# Kaggle kernels, polling, and output download are free.
#
# Usage:
#   bash scripts/pull_kaggle_kernel_output.sh
#   bash scripts/pull_kaggle_kernel_output.sh --slug mykhailokozyrev/bpm-v6-train
#   bash scripts/pull_kaggle_kernel_output.sh --timeout 3600   # seconds

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/kaggle_env.sh"

SLUG=""
TIMEOUT=3600          # 1 hour default
POLL_SECS=30

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slug)    SLUG="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --poll)    POLL_SECS="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,/^set -euo pipefail$/p' "$0"
      exit 0 ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

USERNAME="$(
  kaggle config view 2>/dev/null | awk -F': ' '/^- username:/{print $2; exit}'
)"
SLUG="${SLUG:-$USERNAME/bpm-v6-train}"

OUT_DIR="$REPO_ROOT/output/gpu"
mkdir -p "$OUT_DIR"

echo "Polling kernel $SLUG every ${POLL_SECS}s (timeout ${TIMEOUT}s)"

elapsed=0
last=""
while (( elapsed < TIMEOUT )); do
  status_raw="$(kaggle kernels status "$SLUG" 2>&1 || true)"
  status="$(printf '%s' "$status_raw" | awk -F'"' '/status/{print $2}' | head -1)"
  status="${status:-$(printf '%s' "$status_raw" | tr -d '\n' | head -c 120)}"
  if [[ "$status" != "$last" ]]; then
    printf '[%5ds] %s\n' "$elapsed" "$status"
    last="$status"
  fi
  case "$status" in
    complete|completed|CompleteSuccess|CompleteWithErrors|success|error|cancelled|cancelAcknowledged) break ;;
  esac
  sleep "$POLL_SECS"
  elapsed=$((elapsed + POLL_SECS))
done

echo "Downloading output to $OUT_DIR"
kaggle kernels output "$SLUG" -p "$OUT_DIR"
ls -la "$OUT_DIR"
