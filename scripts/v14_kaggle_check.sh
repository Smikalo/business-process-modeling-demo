#!/usr/bin/env bash
# V14 GlobalNN Kaggle helper.
#
# Usage:
#   ./scripts/v14_kaggle_check.sh status     # check kernel status
#   ./scripts/v14_kaggle_check.sh log        # download + show latest log
#   ./scripts/v14_kaggle_check.sh pull       # download outputs to output/
#   ./scripts/v14_kaggle_check.sh merge      # auto: pull if done, merge to V12.6

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "no .env at $(pwd)/.env" >&2
  exit 1
fi
set -a
. ./.env
set +a

KAGGLE_USER="${KAGGLE_USERNAME:-${KAGGLE_USER:-}}"
if [[ -z "$KAGGLE_USER" ]]; then
  echo "Set KAGGLE_USERNAME in your environment (or .env), e.g.:"
  echo "  export KAGGLE_USERNAME=your-handle"
  exit 1
fi
KERNEL="${KAGGLE_USER}/bpm-v14-globalnn"
OUT="output/v14_kaggle_output"

cmd="${1:-status}"

case "$cmd" in
  status)
    .venv/bin/kaggle kernels status "$KERNEL" 2>&1 | grep -v "outdated" | tail -2
    ;;

  log)
    mkdir -p "$OUT"
    .venv/bin/kaggle kernels output "$KERNEL" -p "$OUT/" 2>&1 | grep -v "outdated" | tail -3
    if [[ -f "$OUT/bpm-v14-globalnn.log" ]]; then
      echo "=== last 30 entries (filtered) ==="
      python3 -c "
import json, sys
log = open('$OUT/bpm-v14-globalnn.log').read()
entries = json.loads(log)
text_log = []
for e in entries:
    text_log.append(e.get('data', '').rstrip())
out = ''.join(text_log).strip()
lines = [l for l in out.split('\n') if l.strip()]
for l in lines[-30:]:
    print(l)
" 2>/dev/null || tail -30 "$OUT/bpm-v14-globalnn.log"
    fi
    ;;

  pull)
    mkdir -p "$OUT"
    .venv/bin/kaggle kernels output "$KERNEL" -p "$OUT/" 2>&1 | grep -v "outdated" | tail -5
    ls -la "$OUT/" | head -10
    ;;

  merge)
    mkdir -p "$OUT"
    .venv/bin/kaggle kernels output "$KERNEL" -p "$OUT/" 2>&1 | grep -v "outdated" > /dev/null
    val="$OUT/preds_v14_globalnn_val.csv"
    test="$OUT/preds_v14_globalnn_test.csv"
    if [[ ! -f "$val" || ! -f "$test" ]]; then
      echo "Predictions not found yet. Files in $OUT/:"
      ls -la "$OUT/"
      echo "Re-run when kernel status is 'COMPLETE'."
      exit 1
    fi
    cp "$val" output/preds_v14_globalnn_val.csv
    cp "$test" output/preds_v14_globalnn_test.csv
    echo "✓ V14 predictions copied to output/"
    echo "Running V12.6 multi-helper joint search with V14_globalnn..."
    PYTHONPATH=. .venv/bin/python -m scripts.v126_multihelper
    ;;

  *)
    echo "usage: $0 {status|log|pull|merge}"
    exit 1
    ;;
esac
