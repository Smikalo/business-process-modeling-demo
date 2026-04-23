#!/usr/bin/env bash
# Publish the V6 GPU notebook as a Kaggle kernel and kick off a run.
#
# Uses the free GPU (T4 x2) — there is no paid variant to accidentally pick.
# The 30 GPU-hours / week quota resets automatically; no billing is ever
# enabled on a Kaggle account.
#
# Prerequisites (one-time, already done if .env is populated):
#   - `KAGGLE_API_TOKEN=KGAT_...` in repo-root .env
#   - Dataset pushed via `scripts/push_to_kaggle.sh`
#
# Usage:
#   bash scripts/push_kaggle_kernel.sh
#   bash scripts/push_kaggle_kernel.sh --notebook notebooks/v6_gpu_template.ipynb
#   bash scripts/push_kaggle_kernel.sh --slug mykhailokozyrev/bpm-v6-train

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/kaggle_env.sh"

NOTEBOOK="notebooks/v6_gpu_template.ipynb"
SLUG=""
DATASET_SLUG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --notebook) NOTEBOOK="$2"; shift 2 ;;
    --slug)     SLUG="$2"; shift 2 ;;
    --dataset)  DATASET_SLUG="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,/^set -euo pipefail$/p' "$0"
      exit 0 ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

USERNAME="$(
  kaggle config view 2>/dev/null | awk -F': ' '/^- username:/{print $2; exit}'
)"
[[ -n "$USERNAME" ]] || { echo "Could not read Kaggle username" >&2; exit 1; }

SLUG="${SLUG:-$USERNAME/bpm-v6-train}"
DATASET_SLUG="${DATASET_SLUG:-$USERNAME/bpm-v6-abt}"

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT

cp "$REPO_ROOT/$NOTEBOOK" "$STAGE_DIR/kernel.ipynb"

TITLE="$(basename "$SLUG")"
cat > "$STAGE_DIR/kernel-metadata.json" <<EOF
{
  "id": "$SLUG",
  "title": "$TITLE",
  "code_file": "kernel.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_tpu": "false",
  "enable_internet": "true",
  "dataset_sources": ["$DATASET_SLUG"],
  "competition_sources": [],
  "kernel_sources": []
}
EOF

echo "Pushing kernel $SLUG"
echo "Attaching dataset $DATASET_SLUG"
echo "GPU: T4 x2 (free quota)"

(cd "$STAGE_DIR" && kaggle kernels push)

echo ""
echo "Kernel queued. Watch progress with:"
echo "  kaggle kernels status $SLUG"
echo "Pull artefacts when done:"
echo "  bash scripts/pull_kaggle_kernel_output.sh --slug $SLUG"
