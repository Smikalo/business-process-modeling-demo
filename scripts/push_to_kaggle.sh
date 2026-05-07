#!/usr/bin/env bash
# Package the V6 ABT + source tree and push to Kaggle as a (free) dataset.
#
# Credentials are loaded from `.env` at the repo root
# (`KAGGLE_API_TOKEN=KGAT_...`). No `~/.kaggle/kaggle.json` required.
#
# Kaggle datasets and kernels are free. This script never invokes any
# paid Kaggle product — only `kaggle datasets {create,version}`, which
# are always free regardless of dataset size up to the 20 GB per-dataset
# limit.
#
# Usage:
#   bash scripts/push_to_kaggle.sh                          # defaults: <kaggle-user>/bpm-v6-abt, auto version note
#   bash scripts/push_to_kaggle.sh --dataset u/slug --version "note"
#   bash scripts/push_to_kaggle.sh --dry-run                # stage only, no upload

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Load .env and credentials from the repo root.
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/kaggle_env.sh"

# Default to the authenticated user's slug when --dataset is not supplied.
DEFAULT_USER="$(
  kaggle config view 2>/dev/null | awk -F': ' '/^- username:/{print $2; exit}'
)"
DATASET="${DEFAULT_USER:+$DEFAULT_USER/bpm-v6-abt}"
VERSION_NOTE="v6 build $(date -u +%Y-%m-%dT%H:%M:%SZ)"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)  DATASET="$2"; shift 2 ;;
    --version)  VERSION_NOTE="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '1,/^set -euo pipefail$/p' "$0"
      exit 0 ;;
    *)
      echo "Unknown flag: $1" >&2
      exit 1 ;;
  esac
done

if [[ -z "$DATASET" ]]; then
  echo "Could not derive Kaggle dataset slug (no --dataset, no username). Aborting." >&2
  exit 1
fi

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT

echo "Staging in $STAGE_DIR"
echo "Target dataset:  $DATASET"
echo "Version note:    $VERSION_NOTE"

for f in output/abt_v6_cached.parquet output/v6_feature_manifest.json; do
  if [[ ! -f "$REPO_ROOT/$f" ]]; then
    echo "Missing required artefact: $f" >&2
    exit 1
  fi
  cp "$REPO_ROOT/$f" "$STAGE_DIR/"
done

# Source snapshot so a Kaggle kernel can `sys.path.insert(0, ...)` without git.
(cd "$REPO_ROOT" && zip -qr "$STAGE_DIR/src_and_scripts.zip" src scripts)

cat > "$STAGE_DIR/dataset-metadata.json" <<EOF
{
  "title": "BPM V6 ABT",
  "id": "$DATASET",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

if [[ "$DRY_RUN" == "1" ]]; then
  echo "--dry-run: staged the following files:"
  ls -la "$STAGE_DIR"
  echo "Would run: kaggle datasets {create|version}"
  exit 0
fi

if kaggle datasets list -m -s "$(basename "$DATASET")" 2>/dev/null | grep -q "$DATASET"; then
  echo "Dataset $DATASET exists — pushing new version."
  (cd "$STAGE_DIR" && kaggle datasets version -m "$VERSION_NOTE")
else
  echo "Creating dataset $DATASET"
  (cd "$STAGE_DIR" && kaggle datasets create)
fi

echo ""
echo "Done.  Next steps:"
echo "  1. bash scripts/push_kaggle_kernel.sh           # push + run GPU notebook"
echo "  2. bash scripts/pull_kaggle_kernel_output.sh    # download artefacts"
