#!/usr/bin/env bash
# Package the V6 ABT + source tree and push to Kaggle as a dataset.
#
# Prerequisites:
#   * `kaggle` CLI installed and authenticated (~/.kaggle/kaggle.json, chmod 600)
#   * A Kaggle account that owns (or collaborates on) the target dataset
#
# Usage:
#   bash scripts/push_to_kaggle.sh \
#        --dataset your-username/bpm-v6-abt \
#        --version "v6 build 2026-04-23"

set -euo pipefail

DATASET=""
VERSION_NOTE="v6 build"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"; shift 2 ;;
    --version)
      VERSION_NOTE="$2"; shift 2 ;;
    *)
      echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$DATASET" ]]; then
  echo "Usage: push_to_kaggle.sh --dataset <owner/slug> [--version 'note']" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT

echo "Staging in $STAGE_DIR"

cp "$REPO_ROOT/output/abt_v6_cached.parquet"    "$STAGE_DIR/"
cp "$REPO_ROOT/output/v6_feature_manifest.json" "$STAGE_DIR/"

# Source code snapshot (src + scripts) as a single zip — lets the Kaggle
# kernel import the repo without needing a git clone.
(cd "$REPO_ROOT" && zip -qr "$STAGE_DIR/src_and_scripts.zip" src scripts)

cat > "$STAGE_DIR/dataset-metadata.json" <<EOF
{
  "title": "BPM V6 ABT",
  "id": "$DATASET",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

# Create or version the dataset.
if kaggle datasets list -m -s "$(basename "$DATASET")" | grep -q "$DATASET"; then
  echo "Dataset exists — pushing new version."
  (cd "$STAGE_DIR" && kaggle datasets version -m "$VERSION_NOTE")
else
  echo "Creating dataset $DATASET"
  (cd "$STAGE_DIR" && kaggle datasets create)
fi

echo "Done. Pull artefacts later with:"
echo "  kaggle kernels output <kernel-slug> -p output/gpu/"
