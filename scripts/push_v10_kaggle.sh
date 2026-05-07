#!/usr/bin/env bash
# Push V10 ABT to Kaggle as dataset, then push the Chronos kernel.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export KAGGLE_API_TOKEN="$(grep KAGGLE_API_TOKEN .env | cut -d= -f2)"
USERNAME="${KAGGLE_USERNAME:-${KAGGLE_USER:-}}"
if [[ -z "$USERNAME" ]]; then
  echo "Set KAGGLE_USERNAME in your environment or .env" >&2
  exit 1
fi
DATASET_SLUG="$USERNAME/bpm-v10-abt"

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT

echo "Staging V10 dataset in $STAGE_DIR"
cp output/abt_v10_cached.parquet "$STAGE_DIR/"
cp output/abt_v9_cached.parquet "$STAGE_DIR/"
cp output/v10_feature_manifest.json "$STAGE_DIR/" || true
(cd "$REPO_ROOT" && zip -qr "$STAGE_DIR/src_and_scripts.zip" src scripts)

cat > "$STAGE_DIR/dataset-metadata.json" <<EOF
{
  "title": "BPM V10 ABT",
  "id": "$DATASET_SLUG",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

if kaggle datasets list -m -s bpm-v10-abt 2>/dev/null | grep -q "$DATASET_SLUG"; then
  echo "Dataset $DATASET_SLUG exists -- pushing new version."
  (cd "$STAGE_DIR" && kaggle datasets version -m "v10 build $(date -u +%Y-%m-%dT%H:%M:%SZ)")
else
  echo "Creating dataset $DATASET_SLUG"
  (cd "$STAGE_DIR" && kaggle datasets create)
fi

echo ""
echo "Now pushing Chronos kernel..."

KERNEL_STAGE="$(mktemp -d)"
cp notebooks/v10_chronos_kaggle.ipynb "$KERNEL_STAGE/kernel.ipynb"
cat > "$KERNEL_STAGE/kernel-metadata.json" <<EOF
{
  "id": "$USERNAME/bpm-v10-chronos",
  "title": "bpm-v10-chronos",
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

(cd "$KERNEL_STAGE" && kaggle kernels push)
rm -rf "$KERNEL_STAGE"
echo "kernel queued: $USERNAME/bpm-v10-chronos"
