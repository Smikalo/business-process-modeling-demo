# Free-GPU workflow (Kaggle / Colab)

This repo ships two self-contained notebooks that let you run the V6
pipeline on a free T4 / P100 GPU.  No credit card required.

* [`notebooks/v6_gpu_template.ipynb`](../notebooks/v6_gpu_template.ipynb) — **Kaggle Kernel** template (30 h/week GPU quota).
* [`notebooks/v6_colab_template.ipynb`](../notebooks/v6_colab_template.ipynb) — **Google Colab** template (12 h/session, T4 GPU).

Both notebooks:

1. Install dependencies pinned to `requirements.txt`.
2. Mount / upload the V6 ABT (`output/abt_v6_cached.parquet`) and the `src/` tree as a single dataset archive.
3. Run training (LightGBM GPU build or PyTorch TFT) with checkpoints every 50 rounds.
4. Save metrics + feature importance back to `/kaggle/working` (or Drive for Colab) so you can pull artefacts locally.

## One-time local setup

```bash
pip install kaggle

# Put your Kaggle API token at:
#   ~/.kaggle/kaggle.json   (chmod 600)
# Obtain it from https://www.kaggle.com/settings  →  "Create new API token".

kaggle --version
```

## Push the ABT to Kaggle as a dataset

```bash
bash scripts/push_to_kaggle.sh \
     --dataset your-username/bpm-v6-abt \
     --version "v6 build YYYYMMDD"
```

The script uploads:

| path | purpose |
|---|---|
| `output/abt_v6_cached.parquet`  | V6 analytical base table |
| `output/v6_feature_manifest.json` | feature list + imputation summary |
| `src/` (zipped)                | full feature engineering + model code |
| `scripts/train_v6.py`          | GPU-runnable training entry point |

## Kaggle kernel workflow

1. **Create kernel** (Notebook, GPU T4×2 accelerator, Internet on).
2. Attach the dataset `your-username/bpm-v6-abt` in the sidebar.
3. Open `notebooks/v6_gpu_template.ipynb` and run all cells.
4. At the end the kernel writes:
   * `output/model_v6_gpu.joblib`
   * `output/v6_rolling_cv_gpu.json`
   * `output/feature_importance_v6_gpu.csv`

## Pull artefacts back locally

```bash
# For a Kaggle Kernel
kaggle kernels output your-username/bpm-v6-template -p output/gpu/

# For a Colab session — download the Drive folder artefacts directly
# or use the "Files" tab → right-click → "Download".
```

## Optional experiments

| Experiment | Notebook section | When to run |
|---|---|---|
| Optuna 500-trial retune | `§3 Optuna` | If the core V6 pipeline does not hit ≈0.47 test WAPE |
| TFT quantile net (PyTorch-Forecasting) | `§4 TFT` | Only when LGB V6 is stable and you need a second opinion |
| GPU LGB `device=gpu` smoke test | `§1 Sanity` | Always run first (≈30 s); confirms drivers + ABT I/O |

## Troubleshooting

* **`kaggle: command not found`** — `pip install kaggle` and make sure
  `~/.local/bin` is on `$PATH`.
* **`401 Unauthorized`** — regenerate the API token; verify
  `~/.kaggle/kaggle.json` has `0600` permissions.
* **Dataset upload failed: "file too big"** — split parquet shards or
  bump `--version` and try again (Kaggle allows up to 20 GB per dataset).
* **Kernel GPU quota exhausted** — switch to Colab or wait 7 days for the
  Kaggle reset; all notebooks work on either platform unchanged.
