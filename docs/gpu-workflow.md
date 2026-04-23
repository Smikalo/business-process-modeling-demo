# Free-GPU workflow (Kaggle)

This repo ships a self-contained notebook that lets us run the V6 /
upcoming V7 pipeline on a free Kaggle GPU (T4 x2, 30 GPU-hours / week
quota, no billing, no credit card).

* [`notebooks/v6_gpu_template.ipynb`](../notebooks/v6_gpu_template.ipynb) - Kaggle Kernel template.
* [`notebooks/v6_colab_template.ipynb`](../notebooks/v6_colab_template.ipynb) - Google Colab template (kept as a browser-only fallback).

Everything below is scriptable end-to-end. The agent can run all three
scripts without any browser interaction once `.env` contains the Kaggle
token.

## Authentication (once)

1. Generate an **API token** at [kaggle.com/settings](https://www.kaggle.com/settings)
   -> "Create New API Token". This produces either the new bearer-style
   token (`KGAT_...`) or the legacy `kaggle.json` (`username` + `key`).
2. Put it in the repo-root `.env` file (already in `.gitignore`):

   ```
   KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   # OR, legacy form:
   # KAGGLE_USERNAME=your-user
   # KAGGLE_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

3. Install the CLI (once per machine):

   ```bash
   python3 -m pip install --upgrade kaggle
   ```

`scripts/kaggle_env.sh` sources `.env` and exports the right environment
variables for the CLI. Every other script in this section sources it
first, so you never have to think about it again.

> Kaggle has no paid tier for its dataset / kernel / GPU features. The
> 30 hour weekly GPU quota resets automatically. The only "upgrade" is
> verifying your phone number, which is also free.

## Run the full V6 GPU pipeline (three commands)

```bash
# 1. Publish ABT + source snapshot as a private Kaggle dataset
bash scripts/push_to_kaggle.sh

# 2. Publish the training notebook as a GPU kernel (queues a run)
bash scripts/push_kaggle_kernel.sh

# 3. Poll the kernel and download its artefacts into output/gpu/
bash scripts/pull_kaggle_kernel_output.sh
```

Each step is resumable and idempotent:

* `push_to_kaggle.sh` creates the dataset the first time, then pushes
  new versions on subsequent calls (dataset slug defaults to
  `<your-kaggle-user>/bpm-v6-abt`).
* `push_kaggle_kernel.sh` creates kernel
  `<your-kaggle-user>/bpm-v6-train` the first time and updates it on
  subsequent calls. `enable_gpu: true`, `is_private: true`,
  `enable_internet: true` are hard-coded.
* `pull_kaggle_kernel_output.sh` polls `kaggle kernels status` every
  30 s until the run is `complete` / `error`, then downloads
  `/kaggle/working/*` into `output/gpu/`.

## What the kernel produces

Inside the Kaggle notebook:

| Cell | What runs | Artefact(s) |
|---|---|---|
| 1  | `nvidia-smi` sanity | none (log only) |
| 3  | `pip install` deps, unpack `src_and_scripts.zip` | `/kaggle/working/repo/` |
| 5  | `python -m scripts.train_v6 --num-boost-round 1500` | `output/model_v6.joblib`, `output/feature_importance_v6.csv`, `output/v6_metrics.csv`, `output/v6_vs_v5.md` |
| 7  | `python -m scripts.rolling_origin_cv --n-origins 8` | `output/v6_rolling_cv_gpu.{json,md}` |
| 9  | (optional) Optuna 500-trial retune | `output/v6_optuna_best_params.json` |
| 11 | copy all of `output/*` to `/kaggle/working/` | these become the kernel "output" that `pull_kaggle_kernel_output.sh` downloads |

## Pull-back contract

The downloaded files land in `output/gpu/` locally. To promote a GPU
run into the main scorecard, rename or copy what you want into
`output/` and re-run `scripts/viz_model_progression.py` +
`scripts/decision_cost_scorecard.py`. There is no automatic promotion;
GPU runs are opt-in additions.

## Optional experiments (V7)

These sit in the same notebook behind conditional cells so the free
30 h / week quota is never exhausted by accident:

| Experiment | Runtime on T4 x2 | Trigger |
|---|---|---|
| Optuna 500-trial retune | 2-3 h | Cell `OPTUNA=True` |
| Per-segment alpha sweep (12 segments x 10 alphas) | 1 h | Cell `PER_SEGMENT_ALPHA=True` |
| TFT (Temporal Fusion Transformer) on the dense head | 3-6 h | Cell `TFT=True` |
| N-BEATS / DeepAR ablation | 3 h | Cell `DEEP_ABLATION=True` |

Set exactly one `True` per kernel run so the total stays well under
10 h; otherwise the kernel will be killed at the 12 h Kaggle limit.

## Troubleshooting

* **`kaggle: command not found`** - activate the project venv first
  (`source .venv/bin/activate`) or install globally:
  `python3 -m pip install --user kaggle`.
* **`401 Unauthorized`** - `.env` token is stale; regenerate at
  [kaggle.com/settings](https://www.kaggle.com/settings) and rewrite
  the `KAGGLE_API_TOKEN=...` line. Do **not** commit `.env`.
* **`403: You must verify your phone number to run GPU kernels`** -
  one-time Kaggle requirement, free. Happens on first GPU kernel only.
* **`Dataset upload failed: too big`** - 20 GB hard cap per dataset;
  split parquet shards or bump `--version`.
* **Kernel GPU quota exhausted** - wait up to 7 days for the Kaggle
  weekly reset, or run with `enable_gpu: false` in a regular CPU
  kernel.
