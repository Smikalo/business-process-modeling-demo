# Google Colab: V11 Chronos zero-shot — step-by-step instructions

This adds Amazon's Chronos foundation model as a zero-shot base in
the V11 LAD ensemble.  Total time: **~30 minutes of mostly waiting**,
of which ~5 minutes are active steps.

## Why Colab and not Kaggle / local?

Tried twice on Kaggle in V10 and both kernels crashed with
`torchvision::nms does not exist` and a CUDA-version mismatch.
Colab Free has a cleaner Python environment for HuggingFace deep-learning
packages, the T4 GPU is more than enough, and your local Mac has no
CUDA-capable GPU.

---

## Step 1 — Open the notebook in Colab

Open this URL in your browser:

> [https://colab.research.google.com/](https://colab.research.google.com/)

In the modal that opens, click **GitHub** tab → paste:

> `https://github.com/<YOUR_REPO>/blob/main/notebooks/v11_chronos_colab.ipynb`

…or alternatively, click **Upload** tab and drag-drop the local file:

> `notebooks/v11_chronos_colab.ipynb`

(Colab also accepts `.ipynb` files directly, which is simpler if your
repo is private.)

---

## Step 2 — Switch the runtime to a free T4 GPU

In the Colab top menu:

1. **Runtime** → **Change runtime type**
2. Hardware accelerator → **T4 GPU**
3. Click **Save**

You'll see "Connected to Python 3 + T4" in the top-right when it's ready.

If you get "GPU not available right now", just wait 5–10 minutes and
retry. Colab Free GPUs are first-come-first-served.

---

## Step 3 — Upload the V10 ABT to your Google Drive

The notebook needs `output/abt_v10_cached.parquet` from this repo
(75 MB).  The simplest workflow:

1. In your local terminal:
   ```bash
   open https://drive.google.com/drive/my-drive
   ```
2. In Drive, create a new folder called **`v11_chronos`** at the root of *My Drive*.
3. Drag `output/abt_v10_cached.parquet` (from this repo) into that folder.

Done. Drive will sync it within seconds.

---

## Step 4 — Run the notebook cells one by one

Click each cell in order and press **Shift + Enter**.

### Cell 1 — install Chronos (~ 60 s, **may need a runtime restart**)

You should see something like:
```
torch 2.10.0+cu128 cuda True
transformers 4.49.0   (must be 4.48 <= x < 5)
GPU: Tesla T4
CUDA cap.: (7, 5)
```

**If `transformers` prints `5.x.x`** (Colab Free ships 5.x by default —
incompatible with `chronos-forecasting==1.5.2`):

1. **Runtime → Restart session** (top menu)
2. Re-run **only Cell 1** once more
3. Now `transformers` will load as 4.49.x and the rest of the
   notebook works.

If you see `cuda False`, go back to Step 2 — runtime is on CPU.

If pip prints a `dependency resolver` warning about
`transformers<5,>=4.48`, ignore it: it appears mid-install before pip
finds a compatible version.  As long as the final `transformers x.y.z`
print is in the 4.48–4.99 range, you are fine.

### Cell 2 — mount Drive (~ 5 s)

A pop-up will ask permission to mount your Drive. Click **Connect to
Google Drive**, authorise the *Colab* app, and Colab returns to the
notebook.

You should see:
```
ABT shape: (316498, 191)
```

If it can't find the file, double-check the path: it must be exactly
`/content/drive/MyDrive/v11_chronos/abt_v10_cached.parquet`.
If you placed it elsewhere, edit the `ABT_PATH` variable in Cell 2.

### Cell 3 — pivot to wide format (~ 5 s)

Output:
```
active pairs: ~4500
wide shape: (~4500, 73)
```

### Cell 4 — load Chronos weights (~ 2 min, first time)

Downloads ~200 MB to Colab's local disk. Output ends with
`OK`.  Subsequent runs (within 12 h) re-use the cache.

### Cell 5 — zero-shot forecasting (~ 25 min)

The progress bar shows **19 periods** (12 val + 7 test).
Each period takes ~1.5 min to forecast for ~4500 SKUs in batches of 256
on T4. You can keep the browser tab in the background — Colab does not
disconnect idle GPU sessions for at least 90 minutes.

If you see `OutOfMemoryError`, lower the `BATCH` variable in Cell 5
from 256 to 128 and re-run only Cell 5.

### Cell 6 — write CSVs to Drive (~ 5 s)

Output ends with the two CSV paths in your Drive. The notebook also
writes a sanity-check, e.g.:
```
wrote /content/drive/MyDrive/v11_chronos/preds_v11_chronos_val.csv (~54000 rows)
wrote /content/drive/MyDrive/v11_chronos/preds_v11_chronos_test.csv (~31000 rows)
```

### Cell 7 — quick metrics (~ 5 s)

Expected ballpark output:
```
VAL : {'rows': ~54000, 'WAPE': 0.6–0.8, 'bias_pct': −20 to +20}
TEST: {'rows': ~31000, 'WAPE': 0.7–0.9, 'bias_pct': −20 to +20}
```

These look "bad" relative to the LightGBM bases — they should!
Chronos is zero-shot. The point is *orthogonal* residuals, not raw
accuracy. Even a WAPE-0.8 base can earn weight in LAD if its errors
are uncorrelated with the LightGBM bases.

If the bias % is huge (`> 50%`), there's likely a problem with the
input series (e.g., training period is treated as zero — check the
pivot in Cell 3 has 73 columns covering 2020-01..2026-01).

---

## Step 5 — Download the CSVs to this repo

In your local terminal:

```bash
cd <repo-root>

# either via gdrive CLI (`brew install rclone gdrive`)…
gdrive download --recursive --path output v11_chronos

# …or simpler: just open Drive in a browser, select both files,
# right-click → "Download". They'll arrive as a .zip — extract into
# output/.
unzip ~/Downloads/v11_chronos*.zip -d output/

# verify they exist:
ls output/preds_v11_chronos_*.csv
```

---

## Step 6 — Re-run the V11 LAD search

```bash
cd <repo-root>
source .venv/bin/activate
PYTHONPATH=. python -m scripts.v11_lad_stack
PYTHONPATH=. python -m scripts.v11_final_blend
```

The LAD script will detect Chronos automatically and add a
`v10+v11_chronos` pool to the search.  If Chronos earns LAD weight,
expect:

* test SIMSCORE another 1–3 % lower
* test bias closer to zero (foundation model has different bias
  direction than LightGBM)
* an explicit fallback when (Партнер, Артикул) is brand-new and
  has no historical lags

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cuda False` in Cell 1 | Runtime was reset to CPU. Go back to Step 2. |
| `RuntimeError: operator torchvision::nms` | `%pip uninstall -y torchvision` and rerun Cell 1. |
| `OOM error in Cell 5` | Reduce `BATCH` to 128 or 64. |
| Notebook disconnects mid-run | Colab Free disconnects after 12 h or 90 min idle. Reconnect, re-run from Cell 4 (weights are cached, Cell 5 picks up where it left off if you save partial predictions). |
| Wrong path in Cell 2 | Edit `ABT_PATH` to wherever you placed the parquet on Drive. |
| Chronos predicts only zeros | Double-check Cell 3 output: wide shape should be (~4500, 73). |

---

## What if Colab's free GPU is unavailable?

Two backup options:

1. **Modal Labs**: cleaner programmatic GPU. Sign up at
   [modal.com](https://modal.com), they give $30/month free credits which is
   enough for ~10 Chronos runs. The notebook would need to be wrapped in
   a `modal.App` script (~30 lines of additional code, see
   `docs/guides/v11_plan.md` Priority 5 for the template).

2. **HuggingFace Spaces (ZeroGPU)**: free A100 minutes. The notebook
   would need to be ported to a Gradio app first — non-trivial, ~1 day
   of work.

If Colab works (it usually does), don't bother with the alternatives.
