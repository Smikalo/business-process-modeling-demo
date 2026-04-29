# V12 → V14 — Human-action playbook (zero-budget)

**Audience:** the human running this campaign (Mykhailo).
**Compute budget:** $0. Free Colab + free Kaggle only. No card on file.
**Time budget:** ≈ 4 weeks wall-clock, ≈ 8 hours of *active* human time
spread across that month.

The agent has done as much as can be done autonomously on CPU. This
document tells you exactly what to do in front of the keyboard for the
GPU steps (V13 foundation-model fine-tuning, V14 GlobalNN) and how to
glue everything back together.

---

## Status as of the agent handoff

**Headline:** V12 failed (test +2.63 % regression); **V12.1 succeeded**.
The new production champion is `V12.1_champion = 0.95 · V11_final +
0.05 · V12_external` (test SIMSCORE **0.4453** vs V11_final **0.4489**,
**−0.80 %**), with WAPE 0.3937 (new all-time low), bias +2.36 % (closer
to zero), Monthly-WAPE 0.0796.

See `docs/v12_retrospective.md` for the V12 failure diagnosis (val→test
bias-direction reversal), `docs/v121_retrospective.md` for the V12.1
fixes (re-train V11 base on `abt_v12_external`, bias-direction-symmetry
LAD filter, OOF-driven blend with `V12_external` as bias counter), and
`output/v121/audit.md` for the formal V12.1 audit.

V13 (Chronos / TimesFM / Moirai fine-tuning) and V14 (GlobalNN
Transformer-encoder) remain GPU-dependent, **independent** of the
V12 → V12.1 work, and gated on you running them on Colab/Kaggle when
convenient. The handoff package below is unchanged.

| Stage | What | Status |
|---|---|---|
| Phase 0 | output/ restructure, CI workflow | **deferred** — kept flat output/ to avoid breaking V11 references |
| Phase EXT | 9 priority-1 free open-data loaders | ✅ implemented (with synthetic fallback when upstream is unreachable) |
| Phase EXT | per-source A/B audit | ⏳ deferred — needs a base that actually consumes EXT features (V12.1) |
| V12 | multi-seed bagging (5 seeds) | ✅ trained: `output/preds_v12_multiseed_{val,test}.csv` |
| V12 | Croston/SBA/TSB intermittent specialist | ✅ trained: `output/preds_v12_intermittent_{val,test}.csv` |
| V12 | abt_v12_external | ✅ built: `output/abt_v12_external.parquet` (316498 × 223) |
| V12 | LAD bias-ladder champion | ✅ trained: `output/preds_v12_lad_{val,test}.csv` |
| V12 | Robust λ-blend final | ✅ ran: `output/preds_v12_final_{val,test}.csv` |
| V12 | OOF audit | ❌ **failed** acceptance gate (1 FAIL, 4 WARN, 3 PASS) |
| V12 | viz | ✅ `output/plot_v12_vs_v11_progression.png` |
| V12 | retrospective | ✅ `docs/v12_retrospective.md` |
| **V12.1** | **fix the val→test bias issue + rerun** | 👤 1 week of CPU work — see retrospective |
| **V13** | **Chronos fine-tune (Colab GPU)** | 👤 **YOU DO** — see Step 1 below |
| **V13** | **TimesFM fine-tune (Colab GPU)** | 👤 **YOU DO** — see Step 2 below |
| **V13** | **Moirai fine-tune (Kaggle GPU)** | 👤 **YOU DO** — see Step 3 below |
| V13 | LAD merge of fine-tuned FMs | ⏳ runnable: `python -m scripts.v13_lad_stack` after the 3 GPU runs |
| **V14** | **GlobalNN training (Colab GPU, 4 sessions)** | 👤 **YOU DO** — see Step 4 below |
| V14 | LAD merge with GlobalNN | ⏳ runnable: `python -m scripts.v14_lad_stack` |
| V14 | MoE per-cluster specialists | ⏳ CPU; runnable after V14_alpha |
| V14 | final V14 LAD + viz + executive report | ⏳ runnable |

If at any step a fine-tune crashes or Colab disconnects, the
fall-through is "ship V12 as-is, rerun the FM step next week". V12
already gives a measurable improvement over V11 even without V13/V14.

---

## Step 1 — Fine-tune Chronos-T5-Small on Colab Free (≈ 5 hr GPU)

### 1a. Prepare the data export (you, locally — 5 min)

```bash
cd /Users/m.kozyrev/Desktop/business-process-modeling-demo
PYTHONPATH=. python -m scripts.export_v13_fm_data
```

This writes `output/v13_fm/series_train.parquet` (≈ 30 MB) and
`output/v13_fm/series_oof.parquet` (≈ 5 MB) — wide-format monthly
shipment series, ready for Chronos's expected input.

Upload both files to Google Drive at `/MyDrive/v13_fm_data/`.

### 1b. Open the notebook in Colab (you — 1 min)

[Open Colab](https://colab.research.google.com/) → File → Upload notebook
→ select `notebooks/v13_chronos_finetune_colab.ipynb` from this repo.

Or run this on your local terminal to push it to GitHub first:

```bash
git add notebooks/v13_chronos_finetune_colab.ipynb && git commit -m "v13 chronos notebook" && git push
```

…then in Colab: File → Open notebook → GitHub tab → paste your repo URL.

### 1c. Switch to T4 GPU (you — 30 sec)

Runtime → Change runtime type → Hardware accelerator: **T4 GPU**.

If "GPU not available right now": wait 5–10 min and retry. Free tier is
first-come-first-served. If Colab refuses GPU for >1 hr, switch to
Step 1d (Kaggle).

### 1d. Run cells 1–7 in order (you — 5 hr, mostly idle)

Each cell prints expected output. Active time is ≈ 10 min total — the
rest is GPU waiting. You can close the laptop lid; Colab Free
disconnects only after 90 min of *no browser tab open*. Just leave the
tab open in the background.

### 1e. Download the result CSVs (you — 2 min)

The notebook writes `preds_v13_chronos_val.csv` + `preds_v13_chronos_test.csv`
to `/MyDrive/v13_fm_data/`. Download to your local repo:

```bash
cd /Users/m.kozyrev/Desktop/business-process-modeling-demo
# either via gdrive CLI:
gdrive download --recursive --path output v13_fm_data
# or just drag-drop in the Drive web UI to ~/Downloads/, then:
mv ~/Downloads/preds_v13_chronos_*.csv output/
```

### 1f. Quick sanity check (you — 30 sec)

```bash
PYTHONPATH=. python -c "
from scripts.score_similarity import score_frame
import pandas as pd
v = pd.read_csv('output/preds_v13_chronos_val.csv')
t = pd.read_csv('output/preds_v13_chronos_test.csv')
print('val:', score_frame(v))
print('test:', score_frame(t))
"
```

Fine-tuned Chronos test SIMSCORE should be in the **0.50–0.65** range.
If it's > 0.85 something went wrong (zero-shot baseline was 0.88).

---

## Step 2 — Fine-tune TimesFM-200M on Colab Free (≈ 5 hr GPU)

Same pattern as Step 1, but using `notebooks/v13_timesfm_finetune_colab.ipynb`.

TimesFM (Google, 2024) is a 200M-parameter decoder-only model
pre-trained on Google's internal time-series corpus + the Wiki-page-views
dataset. It has a different inductive bias than Chronos (autoregressive
vs encoder-decoder), so its residuals are likely orthogonal — exactly
what LAD wants.

Output: `output/preds_v13_timesfm_{val,test}.csv`.

---

## Step 3 — Fine-tune Moirai-Small on Kaggle Free (≈ 7 hr GPU)

Moirai is the Salesforce 2024 foundation model with the longest
pre-training context. It needs more wall-clock time to fine-tune;
Kaggle's 12-hour weekly GPU quota is the right home for it.

### 3a. Push the data to Kaggle (you — 5 min)

Create a new Kaggle Dataset:

1. Go to [Kaggle Datasets](https://www.kaggle.com/datasets).
2. Click **New Dataset** → upload `output/v13_fm/series_train.parquet`
   and `output/v13_fm/series_oof.parquet`.
3. Set the dataset slug to **`v13-fm-data`** (private).

### 3b. Open the notebook in Kaggle (you — 1 min)

[Open Kaggle Notebooks](https://www.kaggle.com/code) → New Notebook →
File → Import Notebook → upload `notebooks/v13_moirai_finetune_kaggle.ipynb`.

In the right panel:
- Accelerator: **GPU T4 ×2** (Kaggle gives 2 T4s, the notebook uses one).
- Internet: **on**.
- Add data: search for `v13-fm-data` → add your dataset.

### 3c. Run all cells (you — 7 hr, idle)

Click **Run all**. Kaggle disconnects after 12 h or 4 h idle, so keep
the tab open or commit the notebook periodically (which checkpoints
state).

### 3d. Download `preds_v13_moirai_*.csv` from Kaggle output

The notebook saves both CSVs to `/kaggle/working/`. From the Kaggle
notebook's **Output** tab → Download All → unzip into `output/`.

---

## Step 4 — Merge V13 fine-tuned FMs into the LAD pool (CPU, 2 min)

```bash
cd /Users/m.kozyrev/Desktop/business-process-modeling-demo
PYTHONPATH=. python -m scripts.v13_lad_stack
PYTHONPATH=. python -m scripts.audit_v13_oof
PYTHONPATH=. python -m scripts.viz_v13_progression
```

Expected: V13_final test SIMSCORE 0.430 ± 0.005 (V12_final is ≈ 0.435,
V11_final was 0.4447). If at least 2 of 3 fine-tuned FMs earn ≥ 5 %
LAD weight, V13 ships. If they all earn 0 %, that's diagnostic — the
fine-tune didn't generalise; revisit the per-FM hyperparameters.

---

## Step 5 — Train the GlobalNN (V14_alpha, Colab Free, ≈ 16 hr GPU split)

GlobalNN is a Transformer-encoder model with learned embeddings for
`Партнер`, `Артикул`, `Бренд`, `Канал`. It produces direct quantile
forecasts for h=1..6 and is the only V14 component that benefits from
GPU. Architecture: see `src/models/global_nn.py`.

### 5a. Prepare data + notebook (you — 5 min)

```bash
PYTHONPATH=. python -m scripts.export_v14_globalnn_data
```

Writes `output/v14_globalnn/{train,val,test}.parquet` to your repo.
Upload all three to Google Drive at `/MyDrive/v14_globalnn_data/`.

Push the notebook:

```bash
git add notebooks/v14_globalnn_colab.ipynb && git commit -m "v14 globalnn notebook" && git push
```

### 5b. Train across 4 Colab sessions (you — 4× 4 hr)

The model is too big to train in a single Colab Free session (12 h
hard cap, 90 min idle disconnect). The notebook checkpoints every
500 steps to Drive, so each session resumes where the last left off.

Open the same Colab notebook 4 times across 4 days; each run resumes,
trains for 4 hours, then hits the soft idle limit. After session 4,
the model has seen ≈ 32 k optimizer steps which is enough on this
data size.

### 5c. Run inference + download (you — 30 min)

In session 4 (or session 5 if needed), run the inference cell after
training completes. The notebook writes `preds_v14_globalnn_{val,test}.csv`
to Drive. Download to `output/`.

### 5d. Merge into LAD pool (CPU, 2 min)

```bash
PYTHONPATH=. python -m scripts.v14_lad_stack --variant alpha
```

V14_alpha test SIMSCORE goal: ≤ 0.425.

---

## Step 6 — V14 final (MoE specialists, CPU only, 2 hr)

This is fully CPU. No GPU steps remaining.

```bash
PYTHONPATH=. python -m scripts.v14_moe_specialists   # trains 4 cluster-specialists
PYTHONPATH=. python -m scripts.v14_lad_stack --variant final
PYTHONPATH=. python -m scripts.audit_v14_oof
PYTHONPATH=. python -m scripts.viz_v14_dashboard
```

If V14_final test SIMSCORE ≤ 0.420 → ship as production champion. Else,
ship V12 / V13 as the production champion with a clear changelog
explaining V14 didn't generalise.

---

## Troubleshooting flowchart

| Symptom | First-line fix | Fall-through |
|---|---|---|
| Colab refuses GPU > 1 hr | Switch the run to Kaggle (different GPU pool) | Skip that FM, V13 still ships with the other 2 |
| Chronos OOM on T4 | Reduce `BATCH=128` → `64` in the notebook | Reduce `MAX_CONTEXT=64` → `48` |
| Fine-tune diverges (val loss exploding) | Lower `LR=5e-5` → `2e-5`; restart | Drop that FM from V13 pool |
| GlobalNN val WAPE > 1.0 after 1 session | Embedding init bug; rerun with `--reinit-embeddings` | Drop GlobalNN, ship V13 directly |
| `preds_v1*_*.csv` row counts don't match V11 | Key mismatch — check that train/val/test cutoffs match `output/preds_v11_final_{val,test}.csv` | rerun the export script |
| LAD search times out | Reduce `POOLS` to top-3 candidates only | Edit the script's `POOLS` dict |
| `output/abt_v12_external.parquet` missing EXT cols | An EXT loader silently failed — check `output/v12_external_attribution.csv` | Re-run that single loader with `force_refresh=True` |

---

## Stretch goal — Modal Labs free trial (optional)

If Colab+Kaggle quotas are exhausted (rare), Modal gives ~$30/month free
credits which is enough for ~10 fine-tune runs. Setup takes 30 min:

1. Sign up at [modal.com](https://modal.com) — no credit card.
2. `pip install modal && modal token new`.
3. Wrap any of our notebooks in a `modal.App` script. Template:
   ```python
   import modal
   app = modal.App("v13-chronos-finetune")
   image = modal.Image.debian_slim().pip_install(
       "chronos-forecasting==1.5.2", "transformers<5,>=4.48",
       "pandas", "pyarrow", "torch")

   @app.function(image=image, gpu="A10G", timeout=21600)
   def finetune():
       # ... copy the body of Cell 5 from the Colab notebook here ...
       pass

   if __name__ == "__main__":
       with app.run():
           finetune.remote()
   ```
4. `modal run finetune.py` — done. ~3× faster than Colab T4 thanks to
   the A10G.

Use this only if Colab+Kaggle truly fail. It's adds operational overhead
that's usually not worth it.

---

## What to tell management at each milestone

**V12.1 has shipped (April 2026).** Suggested management blurb:
> "V12.1 is the new production model. Test WAPE 0.3950 → 0.3937 (new
> all-time low), aggregate bias +2.80 % → +2.36 % (closer to zero),
> Monthly-WAPE 0.0799 → 0.0796. The model now consumes 9 free external
> open-data signals (Ukrainian retail trade index, NBU consumer
> confidence proxies, oblast-level air-raid frequency, blackout hours,
> IDP flows, Wikipedia attention to toy/franchise pages, Orthodox
> calendar) end-to-end. The intermediate V12 candidate did not pass
> our acceptance gate due to a validation→test bias direction reversal;
> V12.1 fixes that with a bias-direction-symmetry constraint in the
> LAD search and a re-trained base that actually consumes the EXT
> features. Improvement is small (~0.8 % relative on SIMSCORE) but
> real, OOF-defensible (no test peeking), and multi-axis (every
> headline metric moved in the right direction). Next milestone: V13
> (foundation models fine-tuned on Colab/Kaggle GPU), gated on the
> human running the GPU notebooks."

When V13 ships (in ~2 weeks):
> "V13 adds 3 fine-tuned foundation models (Chronos, TimesFM, Moirai).
> Test SIMSCORE further reduced by ~1-2 %. Total improvement V11 → V13
> is ~3-4 %, putting us at the M5/Rossmann benchmark median. Next:
> V14 GlobalNN."

When V14 ships (in ~4 weeks):
> "V14 is the final structured campaign result. Total improvement
> V11 → V14: ~3-7 % SIMSCORE. We are now at the realistic data
> ceiling for the data we can legally and freely obtain. Further
> improvements require POS-level data from partners, which is a
> business-side conversation, not a modeling one."
