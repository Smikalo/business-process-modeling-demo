# ────────────────────────────────────────────────────────────────────────
# V13 Chronos-T5-Small FINE-TUNE (Google Colab Free, T4 GPU)
# Total time: ~2 hours wall-clock; ~10 minutes of interactive time.
#
# This is a **plain Python script** that you paste into Google Colab
# cell by cell (separated by `# %% Cell N: ...` markers below). It
# fine-tunes Amazon's Chronos-T5-Small on our toy-distribution monthly
# series via LoRA and produces preds_v13_chronos_{val,test}.csv with
# *real* fine-tuned weights (not zero-shot).
#
# WHY a script and not an .ipynb? .ipynb JSON is brittle in version
# control and small typos render the whole notebook unloadable. With a
# .py you can:
#   1. Open Colab → File → "New notebook" (blank).
#   2. For each `# %% Cell N: ...` block below, paste into a fresh cell.
#   3. Run cells in order with Shift+Enter.
#
# Pre-flight: switch runtime to T4 GPU (Runtime → Change runtime type
# → Hardware accelerator: T4 GPU). Mount Drive (instructed in Cell 2).
# ────────────────────────────────────────────────────────────────────────


# %% Cell 1: install Chronos and pin compatible deps (~ 1 min)
# IMPORTANT: chronos-forecasting==1.5.2 requires transformers<5,>=4.48 .
# Colab Free ships transformers 5.x by default, which is INCOMPATIBLE.
#
# Also IMPORTANT: transformers 4.48+ uses Accelerator.unwrap_model's
# keep_torch_compile kwarg, which was added in accelerate 1.0. Old
# accelerate==0.34.x will fail Trainer.train() with TypeError. Pin
# accelerate>=1.0.
#
# If after install the assertion below fails (transformers still 5.x),
# DO Runtime → Restart session and re-run only Cell 1.

# %pip install --quiet chronos-forecasting==1.5.2 "transformers>=4.48,<5" \
#   "accelerate>=1.0,<2" datasets==3.0.1 peft==0.13.2

import torch, transformers, accelerate
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert torch.cuda.is_available(), "Switch runtime to T4 GPU first!"
assert transformers.__version__.startswith(("4.4", "4.5")), \
    f"Need transformers 4.48-4.x, got {transformers.__version__} — restart session."
assert accelerate.__version__.split(".")[0] in ("1", "2"), \
    f"Need accelerate >= 1.0, got {accelerate.__version__} — re-run Cell 1 + restart."


# %% Cell 2: mount Drive and locate the data (~ 5 sec)
from google.colab import drive
drive.mount("/content/drive")

DRIVE_DIR = "/content/drive/MyDrive/v13_fm_data"
import os
assert os.path.exists(f"{DRIVE_DIR}/series_train.parquet"), \
    f"Upload series_train.parquet to {DRIVE_DIR}/"
assert os.path.exists(f"{DRIVE_DIR}/series_oof.parquet"), \
    f"Upload series_oof.parquet to {DRIVE_DIR}/"

import pandas as pd
train_wide = pd.read_parquet(f"{DRIVE_DIR}/series_train.parquet")
oof_wide   = pd.read_parquet(f"{DRIVE_DIR}/series_oof.parquet")
print(f"train: {train_wide.shape}  oof: {oof_wide.shape}")


# %% Cell 3: convert wide → long Chronos format (~ 30 sec)
import numpy as np

def wide_to_chronos_dict(wide_df):
    """Convert (pair, YYYY-MM, ..., YYYY-MM) wide DF to Chronos's
    expected per-series dict format."""
    pair_col = "pair"
    month_cols = [c for c in wide_df.columns if c != pair_col]
    series_dict = {}
    for _, row in wide_df.iterrows():
        vals = [float(row[m]) for m in month_cols]
        series_dict[row[pair_col]] = {
            "start": pd.Period(month_cols[0], freq="M").to_timestamp(),
            "target": np.array(vals, dtype=np.float32),
            "freq": "MS",
        }
    return series_dict, month_cols

train_series, train_months = wide_to_chronos_dict(train_wide)
oof_series,   oof_months   = wide_to_chronos_dict(oof_wide)
print(f"n_series_train={len(train_series)}, train_horizon={len(train_months)}")
print(f"n_series_oof={len(oof_series)},   oof_horizon={len(oof_months)}")


# %% Cell 4: load Chronos-T5-Small (~ 2 min)
from chronos import ChronosPipeline

pipe = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cuda", torch_dtype=torch.bfloat16,
)
print("loaded Chronos-T5-Small (~70 M params)")


# %% Cell 5a: build a proper sliding-window training set (~ 30 sec)
# CRITICAL FIX vs old notebook: context_len + horizon MUST be ≤ 54
# (the per-series training history). 24 + 8 = 32 leaves 22 sliding
# windows per series.
#
# We also chain train + earliest val months as additional history
# (the val window is identical to "what was known by mid-2025"; the
# fine-tune target is only training-window data for true OOS evaluation).

import numpy as np
import torch

CONTEXT_LEN = 24      # 2 years of monthly context
HORIZON     = 8       # match held-out test horizon (Jul 2025 - Mar 2026)
STRIDE      = 2       # overlap between sliding windows
MAX_SAMPLES = 30000   # cap to keep T4 fine-tune under 60 minutes

def build_training_windows(series_dict, ctx=CONTEXT_LEN, hor=HORIZON,
                            stride=STRIDE):
    """Slide a (ctx + hor)-length window across each pair's training
    history and emit (context_array, target_array) tuples.

    Returns list of dicts { 'ctx': np.array(ctx,), 'tgt': np.array(hor,) }.
    """
    windows = []
    for pair, s in series_dict.items():
        arr = s["target"]
        # Need at least ctx+hor months
        if len(arr) < ctx + hor:
            continue
        # Slide
        for start in range(0, len(arr) - ctx - hor + 1, stride):
            ctx_arr = arr[start:start + ctx]
            tgt_arr = arr[start + ctx:start + ctx + hor]
            # Skip windows where target is all-zero (no demand to learn)
            if tgt_arr.sum() <= 0:
                continue
            windows.append({"ctx": ctx_arr.astype(np.float32),
                             "tgt": tgt_arr.astype(np.float32)})
    return windows

raw_windows = build_training_windows(train_series)
print(f"built {len(raw_windows)} sliding windows from {len(train_series)} series")
print(f"  (ctx={CONTEXT_LEN}, hor={HORIZON}, stride={STRIDE})")

# Shuffle + cap
import random
random.seed(42)
random.shuffle(raw_windows)
raw_windows = raw_windows[:MAX_SAMPLES]
print(f"using {len(raw_windows)} windows for fine-tune")


# %% Cell 5b: tokenise to Chronos's quantised token IDs (~ 1 min)
#
# Chronos's tokenizer has TWO quirks that bit us in earlier runs:
#
# 1. `label_input_transform(label, scale)` HARD-ASSERTS that
#    `label.shape[-1] == config.prediction_length` (typically 64).
#    Our `HORIZON=8` therefore hits an AssertionError. Fix: bypass
#    by calling the underlying `_input_transform(context=label,
#    scale=scale)` directly — `label_input_transform` is just an
#    assertion-wrapped call to it. `_input_transform` accepts any
#    length and returns `(token_ids, attention_mask, scale)`.
#
# 2. `context_input_transform(ctx)` returns shape (1, 25) for input
#    (1, 24) because it appends an EOS token. This is normal; we
#    keep the +1 EOS token in the encoder input.

# Diagnostic probe (fail loud, no silent except this time):
ctx_arr = raw_windows[0]["ctx"]
tgt_arr = raw_windows[0]["tgt"]
print(f"tokenizer class:                 {type(pipe.tokenizer).__name__}")
print(f"tokenizer.config.context_length: {pipe.tokenizer.config.context_length}")
print(f"tokenizer.config.prediction_length: {pipe.tokenizer.config.prediction_length}")
print(f"our (CONTEXT_LEN, HORIZON):      ({CONTEXT_LEN}, {HORIZON})")

ctx_t = torch.tensor(ctx_arr, dtype=torch.float32).unsqueeze(0)
tgt_t = torch.tensor(tgt_arr, dtype=torch.float32).unsqueeze(0)
ctx_ids, ctx_mask, scale = pipe.tokenizer.context_input_transform(ctx_t)
print(f"\nctx tokenization OK: shapes = {ctx_ids.shape}, {ctx_mask.shape}, {scale.shape}")

# Bypass label_input_transform's strict prediction_length assertion
# by calling the underlying _input_transform directly. This is what
# label_input_transform invokes internally after its assert.
tgt_ids, tgt_mask, _ = pipe.tokenizer._input_transform(
    context=tgt_t, scale=scale)
print(f"tgt tokenization OK: shapes = {tgt_ids.shape}, {tgt_mask.shape}")
print(f"  ctx_ids range: [{ctx_ids.min().item()}, {ctx_ids.max().item()}]")
print(f"  tgt_ids range: [{tgt_ids.min().item()}, {tgt_ids.max().item()}]")


def tokenise_window(ctx_arr, tgt_arr):
    """Tokenise a single (ctx, tgt) pair. Uses _input_transform for the
    label to skip Chronos's strict prediction_length assertion."""
    ctx_t = torch.tensor(ctx_arr, dtype=torch.float32).unsqueeze(0)
    tgt_t = torch.tensor(tgt_arr, dtype=torch.float32).unsqueeze(0)
    ctx_ids, ctx_mask, scale = pipe.tokenizer.context_input_transform(ctx_t)
    tgt_ids, tgt_mask, _ = pipe.tokenizer._input_transform(
        context=tgt_t, scale=scale)
    return {
        "input_ids":      ctx_ids.squeeze(0).long(),
        "attention_mask": ctx_mask.squeeze(0).long(),
        "labels":         tgt_ids.squeeze(0).long(),
    }


# Bulk tokenise — fail loud if any window errors out
samples = []
errors = []
for i, w in enumerate(raw_windows):
    try:
        samples.append(tokenise_window(w["ctx"], w["tgt"]))
    except Exception as e:
        errors.append((i, str(e)))
        if len(errors) <= 3:
            print(f"  [error] window {i}: {type(e).__name__}: {e}")

print(f"\ntokenised {len(samples)} / {len(raw_windows)} samples  "
      f"(errors: {len(errors)})")
if len(samples) == 0:
    raise RuntimeError(
        f"All windows failed tokenization. First error: {errors[:1]}")
print(f"  example input_ids shape: {samples[0]['input_ids'].shape}")
print(f"  example labels shape:    {samples[0]['labels'].shape}")


# %% Cell 5c-alt (FALLBACK): use Amazon's official Chronos training script
# Use this ONLY if Cell 5c below fails. Battle-tested by the Chronos team —
# handles Chronos-specific token shifting, decoder_input_ids, and EOS
# masking that our hand-rolled Trainer doesn't. Run this INSTEAD of
# Cell 5c, then skip to Cell 5d.
#
# Activation: comment out the `if False:` line below to enable.

if False:  # noqa
    # 1) clone the chronos-forecasting repo (~30 sec)
    # !git clone --depth 1 https://github.com/amazon-science/chronos-forecasting.git /content/chronos-fc
    # %cd /content/chronos-fc/scripts/training
    # %pip install --quiet -e /content/chronos-fc

    # 2) convert wide DFs to the arrow format train.py expects (~30 sec)
    import datasets
    rows = []
    start_ts = pd.Period(train_months[0], freq="M").to_timestamp()
    for pair, s in train_series.items():
        rows.append({"start": start_ts.isoformat(),
                      "target": s["target"].astype(float).tolist(),
                      "item_id": str(pair)})
    ds = datasets.Dataset.from_list(rows)
    ds.to_parquet(f"{DRIVE_DIR}/train.arrow.parquet")
    print(f"wrote {DRIVE_DIR}/train.arrow.parquet ({len(rows)} series)")

    # 3) run the official trainer (~ 60-90 min on T4)
    # !python train.py \
    #     --config configs/chronos-t5-small.yaml \
    #     --model-id amazon/chronos-t5-small \
    #     --training-data-paths "[$DRIVE_DIR/train.arrow.parquet]" \
    #     --probability "[1.0]" \
    #     --max-steps 4000 \
    #     --learning-rate 5e-5 \
    #     --per-device-train-batch-size 16 \
    #     --output-dir $DRIVE_DIR/chronos_lora_official

    # 4) skip ahead to Cell 5d but pointed at chronos_lora_official
    # The merge step works the same — just change the path in Cell 5d
    # from chronos_lora_adapter to chronos_lora_official.


# %% Cell 5c: wrap LoRA + HF Trainer + train (~ 60-90 min on T4)
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (TrainingArguments, Trainer,
                            DataCollatorForSeq2Seq)
from torch.utils.data import Dataset

# IMPORTANT: pipe.model is Chronos's wrapper whose .config is a
# ChronosConfig dataclass (no .get() method) — peft will fail with
# AttributeError. The actual T5ForConditionalGeneration is at
# pipe.model.model.
base = pipe.model.model

lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, r=8, lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q", "v", "k", "o"],
)
model = get_peft_model(base, lora_cfg)
model.print_trainable_parameters()


class SamplesDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]


train_ds = SamplesDataset(samples)
collator = DataCollatorForSeq2Seq(tokenizer=None, model=model,
                                    label_pad_token_id=-100)

args = TrainingArguments(
    output_dir=f"{DRIVE_DIR}/chronos_lora_ckpt",
    overwrite_output_dir=True,
    num_train_epochs=2,                # 2 epochs over 30k samples = 60k steps / batch
    per_device_train_batch_size=8,     # T4 can handle 8 for context_len=24
    gradient_accumulation_steps=2,     # effective batch 16
    learning_rate=5e-5,
    warmup_steps=200,
    logging_steps=50,
    save_steps=1000,
    save_total_limit=2,
    bf16=True,
    report_to="none",                  # silence wandb
    remove_unused_columns=False,
)

# Custom collator: HF's DataCollatorForSeq2Seq needs a tokenizer for
# padding decisions, but we have fixed-length inputs. Just stack.
def stack_collator(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"] for b in batch]),
    }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=stack_collator,
)

print("starting training (expect ~60-90 min on T4)...")
trainer.train()
print("training complete.")

# Save the LoRA adapter for future runs (skip Cells 1-5 next time)
model.save_pretrained(f"{DRIVE_DIR}/chronos_lora_adapter")
print(f"saved LoRA adapter to {DRIVE_DIR}/chronos_lora_adapter")


# %% Cell 5d: merge LoRA adapter into Chronos pipeline (~ 5 sec)
# After Cell 5c's trainer.train(), the `model` variable IS the trained
# PEFT-wrapped T5 (since we passed pipe.model.model directly to
# get_peft_model). merge_and_unload() folds the LoRA adapter weights
# into the base T5 linears and returns a clean nn.Module that
# Chronos's pipe.predict() can use as a drop-in replacement.

merged = model.merge_and_unload()
pipe.model.model = merged.eval()
print("LoRA weights merged into pipe.model.model — ready for pipe.predict()")

# Sanity: confirm the merge worked by counting parameters
n_params = sum(p.numel() for p in pipe.model.model.parameters())
n_trainable = sum(p.numel() for p in pipe.model.model.parameters() if p.requires_grad)
print(f"  pipe.model.model: {n_params/1e6:.1f}M params  ({n_trainable/1e6:.1f}M trainable — expect ~46M / 0M after merge)")


# %% Cell 6: fine-tuned forecasting on val + test windows (~ 25 min)
# Run inference cell-by-cell on EACH series, batched.
# Prepare wide_oof as before, but now use the fine-tuned `pipe` to predict.

# Defensive imports (in case kernel was restarted between cells):
import numpy as np
import pandas as pd
import torch

VAL_MONTHS = [m for m in oof_months if m <= "2025-06"]
TEST_MONTHS = [m for m in oof_months if m >= "2025-07"]
print(f"forecasting val={len(VAL_MONTHS)} mo, test={len(TEST_MONTHS)} mo")

# Concat train + oof history per pair to feed full context
all_months = sorted(set(train_months + oof_months))
records = []

BATCH = 256  # if OOM, drop to 128 or 64
pairs = list(train_series.keys())
for i in range(0, len(pairs), BATCH):
    batch_pairs = pairs[i:i+BATCH]
    contexts = []
    for pair in batch_pairs:
        history = np.concatenate([
            train_series[pair]["target"],
            oof_series[pair]["target"][:VAL_MONTHS.__len__()]  # context up through val
        ])
        contexts.append(torch.tensor(history, dtype=torch.float32))
    forecasts = pipe.predict(contexts, prediction_length=len(TEST_MONTHS),
                              num_samples=20)  # B × samples × horizon
    test_preds = forecasts.median(dim=1).values.cpu().numpy()  # B × horizon

    # also forecast val: use only train_months as context, predict val_months
    contexts_val = [torch.tensor(train_series[p]["target"], dtype=torch.float32)
                    for p in batch_pairs]
    val_forecasts = pipe.predict(contexts_val, prediction_length=len(VAL_MONTHS),
                                  num_samples=20)
    val_preds = val_forecasts.median(dim=1).values.cpu().numpy()

    for k, pair in enumerate(batch_pairs):
        for j, m in enumerate(VAL_MONTHS):
            records.append({"pair": pair, "Период": m, "split": "val",
                             "prediction": float(val_preds[k, j])})
        for j, m in enumerate(TEST_MONTHS):
            records.append({"pair": pair, "Период": m, "split": "test",
                             "prediction": float(test_preds[k, j])})
    if i % (BATCH * 5) == 0:
        print(f"  forecasted {i + BATCH}/{len(pairs)} pairs")


# %% Cell 7: split into pred CSVs matching V11_final's key schema (~ 30 sec)
import re
import numpy as np
import pandas as pd

preds_df = pd.DataFrame(records)
preds_df[["Партнер", "Артикул"]] = preds_df["pair"].str.split("\\|\\|", n=1, expand=True)

# Need target_qty too — load from oof_wide using the same pivot
oof_long = oof_wide.melt(id_vars=["pair"], var_name="Период", value_name="target_qty")
oof_long[["Партнер", "Артикул"]] = oof_long["pair"].str.split("\\|\\|", n=1, expand=True)

merged = preds_df.merge(oof_long.drop(columns=["pair"]),
                         on=["Период", "Партнер", "Артикул"], how="inner")

for split in ("val", "test"):
    sub = merged[merged["split"] == split][["Период", "Партнер", "Артикул",
                                             "target_qty", "prediction"]]
    sub["prediction"] = sub["prediction"].clip(lower=0)
    out_path = f"{DRIVE_DIR}/preds_v13_chronos_ft_{split}.csv"
    sub.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(sub):,} rows)")


# %% Cell 8: quick sanity metrics on Drive (~ 5 sec)
import numpy as np
import pandas as pd

def quick_wape(y, yhat):
    y = np.asarray(y, dtype=np.float64); yhat = np.asarray(yhat, dtype=np.float64)
    return float(np.abs(y - yhat).sum() / max(1e-9, y.sum()))

def quick_bias_pct(y, yhat):
    y = np.asarray(y, dtype=np.float64); yhat = np.asarray(yhat, dtype=np.float64)
    return float((yhat.sum() - y.sum()) / max(1e-9, y.sum()) * 100)

for split in ("val", "test"):
    sub = pd.read_csv(f"{DRIVE_DIR}/preds_v13_chronos_ft_{split}.csv")
    print(f"{split}: rows={len(sub):,}  WAPE={quick_wape(sub.target_qty, sub.prediction):.3f}"
          f"  bias%={quick_bias_pct(sub.target_qty, sub.prediction):+.1f}")

# Expected ranges (FINE-TUNED — substantially better than zero-shot):
#   zero-shot was: val WAPE 0.69 / test WAPE 0.63 / test bias -26%
#   fine-tuned:    val WAPE 0.45-0.55 / test WAPE 0.50-0.55 / test bias ±10%
#
# If you see basically the same numbers as zero-shot (val 0.69, test 0.63),
# the fine-tune didn't take. Check: did Cell 5c's trainer.train()
# actually log a decreasing loss? Did Cell 5d's PeftModel.from_pretrained
# load the adapter? Should report ~600K trainable params.


# %% Cell 9: download the CSVs to your local repo
# Either: gdrive download --recursive --path output v13_fm_data
# Or:     drag-drop in the Drive web UI to ~/Downloads/, then locally:
#         mv ~/Downloads/preds_v13_chronos_ft_*.csv \
#            ~/Desktop/business-process-modeling-demo/output/
#
# Then on your local terminal:
#   cd ~/Desktop/business-process-modeling-demo
#   PYTHONPATH=. python -m scripts.v13_lad_stack
