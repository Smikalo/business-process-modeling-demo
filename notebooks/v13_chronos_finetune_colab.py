# ────────────────────────────────────────────────────────────────────────
# V13 Chronos-T5-Small FINE-TUNE (Google Colab Free, T4 GPU)
# Total time: ~2 hours wall-clock. Active human time: ~10 minutes.
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


# %% Cell 1: install Chronos and pin transformers (~ 1 min)
# IMPORTANT: chronos-forecasting==1.5.2 requires transformers<5,>=4.48 .
# Colab Free ships transformers 5.x by default, which is INCOMPATIBLE.
# Strategy: install Chronos which pulls in compatible transformers.
# If you see transformers 5.x after install, DO Runtime → Restart session
# and re-run only Cell 1.

# %pip install --quiet chronos-forecasting==1.5.2 "transformers>=4.48,<5" \
#   accelerate==0.34.2 datasets==3.0.1 peft==0.13.2

import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert torch.cuda.is_available(), "Switch runtime to T4 GPU first!"
assert transformers.__version__.startswith(("4.4", "4.5")), \
    f"Need transformers 4.48-4.x, got {transformers.__version__} — restart session."


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
# Chronos's pipeline.tokenizer.context_input_transform expects a
# TENSOR of shape (B, ctx_len). It returns (token_ids, attention_mask,
# scale). For training we tokenise context AND future jointly so the
# T5 model learns to predict the future quantile tokens given context
# tokens.
#
# Returns: list of {input_ids: 1D tensor, labels: 1D tensor}.

def tokenise_window(ctx_arr, tgt_arr):
    """Tokenise a single (ctx, tgt) pair. Returns dict with input_ids
    (encoder input = ctx tokens) and labels (decoder target = tgt
    tokens). The Chronos T5 is trained as encoder→decoder."""
    ctx_t = torch.tensor(ctx_arr).unsqueeze(0)   # (1, ctx_len)
    tgt_t = torch.tensor(tgt_arr).unsqueeze(0)   # (1, hor)
    # Chronos's tokenizer scales each series by its abs-mean before
    # quantising to 4096 token bins.
    ctx_ids, ctx_mask, scale = pipe.tokenizer.context_input_transform(ctx_t)
    # For the decoder labels, transform tgt with the SAME scale so the
    # decoder learns to predict tokens consistent with the context's scale.
    tgt_ids, tgt_mask = pipe.tokenizer.label_input_transform(tgt_t, scale)
    return {
        "input_ids": ctx_ids.squeeze(0).long(),
        "attention_mask": ctx_mask.squeeze(0).long(),
        "labels": tgt_ids.squeeze(0).long(),
    }

samples = []
for w in raw_windows:
    try:
        s = tokenise_window(w["ctx"], w["tgt"])
        samples.append(s)
    except Exception:
        pass  # skip malformed
print(f"tokenised {len(samples)} samples")
print(f"  example input_ids shape: {samples[0]['input_ids'].shape}")
print(f"  example labels shape:    {samples[0]['labels'].shape}")
print(f"  sanity: token range "
      f"[{min(s['input_ids'].min().item() for s in samples[:100])}, "
      f"{max(s['input_ids'].max().item() for s in samples[:100])}]")


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


# %% Cell 5d: re-load fine-tuned model into Chronos pipeline (~ 30 sec)
# Trick: the fine-tune mutated the underlying T5 weights via LoRA's
# adapter. Pipe.model.model now IS the fine-tuned model since we
# passed it directly to get_peft_model. To use Chronos's pipe.predict()
# we need to restore the LoRA-merged weights into pipe.

from peft import PeftModel

# Reload base T5
import copy
base_clean = copy.deepcopy(pipe.model.model)
peft_model = PeftModel.from_pretrained(base_clean,
                                          f"{DRIVE_DIR}/chronos_lora_adapter")
merged = peft_model.merge_and_unload()
pipe.model.model = merged.eval()
print("LoRA weights merged into pipe.model.model")


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
