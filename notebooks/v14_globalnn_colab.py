# ────────────────────────────────────────────────────────────────────────
# V14 GlobalNN — Transformer-encoder fine-tune (Google Colab Free, T4 GPU)
# Total time: ~3-4 hours wall-clock. Active human time: ~10 minutes.
#
# Trains a 192-dim Transformer-encoder with learned categorical
# embeddings (Партнер / Артикул / Бренд / Канал) on the V12-external
# ABT (V11 features + 32 EXT columns from open-data loaders). Outputs
# pinball-loss median forecasts for the held-out val + test windows.
#
# Architecture is defined in src/models/global_nn.py (paste it into
# Cell 4 below to keep this notebook self-contained).
#
# WHY a script and not an .ipynb? .ipynb JSON is brittle in version
# control. Paste cell-by-cell (separated by `# %% Cell N: ...` markers).
#
# Pre-flight: switch runtime to T4 GPU (Runtime → Change runtime type
# → Hardware accelerator: T4 GPU).
# ────────────────────────────────────────────────────────────────────────


# %% Cell 1: install + GPU check (~ 30 sec)
# %pip install --quiet torch>=2.0 numpy pandas pyarrow

import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert torch.cuda.is_available(), "Switch runtime to T4 GPU first!"


# %% Cell 2: mount Drive and locate the data (~ 5 sec)
from google.colab import drive
drive.mount("/content/drive")

DRIVE_DIR = "/content/drive/MyDrive/v14_globalnn_data"
import os, json
for f in ("train.parquet", "val.parquet", "test.parquet",
           "vocab.json", "manifest.json"):
    assert os.path.exists(f"{DRIVE_DIR}/{f}"), f"Upload {f} to {DRIVE_DIR}/"

with open(f"{DRIVE_DIR}/manifest.json") as fh:
    manifest = json.load(fh)
print(f"manifest: n_partners={manifest['n_partners']}  "
      f"n_skus={manifest['n_skus']}  n_brands={manifest['n_brands']}  "
      f"n_channels={manifest['n_channels']}")
print(f"split sizes: {manifest['split_sizes']}")


# %% Cell 3: load + split tensors (~ 1 min)
import pandas as pd
import numpy as np

with open(f"{DRIVE_DIR}/vocab.json") as fh:
    vocab = json.load(fh)

train_df = pd.read_parquet(f"{DRIVE_DIR}/train.parquet")
val_df   = pd.read_parquet(f"{DRIVE_DIR}/val.parquet")
test_df  = pd.read_parquet(f"{DRIVE_DIR}/test.parquet")
print(f"loaded: train={train_df.shape}  val={val_df.shape}  test={test_df.shape}")

# Categorical idx columns are already in the parquets as int32
# (suffix '_idx', e.g. 'Партнер_idx').
CAT_COLS = ["Партнер_idx", "Артикул_idx", "Бренд_idx", "Канал_idx"]
DROP_COLS = ["Период_str", "target_qty"] + CAT_COLS

NUMERIC_COLS = [c for c in train_df.columns if c not in DROP_COLS]
# Exclude any leftover object-dtype columns (shouldn't be any after export)
NUMERIC_COLS = [c for c in NUMERIC_COLS
                 if pd.api.types.is_numeric_dtype(train_df[c])]
print(f"n_numeric features: {len(NUMERIC_COLS)}")

# Impute missing numerics with 0 (most are lag-based and 0 is the
# correct semantics for pre-history rows).
for df in (train_df, val_df, test_df):
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0).astype(np.float32)

# z-score numeric cols using train stats
mu = train_df[NUMERIC_COLS].mean().values.astype(np.float32)
sd = train_df[NUMERIC_COLS].std().replace(0, 1).values.astype(np.float32)


def make_tensors(df):
    p = torch.tensor(df["Партнер_idx"].values, dtype=torch.long)
    s = torch.tensor(df["Артикул_idx"].values, dtype=torch.long)
    b = torch.tensor(df["Бренд_idx"].values,   dtype=torch.long)
    c = torch.tensor(df["Канал_idx"].values,   dtype=torch.long)
    n = (df[NUMERIC_COLS].values.astype(np.float32) - mu) / sd
    n = torch.tensor(n, dtype=torch.float32)
    y = torch.tensor(df["target_qty"].values.astype(np.float32),
                      dtype=torch.float32)
    return p, s, b, c, n, y

train_tensors = make_tensors(train_df)
val_tensors   = make_tensors(val_df)
test_tensors  = make_tensors(test_df)
print(f"train numeric tensor shape: {train_tensors[4].shape}")


# %% Cell 4: paste in src/models/global_nn.py (architecture) (~ 5 sec)
# Inline the architecture so this Colab doesn't need the repo. Same
# code as src/models/global_nn.py.

import torch.nn as nn
from dataclasses import dataclass


@dataclass
class GlobalNNConfig:
    n_partners: int
    n_skus: int
    n_brands: int
    n_channels: int
    n_numeric: int
    emb_dim: int = 32
    num_enc_dim: int = 64
    d_model: int = 192
    nhead: int = 8
    n_layers: int = 4
    dropout: float = 0.10
    quantiles: tuple = (0.1, 0.25, 0.5, 0.75, 0.9)


class GlobalNN(nn.Module):
    def __init__(self, cfg: GlobalNNConfig):
        super().__init__()
        self.cfg = cfg
        self.emb_partner = nn.Embedding(cfg.n_partners, cfg.emb_dim)
        self.emb_sku     = nn.Embedding(cfg.n_skus,     cfg.emb_dim)
        self.emb_brand   = nn.Embedding(cfg.n_brands,   cfg.emb_dim)
        self.emb_channel = nn.Embedding(cfg.n_channels, cfg.emb_dim)
        self.num_enc = nn.Sequential(
            nn.Linear(cfg.n_numeric, cfg.num_enc_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.num_enc_dim, cfg.num_enc_dim),
        )
        assert (4 * cfg.emb_dim + cfg.num_enc_dim) == cfg.d_model
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.q_head = nn.Linear(cfg.d_model, len(cfg.quantiles))

    def forward(self, partner_idx, sku_idx, brand_idx, channel_idx, numeric):
        ep = self.emb_partner(partner_idx)
        es = self.emb_sku(sku_idx)
        eb = self.emb_brand(brand_idx)
        ec = self.emb_channel(channel_idx)
        en = self.num_enc(numeric)
        x = torch.cat([ep, es, eb, ec, en], dim=-1)
        x = x.unsqueeze(1)
        x = self.encoder(x).squeeze(1)
        return self.q_head(x)


def pinball_loss(y_pred, y_true, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        diff = y_true - y_pred[:, i]
        losses.append(torch.maximum(q * diff, (q - 1) * diff))
    return torch.stack(losses, dim=-1).mean()


cfg = GlobalNNConfig(
    n_partners=manifest["n_partners"],
    n_skus=manifest["n_skus"],
    n_brands=manifest["n_brands"],
    n_channels=manifest["n_channels"],
    n_numeric=len(NUMERIC_COLS),
)
model = GlobalNN(cfg).cuda()
n_params = sum(p.numel() for p in model.parameters())
print(f"GlobalNN: {n_params/1e6:.2f}M params  "
      f"(d_model={cfg.d_model}, n_layers={cfg.n_layers})")


# %% Cell 5: train (~ 90-120 min on T4)
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

BATCH = 1024
EPOCHS = 15
LR = 5e-4
WD = 0.01

train_ds = TensorDataset(*train_tensors)
val_ds   = TensorDataset(*val_tensors)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                           num_workers=2, drop_last=True, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH * 4, shuffle=False,
                           num_workers=2, pin_memory=True)

opt = AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = CosineAnnealingLR(opt, T_max=EPOCHS * len(train_loader))

best_val_loss = float("inf")
ckpt_path = f"{DRIVE_DIR}/v14_globalnn_best.pt"

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for step, batch in enumerate(train_loader):
        p, s, b, c, n, y = [t.cuda(non_blocking=True) for t in batch]
        opt.zero_grad()
        pred = model(p, s, b, c, n)
        loss = pinball_loss(pred, y, cfg.quantiles)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        epoch_loss += loss.item()

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for batch in val_loader:
            p, s, b, c, n, y = [t.cuda(non_blocking=True) for t in batch]
            pred = model(p, s, b, c, n)
            val_loss += pinball_loss(pred, y, cfg.quantiles).item()
        val_loss /= len(val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "state_dict": model.state_dict(),
            "cfg":        cfg.__dict__,
            "epoch":      epoch,
            "val_loss":   val_loss,
            "mu":         mu, "sd": sd,
            "numeric_cols": NUMERIC_COLS,
        }, ckpt_path)
        flag = "★"
    else:
        flag = " "
    print(f"epoch {epoch+1:2d}/{EPOCHS}  "
          f"train_pinball={epoch_loss/len(train_loader):.4f}  "
          f"val_pinball={val_loss:.4f}  {flag}")

print(f"\nbest val loss = {best_val_loss:.4f}; checkpoint saved at {ckpt_path}")


# %% Cell 6: predict on val + test → CSVs (~ 5 min)
import pandas as pd

# Reload best checkpoint
ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
model.load_state_dict(ck["state_dict"])
model.eval()

QUANTILES = list(cfg.quantiles)
MEDIAN_IDX = QUANTILES.index(0.5)

def predict(tensors, df):
    p, s, b, c, n, y = [t.cuda() for t in tensors]
    preds = []
    with torch.no_grad():
        for i in range(0, len(p), BATCH * 4):
            sl = slice(i, i + BATCH * 4)
            out = model(p[sl], s[sl], b[sl], c[sl], n[sl])
            preds.append(out.cpu().float().numpy())
    pred_all = np.concatenate(preds, axis=0)  # N × Q
    out_df = df[["Период_str"]].rename(columns={"Период_str": "Период"}).copy()
    # Re-attach Партнер and Артикул from the source DF if available;
    # if not, use vocab inversion via the idx columns.
    inv_partner = {v: k for k, v in vocab["Партнер"].items()}
    inv_sku     = {v: k for k, v in vocab["Артикул"].items()}
    out_df["Партнер"] = df["Партнер_idx"].map(inv_partner).fillna("<UNK>")
    out_df["Артикул"] = df["Артикул_idx"].map(inv_sku).fillna("<UNK>")
    out_df["target_qty"] = df["target_qty"].values
    out_df["prediction"] = np.clip(pred_all[:, MEDIAN_IDX], 0, None)
    return out_df

val_pred  = predict(val_tensors,  val_df)
test_pred = predict(test_tensors, test_df)

val_pred.to_csv(f"{DRIVE_DIR}/preds_v14_globalnn_val.csv", index=False)
test_pred.to_csv(f"{DRIVE_DIR}/preds_v14_globalnn_test.csv", index=False)
print(f"wrote val ({len(val_pred):,} rows), test ({len(test_pred):,} rows)")


# %% Cell 7: quick sanity metrics (~ 5 sec)
def quick_wape(y, yhat):
    y = np.asarray(y, dtype=np.float64); yhat = np.asarray(yhat, dtype=np.float64)
    return float(np.abs(y - yhat).sum() / max(1e-9, y.sum()))

def quick_bias_pct(y, yhat):
    y = np.asarray(y, dtype=np.float64); yhat = np.asarray(yhat, dtype=np.float64)
    return float((yhat.sum() - y.sum()) / max(1e-9, y.sum()) * 100)

for name, df in (("val", val_pred), ("test", test_pred)):
    print(f"{name}: rows={len(df):,}  "
          f"WAPE={quick_wape(df.target_qty, df.prediction):.3f}  "
          f"bias%={quick_bias_pct(df.target_qty, df.prediction):+.1f}")

# Expected ranges (after 15 epochs):
#   val   WAPE 0.32-0.36 (V12.2 has 0.33 — hopefully NN matches or beats)
#   test  WAPE 0.38-0.42 (V12.2 has 0.39 — hopefully matches; if 0.36-0.38 we have a winner)
#
# If val WAPE > 0.45, training didn't converge — try LR=2e-4 + more epochs.


# %% Cell 8: download CSVs to local repo
# In Colab UI: drag-drop /content/drive/MyDrive/v14_globalnn_data/
#   preds_v14_globalnn_*.csv to ~/Downloads/
#
# Locally:
#   mv ~/Downloads/preds_v14_globalnn_*.csv \
#      ~/Desktop/business-process-modeling-demo/output/
#   cd ~/Desktop/business-process-modeling-demo
#   PYTHONPATH=. python -m scripts.v14_lad_stack --variant alpha
#
# scripts.v14_lad_stack will automatically merge the new V14 base
# into the V13 LAD pool and produce preds_v14_alpha_lad_{val,test}.csv
# + preds_v14_alpha_final_{val,test}.csv.
