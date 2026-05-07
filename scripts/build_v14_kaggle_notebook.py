"""Build the V14 GlobalNN Kaggle .ipynb from the Colab paste-script.

Adapts paths (Drive → /kaggle/input + /kaggle/working) and writes a
proper nbformat .ipynb to output/v14_kaggle_kernel/v14_globalnn.ipynb.

Run: PYTHONPATH=. python -m scripts.build_v14_kaggle_notebook
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEST = REPO / "output" / "v14_kaggle_kernel" / "v14_globalnn.ipynb"
DATA_DIR = "/kaggle/input/bpm-v14-globalnn-data"
OUT_DIR = "/kaggle/working"


CELLS_SOURCE: list[tuple[str, str]] = [
    ("md",
     "# V14 GlobalNN — Kaggle T4×2 / P100 fine-tune\n"
     "\n"
     "Transformer-encoder with learned categorical embeddings for "
     "Партнер/Артикул/Бренд/Канал. 192-dim, 4 layers, 8 heads, "
     "5-quantile pinball head. Trains in one Kaggle session "
     "(~90 min on T4 ×2).\n"
     "\n"
     "**Output:** `preds_v14_globalnn_{val,test}.csv` to `/kaggle/working/`. "
     "Pull via `kaggle kernels output <user>/bpm-v14-globalnn` (use your Kaggle handle)."),

    ("code",
     "# Cell 1: GPU compatibility detect + force-install torch BEFORE any import\n"
     "# This cell does NOT import torch — that happens in Cell 1b after install.\n"
     "# Reason: torch's C extensions can't be cleanly reloaded mid-process\n"
     "# (RuntimeError: function '_has_torch_function' already has a docstring).\n"
     "import subprocess, sys, os\n"
     "\n"
     "# Quick way to check GPU type without importing torch: nvidia-smi.\n"
     "try:\n"
     "    smi = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap',\n"
     "                                     '--format=csv,noheader']).decode().strip()\n"
     "    print('nvidia-smi:', smi)\n"
     "    name, cap = [s.strip() for s in smi.split(',')]\n"
     "    cap_major = int(cap.split('.')[0]) if '.' in cap else 7\n"
     "except Exception as e:\n"
     "    print(f'could not query GPU: {e} — assuming compatible')\n"
     "    cap_major = 7\n"
     "\n"
     "if cap_major < 7:\n"
     "    print(f'GPU sm_{cap}: incompatible with default torch — installing 2.5.1+cu121...')\n"
     "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet',\n"
     "        '--force-reinstall', '--no-deps',\n"
     "        'torch==2.5.1', 'torchvision==0.20.1', 'torchaudio==2.5.1',\n"
     "        '--index-url', 'https://download.pytorch.org/whl/cu121'])\n"
     "    print('✓ torch 2.5.1 installed (will be loaded fresh in Cell 1b)')\n"
     "else:\n"
     "    print(f'GPU sm_{cap}: compatible — keeping default torch')"),

    ("code",
     "# Cell 1b: NOW import torch (first time in this Python process)\n"
     "# Even if Cell 1 reinstalled, this is the first 'import torch' so we\n"
     "# get the fresh version from disk.\n"
     "import torch\n"
     "import numpy as np, pandas as pd, json, os\n"
     "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\n"
     "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')\n"
     "if torch.cuda.is_available():\n"
     "    cap = torch.cuda.get_device_capability(0)\n"
     "    print(f'GPU capability: sm_{cap[0]}{cap[1]}')\n"
     "    # Smoke test: tiny CUDA op\n"
     "    x = torch.zeros(2, device='cuda'); y = x + 1\n"
     "    print(f'CUDA smoke test OK: y={y.tolist()}')\n"
     "assert torch.cuda.is_available(), 'GPU required'"),

    ("code",
     "# Cell 2: locate the data via recursive search\n"
     "# Kaggle mounts attached datasets at varying paths:\n"
     "#   /kaggle/input/<slug>/                       (public datasets)\n"
     "#   /kaggle/input/datasets/<owner>/<slug>/      (private datasets via API)\n"
     "# Use recursive glob to find train.parquet wherever it landed.\n"
     "import glob\n"
     f"OUT_DIR  = '{OUT_DIR}'\n"
     "\n"
     "# First print the tree for diagnostic visibility\n"
     "print('/kaggle/input/ tree (first 3 levels):')\n"
     "for root, dirs, files in os.walk('/kaggle/input', topdown=True):\n"
     "    depth = root.count(os.sep) - '/kaggle/input'.count(os.sep)\n"
     "    if depth > 3: dirs.clear(); continue\n"
     "    indent = '  ' * depth\n"
     "    print(f'{indent}{os.path.basename(root) or root}/')\n"
     "    for f in files[:5]:\n"
     "        print(f'{indent}  {f}')\n"
     "    if len(files) > 5:\n"
     "        print(f'{indent}  ... ({len(files)-5} more files)')\n"
     "\n"
     "# Recursive search for train.parquet — use that as the anchor.\n"
     "matches = glob.glob('/kaggle/input/**/train.parquet', recursive=True)\n"
     "print(f'\\nfound train.parquet at: {matches}')\n"
     "if not matches:\n"
     "    raise RuntimeError('train.parquet not found anywhere in /kaggle/input/')\n"
     "DATA_DIR = os.path.dirname(matches[0])\n"
     "print(f'→ using DATA_DIR = {DATA_DIR}')\n"
     "\n"
     "for f in ('train.parquet', 'val.parquet', 'test.parquet',\n"
     "          'vocab.json', 'manifest.json'):\n"
     "    assert os.path.exists(f'{DATA_DIR}/{f}'), f'missing {f} in {DATA_DIR}'\n"
     "\n"
     "with open(f'{DATA_DIR}/manifest.json') as fh:\n"
     "    manifest = json.load(fh)\n"
     "with open(f'{DATA_DIR}/vocab.json') as fh:\n"
     "    vocab = json.load(fh)\n"
     "print('manifest:', manifest['n_partners'], 'partners,',\n"
     "      manifest['n_skus'], 'skus,',\n"
     "      manifest['n_brands'], 'brands,',\n"
     "      manifest['n_channels'], 'channels')"),

    ("code",
     "# Cell 3: load + scrub + tensorise\n"
     "train_df = pd.read_parquet(f'{DATA_DIR}/train.parquet')\n"
     "val_df   = pd.read_parquet(f'{DATA_DIR}/val.parquet')\n"
     "test_df  = pd.read_parquet(f'{DATA_DIR}/test.parquet')\n"
     "print('loaded:', train_df.shape, val_df.shape, test_df.shape)\n"
     "\n"
     "CAT_COLS = ['Партнер_idx', 'Артикул_idx', 'Бренд_idx', 'Канал_idx']\n"
     "DROP_COLS = ['Период_str', 'target_qty'] + CAT_COLS\n"
     "NUMERIC_COLS = [c for c in train_df.columns if c not in DROP_COLS\n"
     "                 and pd.api.types.is_numeric_dtype(train_df[c])]\n"
     "print('n_numeric:', len(NUMERIC_COLS))\n"
     "\n"
     "# Robust scrub: catch BOTH NaN and Inf (upstream feature engineering\n"
     "# can produce Inf via division-by-zero in ratio columns)\n"
     "for df in (train_df, val_df, test_df):\n"
     "    arr = df[NUMERIC_COLS].values.astype(np.float32)\n"
     "    arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)\n"
     "    df[NUMERIC_COLS] = arr\n"
     "for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:\n"
     "    bad = df['target_qty'].isna() | np.isinf(df['target_qty'])\n"
     "    if bad.any():\n"
     "        df.drop(df.index[bad], inplace=True)\n"
     "        print(f'  [scrub] {split_name}: dropped {bad.sum()} bad target rows')\n"
     "\n"
     "mu = train_df[NUMERIC_COLS].mean().values.astype(np.float32)\n"
     "sd = train_df[NUMERIC_COLS].std().fillna(1.0).replace(0, 1).values.astype(np.float32)\n"
     "mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)\n"
     "sd = np.nan_to_num(sd, nan=1.0, posinf=1.0, neginf=1.0)\n"
     "print(f'  mu range [{mu.min():.3g}, {mu.max():.3g}]  sd range [{sd.min():.3g}, {sd.max():.3g}]')\n"
     "\n"
     "def make_tensors(df):\n"
     "    p = torch.tensor(df['Партнер_idx'].values, dtype=torch.long)\n"
     "    s = torch.tensor(df['Артикул_idx'].values, dtype=torch.long)\n"
     "    b = torch.tensor(df['Бренд_idx'].values,   dtype=torch.long)\n"
     "    c = torch.tensor(df['Канал_idx'].values,   dtype=torch.long)\n"
     "    n = (df[NUMERIC_COLS].values.astype(np.float32) - mu) / sd\n"
     "    n = np.nan_to_num(n, nan=0.0, posinf=10.0, neginf=-10.0)\n"
     "    n = np.clip(n, -10.0, 10.0)\n"
     "    n = torch.tensor(n, dtype=torch.float32)\n"
     "    y = torch.tensor(df['target_qty'].values.astype(np.float32), dtype=torch.float32)\n"
     "    return p, s, b, c, n, y\n"
     "\n"
     "train_tensors = make_tensors(train_df)\n"
     "val_tensors   = make_tensors(val_df)\n"
     "test_tensors  = make_tensors(test_df)\n"
     "for name, t in [('train_n', train_tensors[4]), ('train_y', train_tensors[5]),\n"
     "                ('val_n', val_tensors[4]),     ('test_n', test_tensors[4])]:\n"
     "    assert torch.isfinite(t).all(), f'NaN/Inf in {name}!'\n"
     "print('✓ all tensors finite')"),

    ("code",
     "# Cell 4: GlobalNN architecture (192-dim Transformer-encoder)\n"
     "import torch.nn as nn\n"
     "from dataclasses import dataclass\n"
     "\n"
     "@dataclass\n"
     "class Cfg:\n"
     "    n_partners: int; n_skus: int; n_brands: int; n_channels: int; n_numeric: int\n"
     "    emb_dim: int = 32\n"
     "    num_enc_dim: int = 64\n"
     "    d_model: int = 192\n"
     "    nhead: int = 8\n"
     "    n_layers: int = 4\n"
     "    dropout: float = 0.10\n"
     "    quantiles: tuple = (0.1, 0.25, 0.5, 0.75, 0.9)\n"
     "\n"
     "class GlobalNN(nn.Module):\n"
     "    def __init__(self, cfg):\n"
     "        super().__init__(); self.cfg = cfg\n"
     "        self.emb_partner = nn.Embedding(cfg.n_partners, cfg.emb_dim)\n"
     "        self.emb_sku     = nn.Embedding(cfg.n_skus,     cfg.emb_dim)\n"
     "        self.emb_brand   = nn.Embedding(cfg.n_brands,   cfg.emb_dim)\n"
     "        self.emb_channel = nn.Embedding(cfg.n_channels, cfg.emb_dim)\n"
     "        self.num_enc = nn.Sequential(\n"
     "            nn.Linear(cfg.n_numeric, cfg.num_enc_dim),\n"
     "            nn.GELU(), nn.Dropout(cfg.dropout),\n"
     "            nn.Linear(cfg.num_enc_dim, cfg.num_enc_dim))\n"
     "        assert (4*cfg.emb_dim + cfg.num_enc_dim) == cfg.d_model\n"
     "        enc_layer = nn.TransformerEncoderLayer(\n"
     "            d_model=cfg.d_model, nhead=cfg.nhead,\n"
     "            dim_feedforward=cfg.d_model*4,\n"
     "            dropout=cfg.dropout, batch_first=True)\n"
     "        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)\n"
     "        self.q_head = nn.Linear(cfg.d_model, len(cfg.quantiles))\n"
     "    def forward(self, p, s, b, c, n):\n"
     "        x = torch.cat([self.emb_partner(p), self.emb_sku(s),\n"
     "                       self.emb_brand(b), self.emb_channel(c),\n"
     "                       self.num_enc(n)], dim=-1).unsqueeze(1)\n"
     "        return self.q_head(self.encoder(x).squeeze(1))\n"
     "\n"
     "def pinball_loss(y_pred, y_true, quantiles):\n"
     "    losses = []\n"
     "    for i, q in enumerate(quantiles):\n"
     "        diff = y_true - y_pred[:, i]\n"
     "        losses.append(torch.maximum(q*diff, (q-1)*diff))\n"
     "    return torch.stack(losses, dim=-1).mean()\n"
     "\n"
     "cfg = Cfg(n_partners=manifest['n_partners'], n_skus=manifest['n_skus'],\n"
     "          n_brands=manifest['n_brands'], n_channels=manifest['n_channels'],\n"
     "          n_numeric=len(NUMERIC_COLS))\n"
     "model = GlobalNN(cfg).cuda()\n"
     "n_params = sum(p.numel() for p in model.parameters())\n"
     "print(f'GlobalNN: {n_params/1e6:.2f}M params')"),

    ("code",
     "# Cell 5: train (~90-120 min on T4)\n"
     "from torch.utils.data import DataLoader, TensorDataset\n"
     "from torch.optim import AdamW\n"
     "from torch.optim.lr_scheduler import CosineAnnealingLR\n"
     "\n"
     "BATCH = 1024; EPOCHS = 15; LR = 5e-4; WD = 0.01\n"
     "\n"
     "train_ds = TensorDataset(*train_tensors)\n"
     "val_ds   = TensorDataset(*val_tensors)\n"
     "train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,\n"
     "                          num_workers=2, drop_last=True, pin_memory=True)\n"
     "val_loader   = DataLoader(val_ds, batch_size=BATCH*4, shuffle=False,\n"
     "                          num_workers=2, pin_memory=True)\n"
     "\n"
     "opt = AdamW(model.parameters(), lr=LR, weight_decay=WD)\n"
     "sched = CosineAnnealingLR(opt, T_max=EPOCHS*len(train_loader))\n"
     "\n"
     "# Smoke test BEFORE the loop\n"
     "model.train()\n"
     "smoke = [t[:64].cuda() for t in train_tensors]\n"
     "sm_pred = model(smoke[0], smoke[1], smoke[2], smoke[3], smoke[4])\n"
     "sm_loss = pinball_loss(sm_pred, smoke[5], cfg.quantiles)\n"
     "print(f'smoke: loss={sm_loss.item():.4f}  finite={torch.isfinite(sm_loss).item()}')\n"
     "assert torch.isfinite(sm_loss).item(), 'non-finite loss — abort'\n"
     "print('✓ starting training')\n"
     "\n"
     "best_val = float('inf')\n"
     f"ckpt = '{OUT_DIR}/v14_globalnn_best.pt'\n"
     "\n"
     "for epoch in range(EPOCHS):\n"
     "    model.train()\n"
     "    epoch_loss = 0.0\n"
     "    for batch in train_loader:\n"
     "        p, s, b, c, n, y = [t.cuda(non_blocking=True) for t in batch]\n"
     "        opt.zero_grad()\n"
     "        loss = pinball_loss(model(p, s, b, c, n), y, cfg.quantiles)\n"
     "        loss.backward()\n"
     "        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n"
     "        opt.step(); sched.step()\n"
     "        epoch_loss += loss.item()\n"
     "    model.eval()\n"
     "    with torch.no_grad():\n"
     "        v_loss = 0.0\n"
     "        for batch in val_loader:\n"
     "            p, s, b, c, n, y = [t.cuda(non_blocking=True) for t in batch]\n"
     "            v_loss += pinball_loss(model(p, s, b, c, n), y, cfg.quantiles).item()\n"
     "        v_loss /= len(val_loader)\n"
     "    flag = '★' if v_loss < best_val else ' '\n"
     "    if v_loss < best_val:\n"
     "        best_val = v_loss\n"
     "        torch.save({'sd': model.state_dict(), 'mu': mu, 'sd_arr': sd,\n"
     "                    'numeric_cols': NUMERIC_COLS, 'cfg': cfg.__dict__},\n"
     "                   ckpt)\n"
     "    print(f'epoch {epoch+1:2d}/{EPOCHS}  '\n"
     "          f'train={epoch_loss/len(train_loader):.4f}  val={v_loss:.4f}  {flag}')\n"
     "print(f'\\nbest val loss = {best_val:.4f}; saved to {ckpt}')"),

    ("code",
     "# Cell 6: forecast on val + test (~5 min)\n"
     "ck = torch.load(ckpt, map_location='cuda', weights_only=False)\n"
     "model.load_state_dict(ck['sd']); model.eval()\n"
     "\n"
     "QUANTILES = list(cfg.quantiles)\n"
     "MEDIAN_IDX = QUANTILES.index(0.5)\n"
     "inv_partner = {v: k for k, v in vocab['Партнер'].items()}\n"
     "inv_sku     = {v: k for k, v in vocab['Артикул'].items()}\n"
     "\n"
     "def predict(tensors, df):\n"
     "    p, s, b, c, n, y = [t.cuda() for t in tensors]\n"
     "    preds = []\n"
     "    with torch.no_grad():\n"
     "        for i in range(0, len(p), BATCH*4):\n"
     "            sl = slice(i, i+BATCH*4)\n"
     "            preds.append(model(p[sl], s[sl], b[sl], c[sl], n[sl]).cpu().float().numpy())\n"
     "    pred_all = np.concatenate(preds, axis=0)\n"
     "    out = df[['Период_str']].rename(columns={'Период_str': 'Период'}).copy()\n"
     "    out['Партнер'] = df['Партнер_idx'].map(inv_partner).fillna('<UNK>')\n"
     "    out['Артикул'] = df['Артикул_idx'].map(inv_sku).fillna('<UNK>')\n"
     "    out['target_qty'] = df['target_qty'].values\n"
     "    out['prediction'] = np.clip(pred_all[:, MEDIAN_IDX], 0, None)\n"
     "    return out\n"
     "\n"
     "val_pred  = predict(val_tensors,  val_df)\n"
     "test_pred = predict(test_tensors, test_df)\n"
     f"val_pred.to_csv('{OUT_DIR}/preds_v14_globalnn_val.csv', index=False)\n"
     f"test_pred.to_csv('{OUT_DIR}/preds_v14_globalnn_test.csv', index=False)\n"
     "print(f'wrote val ({len(val_pred):,}), test ({len(test_pred):,})')\n"
     "\n"
     "for name, df in (('val', val_pred), ('test', test_pred)):\n"
     "    y, yh = df['target_qty'].values.astype(float), df['prediction'].values.astype(float)\n"
     "    wape = np.abs(y-yh).sum() / max(1e-9, y.sum())\n"
     "    bias = (yh.sum()-y.sum()) / max(1e-9, y.sum()) * 100\n"
     "    print(f'{name}: rows={len(df):,}  WAPE={wape:.3f}  bias%={bias:+.1f}')"),
]


def main() -> int:
    cells = []
    for kind, src in CELLS_SOURCE:
        if kind == "md":
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": src,
            })
        else:
            cells.append({
                "cell_type": "code",
                "metadata": {},
                "source": src,
                "outputs": [],
                "execution_count": None,
            })

    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3",
                           "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "cells": cells,
    }
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {DEST} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
