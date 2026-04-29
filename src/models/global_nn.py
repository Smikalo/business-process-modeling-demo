"""V14 GlobalNN — Transformer-encoder model with learned categorical
embeddings for Партнер / Артикул / Бренд / Канал.

Single global model that outputs h-step quantile forecasts. Designed to
be trained on Colab Free T4 (memory budget ~10 GB) across 4 sessions
with checkpointing every 500 steps.

Architecture:
  - Categorical embeddings: 4× nn.Embedding(vocab_size, dim=32)
  - Numeric features: passed through a 2-layer MLP encoder → 64 dims
  - Concatenation: [4×32 emb + 64 num] = 192 dims
  - Transformer encoder: 4 layers, 8 heads, 192 → 192
  - Quantile head: nn.Linear(192, n_quantiles=5)
    (q0.1, q0.25, q0.5, q0.75, q0.9 — q0.5 is the point forecast)
  - Loss: pinball loss summed across the 5 quantiles
  - Optimizer: AdamW lr=5e-4, cosine schedule, weight_decay=0.01

This file imports torch lazily so importing the package on a CPU box
(e.g. for reading manifests) doesn't require torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


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
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


def build_model(cfg: GlobalNNConfig):
    """Construct the GlobalNN. Imports torch on demand."""
    import torch
    import torch.nn as nn

    class GlobalNN(nn.Module):
        def __init__(self):
            super().__init__()
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

            assert (4 * cfg.emb_dim + cfg.num_enc_dim) == cfg.d_model, \
                "embedding+numeric dims must sum to d_model"

            enc_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.nhead,
                dim_feedforward=cfg.d_model * 4,
                dropout=cfg.dropout, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer,
                                                   num_layers=cfg.n_layers)

            self.q_head = nn.Linear(cfg.d_model, len(cfg.quantiles))

        def forward(self, partner_idx, sku_idx, brand_idx, channel_idx,
                    numeric):
            ep = self.emb_partner(partner_idx)
            es = self.emb_sku(sku_idx)
            eb = self.emb_brand(brand_idx)
            ec = self.emb_channel(channel_idx)
            en = self.num_enc(numeric)
            x = torch.cat([ep, es, eb, ec, en], dim=-1)  # B × d_model
            x = x.unsqueeze(1)                            # B × 1 × d_model
            x = self.encoder(x).squeeze(1)                # B × d_model
            return self.q_head(x)                         # B × n_quantiles

    return GlobalNN()


def pinball_loss(y_pred, y_true, quantiles):
    """Pinball loss summed over quantiles. y_pred is B × Q,
    y_true is B."""
    import torch
    losses = []
    for i, q in enumerate(quantiles):
        diff = y_true - y_pred[:, i]
        losses.append(torch.maximum(q * diff, (q - 1) * diff))
    return torch.stack(losses, dim=-1).mean()
