"""Canonical file paths and project-wide constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# ── Transactional data ──────────────────────────────────────────────────────
SALES_PATH = DATA_DIR / "Продажи 2020-2026.txt"
SHIPMENT_PATH = DATA_DIR / "Отгрузки 2020-2026.txt"
RESTS_ORC_PATH = DATA_DIR / "Остатки ОРЦ 2020-2025.txt"
RESTS_TT_PATH = DATA_DIR / "Остатки ТТ 2020-2025.txt"
RECEIPTS_ORC_PATH = DATA_DIR / "Поступление ОРЦ 2020-2025.xlsx"

# ── Reference / dimensional data ────────────────────────────────────────────
NOMENCLATURE_PATH = DATA_DIR / "Справочник номенклатуры.xlsx"
PARTNERS_PATH = DATA_DIR / "Справочник партнеров.xlsx"
PRICE_DJECO_PATH = DATA_DIR / "Прайс Djeco.xlsx"
PRICE_CUBICFUN_PATH = DATA_DIR / "Прайс CubicFun.xlsx"
PRICE_INFANTINO_PATH = DATA_DIR / "Прайс Infantino.xlsx"
PROMOTIONS_PATH = DATA_DIR / "Нац. акции 2024.xlsx"

# ── Period boundaries ───────────────────────────────────────────────────────
PERIOD_START = "2020-01"
PERIOD_END = "2026-02"

# ── Train / Validation / Test split boundaries (month-end inclusive) ────────
TRAIN_END = "2024-06"
VAL_END = "2025-06"
# Test: everything after VAL_END up to PERIOD_END
