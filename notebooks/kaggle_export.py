"""Kaggle-compatible single-file version of the full pipeline.

This script can run standalone in Kaggle Notebooks (CPU mode, ~30h/week free).
Upload the data/ folder as a Kaggle dataset, then run this script.

Dependencies: pip install pandas numpy openpyxl python-calamine lightgbm scikit-learn optuna joblib
"""

# %% [markdown]
# # Demand Forecasting PoC — Kaggle-Compatible Pipeline
# SKU-level forecasting for Ukrainian toy distributor (Djeco, CubicFun, Infantino)

# %%
import logging
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger("poc")

# Kaggle input path — adjust if using different dataset name
DATA_DIR = Path("/kaggle/input/toy-distributor-data") if Path("/kaggle/input").exists() else Path("data")

t0 = time.time()

# %% [markdown]
# ## 1. Data Ingestion

# %%
def clean_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(r"\s+", "", regex=True).str.replace(",", ".", regex=False),
        errors="coerce",
    ).fillna(0.0)

def to_period(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%d.%m.%Y", errors="coerce").dt.to_period("M")

# Sales
sales = pd.read_csv(DATA_DIR / "Продажи 2020-2026.txt", sep="\t", skiprows=6, encoding="utf-8",
    names=["Партнер","Артикул","Дата","Количество","Выручка"],
    dtype={"Артикул":str,"Партнер":str,"Количество":str,"Выручка":str}, on_bad_lines="warn")
sales = sales.dropna(subset=["Артикул"])
for c in ["Артикул","Партнер"]: sales[c] = sales[c].str.strip()
for c in ["Количество","Выручка"]: sales[c] = clean_numeric(sales[c])
sales["Период"] = to_period(sales["Дата"])
sales = sales.drop(columns=["Дата"]).dropna(subset=["Период"])
log.info("Sales: %d rows", len(sales))

# Shipments
ship = pd.read_csv(DATA_DIR / "Отгрузки 2020-2026.txt", sep="\t", skiprows=7, encoding="utf-8",
    names=["Партнер","Артикул","Дата","Количество","Выручка"],
    dtype={"Артикул":str,"Партнер":str,"Количество":str,"Выручка":str}, on_bad_lines="warn")
ship = ship.dropna(subset=["Артикул"])
for c in ["Артикул","Партнер"]: ship[c] = ship[c].str.strip()
for c in ["Количество","Выручка"]: ship[c] = clean_numeric(ship[c])
ship["Период"] = to_period(ship["Дата"])
ship = ship.drop(columns=["Дата"]).dropna(subset=["Период"])

# ORC Rests
orc = pd.read_csv(DATA_DIR / "Остатки ОРЦ 2020-2025.txt", sep="\t", skiprows=1, encoding="cp1251",
    header=None, names=["Дата","Артикул","Количество","Стоимость"],
    dtype={"Артикул":str,"Количество":str,"Стоимость":str}, on_bad_lines="warn")
orc = orc.dropna(subset=["Артикул"])
orc["Артикул"] = orc["Артикул"].str.strip()
for c in ["Количество","Стоимость"]: orc[c] = clean_numeric(orc[c])
orc["Период"] = to_period(orc["Дата"])
orc = orc.drop(columns=["Дата"]).dropna(subset=["Период"])

# TT Rests
tt = pd.read_csv(DATA_DIR / "Остатки ТТ 2020-2025.txt", sep="\t", skiprows=1, encoding="cp1251",
    header=None, names=["Дата","Партнер","Артикул","Количество","Стоимость"],
    dtype={"Артикул":str,"Партнер":str,"Количество":str,"Стоимость":str}, on_bad_lines="warn")
tt = tt.dropna(subset=["Артикул"])
for c in ["Артикул","Партнер"]: tt[c] = tt[c].str.strip()
for c in ["Количество","Стоимость"]: tt[c] = clean_numeric(tt[c])
tt["Период"] = to_period(tt["Дата"])
tt = tt.drop(columns=["Дата"]).dropna(subset=["Период"])

# ORC Receipts
rec = pd.read_excel(DATA_DIR / "Поступление ОРЦ 2020-2025.xlsx")
rec["Артикул"] = rec["Артикул"].astype(str).str.strip()
rec["Период"] = rec["Дата"].dt.to_period("M")
rec = rec.drop(columns=["Дата"])

# Partners
part = pd.read_excel(DATA_DIR / "Справочник партнеров.xlsx")
part.columns = part.columns.str.strip()
part["Партнер"] = part["Партнер"].str.strip()
part = part.rename(columns={"Направление":"Канал","Соглашение":"Тип_соглашения"})
part = part.sort_values("Тип_соглашения").drop_duplicates(subset=["Партнер"], keep="last")

log.info("All sources loaded in %.0fs", time.time() - t0)

# %% [markdown]
# ## 2. Aggregation & Skeleton

# %%
a_sales = sales.groupby(["Период","Партнер","Артикул"], as_index=False).agg(Количество_sales=("Количество","sum"), Выручка_sales=("Выручка","sum"))
a_ship = ship.groupby(["Период","Партнер","Артикул"], as_index=False).agg(Количество_ship=("Количество","sum"))
a_tt = tt.groupby(["Период","Партнер","Артикул"], as_index=False).agg(Количество_tt=("Количество","sum"))
a_tt["Количество_tt"] = a_tt["Количество_tt"].clip(lower=0)
a_orc = orc.groupby(["Период","Артикул"], as_index=False).agg(Количество_orc=("Количество","sum"))
a_orc["Количество_orc"] = a_orc["Количество_orc"].clip(lower=0)
a_rec = rec.groupby(["Период","Артикул"], as_index=False).agg(Количество_receipts=("Количество","sum"))

pairs = pd.concat([a_sales[["Партнер","Артикул"]], a_ship[["Партнер","Артикул"]], a_tt[["Партнер","Артикул"]]]).drop_duplicates()
all_periods = pd.period_range("2020-01", "2026-02", freq="M")
idx = pd.MultiIndex.from_product([all_periods, pairs.index], names=["Период","_i"])
skeleton = pd.DataFrame(index=idx).reset_index().merge(pairs, left_on="_i", right_index=True).drop(columns=["_i"])

df = (skeleton
    .merge(a_sales, on=["Период","Партнер","Артикул"], how="left")
    .merge(a_ship, on=["Период","Партнер","Артикул"], how="left")
    .merge(a_tt, on=["Период","Партнер","Артикул"], how="left")
    .merge(a_orc, on=["Период","Артикул"], how="left")
    .merge(a_rec, on=["Период","Артикул"], how="left")
)
metric_cols = [c for c in df.columns if c not in ("Период","Партнер","Артикул")]
df[metric_cols] = df[metric_cols].fillna(0.0)

# Partners
df = df.merge(part[["Партнер","Канал","Тип_соглашения"]], on="Партнер", how="left")
df["Канал"] = df["Канал"].fillna("unknown")
df["Тип_соглашения"] = df["Тип_соглашения"].fillna("unknown")
df["target_qty"] = np.where(df["Тип_соглашения"]=="Выкуп", df["Количество_ship"], df["Количество_sales"])

log.info("Master: %d rows × %d cols", *df.shape)

# %% [markdown]
# ## 3. Feature Engineering

# %%
GRP = ["Партнер","Артикул"]
df = df.sort_values(GRP + ["Период"]).reset_index(drop=True)
g = df.groupby(GRP, sort=False)

for lag in [1,2,3,6,12]:
    df[f"lag_{lag}"] = g["target_qty"].shift(lag).fillna(0)
df["lag_1_orc"] = g["Количество_orc"].shift(1).fillna(0)

df["rmean_3"] = (df["lag_1"]+df["lag_2"]+df["lag_3"])/3
df["rmean_6"] = (df["lag_1"]+df["lag_2"]+df["lag_3"]+df["lag_6"])/4
df["rmean_12"] = (df["lag_1"]+df["lag_12"])/2
df["rstd_3"] = df[["lag_1","lag_2","lag_3"]].std(axis=1)

m = df["Период"].dt.month; y = df["Период"].dt.year
df["month"] = m.astype(np.int8)
df["quarter"] = df["Период"].dt.quarter.astype(np.int8)
df["year"] = y.astype(np.int16)
df["month_sin"] = np.sin(2*np.pi*m/12).astype(np.float32)
df["month_cos"] = np.cos(2*np.pi*m/12).astype(np.float32)
df["is_wartime"] = ((y>2022)|((y==2022)&(m>=2))).astype(np.int8)
df["is_q4"] = (df["quarter"]==4).astype(np.int8)

df["stockout_orc"] = (df["Количество_orc"]==0).astype(np.int8)
df["stockout_tt"] = (df["Количество_tt"]==0).astype(np.int8)
lag1 = df["lag_1"].clip(lower=0)
df["inv_to_sales_orc"] = (df["Количество_orc"]/(lag1+1e-9)).clip(upper=999).astype(np.float32)

brand_agg = df.groupby(["Канал","Период"], as_index=False).agg(brand_total=("target_qty","sum"))
brand_agg["Период"] = brand_agg["Период"]+1
df = df.merge(brand_agg, on=["Канал","Период"], how="left")
df["brand_total"] = df["brand_total"].fillna(0).astype(np.float32)

log.info("Features: %d cols", df.shape[1])

# %% [markdown]
# ## 4. Train & Evaluate

# %%
TRAIN_END = pd.Period("2024-06","M")
VAL_END = pd.Period("2025-06","M")
df_train = df[df["Период"]<=TRAIN_END]
df_val = df[(df["Период"]>TRAIN_END)&(df["Период"]<=VAL_END)]
df_test = df[df["Период"]>VAL_END]

exclude = {"Период","Партнер","Артикул","Канал","Тип_соглашения","target_qty",
           "Количество_sales","Выручка_sales","Количество_ship","Количество_tt","Количество_orc","Количество_receipts"}
feat = [c for c in df.columns if c not in exclude and df[c].dtype.kind in ("f","i","u")]

params = {"objective":"regression","metric":"mae","num_leaves":127,"learning_rate":0.05,
          "feature_fraction":0.8,"bagging_fraction":0.8,"bagging_freq":5,"n_jobs":-1,"device":"cpu","verbose":-1}
ts = lgb.Dataset(df_train[feat], label=df_train["target_qty"])
vs = lgb.Dataset(df_val[feat], label=df_val["target_qty"])
model = lgb.train(params, ts, 500, valid_sets=[vs], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])

preds = model.predict(df_test[feat]).clip(min=0)
actual = df_test["target_qty"].values
total = np.abs(actual).sum()
wape = np.abs(actual-preds).sum()/total if total>0 else 0
nz = actual>0
mape_nz = np.abs((actual[nz]-preds[nz])/actual[nz]).mean() if nz.sum()>0 else 0

print(f"\nTest WAPE={wape:.4f}, MAPE_nz={mape_nz:.4f}, RMSE={np.sqrt(np.mean((actual-preds)**2)):.4f}")
print(f"Total time: {time.time()-t0:.0f}s")
