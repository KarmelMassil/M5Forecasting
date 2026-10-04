"""
Phase 3 — Modeling  (v3)
Walmart Revenue Forecasting Competition

Fixes vs v2:
  1. Store 0 excluded from all model training. Aggregate is always built
     bottom-up via reconcile_hierarchy(). Prevents the ~10x larger revenue
     from dominating LightGBM gain-based splits.
  2. Ensemble weights optimised on 3 Q4 folds (2012, 2013, 2014) combined
     rather than a single Q4 2014 window, reducing overfitting risk.

Run:  python models.py
"""

import os, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import logging

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#f8f8f6",
    "axes.grid": True, "grid.color": "white", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif",
})

from config import FEATURES_DIR as _FD, MODELS_DIR as _MD, CALENDAR_CSV as _CC, PARAMS_JSON as _PJ
FEAT_DIR = str(_FD)
CAL_PATH = str(_CC)
OUT      = str(_MD)
os.makedirs(OUT, exist_ok=True)

FORECAST_START  = pd.Timestamp("2015-10-01")
FORECAST_END    = pd.Timestamp("2015-12-31")
TRAIN_STORE_IDS = list(range(1, 11))   # store 0 always reconciled bottom-up
PALETTE = ["#2271B3","#E76F51","#2A9D8F","#8338EC","#F4A261",
           "#06D6A0","#EF233C","#4361EE","#F77F00","#7B2D8B"]

# Three Q4 validation folds — same seasonal window as the forecast target
FOLDS = [
    {"name": "Q4_2012", "cutoff": pd.Timestamp("2012-09-30"),
     "val_start": pd.Timestamp("2012-10-01"), "val_end": pd.Timestamp("2012-12-31")},
    {"name": "Q4_2013", "cutoff": pd.Timestamp("2013-09-30"),
     "val_start": pd.Timestamp("2013-10-01"), "val_end": pd.Timestamp("2013-12-31")},
    {"name": "Q4_2014", "cutoff": pd.Timestamp("2014-09-30"),
     "val_start": pd.Timestamp("2014-10-01"), "val_end": pd.Timestamp("2014-12-31")},
]

print("=" * 64)
print("  Phase 3 — Modeling  (v3)")
print("=" * 64)


# ══════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ══════════════════════════════════════════════════════════════════

train_all = pd.read_csv(f"{FEAT_DIR}/train_features.csv", parse_dates=["date"])
test_raw  = pd.read_csv(f"{FEAT_DIR}/test_features.csv",  parse_dates=["date"])
cal_raw   = pd.read_csv(CAL_PATH, parse_dates=["date"])

print(f"\n✓ {len(train_all):,} training rows | {len(test_raw):,} test rows | "
      f"{len(cal_raw)} calendar events")


# ══════════════════════════════════════════════════════════════════
# 2.  SAFE FEATURES — semantic whitelist (no NaN probe)
# ══════════════════════════════════════════════════════════════════

meta_cols     = ["store_id", "date", "revenue", "is_forecast"]
all_feat_cols = [c for c in test_raw.columns if c not in meta_cols]

_DATE   = ["year","month","day","dayofweek","dayofyear","weekofyear","quarter",
           "is_weekend","is_month_start","is_month_end","day_of_quarter","time_idx",
           "sin_dow","cos_dow","sin_doy","cos_doy","sin_month","cos_month"]
_LAGS   = [
    # lag ≥ 92 → safe for full 92-day forecast window
    "lag_92",  "lag_93",  "lag_94",  "lag_95",  "lag_96",  "lag_97",  "lag_98",
    "lag_182",
    "lag_364", "lag_365", "lag_371",
    "lag_728", "lag_729", "lag_730", "lag_731",
    # calendar-shift-aware lag: max(lag_364, lag_365, lag_371)
    # avoids the holiday-zero problem (e.g. lag_364 for Dec 24 → Christmas Day ~$0)
    "best_lag",
]
# is_lag364_holiday and is_lag365_holiday are auto-captured by _EVENTS (is_ prefix)
_ANCHOR_ROLLING = [
    "rolling_mean_7_lag92",
    "rolling_mean_28_lag92",
    "rolling_std_28_lag92",
    "rolling_mean_28_lag364",
]
_INTER  = [
    "event_x_weekend", "pre_xmas_weekend",
    "yoy_growth_ratio",    # lag_364 / lag_728 — YoY growth factor (safe)
    # trend_adj_lag364 removed: ablation confirmed redundant (rank 29, collinear)
]
_EVENTS = [c for c in all_feat_cols
           if (c.startswith("event_") or c.startswith("is_") or
               c.startswith("days_") or c.startswith("in_"))
           # Exclude meta column and its pandas-renamed duplicate from the
           # old CSV bug where is_forecast appeared twice in FEATURE_COLS.
           # features.py v2 is now fixed, but guard both names for safety.
           and c not in ("is_forecast", "is_forecast.1")]
_STORE  = ["store_id",                          # ← added: global model must know which store
           "region_enc", "store_mean", "store_std", "lag364_vs_store_mean"]
# rolling28_vs_store_mean excluded: derived from rolling_mean_28 (leaky)
# store_id is safe because store 0 is excluded from training (TRAIN_STORE_IDS)
_INTER  = ["event_x_weekend","pre_xmas_weekend"]

_seen = set()
SAFE_FEATURES = []
for f in (_DATE + _LAGS + _ANCHOR_ROLLING + _EVENTS + _STORE + _INTER):
    if f in all_feat_cols and f not in _seen:
        SAFE_FEATURES.append(f)
        _seen.add(f)

# store_id lives in meta_cols so it is excluded from all_feat_cols, but we
# need it in the global model so it can distinguish stores 1-10.
# Prepend it so LightGBM sees it first (it will be registered as categorical).
if "store_id" not in SAFE_FEATURES:
    SAFE_FEATURES.insert(0, "store_id")

_bad = [f for f in SAFE_FEATURES if test_raw[f].isna().any()]
if _bad:
    print(f"  WARNING: removing {len(_bad)} features with test NaNs: {_bad}")
    SAFE_FEATURES = [f for f in SAFE_FEATURES if f not in _bad]

CAT_FEATURES   = ["store_id", "region_enc", "dayofweek", "month", "quarter"]
_PS_DROP       = ["store_id","region_enc","store_mean","store_std","lag364_vs_store_mean"]
PS_FEATURES    = [f for f in SAFE_FEATURES if f not in _PS_DROP]
PS_CAT         = [c for c in CAT_FEATURES if c in PS_FEATURES]

print(f"✓ {len(SAFE_FEATURES)} direct-forecast-safe features | "
      f"{len(PS_FEATURES)} per-store features")


# ══════════════════════════════════════════════════════════════════
# 3.  HELPERS
# ══════════════════════════════════════════════════════════════════

def apply_store_stats(df, stats_df):
    """Overwrite store_mean, store_std, lag364_vs_store_mean using scoped stats."""
    df = df.copy()
    for sid in df["store_id"].unique():
        if sid not in stats_df.index:
            continue
        mask = df["store_id"] == sid
        m = stats_df.loc[sid, "store_mean"]
        s = stats_df.loc[sid, "store_std"]
        df.loc[mask, "store_mean"]          = m
        df.loc[mask, "store_std"]           = s
        df.loc[mask, "lag364_vs_store_mean"] = df.loc[mask, "lag_364"] / m
    return df


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_preds(preds_df, actual_df, label=""):
    merged = preds_df.merge(actual_df[["store_id","date","revenue"]],
                            on=["store_id","date"], how="inner")
    overall  = rmse(merged["revenue"], merged["prediction"])
    per_store = (merged.groupby("store_id")
                 .apply(lambda g: rmse(g["revenue"], g["prediction"]))
                 .rename("rmse"))
    if label:
        s0 = per_store.get(0, np.nan)
        print(f"    [{label:20s}]  overall={overall:>8,.0f}  store-0={s0:>8,.0f}")
    return overall, per_store


def hard_overrides(preds_df, cal_df):
    """Zero out revenue on store-closed holidays (Christmas, New Year)."""
    closed = set(cal_df[cal_df["event"].str.contains(
        "Christmas|NewYear", na=False)]["date"])
    out = preds_df.copy()
    out.loc[out["date"].isin(closed), "prediction"] = 0.0
    out["prediction"] = out["prediction"].clip(lower=0)
    return out


def reconcile_hierarchy(preds_df):
    """Bottom-up: store 0 = sum(stores 1-10)."""
    ind = preds_df[preds_df["store_id"] != 0].copy()
    agg = (ind.groupby("date")["prediction"].sum()
              .reset_index()
              .assign(store_id=0))
    return (pd.concat([ind, agg], ignore_index=True)
              .sort_values(["store_id","date"])
              .reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════
# 4.  MODEL TRAINING FUNCTIONS
# ══════════════════════════════════════════════════════════════════

LGBM_PARAMS = dict(
    n_estimators=2000, learning_rate=0.03, num_leaves=63,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, verbose=-1,
)


def train_lgbm_global(train_df, valid_df_=None):
    """Global LGBM — expects stores 1-10 only (caller's responsibility).

    Sample weights: proportional to store_mean so higher-revenue stores
    get more weight in training. This aligns the training objective with
    the competition metric, where higher-revenue stores (and their summed
    store-0 aggregate) contribute larger absolute errors.
    Weights are normalised to mean=1 to avoid changing the effective
    learning rate.
    """
    X_tr, y_tr = train_df[SAFE_FEATURES], train_df["revenue"]
    cat_idx = [SAFE_FEATURES.index(c) for c in CAT_FEATURES if c in SAFE_FEATURES]

    sw = train_df["store_mean"].values
    sw = sw / sw.mean()   # normalise: mean weight = 1

    m = lgb.LGBMRegressor(**LGBM_PARAMS)
    if valid_df_ is not None:
        m.fit(X_tr, y_tr,
              sample_weight=sw,
              eval_set=[(valid_df_[SAFE_FEATURES], valid_df_["revenue"])],
              eval_metric="rmse",
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(500)],
              categorical_feature=cat_idx)
    else:
        m.fit(X_tr, y_tr, sample_weight=sw, categorical_feature=cat_idx)
    return m


def train_lgbm_per_store(train_df, valid_df_=None):
    """Per-store LGBM — loops over TRAIN_STORE_IDS (1-10), never store 0."""
    models = {}
    for sid in TRAIN_STORE_IDS:
        tr = train_df[train_df["store_id"] == sid]
        m  = lgb.LGBMRegressor(**{**LGBM_PARAMS, "n_estimators": 1500})
        cat_idx = [PS_FEATURES.index(c) for c in PS_CAT]
        if valid_df_ is not None:
            vl = valid_df_[valid_df_["store_id"] == sid]
            m.fit(tr[PS_FEATURES], tr["revenue"],
                  eval_set=[(vl[PS_FEATURES], vl["revenue"])],
                  eval_metric="rmse",
                  callbacks=[lgb.early_stopping(100, verbose=False),
                             lgb.log_evaluation(9999)],
                  categorical_feature=cat_idx)
        else:
            m.fit(tr[PS_FEATURES], tr["revenue"], categorical_feature=cat_idx)
        models[sid] = m
    return models


def predict_per_store(models, df):
    preds = []
    for sid, m in models.items():
        sub = df[df["store_id"] == sid].copy()
        sub["prediction"] = m.predict(sub[PS_FEATURES]).clip(0)
        preds.append(sub[["store_id","date","prediction"]])
    return pd.concat(preds, ignore_index=True)

# ══════════════════════════════════════════════════════════════════
# 5b.  LOAD TUNED LGBM PARAMS  (produced by tune.py)
# ══════════════════════════════════════════════════════════════════

import json, pathlib
_PARAMS_PATH = _PJ
if _PARAMS_PATH.exists():
    with open(_PARAMS_PATH) as _f:
        _tuned = json.load(_f)
    LGBM_PARAMS.update(_tuned)
    LGBM_PARAMS["n_estimators"] = 2000   # always use full budget for final training
    print(f"\n✓ Loaded tuned LGBM params from {_PARAMS_PATH}")
    for k, v in sorted(_tuned.items()):
        print(f"    {k:25s}: {v:.5g}")
else:
    print(f"\n  (No tuned params found at {_PARAMS_PATH} — using defaults)")


# ══════════════════════════════════════════════════════════════════
# 5.  run_fold — trains Naive, LGBM Global, LGBM Per-Store
# ══════════════════════════════════════════════════════════════════

# Prophet removed from ensemble:
#   - 68% worse RMSE than LGBM Global on Q4 2013+2014
#   - ≈0% ensemble weight across all runs
#   - ~12 min training cost per full pipeline run
MODEL_COLS = ["naive", "lgbm_global", "lgbm_ps"]


def run_fold(fold_def, train_all_df):
    """
    1. Split on fold_def["cutoff"].
    2. Scope store stats to training data only.
    3. Train all models on stores 1-10 only.
    4. Apply hard_overrides + reconcile_hierarchy to each model's val preds.
    5. Return stacked DataFrame:
         [store_id, date, revenue, naive, lgbm_global, lgbm_ps, prophet]
       for ALL stores 0-10 (store 0 from bottom-up reconciliation).
    """
    cutoff    = fold_def["cutoff"]
    val_start = fold_def["val_start"]
    val_end   = fold_def["val_end"]

    train_f = train_all_df[train_all_df["date"] <= cutoff].copy()
    valid_f = train_all_df[(train_all_df["date"] >= val_start) &
                           (train_all_df["date"] <= val_end)].copy()

    # Scope store stats to this fold's training window
    stats = train_f.groupby("store_id")["revenue"].agg(store_mean="mean", store_std="std")
    train_f = apply_store_stats(train_f, stats)
    valid_f = apply_store_stats(valid_f, stats)

    # Stores 1-10 slices for model training + LGBM early stopping
    tr_ind = train_f[train_f["store_id"].isin(TRAIN_STORE_IDS)]
    vl_ind = valid_f[valid_f["store_id"].isin(TRAIN_STORE_IDS)]

    # ── Naive ──────────────────────────────────────────────────
    naive_v = valid_f[["store_id","date"]].assign(prediction=valid_f["lag_364"].values)
    naive_v = hard_overrides(naive_v, cal_raw)
    naive_v = reconcile_hierarchy(naive_v)

    # ── LightGBM Global ────────────────────────────────────────
    lgbm_g   = train_lgbm_global(tr_ind, vl_ind)
    lgbm_g_v = vl_ind[["store_id","date"]].assign(
        prediction=lgbm_g.predict(vl_ind[SAFE_FEATURES]).clip(0))
    lgbm_g_v = hard_overrides(lgbm_g_v, cal_raw)
    lgbm_g_v = reconcile_hierarchy(lgbm_g_v)

    # ── LightGBM Per-Store ─────────────────────────────────────
    ps_models = train_lgbm_per_store(tr_ind, vl_ind)
    ps_v      = predict_per_store(ps_models, vl_ind)
    ps_v      = hard_overrides(ps_v, cal_raw)
    ps_v      = reconcile_hierarchy(ps_v)

    # ── Stack (all stores via reconciled predictions) ──────────
    stack = (valid_f[["store_id","date","revenue"]]
             .merge(naive_v .rename(columns={"prediction":"naive"}),       on=["store_id","date"])
             .merge(lgbm_g_v.rename(columns={"prediction":"lgbm_global"}), on=["store_id","date"])
             .merge(ps_v    .rename(columns={"prediction":"lgbm_ps"}),     on=["store_id","date"]))
    return stack


# ══════════════════════════════════════════════════════════════════
# 6.  3-FOLD CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════

print("\n── 3-Fold Cross-Validation (Q4 2012, Q4 2013, Q4 2014) ──")

fold_stacks  = []
fold_results = []   # list of {name, naive, lgbm_global, lgbm_ps}

for fold in FOLDS:
    print(f"\n  Fold: {fold['name']}")
    t0    = time.time()
    stack = run_fold(fold, train_all)
    fold_stacks.append(stack)

    res = {"name": fold["name"]}
    for m in MODEL_COLS:
        p = stack[["store_id","date"]].assign(prediction=stack[m])
        o, ps = evaluate_preds(p, stack[["store_id","date","revenue"]], m)
        res[m]        = o
        res[f"{m}_s0"] = ps.get(0, np.nan)
    fold_results.append(res)
    print(f"    Elapsed: {time.time()-t0:.0f}s")

combined_stack = pd.concat(fold_stacks, ignore_index=True)
print(f"\n  Combined stack: {len(combined_stack):,} rows "
      f"({len(fold_stacks)} folds × {len(fold_stacks[0])} rows)")


# ══════════════════════════════════════════════════════════════════
# 7.  OPTIMISE ENSEMBLE WEIGHTS ON COMBINED 3-FOLD STACK
# ══════════════════════════════════════════════════════════════════

print("\n── Optimising ensemble weights on combined 3-fold OOF ──")

# Use only data-rich folds for weight optimisation.
# Q4 2012 (8 months training, 0% lag_728 coverage) is a different regime
# from Q4 2015 (44 months, 100% coverage) and inflates per-store weight
# at the expense of global model performance on the actual test fold.
# We report Q4 2012 RMSE in the summary table but exclude it from weighting.
_opt_fold_names = {f["name"] for f in FOLDS if f["name"] != "Q4_2012"}
_opt_stacks     = [s for s, f in zip(fold_stacks, FOLDS)
                   if f["name"] in _opt_fold_names]
ind_stack = (pd.concat(_opt_stacks, ignore_index=True)
               .query("store_id != 0")
               .copy())

print(f"  Optimising on {len(_opt_fold_names)} folds "
      f"({', '.join(sorted(_opt_fold_names))}) — {len(ind_stack):,} rows")

def blend_rmse(weights):
    w = np.abs(weights) / np.abs(weights).sum()
    blended = (ind_stack[MODEL_COLS].values * w).sum(axis=1)
    return rmse(ind_stack["revenue"], blended)

result = minimize(blend_rmse, x0=[0.10, 0.70, 0.20],
                  method="Nelder-Mead",
                  options={"maxiter": 10000, "xatol": 1e-7})
OPT_WEIGHTS = np.abs(result.x) / np.abs(result.x).sum()

print("  Weights (3-fold, individual stores only):")
for name, w in zip(MODEL_COLS, OPT_WEIGHTS):
    print(f"    {name:16s}: {w:.4f}")

# Ensemble combined RMSE on the 3-fold stack
ens_blend   = (combined_stack[MODEL_COLS].values * OPT_WEIGHTS).sum(axis=1).clip(0)
ens_val_all = combined_stack[["store_id","date"]].assign(prediction=ens_blend)
ens_val_all = hard_overrides(ens_val_all, cal_raw)
ens_val_all = reconcile_hierarchy(ens_val_all)
ens_combined, ens_ps = evaluate_preds(
    ens_val_all, combined_stack[["store_id","date","revenue"]], "Ensemble (3-fold)")


# ══════════════════════════════════════════════════════════════════
# 8.  FINAL MODELS — FULL TRAINING DATA (stores 1-10 only)
# ══════════════════════════════════════════════════════════════════

print("\n── Training final models on full data (stores 1-10 only) ──")

full_train = train_all.copy()
stats_full = full_train.groupby("store_id")["revenue"].agg(store_mean="mean", store_std="std")
full_train = apply_store_stats(full_train, stats_full)
test_raw   = apply_store_stats(test_raw,   stats_full)

full_ind = full_train[full_train["store_id"].isin(TRAIN_STORE_IDS)]
test_ind  = test_raw [test_raw ["store_id"].isin(TRAIN_STORE_IDS)]

t0 = time.time()
lgbm_global_full = train_lgbm_global(full_ind)
print(f"  LGBM Global     ({time.time()-t0:.0f}s)")

t0 = time.time()
ps_full = train_lgbm_per_store(full_ind)
print(f"  LGBM Per-Store  ({time.time()-t0:.0f}s, {len(ps_full)} stores)")


# ══════════════════════════════════════════════════════════════════
# 9.  FINAL TEST PREDICTIONS
# ══════════════════════════════════════════════════════════════════

print("\n── Generating final test predictions ──")


def predict_all_models(test_df, test_ind_df, lgbm_g, ps_models):
    """Predict 3 models → stacked DataFrame for test window."""
    # Naive
    naive = test_df[["store_id","date"]].assign(prediction=test_df["lag_364"].values)
    naive = hard_overrides(naive, cal_raw)
    naive = reconcile_hierarchy(naive)

    # LGBM Global
    lg = test_ind_df[["store_id","date"]].assign(
        prediction=lgbm_g.predict(test_ind_df[SAFE_FEATURES]).clip(0))
    lg = hard_overrides(lg, cal_raw)
    lg = reconcile_hierarchy(lg)

    # LGBM Per-Store
    lp = predict_per_store(ps_models, test_ind_df)
    lp = hard_overrides(lp, cal_raw)
    lp = reconcile_hierarchy(lp)

    return (test_df[["store_id","date"]]
            .merge(naive.rename(columns={"prediction":"naive"}),       on=["store_id","date"])
            .merge(lg   .rename(columns={"prediction":"lgbm_global"}), on=["store_id","date"])
            .merge(lp   .rename(columns={"prediction":"lgbm_ps"}),     on=["store_id","date"]))


test_stack  = predict_all_models(test_raw, test_ind, lgbm_global_full, ps_full)

test_blend  = (test_stack[MODEL_COLS].values * OPT_WEIGHTS).sum(axis=1).clip(0)
ensemble_test = test_stack[["store_id","date"]].assign(prediction=test_blend)
ensemble_test = hard_overrides(ensemble_test, cal_raw)
ensemble_test = reconcile_hierarchy(ensemble_test)

print(f"  Range: ${ensemble_test['prediction'].min():,.0f} – "
      f"${ensemble_test['prediction'].max():,.0f}")
print(f"  Zero predictions (closed-store days): "
      f"{(ensemble_test['prediction'] == 0).sum()}")


# ══════════════════════════════════════════════════════════════════
# 10.  VALIDATION SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  VALIDATION — 3-Fold Q4 Overall RMSE  (lower is better)")
print("=" * 70)

col_w = 11
fold_names = [f["name"] for f in FOLDS]
header = f"  {'Model':<18}" + "".join(f"{n:>{col_w}}" for n in fold_names) + f"{'Combined':>{col_w}}"
print(header)
print("  " + "-" * (len(header) - 2))

for m in MODEL_COLS:
    fold_vals = [fr[m] for fr in fold_results]
    comb_preds = combined_stack[["store_id","date"]].assign(prediction=combined_stack[m])
    c_rmse, _ = evaluate_preds(comb_preds, combined_stack[["store_id","date","revenue"]])
    row = f"  {m:<18}" + "".join(f"{v:>{col_w},.0f}" for v in fold_vals) + f"{c_rmse:>{col_w},.0f}"
    print(row)

# Ensemble per-fold RMSE
ens_fold_rmses = []
for stack in fold_stacks:
    b = (stack[MODEL_COLS].values * OPT_WEIGHTS).sum(axis=1).clip(0)
    p = stack[["store_id","date"]].assign(prediction=b)
    p = hard_overrides(p, cal_raw)
    p = reconcile_hierarchy(p)
    o, _ = evaluate_preds(p, stack[["store_id","date","revenue"]])
    ens_fold_rmses.append(o)

ens_row = (f"  {'ensemble (opt)':<18}" +
           "".join(f"{v:>{col_w},.0f}" for v in ens_fold_rmses) +
           f"{ens_combined:>{col_w},.0f}")
print("  " + "-" * (len(header) - 2))
print(ens_row)
print("=" * 70)
print(f"\n  Optimised weights: " +
      "  ".join(f"{m}={w:.3f}" for m, w in zip(MODEL_COLS, OPT_WEIGHTS)))

# Save summary CSV
summary_rows = []
for fr in fold_results:
    for m in MODEL_COLS:
        summary_rows.append({"fold": fr["name"], "model": m,
                             "overall_rmse": fr[m], "store0_rmse": fr[f"{m}_s0"]})
pd.DataFrame(summary_rows).to_csv(f"{OUT}/validation_summary.csv", index=False)


# ══════════════════════════════════════════════════════════════════
# 11.  SUBMISSION FILE
# ══════════════════════════════════════════════════════════════════

sub = ensemble_test.copy()
sub["id"] = sub["store_id"].astype(str) + "_" + sub["date"].dt.strftime("%Y%m%d")
sub = sub[["id","prediction"]].sort_values("id").reset_index(drop=True)

assert len(sub)                == 1012, f"Expected 1012 rows, got {len(sub)}"
assert sub["prediction"].min() >= 0,   "Negative predictions found"
assert sub["id"].nunique()     == 1012, "Duplicate IDs"

sub.to_csv(f"{OUT}/submission.csv", index=False)
print(f"\n✓ submission.csv  ({len(sub)} rows)")


# ══════════════════════════════════════════════════════════════════
# 12.  PLOTS
# ══════════════════════════════════════════════════════════════════

print("\n── Generating plots ──")

# ── Fig 1: Per-fold RMSE grouped bar chart + ensemble weights ──────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ax = axes[0]
x     = np.arange(len(MODEL_COLS) + 1)
width = 0.26
bar_colors = ["#bbddf7","#7abfee","#2271B3"]

for i, (fr, stack) in enumerate(zip(fold_results, fold_stacks)):
    vals = [fr[m] for m in MODEL_COLS]
    # ensemble for this fold
    b = (stack[MODEL_COLS].values * OPT_WEIGHTS).sum(axis=1).clip(0)
    p = stack[["store_id","date"]].assign(prediction=b)
    p = hard_overrides(p, cal_raw)
    p = reconcile_hierarchy(p)
    o, _ = evaluate_preds(p, stack[["store_id","date","revenue"]])
    vals.append(o)
    bars = ax.bar(x + i*width, vals, width, label=fr["name"],
                  color=bar_colors[i], edgecolor="white")

ax.set_xticks(x + width)
ax.set_xticklabels(["Naive","LGBM\nGlobal","LGBM\nPer-Store","Ensemble"], fontsize=8)
ax.set_ylabel("Overall RMSE")
ax.set_title("Per-Fold RMSE by Model\n(3 Q4 windows — each ~1012 rows)")
ax.legend(fontsize=8)

ax2 = axes[1]
bars2 = ax2.bar(MODEL_COLS, OPT_WEIGHTS * 100,
                color=PALETTE[:len(MODEL_COLS)], edgecolor="white", width=0.5)
ax2.bar_label(bars2, labels=[f"{w*100:.1f}%" for w in OPT_WEIGHTS], padding=3, fontsize=9)
ax2.set_ylabel("Ensemble weight (%)")
ax2.set_title("Optimised Ensemble Weights\n(3-fold combined OOF, not single Q4 2014)")

fig.tight_layout()
fig.savefig(f"{OUT}/fig1_fold_rmse.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("  ✓ fig1_fold_rmse.png")

# ── Fig 2: Model consistency across folds (line per model) ────────────
fig, ax = plt.subplots(figsize=(10, 5))
for m, color in zip(MODEL_COLS, PALETTE):
    vals = [fr[m] for fr in fold_results]
    ax.plot(fold_names, vals, marker="o", color=color, linewidth=2, label=m)
# ensemble line
ax.plot(fold_names, ens_fold_rmses, marker="D", color="#333",
        linewidth=2.5, linestyle="--", label="ensemble")
ax.set_ylabel("Overall RMSE")
ax.set_title("RMSE Stability Across Folds\n(flat = generalises well; sloping = fold-specific)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_fold_stability.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("  ✓ fig2_fold_stability.png")

# ── Fig 3: Final Q4 2015 forecast ─────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

ax = axes[0]
s0_hist = full_train[(full_train["store_id"] == 0) &
                     (full_train["date"] >= "2015-07-01")].sort_values("date")
ax.plot(s0_hist["date"], s0_hist["revenue"],
        color="#aaa", linewidth=1.2, label="Training (2015)")
s0_fc = ensemble_test[ensemble_test["store_id"] == 0].sort_values("date")
ax.plot(s0_fc["date"], s0_fc["prediction"],
        color="#2271B3", linewidth=2.4, label="Ensemble forecast")
q4_ref = full_train[(full_train["store_id"] == 0) &
                    (full_train["date"] >= "2014-10-01") &
                    (full_train["date"] <= "2014-12-31")].sort_values("date")
ax.plot(q4_ref["date"] + pd.DateOffset(years=1), q4_ref["revenue"].values,
        color="#E76F51", linewidth=1.8, linestyle="--", alpha=0.7,
        label="Q4 2014 (reference)")
q4_2015 = cal_raw[(cal_raw["date"] >= "2015-10-01") & (cal_raw["date"] <= "2015-12-31")]
for _, row in q4_2015.iterrows():
    ax.axvline(row["date"], color="#ccc", linewidth=1, linestyle="--")
    ax.text(row["date"], s0_fc["prediction"].max()*0.98,
            row["event"].split(",")[0], rotation=90, fontsize=7.5, va="top", color="#c0392b")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.set_ylabel("All-Stores Revenue ($)")
ax.set_title("Final Forecast — All Stores (Q4 2015)")
ax.legend(loc="upper left", fontsize=9)

ax2 = axes[1]
for sid in range(1, 11):
    s_fc = ensemble_test[ensemble_test["store_id"] == sid].sort_values("date")
    ax2.plot(s_fc["date"], s_fc["prediction"],
             color=PALETTE[sid-1], linewidth=1.5, alpha=0.8, label=f"Store {sid}")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax2.set_ylabel("Revenue ($)")
ax2.set_title("Final Forecast — Individual Stores 1-10")
ax2.legend(ncol=5, fontsize=8, loc="upper left")
for _, row in q4_2015.iterrows():
    ax2.axvline(row["date"], color="#ccc", linewidth=0.8, linestyle="--")
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_final_forecast.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("  ✓ fig3_final_forecast.png")

# ── Fig 4: Feature importance (global model, stores 1-10 only) ────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, itype in zip(axes, ["split","gain"]):
    fi = (pd.Series(lgbm_global_full.booster_.feature_importance(importance_type=itype),
                    index=SAFE_FEATURES)
          .sort_values(ascending=True).tail(25))
    cols_fi = ["#E76F51" if "lag" in i
               else "#2271B3" if any(k in i for k in ["event","is_","days_","in_"])
               else "#2A9D8F" for i in fi.index]
    ax.barh(fi.index, fi.values, color=cols_fi, edgecolor="white", height=0.65)
    ax.set_title(f"Feature importance ({itype}) — Top 25\nLGBM Global, trained on stores 1-10 only")
axes[1].legend(handles=[
    plt.Rectangle((0,0),1,1, color="#E76F51", label="Lag"),
    plt.Rectangle((0,0),1,1, color="#2271B3", label="Event / holiday"),
    plt.Rectangle((0,0),1,1, color="#2A9D8F", label="Date / store"),
], loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_feature_importance.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("  ✓ fig4_feature_importance.png")

print()
print("=" * 64)
print("  ALL DONE — Phase 3 v3")
print("=" * 64)
print(f"\n  3-fold ensemble RMSE : {ens_combined:,.0f}")
print(f"  Submission rows      : {len(sub)}")
print(f"  Files → {OUT}/")
