"""
tune.py — Optuna hyperparameter search for LightGBM Global.

Runs independently of the main model pipeline so its wall-clock time
doesn't block submission generation.

Saves best params to /home/claude/best_lgbm_params.json.
models.py reads that file at startup and uses it for all LGBM training.

Run:  python tune.py
"""

import json, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── paths ──────────────────────────────────────────────────────────────────────
from config import FEATURES_DIR as _FD, CALENDAR_CSV as _CC, PARAMS_JSON as _PJ
FEAT_DIR  = str(_FD)
CAL_PATH  = str(_CC)
OUT_JSON  = str(_PJ)

# ── constants (must mirror models.py) ─────────────────────────────────────────
TRAIN_STORE_IDS = list(range(1, 11))

# Use only Q4 2013 and Q4 2014 — data-rich, representative of the test scenario.
# Q4 2012 (8 months training) is excluded from both tuning and ensemble weighting.
TUNE_FOLDS = [
    {"name": "Q4_2013", "cutoff": pd.Timestamp("2013-09-30"),
     "val_start": pd.Timestamp("2013-10-01"), "val_end": pd.Timestamp("2013-12-31")},
    {"name": "Q4_2014", "cutoff": pd.Timestamp("2014-09-30"),
     "val_start": pd.Timestamp("2014-10-01"), "val_end": pd.Timestamp("2014-12-31")},
]

N_TRIALS = 60   # was 60 TPE finds strong params well within 60 trials
TIMEOUT  = 900  # was 240 hard stop at 4 minutes regardless of trial count

# ── load data ──────────────────────────────────────────────────────────────────
print("Loading features …")
train_all = pd.read_csv(f"{FEAT_DIR}/train_features.csv", parse_dates=["date"])
cal_raw   = pd.read_csv(CAL_PATH, parse_dates=["date"])

# ── safe feature whitelist (must mirror models.py) ─────────────────────────────
meta_cols     = ["store_id", "date", "revenue", "is_forecast"]
all_feat_cols = [c for c in
                 pd.read_csv(f"{FEAT_DIR}/test_features.csv", nrows=1).columns
                 if c not in meta_cols]

_DATE   = ["year","month","day","dayofweek","dayofyear","weekofyear","quarter",
           "is_weekend","is_month_start","is_month_end","day_of_quarter","time_idx",
           "sin_dow","cos_dow","sin_doy","cos_doy","sin_month","cos_month"]
_LAGS   = ["lag_92","lag_93","lag_94","lag_95","lag_96","lag_97","lag_98",
           "lag_182","lag_364","lag_365","lag_371",
           "lag_728","lag_729","lag_730","lag_731","best_lag"]
_ANCHOR = ["rolling_mean_7_lag92","rolling_mean_28_lag92",
           "rolling_std_28_lag92","rolling_mean_28_lag364"]
_EVENTS = [c for c in all_feat_cols
           if (c.startswith("event_") or c.startswith("is_") or
               c.startswith("days_") or c.startswith("in_"))
           and c not in ("is_forecast","is_forecast.1")]
_STORE  = ["region_enc","store_mean","store_std","lag364_vs_store_mean"]
_INTER  = ["event_x_weekend","pre_xmas_weekend","yoy_growth_ratio","trend_adj_lag364"]

test_raw = pd.read_csv(f"{FEAT_DIR}/test_features.csv")
_seen, SAFE_FEATURES = set(), []
for f in (_DATE + _LAGS + _ANCHOR + _EVENTS + _STORE + _INTER):
    if f in all_feat_cols and f not in _seen:
        SAFE_FEATURES.append(f); _seen.add(f)
if "store_id" not in SAFE_FEATURES:
    SAFE_FEATURES.insert(0, "store_id")

SAFE_FEATURES = [f for f in SAFE_FEATURES if test_raw[f].isna().sum() == 0]
CAT_FEATURES  = ["store_id","region_enc","dayofweek","month","quarter"]
print(f"  {len(SAFE_FEATURES)} safe features, {len(TUNE_FOLDS)} tuning folds")


# ── helpers ────────────────────────────────────────────────────────────────────
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def apply_store_stats(df, stats_df):
    df = df.copy()
    for sid in df["store_id"].unique():
        if sid not in stats_df.index:
            continue
        mask = df["store_id"] == sid
        m = stats_df.loc[sid, "store_mean"]
        s = stats_df.loc[sid, "store_std"]
        df.loc[mask, "store_mean"]           = m
        df.loc[mask, "store_std"]            = s
        df.loc[mask, "lag364_vs_store_mean"] = df.loc[mask, "lag_364"] / m
    return df

def hard_overrides(preds_df):
    closed = set(cal_raw[cal_raw["event"].str.contains(
        "Christmas|NewYear", na=False)]["date"])
    out = preds_df.copy()
    out.loc[out["date"].isin(closed), "prediction"] = 0.0
    out["prediction"] = out["prediction"].clip(lower=0)
    return out

def reconcile_hierarchy(preds_df):
    ind = preds_df[preds_df["store_id"] != 0].copy()
    agg = ind.groupby("date")["prediction"].sum().reset_index().assign(store_id=0)
    return pd.concat([ind, agg], ignore_index=True)

def evaluate(preds_df, actual_df):
    m = preds_df.merge(actual_df[["store_id","date","revenue"]],
                       on=["store_id","date"], how="inner")
    return rmse(m["revenue"], m["prediction"])


# ── Optuna objective ───────────────────────────────────────────────────────────
def objective(trial):
    params = dict(
        n_estimators      = 600,   # capped; early stopping finds best iteration
        learning_rate     = trial.suggest_float("learning_rate",     0.005, 0.15, log=True),
        num_leaves        = trial.suggest_int  ("num_leaves",         15,  255),
        min_child_samples = trial.suggest_int  ("min_child_samples",   5,  100),
        colsample_bytree  = trial.suggest_float("colsample_bytree",  0.4,  1.0),
        subsample         = trial.suggest_float("subsample",          0.4,  1.0),
        reg_alpha         = trial.suggest_float("reg_alpha",          0.0,  5.0),
        reg_lambda        = trial.suggest_float("reg_lambda",         0.0,  5.0),
        random_state=42, n_jobs=-1, verbose=-1,
    )

    fold_rmses = []
    for fold_def in TUNE_FOLDS:
        tr_f = train_all[train_all["date"] <= fold_def["cutoff"]].copy()
        vl_f = train_all[(train_all["date"] >= fold_def["val_start"]) &
                         (train_all["date"] <= fold_def["val_end"])].copy()

        stats = tr_f.groupby("store_id")["revenue"].agg(
            store_mean="mean", store_std="std")
        tr_f = apply_store_stats(tr_f, stats)
        vl_f = apply_store_stats(vl_f, stats)

        tr_ind = tr_f[tr_f["store_id"].isin(TRAIN_STORE_IDS)]
        vl_ind = vl_f[vl_f["store_id"].isin(TRAIN_STORE_IDS)]

        sw      = tr_ind["store_mean"].values / tr_ind["store_mean"].mean()
        cat_idx = [SAFE_FEATURES.index(c) for c in CAT_FEATURES if c in SAFE_FEATURES]

        m = lgb.LGBMRegressor(**params)
        m.fit(tr_ind[SAFE_FEATURES], tr_ind["revenue"],
              sample_weight=sw,
              eval_set=[(vl_ind[SAFE_FEATURES], vl_ind["revenue"])],
              eval_metric="rmse",
              callbacks=[lgb.early_stopping(40, verbose=False),
                         lgb.log_evaluation(9999)],
              categorical_feature=cat_idx)

        preds = vl_ind[["store_id","date"]].assign(
            prediction=m.predict(vl_ind[SAFE_FEATURES]).clip(0))
        preds = hard_overrides(preds)
        preds = reconcile_hierarchy(preds)
        fold_rmses.append(evaluate(preds, vl_f))

    return np.mean(fold_rmses)


# ── run study ──────────────────────────────────────────────────────────────────
print(f"\nRunning Optuna ({N_TRIALS} trials, timeout={TIMEOUT}s) …")
t0 = time.time()
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT,
               show_progress_bar=False)

elapsed   = time.time() - t0
completed = len(study.trials)
print(f"  Completed {completed} trials in {elapsed:.0f}s")
print(f"  Best 2-fold RMSE: {study.best_value:,.0f}")
print("  Best params:")
for k, v in sorted(study.best_params.items()):
    print(f"    {k:25s}: {v:.5g}")

# print top-5 trials
print("\n  Top 5 trials:")
sorted_trials = sorted(study.trials, key=lambda t: t.value)[:5]
for rank, t in enumerate(sorted_trials, 1):
    print(f"    #{rank}  RMSE={t.value:,.0f}  "
          f"lr={t.params.get('learning_rate','-'):.4g}  "
          f"leaves={t.params.get('num_leaves','-')}")

# ── save best params ───────────────────────────────────────────────────────────
with open(OUT_JSON, "w") as f:
    json.dump(study.best_params, f, indent=2)
print(f"\n✓ Saved to {OUT_JSON}  (models.py will load these at startup)")
