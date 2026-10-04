"""
Phase 2 — Feature Engineering
Walmart Revenue Forecasting Competition

Outputs:
  features/train_features.csv   — full training set, all features, ready for modelling
  features/test_features.csv    — forecast window, all safe features filled
  features/feature_catalog.txt  — description of every column
  features/fig_*.png            — validation plots

Run:  python features.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f8f6",
    "axes.grid":        True,
    "grid.color":       "white",
    "grid.linewidth":   0.8,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.family":      "sans-serif",
})

# ── paths ─────────────────────────────────────────────────────────────────────
from config import DATA_DIR as _C, FEATURES_DIR as _F, CALENDAR_CSV
DATA = str(_C)
OUT  = str(_F)
os.makedirs(OUT, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
TRAIN_END        = pd.Timestamp("2015-09-30")
FORECAST_START   = pd.Timestamp("2015-10-01")
FORECAST_END     = pd.Timestamp("2015-12-31")

# Stores closed these days → revenue must be 0 (hard override in model)
HARD_ZERO_EVENTS = {"Christmas", "NewYear"}
# Note: SOFT_ZERO_EVENTS (Thanksgiving) removed — stores are open but revenue is low;
# this is learned from the data via event flags, not forced to zero.

# Event taxonomy
SPORTS_EVENTS    = {"SuperBowl","NBAFinalsStart","NBAFinalsEnd"}
NATIONAL_EVENTS  = {"NewYear","MartinLutherKingDay","PresidentsDay","MemorialDay",
                    "IndependenceDay","LaborDay","ColumbusDay","VeteransDay","Thanksgiving"}
RELIGIOUS_EVENTS = {"Christmas","Easter","OrthodoxEaster","OrthodoxChristmas",
                    "Chanukah End","LentStart","LentWeek2","Pesach End","Purim End",
                    "Ramadan starts","Eid al-Fitr","EidAlAdha"}
CULTURAL_EVENTS  = {"Halloween","ValentinesDay","StPatricksDay","Cinco De Mayo",
                    "Mother's day","Father's day"}

# Days around a holiday to build pre/post ramp features
RAMP_HOLIDAYS = {
    "Thanksgiving": (-7, +3),   # 7 days before, 3 days after
    "Christmas":    (-14, +1),  # 14 days before (holiday shopping ramp)
    "Halloween":    (-3, +1),
    "SuperBowl":    (-2, +1),
    "LaborDay":     (-1, +2),
}

print("=" * 62)
print("  Phase 2 — Feature Engineering")
print("=" * 62)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD RAW DATA
# ══════════════════════════════════════════════════════════════════════════════

train_raw = pd.read_csv(f"{DATA}/train.csv", parse_dates=["date"])
cal_raw   = pd.read_csv(f"{DATA}/calendar_events.csv", parse_dates=["date"])
sub_raw   = pd.read_csv(f"{DATA}/forecast_submission.csv")

# Parse the submission IDs into store_id + date
sub_raw["store_id"] = sub_raw["id"].str.extract(r"^(\d+)_").astype(int)
sub_raw["date"]     = pd.to_datetime(sub_raw["id"].str.extract(r"_(\d{8})$")[0],
                                     format="%Y%m%d")

# Store metadata (id → name, region)
store_meta = (train_raw[["store_id","store_name"]]
              .drop_duplicates()
              .set_index("store_id"))

def region_from_name(name):
    if "California" in str(name):  return "california"
    if "Texas"      in str(name):  return "texas"
    if "Wisconsin"  in str(name):  return "wisconsin"
    return "aggregate"

store_meta["region"] = store_meta["store_name"].apply(region_from_name)
region_enc = {"aggregate": 0, "california": 1, "texas": 2, "wisconsin": 3}
store_meta["region_enc"] = store_meta["region"].map(region_enc)

# Per-store revenue statistics (from training — used as scale features)
store_stats = (train_raw
               .groupby("store_id")["revenue"]
               .agg(store_mean="mean", store_std="std",
                    store_min="min", store_max="max"))

print(f"\n✓ Loaded {len(train_raw):,} training rows, "
      f"{sub_raw['date'].nunique()} forecast dates, "
      f"{len(cal_raw)} calendar events")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CALENDAR EVENT TABLES
# ══════════════════════════════════════════════════════════════════════════════

def parse_event_cell(cell):
    """'Easter, OrthodoxEaster' → ['Easter', 'OrthodoxEaster']"""
    return [e.strip() for e in str(cell).split(",")]

# Flat event-date lookup  {date: [event, ...]}
event_by_date = {}
for _, row in cal_raw.iterrows():
    events = parse_event_cell(row["event"])
    event_by_date[row["date"]] = events

# Pre-compute ramp windows for key holidays
#   ramp_windows[(event_name, date)] = (days_before, days_after)
ramp_lookup = {}   # date → {holiday: days_offset}
for _, row in cal_raw.iterrows():
    for ev in parse_event_cell(row["event"]):
        if ev in RAMP_HOLIDAYS:
            before, after = RAMP_HOLIDAYS[ev]
            for offset in range(before, after + 1):
                d = row["date"] + pd.Timedelta(days=offset)
                ramp_lookup.setdefault(d, {})[ev] = offset   # negative = before

print("✓ Built calendar event lookup")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BUILD FULL TIME SPINE  (training + forecast, all stores)
# ══════════════════════════════════════════════════════════════════════════════

all_dates  = pd.date_range(train_raw["date"].min(), FORECAST_END)
all_stores = sorted(train_raw["store_id"].unique())

spine = pd.DataFrame(
    [(s, d) for s in all_stores for d in all_dates],
    columns=["store_id", "date"]
)

# Merge known revenue (NaN for forecast dates)
spine = spine.merge(
    train_raw[["store_id","date","revenue"]],
    on=["store_id","date"], how="left"
)

# Flag splits
spine["is_forecast"] = spine["date"] > TRAIN_END

print(f"✓ Full time spine: {len(spine):,} rows "
      f"({spine[~spine['is_forecast']].shape[0]:,} train + "
      f"{spine[spine['is_forecast']].shape[0]:,} test)")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FEATURE ENGINEERING  (all operations on the full spine)
# ══════════════════════════════════════════════════════════════════════════════

df = spine.copy()
df = df.sort_values(["store_id","date"]).reset_index(drop=True)


# ── 4a. Date features ─────────────────────────────────────────────────────────
print("\nBuilding date features …")

df["year"]           = df["date"].dt.year
df["month"]          = df["date"].dt.month
df["day"]            = df["date"].dt.day
df["dayofweek"]      = df["date"].dt.dayofweek          # 0=Mon … 6=Sun
df["dayofyear"]      = df["date"].dt.dayofyear
df["weekofyear"]     = df["date"].dt.isocalendar().week.astype(int)
df["quarter"]        = df["date"].dt.quarter
df["is_weekend"]     = (df["dayofweek"] >= 5).astype(int)
df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)

# Day within quarter (0 = first day of Q1/Q2/Q3/Q4)
quarter_starts = {1: "-01-01", 2: "-04-01", 3: "-07-01", 4: "-10-01"}
df["day_of_quarter"] = df.apply(
    lambda r: (r["date"] - pd.Timestamp(
        str(r["year"]) + quarter_starts[r["quarter"]])).days, axis=1)

# Integer time index (days since first training date — encodes trend)
t0 = df["date"].min()
df["time_idx"] = (df["date"] - t0).dt.days

# Cyclical encodings (preserves periodicity for tree models too, and is great
# for neural nets / linear models)
df["sin_dow"]   = np.sin(2 * np.pi * df["dayofweek"]   / 7)
df["cos_dow"]   = np.cos(2 * np.pi * df["dayofweek"]   / 7)
df["sin_doy"]   = np.sin(2 * np.pi * df["dayofyear"]   / 365)
df["cos_doy"]   = np.cos(2 * np.pi * df["dayofyear"]   / 365)
df["sin_month"] = np.sin(2 * np.pi * df["month"]       / 12)
df["cos_month"] = np.cos(2 * np.pi * df["month"]       / 12)

date_features = [
    "year","month","day","dayofweek","dayofyear","weekofyear","quarter",
    "is_weekend","is_month_start","is_month_end","day_of_quarter","time_idx",
    "sin_dow","cos_dow","sin_doy","cos_doy","sin_month","cos_month",
]
print(f"  → {len(date_features)} date features")


# ── 4b. Lag features ──────────────────────────────────────────────────────────
print("Building lag features …")
# Note on safety for the 92-day forecast window:
#   SAFE (entire window, no recursion needed):  lag_364, lag_365, lag_371
#   PARTIAL (first N days, then needs recursive): lag_7 (7 days), lag_28 (28 days) etc.
#   Strategy: compute all lags; NaN propagates honestly; fill_recursive() in model step

df = df.sort_values(["store_id","date"])

# Short lags — only first N days of the 92-day forecast window are safe
SHORT_LAGS = [7, 14, 21, 28, 35, 42, 56]

# Safe lags for the FULL 92-day forecast window:
#   lag_N is safe iff  (FORECAST_END - N) < FORECAST_START
#   i.e. N > (FORECAST_END - FORECAST_START).days = 91  → N ≥ 92
#
#   lag_92  : Dec 31 - 92 = Sep 30 (last training day)  ✓
#   lag_92-98: quarter-ago window, avoids all test-period actuals
#   lag_182  : half-year ago
#   lag_364/365/371: same period last year
#   lag_728-731: same period two years ago

SAFE_LAGS = [92, 93, 94, 95, 96, 97, 98, 182, 364, 365, 371, 728, 729, 730, 731]

for lag in SHORT_LAGS + SAFE_LAGS:
    df[f"lag_{lag}"] = df.groupby("store_id")["revenue"].shift(lag)

lag_features       = [f"lag_{l}" for l in SHORT_LAGS + SAFE_LAGS]
safe_lag_features  = [f"lag_{l}" for l in SAFE_LAGS]
short_lag_features = [f"lag_{l}" for l in SHORT_LAGS]

# ── Calendar-shift-aware lag ──────────────────────────────────────────────────
# Problem: lag_364 for Dec 24 2014 references Dec 25 2013 (Christmas, ~$0),
# causing a $226k prediction error on the biggest shopping day of the year.
# Fix: best_lag = max(lag_364, lag_365, lag_371) selects whichever reference
# year gives the highest (non-holiday-zero) revenue. On normal days all three
# are similar; on holiday-adjacent days the best one wins.
df["best_lag"] = df[["lag_364","lag_365","lag_371"]].max(axis=1)

# Explicit holiday-reference flags — tells the model *why* a lag is near zero.
# When lag_364 references a Christmas/NewYear date, the model should discount
# it and rely on lag_365 or best_lag instead.
_hard_zero_dates = set(
    cal_raw[cal_raw["event"].str.contains("Christmas|NewYear", na=False)]["date"]
)
df["is_lag364_holiday"] = df["date"].apply(
    lambda d: int((d - pd.Timedelta(days=364)) in _hard_zero_dates)
)
df["is_lag365_holiday"] = df["date"].apply(
    lambda d: int((d - pd.Timedelta(days=365)) in _hard_zero_dates)
)

lag_features = lag_features + ["best_lag"]   # best_lag is a derived lag feature
print(f"  → {len(lag_features)} lag features "
      f"({len(safe_lag_features)} fully safe for 92-day window, "
      f"{len(short_lag_features)} need recursive fill, + best_lag)")


# ── 4c. Rolling statistics ────────────────────────────────────────────────────
print("Building rolling features …")
# Shift by 1 first so rolling never includes the current date (no leakage)
# For test rows, these anchor to the last known training values.

for window in [7, 14, 28, 56]:
    shifted = df.groupby("store_id")["revenue"].shift(1)
    df[f"rolling_mean_{window}"] = (
        shifted.groupby(df["store_id"]).transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
    df[f"rolling_std_{window}"]  = (
        shifted.groupby(df["store_id"]).transform(lambda x: x.rolling(window, min_periods=2).std())
    )

# Expanding mean — computed with shift(1) so current day not included.
# NOTE: expanding_mean is leaky for VALIDATION (val rows are in training spine
# and see within-period actuals). Excluded from direct-forecast model feature set.
df["expanding_mean"] = (
    df.groupby("store_id")["revenue"]
    .transform(lambda x: x.shift(1).expanding(min_periods=7).mean())
)

# Year-over-year rolling ratio — FIXED to use groupby so it never crosses
# store boundaries. Still excluded from direct model (rolling_mean_28 is leaky).
df["yoy_ratio_28d"] = (
    df["rolling_mean_28"] /
    df.groupby("store_id")["lag_364"]
      .transform(lambda x: x.rolling(28, min_periods=1).mean())
)

# ── Safe anchor-lag rolling features ─────────────────────────────────────────
# These are safe for the full 92-day forecast window because they shift by a
# safe lag first (≥92 days), then roll over a window that never reaches into
# the forecast period.
#
#   rolling_mean_7_lag92  : mean of 7 days ending at (date − 92)
#     → for Dec 31 2015: covers Sep 24–30 2015 (all in training) ✓
#   rolling_mean_28_lag92 : mean of 28 days ending at (date − 92)
#     → for Dec 31 2015: covers Sep 3–30 2015 ✓
#   rolling_std_28_lag92  : std of same window
#   rolling_mean_28_lag364: mean of 28 days ending at (date − 364)
#     → for Dec 31 2015: covers Jan 1–28 2015 ✓

for window, anchor in [(7, 92), (28, 92), (28, 364)]:
    col = f"rolling_mean_{window}_lag{anchor}"
    df[col] = (
        df.groupby("store_id")["revenue"]
          .transform(lambda x, w=window, a=anchor:
                     x.shift(a).rolling(w, min_periods=1).mean())
    )

df["rolling_std_28_lag92"] = (
    df.groupby("store_id")["revenue"]
      .transform(lambda x: x.shift(92).rolling(28, min_periods=2).std())
)

rolling_features = (
    [f"rolling_mean_{w}" for w in [7, 14, 28, 56]] +
    [f"rolling_std_{w}"  for w in [7, 14, 28, 56]] +
    ["expanding_mean", "yoy_ratio_28d",
     "rolling_mean_7_lag92", "rolling_mean_28_lag92",
     "rolling_std_28_lag92", "rolling_mean_28_lag364"]
)
print(f"  → {len(rolling_features)} rolling features "
      f"(4 anchor-lag features safe for full 92-day window)")


# ── 4d. Calendar event features ──────────────────────────────────────────────
print("Building calendar event features …")

# Binary event flags
def make_event_flag(event_name):
    """1 on dates where event_name matches exactly (case-insensitive).
    Uses exact equality — NOT substring — to prevent 'Easter' matching
    'OrthodoxEaster'. parse_event_cell() already splits comma-separated
    events into individual strings before they reach event_by_date.
    """
    s = set(
        d for d, evs in event_by_date.items()
        if any(event_name.lower() == e.lower() for e in evs)
    )
    return df["date"].isin(s).astype(int)

for ev in ["Christmas","Thanksgiving","Halloween","SuperBowl",
           "LaborDay","IndependenceDay","Chanukah End","NewYear",
           "Easter","OrthodoxEaster","VeteransDay","ColumbusDay"]:
    col = "event_" + ev.lower().replace(" ","_").replace("'","")
    df[col] = make_event_flag(ev)

# Hard-zero flag (stores physically closed)
hard_zero_dates = set(
    d for d, evs in event_by_date.items()
    if any(e in HARD_ZERO_EVENTS for e in evs)
)
df["is_store_closed"] = df["date"].isin(hard_zero_dates).astype(int)

# Event type one-hot
def event_type_flag(event_set):
    return df["date"].apply(
        lambda d: int(any(e in event_set
                          for e in event_by_date.get(d, [])))
    )

df["event_type_sports"]   = event_type_flag(SPORTS_EVENTS)
df["event_type_national"] = event_type_flag(NATIONAL_EVENTS)
df["event_type_religious"]= event_type_flag(RELIGIOUS_EVENTS)
df["event_type_cultural"] = event_type_flag(CULTURAL_EVENTS)
df["is_any_event"]        = (
    df["event_type_sports"] | df["event_type_national"] |
    df["event_type_religious"] | df["event_type_cultural"]
).astype(int)

# Countdown features: days until / since nearest event
all_event_dates = sorted(cal_raw["date"].unique())

def days_to_next_event(date):
    future = [d for d in all_event_dates if d > date]
    return (future[0] - date).days if future else 999

def days_since_last_event(date):
    past = [d for d in all_event_dates if d <= date]
    return (date - past[-1]).days if past else 999

df["days_to_next_event"]   = df["date"].map(days_to_next_event)
df["days_since_last_event"]= df["date"].map(days_since_last_event)

# Pre/post-event ramp features for major holidays
for holiday in RAMP_HOLIDAYS:
    col = f"days_to_{holiday.lower().replace(' ','_')}"
    def ramp_offset(date, h=holiday):
        if date in ramp_lookup and h in ramp_lookup[date]:
            return ramp_lookup[date][h]   # negative = before event, positive = after
        return np.nan
    df[col] = df["date"].apply(ramp_offset)
    # Also binary "within ramp window"
    df[f"in_{holiday.lower().replace(' ','_')}_window"] = (~df[col].isna()).astype(int)
    df[col] = df[col].fillna(0)   # 0 = "not in the ramp window"

# Black Friday (day after US Thanksgiving — biggest retail day of the year)
thanksgiving_dates = [d for d, evs in event_by_date.items() if "Thanksgiving" in evs]
black_friday_dates = {d + pd.Timedelta(days=1) for d in thanksgiving_dates}
df["is_black_friday"] = df["date"].isin(black_friday_dates).astype(int)

# Cyber Monday (Mon after Black Friday)
cyber_monday_dates  = {d + pd.Timedelta(days=4) for d in thanksgiving_dates}
df["is_cyber_monday"] = df["date"].isin(cyber_monday_dates).astype(int)

event_features = [c for c in df.columns
                  if c.startswith("event_") or c.startswith("is_") or
                     c.startswith("days_") or c.startswith("in_")
                  and c != "is_forecast"]
# Guard: is_forecast must never appear as a model feature. The is_ prefix
# filter above would catch it without the explicit exclusion on combined
# conditions, but be explicit for safety.
event_features = [c for c in event_features if c != "is_forecast"]
print(f"  → {len(event_features)} event/holiday features")


# ── 4e. Store features ────────────────────────────────────────────────────────
print("Building store features …")

df = df.merge(store_meta[["region_enc"]], on="store_id", how="left")
df = df.merge(store_stats[["store_mean","store_std"]], on="store_id", how="left")

# Normalized revenue — kept in CSV for reference but excluded from FEATURE_COLS.
# It uses the same store_mean/store_std that models.py re-scopes per fold,
# so including it would create a parallel scaling column with stale stats.
df["revenue_normalized"] = (df["revenue"] - df["store_mean"]) / df["store_std"]

# Store-level lag ratios
df["lag364_vs_store_mean"]    = df["lag_364"]         / df["store_mean"]
df["rolling28_vs_store_mean"] = df["rolling_mean_28"] / df["store_mean"]

store_features = ["region_enc", "store_mean", "store_std", "lag364_vs_store_mean"]
# rolling28_vs_store_mean excluded: derived from rolling_mean_28 (leaky for val)
# revenue_normalized excluded: stale stats; models.py re-scopes stats per fold
print(f"  → {len(store_features)} store features (revenue_normalized kept in CSV, not in FEATURE_COLS)")


# ── 4f. Interaction / derived features ────────────────────────────────────────
print("Building interaction features …")

df["event_x_weekend"]   = df["is_any_event"] * df["is_weekend"]

df["pre_xmas_weekend"]  = (
    (df["day_of_quarter"] >= 72) &
    (df["quarter"] == 4) &
    (df["is_weekend"] == 1)
).astype(int)

# yoy_delta_364 = revenue − lag_364.
# THIS IS TARGET-DERIVED and must never be used as a prediction feature.
# It is kept in the CSV for EDA/diagnostics only and is explicitly absent
# from FEATURE_COLS. It is NaN in test rows (revenue is NaN there), so it
# would be auto-excluded by any NaN probe, but we exclude it by name too.
df["yoy_delta_364"] = df.groupby("store_id")["revenue"].transform(
    lambda x: x - x.shift(364)
)

# lag7_vs_lag14: short-lag momentum. Kept in CSV but excluded from direct
# model (both components are leaky for most of the 92-day window).
df["lag7_vs_lag14"] = df["lag_7"] - df["lag_14"]

interaction_features = ["event_x_weekend", "pre_xmas_weekend"]

# ── YoY trend-adjustment features (safe — both lags reference training data) ──
# yoy_growth_ratio = lag_364 / lag_728
#   Captures systematic growth: if 2014 Q4 was 5% above 2013 Q4,
#   the model can project a similar lift into 2015 Q4.
#   Safe: lag_364 ≥ 92 and lag_728 ≥ 92 → both reference pre-forecast data ✓
#   Clip to [0.5, 2.0] to prevent extreme values on holiday-zero days.
#   Fill NaN (first ~728 training days where lag_728 is not yet available) → 1.0 (no growth).
df["yoy_growth_ratio"] = (
    (df["lag_364"] / df["lag_728"].replace(0, np.nan))
    .clip(0.5, 2.0)
    .fillna(1.0)
)

# trend_adj_lag364 = lag_364 × yoy_growth_ratio
#   A trend-extrapolated forecast anchor: "if last year's Q4 was $X and
#   revenue grew by Y% YoY, predict $X * Y% for this year's Q4."
df["trend_adj_lag364"] = df["lag_364"] * df["yoy_growth_ratio"]

interaction_features = interaction_features + ["yoy_growth_ratio"]
# trend_adj_lag364 removed: redundant with lag_364 × yoy_growth_ratio already in model.
# Ablation confirmed it adds no RMSE improvement (ranked 29th, collinear with parents).

# yoy_delta_364 and lag7_vs_lag14 intentionally excluded from FEATURE_COLS
print(f"  → {len(interaction_features)} interaction features in FEATURE_COLS "
      f"(trend_adj_lag364, yoy_delta_364, lag7_vs_lag14 kept in CSV only)")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SPLIT BACK INTO TRAIN / TEST
# ══════════════════════════════════════════════════════════════════════════════

# All features used in modelling
FEATURE_COLS = (
    date_features +
    lag_features +
    rolling_features +
    event_features +
    store_features +
    interaction_features
)
# Remove duplicates while preserving order
seen = set()
FEATURE_COLS = [f for f in FEATURE_COLS if not (f in seen or seen.add(f))]

TARGET_COL = "revenue"
ID_COLS    = ["store_id","date","is_forecast"]

train_df = df[~df["is_forecast"]].copy()
test_df  = df[df["is_forecast"]].copy()

# Drop rows where SAFE lags are NaN (first ~364 days of training — no prior year)
train_df = train_df.dropna(subset=["lag_364","lag_365","lag_371"])

# Drop store-0 from direct training? — keep it; hierarchical reconciliation handles it
print(f"\n✓ Training set after lag warm-up dropped: {len(train_df):,} rows")
print(f"✓ Test set: {len(test_df):,} rows")

# How many NaN values per feature in TEST set?
nan_counts = test_df[FEATURE_COLS].isna().sum()
nan_features = nan_counts[nan_counts > 0].sort_values(ascending=False)
print(f"\n  NaN feature counts in test set (first {len(nan_features)} features):")
for feat, n in nan_features.items():
    safe_str = " ← needs recursive fill in model step" if feat in short_lag_features else ""
    print(f"    {feat:35s}  {n:4d} NaN{safe_str}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SAVE FEATURE FILES
# ══════════════════════════════════════════════════════════════════════════════

out_cols = ID_COLS + [TARGET_COL] + FEATURE_COLS

train_df[out_cols].to_csv(f"{OUT}/train_features.csv", index=False)
test_df[out_cols].to_csv(f"{OUT}/test_features.csv",  index=False)

print(f"\n✓ Saved train_features.csv  ({len(train_df):,} rows × {len(FEATURE_COLS)} features)")
print(f"✓ Saved test_features.csv   ({len(test_df):,}  rows × {len(FEATURE_COLS)} features)")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  WRITE FEATURE CATALOG
# ══════════════════════════════════════════════════════════════════════════════

catalog_lines = [
    "FEATURE CATALOG — Walmart Revenue Forecasting",
    "=" * 60,
    "",
    "TARGET COLUMN",
    "  revenue                    Daily store revenue ($). NaN in test set.",
    "",
    f"DATE FEATURES  ({len(date_features)} columns)",
    "  year/month/day/dayofweek   Raw calendar components",
    "  dayofyear / weekofyear     Day/week position within year",
    "  quarter                    1–4",
    "  is_weekend                 1 if Sat/Sun else 0",
    "  is_month_start/end         First/last day of month",
    "  day_of_quarter             Day within Q1–Q4 (0-indexed)",
    "  time_idx                   Days since 2011-01-29 (trend proxy)",
    "  sin/cos_dow                Cyclical day-of-week encoding",
    "  sin/cos_doy                Cyclical day-of-year encoding",
    "  sin/cos_month              Cyclical month encoding",
    "",
    f"LAG FEATURES  ({len(lag_features)} columns)",
    "  lag_7/14/21/28/35/42/56    Revenue N days ago — NEED RECURSIVE FILL for test",
    "  lag_364/365/371            Same period last year — SAFE for full test window",
    "  *** For direct (non-recursive) models: use only lag_364/365/371 ***",
    "",
    f"ROLLING FEATURES  ({len(rolling_features)} columns)",
    "  rolling_mean_7/14/28/56    Mean revenue over last N days (shift=1, no leakage)",
    "  rolling_std_7/14/28/56     Std dev over last N days",
    "  expanding_mean             Mean over all prior history",
    "  yoy_ratio_28d              rolling_mean_28 / lag_364.rolling(28)",
    "",
    f"EVENT / HOLIDAY FEATURES  ({len(event_features)} columns)",
    "  event_<name>               Binary: 1 on that specific event day",
    "  is_store_closed            1 on Christmas + New Year (revenue hard-zeroed)",
    "  event_type_*               Sports / National / Religious / Cultural flag",
    "  is_any_event               1 if any event that day",
    "  days_to_next_event         Days until next calendar event",
    "  days_since_last_event      Days since last calendar event",
    "  days_to_thanksgiving       Signed offset within ramp window (neg=before)",
    "  in_thanksgiving_window     1 if within ±7 days of Thanksgiving",
    "  days_to_christmas          Signed offset within ramp window (neg=before)",
    "  in_christmas_window        1 if within ±14 days of Christmas",
    "  is_black_friday            1 on the day after Thanksgiving",
    "  is_cyber_monday            1 on the Monday after Thanksgiving",
    "",
    f"STORE FEATURES  ({len(store_features)} columns)",
    "  region_enc                 0=aggregate 1=California 2=Texas 3=Wisconsin",
    "  store_mean / store_std     Revenue baseline and spread for that store",
    "  lag364_vs_store_mean       lag_364 / store_mean (relative scale)",
    "  rolling28_vs_store_mean    rolling_mean_28 / store_mean",
    "",
    f"INTERACTION FEATURES  ({len(interaction_features)} columns)",
    "  event_x_weekend            is_any_event × is_weekend",
    "  pre_xmas_weekend           Weekend AND Dec 12+ AND Q4",
    "  yoy_delta_364              revenue - lag_364 (YoY change; NaN in test)",
    "  lag7_vs_lag14              lag_7 - lag_14 (short-term momentum)",
    "",
    f"TOTAL FEATURE COUNT: {len(FEATURE_COLS)}",
    "",
    "MODELING NOTES",
    "  1. is_store_closed=1 → force prediction to 0 (Christmas, New Year)",
    "  2. Hierarchical: store_0 prediction should equal sum(stores 1–10)",
    "  3. For LightGBM/XGBoost: use lag_364+365+371 + date + event features",
    "     for a DIRECT forecast (no recursion required)",
    "  4. For recursive forecasting: predict day-by-day, fill short lags",
    "     from prior predictions before moving to the next day",
    "  5. Prophet: pass calendar events as custom holidays",
]

with open(f"{OUT}/feature_catalog.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(catalog_lines))

print("✓ Saved feature_catalog.txt")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  VALIDATION PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating validation plots …")

PALETTE = ["#2271B3","#E76F51","#2A9D8F","#8338EC","#F4A261",
           "#06D6A0","#EF233C","#4361EE","#F77F00","#7B2D8B"]

# ── Plot A: Feature correlation with revenue (training, store 0) ──────────────
s0_train = train_df[train_df["store_id"] == 0].copy()

numeric_feats = [c for c in FEATURE_COLS if s0_train[c].dtype in [np.float64, np.int64, int, float]
                 and s0_train[c].nunique() > 2]
corrs = (s0_train[numeric_feats + [TARGET_COL]]
         .corr()[TARGET_COL]
         .drop(TARGET_COL)
         .dropna()
         .sort_values())

top_n    = 20
top_corr = pd.concat([corrs.head(top_n//2), corrs.tail(top_n//2)])

fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#E76F51" if v > 0 else "#4aaed9" for v in top_corr.values]
ax.barh(top_corr.index, top_corr.values, color=colors, edgecolor="white", height=0.65)
ax.axvline(0, color="#333", linewidth=0.8)
ax.set_xlabel("Pearson Correlation with Revenue (Store 0, training set)")
ax.set_title(f"Top {top_n} Feature Correlations with Revenue")
fig.tight_layout()
fig.savefig(f"{OUT}/figA_feature_correlations.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("  ✓ figA_feature_correlations.png")


# ── Plot B: lag_364 vs actual revenue scatter ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
sample = s0_train.dropna(subset=["lag_364"]).sample(min(500, len(s0_train)), random_state=42)
ax.scatter(sample["lag_364"], sample["revenue"],
           alpha=0.35, s=18, color="#2271B3")
mn = min(sample["lag_364"].min(), sample["revenue"].min())
mx = max(sample["lag_364"].max(), sample["revenue"].max())
ax.plot([mn, mx], [mn, mx], color="#E76F51", linewidth=1.5, linestyle="--",
        label="y = x (perfect same-as-last-year)")
ax.set_xlabel("lag_364 (same day last year)")
ax.set_ylabel("Actual Revenue")
ax.set_title("lag_364 vs Revenue — Store 0\n(key feature for direct forecasting)")
ax.legend(fontsize=8)
corr_val = sample[["lag_364","revenue"]].corr().iloc[0,1]
ax.text(0.04, 0.95, f"r = {corr_val:.3f}", transform=ax.transAxes,
        va="top", bbox=dict(fc="white", ec="#ddd", pad=3))

ax = axes[1]
# DOW box plots coloured by lag_364 availability in test
dow_means = s0_train.groupby("dayofweek")["revenue"].mean()
colors_dow = ["#4aaed9"] * 5 + ["#E76F51"] * 2
ax.bar(range(7), dow_means.values, color=colors_dow, edgecolor="white", width=0.65)
ax.set_xticks(range(7))
ax.set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
ax.set_ylabel("Mean Revenue ($)")
ax.set_title("Revenue by Day-of-Week — Store 0\n(strongest single date feature)")
ax.legend(handles=[
    plt.Rectangle((0,0),1,1, color="#4aaed9", label="Weekday"),
    plt.Rectangle((0,0),1,1, color="#E76F51", label="Weekend"),
], loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/figB_key_features.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("  ✓ figB_key_features.png")


# ── Plot C: Holiday ramp features (days_to_thanksgiving, days_to_christmas) ───
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, holiday, col, window in [
    (axes[0], "Thanksgiving", "days_to_thanksgiving", (-7, 3)),
    (axes[1], "Christmas",    "days_to_christmas",    (-14, 1)),
]:
    ramp_rows = s0_train[s0_train[f"in_{holiday.lower()}_window"] == 1].copy()
    if ramp_rows.empty:
        continue
    agg = ramp_rows.groupby(col)["revenue"].mean().reset_index()
    agg = agg.sort_values(col)
    bar_colors = ["#E76F51" if x >= 0 else "#2271B3" for x in agg[col]]
    ax.bar(agg[col], agg["revenue"] / 1000, color=bar_colors, edgecolor="white", width=0.7)
    ax.axvline(0, color="#333", linewidth=1.2, linestyle="--")
    ax.set_xlabel(f"Days relative to {holiday} (0 = event day)")
    ax.set_ylabel("Mean Revenue ($k)")
    ax.set_title(f"{holiday} Revenue Ramp\n(blue = before, orange = after)")
    ax.text(0.02, 0.97, "← before   event day   after →",
            transform=ax.transAxes, va="top", fontsize=8, color="#666")

fig.suptitle("Pre/Post Holiday Revenue Ramp (All-Stores, all training years)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/figC_holiday_ramps.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("  ✓ figC_holiday_ramps.png")


# ── Plot D: Feature completeness for test set ─────────────────────────────────
completeness = (test_df[FEATURE_COLS].notna().mean() * 100).sort_values()

fig, ax = plt.subplots(figsize=(10, max(6, len(FEATURE_COLS) * 0.18)))
colors_c = ["#E76F51" if v < 100 else "#2A9D8F" for v in completeness.values]
ax.barh(completeness.index, completeness.values, color=colors_c, edgecolor="white", height=0.6)
ax.axvline(100, color="#333", linewidth=0.8, linestyle="--")
ax.set_xlim(0, 110)
ax.set_xlabel("% Non-NaN in test set")
ax.set_title("Feature Completeness for Test/Forecast Window\n(green=100% safe, orange=needs recursive fill)")
fig.tight_layout()
fig.savefig(f"{OUT}/figD_feature_completeness.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("  ✓ figD_feature_completeness.png")


# ── Print final summary ───────────────────────────────────────────────────────
print()
print("=" * 62)
print("  FEATURE ENGINEERING SUMMARY")
print("=" * 62)
print(f"\n  Total features   : {len(FEATURE_COLS)}")
print(f"  Date features    : {len(date_features)}")
print(f"  Lag features     : {len(lag_features)}")
print(f"  Rolling features : {len(rolling_features)}")
print(f"  Event features   : {len(event_features)}")
print(f"  Store features   : {len(store_features)}")
print(f"  Interaction feat : {len(interaction_features)}")

safe_in_test = [f for f in FEATURE_COLS if test_df[f].notna().all()]
needs_fill   = [f for f in FEATURE_COLS if test_df[f].isna().any()]
print(f"\n  Fully safe in test set    : {len(safe_in_test)}")
print(f"  Need recursive fill       : {len(needs_fill)}")
print(f"\n  Strongest correlators with revenue (store 0, r>0.6):")
strong = corrs[corrs.abs() > 0.6].sort_values(ascending=False)
for feat, val in strong.items():
    print(f"    {feat:35s}  r = {val:+.3f}")

print(f"\n  Files saved to {OUT}/")
print("=" * 62)
