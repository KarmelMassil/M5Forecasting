"""
config.py — Edit the paths here, everything else works automatically.

Set BASE_DIR to the folder where you'll put train.csv, calendar_events.csv,
and forecast_submission.csv.
"""

import pathlib, os

# ── The one thing you need to change ─────────────────────────────────────────
# Point this to the folder that contains train.csv, calendar_events.csv,
# and forecast_submission.csv (the competition data files).

BASE_DIR = pathlib.Path(__file__).parent   # same folder as config.py by default

# ── Everything below is derived automatically — no need to change ─────────────
DATA_DIR    = BASE_DIR / "competition_data"   # raw CSVs go here
FEATURES_DIR= BASE_DIR / "features"           # generated feature CSVs
MODELS_DIR  = BASE_DIR / "models"             # plots, submission, validation
PARAMS_JSON = BASE_DIR / "best_lgbm_params.json"  # Optuna output

# Source data files
TRAIN_CSV      = DATA_DIR / "train.csv"
CALENDAR_CSV   = DATA_DIR / "calendar_events.csv"
SUBMISSION_CSV = DATA_DIR / "forecast_submission.csv"

# Create output directories if they don't exist
for d in [DATA_DIR, FEATURES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Sanity check ──────────────────────────────────────────────────────────────
def check_data():
    missing = [str(f) for f in [TRAIN_CSV, CALENDAR_CSV, SUBMISSION_CSV] if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"\n  Missing data files:\n" +
            "\n".join(f"    {m}" for m in missing) +
            f"\n\n  Put train.csv, calendar_events.csv, and forecast_submission.csv"
            f"\n  in:  {DATA_DIR}\n"
        )
    return True
