"""
Phase 1 — Exploratory Data Analysis
Walmart Revenue Forecasting Competition

Run:  python eda.py
Outputs saved to /home/claude/eda_output/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")

# ── styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f8f6",
    "axes.grid":        True,
    "grid.color":       "white",
    "grid.linewidth":   0.8,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.labelsize":   11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "font.family":      "sans-serif",
    "lines.linewidth":  1.6,
})

OUT = "./eda_output"
os.makedirs(OUT, exist_ok=True)

PALETTE_10 = [
    "#2271B3","#E76F51","#2A9D8F","#8338EC","#F4A261",
    "#06D6A0","#EF233C","#4361EE","#F77F00","#7B2D8B",
]

# ── load data ─────────────────────────────────────────────────────────────────
DATA = "./competition_data"

train = pd.read_csv(f"{DATA}/train.csv", parse_dates=["date"])
cal   = pd.read_csv(f"{DATA}/calendar_events.csv", parse_dates=["date"])
sub   = pd.read_csv(f"{DATA}/forecast_submission.csv")

# Separate aggregate vs individual stores
all_stores = train[train["store_id"] == 0].copy()
ind_stores  = train[train["store_id"] != 0].copy()
store_ids   = sorted(ind_stores["store_id"].unique())    # [1..10]
store_names = train.groupby("store_id")["store_name"].first()

FORECAST_START = pd.Timestamp("2015-10-01")
FORECAST_END   = pd.Timestamp("2015-12-31")

print("=" * 60)
print("  Walmart Revenue Forecasting — EDA")
print("=" * 60)
print(f"Training period : {train['date'].min().date()} → {train['date'].max().date()}")
print(f"Forecast period : {FORECAST_START.date()} → {FORECAST_END.date()}")
print(f"Stores          : {train['store_id'].nunique()} (0=aggregate, 1–10=individual)")
print(f"Training rows   : {len(train):,}")
print(f"Submission rows : {len(sub):,}")
print(f"Calendar events : {len(cal)}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# FIG 1 — All-Stores Revenue + Event Markers
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(all_stores["date"], all_stores["revenue"],
        color="#2271B3", alpha=0.85, linewidth=1.2, label="All Stores (store 0)")

# shade forecast window
ax.axvspan(FORECAST_START, FORECAST_END, color="#E76F51", alpha=0.12, label="Forecast window")

# mark Q4 events in the final year
q4_events = cal[(cal["date"] >= FORECAST_START) & (cal["date"] <= FORECAST_END)]
for _, row in q4_events.iterrows():
    ax.axvline(row["date"], color="#E76F51", linewidth=1, linestyle="--", alpha=0.7)
    ax.text(row["date"], ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 380000,
            row["event"].split(",")[0], rotation=90, va="top",
            fontsize=7.5, color="#c0392b", alpha=0.85)

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_ylabel("Daily Revenue ($)")
ax.set_title("All-Stores Daily Revenue — Full Training History")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_all_stores_history.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("✓ fig1_all_stores_history.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 2 — Individual Store Time Series (2-col grid)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(5, 2, figsize=(16, 16), sharex=True)
axes = axes.flatten()

for idx, sid in enumerate(store_ids):
    df = ind_stores[ind_stores["store_id"] == sid].sort_values("date")
    axes[idx].plot(df["date"], df["revenue"],
                   color=PALETTE_10[idx], linewidth=0.9, alpha=0.85)
    axes[idx].axvspan(FORECAST_START, FORECAST_END, color="#E76F51", alpha=0.10)
    axes[idx].set_title(f"Store {sid} — {store_names[sid]}", fontsize=11)
    axes[idx].set_ylabel("Revenue ($)", fontsize=9)
    axes[idx].tick_params(axis="x", labelsize=8)
    axes[idx].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator())

fig.suptitle("Individual Store Daily Revenue (orange = forecast window)", y=1.002, fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_individual_stores.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("✓ fig2_individual_stores.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 — Weekly Seasonality: Day-of-Week Distributions
# ═══════════════════════════════════════════════════════════════════════════
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
train["dow"] = train["date"].dt.dayofweek

# for store 0 and individual stores separately
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# store 0
dow_s0 = all_stores.copy()
dow_s0["dow"] = dow_s0["date"].dt.dayofweek
ax = axes[0]
means = dow_s0.groupby("dow")["revenue"].mean()
colors = ["#4aaed9" if d < 5 else "#E76F51" for d in range(7)]
bars = ax.bar(dow_labels, means.values, color=colors, edgecolor="white", width=0.6)
for bar, val in zip(bars, means.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
            f"${val/1000:.0f}k", ha="center", va="bottom", fontsize=9)
ax.set_title("All Stores — Avg Revenue by Day of Week")
ax.set_ylabel("Mean Daily Revenue ($)")
ax.legend(handles=[
    plt.Rectangle((0,0),1,1, color="#4aaed9", label="Weekday"),
    plt.Rectangle((0,0),1,1, color="#E76F51", label="Weekend"),
], loc="lower right")

# weekend premium
wd = dow_s0[dow_s0["dow"] < 5]["revenue"].mean()
we = dow_s0[dow_s0["dow"] >= 5]["revenue"].mean()
ax.text(0.02, 0.97, f"Weekend premium: +{(we/wd-1)*100:.0f}%",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ddd"))

# individual stores — boxplot grouped by DOW
ax = axes[1]
plot_data = [ind_stores[ind_stores["date"].dt.dayofweek == d]["revenue"].values
             for d in range(7)]
bp = ax.boxplot(plot_data, labels=dow_labels, patch_artist=True,
                medianprops=dict(color="#333", linewidth=1.5),
                flierprops=dict(marker=".", markersize=2, alpha=0.3))
for i, (patch, label) in enumerate(zip(bp["boxes"], dow_labels)):
    patch.set_facecolor("#4aaed9" if i < 5 else "#E76F51")
    patch.set_alpha(0.7)
ax.set_title("Individual Stores — Revenue by Day of Week")
ax.set_ylabel("Daily Revenue ($)")

fig.suptitle("Weekly Seasonality Pattern", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_weekly_seasonality.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# print key stats
print(f"✓ fig3_weekly_seasonality.png  (Weekend premium all-stores: +{(we/wd-1)*100:.0f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4 — Monthly Seasonality Heatmap (year × month)
# ═══════════════════════════════════════════════════════════════════════════
all_stores["year"]  = all_stores["date"].dt.year
all_stores["month"] = all_stores["date"].dt.month

monthly = (all_stores
           .groupby(["year","month"])["revenue"]
           .mean()
           .unstack("month"))
monthly.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(13, 4))
sns.heatmap(monthly, annot=True, fmt=".0f", cmap="YlOrRd",
            linewidths=0.4, linecolor="white",
            ax=ax, annot_kws={"size": 9})
ax.set_title("All-Stores Mean Daily Revenue by Year × Month (missing = not in training)")
ax.set_ylabel("Year")
ax.set_xlabel("Month")
# highlight Q4 columns
for col_idx in [9, 10, 11]:   # Oct=9, Nov=10, Dec=11 (0-indexed)
    ax.add_patch(plt.Rectangle((col_idx, 0), 1, len(monthly),
                               fill=False, edgecolor="#E76F51", linewidth=2.5))
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_monthly_heatmap.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("✓ fig4_monthly_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 5 — Year-over-Year Q4 Comparison (your forecast target)
# ═══════════════════════════════════════════════════════════════════════════
q4_years = {}
for year in [2011, 2012, 2013, 2014]:
    mask = (all_stores["date"] >= f"{year}-10-01") & (all_stores["date"] <= f"{year}-12-31")
    df = all_stores[mask].copy()
    df["day_of_q4"] = (df["date"] - pd.Timestamp(f"{year}-10-01")).dt.days
    q4_years[year] = df

fig, axes = plt.subplots(2, 1, figsize=(14, 9))

# top: overlaid Q4 traces
ax = axes[0]
colors_yoy = ["#bbddf7","#7abfee","#3a9ad9","#0a6eb4"]
for year, color in zip([2011, 2012, 2013, 2014], colors_yoy):
    df = q4_years[year]
    ax.plot(df["day_of_q4"], df["revenue"], color=color,
            label=str(year), linewidth=1.5 if year < 2014 else 2.5,
            alpha=0.7 if year < 2014 else 1.0)

# add event markers for 2014 (the closest reference)
q4_2014_events = cal[(cal["date"] >= "2014-10-01") & (cal["date"] <= "2014-12-31")]
for _, row in q4_2014_events.iterrows():
    doy = (row["date"] - pd.Timestamp("2014-10-01")).days
    ax.axvline(doy, linestyle=":", color="#c0392b", linewidth=1, alpha=0.6)
    ax.text(doy + 0.5, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 390000,
            row["event"].split(",")[0], fontsize=7.5, color="#c0392b",
            rotation=90, va="top")

ax.set_title("Q4 (Oct–Dec) Revenue — Year-over-Year Overlay")
ax.set_xlabel("Day of Q4 (0 = Oct 1)")
ax.set_ylabel("All-Stores Revenue ($)")
ax.legend(title="Year", loc="upper left")
ax.set_xticks([0,30,61,91])
ax.set_xticklabels(["Oct 1","Nov 1","Dec 1","Dec 31"])

# bottom: % growth Q4 2012→2014
ax2 = axes[1]
for i, (yr_from, yr_to) in enumerate([(2011,2012),(2012,2013),(2013,2014)]):
    df_a = q4_years[yr_from].set_index("day_of_q4")["revenue"]
    df_b = q4_years[yr_to].set_index("day_of_q4")["revenue"]
    common = df_a.index.intersection(df_b.index)
    growth = ((df_b.loc[common] - df_a.loc[common]) / df_a.loc[common]) * 100
    ax2.plot(common, growth,
             color=PALETTE_10[i], linewidth=1.5, alpha=0.8,
             label=f"{yr_from}→{yr_to}")
ax2.axhline(0, color="#999", linewidth=0.8, linestyle="--")
ax2.set_title("Q4 Year-over-Year Revenue Growth (%)")
ax2.set_xlabel("Day of Q4")
ax2.set_ylabel("YoY Growth (%)")
ax2.set_xticks([0,30,61,91])
ax2.set_xticklabels(["Oct 1","Nov 1","Dec 1","Dec 31"])
ax2.legend(loc="upper left")

fig.tight_layout()
fig.savefig(f"{OUT}/fig5_yoy_q4_comparison.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("✓ fig5_yoy_q4_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 6 — Holiday Event Impact Analysis
# ═══════════════════════════════════════════════════════════════════════════
# measure average revenue on event days vs surrounding baseline window
event_impacts = []
ts0 = all_stores.set_index("date")["revenue"]

for _, row in cal.iterrows():
    ed = row["date"]
    if ed not in ts0.index:
        continue
    # 7-day window around the event (exclude event day itself)
    window = pd.date_range(ed - pd.Timedelta(days=7), ed + pd.Timedelta(days=7))
    window = [d for d in window if d in ts0.index and d != ed]
    if len(window) < 10:
        continue
    baseline  = ts0[window].mean()
    event_rev = ts0[ed]
    lift_pct  = (event_rev / baseline - 1) * 100
    event_impacts.append({
        "event":     row["event"].split(",")[0].strip(),
        "date":      ed,
        "revenue":   event_rev,
        "baseline":  baseline,
        "lift_pct":  lift_pct,
    })

impacts_df = pd.DataFrame(event_impacts)

# aggregate by event name (mean lift across years)
agg = (impacts_df
       .groupby("event")["lift_pct"]
       .agg(["mean","std","count"])
       .rename(columns={"mean":"avg_lift","std":"std_lift","count":"n_years"})
       .sort_values("avg_lift", ascending=True))

fig, ax = plt.subplots(figsize=(12, max(6, len(agg)*0.38)))
colors = ["#E76F51" if v > 0 else "#4aaed9" for v in agg["avg_lift"]]
bars = ax.barh(agg.index, agg["avg_lift"], color=colors, edgecolor="white", height=0.65)

# error bars
ax.errorbar(agg["avg_lift"], range(len(agg)),
            xerr=agg["std_lift"].fillna(0),
            fmt="none", color="#555", capsize=3, linewidth=1)

for bar, val in zip(bars, agg["avg_lift"]):
    ax.text(val + (1 if val >= 0 else -1),
            bar.get_y() + bar.get_height()/2,
            f"{val:+.1f}%", va="center", ha="left" if val >= 0 else "right",
            fontsize=8)

ax.axvline(0, color="#333", linewidth=0.8)
ax.set_xlabel("Average Revenue Lift vs 7-day Baseline (%)")
ax.set_title("Event Impact on All-Stores Revenue\n(error bars = ±1 std across years)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig6_event_impact.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# print top events
top5  = agg.nlargest(5, "avg_lift")
bot5  = agg.nsmallest(5, "avg_lift")
print("✓ fig6_event_impact.png")
print()
print("  Top 5 positive event lifts:")
for ev, row in top5.iterrows():
    print(f"    {ev:28s}  {row['avg_lift']:+.1f}% (n={int(row['n_years'])})")
print("  Top 5 negative event impacts:")
for ev, row in bot5.iterrows():
    print(f"    {ev:28s}  {row['avg_lift']:+.1f}% (n={int(row['n_years'])})")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 7 — STL Decomposition on All-Stores (weekly period)
# ═══════════════════════════════════════════════════════════════════════════
ts_full = all_stores.sort_values("date").set_index("date")["revenue"]

stl = STL(ts_full, period=7, robust=True)
result = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
components = {
    "Observed":  ts_full,
    "Trend":     result.trend,
    "Seasonal":  result.seasonal,
    "Residual":  result.resid,
}
colors_stl = ["#2271B3", "#2A9D8F", "#E76F51", "#8338EC"]

for ax, (label, series), color in zip(axes, components.items(), colors_stl):
    ax.plot(series.index, series.values, color=color, linewidth=0.9, alpha=0.85)
    ax.axvspan(FORECAST_START, FORECAST_END, color="#E76F51", alpha=0.08)
    ax.set_ylabel(label, fontsize=10)
    ax.yaxis.set_label_position("right")

axes[0].set_title("STL Decomposition — All Stores (weekly seasonality)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig7_stl_decomposition.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("\n✓ fig7_stl_decomposition.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 8 — Store Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════
pivot = (ind_stores
         .pivot_table(index="date", columns="store_id", values="revenue")
         .sort_index())
corr = pivot.corr()
# rename columns for readability
corr.index   = [f"Store {i}" for i in corr.index]
corr.columns = [f"Store {i}" for i in corr.columns]

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues",
            mask=mask, linewidths=0.4, linecolor="white",
            vmin=0.5, vmax=1.0, ax=ax, annot_kws={"size": 9})
ax.set_title("Pairwise Correlation Between Individual Stores\n(higher = more similar revenue patterns)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig8_store_correlation.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("✓ fig8_store_correlation.png")

# print correlation insights
min_corr = corr.values[~mask].min()
max_corr = corr.values[~mask].max()
print(f"   Correlation range: {min_corr:.2f} – {max_corr:.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 9 — Revenue Distribution per Store (violin)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))

plot_data = [ind_stores[ind_stores["store_id"] == sid]["revenue"].values
             for sid in store_ids]
parts = ax.violinplot(plot_data, positions=store_ids,
                      showmedians=True, widths=0.7)

for pc, color in zip(parts["bodies"], PALETTE_10):
    pc.set_facecolor(color)
    pc.set_alpha(0.6)
parts["cmedians"].set_color("#333")
parts["cmedians"].set_linewidth(2)
for comp in ["cmins","cmaxes","cbars"]:
    parts[comp].set_color("#666")
    parts[comp].set_linewidth(1)

# add mean labels
for sid in store_ids:
    m = ind_stores[ind_stores["store_id"] == sid]["revenue"].mean()
    ax.text(sid, m + 300, f"${m/1000:.0f}k", ha="center", va="bottom", fontsize=8)

ax.set_xticks(store_ids)
ax.set_xticklabels([f"Store {s}" for s in store_ids])
ax.set_ylabel("Daily Revenue ($)")
ax.set_title("Revenue Distribution by Store (violin = distribution, line = median)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig9_store_distributions.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("✓ fig9_store_distributions.png")


# ═══════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY STATS TABLE
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("  KEY STATS SUMMARY")
print("=" * 60)

summary = train.groupby(["store_id","store_name"])["revenue"].agg(
    mean="mean", std="std", min="min", max="max",
    cv=lambda x: x.std()/x.mean()
).round(0)
summary["cv"] = summary["cv"].round(3)
print("\nPer-store descriptive stats:")
print(summary.to_string())

# Q4 vs rest-of-year
all_stores["quarter"] = all_stores["date"].dt.quarter
q4_mean  = all_stores[all_stores["quarter"] == 4]["revenue"].mean()
rest_mean = all_stores[all_stores["quarter"] != 4]["revenue"].mean()
print(f"\nAll-Stores Q4 mean revenue  : ${q4_mean:,.0f}")
print(f"All-Stores non-Q4 mean rev  : ${rest_mean:,.0f}")
print(f"Q4 premium vs rest of year  : +{(q4_mean/rest_mean-1)*100:.1f}%")

# % variance explained by store 0
all_var   = all_stores["revenue"].var()
ind_var   = ind_stores["revenue"].var()
print(f"\nStd dev — All Stores (store 0) : ${all_stores['revenue'].std():,.0f}")
print(f"RMSE impact ratio (store 0 std / avg individual store std): "
      f"{all_stores['revenue'].std() / ind_stores.groupby('store_id')['revenue'].std().mean():.1f}×")

print()
print("=" * 60)
print(f"  All figures saved to {OUT}")
print("=" * 60)
