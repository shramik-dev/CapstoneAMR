"""
01_eda.py  --  Exploratory Data Analysis for the AMR forecasting project
===========================================================================

Builds the merged panel (via data_prep.build_panel) and produces:

  * A console EDA report (shape, coverage, missingness, describe, outliers)
  * UNIVARIATE graphs   -> figures in ./eda_figs/uni_*
  * BIVARIATE graphs    -> figures in ./eda_figs/bi_*

Data sources (all authentic, public):
  EARS-Net (resistance)  |  ESAC-Net (consumption)  |
  Eurostat degree days (climate)  |  Eurostat population density (demographics)

Run:  python 01_eda.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.getenv("PROJECT_ROOT", os.path.dirname(_HERE))
DATA = os.getenv("DATA_DIR",  os.path.join(ROOT, "data"))
OUT  = os.getenv("OUT_DIR",   os.path.join(ROOT, "outputs"))
FIGS = os.getenv("FIGS_DIR",  os.path.join(ROOT, "figures"))
sys.path.insert(0, _HERE)
os.makedirs(OUT, exist_ok=True)
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # file output, no display needed
import matplotlib.pyplot as plt

from data_prep import build_panel

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

OUT = os.path.join(FIGS, "eda")
os.makedirs(OUT, exist_ok=True)

DDD_COLS = ["ddd_J01A", "ddd_J01B", "ddd_J01C", "ddd_J01D", "ddd_J01E",
            "ddd_J01F", "ddd_J01G", "ddd_J01M", "ddd_J01R", "ddd_J01X"]
ATC_LABELS = {
    "ddd_J01A": "Tetracyclines", "ddd_J01B": "Amphenicols",
    "ddd_J01C": "Penicillins", "ddd_J01D": "Other beta-lactams",
    "ddd_J01E": "Sulfonamides/trim.", "ddd_J01F": "Macrolides/linc.",
    "ddd_J01G": "Aminoglycosides", "ddd_J01M": "Quinolones",
    "ddd_J01R": "Combinations", "ddd_J01X": "Other antibac.",
}
NUMERIC = ["resistance_pct", "ddd_total", "hdd", "cdd", "pop_density"] + DDD_COLS


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  saved", path)


# ===========================================================================
# 1. DATA OVERVIEW
# ===========================================================================
def data_overview(df):
    print("=" * 74)
    print("SECTION 1  DATA OVERVIEW")
    print("=" * 74)
    print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
    print(f"Years: {df.year.min()}-{df.year.max()}   "
          f"Countries: {df.country_code.nunique()}   "
          f"Pathogen-drug combos: {df.pathogen_drug.nunique()}")
    print("\nPathogen-drug combinations:")
    for c in sorted(df.pathogen_drug.unique()):
        print("   ", c)
    print("\nDtypes:")
    print(df.dtypes.to_string())

    print("\nMissing values (% of rows):")
    miss = (df.isna().mean() * 100).round(1)
    print(miss[miss > 0].sort_values(ascending=False).to_string() or "  none")

    print("\nNumeric summary (describe):")
    print(df[NUMERIC].describe().T.round(2).to_string())

    # coverage grid: how many country-years per combo
    print("\nObservations per pathogen-drug combo:")
    print(df.groupby("pathogen_drug").size().to_string())


# ===========================================================================
# 2. UNIVARIATE ANALYSIS
# ===========================================================================
def univariate(df):
    print("\n" + "=" * 74)
    print("SECTION 2  UNIVARIATE ANALYSIS")
    print("=" * 74)

    # --- 2a target distribution ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(df["resistance_pct"].dropna(), bins=30,
               color="#c0392b", edgecolor="white")
    ax[0].set(title="Resistance % — distribution", xlabel="Resistance (%)",
              ylabel="Count")
    ax[1].boxplot(df["resistance_pct"].dropna(), vert=True)
    ax[1].set(title="Resistance % — boxplot", ylabel="Resistance (%)")
    _save(fig, "uni_resistance.png")

    # --- 2b resistance by pathogen-drug (the whole point of the study) ---
    order = (df.groupby("pathogen_drug")["resistance_pct"]
               .median().sort_values().index.tolist())
    data = [df.loc[df.pathogen_drug == c, "resistance_pct"].dropna() for c in order]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=[c.replace("-", "\n") for c in order])
    ax.set(title="Resistance % by pathogen-drug combination",
           ylabel="Resistance (%)")
    ax.grid(alpha=.3, axis="y")
    _save(fig, "uni_resistance_by_combo.png")

    # --- 2c consumption: total DDD distribution + per-class means ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(df["ddd_total"].dropna(), bins=30,
               color="#2980b9", edgecolor="white")
    ax[0].set(title="Total antibiotic consumption (DDD/1000/day)",
              xlabel="DDD", ylabel="Count")
    class_means = df[DDD_COLS].mean().sort_values()
    ax[1].barh([ATC_LABELS[c] for c in class_means.index],
               class_means.values, color="#16a085")
    ax[1].set(title="Mean consumption by ATC class", xlabel="DDD/1000/day")
    _save(fig, "uni_consumption.png")

    # --- 2d covariates: climate + density ---
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, col, title, color in zip(
            ax, ["hdd", "cdd", "pop_density"],
            ["Heating degree days", "Cooling degree days", "Population density"],
            ["#8e44ad", "#e67e22", "#34495e"]):
        a.hist(df[col].dropna(), bins=25, color=color, edgecolor="white")
        a.set(title=title, ylabel="Count")
    _save(fig, "uni_covariates.png")

    # --- console: skew + outlier counts (IQR rule) ---
    print("\nSkewness & IQR-outlier counts:")
    rows = []
    for c in NUMERIC:
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
        rows.append((c, round(s.skew(), 2), out))
    print(pd.DataFrame(rows, columns=["variable", "skew", "n_outliers"])
          .to_string(index=False))


# ===========================================================================
# 3. BIVARIATE ANALYSIS
# ===========================================================================
def bivariate(df):
    print("\n" + "=" * 74)
    print("SECTION 3  BIVARIATE ANALYSIS")
    print("=" * 74)

    # --- 3a correlation heatmap (numeric block) ---
    corr = df[NUMERIC].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(NUMERIC)))
    ax.set_xticklabels(NUMERIC, rotation=90, fontsize=8)
    ax.set_yticks(range(len(NUMERIC)))
    ax.set_yticklabels(NUMERIC, fontsize=8)
    for i in range(len(NUMERIC)):
        for j in range(len(NUMERIC)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")
    fig.colorbar(im, fraction=.046, pad=.04)
    ax.set_title("Correlation matrix (numeric variables)")
    _save(fig, "bi_correlation_heatmap.png")

    print("\nPearson correlation of each predictor with resistance_pct:")
    tgt = corr["resistance_pct"].drop("resistance_pct").sort_values(key=abs,
                                                                    ascending=False)
    print(tgt.round(3).to_string())

    # --- 3b consumption vs resistance, contemporaneous, per combo ---
    combos = sorted(df.pathogen_drug.unique())
    fig, axes = plt.subplots(1, len(combos), figsize=(4 * len(combos), 4),
                             sharey=True)
    for ax, combo in zip(np.atleast_1d(axes), combos):
        sub = df[(df.pathogen_drug == combo)].dropna(
            subset=["ddd_total", "resistance_pct"])
        ax.scatter(sub.ddd_total, sub.resistance_pct, s=10, alpha=.35,
                   color="#c0392b")
        if len(sub) > 2:
            b, a = np.polyfit(sub.ddd_total, sub.resistance_pct, 1)
            xs = np.linspace(sub.ddd_total.min(), sub.ddd_total.max(), 50)
            ax.plot(xs, a + b * xs, color="black", lw=1)
            r = sub.ddd_total.corr(sub.resistance_pct)
            ax.set_title(f"{combo}\nr={r:.2f}", fontsize=8)
        ax.set_xlabel("Total DDD")
    np.atleast_1d(axes)[0].set_ylabel("Resistance (%)")
    fig.suptitle("Contemporaneous consumption vs resistance, by combo", y=1.03)
    _save(fig, "bi_consumption_vs_resistance.png")

    # --- 3c LAGGED consumption vs resistance (motivates RQ1) ---
    #     For each combo, correlate resistance_t against total DDD at lags 0..4.
    print("\nLagged corr(total DDD_{t-L}, resistance_t)  -- motivates RQ1:")
    lag_tab = []
    for combo in combos:
        sub = df[df.pathogen_drug == combo].sort_values(
            ["country_code", "year"]).copy()
        row = {"combo": combo}
        for L in range(0, 5):
            sub[f"lag{L}"] = sub.groupby("country_code")["ddd_total"].shift(L)
            s = sub.dropna(subset=[f"lag{L}", "resistance_pct"])
            row[f"lag{L}"] = round(s[f"lag{L}"].corr(s["resistance_pct"]), 3) \
                if len(s) > 5 else np.nan
        lag_tab.append(row)
    lag_df = pd.DataFrame(lag_tab).set_index("combo")
    print(lag_df.to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    for combo in combos:
        ax.plot(range(5), lag_df.loc[combo].values, marker="o", label=combo)
    ax.axhline(0, color="grey", lw=.8)
    ax.set(xlabel="Lag L (years)  consumption_{t-L} vs resistance_t",
           ylabel="Pearson r", title="Consumption-resistance correlation by lag")
    ax.set_xticks(range(5))
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    _save(fig, "bi_lag_correlation.png")

    # --- 3d resistance trend over time by combo (pooled median) ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for combo in combos:
        s = (df[df.pathogen_drug == combo]
             .groupby("year")["resistance_pct"].median())
        ax.plot(s.index, s.values, marker=".", label=combo)
    ax.set(xlabel="Year", ylabel="Median resistance (%)",
           title="Resistance trend over time (median across countries)")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    _save(fig, "bi_resistance_trend.png")

    # --- 3e climate vs resistance (does warmer -> more resistance?) ---
    sub = df.dropna(subset=["cdd", "resistance_pct"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(sub.cdd, sub.resistance_pct, s=8, alpha=.3, color="#e67e22")
    ax[0].set(xlabel="Cooling degree days", ylabel="Resistance (%)",
              title=f"CDD vs resistance (r={sub.cdd.corr(sub.resistance_pct):.2f})")
    sub2 = df.dropna(subset=["pop_density", "resistance_pct"])
    ax[1].scatter(sub2.pop_density, sub2.resistance_pct, s=8, alpha=.3,
                  color="#34495e")
    ax[1].set(xlabel="Population density", ylabel="Resistance (%)",
              title=f"Density vs resistance "
                    f"(r={sub2.pop_density.corr(sub2.resistance_pct):.2f})")
    _save(fig, "bi_climate_density_vs_resistance.png")


def main():
    df = build_panel(write="amr_panel.csv")
    data_overview(df)
    univariate(df)
    bivariate(df)
    print("\n" + "=" * 74)
    print(f"DONE. Panel written to amr_panel.csv. Figures in ./{OUT}/")
    print("=" * 74)


if __name__ == "__main__":
    main()
