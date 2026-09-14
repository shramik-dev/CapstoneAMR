"""
02_modeling.py  --  Model building for the AMR forecasting project
===================================================================

Answers the four research questions using the merged panel from data_prep,
with a proper THREE-WAY TEMPORAL split and expanding-window cross-validation
for hyperparameter tuning.

  RQ1  Optimal consumption -> resistance time lag per pathogen-drug combo.
  RQ2  How well country-level future resistance can be predicted from lagged
       consumption + demographic + climate covariates.
  RQ3  Which predictors matter most, and how that differs across pathogens.
  RQ4  Does a pooled cross-country ML model beat naive + single-country
       baselines on future, unseen years.

------------------------------------------------------------------
Data-splitting design (why it is done this way)
------------------------------------------------------------------
Forecasting is temporal, so we NEVER use a random train/test split -- that
would leak future years into training. Instead, per pathogen-drug combo:

    | ---------- TRAIN years ---------- | -- VALIDATION -- | -- TEST -- |
    (earliest ... T-TEST-VAL)            (tuning, via CV )   (final, once)

  * TEST  = the last TEST_YEARS years. Held out completely; scored ONCE at the
            end. Never seen during fitting or tuning.
  * The remaining "development" years are split again in time: hyperparameters
    are chosen by EXPANDING-WINDOW cross-validation (sklearn TimeSeriesSplit)
    over the development years -- train on an early block, validate on the next
    block, expand, repeat, average. More robust than one fixed validation year
    and mirrors real use (always predicting forward from all history so far).
  * The tuned model is then refit on ALL development years and evaluated once
    on TEST. That is the number we report.

Run:  python 02_modeling.py
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

from data_prep import build_panel

warnings.filterwarnings("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 60)

OUT = os.path.join(FIGS, "model")
os.makedirs(OUT, exist_ok=True)

DDD_COLS = ["ddd_J01A", "ddd_J01C", "ddd_J01D", "ddd_J01E",
            "ddd_J01F", "ddd_J01G", "ddd_J01M", "ddd_J01X"]
COVARS = ["hdd", "cdd", "pop_density"]
MAX_LAG = 5
TEST_YEARS = 3          # final years held out for the one-shot test
N_CV_SPLITS = 4         # expanding-window folds used for tuning

RF_GRID = {"n_estimators": [300],
           "min_samples_leaf": [1, 2, 4],
           "max_features": ["sqrt"]}
GBR_GRID = {"n_estimators": [300],
            "max_depth": [2, 3],
            "learning_rate": [0.03, 0.05, 0.1]}


def _save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  saved", p)


def rmse(y, yhat):
    return float(np.sqrt(mean_squared_error(y, yhat)))


# ===========================================================================
# Feature engineering
# ===========================================================================
def make_lagged(df, combo, max_lag=MAX_LAG, autoregressive=True):
    """One combo's panel with consumption + covariates lagged 1..max_lag.

    autoregressive=True  -> include resistance_{t-1} (best raw accuracy, but it
                            dominates and turns the model into persistence).
    autoregressive=False -> covariates only; used for RQ3/RQ4 attribution so
                            consumption/climate/demographics carry the signal.
    Rows are returned sorted by (country, year), which the time-ordered
    splitters below rely on.
    """
    d = df[df.pathogen_drug == combo].sort_values(
        ["country_code", "year"]).copy()
    feat_cols = []
    for lag in range(1, max_lag + 1):
        for c in DDD_COLS:
            col = f"{c}_lag{lag}"
            d[col] = d.groupby("country_code")[c].shift(lag)
            feat_cols.append(col)
    feat_cols += COVARS
    d["resist_lag1"] = d.groupby("country_code")["resistance_pct"].shift(1)
    if autoregressive:
        feat_cols.append("resist_lag1")
    return d, feat_cols


# ===========================================================================
# Three-way temporal split helpers
# ===========================================================================
def dev_test_split(d, feat_cols, test_years=TEST_YEARS):
    """Return (development_df, test_df, test_start_year).
    TEST = last `test_years` calendar years; development = everything before."""
    d = d.dropna(subset=feat_cols + ["resistance_pct"]).sort_values(
        ["year", "country_code"])
    test_start = d.year.max() - test_years + 1
    dev = d[d.year < test_start]
    test = d[d.year >= test_start]
    return dev, test, int(test_start)


def year_ordered_cv(dev, n_splits=N_CV_SPLITS):
    """Expanding-window CV *by row position* on year-sorted dev data.
    Because rows are year-sorted, each validation fold is later in time than
    its training fold. Returns (sorted_dev, folds) for GridSearchCV's cv=."""
    dev = dev.sort_values(["year", "country_code"]).reset_index(drop=True)
    n_splits = min(n_splits, max(2, dev.year.nunique() - 1))
    tss = TimeSeriesSplit(n_splits=n_splits)
    return dev, list(tss.split(dev))


def tune(model, grid, dev, feat_cols):
    """Grid-search over `grid` with expanding-window CV on `dev`.
    Returns (best_estimator refit on all dev, best_params)."""
    dev_sorted, folds = year_ordered_cv(dev)
    gs = GridSearchCV(model, grid, cv=folds,
                      scoring="neg_mean_absolute_error", n_jobs=-1, refit=True)
    gs.fit(dev_sorted[feat_cols], dev_sorted["resistance_pct"])
    return gs.best_estimator_, gs.best_params_


# ===========================================================================
# RQ1
# ===========================================================================
def rq1_optimal_lag(df):
    print("=" * 74)
    print("RQ1  OPTIMAL CONSUMPTION -> RESISTANCE LAG (per pathogen-drug)")
    print("=" * 74)
    combos = sorted(df.pathogen_drug.unique())
    best = {}
    fig, ax = plt.subplots(figsize=(9, 5))
    for combo in combos:
        d = df[df.pathogen_drug == combo].sort_values(
            ["country_code", "year"]).copy()
        corrs = []
        for lag in range(0, MAX_LAG + 1):
            d[f"c{lag}"] = d.groupby("country_code")["ddd_total"].shift(lag)
            s = d.dropna(subset=[f"c{lag}", "resistance_pct"])
            g = s.groupby("country_code")
            x = s[f"c{lag}"] - g[f"c{lag}"].transform("mean")
            y = s["resistance_pct"] - g["resistance_pct"].transform("mean")
            corrs.append(x.corr(y) if len(s) > 10 else np.nan)
        best[combo] = int(np.nanargmax(np.abs(corrs)))
        ax.plot(range(MAX_LAG + 1), corrs, marker="o", label=combo)
    ax.axhline(0, color="grey", lw=.8)
    ax.set(xlabel="Lag (years)", ylabel="Within-country partial correlation",
           title="RQ1: consumption-resistance correlation vs lag")
    ax.set_xticks(range(MAX_LAG + 1))
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    _save(fig, "rq1_optimal_lag.png")
    print("Best (|correlation|) lag per combo:")
    for k, v in best.items():
        print(f"   {k:35s} lag {v}")
    return best


# ===========================================================================
# RQ2
# ===========================================================================
def rq2_forecast_accuracy(df):
    print("\n" + "=" * 74)
    print("RQ2  FORECAST ACCURACY  (tuned on expanding-window CV, tested once)")
    print(f"     TEST = last {TEST_YEARS} years; tuning via up to {N_CV_SPLITS}"
          "-fold expanding-window CV on earlier years")
    print("     [AR]  includes resistance_{t-1};  [COV] covariates only")
    print("=" * 74)
    combos = sorted(df.pathogen_drug.unique())
    rows, preds_store = [], {}
    for combo in combos:
        entry = {"combo": combo}
        for tag, ar in [("AR", True), ("COV", False)]:
            d, feat_cols = make_lagged(df, combo, autoregressive=ar)
            dev, test, tstart = dev_test_split(d, feat_cols)
            if len(test) < 5 or dev.year.nunique() < 3:
                continue
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            best_rf, params = tune(rf, RF_GRID, dev, feat_cols)
            pred = best_rf.predict(test[feat_cols])
            entry[f"MAE_{tag}"] = mean_absolute_error(test.resistance_pct, pred)
            entry[f"R2_{tag}"] = r2_score(test.resistance_pct, pred)
            entry["test_from"] = tstart
            entry["n_dev"] = len(dev)
            entry["n_test"] = len(test)
            if tag == "COV":
                preds_store[combo] = (test, pred)
                entry["best_params"] = params
        rows.append(entry)

    res = pd.DataFrame(rows).set_index("combo")
    print(res[["test_from", "n_dev", "n_test",
               "MAE_AR", "R2_AR", "MAE_COV", "R2_COV"]].round(3).to_string())
    print("\nChosen hyperparameters (covariate model):")
    for combo in res.index:
        print(f"   {combo:35s} {res.loc[combo, 'best_params']}")

    fig, ax = plt.subplots(figsize=(6, 6))
    for combo, (te, pred) in preds_store.items():
        ax.scatter(te["resistance_pct"], pred, s=14, alpha=.5, label=combo)
    lim = [0, max(df.resistance_pct.max(), 1)]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlabel="Actual resistance (%)", ylabel="Predicted resistance (%)",
           title="RQ2: covariate model, predicted vs actual (TEST years)")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    _save(fig, "rq2_pred_vs_actual.png")
    return res


# ===========================================================================
# RQ3
# ===========================================================================
def rq3_importance(df):
    print("\n" + "=" * 74)
    print("RQ3  PREDICTOR IMPORTANCE PER PATHOGEN (permutation importance)")
    print("     covariate-only, tuned model; importance measured on TEST years")
    print("=" * 74)
    combos = sorted(df.pathogen_drug.unique())
    imp_by_combo = {}
    for combo in combos:
        d, feat_cols = make_lagged(df, combo, autoregressive=False)
        dev, test, _ = dev_test_split(d, feat_cols)
        if len(test) < 5 or dev.year.nunique() < 3:
            continue
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        best_rf, _ = tune(rf, RF_GRID, dev, feat_cols)
        pi = permutation_importance(best_rf, test[feat_cols], test.resistance_pct,
                                    n_repeats=20, random_state=42, n_jobs=-1)
        imp = pd.Series(pi.importances_mean, index=feat_cols)
        imp_by_combo[combo] = imp
        print(f"\n{combo}  -- top 8 predictors:")
        print(imp.sort_values(ascending=False).head(8).round(4).to_string())

    def family(name):
        if name == "resist_lag1":
            return "autoregressive"
        if name in COVARS:
            return "climate" if name in ("hdd", "cdd") else "demographic"
        return "consumption"

    fam_tab = {}
    for combo, imp in imp_by_combo.items():
        fam = imp.groupby(family).sum()
        fam_tab[combo] = fam / fam.sum()
    fam_df = pd.DataFrame(fam_tab).T.fillna(0)
    print("\nShare of importance by predictor family (rows sum to 1):")
    print(fam_df.round(3).to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(fam_df))
    for col in fam_df.columns:
        ax.bar(fam_df.index, fam_df[col], bottom=bottom, label=col)
        bottom += fam_df[col].values
    ax.set_ylabel("Share of permutation importance")
    ax.set_title("RQ3: predictor-family contribution by pathogen")
    ax.set_xticklabels(fam_df.index, rotation=25, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    _save(fig, "rq3_family_importance.png")
    return fam_df


# ===========================================================================
# RQ4
# ===========================================================================
def rq4_pooled_vs_baselines(df):
    print("\n" + "=" * 74)
    print("RQ4  POOLED (tuned) ML vs NAIVE and SINGLE-COUNTRY BASELINES")
    print("     all models covariate-only; evaluated on the held-out TEST years")
    print("=" * 74)
    combos = sorted(df.pathogen_drug.unique())
    rows = []
    for combo in combos:
        d, feat_cols = make_lagged(df, combo, autoregressive=False)
        d = d.dropna(subset=feat_cols + ["resistance_pct", "resist_lag1"])
        dev, test, _ = dev_test_split(d, feat_cols)
        if len(test) < 5 or dev.year.nunique() < 3:
            continue

        # baseline 1: naive persistence
        naive_pred = test["resist_lag1"].values
        naive_mae = mean_absolute_error(test.resistance_pct, naive_pred)

        # baseline 2: single-country Ridge (alpha tuned on dev when possible)
        sc_true, sc_pred = [], []
        for cc, gdev in dev.groupby("country_code"):
            gte = test[test.country_code == cc]
            if len(gdev) < 6 or gte.empty:
                continue
            if gdev.year.nunique() > 2:
                best_ridge, _ = tune(Ridge(), {"alpha": [1.0, 10.0, 50.0]},
                                     gdev, feat_cols)
            else:
                best_ridge = Ridge(alpha=10.0).fit(gdev[feat_cols],
                                                   gdev.resistance_pct)
            sc_pred += list(best_ridge.predict(gte[feat_cols]))
            sc_true += list(gte.resistance_pct)
        single_mae = mean_absolute_error(sc_true, sc_pred) \
            if len(sc_true) >= 5 else np.nan

        # pooled ML: gradient boosting, tuned by expanding-window CV
        best_gbr, gparams = tune(GradientBoostingRegressor(random_state=42),
                                 GBR_GRID, dev, feat_cols)
        gb_pred = best_gbr.predict(test[feat_cols])
        pooled_mae = mean_absolute_error(test.resistance_pct, gb_pred)

        rows.append({"combo": combo, "naive_MAE": naive_mae,
                     "single_MAE": single_mae, "pooled_MAE": pooled_mae,
                     "pooled_params": gparams})

    res = pd.DataFrame(rows).set_index("combo")
    print(res[["naive_MAE", "single_MAE", "pooled_MAE"]].round(3).to_string())
    print("\nSkill vs naive (positive = pooled better):")
    skill = (res.naive_MAE - res.pooled_MAE) / res.naive_MAE * 100
    print(skill.round(1).astype(str).add(" %").to_string())

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(res)); w = 0.27
    ax.bar(x - w, res.naive_MAE, w, label="Naive persistence")
    ax.bar(x,     res.single_MAE, w, label="Single-country Ridge")
    ax.bar(x + w, res.pooled_MAE, w, label="Pooled ML (GBR, tuned)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("-", "\n") for c in res.index], fontsize=7)
    ax.set_ylabel("MAE on held-out TEST years")
    ax.set_title("RQ4: pooled ML vs baseline forecasts")
    ax.legend()
    ax.grid(alpha=.3, axis="y")
    _save(fig, "rq4_model_comparison.png")
    return res


def main():
    df = build_panel(write="amr_panel.csv")
    rq1_optimal_lag(df)
    rq2_forecast_accuracy(df)
    rq3_importance(df)
    rq4_pooled_vs_baselines(df)
    print("\n" + "=" * 74)
    print(f"DONE. Figures in {OUT}")
    print("=" * 74)


if __name__ == "__main__":
    main()
