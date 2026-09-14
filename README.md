# Forecasting Antimicrobial Resistance in the EU/EEA

An integrated consumption–lag, demographic, and climate pipeline for forecasting
country-level antibiotic resistance across the EU/EEA.

**QM640: Data Analytics Capstone — Walsh College**
Author: Shramik Wavekar · Mentor: Vikas S

---

## What this project does

Antimicrobial resistance is associated with more than 35,000 attributable deaths a
year in the EU/EEA. Antibiotic consumption is the main modifiable driver, but its
effect is **delayed** — a change in prescribing shows up in resistance rates years
later. Most published analyses correlate consumption and resistance within the same
calendar year and therefore miss this.

This project builds a lag-aware forecasting pipeline that:

1. merges four authentic public data sources into one country–year panel,
2. estimates the optimal consumption→resistance lag per pathogen–drug combination,
3. forecasts next-year resistance from lagged consumption plus climate and demographic covariates, and
4. benchmarks the learned models against a naive persistence baseline under strict temporal validation.

### Research questions

| RQ | Question |
|----|----------|
| RQ1 | What is the optimal time lag between antibiotic consumption and the corresponding change in resistance? |
| RQ2 | How well can future resistance be predicted from lagged consumption plus demographic and climate covariates? |
| RQ3 | Which predictors contribute most, and does this differ across pathogens? |
| RQ4 | Does a pooled cross-country model beat naive and single-country baselines on unseen years? |

---

## Repository structure

```
Capstone/
├── data/                                       raw public data (unmodified)
│   ├── ears_combined.csv                       EARS-Net resistance (%)
│   ├── consumption_ALL_with_Germany.csv        ESAC-Net consumption (DDD)
│   ├── nrg_chdd_a_defaultview_spreadsheet.xlsx Eurostat heating/cooling degree days
│   └── demo_r_d3dens__custom_22308386_spreadsheet.xlsx  Eurostat population density
├── src/
│   ├── data_prep.py        loads, harmonises and merges all four sources
│   ├── 01_eda.py           univariate, bivariate and temporal analysis
│   ├── 02_modeling.py      lag features, temporal split, CV, RQ1–RQ4
│   ├── stats.py            hypothesis tests and power analysis
│   ├── unified.py          single comparable experiment across all four models
│   └── gaps.py             per-RQ sample size, IQR/kurtosis, paired model tests
├── figures/
│   ├── eda/                figures produced by 01_eda.py
│   ├── model/              figures produced by 02_modeling.py
│   └── report/             figures used in the written report
├── outputs/
│   ├── amr_panel.csv       merged analytical panel (3,257 rows × 18 columns)
│   └── *.csv               result tables produced by the scripts
├── requirements.txt
└── README.md
```

---

## Data sources

All four are public and require no authentication.

| Source | Variable | Link |
|--------|----------|------|
| ECDC EARS-Net | Resistance % by country, year, pathogen–drug | https://atlas.ecdc.europa.eu/public/ |
| ECDC ESAC-Net | Consumption in DDD per 1,000 inhabitants/day by ATC class | https://www.ecdc.europa.eu/en/antimicrobial-consumption/surveillance-and-disease-data/database |
| Eurostat `demo_r_d3dens` | Population density | https://ec.europa.eu/eurostat/databrowser/ |
| Eurostat `nrg_chdd_a` | Heating and cooling degree days | https://ec.europa.eu/eurostat/databrowser/ |

**Panel:** 3,257 country–year–combination rows · 30 EU/EEA countries · 2000–2024 ·
5 pathogen–antibiotic combinations (E. coli vs fluoroquinolones, E. coli vs 3rd-gen
cephalosporins, K. pneumoniae vs carbapenems, S. aureus vs methicillin/MRSA,
S. pneumoniae vs macrolides).

---

## Running on Google Colab (easiest)

Open **`AMR_Capstone_Colab.ipynb`** in Colab and run the cells in order:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shramik-dev/Capstone/blob/main/AMR_Capstone_Colab.ipynb)

Cell 1 installs dependencies and clones this repository. If the clone fails (private
repo, no network), set `USE_UPLOAD = True` in Cell 1 and upload `Capstone.zip`
instead. Every figure renders inline; the last cell zips the results for download.

Cells 5–7 run grid-search tuning across five pathogen–drug combinations and take
several minutes each — this is expected.

---

## Setup (local)

```bash
git clone https://github.com/shramik-dev/Capstone.git
cd Capstone
pip install -r requirements.txt
```

Python 3.9 or later. CPU only — no GPU required.

## Running the pipeline

Run in order from the repository root:

```bash
python src/data_prep.py     # builds outputs/amr_panel.csv
python src/01_eda.py        # EDA figures  -> figures/eda/
python src/02_modeling.py   # RQ1–RQ4      -> figures/model/
```

Optional supporting analyses (these reproduce specific report tables):

```bash
python src/stats.py         # Shapiro–Wilk, ANOVA, Kruskal–Wallis, Spearman, power
python src/unified.py       # one comparable experiment across all four models
python src/gaps.py          # per-RQ sample size, IQR/kurtosis, paired model tests
```

`02_modeling.py` performs grid-search tuning across five pathogen–drug combinations
and takes several minutes.

---

## Method summary

**Feature engineering.** Consumption is lagged 1–5 years *within each country*,
producing 40 lagged predictors across eight ATC classes, plus heating degree days,
cooling degree days and population density (43 predictors total). The per-country
shift guarantees no value crosses a national boundary and no feature is dated at or
after its prediction target.

**Validation.** A strict temporal train–validation–test framework. The final three
years are held out and scored exactly once; hyperparameters are selected on earlier
years by expanding-window cross-validation (`TimeSeriesSplit`). A random split is
never used, since it would leak future information.

> **Lag ≠ horizon.** The task is *one-year-ahead* prediction. "Lag 1–5" describes how
> far back the predictors reach, not how far forward the model forecasts.

**Models.** Naive last-value baseline · Ridge regression · Random forest ·
Gradient-boosted trees. All four share the identical temporal protocol.

**Metrics.** MAE, RMSE, R², and directional accuracy, defined as the share of
held-out country–years where `sign(ŷ_t − y_{t−1}) = sign(y_t − y_{t−1})`.

---

## Headline results

| Model | MAE | RMSE | R² | Directional accuracy |
|-------|-----|------|-----|----------------------|
| Naive last-value | 1.96 | 3.02 | 0.908 | 0.000 |
| Ridge regression | 4.92 | 6.74 | 0.670 | 0.516 |
| **Random forest** | **3.82** | **5.85** | **0.733** | **0.546** |
| Gradient boosting | 4.10 | 6.29 | 0.674 | 0.497 |

*Held-out test years; MAE and RMSE in percentage points.*

**The key finding is a trade-off, not a clean win.** The naive baseline has the best
R² and the lowest error — significantly so (paired t-tests, all p < .001) — because
resistance is strongly autocorrelated. But its directional accuracy is zero by
construction: it never predicts change. The learned models trade error for the
ability to anticipate whether resistance will rise or fall, which is what a
stewardship early-warning tool actually needs.

**On the lag (RQ1).** Pooled correlations peak at 1–5 year lags and are strong
(r up to 0.716). However, once each country's mean is removed, within-country
correlations fall to between −0.17 and +0.22. Most of the pooled association is
*cross-country* — nations that prescribe more have more resistance — rather than
year-to-year dynamics within a country. This is reported openly rather than set aside.

---

## Known limitations

- Consumption is reported by ATC class while resistance is drug-specific; carbapenems and cephalosporins share group J01D.
- S. aureus–MRSA has no clean single-class consumption driver and is excluded from the class-specific lag mapping.
- Degree-day data covers only 2016–2024, so the climate contribution is underpowered.
- Short interior gaps in consumption were linearly interpolated before the temporal split, which carries limited look-ahead risk for those cells.
- Repeated yearly measurements per country are not independent, so reported p-values are approximate.
- The design is observational: it establishes association consistent with a known biological mechanism, not causation.

---

## License and attribution

Code released under the  License (see `LICENSE`).

The data files in `data/` are public exports from ECDC and Eurostat and remain
subject to their original terms of use. Please cite the original sources rather than
this repository when reusing the data.
