"""
data_prep.py  -- Load & clean the four Eurostat/ECDC sources into ONE tidy panel.

Outputs a long panel keyed by (country_code, year, pathogen_drug) with:
  resistance_pct            -- target (from EARS-Net)
  ddd_J01A ... ddd_J01X     -- antibiotic consumption per ATC class (ESAC-Net)
  ddd_total                 -- total antibiotic consumption
  hdd, cdd                  -- heating / cooling degree days (climate)
  pop_density               -- persons / km^2 (demographics)

Run standalone to write  amr_panel.csv  and print a summary.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.getenv("PROJECT_ROOT", os.path.dirname(_HERE))
DATA = os.getenv("DATA_DIR",  os.path.join(ROOT, "data"))
OUT  = os.getenv("OUT_DIR",   os.path.join(ROOT, "outputs"))
FIGS = os.getenv("FIGS_DIR",  os.path.join(ROOT, "figures"))
sys.path.insert(0, _HERE)
os.makedirs(OUT, exist_ok=True)
import re
import numpy as np
import pandas as pd

# ---- country name -> ISO2 code harmonisation ----------------------------------
NAME2CODE = {
    "Austria":"AT","Belgium":"BE","Bulgaria":"BG","Croatia":"HR","Cyprus":"CY",
    "Czechia":"CZ","Czech Republic":"CZ","Denmark":"DK","Estonia":"EE","Finland":"FI",
    "France":"FR","Germany":"DE","Greece":"EL","Hungary":"HU","Iceland":"IS",
    "Ireland":"IE","Italy":"IT","Latvia":"LV","Lithuania":"LT","Luxembourg":"LU",
    "Malta":"MT","Netherlands":"NL","Norway":"NO","Poland":"PL","Portugal":"PT",
    "Romania":"RO","Slovakia":"SK","Slovenia":"SI","Spain":"ES","Sweden":"SE",
    "Switzerland":"CH","United Kingdom":"UK","Liechtenstein":"LI",
}

def _num(s):
    """Coerce Eurostat cells (':' = missing, may carry flag letters) to float."""
    return pd.to_numeric(
        s.astype(str).str.replace(":", "", regex=False).str.strip().replace("", np.nan),
        errors="coerce",
    )

# ---- 1. EARS-Net resistance ---------------------------------------------------
def load_resistance(path=None):
    path = path or os.getenv("EARS_PATH", os.path.join(DATA, "ears_combined.csv"))
    df = pd.read_csv(path)
    df = df.rename(columns=str.lower)
    df["country_code"] = df["country_code"].replace({"GR": "EL", "GB": "UK"})
    df["year"] = df["year"].astype(int)
    return df[["country_code", "year", "pathogen_drug", "resistance_pct"]]

# ---- 2. ESAC-Net consumption (long -> wide by ATC class) ----------------------
def load_consumption(path=None):
    path = path or os.getenv("CONSUMPTION_PATH", os.path.join(DATA, "consumption_ALL_with_Germany.csv"))
    df = pd.read_csv(path)
    df["country_code"] = df["Country"].map(NAME2CODE)
    df = df.dropna(subset=["country_code"])
    df["Year"] = df["Year"].astype(int)
    wide = (df.pivot_table(index=["country_code", "Year"],
                           columns="atc_level", values="ddd", aggfunc="sum")
              .reset_index().rename(columns={"Year": "year"}))
    wide.columns = [c if c in ("country_code", "year") else f"ddd_{c}"
                    for c in wide.columns]
    ddd_cols = [c for c in wide.columns if c.startswith("ddd_")]
    wide["ddd_total"] = wide[ddd_cols].sum(axis=1)
    return wide

# ---- 3. Climate: HDD + CDD (wide Eurostat sheets) -----------------------------
def _load_chdd_sheet(path, sheet, valname):
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    # locate the TIME header row
    trow = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "TIME"][0]
    years = _num(raw.iloc[trow]).tolist()
    ycols = {i: int(years[i]) for i in range(len(years)) if pd.notna(years[i])}
    start = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "GEO (Labels)"][0] + 1
    recs = []
    for r in range(start, len(raw)):
        name = str(raw.iat[r, 0]).strip()
        if name in ("", "nan", "Special value", ":"):
            continue
        code = NAME2CODE.get(name)
        if code is None:
            continue
        for ci, yr in ycols.items():
            recs.append((code, yr, _num(pd.Series([raw.iat[r, ci]]))[0]))
    return pd.DataFrame(recs, columns=["country_code", "year", valname])

def load_climate(path=None):
    path = path or os.getenv("CLIMATE_PATH", os.path.join(DATA, "nrg_chdd_a_defaultview_spreadsheet.xlsx"))
    hdd = _load_chdd_sheet(path, "Sheet 1", "hdd")
    cdd = _load_chdd_sheet(path, "Sheet 2", "cdd")
    return hdd.merge(cdd, on=["country_code", "year"], how="outer")

# ---- 4. Demographics: population density (country = 2-letter GEO code) ---------
def load_density(path=None):
    path = path or os.getenv("DENSITY_PATH", os.path.join(DATA, "demo_r_d3dens__custom_22308386_spreadsheet.xlsx"))
    raw = pd.read_excel(path, sheet_name="Sheet 1", header=None)
    trow = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "TIME"][0]
    years = _num(raw.iloc[trow]).tolist()
    ycols = {i: int(years[i]) for i in range(len(years)) if pd.notna(years[i])}
    recs = []
    for r in range(len(raw)):
        code = str(raw.iat[r, 0]).strip()
        if not re.fullmatch(r"[A-Z]{2}", code):      # NUTS-0 only
            continue
        code = {"GR": "EL", "GB": "UK"}.get(code, code)
        for ci, yr in ycols.items():
            recs.append((code, yr, _num(pd.Series([raw.iat[r, ci]]))[0]))
    return pd.DataFrame(recs, columns=["country_code", "year", "pop_density"])

# ---- master builder -----------------------------------------------------------
def build_panel(write="amr_panel.csv"):
    if write:
        write = os.path.join(OUT, os.path.basename(write))
    res  = load_resistance()
    cons = load_consumption()
    clim = load_climate()
    dens = load_density()

    panel = (res
             .merge(cons, on=["country_code", "year"], how="left")
             .merge(clim, on=["country_code", "year"], how="left")
             .merge(dens, on=["country_code", "year"], how="left")
             .sort_values(["pathogen_drug", "country_code", "year"])
             .reset_index(drop=True))
    if write:
        panel.to_csv(write, index=False)
    return panel

if __name__ == "__main__":
    p = build_panel()
    print("PANEL:", p.shape)
    print("years:", p.year.min(), "-", p.year.max(),
          "| countries:", p.country_code.nunique(),
          "| combos:", p.pathogen_drug.nunique())
    print("\ncolumns:", list(p.columns))
    print("\nmissing (%):")
    print((p.isna().mean() * 100).round(1).to_string())
    print("\nsample:")
    print(p.head(4).to_string())
