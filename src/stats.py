import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.getenv("PROJECT_ROOT", os.path.dirname(_HERE))
DATA = os.getenv("DATA_DIR",  os.path.join(ROOT, "data"))
OUT  = os.getenv("OUT_DIR",   os.path.join(ROOT, "outputs"))
FIGS = os.getenv("FIGS_DIR",  os.path.join(ROOT, "figures"))
sys.path.insert(0, _HERE)
os.makedirs(OUT, exist_ok=True)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats

df = pd.read_csv(os.path.join(OUT,"amr_panel.csv"))
r = df.resistance_pct.dropna()
print("=== TARGET ===")
print(f"n={len(r)} mean={r.mean():.2f} median={r.median():.2f} sd={r.std():.2f} skew={r.skew():.2f}")
W,pw = stats.shapiro(r.sample(min(len(r),4000), random_state=42))
print(f"Shapiro-Wilk W={W:.3f} p={pw:.2e}")

print("\n=== GROUP DIFFERENCES ACROSS COMBOS ===")
groups=[g.resistance_pct.dropna().values for _,g in df.groupby("pathogen_drug")]
F,pF = stats.f_oneway(*groups)
H,pH = stats.kruskal(*groups)
k=len(groups); N=sum(len(g) for g in groups)
# eta squared from F
eta2 = (F*(k-1))/(F*(k-1)+(N-k))
f_eff = np.sqrt(eta2/(1-eta2))
print(f"ANOVA F({k-1},{N-k})={F:.1f} p={pF:.3e}")
print(f"Kruskal-Wallis H({k-1})={H:.1f} p={pH:.3e}")
print(f"eta^2={eta2:.4f}  Cohen f={f_eff:.3f}  N={N} k={k}")

print("\n=== RQ1: lagged consumption vs resistance (Spearman, lag1) ===")
for combo,g in df.groupby("pathogen_drug"):
    g=g.sort_values(["country_code","year"]).copy()
    g["lag1"]=g.groupby("country_code")["ddd_total"].shift(1)
    s=g.dropna(subset=["lag1","resistance_pct"])
    if len(s)>10:
        rho,p = stats.spearmanr(s.lag1, s.resistance_pct)
        print(f"  {combo:34s} n={len(s):4d} rho={rho:+.3f} p={p:.2e}")

print("\n=== POWER / MIN SAMPLE SIZE (one-way ANOVA, k=5) ===")
try:
    from statsmodels.stats.power import FTestAnovaPower
    for f_target in (0.10, 0.25, 0.40):
        n_req = FTestAnovaPower().solve_power(effect_size=f_target, alpha=0.05, power=0.80, k_groups=5)
        print(f"  Cohen f={f_target}: total N required = {np.ceil(n_req):.0f}")
    ach = FTestAnovaPower().solve_power(effect_size=f_eff, alpha=0.05, nobs=N, k_groups=5)
    print(f"  Achieved power at observed f={f_eff:.3f}, N={N}: {ach:.4f}")
except ImportError:
    print("  statsmodels unavailable")
