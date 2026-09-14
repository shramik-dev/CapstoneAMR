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
import importlib.util as u
spec=u.spec_from_file_location("m", os.path.join(_HERE,"02_modeling.py")); m=u.module_from_spec(spec); spec.loader.exec_module(m)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

df=pd.read_csv(os.path.join(OUT,"amr_panel.csv"))

print("=== 1. DESCRIPTIVES with IQR + kurtosis ===")
for c in ["resistance_pct","ddd_total","pop_density","hdd","cdd"]:
    s=df[c].dropna()
    q1,q3=s.quantile([.25,.75])
    print(f"{c:16s} n={len(s):5d} mean={s.mean():8.2f} med={s.median():8.2f} sd={s.std():8.2f} "
          f"Q1={q1:8.2f} Q3={q3:8.2f} IQR={q3-q1:8.2f} skew={s.skew():6.2f} kurt={s.kurtosis():7.2f}")

print("\n=== 2. MIN SAMPLE SIZE PER RQ ===")
# RQ1: Fisher z for correlation, two-tailed alpha=.05, power=.80
za, zb = stats.norm.ppf(1-0.05/2), stats.norm.ppf(0.80)
for r in (0.10,0.20,0.30,0.50):
    z=0.5*np.log((1+r)/(1-r)); n=((za+zb)/z)**2+3
    print(f"  RQ1 Fisher z: detect r={r:.2f} -> n >= {np.ceil(n):.0f}")
# RQ2: Green's rule N >= 50 + 8m (overall model), 104 + m (individual predictors)
for mm in (43,):
    print(f"  RQ2 Green's rule (m={mm} predictors): overall N >= {50+8*mm}; per-predictor N >= {104+mm}")
# RQ4: paired comparison of forecast errors
print("  RQ4: paired test on n_test forecast errors (see below)")

print("\n=== 3. PAIRED MODEL COMPARISON (RQ4) ===")
combos=sorted(df.pathogen_drug.unique())
PANEL=m.build_panel(write=None)   # build once, reuse (was rebuilt 20x)
P={}
for name in ["Naive","Ridge","RandomForest","GradientBoosting"]:
    err=[]
    for c in combos:
        d,feat=m.make_lagged(PANEL,c,autoregressive=False)
        d=d.dropna(subset=feat+["resistance_pct","resist_lag1"])
        cut=d.year.max()-m.TEST_YEARS
        dev,test=d[d.year<=cut],d[d.year>cut]
        if len(test)<5 or dev.year.nunique()<3: continue
        if name=="Naive": pred=test.resist_lag1.values
        else:
            if name=="Ridge": est,_=m.tune(Ridge(),{"alpha":[1.0,10.0,50.0]},dev,feat)
            elif name=="RandomForest": est,_=m.tune(RandomForestRegressor(random_state=42,n_jobs=-1),m.RF_GRID,dev,feat)
            else: est,_=m.tune(GradientBoostingRegressor(random_state=42),m.GBR_GRID,dev,feat)
            pred=est.predict(test[feat])
        err.extend(np.abs(test.resistance_pct.values-pred))
    P[name]=np.array(err)
    print(f"  {name:18s} n={len(err)} mean|e|={np.mean(err):.3f}")

print("\n  Paired t-tests on absolute errors (Diebold-Mariano style, MAE loss):")
import itertools
for a,b in itertools.combinations(P,2):
    x,y=P[a],P[b]; n=min(len(x),len(y)); x,y=x[:n],y[:n]
    t,p=stats.ttest_rel(x,y); w,pw=stats.wilcoxon(x,y)
    print(f"    {a:17s} vs {b:17s} diff={np.mean(x-y):+.3f}  t={t:+.2f} p={p:.3e}  Wilcoxon p={pw:.3e}")
