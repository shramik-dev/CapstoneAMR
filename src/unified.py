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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import importlib.util as u
spec=u.spec_from_file_location("m", os.path.join(_HERE,"02_modeling.py")); m=u.module_from_spec(spec); spec.loader.exec_module(m)

df=m.build_panel(write=None)
combos=sorted(df.pathogen_drug.unique())
def rmse(y,p): return float(np.sqrt(mean_squared_error(y,p)))

def diracc(prev, actual, pred):
    """sign(pred_t - actual_{t-1}) == sign(actual_t - actual_{t-1}); ties count as miss."""
    a=np.sign(np.asarray(actual)-np.asarray(prev)); p=np.sign(np.asarray(pred)-np.asarray(prev))
    return float(np.mean((a==p) & (a!=0)))

rows=[]
for name in ["Naive","Ridge","RandomForest","GradientBoosting"]:
    TR=dict(mae=[],rmse=[],r2=[],da=[],n=[]); TE=dict(mae=[],rmse=[],r2=[],da=[],n=[])
    for c in combos:
        d,feat=m.make_lagged(df,c,autoregressive=False)
        d=d.dropna(subset=feat+["resistance_pct","resist_lag1"])
        cut=d.year.max()-m.TEST_YEARS
        dev,test=d[d.year<=cut],d[d.year>cut]
        if len(test)<5 or dev.year.nunique()<3: continue
        if name=="Naive":
            trp=dev.resist_lag1.values; tep=test.resist_lag1.values
        else:
            if name=="Ridge": est,_=m.tune(Ridge(),{"alpha":[1.0,10.0,50.0]},dev,feat)
            elif name=="RandomForest": est,_=m.tune(RandomForestRegressor(random_state=42,n_jobs=-1),m.RF_GRID,dev,feat)
            else: est,_=m.tune(GradientBoostingRegressor(random_state=42),m.GBR_GRID,dev,feat)
            trp=est.predict(dev[feat]); tep=est.predict(test[feat])
        for S,P,D in ((TR,trp,dev),(TE,tep,test)):
            S["mae"].append(mean_absolute_error(D.resistance_pct,P)); S["rmse"].append(rmse(D.resistance_pct,P))
            S["r2"].append(r2_score(D.resistance_pct,P)); S["da"].append(diracc(D.resist_lag1,D.resistance_pct,P))
            S["n"].append(len(D))
    rows.append(dict(model=name,
        train_MAE=np.mean(TR["mae"]), test_MAE=np.mean(TE["mae"]),
        train_RMSE=np.mean(TR["rmse"]), test_RMSE=np.mean(TE["rmse"]),
        train_R2=np.mean(TR["r2"]), test_R2=np.mean(TE["r2"]),
        train_DA=np.mean(TR["da"]), test_DA=np.mean(TE["da"]), n_test=int(np.sum(TE["n"]))))
res=pd.DataFrame(rows).set_index("model")
res.to_csv(os.path.join(OUT,"unified_results.csv"))
print(res.round(3).to_string())
