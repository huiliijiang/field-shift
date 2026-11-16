"""Train and compare yield-anomaly models with leave-one-year-out testing.

Target: relative yield anomaly = (yield - trend) / trend, where the trend is a
straight line per RM x crop fitted ONLY on training years (no leakage).

Feature sets
  in_season  : pre-season + in-season weather  -> "what drives a bad year?"
  pre_season : only what is known by ~May 1    -> "can we see it coming?"

Models
  trend_only        predicts 0 anomaly (yield = trend). The floor to beat.
  ridge             linear, standardised features + crop one-hot
  random_forest     sklearn RandomForestRegressor
  gbm               gradient boosting (HistGradientBoosting, LightGBM-style)
  gbm_no_enso       MAIN MODEL: gbm without El Niño, plus soil (SoilGrids) and, in season, satellite
                    greenness (MODIS NDVI deviation from training-year mean). See scripts/extra_data.py
  gbm_q10 / q50     quantile gradient boosting: bad-year (10th pct) / median
  fail_logit / fail_gbm   P(anomaly < -20%) classifiers, class-weighted

Outputs -> results/
"""
import json, pathlib, warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
P, OUT = ROOT / "data" / "processed", ROOT / "results"
OUT.mkdir(exist_ok=True)

import sys; sys.path.insert(0, str(ROOT / "scripts"))
from build_features import PRE, IN  # noqa: E402
import extra_data as XD  # noqa: E402

CROPS = ["Canola", "Spring Wheat", "Durum", "Barley", "Oats", "Peas", "Lentils", "Flax"]
ENSO = ["oni_son_prev", "oni_djf"]
FAIL = -0.20          # "failure" = 20%+ below trend
FLAG = -0.10          # a prediction this low counts as "flagged a bad year"
SEED = 0


def load():
    y = pd.read_csv(P / "sk_yields.csv")
    y = y[y.crop.isin(CROPS) & (y.year >= 1982)]
    n = y.groupby(["rm", "crop"]).year.transform("count")
    y = y[n >= 20]
    f = pd.read_csv(P / "features.csv")
    return XD.add_ndvi(XD.add_soil(y.merge(f, on=["rm", "year"]))).reset_index(drop=True)


def anomalies(df, test_year):
    """Relative anomaly vs a linear trend fitted on years != test_year, per RM x crop."""
    out = np.full(len(df), np.nan)
    for _, g in df.groupby(["rm", "crop"]):
        tr = g[g.year != test_year]
        if len(tr) < 10:
            continue
        b, a = np.polyfit(tr.year, tr["yield"], 1)
        trend = np.maximum(a + b * g.year, 1e-6)
        out[g.index] = ((g["yield"] - trend) / trend).clip(-1, 1)
    return out


def X(df, cols):
    return pd.get_dummies(df[cols + ["crop"]], columns=["crop"], dtype=float)


def make(name):
    num = lambda cols: cols  # noqa: E731
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=400, min_samples_leaf=10, max_features=0.5,
                                     n_jobs=-1, random_state=SEED)
    kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
              l2_regularization=1.0, random_state=SEED)
    if name in ("gbm", "gbm_no_enso"):
        return HistGradientBoostingRegressor(**kw)
    if name == "gbm_q10":
        return HistGradientBoostingRegressor(loss="quantile", quantile=0.10, **kw)
    if name == "gbm_q50":
        return HistGradientBoostingRegressor(loss="quantile", quantile=0.50, **kw)
    if name == "fail_logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))
    if name == "fail_gbm":
        return HistGradientBoostingClassifier(class_weight="balanced", **kw)
    raise ValueError(name)


REG = ["ridge", "random_forest", "gbm", "gbm_no_enso", "gbm_q10", "gbm_q50"]
CLF = ["fail_logit", "fail_gbm"]


def loyo(df, feats):
    df = df.copy()
    """Leave-one-year-out predictions for every model."""
    preds = []
    for t in sorted(df.year.unique()):
        a = anomalies(df, t)
        ok = ~np.isnan(a)
        tr, te = ok & (df.year != t).values, ok & (df.year == t).values
        if te.sum() == 0:
            continue
        row = df.loc[te, ["year", "rm", "crop", "yield"]].copy()
        row["obs"] = a[te]
        row["trend_only"] = 0.0
        XD.ndvi_dev(df, [y for y in XD.NDVI_YEARS if y != t])
        for m in REG + CLF:
            cols = feats
            if m == "gbm_no_enso":
                cols = [c for c in feats if c not in ENSO] + XD.SOIL
                if any(c in IN for c in feats):
                    cols = cols + XD.SOILX + XD.NDEV
            Xtr, Xte = X(df[tr], cols), X(df[te], cols).reindex(columns=X(df[tr], cols).columns, fill_value=0)
            if m in CLF:
                ytr = (a[tr] < FAIL).astype(int)
                row[m] = make(m).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
            else:
                row[m] = make(m).fit(Xtr, a[tr]).predict(Xte)
        preds.append(row)
    return pd.concat(preds, ignore_index=True)


def score(pr):
    rows = []
    bad = pr.obs < FAIL
    for m in ["trend_only"] + REG:
        e = pr[m] - pr.obs
        r = dict(model=m,
                 rmse=np.sqrt(mean_squared_error(pr.obs, pr[m])),
                 mae=mean_absolute_error(pr.obs, pr[m]),
                 r2=r2_score(pr.obs, pr[m]),
                 mae_bad_years=e[bad].abs().mean(),
                 bad_years_flagged=(pr.loc[bad, m] < FLAG).mean(),
                 false_alarms=(pr.loc[~bad, m] < FLAG).mean())
        if m == "gbm_q10":
            r["share_below_q10"] = (pr.obs < pr[m]).mean()  # ideal ~0.10
        rows.append(r)
    for m in CLF:
        p = pr[m]
        rows.append(dict(model=m, auc=roc_auc_score(bad, p),
                         bad_years_flagged=(p[bad] >= 0.5).mean(),
                         false_alarms=(p[~bad] >= 0.5).mean()))
    return pd.DataFrame(rows)


def main():
    df = load()
    print(f"{len(df)} RM-crop-years, {df.year.nunique()} years, {df.rm.nunique()} RMs, crops: {sorted(df.crop.unique())}")
    allm, allp = [], []
    for name, feats in [("in_season", PRE + IN), ("pre_season", PRE)]:
        pr = loyo(df, feats)
        pr.insert(0, "feature_set", name)
        s = score(pr)
        s.insert(0, "feature_set", name)
        allm.append(s); allp.append(pr)
        print(f"\n== {name} ==\n", s.round(3).to_string(index=False))
    pd.concat(allm).to_csv(OUT / "metrics.csv", index=False)
    pd.concat(allp).to_csv(OUT / "predictions.csv", index=False)
    json.dump(dict(fail_threshold=FAIL, flag_threshold=FLAG, crops=CROPS, rows=len(df)),
              open(OUT / "run_info.json", "w"), indent=2)


if __name__ == "__main__":
    main()
