"""Scenario forecast of 2027 yields per RM x crop (Saskatchewan demo).

Next summer's weather can't be predicted (pre-season models have no skill, see
results/README.md), so the forecast is a DISTRIBUTION over weather scenarios:

  1. Weather scenarios: every past season 1982-2025 is an equally likely
     weather year. For RM r, scenario s = r's own in-season features of year s.
     Pre-season features (fall-winter rain, May-1 soil wetness) are set to r's
     climatological median, because they carry no skill and 2027's values are
     not known yet. ENSO features are not used (they hurt held-out skill).
  2. Model: gradient boosting without ENSO (main) and a random forest
     (comparison). Scenario s is predicted by a model trained WITHOUT year s
     (leave-one-year-out). This stops the model from "remembering" year s's
     yields, which would make scenarios too spread out (in-sample predictions
     have SD 0.21 vs 0.17 out of sample).
  3. Honest spread: models shrink extremes, so we add a residual bootstrap.
     Outcome = scenario prediction + e, with e drawn from the out-of-sample
     (leave-one-year-out) residuals of the same crop, pooled over RMs and years.
     This keeps the skew and fat lower tail of real crop failures, which a
     symmetric conformal widening would not.
  4. Yield = 2027 trend (straight line per RM x crop, all years) x (1 + anomaly).
  5. Optional ENSO analogue weighting (Gaussian kernel on autumn ONI).
  6. Validation: leave-one-year-out of the WHOLE procedure (pair-wise refits:
     model trained without the held-out year t AND the scenario year s),
     compared with a climatology baseline (trend + the RM x crop's own past
     anomalies, no model).
  7. In-season update at July 1: pre-season + May-June features observed,
     the rest from scenarios.

Importable API (for the recommendation step / Streamlit):
    from forecast import forecast_distribution, summarize
    d = forecast_distribution(159, "Canola")            # one row per weather year
    summarize(d)                                         # expected, P10/P50/P90, P(fail)
    forecast_distribution(159, "Canola", weighting="enso")
    forecast_distribution(159, "Canola", early_obs={"rain_may_jun": 60, "last_spring_frost": 130})

Run:  .venv/bin/python scripts/forecast.py          (~3 min on 10 cores)
"""
import os, pathlib, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")   # HistGB threads oversubscribe inside joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_features import IN  # noqa: E402
from train_models import CROPS, FAIL, P, X, anomalies, load, make  # noqa: E402
import extra_data as XD  # noqa: E402

OUT = ROOT / "results" / "forecast"
PRE2 = ["rain_fall_winter", "wet_root_may"]          # pre-season, no ENSO
WX = IN + XD.SOIL + XD.SOILX                         # per RM x weather year (soil static, known pre-planting)
FEATS = PRE2 + WX
EARLY = ["rain_may_jun", "last_spring_frost"]         # fully observed by July 1
LATE = [c for c in IN if c not in EARLY]              # come from scenarios at July 1
TARGET_YEAR = 2027
ONI_NOW = 2.46        # ASO 2026 (latest available; SON 2026 not yet published)
ENSO_BW = 0.5         # kernel bandwidth in ONI units
QS = (0.10, 0.50, 0.90)
UNITS = {c: "bu/ac" for c in CROPS} | {"Lentils": "lb/ac", "Chickpeas": "lb/ac"}
MODELS = ("gbm", "rf")
NAMED = {"bad_2021": 2021, "wet_2010": 2010}


def make_model(name, n_jobs=1):
    if name == "gbm":
        return make("gbm_no_enso")
    return RandomForestRegressor(n_estimators=150, min_samples_leaf=10, max_features=0.5,
                                 n_jobs=n_jobs, random_state=0)


# ---------------------------------------------------------------- data helpers
def _data():
    df = load()
    f = XD.add_soil(pd.read_csv(P / "features.csv"))
    cols = X(df, FEATS).columns
    return df, f, cols


def _xm(frame, cols):
    return X(frame, FEATS).reindex(columns=cols, fill_value=0).values


def scen_frame(R, f, s, pre, early=None):
    """Feature rows for targets R (rm, crop) under weather year s.
    pre: DataFrame indexed by rm with PRE2 columns. early: same for EARLY cols (or None)."""
    w = f[f.year == s].set_index("rm")[WX]
    out = R[["rm", "crop"]].reset_index(drop=True).copy()
    out[WX] = w.loc[out.rm].values
    out[PRE2] = pre.loc[out.rm, PRE2].values
    if early is not None:
        out[EARLY] = early.loc[out.rm, EARLY].values
    return XD.soil_interactions(out)


def oni_prev(f):
    return f.groupby("year").oni_son_prev.first()


def enso_weights(years, oni_target, oni_by_year, bw=ENSO_BW):
    d = oni_by_year.loc[years].values - oni_target
    w = np.exp(-0.5 * (d / bw) ** 2)
    return w / w.sum()


# --------------------------------------------------------- mixture statistics
def mixture(scen, w, resid):
    """Samples = scen[s] + resid[j], weight w[s]/len(resid). Clipped to [-1, 1]."""
    x = (scen[:, None] + resid[None, :]).ravel()
    ww = np.repeat(w / w.sum(), len(resid)) / len(resid)
    return np.clip(x, -1, 1), ww


def wquant(x, w, qs=QS):
    o = np.argsort(x)
    x, c = x[o], np.cumsum(w[o])
    return np.array([x[min(np.searchsorted(c, q), len(x) - 1)] for q in qs])


def crps(x, w, y):
    o = np.argsort(x)
    x, w = x[o], w[o] / w.sum()
    c = np.cumsum(w)
    e_xy = np.sum(w * np.abs(x - y))
    e_xx = 2 * np.sum(w * x * (2 * c - w - 1))
    return e_xy - 0.5 * e_xx


def pinball(qv, y, qs=QS):
    return np.mean([max(q * (y - v), (q - 1) * (y - v)) for q, v in zip(qs, qv)])


def score_dist(x, w, y):
    q = wquant(x, w)
    pf = w[x < FAIL].sum()
    return dict(p10=q[0], p50=q[1], p90=q[2], mean=np.sum(w * x), p_fail=pf,
                crps=crps(x, w, y), pinball=pinball(q, y),
                brier=(pf - (y < FAIL)) ** 2,
                inside=float(q[0] <= y <= q[2]), below=float(y < q[0]), above=float(y > q[2]))


# -------------------------------------------------- production (2027) fitting
def _prod_job(s, name, df, f, a, cols, pre_med, R):
    ok = ~np.isnan(a)
    tr = ok & (df.year != s).values
    m = make_model(name).fit(_xm(df[tr], cols), a[tr])
    scen = m.predict(_xm(scen_frame(R, f, s, pre_med), cols))
    rs = ok & (df.year == s).values
    res = df.loc[rs, ["rm", "crop", "year"]].copy()
    res["resid"] = a[rs] - m.predict(_xm(df[rs], cols))
    return s, m, scen, res


def build_2027(df, f, cols):
    a = anomalies(df, -1)                       # trend on all years
    years = sorted(df.year.unique())
    R = df[["rm", "crop"]].drop_duplicates().sort_values(["rm", "crop"]).reset_index(drop=True)
    pre_med = f.groupby("rm")[PRE2].median()
    # trend projected to 2027
    tr = []
    for (rm, crop), g in df.groupby(["rm", "crop"]):
        b, c0 = np.polyfit(g.year, g["yield"], 1)
        tr.append((rm, crop, max(c0 + b * TARGET_YEAR, 1e-6), b))
    trend = pd.DataFrame(tr, columns=["rm", "crop", "trend_2027", "trend_slope"])
    long, resid, models = [], [], {}
    for name in MODELS:
        jobs = Parallel(n_jobs=-1)(delayed(_prod_job)(s, name, df, f, a, cols, pre_med, R) for s in years)
        models[name] = {s: m for s, m, _, _ in jobs}
        for s, _, scen, res in jobs:
            long.append(R.assign(model=name, scenario_year=s, anomaly=scen))
            resid.append(res.assign(model=name))
        # in-sample alternative (one model on all years) -- for comparison only
        if name == "gbm":
            ok = ~np.isnan(a)
            mall = make_model(name).fit(_xm(df[ok], cols), a[ok])
            ins = [R.assign(scenario_year=s, anomaly_insample=mall.predict(_xm(scen_frame(R, f, s, pre_med), cols)))
                   for s in years]
    long = pd.concat(long).pivot_table(index=["rm", "crop", "scenario_year"], columns="model",
                                       values="anomaly").reset_index()
    long = long.rename(columns={m: f"anomaly_{m}" for m in MODELS})
    long = long.merge(pd.concat(ins), on=["rm", "crop", "scenario_year"])
    resid = pd.concat(resid).pivot_table(index=["rm", "crop", "year"], columns="model",
                                         values="resid").reset_index()
    resid = resid.rename(columns={m: f"resid_{m}" for m in MODELS})
    return long, resid, trend, models


# ------------------------------------------------------------ public API
_CACHE = {}


def _load_cached():
    if not _CACHE:
        _CACHE["scen"] = pd.read_csv(OUT / "scenarios_2027.csv")
        _CACHE["resid"] = pd.read_csv(OUT / "residuals.csv")
        _CACHE["f"] = XD.add_soil(pd.read_csv(P / "features.csv"))
    return _CACHE


def _fit_loyo(s, df, a, cols, name="gbm"):
    ok = ~np.isnan(a)
    tr = ok & (df.year != s).values
    return s, make_model(name).fit(_xm(df[tr], cols), a[tr])


def _load_models():
    """The 44 leave-one-year-out GB models, refitted on first use (~10 s, then cached in
    memory). Refitting is cheaper to ship than a 30 MB pickle."""
    if "models" not in _CACHE:
        df, _, cols = _data()
        a = anomalies(df, -1)
        fits = Parallel(n_jobs=-1)(delayed(_fit_loyo)(s, df, a, cols) for s in sorted(df.year.unique()))
        _CACHE["models"], _CACHE["cols"] = dict(fits), cols
    return _CACHE["models"], _CACHE["cols"]


def forecast_distribution(rm, crop, model="gbm", weighting="equal", oni=ONI_NOW, early_obs=None,
                          pre_obs=None):
    """Scenario outcomes for one RM x crop in 2027: one row per weather year.

    weighting : "equal" or "enso" (kernel weight on autumn ONI vs `oni`).
    early_obs : dict of observed May-June features (rain_may_jun, last_spring_frost) for an
                in-season (July 1) update. Re-predicts with the saved GB models (model="gbm" only).
    pre_obs   : dict of observed pre-season features (rain_fall_winter, wet_root_may);
                default = the RM's climatological median.
    Columns: scenario_year, weight, anomaly (model mean under that weather), yield, the
    residual-adjusted P10/P50/P90 and P(fail) under that weather, unit, trend_2027.
    Use summarize(df) for the full-mixture summary.
    """
    c = _load_cached()
    d = c["scen"][(c["scen"].rm == rm) & (c["scen"].crop == crop)].sort_values("scenario_year").copy()
    if d.empty:
        raise ValueError(f"no data for RM {rm} {crop}")
    if early_obs is not None or pre_obs is not None:
        if model != "gbm":
            raise ValueError("in-season update uses the saved GB models only")
        models, cols = _load_models()
        f = c["f"]
        R = pd.DataFrame({"rm": [rm], "crop": [crop]})
        pre = f[f.rm == rm][PRE2].median().to_frame().T.set_index(pd.Index([rm]))
        for k, v in (pre_obs or {}).items():
            pre[k] = v
        early = None
        if early_obs is not None:
            early = pd.DataFrame([early_obs], index=[rm])
            for k in EARLY:
                if k not in early:
                    raise ValueError(f"early_obs needs {EARLY}")
        d["anomaly_gbm"] = [models[s].predict(_xm(scen_frame(R, f, s, pre, early), cols))[0]
                            for s in d.scenario_year]
    years = d.scenario_year.values
    if weighting == "enso":
        w = enso_weights(years, oni, oni_prev(c["f"]))
    else:
        w = np.full(len(years), 1 / len(years))
    res = c["resid"][c["resid"].crop == crop][f"resid_{model}"].dropna().values
    t = d.trend_2027.iloc[0]
    rows = []
    for s, sc, wi in zip(years, d[f"anomaly_{model}"].values, w):
        x = np.clip(sc + res, -1, 1)
        q = np.quantile(x, QS)
        rows.append(dict(rm=rm, crop=crop, scenario_year=s, weight=wi, anomaly=sc,
                         yield_=t * (1 + sc), yield_p10=t * (1 + q[0]), yield_p50=t * (1 + q[1]),
                         yield_p90=t * (1 + q[2]), p_fail=(x < FAIL).mean()))
    out = pd.DataFrame(rows).rename(columns={"yield_": "yield"})
    out["unit"] = UNITS.get(crop, "bu/ac")
    out["trend_2027"] = t
    out["model"] = model
    out.attrs["resid"] = res
    return out


def summarize(dist):
    """Full mixture summary (scenario x residual) of a forecast_distribution() frame."""
    res = dist.attrs.get("resid")
    if res is None:
        c = _load_cached()
        res = c["resid"][c["resid"].crop == dist.crop.iloc[0]][f"resid_{dist.model.iloc[0]}"].dropna().values
    x, w = mixture(dist.anomaly.values, dist.weight.values, res)
    t = dist.trend_2027.iloc[0]
    q = wquant(x, w)
    return dict(rm=dist.rm.iloc[0], crop=dist.crop.iloc[0], unit=dist.unit.iloc[0], trend_2027=t,
                expected=t * (1 + np.sum(w * x)), p10=t * (1 + q[0]), p50=t * (1 + q[1]),
                p90=t * (1 + q[2]), p_fail=w[x < FAIL].sum(),
                anom_expected=np.sum(w * x), anom_p10=q[0], anom_p50=q[1], anom_p90=q[2])


# ------------------------------------------------------------ validation
def _val_job(t, name, df, f, cols, oni_by):
    """Leave year t out of everything. For each scenario year s != t a model is trained
    without {s, t}: it gives the scenario prediction for t's rows under weather s
    (pre-season = RM median of years != t), the July-1 variant (t's own pre-season and
    May-June), and the out-of-sample residuals at year s (residual pool without t)."""
    a = anomalies(df, t)
    ok = ~np.isnan(a)
    te = ok & (df.year == t).values
    R = df[te].reset_index(drop=True)
    pre_med = f[f.year != t].groupby("rm")[PRE2].median()
    own = f[f.year == t].set_index("rm")
    years = [s for s in sorted(df.year.unique()) if s != t]
    S_pre, S_early, pool = [], [], []
    for s in years:
        tr = ok & ~df.year.isin([s, t]).values
        m = make_model(name).fit(_xm(df[tr], cols), a[tr])
        S_pre.append(m.predict(_xm(scen_frame(R, f, s, pre_med), cols)))
        S_early.append(m.predict(_xm(scen_frame(R, f, s, own[PRE2], own[EARLY]), cols)))
        rs = ok & (df.year == s).values
        pool.append(pd.DataFrame({"crop": df.crop[rs].values, "resid": a[rs] - m.predict(_xm(df[rs], cols))}))
    S_pre, S_early, pool = np.array(S_pre), np.array(S_early), pd.concat(pool)
    S_ins = None
    if name == "gbm":   # in-sample alternative: one model without t, scenarios in-sample
        mt = make_model(name).fit(_xm(df[ok & (df.year != t).values], cols), a[ok & (df.year != t).values])
        S_ins = np.array([mt.predict(_xm(scen_frame(R, f, s, pre_med), cols)) for s in years])
    # climatology: RM x crop's own anomalies in years != t
    clim = pd.Series(a, index=df.index)[ok & (df.year != t).values]
    climg = {k: clim.loc[g.index].values for k, g in df[ok & (df.year != t).values].groupby(["rm", "crop"])}
    w_eq = np.full(len(years), 1 / len(years))
    w_en = enso_weights(years, oni_by.loc[t], oni_by)
    pools = {c: g.resid.values for c, g in pool.groupby("crop")}
    rows = []
    for i, r in R.iterrows():
        y = a[te][i]
        res = pools[r.crop]
        base = dict(year=t, rm=r.rm, crop=r.crop, obs=y)
        variants = {
            f"{name}_scenarios": mixture(S_pre[:, i], w_eq, res),
            f"{name}_scenarios_enso": mixture(S_pre[:, i], w_en, res),
            f"{name}_july1_update": mixture(S_early[:, i], w_eq, res),
            f"{name}_scenarios_no_resid": (np.clip(S_pre[:, i], -1, 1), w_eq),
        }
        if S_ins is not None:
            variants["gbm_insample_scenarios"] = mixture(S_ins[:, i], w_eq, res)
        if name == "gbm":   # baselines once
            ca = climg[(r.rm, r.crop)]
            cy = df.loc[ok & (df.year != t).values & (df.rm == r.rm).values & (df.crop == r.crop).values, "year"].values
            variants["climatology"] = (ca, np.full(len(ca), 1 / len(ca)))
            variants["climatology_enso"] = (ca, enso_weights(cy, oni_by.loc[t], oni_by))
        for v, (x, w) in variants.items():
            rows.append(base | dict(method=v) | score_dist(x, w, y))
    return pd.DataFrame(rows)


def validate(df, f, cols):
    oni_by = oni_prev(f)
    years = sorted(df.year.unique())
    out = []
    for name in MODELS:
        t0 = time.time()
        out += Parallel(n_jobs=-1)(delayed(_val_job)(t, name, df, f, cols, oni_by) for t in years)
        print(f"  validation {name}: {time.time() - t0:.0f}s")
    return pd.concat(out, ignore_index=True)


def summarize_validation(v):
    g = v.groupby("method")
    s = g.agg(n=("obs", "size"), coverage_p10_p90=("inside", "mean"), share_below_p10=("below", "mean"),
              share_above_p90=("above", "mean"), crps=("crps", "mean"), pinball=("pinball", "mean"),
              brier_fail=("brier", "mean"), mean_p_fail=("p_fail", "mean"),
              width_p10_p90=("p90", "mean")).reset_index()
    s["width_p10_p90"] = g.apply(lambda d: (d.p90 - d.p10).mean()).values
    s["obs_fail_rate"] = (v.groupby("method").obs.apply(lambda o: (o < FAIL).mean())).values
    ref = s.set_index("method").loc["climatology"]
    for k in ("crps", "pinball", "brier_fail"):
        s[f"{k}_skill_vs_clim"] = 1 - s[k] / ref[k]
    # year-level bootstrap CI of CRPS skill (resampling held-out years)
    rng = np.random.default_rng(0)
    yrs = v.year.unique()
    cy = v.groupby(["method", "year"]).crps.sum().unstack()
    nb = v.groupby(["method", "year"]).size().unstack()
    lo, hi = [], []
    for m in s.method:
        sk = []
        for _ in range(2000):
            b = rng.choice(yrs, len(yrs))
            sk.append(1 - cy.loc[m, b].sum() / cy.loc["climatology", b].sum())
        lo.append(np.percentile(sk, 5)); hi.append(np.percentile(sk, 95))
    s["crps_skill_ci90_lo"], s["crps_skill_ci90_hi"] = lo, hi
    order = ["climatology", "climatology_enso", "gbm_scenarios", "gbm_scenarios_enso", "gbm_insample_scenarios",
             "gbm_scenarios_no_resid", "rf_scenarios", "rf_scenarios_enso", "rf_scenarios_no_resid",
             "gbm_july1_update", "rf_july1_update"]
    return s.set_index("method").loc[[o for o in order if o in s.method.values]].reset_index()


# ------------------------------------------------------------ outputs
def median_outcome_year(scen, rm):
    """Weather year whose model anomaly, averaged over the RM's crops, is the median one."""
    m = scen[scen.rm == rm].groupby("scenario_year").anomaly_gbm.mean()
    target = np.sort(m.values)[(len(m) - 1) // 2]      # lower middle value: no ties with 44 years
    return int(m.index[np.argmin(np.abs(m.values - target))])


def median_weather_year(f, rm):
    """Weather year whose in-season features are closest to the RM's median (standardised)."""
    g = f[f.rm == rm].set_index("year")[IN]
    z = (g - g.median()) / g.std()
    return int((z ** 2).sum(axis=1).idxmin())


def write_forecast(long, resid, trend, f, df):
    scen = long.merge(trend, on=["rm", "crop"])
    for m in MODELS:
        scen[f"yield_{m}"] = scen.trend_2027 * (1 + scen[f"anomaly_{m}"])
    oni_by = oni_prev(f)
    scen["weight_equal"] = 1 / scen.groupby(["rm", "crop"]).scenario_year.transform("count")
    scen["weight_enso"] = scen.groupby(["rm", "crop"]).scenario_year.transform(
        lambda ys: enso_weights(ys.values, ONI_NOW, oni_by))
    scen["unit"] = scen.crop.map(UNITS)
    scen = scen[["rm", "crop", "scenario_year", "weight_equal", "weight_enso", "anomaly_gbm", "anomaly_rf",
                 "anomaly_insample", "yield_gbm", "yield_rf", "unit", "trend_2027", "trend_slope"]]
    scen.round(5).to_csv(OUT / "scenarios_2027.csv", index=False)
    resid.round(5).to_csv(OUT / "residuals.csv", index=False)
    _CACHE.clear()
    names = pd.read_csv(ROOT / "data" / "rm_points.csv").set_index("rm").rm_name
    act = dict(zip(zip(df.rm, df.crop, df.year), anomalies(df, -1)))   # actual anomalies, all-year trend
    rows = []
    for (rm, crop), _ in scen.groupby(["rm", "crop"]):
        d = forecast_distribution(rm, crop)
        s = summarize(d)
        se = summarize(forecast_distribution(rm, crop, weighting="enso"))
        sr = summarize(forecast_distribution(rm, crop, model="rf"))
        ny = median_outcome_year(scen, rm)
        wy = int(d.loc[d.anomaly.idxmin(), "scenario_year"])
        pick = lambda yr: d.loc[d.scenario_year == yr, "yield"].iloc[0]  # noqa: E731
        replay = lambda yr: s["trend_2027"] * (1 + act.get((rm, crop, yr), np.nan))  # noqa: E731
        rows.append(dict(rm=rm, rm_name=names[rm], crop=crop, unit=s["unit"], trend_2027=s["trend_2027"],
                         expected=s["expected"], p10=s["p10"], p50=s["p50"], p90=s["p90"], p_fail=s["p_fail"],
                         normal_year=ny, yield_normal_year=pick(ny), yield_bad_2021=pick(2021),
                         yield_wet_2010=pick(2010), worst_weather_year=wy, yield_worst_weather_year=pick(wy),
                         replay_2021=replay(2021), replay_2010=replay(2010),
                         median_weather_year=median_weather_year(f, rm),
                         expected_enso=se["expected"], p10_enso=se["p10"], p50_enso=se["p50"],
                         p90_enso=se["p90"], p_fail_enso=se["p_fail"],
                         expected_rf=sr["expected"], p10_rf=sr["p10"], p50_rf=sr["p50"], p90_rf=sr["p90"],
                         p_fail_rf=sr["p_fail"]))
    out = pd.DataFrame(rows)
    out.round(3).to_csv(OUT / "forecast_2027.csv", index=False)
    return out


def charts(fc, v, vs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2})

    # 1. RM 159 fan: yield distribution per crop, % of 2027 trend
    rm = 159
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sub = fc[fc.rm == rm].reset_index(drop=True)
    for i, r in sub.iterrows():
        d = forecast_distribution(rm, r.crop)
        x, w = mixture(d.anomaly.values, d.weight.values, d.attrs["resid"])
        q = wquant(x, w, (0.1, 0.25, 0.5, 0.75, 0.9)) * 100 + 100
        ax.plot([q[0], q[4]], [i, i], color=BLUE, lw=2, alpha=0.45, solid_capstyle="round")
        ax.plot([q[1], q[3]], [i, i], color=BLUE, lw=7, solid_capstyle="round")
        ax.plot(q[2], i, "o", ms=9, color="white", mec=BLUE, mew=2)
        for col, mk, c in (("yield_bad_2021", "v", ORANGE), ("yield_wet_2010", "^", AQUA)):
            ax.plot(r[col] / r.trend_2027 * 100, i, mk, ms=8, color=c, mec="white", mew=1)
        ax.text(152, i, f"P50 {r.p50:,.0f} {r.unit}   P(fail) {r.p_fail:.0%}", va="center", fontsize=9, color=INK2)
    ax.axvline(100, color=GRAY, lw=1, ls=":")
    ax.axvline(80, color=ORANGE, lw=1, ls="--", alpha=0.6)
    ax.text(79, len(sub) - 0.75, "failure line (-20%)", color=ORANGE, fontsize=8, ha="right")
    ax.set_yticks(range(len(sub)), sub.crop)
    ax.invert_yaxis()
    ax.set_xlim(40, 150)
    ax.set_xlabel("2027 yield, % of trend")
    ax.set_title(f"RM {rm} Sherwood: 2027 yield scenarios (thin P10-P90, thick P25-P75, dot P50)",
                 loc="left", fontsize=11, color=INK)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=7, label="all 44 weather years + model error"),
                       Line2D([], [], marker="v", ls="", color=ORANGE, ms=8, label="2021-like weather (model mean)"),
                       Line2D([], [], marker="^", ls="", color=AQUA, ms=8, label="2010-like weather (model mean)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=False)
    fig.subplots_adjust(right=0.75, left=0.13, bottom=0.2)
    fig.savefig(OUT / "chart_rm159_2027.png", dpi=150)
    plt.close(fig)

    # 2. Calibration: coverage + below P10 per method; CRPS skill
    show = [m for m in ["climatology", "gbm_scenarios", "rf_scenarios", "gbm_scenarios_no_resid",
                        "gbm_insample_scenarios", "gbm_scenarios_enso", "gbm_july1_update", "rf_july1_update"]
            if m in vs.method.values]
    s = vs.set_index("method").loc[show]
    lab = {"climatology": "Climatology (no model)", "gbm_scenarios": "GB scenarios + residuals",
           "rf_scenarios": "RF scenarios + residuals", "gbm_scenarios_no_resid": "GB scenarios, no residuals",
           "gbm_insample_scenarios": "GB in-sample scenarios", "gbm_scenarios_enso": "GB + El Nino weights",
           "gbm_july1_update": "GB, July 1 update", "rf_july1_update": "RF, July 1 update"}
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    yy = np.arange(len(s))
    for ax, col, tgt, title in ((axs[0], "coverage_p10_p90", 0.8, "Actual inside P10-P90 (target 80%)"),
                                (axs[1], "share_below_p10", 0.1, "Actual below P10 (target 10%)")):
        ax.barh(yy, s[col] * 100, color=[GRAY if m == "climatology" else BLUE for m in s.index], height=0.6)
        ax.axvline(tgt * 100, color=INK, lw=1, ls="--")
        for i, val in enumerate(s[col]):
            ax.text(val * 100 + 1, i, f"{val:.0%}", va="center", fontsize=8, color=INK2)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    sk = s.crps_skill_vs_clim * 100
    axs[2].barh(yy, sk, color=[GRAY if m == "climatology" else (BLUE if v >= 0 else ORANGE)
                               for m, v in zip(s.index, sk)], height=0.6)
    axs[2].errorbar(sk, yy, xerr=[sk - s.crps_skill_ci90_lo * 100, s.crps_skill_ci90_hi * 100 - sk],
                    fmt="none", ecolor=INK2, lw=1, capsize=2)
    axs[2].axvline(0, color=INK, lw=1)
    axs[2].set_title("CRPS skill vs climatology, % (90% CI)", loc="left", fontsize=10, color=INK)
    axs[0].set_yticks(yy, [lab[m] for m in s.index])
    axs[0].invert_yaxis()
    fig.suptitle("Leave-one-year-out check of the scenario forecast (all RMs x crops, 1982-2025)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "chart_validation.png", dpi=150)
    plt.close(fig)

    # 3. July 1 narrowing for RM 159 canola, example years
    d = v[(v.rm == 159) & (v.crop == "Canola")]
    yrs = [1988, 2002, 2021, 2010, 2016, 2013]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, yr in enumerate(yrs):
        for off, meth, c, lb in ((-0.15, "gbm_scenarios", GRAY, "before planting"),
                                 (0.15, "gbm_july1_update", BLUE, "July 1 update")):
            r = d[(d.year == yr) & (d.method == meth)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.plot([r.p10 * 100, r.p90 * 100], [i + off] * 2, color=c, lw=5, solid_capstyle="round",
                    label=lb if i == 0 else None)
            ax.plot(r.p50 * 100, i + off, "o", ms=7, color="white", mec=c, mew=2)
        ob = d[d.year == yr].obs.iloc[0] * 100
        ax.plot(ob, i, "D", color=ORANGE, ms=7, mec="white", label="actual" if i == 0 else None)
    ax.axvline(0, color=GRAY, lw=1, ls=":")
    ax.set_yticks(range(len(yrs)), yrs)
    ax.invert_yaxis()
    ax.set_xlabel("yield anomaly vs trend, % (P10-P90 bar, P50 dot)")
    ax.set_title("RM 159 canola: range before planting vs at July 1 (held-out years)",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "chart_july1_update.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, f, cols = _data()
    print(f"{len(df)} RM-crop-years; {df[['rm', 'crop']].drop_duplicates().shape[0]} RM x crop combos")
    t0 = time.time()
    long, resid, trend, models = build_2027(df, f, cols)
    fc = write_forecast(long, resid, trend, f, df)
    print(f"2027 forecast: {time.time() - t0:.0f}s")
    v = validate(df, f, cols)
    v.round(5).to_csv(OUT / "validation_detail.csv", index=False)
    vs = summarize_validation(v)
    vs.round(4).to_csv(OUT / "validation.csv", index=False)
    # per-crop calibration of the main method vs climatology
    pc = v[v.method.isin(["climatology", "gbm_scenarios", "gbm_july1_update"])].groupby(["crop", "method"]).agg(
        coverage=("inside", "mean"), below_p10=("below", "mean"), crps=("crps", "mean")).unstack()
    pc.round(4).to_csv(OUT / "validation_by_crop.csv")
    print(vs.round(3).to_string(index=False))
    print(pc.round(3).to_string())
    charts(fc, v, vs)
    print(fc[fc.rm == 159].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
