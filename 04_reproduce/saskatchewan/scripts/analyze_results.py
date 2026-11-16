"""Explain and summarise the trained models.

  - year-level skill: average anomaly across all RMs x crops per year (obs vs predicted)
  - key drought years (1988, 2001, 2002, 2021)
  - per-crop skill
  - SHAP importance + direction for the in-season gradient-boosting model
  - surrogate depth-3 decision tree (plain rules) and its fidelity
  - El Niño check: anomaly in El Niño / La Niña / neutral winters
  - charts -> results/*.png
"""
import pathlib, sys, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, export_text

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
from build_features import PRE, IN  # noqa: E402
from train_models import load, anomalies, X, make, FAIL  # noqa: E402

LABEL = dict(rain_fall_winter="Fall–winter rain (mm)", wet_root_may="Root soil wetness, May 1",
             oni_son_prev="El Niño index, prev. autumn", oni_djf="El Niño index, winter",
             rain_may_jun="May–Jun rain (mm)", rain_jul_aug="Jul–Aug rain (mm)",
             dry_spell_max="Longest dry spell (days)", hot_days_flower="Days > 30 °C at flowering",
             tmax_jul="July mean max temp (°C)", gdd_may_aug="Growing degree days",
             last_spring_frost="Last spring frost (day of year)", first_fall_frost="First fall frost (day of year)",
             wet_root_min="Lowest root soil wetness, summer")
INK, MUTED, GRID = "#1F1E1D", "#5F5E5A", "#D3D1C7"
C = {"obs": "#1F1E1D", "trend_only": "#888780", "ridge": "#378ADD", "random_forest": "#1D9E75",
     "gbm": "#534AB7", "gbm_no_enso": "#7F77DD", "gbm_q50": "#AFA9EC", "gbm_q10": "#D85A30"}
NAMES = {"trend_only": "Trend only", "ridge": "Ridge", "random_forest": "Random forest", "gbm": "Gradient boosting",
         "gbm_no_enso": "Main: GB + soil + NDVI", "gbm_q50": "GB median (q50)", "gbm_q10": "GB bad-year (q10)"}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def main():
    pr = pd.read_csv(OUT / "predictions.csv")
    met = pd.read_csv(OUT / "metrics.csv")
    ins = pr[pr.feature_set == "in_season"]
    pre = pr[pr.feature_set == "pre_season"]
    models = ["trend_only", "ridge", "random_forest", "gbm", "gbm_no_enso", "gbm_q50"]

    # --- year level -------------------------------------------------------
    yr = ins.groupby("year")[["obs"] + models + ["gbm_q10"]].mean()
    yr_pre = pre.groupby("year")[["obs"] + models].mean()
    ylev = pd.DataFrame({
        "model": models,
        "in_season_r2_year": [r2_score(yr.obs, yr[m]) for m in models],
        "in_season_corr_year": [np.corrcoef(yr.obs, yr[m])[0, 1] if yr[m].std() > 0 else np.nan for m in models],
        "pre_season_r2_year": [r2_score(yr_pre.obs, yr_pre[m]) for m in models],
    })
    ylev.to_csv(OUT / "year_level_skill.csv", index=False)
    key = yr.loc[[1988, 2001, 2002, 2021, 2023], ["obs", "ridge", "random_forest", "gbm_no_enso", "gbm_q10"]]
    key.to_csv(OUT / "key_years.csv")

    # --- per crop ---------------------------------------------------------
    pc = ins.groupby("crop").apply(lambda g: pd.Series({
        "n": len(g), "r2_random_forest": r2_score(g.obs, g.random_forest),
        "r2_gbm_no_enso": r2_score(g.obs, g.gbm_no_enso), "obs_sd": g.obs.std()})).sort_values("r2_random_forest", ascending=False)
    pc.to_csv(OUT / "per_crop_skill.csv")

    # --- fit on all years for SHAP + surrogate ---------------------------
    df = load()
    df["anom"] = anomalies(df, test_year=-1)
    df = df.dropna(subset=["anom"]).reset_index(drop=True)
    feats = PRE + IN
    Xa = X(df, feats)
    gbm = make("gbm").fit(Xa, df.anom)
    sv = shap.TreeExplainer(gbm).shap_values(Xa)
    imp = pd.DataFrame({"feature": Xa.columns, "mean_abs_shap": np.abs(sv).mean(0),
                        "corr_value_vs_shap": [np.corrcoef(Xa[c], sv[:, i])[0, 1] if Xa[c].std() > 0 else 0
                                               for i, c in enumerate(Xa.columns)]})
    imp = imp[~imp.feature.str.startswith("crop_")].sort_values("mean_abs_shap", ascending=False)
    imp["label"] = imp.feature.map(LABEL)
    imp.to_csv(OUT / "shap_importance.csv", index=False)

    # surrogate: shallow tree mimicking the forest (weather only, crop-agnostic rules)
    rf = make("random_forest").fit(Xa, df.anom)
    target = rf.predict(Xa)
    Xw = df[IN + PRE].rename(columns=LABEL)
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=60, random_state=0).fit(Xw, target)
    fid = r2_score(target, tree.predict(Xw))
    rules = export_text(tree, feature_names=list(Xw.columns), decimals=3)
    rules = "\n".join(
        line.replace("value: [", "avg anomaly: ").replace("]", "") for line in rules.splitlines())
    (OUT / "surrogate_tree.txt").write_text(
        f"Shallow tree (depth 3) that mimics the random forest.\nFidelity R² to the forest: {fid:.2f}\n"
        f"Leaf values = average relative yield anomaly (−0.20 = 20% below trend).\n\n{rules}\n")

    # --- El Niño check ----------------------------------------------------
    df["enso"] = pd.cut(df.oni_djf, [-9, -0.5, 0.5, 9], labels=["La Niña", "Neutral", "El Niño"])
    enso = df.groupby(["enso", "crop"]).anom.mean().unstack(0)
    enso["all_crops_n_years"] = np.nan
    eyrs = df.groupby("enso").year.nunique()
    enso_all = df.groupby("enso").agg(mean_anom=("anom", "mean"), fail_rate=("anom", lambda s: (s < FAIL).mean()))
    enso_all["years"] = eyrs
    enso_all.to_csv(OUT / "enso_check.csv")
    enso.drop(columns="all_crops_n_years").to_csv(OUT / "enso_by_crop.csv")

    # --- charts -----------------------------------------------------------
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    m_in = met[met.feature_set == "in_season"].set_index("model")
    m_pre = met[met.feature_set == "pre_season"].set_index("model")

    # 1) model comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    order = ["trend_only", "ridge", "random_forest", "gbm", "gbm_no_enso", "gbm_q50"]
    for ax, col, title, fmt in [(axes[0], "mae", "Average error (lower is better)", "{:.0%}"),
                                (axes[1], "bad_years_flagged", "Bad years flagged (higher is better)", "{:.0%}")]:
        v_in, v_pre = m_in.loc[order, col], m_pre.loc[order, col]
        yy = np.arange(len(order))
        ax.barh(yy - 0.2, v_in, 0.38, color=[C[m] for m in order])
        ax.barh(yy + 0.2, v_pre, 0.38, color=[C[m] for m in order], alpha=0.35)
        for i, (a, b) in enumerate(zip(v_in, v_pre)):
            ax.text(a + 0.005, i - 0.2, fmt.format(a), va="center", fontsize=8, color=INK)
            ax.text(b + 0.005, i + 0.2, fmt.format(b), va="center", fontsize=8, color=MUTED)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_yticks(yy, [NAMES[m] for m in order]); ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=11, color=INK); style(ax); ax.grid(axis="x", color=GRID, lw=.6); ax.grid(axis="y", visible=False)
    axes[1].set_yticklabels([])
    fig.suptitle("Leave-one-year-out test, 10 Saskatchewan RMs, 8 crops, 1982–2025\n"
                 "Solid bar = with in-season weather · faded bar = pre-season data only",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(); fig.savefig(OUT / "chart_model_comparison.png", dpi=160); plt.close(fig)

    # 2) year-level obs vs predicted
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axhline(0, color=GRID, lw=1)
    ax.fill_between(yr.index, yr.gbm_q10, 0, color=C["gbm_q10"], alpha=0.12, label="Bad-year range (q10 model)")
    ax.plot(yr.index, yr.obs, color=C["obs"], lw=2, marker="o", ms=3, label="Actual")
    ax.plot(yr.index, yr.random_forest, color=C["random_forest"], lw=1.6, label="Random forest (in-season)")
    ax.plot(yr_pre.index, yr_pre.random_forest, color=C["random_forest"], lw=1.2, ls="--", alpha=0.7, label="Random forest (pre-season only)")
    for y_, t in [(1988, "1988"), (2002, "2002"), (2021, "2021")]:
        ax.annotate(t, (y_, yr.obs[y_]), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Yield vs trend", color=MUTED); style(ax)
    ax.set_title("Each year, averaged over all RMs and crops: actual vs predicted (year hidden from training)", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(); fig.savefig(OUT / "chart_years.png", dpi=160); plt.close(fig)

    # 3) SHAP importance
    top = imp.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    cols = ["#D85A30" if c < 0 else "#1D9E75" for c in top.corr_value_vs_shap]
    ax.barh(top.label, top.mean_abs_shap, color=cols)
    ax.set_title("What drives the yield anomaly (SHAP, gradient boosting)", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("Average impact on yield anomaly", color=MUTED); style(ax); ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=.6)
    ax.text(0.99, 0.02, "orange = more of it lowers yield\ngreen = more of it raises yield", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=MUTED)
    fig.tight_layout(); fig.savefig(OUT / "chart_shap.png", dpi=160); plt.close(fig)

    # 4) hot days vs anomaly
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.scatter(df.hot_days_flower + np.random.default_rng(0).uniform(-.3, .3, len(df)), df.anom, s=6, alpha=0.25, color="#534AB7")
    b = df.groupby(pd.cut(df.hot_days_flower, [-1, 4, 9, 14, 19, 40])).anom.median()
    ax.plot([2, 7, 12, 17, 24], b.values, color="#D85A30", lw=2.5, marker="o", label="Median")
    ax.axhline(0, color=GRID, lw=1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Days above 30 °C, Jun 15 – Jul 31", color=MUTED); ax.set_ylabel("Yield vs trend", color=MUTED); style(ax)
    ax.set_title("Heat at flowering vs yield (all RMs, crops, years)", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "chart_heat.png", dpi=160); plt.close(fig)

    print(ylev.round(3).to_string(index=False)); print(key.round(3)); print(pc.round(3))
    print(imp[["label", "mean_abs_shap", "corr_value_vs_shap"]].round(4).to_string(index=False))
    print(f"surrogate fidelity {fid:.2f}"); print(rules)
    print(enso_all.round(3)); print(enso.drop(columns='all_crops_n_years').round(3))
    print("heat bins median:", b.round(3).to_dict())


if __name__ == "__main__":
    main()
