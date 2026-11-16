"""Soil and satellite inputs of the main model. Both passed the written adopt rule and the one-time exam on
untouched 2018-2025 (see 03_evidence/canada/exam_soil.csv and exam_ndvi.csv).

  SOIL + SOILX : SoilGrids soil per RM (static, known before planting) and soil water x weather.
                 Used in the in-season model AND the 2027 scenario engine.
  NDEV         : MODIS NDVI (2000-2025) as deviation from the RM's mean over TRAINING years only.
                 Known only once the season is under way, so in-season model only. Missing before 2000
                 (gradient boosting handles missing values).
"""
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOIL = ["awc_100", "awc_30", "clay_30", "sand_30", "soc_30"]
SOILX = ["awc_x_heat", "awc_x_rain"]
ND = ["ndvi_jul1", "ndvi_peak", "ndvi_jul", "ndvi_int"]
NDEV = [c + "_d" for c in ND]
NDVI_YEARS = range(2000, 2026)


def soil_interactions(d):
    d["awc_x_heat"] = d.awc_100 * d.hot_days_flower
    d["awc_x_rain"] = d.awc_100 * d.rain_may_jun
    return d


def add_soil(d):
    s = pd.read_csv(ROOT / "data/experiments/soil_features.csv")[["rm"] + SOIL]
    return soil_interactions(d.drop(columns=[c for c in SOIL if c in d.columns]).merge(s, on="rm", how="left"))


def add_ndvi(d):
    n = pd.read_csv(ROOT / "data/experiments/ndvi_features_2000_2025.csv")[["rm", "year"] + ND]
    return d.merge(n, on=["rm", "year"], how="left")


def ndvi_dev(d, train_years):
    """NDVI minus the RM's mean over `train_years` (no test-year information)."""
    m = d[d.year.isin(list(train_years))].drop_duplicates(["rm", "year"]).groupby("rm")[ND].mean()
    for c in ND:
        d[c + "_d"] = d[c] - d.rm.map(m[c])
    return d
