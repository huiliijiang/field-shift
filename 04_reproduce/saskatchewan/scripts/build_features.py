"""Build one row of season features per RM-year from NASA POWER daily data + ONI.

Two kinds of features:
  PRE-SEASON (known by ~May 1, when the planting decision is final)
    rain_fall_winter   Sep(Y-1)–Apr(Y) precipitation, mm (stored water + snow)
    wet_root_may       root-zone soil wetness, Apr 25–May 5 mean (0–1)
    oni_son_prev       El Niño index, Sep–Nov of the year before (known in autumn)
    oni_djf            El Niño index, Dec–Feb before planting
  IN-SEASON (what actually happened during the growing season)
    rain_may_jun, rain_jul_aug      mm
    dry_spell_max      longest run of days with < 1 mm rain, May 15–Aug 31
    hot_days_flower    days with Tmax > 30 °C, Jun 15–Jul 31 (flowering)
    tmax_jul           mean daily max temperature in July, °C
    gdd_may_aug        growing degree days, base 5 °C
    last_spring_frost  day-of-year of last Tmin <= -2 °C between Mar 1 and Jun 30
    first_fall_frost   day-of-year of first Tmin <= -2 °C after Aug 1
    wet_root_min       lowest root-zone wetness, Jun–Aug

Output: data/processed/features.csv
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"

PRE = ["rain_fall_winter", "wet_root_may", "oni_son_prev", "oni_djf"]
IN = ["rain_may_jun", "rain_jul_aug", "dry_spell_max", "hot_days_flower", "tmax_jul",
      "gdd_may_aug", "last_spring_frost", "first_fall_frost", "wet_root_min"]


def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def season_features(d, year):
    """d: daily frame for one RM indexed by date."""
    def win(a, b):
        return d.loc[a:b]
    f = {}
    f["rain_fall_winter"] = win(f"{year-1}-09-01", f"{year}-04-30").PRECTOTCORR.sum()
    f["wet_root_may"] = win(f"{year}-04-25", f"{year}-05-05").GWETROOT.mean()
    f["rain_may_jun"] = win(f"{year}-05-01", f"{year}-06-30").PRECTOTCORR.sum()
    f["rain_jul_aug"] = win(f"{year}-07-01", f"{year}-08-31").PRECTOTCORR.sum()
    f["dry_spell_max"] = longest_run((win(f"{year}-05-15", f"{year}-08-31").PRECTOTCORR < 1).values)
    f["hot_days_flower"] = int((win(f"{year}-06-15", f"{year}-07-31").T2M_MAX > 30).sum())
    f["tmax_jul"] = win(f"{year}-07-01", f"{year}-07-31").T2M_MAX.mean()
    f["gdd_may_aug"] = (win(f"{year}-05-01", f"{year}-08-31").T2M - 5).clip(lower=0).sum()
    spring = win(f"{year}-03-01", f"{year}-06-30")
    frost = spring.index[spring.T2M_MIN <= -2]
    f["last_spring_frost"] = frost.max().dayofyear if len(frost) else 60
    fall = win(f"{year}-08-01", f"{year}-12-31")
    frost = fall.index[fall.T2M_MIN <= -2]
    f["first_fall_frost"] = frost.min().dayofyear if len(frost) else 365
    f["wet_root_min"] = win(f"{year}-06-01", f"{year}-08-31").GWETROOT.min()
    return f


def main():
    w = pd.read_csv(P / "power_daily.csv", parse_dates=["date"])
    oni = pd.read_csv(P / "oni.csv")
    son = oni[oni.season == "SON"].set_index("year").oni
    djf = oni[oni.season == "DJF"].set_index("year").oni

    rows = []
    for rm, d in w.groupby("rm"):
        d = d.set_index("date").sort_index()
        for year in range(1982, 2026):  # 1981 lacks the previous autumn
            f = season_features(d, year)
            f.update(rm=rm, year=year, oni_son_prev=son.get(year - 1, np.nan), oni_djf=djf.get(year, np.nan))
            rows.append(f)
    feats = pd.DataFrame(rows)[["rm", "year"] + PRE + IN]
    feats.to_csv(P / "features.csv", index=False)
    print(feats.shape)
    print(feats.describe().T[["mean", "min", "max"]].round(2))


if __name__ == "__main__":
    main()
