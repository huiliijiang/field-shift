"""Export the Saskatchewan replay for the static Field Shift demo (02_demo/canada/index.html).

The browser page does no modelling. It filters and formats the rows written here, the same rows the Streamlit
app shows (app/data.py -> scripts/recommend_yield.py, scripts/recommend.py).

Output: 02_demo/canada/data.json
  rotations[rm][crop_2026] = all 64 two-year rotations, scored on the 44 replayed summers (rules NOT applied)
  rules["2025|2026|2027|2028"] = broken rotation rules for that sequence ("-" = 2025 crop unknown)
  alerts[rm] = in-season alerts and how often each fired
"""
import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import data as D  # noqa: E402

R, Y = D.R, D.Y
OUT = ROOT.parents[1] / "02_demo" / "canada" / "data.json"
CROPS = list(R.CROP_MAP)


def r3(x):
    return round(float(x), 7)


def main():
    pts = D.rm_points()
    rms = [dict(rm=int(r.rm), label=r.label, zone=r.guide_soil_zone, lat=float(r.lat), lon=float(r.lon))
           for r in pts.itertuples()]
    # district history: each summer's yield as a share of normal (all crops averaged), plus hot days at flowering
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_models import anomalies, load
    df = load(); df["anom"] = anomalies(df, -1)   # trend on all years, descriptive only (as in recommend.alerts_table)
    feat = pd.read_csv(ROOT / "data" / "processed" / "features.csv")[["rm", "year", "hot_days_flower", "rain_may_jun"]]
    yr = df.groupby(["rm", "year"]).anom.mean().reset_index().merge(feat, on=["rm", "year"])
    history = {int(rm): [dict(y=int(r.year), rel=round(1 + float(r.anom), 4), hot=int(r.hot_days_flower),
                              rain=round(float(r.rain_may_jun)))
                         for r in g.sort_values("year").itertuples()] for rm, g in yr.groupby("rm")}
    rotations, alerts = {}, {}
    for rm in [x["rm"] for x in rms]:
        rotations[rm] = {}
        for prev in CROPS:
            d = D._scored(rm, prev, check_rules=False)
            rotations[rm][prev] = [dict(
                c1=r.crop_2027, c2=r.crop_2028, avg=r3(r.avg_rel), pfail=r3(r.p_fail), cvar=r3(r.cvar10),
                k21=r3(r.kept_2021), n1=r3(r.normal_2027), n2=r3(r.normal_2028), d1=r3(r.y2021_2027),
                d2=r3(r.y2021_2028), m=r3(r.margin_yr), m21=r3(r.margin_2021_yr), ploss=r3(r.p_loss),
                e1=r3(R.yield_mult(prev, r.crop_2027) - 1), e2=r3(R.yield_mult(r.crop_2027, r.crop_2028) - 1))
                for r in d.itertuples()]
        a = R.alerts(rm)
        alerts[rm] = [dict(alert=r.trigger, known_by=r.known_by, action=r.suggested_action,
                           fired=int(r.years_fired), n=int(r.n_rm_years),
                           years=r.years_list if isinstance(r.years_list, str) else "") for r in a.itertuples()]
        print("rm", rm, "done")
    rules = {}
    for b in [None] + CROPS:
        for p in CROPS:
            if b and R.violations([b, p]):
                rules[f"{b}|{p}"] = "history"  # 2025 -> 2026 already breaks a rule: the app treats 2025 as unknown
            for c1 in CROPS:
                for c2 in CROPS:
                    v = R.violations([b, p, c1, c2])
                    if v:
                        rules[f"{b or '-'}|{p}|{c1}|{c2}"] = "; ".join(v)
    out = dict(source="04_reproduce/saskatchewan (scripts/export_demo.py)", crops=CROPS,
               nice={c: R.nice(c) for c in CROPS}, units=R.UNITS, pulses=sorted(R.PULSES), cereals=sorted(R.CEREALS),
               profiles=Y.PROFILES, tie=D.TIE, rms=rms, history=history, rotations=rotations, rules=rules, alerts=alerts)
    OUT.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
    print("wrote", OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
