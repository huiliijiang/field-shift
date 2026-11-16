"""Yield-only recommendation (no prices). Money is an optional separate view (scripts/recommend.py).

Each crop is scored against its OWN normal (trend) yield, so canola bushels and lentil pounds are comparable:
  relative yield r = yield × rotation effect / trend      (1.0 = a normal crop)
Per 2-year rotation (2027, 2028), over the joint weather scenarios from the forecast layer:
  avg_rel      average relative yield over the 2 years
  p_fail       chance either year falls > 20% below normal
  cvar10       average of the worst 10% of 2-year outcomes
  kept_2021    relative yield kept if a 2021-like summer hits (the worse of the two years)
Profiles (weight on the bad tail): best_average 0, balanced 0.5, most_reliable 0.8.
Backtest (staged: pilot 4 RMs, confirm 6 RMs): past-only picks scored on ACTUAL yields vs own past trend.
"""
import pathlib, sys
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
import recommend as R  # noqa: E402
OUT = ROOT / "results" / "recommend_yield"
OUT.mkdir(parents=True, exist_ok=True)
PROFILES = {"best_average": 0.0, "balanced": 0.5, "most_reliable": 0.8}
FAIL = 0.80


def score(rm, history=(None, "Canola"), effects=True, check_rules=True):
    """All 2-year rotations for an RM. check_rules=False also scores rule-breaking ones (e.g. a farmer's usual plan)
    and lists the broken rules in `rule_issue`."""
    sp = R._sampler(rm); crops = sp["crops"]
    Y1, _ = R.joint_yields(rm, 2027); Y2, _ = R.joint_yields(rm, 2028)
    t27, t28 = sp["t27"], sp["t27"] + sp["slope"]
    rows = []
    for i, c1 in enumerate(crops):
        for j, c2 in enumerate(crops):
            issue = R.violations(list(history) + [c1, c2])
            if issue and check_rules:
                continue
            m1 = R.yield_mult(history[-1], c1) if effects else 1.0
            m2 = R.yield_mult(c1, c2) if effects else 1.0
            r1, r2 = Y1[:, i] * m1 / t27[i], Y2[:, j] * m2 / t28[j]
            tot = (r1 + r2) / 2
            k21 = min((1 + sp["rep2021"][i]) * m1, (1 + sp["rep2021"][j]) * m2)
            rows.append(dict(rm=rm, crop_2027=c1, crop_2028=c2, rotation=f"{R.nice(c1)} → {R.nice(c2)}",
                             avg_rel=tot.mean(), p_fail=((r1 < FAIL) | (r2 < FAIL)).mean(), cvar10=R.cvar(tot),
                             kept_2021=k21, rotation_boost=(m1 + m2) / 2 - 1,
                             normal_2027=t27[i] * m1, normal_2028=t28[j] * m2,
                             y2021_2027=(1 + sp["rep2021"][i]) * m1 * t27[i], y2021_2028=(1 + sp["rep2021"][j]) * m2 * t28[j],
                             unit_2027=R.UNITS[c1], unit_2028=R.UNITS[c2], rule_issue="; ".join(issue)))
    d = pd.DataFrame(rows)
    for p, w in PROFILES.items():
        d[f"score_{p}"] = (1 - w) * d.avg_rel + w * d.cvar10
    d["rank_2021"] = d.kept_2021.rank(ascending=False, method="min").astype(int)
    return d


def risk(p_fail, cv):
    return "🟢 low" if p_fail <= 0.25 and cv >= 0.70 else ("🔴 high" if p_fail > 0.45 or cv < 0.55 else "🟡 medium")


def recommend(rm, previous_crop="Canola", profile="balanced", n_top=3, effects=True):
    d = score(rm, (None, previous_crop), effects).sort_values(f"score_{profile}", ascending=False).head(n_top).copy()
    n = len(score(rm, (None, previous_crop), effects))
    d.insert(0, "rank", range(1, len(d) + 1))
    d["risk_level"] = [risk(p, c) for p, c in zip(d.p_fail, d.cvar10)]
    d["bad_year_like_2021"] = [f"keeps ~{k:.0%} of a normal crop (#{r} of {n})" for k, r in zip(d.kept_2021, d.rank_2021)]
    d["reason"] = [("pulse first: the cereal after it gets a yield boost; " if c1 in R.PULSES and c2 in R.CEREALS else "")
                   + f"fails (>20% below normal) in {p:.0%} of scenarios" for c1, c2, p in zip(d.crop_2027, d.crop_2028, d.p_fail)]
    d["profile"] = profile
    return d


# ----------------------------------------------------------------- backtest (yield only)
def backtest(rms, effects=True, n_random=50):
    Yp = R._bt_data(); rows = []
    for rm in rms:
        crops = [c for c in R.CROP_MAP if c in Yp.columns and Yp.loc[rm][c].notna().sum() >= 10]
        paths = {p: [None, "Canola"] for p in PROFILES} | {"canola_wheat": [None, "Canola"]}
        rp = [[None, "Canola"] for _ in range(n_random)]
        rng = np.random.default_rng(rm)
        for t in range(R.BT_START, R.BT_END, 2):
            avail = [c for c in crops if not np.isnan(Yp.loc[(rm, t)].get(c, np.nan)) and not np.isnan(Yp.loc[(rm, t + 1)].get(c, np.nan))]
            ok, A, tr0, tr1 = R._past_dist(Yp, rm, t, avail, rng)
            idx = {c: k for k, c in enumerate(ok)}

            def pick(hist, w):
                best, bs = None, -np.inf
                for c1 in ok:
                    for c2 in ok:
                        if R.violations(list(hist) + [c1, c2]):
                            continue
                        m1 = R.yield_mult(hist[-1], c1) if effects else 1.0
                        m2 = R.yield_mult(c1, c2) if effects else 1.0
                        tot = (((1 + A[:, idx[c1]]) * m1)[:, None] + ((1 + A[:, idx[c2]]) * m2)[None, :]).ravel() / 2
                        s = (1 - w) * tot.mean() + w * R.cvar(tot)
                        if s > bs:
                            best, bs = (c1, c2), s
                return best

            picks = {p: pick(tuple(paths[p][-2:]), w) for p, w in PROFILES.items()}
            picks["canola_wheat"] = ("Spring Wheat", "Canola")
            for s, (c1, c2) in picks.items():
                prev = paths[s][-1]
                for c, p, yy, tr in ((c1, prev, t, tr0), (c2, c1, t + 1, tr1)):
                    rel = (Yp.loc[(rm, yy)][c] * (R.yield_mult(p, c) if effects else 1.0) / tr[idx[c]]) if c in idx else np.nan
                    rows.append(dict(rm=rm, year=yy, strategy=s, crop=c, rel=rel))
                paths[s] += [c1, c2]
            for r in rp:
                seqs = [(a, b) for a in ok for b in ok if not R.violations(r[-2:] + [a, b])]
                c1, c2 = seqs[rng.integers(len(seqs))]
                for c, p, yy, tr in ((c1, r[-1], t, tr0), (c2, c1, t + 1, tr1)):
                    rows.append(dict(rm=rm, year=yy, strategy="random", crop=c,
                                     rel=Yp.loc[(rm, yy)][c] * (R.yield_mult(p, c) if effects else 1.0) / tr[idx[c]]))
                r += [c1, c2]
    return pd.DataFrame(rows)


def summarize(bt):
    bt = bt.dropna(subset=["rel"])
    g = bt.groupby("strategy").rel
    s = pd.DataFrame({"avg_rel_yield": g.mean(), "fail_rate": g.apply(lambda x: (x < FAIL).mean()),
                      "worst10_avg": g.apply(lambda x: x[x <= x.quantile(0.1)].mean())})
    yr = bt.groupby(["strategy", "year"]).rel.mean().unstack(0)
    s["worst_year_avg"] = yr.min()
    s["yr_2021"] = yr.loc[2021] if 2021 in yr.index else np.nan
    return s


def main():
    # pilot RMs (used for tuning) and confirm RMs (never used for tuning)
    PILOT, CONFIRM = [137, 159, 5, 244], [67, 161, 290, 344, 437, 461]
    out = []
    for rm in PILOT + CONFIRM:
        for p in PROFILES:
            out.append(recommend(rm, "Canola", p))
    rec = pd.concat(out); rec.to_csv(OUT / "recommendations_yield.csv", index=False)
    print(rec[rec.rm == 159][["profile", "rank", "rotation", "risk_level", "bad_year_like_2021", "avg_rel", "p_fail", "cvar10"]].round(3).to_string(index=False))
    for st, rms in [("pilot", PILOT), ("confirm", CONFIRM)]:
        for eff in (True, False):
            bt = backtest(rms, eff); s = summarize(bt)
            s.insert(0, "rotation_effects", eff); s.insert(0, "stage", st)
            s.to_csv(OUT / f"backtest_{st}_{'fx' if eff else 'nofx'}.csv")
            print(f"\n== backtest {st}, rotation effects={eff} ==\n", s.round(3).to_string())


if __name__ == "__main__":
    main()
