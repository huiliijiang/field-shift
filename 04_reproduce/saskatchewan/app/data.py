"""Selection and formatting layer behind the Canada page (02_demo/canada), used by scripts/export_demo.py.

Every number shown on the page comes from scripts/recommend_yield.py, scripts/recommend.py or the CSVs
under data/ and results/. This module only selects, renames and formats.
"""
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT / "scripts", ROOT / "scripts" / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import recommend as R  # noqa: E402  (money view, crop mix, insurance, alerts)
import recommend_yield as Y  # noqa: E402  (yield-only view, the default)

PREVIOUS_CROPS = list(R.CROP_MAP)  # all 8 crops
PRIORITIES = ["Best average", "Balanced", "Most reliable"]
YIELD_PROFILE = {"Best average": "best_average", "Balanced": "balanced", "Most reliable": "most_reliable"}
MONEY_PROFILE = {"Best average": "max_profit", "Balanced": "balanced", "Most reliable": "steady_income"}
CHART_YEARS = ROOT / "results" / "chart_years.png"


def rm_points():
    pts = pd.read_csv(ROOT / "data" / "rm_points.csv")
    zone = pd.read_csv(ROOT / "data" / "economics" / "rm_soil_zone.csv")[["rm", "guide_soil_zone"]]
    pts = pts.merge(zone, on="rm", how="left")
    pts["label"] = pts.rm_name + " (RM " + pts.rm.astype(str) + ")"
    return pts


def _light(risk_text):
    t = str(risk_text).lower()
    if "low" in t:
        return "🟢", "Low risk"
    if "high" in t:
        return "🔴", "High risk"
    return "🟡", "Medium risk"


def _part(loss_control, prefix):
    for s in str(loss_control).split(" | "):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
            return s[:1].upper() + s[1:]
    return None


def _amt(x, unit):
    return f"{x:,.0f} {unit}"


def _yields_line(r):
    c1, c2 = R.nice(r.crop_2027).lower(), R.nice(r.crop_2028).lower()
    return (f"District-average yield in a normal year: **{c1} ~{_amt(r.normal_2027, r.unit_2027)}** (2027), "
            f"**{c2} ~{_amt(r.normal_2028, r.unit_2028)}** (2028). "
            f"If a summer like 2021 came back: ~{_amt(r.y2021_2027, r.unit_2027)} and ~{_amt(r.y2021_2028, r.unit_2028)}.")


def history_issue(previous_crop, crop_before):
    """A 2025→2026 history that already breaks a rule (e.g. canola on canola) is treated as unknown 2025."""
    return R.violations([crop_before, previous_crop]) if crop_before else []


def _add_margins(d, rm, previous_crop):
    """$/ac at 2026 guide prices for each rotation, same scenarios as the yield scores (per-year averages)."""
    sp = R._sampler(rm); crops = list(sp["crops"])
    Y1, _ = R.joint_yields(rm, 2027); Y2, _ = R.joint_yields(rm, 2028)
    t28 = sp["t27"] + sp["slope"]
    em, cv, b21, pl = [], [], [], []
    for c1, c2 in zip(d.crop_2027, d.crop_2028):
        i, j = crops.index(c1), crops.index(c2)
        m1 = R.margin(rm, c1, previous_crop, Y1[:, i]); m2 = R.margin(rm, c2, c1, Y2[:, j]); tot = m1 + m2
        r1 = R.margin(rm, c1, previous_crop, sp["t27"][i] * (1 + sp["rep2021"][i]))
        r2 = R.margin(rm, c2, c1, t28[j] * (1 + sp["rep2021"][j]))
        em.append(tot.mean() / 2); cv.append(R.cvar(tot)); b21.append(min(r1 + m2.mean(), m1.mean() + r2) / 2)
        pl.append((tot < 0).mean())
    d = d.assign(margin_yr=em, margin_cvar_2yr=cv, margin_2021_yr=b21, p_loss=pl)
    for p, w in Y.PROFILES.items():
        d[f"money_{p}"] = (1 - w) * d.margin_yr * 2 + w * d.margin_cvar_2yr
    return d


def _scored(rm, previous_crop, crop_before=None, exclude=(), check_rules=True):
    if history_issue(previous_crop, crop_before):
        crop_before = None
    d = Y.score(rm, (crop_before, previous_crop), check_rules=check_rules)
    if exclude:
        d = d[~d.crop_2027.isin(exclude) & ~d.crop_2028.isin(exclude)]
    return _add_margins(d.reset_index(drop=True), rm, previous_crop)


def usual_plan(rm, previous_crop, crop_2027, crop_2028, crop_before=None):
    """The farmer's own plan, scored on the same past summers, even if it breaks a rotation rule."""
    d = _scored(rm, previous_crop, crop_before, check_rules=False)
    d = d[(d.crop_2027 == crop_2027) & (d.crop_2028 == crop_2028)]
    return None if d.empty else d.iloc[0]


def _money_amt(x):
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def _md(t):
    """Escape $ for markdown renderers that read it as math."""
    return t.replace("$", "\\$")


def _crop_amts(r, a, b):
    u1, u2 = r.unit_2027.split("/")[0], r.unit_2028.split("/")[0]
    return (f"{R.nice(r.crop_2027).lower()} {getattr(r, a):,.0f} {u1} · {R.nice(r.crop_2028).lower()} {getattr(r, b):,.0f} {u2}")


TIE = 0.02  # rotations within 2 points of a normal crop of the best score are treated as tied (rank_stability.py)


def _verdict(g, u, rm_label, u_in):
    """One plain bottom line about the tied group. Likelihoods from past summers, never a promise."""
    lo, hi = g.avg_rel.min(), g.avg_rel.max()
    if len(g) == 1:
        r = g.iloc[0]
        txt = (f"**Bottom line for {rm_label}:** across 44 past summers, **{r.rotation}** held up best for your settings "
               f"({r.avg_rel:.0%} of a normal crop on average); no other allowed rotation came within 2 points. "
               f"In a drought like 2021 it kept about {r.kept_2021:.0%} of a normal crop, and it earned about "
               f"{_money_amt(r.margin_yr)}/ac a year at 2026 guide prices.")
        if u is not None:
            txt += (f" Your usual plan (**{u.rotation}**): {u.avg_rel:.0%} on average, {u.kept_2021:.0%} in a 2021-like "
                    f"drought, {_money_amt(u.margin_yr)}/ac a year.")
        return _md(txt)
    txt = (f"**Bottom line for {rm_label}:** across 44 past summers, **{len(g)} rotations held up about the same** "
           f"on average ({lo:.0%}–{hi:.0%} of a normal crop). Differences that small are not a reliable reason to pick one.")
    if u is not None:
        if u.rule_issue:
            txt += f" Your usual plan (**{u.rotation}**) breaks a rotation guideline, so it is not in the group."
        elif u_in:
            txt += f" Your usual plan (**{u.rotation}**) is one of them."
        else:
            txt += (f" Your usual plan (**{u.rotation}**) did a little worse on average ({u.avg_rel:.0%} of normal).")
    d = g.sort_values("kept_2021"); m = g.sort_values("margin_yr")
    txt += (f" **What does differ:** in the 2021 drought, the group ranged from about {d.kept_2021.iloc[0]:.0%} "
            f"({d.rotation.iloc[0]}) to {d.kept_2021.iloc[-1]:.0%} ({d.rotation.iloc[-1]}) of a normal crop kept, and "
            f"at 2026 guide prices from {_money_amt(m.margin_yr.iloc[0])} to {_money_amt(m.margin_yr.iloc[-1])}/ac a year. "
            "Treat 2021 as history, not protection: in a past-only test, picking the rotation that held up best in "
            "earlier droughts did not keep more in the next one. The $/ac gap depends on prices, which you can check.")
    return _md(txt)


def _rot_effect(prev, r):
    """Measured previous-crop effect (MASC, shrunk), as used in the scores."""
    m1, m2 = R.yield_mult(prev, r.crop_2027), R.yield_mult(r.crop_2027, r.crop_2028)
    parts = []
    for m, c, after in ((m1, r.crop_2027, prev), (m2, r.crop_2028, r.crop_2027)):
        if abs(m - 1) >= 0.005:
            parts.append(f"{R.nice(c).lower()} on {R.nice(after).lower()} stubble {m - 1:+.0%}")
    if not parts:
        return "No measurable previous-crop effect on yield for these two years."
    txt = "Yield effect of the previous crop: " + "; ".join(parts) + " (Manitoba fields, MASC 2011–20)."
    if r.crop_2027 in R.PULSES and r.crop_2028 in R.CEREALS:
        txt += " The cereal after a pulse also needs less nitrogen."
    return txt


def _group_line(r, g, best):
    if r.rule_issue:
        return "Not in the tied group: it breaks a rotation guideline."
    if ((g.crop_2027 == r.crop_2027) & (g.crop_2028 == r.crop_2028)).any():
        return ("The only rotation within 2 points of the best." if len(g) == 1 else
                f"In the tied group: held up about the same as the other {len(g) - 1} on average.")
    return (f"Not in the tied group: its score is {(best - r.score) * 100:.0f} points of a normal crop below the best "
            "for your settings.")


def _standouts(g):
    """Up to 2 members of the tied group, each labelled by what it is best at (not a rank)."""
    picks = []
    # bad-year chance barely varies inside the group (a few points), so it is not used to pick standouts
    for lab, col, asc in (("Best in 2021", "kept_2021", False),
                          ("Highest $/ac", "margin_yr", False)):
        i = g.sort_values([col, "score"], ascending=[asc, False]).index[0]
        hit = next((p for p in picks if p[0] == i), None)
        if hit:
            hit[1].append(lab)
        else:
            picks.append((i, [lab]))
    return [(g.loc[i], " · ".join(labs)) for i, labs in picks]


def options(rm, rm_label, previous_crop, priority, basis="yield", crop_before=None, exclude=(), plan=None):
    """Everything screen 1 needs: bottom line, the tied group, comparison table, chart data, per-option detail and the
    full list. Rotations are scored on yield (priority sets the weight on bad years). All rotations within TIE of the
    best score form the tied group: rank_stability.py showed their order comes from a small, uncertain rotation effect.
    `basis` is kept for old callers and ignored."""
    prof = YIELD_PROFILE[priority]
    allr = _scored(rm, previous_crop, crop_before, exclude)
    if allr.empty:
        return None
    allr["score"] = allr[f"score_{prof}"]
    allr = allr.sort_values("score", ascending=False).reset_index(drop=True)
    best = allr.score.iloc[0]
    g = allr[allr.score >= best - TIE]
    picks = _standouts(g)
    u = usual_plan(rm, previous_crop, plan[0], plan[1], crop_before) if plan else None
    if u is not None:
        u = u.copy(); u["score"] = u[f"score_{prof}"]
    u_in = u is not None and not u.rule_issue and ((g.crop_2027 == u.crop_2027) & (g.crop_2028 == u.crop_2028)).any()
    rows, labels = [], []
    if u is not None:
        rows.append(u); labels.append(f"Your plan: {u.rotation}")
    for r, lab in picks:
        if u is not None and r.crop_2027 == u.crop_2027 and r.crop_2028 == u.crop_2028:
            labels[0] += f" (also {lab.lower()} in group)"
            continue
        rows.append(r); labels.append(f"{r.rotation} ({lab.lower()})")
    t = pd.DataFrame(rows).reset_index(drop=True)
    t["label"] = labels
    t["short"] = ["Your plan: " + r.rotation if i == 0 and u is not None else r.rotation for i, r in enumerate(t.itertuples())]
    t["light"] = [_light(Y.risk(r.p_fail, r.cvar10))[0] for r in t.itertuples()]
    table = pd.DataFrame({
        "Option": [f"{lt} {l}" + (" (breaks a guideline)" if r.rule_issue else "")
                   for lt, l, r in zip(t.light, t.label, t.itertuples())],
        "Avg yield": [f"{r.avg_rel:.0%}" for r in t.itertuples()],
        "2021 drought": [f"{r.kept_2021:.0%}" for r in t.itertuples()],
        "$/ac": [_money_amt(r.margin_yr) for r in t.itertuples()],
        "$/ac 2021": [_money_amt(r.margin_2021_yr) for r in t.itertuples()],
    })
    chart = pd.concat([
        pd.DataFrame({"Option": t.short, "order": t.index, "view": v, "case": c, "value": t[col] * k})
        for v, c, col, k in (("Yield (% of normal)", "Average year", "avg_rel", 100),
                             ("Yield (% of normal)", "2021-like drought", "kept_2021", 100),
                             ("$/ac a year", "Average year", "margin_yr", 1),
                             ("$/ac a year", "2021-like drought", "margin_2021_yr", 1))], ignore_index=True)
    details = []
    for r in t.itertuples():
        u1, u2 = r.unit_2027, r.unit_2028
        yt = pd.DataFrame({
            "Year": ["2027", "2028"],
            "Crop": [R.nice(r.crop_2027), R.nice(r.crop_2028)],
            "Normal year": [_amt(r.normal_2027, u1), _amt(r.normal_2028, u2)],
            "2021-like drought": [_amt(r.y2021_2027, u1), _amt(r.y2021_2028, u2)],
        })
        details.append(dict(
            label=r.short, light=r.light, crops=(r.crop_2027, r.crop_2028), yields=yt,
            summary=_md(f"**Yield:** {r.avg_rel:.0%} of a normal crop on average; a bad year in {r.p_fail:.0%} of replays; "
                        f"{r.kept_2021:.0%} kept in a 2021-like drought (worse of the two years).  \n"
                        f"**Margin:** {_money_amt(r.margin_yr)}/ac a year on average, {_money_amt(r.margin_2021_yr)} "
                        f"in a 2021-like year; money lost over the two years in {r.p_loss:.0%} of replays."),
            rank=_group_line(r, g, best), effect=_rot_effect(previous_crop, r),
            rule_issue=r.rule_issue or None))
    in_g = allr.index < len(g)
    full = pd.DataFrame({
        "Tied group": ["✓" if x else "" for x in in_g],
        "Risk": [_light(Y.risk(r.p_fail, r.cvar10))[0] for r in allr.itertuples()],
        "Rotation": allr.rotation,
        "Avg yield %": (allr.avg_rel * 100).round(0),
        "2021-like %": (allr.kept_2021 * 100).round(0),
        "Worst 10% %": (allr.cvar10 * 100).round(0),
        "Bad-year chance %": (allr.p_fail * 100).round(0),
        "$/ac/yr": allr.margin_yr.round(0),
        "$/ac/yr 2021-like": allr.margin_2021_yr.round(0),
        "2-yr loss chance %": (allr.p_loss * 100).round(0),
    })
    return dict(verdict=_verdict(g, u, rm_label, u_in), table=table, chart=chart, details=details, full=full,
                top=picks[0][0], usual=u, n_options=len(allr), n_group=len(g), usual_in_group=u_in,
                warning=(f"Your usual plan breaks a rotation guideline ({u.rule_issue}). Past yields don't show the disease "
                         "and weed build-up this causes, so its numbers look better than they would be."
                         if u is not None and u.rule_issue else None))


def crop_ranges(rm, crops):
    """Per-crop 2027 scenario range relative to the crop's own normal (trend) yield."""
    f = pd.read_csv(ROOT / "results" / "forecast" / "forecast_2027.csv")
    f = f[(f.rm == rm) & (f.crop.isin(crops))].copy()
    out = pd.DataFrame({
        "Crop": [R.nice(c) for c in f.crop],
        "Typical year": [f"{x:.0%} of normal" for x in f.p50 / f.trend_2027],
        "Low end (1 year in 10)": [f"{x:.0%} of normal" for x in f.p10 / f.trend_2027],
        "High end (1 year in 10)": [f"{x:.0%} of normal" for x in f.p90 / f.trend_2027],
        "Bad-year chance": [f"{x:.0%}" for x in f.p_fail],
    })
    return out.reset_index(drop=True)


def protect(rm, previous_crop, crop_2027, priority):
    """Crop mix, insurance (for one rotation's 2027 crop) and alerts. Mix and insurance are priced at 2026 guide prices."""
    lc = R.loss_control(rm, previous_crop, crop_2027, MONEY_PROFILE[priority])
    res = dict(mix_text=_part(lc, "Crop mix:"), insurance_text=_part(lc, "Insurance:"), insurance_crop=crop_2027)
    s = R.crop_mix(rm, previous_crop)["summary"]
    s = s[s.option.str.contains("25% steps") | s.option.str.startswith("best single")]
    res["mix_table"] = pd.DataFrame({
        "Option": s.option.str.replace(" (25% steps)", "", regex=False),
        "2027 acres": s.mix,
        "Expected ($/ac)": s.exp_margin.round(0).astype(int),
        "Worst 10% avg ($/ac)": s.cvar10.round(0).astype(int),
        "Chance of a loss": [f"{x:.0%}" for x in s.p_loss],
        "2021 replay ($/ac)": s.bad2021_margin.round(0).astype(int)}).reset_index(drop=True)
    it = R.insurance_table(rm, crop_2027, previous_crop)
    res["insurance_table"] = pd.DataFrame({
        "Coverage": ["none" if c == 0 else f"{c:.0%}" for c in it.coverage],
        "Premium ($/ac)": it.premium.round(1),
        "Expected margin ($/ac)": it.exp_margin.round(0).astype(int),
        "Worst 10% avg ($/ac)": it.cvar10.round(0).astype(int),
        "2021 replay ($/ac)": it.bad2021_margin.round(0).astype(int)})
    a = R.alerts(rm)
    res["alerts"] = pd.DataFrame({
        "Alert": a.trigger,
        "Known by": a.known_by,
        "What to do": a.suggested_action,
        "Fired here": [f"{int(k)} of {int(n)} seasons" + (f" ({y})" if isinstance(y, str) and y else "")
                       for k, n, y in zip(a.years_fired, a.n_rm_years, a.years_list)],
    })
    heat = a[a.trigger_id == "heat_flowering"]
    res["heat_here"] = None if heat.empty or heat.years_fired.iloc[0] == 0 else dict(
        fired=int(heat.years_fired.iloc[0]), n=int(heat.n_rm_years.iloc[0]),
        anom=float(heat.mean_anom_fired.iloc[0]), anom_else=float(heat.mean_anom_not_fired.iloc[0]))
    return res
