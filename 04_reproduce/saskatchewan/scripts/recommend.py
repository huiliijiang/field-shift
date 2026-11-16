"""Recommendation + loss-control layer of the crop rotation advisor (Saskatchewan demo).

Builds on the 2027 scenario forecast (results/forecast, scripts/forecast.py) and the
2026 Sask Crop Planning Guide economics (data/economics).

  1. Margin per acre per scenario = forecast yield x guide price - guide variable cost
     (main view). Secondary view also subtracts the guide's other/fixed cost.
  2. Rotation candidates = 2-year sequences (2027, 2028) after the field's recent crops,
     filtered by hard rules, adjusted by soft effects (yield gains, N credit).
  3. Scoring: expected 2-yr margin, CVaR10, P(2-yr margin < 0), "2021 again" margin.
     Profiles = weighted mix of expected margin and CVaR10.
  4. Loss control: crop mix (efficient frontier), crop insurance approximation,
     in-season alerts.
  5. Backtest of the recommendation rule 1992-2025 with past-only information.

Joint sampling (keeps the cross-crop correlation):
  one draw = (weather year s, residual key k). All crops of the RM get the SAME s and the
  SAME residual key k = (RM', year') from results/forecast/residuals.csv, so a bad weather
  year and a bad "model error" year hit all crops together. Residual cells that are missing
  for a crop (crop not grown at RM' that year) are filled with a random draw from that crop's
  pool (independent: small share). 2027 and 2028 are independent draws.

Importable API (Streamlit):
    from recommend import recommend, crop_mix, insurance_table, alerts, score_sequences
    recommend(159, "Canola", "balanced")      # top-3 2-year sequences with reasons
    crop_mix(159, "Canola")                     # dict: summary rows + frontier
    insurance_table(159, "Lentils")
    alerts(159)

Run:  .venv/bin/python scripts/recommend.py      (~1-2 min)
"""
import functools, itertools, pathlib, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
ECON = ROOT / "data" / "economics"
FC = ROOT / "results" / "forecast"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "recommend"

# ------------------------------------------------------------------ assumptions
N_SAMPLES = 20000
CVAR_Q = 0.10
CROP_MAP = {"Canola": "canola", "Spring Wheat": "spring_wheat_cwrs", "Durum": "durum",
            "Barley": "barley_feed", "Oats": "oats", "Peas": "field_peas_yellow",
            "Lentils": "lentils_red", "Flax": "flax"}
CEREALS = {"Spring Wheat", "Durum", "Barley", "Oats"}
PULSES = {"Peas", "Lentils"}
UNITS = {c: "bu/ac" for c in CROP_MAP} | {"Lentils": "lb/ac"}
# N price: implied by the guide's own fertilizer budget: HRS wheat Dark Brown $74.50 for 91 lb N,
# Black $90.05 for 110 lb N, feed barley DB $65.49 for 80 lb N -> $0.82 per lb N (all three).
N_PRICE = 0.82
# lb N/ac credited to the crop after a pulse. Black 25 (SPG manual via Western Producer),
# Brown 10 (same article). Dark Brown 10: Gan et al. 2004 rate for Dark Brown (0.004 lb N per lb
# seed) x ~2,400 lb/ac pea crop ~ 10 lb (derived). Lentils assumed same as peas (ASSUMPTION).
N_CREDIT_LB = {"Brown": 10, "Dark Brown": 10, "Black": 25}
CEREAL_AFTER_PULSE = 0.15   # Gan 2004: wheat after pea +15-34% (low end); extended to all cereals (ASSUMPTION)
WHEAT_AFTER_CANOLA = 0.03   # CCC citing Manitoba crop insurance: wheat on canola stubble 103%; applied to wheat+durum
CANOLA_ON_CANOLA = -0.15    # CCC 10-20%, midpoint. Unreachable under the default hard rules.
INS_REF_COVERAGE = 0.70     # ASSUMPTION: the guide's premium corresponds to 70% coverage
PROFILES = {"max_profit": 0.0, "balanced": 0.5, "steady_income": 0.8}   # weight on CVaR10
PROFILE_LABEL = {"max_profit": "Max profit", "balanced": "Balanced", "steady_income": "Steady income"}
BT_START, BT_END, BT_MIN_YEARS = 1992, 2025, 8

# palette (dataviz reference instance; same as forecast charts)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, GRAY, SURF = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"
CROP_COLOR = {"Canola": YELLOW, "Spring Wheat": BLUE, "Durum": VIOLET, "Barley": AQUA, "Oats": GRAY,
              "Peas": GREEN, "Lentils": ORANGE, "Flax": MAGENTA}


# ------------------------------------------------------------------ data
@functools.lru_cache(None)
def _econ():
    e = pd.read_csv(ECON / "sk_crop_economics.csv")
    z = pd.read_csv(ECON / "rm_soil_zone.csv")
    return e, z.set_index("rm")


def zone(rm):
    return _econ()[1].loc[rm, "guide_soil_zone"]


def rm_name(rm):
    return _econ()[1].loc[rm, "rm_name"]


def econ(rm, crop):
    e = _econ()[0]
    r = e[(e.soil_zone == zone(rm)) & (e.crop == CROP_MAP[crop])]
    if r.empty:
        raise ValueError(f"no guide economics for {crop} in {zone(rm)}")
    r = r.iloc[0]
    return dict(price=r.price_per_unit, var=r.variable_cost_per_acre, fixed=r.fixed_cost_per_acre,
                premium=r.crop_insurance_premium_per_acre, guide_return=r.return_over_variable,
                guide_yield=r.expected_yield, guide_avg_yield=r.provincial_avg_yield, unit=r.yield_unit)


@functools.lru_cache(None)
def _fc():
    scen = pd.read_csv(FC / "scenarios_2027.csv")
    resid = pd.read_csv(FC / "residuals.csv")
    f27 = pd.read_csv(FC / "forecast_2027.csv")
    return scen, resid, f27


def rm_crops(rm):
    scen = _fc()[0]
    return [c for c in CROP_MAP if ((scen.rm == rm) & (scen.crop == c)).any()]


@functools.lru_cache(None)
def _sampler(rm):
    """Scenario anomaly matrix S (44 x K), residual matrix R (keys x K), trends, 2021 replay."""
    scen, resid, f27 = _fc()
    crops = rm_crops(rm)
    d = scen[scen.rm == rm]
    S = d.pivot(index="scenario_year", columns="crop", values="anomaly_gbm")[crops]
    t27 = d.groupby("crop").trend_2027.first()[crops].values
    slope = d.groupby("crop").trend_slope.first()[crops].values
    R = resid.pivot_table(index=["rm", "year"], columns="crop", values="resid_gbm")[crops]
    rng = np.random.default_rng(1000 + rm)
    Rv = R.values.copy()
    for j, c in enumerate(crops):
        pool = Rv[~np.isnan(Rv[:, j]), j]
        miss = np.isnan(Rv[:, j])
        Rv[miss, j] = rng.choice(pool, miss.sum())
    fr = f27[f27.rm == rm].set_index("crop").loc[crops]
    rep = (fr.replay_2021 / fr.trend_2027 - 1).values
    rep_model = (fr.yield_bad_2021 / fr.trend_2027 - 1).values
    rep_src = np.where(np.isnan(rep), "model (no 2021 yield reported)", "replay")
    rep = np.where(np.isnan(rep), rep_model, rep)
    return dict(crops=crops, S=S.values, years=S.index.values, R=Rv, t27=t27, slope=slope,
                rep2021=rep, rep2021_model=rep_model, rep_src=rep_src, fr=fr)


@functools.lru_cache(None)
def joint_yields(rm, year=2027, n=N_SAMPLES):
    """n x K joint yield samples (bu/ac or lb/ac). Same (s, k) for every crop in a draw."""
    sp = _sampler(rm)
    rng = np.random.default_rng(rm * 10 + (year - 2027))
    s = rng.integers(0, sp["S"].shape[0], n)
    k = rng.integers(0, sp["R"].shape[0], n)
    a = np.clip(sp["S"][s] + sp["R"][k], -1, 1)
    trend = sp["t27"] + sp["slope"] * (year - 2027)
    return trend * (1 + a), s


# ------------------------------------------------------------------ rotation rules
def violations(seq):
    """Hard-rule violations in a crop sequence (oldest first; None = unknown)."""
    v = []
    for i, c in enumerate(seq):
        if c is None:
            continue
        prev = [seq[j] for j in range(max(0, i - 2), i)]
        if c in ("Canola", "Lentils") and c in prev:
            v.append(f"{c}: needs a 2-year break")
        if c == "Flax" and c in seq[max(0, i - 3):i]:   # Flax Council of Canada: at least 3 years between flax crops
            v.append("Flax: needs a 3-year break")
        p = seq[i - 1] if i > 0 else None
        if p is None:
            continue
        if p in PULSES and c in PULSES and p != c:
            v.append(f"{c} after {p}: pulse after pulse")
        if p == c and c not in ("Canola", "Lentils") and not USE_MASC:
            v.append(f"{c} after {c}: same crop back-to-back")
    return sorted(set(v))


USE_MASC = True   # measured yield-by-previous-crop (MASC 2011-20); False = old assumed effects
_MASC_NAME = {"Spring Wheat": "Spring Wheat", "Durum": "Spring Wheat", "Barley": "Barley", "Oats": "Oat",
              "Canola": "Canola", "Flax": "Flax", "Peas": "Pea", "Lentils": "Pea"}


@functools.lru_cache(None)
def _masc():
    return pd.read_csv(ECON / "masc_stubble_response.csv", index_col=0)


MASC_SHRINK_K = 500   # fields; a cell based on 500 fields keeps half its effect


@functools.lru_cache(None)
def _masc_freq():
    return pd.read_csv(ECON / "masc_stubble_frequency.csv", index_col=0)


def _masc_n(r, c):
    f = _masc_freq()
    return f.loc[r, c] / 100 * f.loc["n_fields", c]


def yield_mult(prev, crop):
    if USE_MASC:
        if prev is None:
            return 1.0
        t = _masc()
        r, c = _MASC_NAME.get(prev), _MASC_NAME.get(crop)
        v = t.loc[r, c] if (r in t.index and c in t.columns) else np.nan
        if pd.isna(v):
            return 1.0
        n = _masc_n(r, c)   # fields behind this cell; shrink small samples toward 100%
        return 1.0 + (float(v) / 100 - 1.0) * n / (n + MASC_SHRINK_K)
    m = 1.0
    if prev in PULSES and crop in CEREALS:
        m += CEREAL_AFTER_PULSE
    if prev == "Canola" and crop in ("Spring Wheat", "Durum"):
        m += WHEAT_AFTER_CANOLA
    if prev == "Canola" and crop == "Canola":
        m += CANOLA_ON_CANOLA
    return m


def n_credit(rm, prev, crop):
    if prev in PULSES and crop not in PULSES:
        return N_CREDIT_LB[zone(rm)] * N_PRICE
    return 0.0


def margin(rm, crop, prev, y, effects=True, total=False):
    e = econ(rm, crop)
    mult = yield_mult(prev, crop) if effects else 1.0
    nc = n_credit(rm, prev, crop) if effects else 0.0
    return y * mult * e["price"] - e["var"] + nc - (e["fixed"] if total else 0.0)


def cvar(x, q=CVAR_Q, axis=-1):
    x = np.asarray(x)
    k = max(1, int(round(q * x.shape[axis])))
    return np.partition(x, k - 1, axis=axis).take(range(k), axis=axis).mean(axis=axis)


# ------------------------------------------------------------------ sequences
@functools.lru_cache(None)
def score_sequences(rm, history=(None, "Canola"), effects=True):
    """All rule-valid (2027, 2028) sequences after `history` (oldest first) with scores."""
    sp = _sampler(rm)
    crops = sp["crops"]
    Y1, _ = joint_yields(rm, 2027)
    Y2, _ = joint_yields(rm, 2028)
    t28 = sp["t27"] + sp["slope"]
    prev0 = history[-1]
    rows = []
    for i, c1 in enumerate(crops):
        for j, c2 in enumerate(crops):
            seq = list(history) + [c1, c2]
            if violations(seq):
                continue
            m1 = margin(rm, c1, prev0, Y1[:, i], effects)
            m2 = margin(rm, c2, c1, Y2[:, j], effects)
            tot = m1 + m2
            fixed = econ(rm, c1)["fixed"] + econ(rm, c2)["fixed"]
            r1 = margin(rm, c1, prev0, sp["t27"][i] * (1 + sp["rep2021"][i]), effects)
            r2 = margin(rm, c2, c1, t28[j] * (1 + sp["rep2021"][j]), effects)
            g1 = margin(rm, c1, prev0, sp["t27"][i] * (1 + sp["rep2021_model"][i]), effects)
            g2 = margin(rm, c2, c1, t28[j] * (1 + sp["rep2021_model"][j]), effects)
            b21_y1, b21_y2 = r1 + m2.mean(), m1.mean() + r2
            rows.append(dict(
                rm=rm, rm_name=rm_name(rm), soil_zone=zone(rm),
                history="-".join(h or "?" for h in history), crop_2027=c1, crop_2028=c2,
                sequence=f"{c1} -> {c2}",
                exp_margin_2yr=tot.mean(), cvar10_2yr=cvar(tot), p_loss_2yr=(tot < 0).mean(),
                p10_2yr=np.quantile(tot, 0.1), p50_2yr=np.median(tot),
                exp_margin_2027=m1.mean(), exp_margin_2028=m2.mean(),
                cvar10_2027=cvar(m1), cvar10_2028=cvar(m2),
                bad2021_margin_2yr=min(b21_y1, b21_y2),
                bad2021_hits=("2027" if b21_y1 <= b21_y2 else "2028"),
                bad2021_model_margin_2yr=min(g1 + m2.mean(), m1.mean() + g2),
                bad2021_source=("/".join(sorted({sp["rep_src"][i], sp["rep_src"][j]}))),
                exp_margin_total_cost_2yr=tot.mean() - fixed, cvar10_total_cost_2yr=cvar(tot) - fixed,
                p_loss_total_cost_2yr=(tot < fixed).mean(),
                n_credit_2028=n_credit(rm, c1, c2) if effects else 0.0,
                yield_mult_2028=yield_mult(c1, c2) if effects else 1.0,
                yield_mult_2027=yield_mult(prev0, c1) if effects else 1.0))
    d = pd.DataFrame(rows)
    d["rank_expected"] = d.exp_margin_2yr.rank(ascending=False, method="min").astype(int)
    d["rank_cvar"] = d.cvar10_2yr.rank(ascending=False, method="min").astype(int)
    d["rank_bad2021"] = d.bad2021_margin_2yr.rank(ascending=False, method="min").astype(int)
    d["n_sequences"] = len(d)
    for p, w in PROFILES.items():
        d[f"score_{p}"] = (1 - w) * d.exp_margin_2yr + w * d.cvar10_2yr
    return d


def _money(x):
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def _reason(r, d, profile):
    """Plain-language reason built only from the numbers in d (checked in check_reasons)."""
    n = len(d)
    best_e = d.loc[d.exp_margin_2yr.idxmax()]
    best_c = d.loc[d.cvar10_2yr.idxmax()]
    parts = [f"{r.sequence}: expected {_money(r.exp_margin_2yr)}/ac over 2 years (#{r.rank_expected} of {n}), "
             f"worst-10% average {_money(r.cvar10_2yr)} (#{r.rank_cvar}), "
             f"P(2-yr loss) {r.p_loss_2yr:.0%}, if 2021 repeats {_money(r.bad2021_margin_2yr)} (#{r.rank_bad2021})."]
    if r.sequence != best_e.sequence:
        de, dc = best_e.exp_margin_2yr - r.exp_margin_2yr, r.cvar10_2yr - best_e.cvar10_2yr
        parts.append(f"vs top-profit {best_e.sequence}: {_money(de)} lower expected, bad-tail average "
                     f"{_money(abs(dc))} {'better' if dc >= 0 else 'worse'}.")
    elif r.sequence != best_c.sequence:
        dc = best_c.cvar10_2yr - r.cvar10_2yr
        parts.append(f"Highest expected margin; the safest option {best_c.sequence} has a "
                     f"{_money(dc)} better bad-tail average.")
    why = []
    if r.n_credit_2028 > 0 or r.yield_mult_2028 > 1.0:
        bits = []
        if r.yield_mult_2028 > 1:
            bits.append(f"+{(r.yield_mult_2028 - 1):.0%} {r.crop_2028} yield (MASC, measured)")
        if r.n_credit_2028 > 0:
            bits.append(f"{_money(r.n_credit_2028)}/ac N credit")
        why.append(f"{r.crop_2027} first gives {' and '.join(bits)} in 2028")
    if r.yield_mult_2027 > 1.0:
        why.append(f"{r.crop_2027} on {r.history.split('-')[-1].lower()} stubble +{r.yield_mult_2027 - 1:.0%} yield")
    one = d.groupby("crop_2027").cvar10_2027.max()
    if one.idxmax() == r.crop_2027:
        why.append(f"{r.crop_2027} has the best single-year bad-tail margin here ({_money(one.max())})")
    why.append(f"{r.crop_2027} {_money(r.exp_margin_2027)} in 2027, {r.crop_2028} {_money(r.exp_margin_2028)} in 2028 on average")
    parts.append("Why: " + "; ".join(why) + ".")
    return " ".join(parts)


# risk level of a 2-year sequence (stated thresholds, $/ac over the two years)
RISK_LOW = dict(p_loss=0.05, cvar=-25.0)     # low: P(2-yr loss) <= 5% AND worst-10% average >= -$25
RISK_HIGH = dict(p_loss=0.15, cvar=-100.0)   # high: P(2-yr loss) > 15% OR worst-10% average < -$100
NICE = {"Spring Wheat": "Wheat", "Lentils": "Lentil", "Peas": "Pea"}


def nice(c):
    return NICE.get(c, c)


def risk_level(p_loss, cv):
    if p_loss > RISK_HIGH["p_loss"] or cv < RISK_HIGH["cvar"]:
        return "high"
    if p_loss <= RISK_LOW["p_loss"] and cv >= RISK_LOW["cvar"]:
        return "low"
    return "medium"


def bad_year_words(r, n):
    k = r.rank_bad2021
    if k == 1:
        w = "holds up best of all allowed rotations"
    elif k <= max(2, n / 3):
        w = "holds up well (top third of options)"
    elif k > 2 * n / 3:
        w = "among the biggest drops"
    else:
        w = "middle of the pack"
    if r.bad2021_margin_2yr < 0:
        w += "; the two years would lose money"
    return w


def _plain_reason(r, d):
    """Recommendation-first reason in words (no yields or dollars)."""
    n = len(d)
    bits = []
    if r.rank_expected == 1 and r.rank_cvar == 1:
        bits.append("most profitable allowed rotation on average and also the best protected in the worst years")
    elif r.rank_expected == 1:
        bits.append("most profitable allowed rotation on average, but not the safest in bad years")
    elif r.rank_cvar == 1:
        bits.append("best protected in the worst years, giving up some average profit")
    else:
        e = "high" if r.rank_expected <= n / 4 else "moderate"
        c = "good" if r.rank_cvar <= n / 4 else "moderate"
        bits.append(f"{e} average profit with {c} protection in the worst years")
    if r.crop_2027 in PULSES and r.crop_2028 not in PULSES:
        extra = " and a measured yield boost (MASC)" if r.yield_mult_2028 > 1 else ""
        bits.append(f"{nice(r.crop_2027).lower()} leaves nitrogen for the {nice(r.crop_2028).lower()} crop{extra}")
    if r.yield_mult_2027 > 1:
        bits.append(f"{nice(r.crop_2027).lower()} does a little better on {nice(r.history.split('-')[-1]).lower()} stubble")
    one = d.groupby("crop_2027").cvar10_2027.max()
    if one.idxmax() == r.crop_2027:
        bits.append(f"{nice(r.crop_2027).lower()} has the smallest bad-year hit of any single crop here")
    best1 = d.groupby("crop_2027").exp_margin_2027.max()
    if best1.idxmax() == r.crop_2027:
        bits.append(f"{nice(r.crop_2027).lower()} earns the most per acre in 2027 at 2026 guide prices")
    s = "; ".join(bits)
    return s[0].upper() + s[1:] + "."


@functools.lru_cache(None)
def _mix_cached(rm, previous_crop):
    return crop_mix(rm, previous_crop)


@functools.lru_cache(None)
def _ins_cached(rm, crop, previous_crop):
    return insurance_table(rm, crop, previous_crop)


def loss_control(rm, previous_crop, crop_2027, profile):
    """Action list (words): crop mix, insurance level, alerts to watch."""
    acts = []
    try:
        s = _mix_cached(rm, previous_crop)["summary"].set_index("option")
        o = s.loc["max expected s.t. CVaR10 >= best single (25% steps)"]
        if o.mix.startswith("100%"):
            o2 = s.loc["max CVaR10 s.t. expected >= 95% of best single (25% steps)"]
            if not o2.mix.startswith("100%"):
                acts.append(f"Crop mix: split 2027 acres {o2.mix} to soften a bad year for a small cost in average profit")
            else:
                o3 = s.loc["max CVaR10 (25% steps)"]
                acts.append(f"Crop mix: {o.mix.split('% ')[1].lower()} alone is best here on average and in bad years; "
                            f"a split such as {o3.mix} softens bad years further but costs clearly more than 5% of average profit")
        else:
            acts.append(f"Crop mix: split 2027 acres {o.mix}; nearly the same average profit with a smaller bad-year hit than any single crop")
    except Exception:
        pass
    it = _ins_cached(rm, crop_2027, previous_crop)
    w = PROFILES[profile]
    sc = (1 - w) * it.exp_margin + w * it.cvar10
    lvl = it.coverage[sc.idxmax()]
    if lvl == 0:
        acts.append(f"Insurance: for pure profit, cover is optional for {nice(crop_2027).lower()} in this model")
    else:
        acts.append(f"Insurance: take {lvl:.0%} SCIC coverage on {nice(crop_2027).lower()}; it raises the bad-year floor most "
                    "(approximate model, confirm premiums with SCIC by March 31)")
    a = alerts(rm).set_index("trigger_id")
    watch = []
    for tid, txt in (("heat_flowering", "more than 15 days above 30 C at flowering (Jun 15-Jul 31)"),
                     ("dry_spring", "May-June rain under about 72 mm"),
                     ("very_wet_spring", "May-June rain over about 174 mm (seeding, root rot)")):
        if tid in a.index and a.loc[tid, "years_fired"] > 0:
            watch.append(f"{txt}, seen {int(a.loc[tid, 'years_fired'])} of {int(a.loc[tid, 'n_rm_years'])} seasons here")
    if watch:
        acts.append("Alerts to watch: " + "; ".join(watch))
    return " | ".join(acts)


def recommend(rm, previous_crop="Canola", profile="balanced", n_top=3, history=None, effects=True):
    """Top-n 2-year sequences for an RM after `previous_crop` (or a full `history` tuple).
    Recommendation-first columns come first; yield/margin numbers stay in the detail columns."""
    hist = tuple(history) if history is not None else (None, previous_crop)
    d = score_sequences(rm, hist, effects)
    top = d.sort_values([f"score_{profile}", "exp_margin_2yr"], ascending=False).head(n_top).copy()
    top["score"] = top[f"score_{profile}"]
    top["details_reason"] = [_reason(r, d, profile) for r in top.itertuples()]
    front = pd.DataFrame(dict(
        profile=profile, rank=range(1, len(top) + 1),
        rotation=[f"{nice(a)} \u2192 {nice(b)}" for a, b in zip(top.crop_2027, top.crop_2028)],
        risk_level=[risk_level(p, c) for p, c in zip(top.p_loss_2yr, top.cvar10_2yr)],
        bad_year_like_2021=[bad_year_words(r, len(d)) for r in top.itertuples()],
        reason=[_plain_reason(r, d) for r in top.itertuples()],
        loss_control=[loss_control(rm, hist[-1], c, profile) for c in top.crop_2027]), index=top.index)
    top = pd.concat([front, top], axis=1)
    return top.drop(columns=[c for c in top.columns if c.startswith("score_")]).reset_index(drop=True)


def check_reasons(rec, d):
    """Re-derive every number/word in the reasons from the score table."""
    for r in rec.itertuples():
        row = d[d.sequence == r.sequence].iloc[0]
        assert _money(row.exp_margin_2yr) in r.details_reason and f"#{row.rank_expected} of" in r.details_reason
        assert _money(row.cvar10_2yr) in r.details_reason and f"#{row.rank_cvar})" in r.details_reason
        assert r.risk_level == risk_level(row.p_loss_2yr, row.cvar10_2yr)
        if "most profitable" in r.reason:
            assert row.rank_expected == 1
        if "best protected" in r.reason:
            assert row.rank_cvar == 1
        if "holds up best" in r.bad_year_like_2021:
            assert row.rank_bad2021 == 1
        if "biggest drops" in r.bad_year_like_2021:
            assert row.rank_bad2021 > 2 * len(d) / 3
        best_e = d.loc[d.exp_margin_2yr.idxmax()]
        if r.sequence != best_e.sequence:
            word = "better" if row.cvar10_2yr >= best_e.cvar10_2yr else "worse"
            assert word in r.details_reason.split("Why:")[0]
            assert row.exp_margin_2yr <= best_e.exp_margin_2yr
        hist = [None if h == "?" else h for h in r.history.split("-")]
        assert not violations(hist + [r.crop_2027, r.crop_2028])
    return True


# ------------------------------------------------------------------ crop mix
def _grid(k, step, max_crops):
    u = int(round(1 / step))
    out = []
    for combo in itertools.combinations_with_replacement(range(k), u):
        w = np.bincount(combo, minlength=k)
        if (w > 0).sum() <= max_crops:
            out.append(w / u)
    return np.unique(np.array(out), axis=0)


def crop_mix(rm, previous_crop="Canola", step=0.25, max_crops=3, frontier_step=0.10):
    """2027 acreage split. Returns dict(summary=DataFrame, frontier=DataFrame, points=DataFrame)."""
    sp = _sampler(rm)
    hist = [None, previous_crop]
    crops = [c for c in sp["crops"] if not violations(hist + [c])]
    idx = [sp["crops"].index(c) for c in crops]
    Y, _ = joint_yields(rm, 2027)
    M = np.column_stack([margin(rm, c, previous_crop, Y[:, i]) for c, i in zip(crops, idx)])
    rep = np.array([margin(rm, c, previous_crop, sp["t27"][i] * (1 + sp["rep2021"][i])) for c, i in zip(crops, idx)])

    def evaluate(W):
        E = W @ M.mean(0)
        C = np.empty(len(W)); PL = np.empty(len(W))
        for a in range(0, len(W), 500):
            X = (W[a:a + 500] @ M.T).astype(np.float32)
            C[a:a + 500] = cvar(X, axis=1)
            PL[a:a + 500] = (X < 0).mean(1)
        return pd.DataFrame(dict(exp_margin=E, cvar10=C, p_loss=PL, bad2021_margin=W @ rep))

    def label(w):
        return " / ".join(f"{int(round(x * 100))}% {c}" for c, x in sorted(zip(crops, w), key=lambda t: -t[1]) if x > 0)

    pts = {}
    for name, st in (("coarse", step), ("fine", frontier_step)):
        W = _grid(len(crops), st, max_crops)
        ev = evaluate(W)
        ev["mix"] = [label(w) for w in W]
        ev[crops] = W
        ev["n_crops"] = (W > 0).sum(1)
        pts[name] = ev
    singles = pts["coarse"][pts["coarse"].n_crops == 1]
    best_e = singles.loc[singles.exp_margin.idxmax()]
    best_c = singles.loc[singles.cvar10.idxmax()]
    rows = [("best single crop (expected)", best_e), ("best single crop (CVaR10)", best_c)]
    for name, lab in (("coarse", f"{int(step * 100)}% steps"), ("fine", f"{int(frontier_step * 100)}% steps")):
        ev = pts[name]
        ok = ev[ev.cvar10 >= best_c.cvar10 - 1e-9]
        rows.append((f"max expected s.t. CVaR10 >= best single ({lab})", ok.loc[ok.exp_margin.idxmax()]))
        rows.append((f"max CVaR10 ({lab})", ev.loc[ev.cvar10.idxmax()]))
        ok = ev[ev.exp_margin >= 0.95 * best_e.exp_margin]
        rows.append((f"max CVaR10 s.t. expected >= 95% of best single ({lab})", ok.loc[ok.cvar10.idxmax()]))
    summ = pd.DataFrame([dict(rm=rm, rm_name=rm_name(rm), previous_crop=previous_crop, option=o) |
                         {k: r[k] for k in ["mix", "exp_margin", "cvar10", "p_loss", "bad2021_margin"]} |
                         {f"share_{c}": r[c] for c in crops} for o, r in rows])
    ev = pts["fine"].sort_values(["cvar10", "exp_margin"], ascending=[False, False])
    fr = ev[ev.exp_margin >= ev.exp_margin.cummax() - 1e-9].copy()
    fr = fr[fr.exp_margin > fr.exp_margin.shift(1).fillna(-np.inf)]
    fr.insert(0, "rm", rm)
    return dict(summary=summ, frontier=fr, points=pts["fine"], crops=crops)


# ------------------------------------------------------------------ insurance
def insurance_table(rm, crop, previous_crop=None, levels=(0.0, 0.5, 0.6, 0.7, 0.8)):
    """Approximate SCIC-style yield-loss cover for 2027. Probable yield = 2027 RM trend.
    Payout = max(0, cov x probable - actual) x guide price. Premium = guide premium (assumed to be
    at 70% coverage) scaled by the model's expected loss at each level relative to 70%."""
    sp = _sampler(rm)
    i = sp["crops"].index(crop)
    e = econ(rm, crop)
    Y, _ = joint_yields(rm, 2027)
    mult = yield_mult(previous_crop, crop)
    y = Y[:, i] * mult
    T = sp["t27"][i]
    y21 = T * (1 + sp["rep2021"][i]) * mult
    base = margin(rm, crop, previous_crop, Y[:, i]) + e["premium"]     # guide var cost includes the premium
    base21 = margin(rm, crop, previous_crop, T * (1 + sp["rep2021"][i])) + e["premium"]
    el = {c: (np.maximum(0, c * T - y) * e["price"]).mean() for c in levels}
    rows = []
    for c in levels:
        pay = np.maximum(0, c * T - y) * e["price"]
        prem = 0.0 if c == 0 else e["premium"] * el[c] / el[INS_REF_COVERAGE]
        m = base - prem + pay
        rows.append(dict(rm=rm, crop=crop, unit=UNITS[crop], coverage=c, probable_yield=T,
                         guarantee_yield=c * T, premium=prem, exp_payout=pay.mean(),
                         p_payout=(pay > 0).mean(), payout_to_premium=(pay.mean() / prem if prem else np.nan),
                         exp_margin=m.mean(), cvar10=cvar(m), p_loss=(m < 0).mean(),
                         bad2021_margin=base21 - prem + max(0, c * T - y21) * e["price"]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ alerts
TRIGGERS = [
    # id, label, test, when known, meaning, action
    ("dry_spring", "May-Jun rain <= 72 mm",
     lambda f: f.rain_may_jun <= 72.2, "July 1",
     "Very dry spring. Tree: about -20% if July then turns hot (> 26.9 C average max)",
     "Scout for moisture stress; hold back in-crop N top-ups and fungicide unless the crop is clean and moisture improves; do not forward-sell beyond insured bushels"),
    ("dry_spring_hot_july", "May-Jun rain <= 72 mm AND July avg max > 26.9 C",
     lambda f: (f.rain_may_jun <= 72.2) & (f.tmax_jul > 26.9) & (f.hot_days_flower <= 15.5), "July 31",
     "Tree leaf: about -20% vs trend",
     "Expect a below-trend crop: review the SCIC claim process, stop extra inputs, price only the insured share"),
    ("heat_flowering", "Days > 30 C at flowering (Jun 15-Jul 31) > 15",
     lambda f: f.hot_days_flower > 15.5, "July 31 (count builds daily)",
     "Heat blast at flowering. Median about -20% (results/README.md)",
     "Scout pod/head fill (canola blanks, pulse flower abortion); skip late fungicide on a heat-damaged crop; review marketing and insurance"),
    ("heat_dry", "Heat at flowering > 15 days AND May-Jun rain <= 103 mm",
     lambda f: (f.hot_days_flower > 15.5) & (f.rain_may_jun <= 103.1), "July 31",
     "Tree leaf: about -30% (-50% if summer root soil also very dry)",
     "Likely loss year: notify SCIC early if yield will fall below the guarantee, consider feed/salvage options, keep stored grain unpriced decisions under review"),
    ("heat_dry_soil", "... AND lowest summer root soil wetness <= 0.345",
     lambda f: (f.hot_days_flower > 15.5) & (f.rain_may_jun <= 103.1) & (f.wet_root_min <= 0.345), "August",
     "Tree's worst leaf: about -50%",
     "Treat as a crop-failure year: claim, cash-flow plan, decide early on cutting for feed/greenfeed"),
    ("very_wet_spring", "May-Jun rain > 174 mm",
     lambda f: f.rain_may_jun > 174.3, "July 1",
     "Very wet spring (2010-type). Per-acre yield near trend in the tree, but unseeded acres and disease risk",
     "File SCIC unseeded-acreage claim before the deadline; scout for root rot (peas/lentils) and leaf disease; fungicide decisions based on scouting"),
]


@functools.lru_cache(None)
def alerts_table():
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_models import anomalies, load
    df = load()
    df["anom"] = anomalies(df, -1)          # trend on all years (descriptive only)
    f = pd.read_csv(PROC / "features.csv")
    yr = df.groupby(["rm", "year"]).agg(anom=("anom", "mean")).reset_index().merge(f, on=["rm", "year"])
    cases = df[["rm", "year", "crop", "anom"]].merge(f, on=["rm", "year"])
    rows = []
    for tid, lab, test, when, meaning, action in TRIGGERS:
        for rm in [None] + sorted(yr.rm.unique()):
            y = yr if rm is None else yr[yr.rm == rm]
            c = cases if rm is None else cases[cases.rm == rm]
            fy, fc = test(y), test(c)
            rows.append(dict(rm="all" if rm is None else rm, trigger_id=tid, trigger=lab, known_by=when,
                             meaning=meaning, suggested_action=action,
                             years_fired=int(fy.sum()), n_rm_years=len(y), freq=fy.mean(),
                             years_list=", ".join(map(str, y.year[fy].tolist())) if rm is not None else "",
                             mean_anom_fired=c.anom[fc].mean(), mean_anom_not_fired=c.anom[~fc].mean(),
                             fail_rate_fired=(c.anom[fc] < -0.2).mean(), fail_rate_not_fired=(c.anom[~fc] < -0.2).mean(),
                             n_cases_fired=int(fc.sum())))
    return pd.DataFrame(rows)


def alerts(rm):
    a = alerts_table()
    return a[a.rm.astype(str) == str(rm)].reset_index(drop=True)


# ------------------------------------------------------------------ backtest
def _bt_data():
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_models import load
    y = load()[["rm", "crop", "year", "yield"]]
    return y.pivot_table(index=["rm", "year"], columns="crop", values="yield")


def _past_dist(Yp, rm, t, crops, rng):
    """Past-only climatology of anomalies: per crop, linear trend on years < t; joint anomaly
    vectors by past year (correlation kept). Returns crops_ok, A (years x K), trend_t, trend_t1."""
    g = Yp.loc[rm]
    past = g[g.index < t]
    ok, tr0, tr1, cols = [], [], [], []
    for c in crops:
        s = past[c].dropna()
        if len(s) < BT_MIN_YEARS:
            continue
        b, a = np.polyfit(s.index, s.values, 1)
        fit = np.maximum(a + b * past.index.values, 1e-6)
        cols.append(((past[c].values - fit) / fit).clip(-1, 1))
        ok.append(c); tr0.append(max(a + b * t, 1e-6)); tr1.append(max(a + b * (t + 1), 1e-6))
    A = np.column_stack(cols)
    for j in range(A.shape[1]):
        m = np.isnan(A[:, j])
        A[m, j] = rng.choice(A[~m, j], m.sum())
    return ok, A, np.array(tr0), np.array(tr1)


def _bt_pick(rm, hist, ok, A, tr0, tr1, profile, effects):
    n = A.shape[0]
    best, best_s = None, -np.inf
    for i, c1 in enumerate(ok):
        for j, c2 in enumerate(ok):
            if violations(list(hist) + [c1, c2]):
                continue
            m1 = margin(rm, c1, hist[-1], tr0[i] * (1 + A[:, i]), effects)
            m2 = margin(rm, c2, c1, tr1[j] * (1 + A[:, j]), effects)
            tot = (m1[:, None] + m2[None, :]).ravel()
            w = PROFILES[profile]
            s = (1 - w) * tot.mean() + w * cvar(tot)
            if s > best_s:
                best, best_s = (c1, c2), s
    return best


def backtest(effects=True, n_random=100):
    Yp = _bt_data()
    rms = sorted(Yp.index.get_level_values(0).unique())
    adj = {}   # trend-adjust factor to 2025 level (scoring view only, full-sample trend)
    for rm in rms:
        g = Yp.loc[rm]
        for c in CROP_MAP:
            s = g[c].dropna() if c in g else pd.Series(dtype=float)
            if len(s) >= 10:
                b, a = np.polyfit(s.index, s.values, 1)
                for yy in g.index:
                    adj[(rm, c, yy)] = max(a + b * 2025, 1e-6) / max(a + b * yy, 1e-6)
    blocks = list(range(BT_START, BT_END, 2))
    rows = []

    def realized(rm, c, prev, yy):
        y = Yp.loc[(rm, yy)].get(c, np.nan)
        m = margin(rm, c, prev, y, effects)
        m_adj = margin(rm, c, prev, y * adj.get((rm, c, yy), np.nan), effects)
        return y, m, m_adj

    for rm in rms:
        crops = [c for c in CROP_MAP if c in Yp.columns]
        crops = [c for c in crops if Yp.loc[rm][c].notna().sum() >= 10 and _has_econ(rm, c)]
        paths = {p: [None, "Canola"] for p in PROFILES} | {"canola_wheat": [None, "Canola"], "guide_top": [None, "Canola"]}
        rpaths = [[None, "Canola"] for _ in range(n_random)]
        rng = np.random.default_rng(rm)
        for t in blocks:
            avail = [c for c in crops if not np.isnan(Yp.loc[(rm, t)].get(c, np.nan))
                     and not np.isnan(Yp.loc[(rm, t + 1)].get(c, np.nan))]
            ok, A, tr0, tr1 = _past_dist(Yp, rm, t, avail, rng)
            picks = {p: _bt_pick(rm, tuple(paths[p][-2:]), ok, A, tr0, tr1, p, effects) for p in PROFILES}
            picks["canola_wheat"] = ("Spring Wheat", "Canola")
            # guide's highest return-over-variable crop allowed by the rules, year by year
            h = list(paths["guide_top"][-2:])
            gt = []
            for _ in range(2):
                cand = sorted([c for c in avail if not violations(h[-2:] + [c])],
                              key=lambda c: -econ(rm, c)["guide_return"])
                gt.append(cand[0]); h.append(cand[0])
            picks["guide_top"] = tuple(gt)
            for s, (c1, c2) in picks.items():
                prev = paths[s][-1]
                for k, (c, p, yy) in enumerate(((c1, prev, t), (c2, c1, t + 1))):
                    y, m, ma = realized(rm, c, p, yy)
                    rows.append(dict(rm=rm, block=t, year=yy, strategy=s, crop=c, prev=p, yield_=y,
                                     margin=m, margin_trend_adj=ma,
                                     rule_violation="; ".join(violations(paths[s][-2:] + ([c1] if k else []) + [c]))))
                paths[s] += [c1, c2]
            # random allowed rotations
            for rp in rpaths:
                seqs = [(a, b) for a in avail for b in avail if not violations(rp[-2:] + [a, b])]
                c1, c2 = seqs[rng.integers(len(seqs))]
                for k, (c, p, yy) in enumerate(((c1, rp[-1], t), (c2, c1, t + 1))):
                    y, m, ma = realized(rm, c, p, yy)
                    rows.append(dict(rm=rm, block=t, year=yy, strategy="random", crop=c, prev=p, yield_=y,
                                     margin=m, margin_trend_adj=ma, rule_violation=""))
                rp += [c1, c2]
    bt = pd.DataFrame(rows).rename(columns={"yield_": "yield"})
    bt["effects"] = effects
    return bt


def _has_econ(rm, c):
    try:
        econ(rm, c)
        return True
    except ValueError:
        return False


def summarize_backtest(bt):
    # random: average over paths first
    g = bt.groupby(["strategy", "rm", "block", "year"])[["margin", "margin_trend_adj"]].mean().reset_index()
    blk = g.groupby(["strategy", "rm", "block"])[["margin", "margin_trend_adj"]].sum().reset_index()
    ref = blk[blk.strategy == "canola_wheat"].set_index(["rm", "block"])
    refg = blk[blk.strategy == "guide_top"].set_index(["rm", "block"])
    rows = []
    order = list(PROFILES) + ["canola_wheat", "guide_top", "random"]
    for s in order:
        a = g[g.strategy == s]
        b = blk[blk.strategy == s].set_index(["rm", "block"])
        yr = a.groupby("year")[["margin", "margin_trend_adj"]].mean()
        diff = (b.margin - ref.margin).dropna()
        diffa = (b.margin_trend_adj - ref.margin_trend_adj).dropna()
        rows.append(dict(strategy=s, n_rm_years=len(a), avg_margin_per_year=a.margin.mean(),
                         avg_margin_trend_adj=a.margin_trend_adj.mean(),
                         worst_rm_year=a.margin.min(), worst_rm_year_trend_adj=a.margin_trend_adj.min(),
                         worst_year_avg_over_rms=yr.margin.min(), worst_year=int(yr.margin.idxmin()),
                         worst_year_avg_trend_adj=yr.margin_trend_adj.min(),
                         p10_rm_year_trend_adj=a.margin_trend_adj.quantile(0.1),
                         share_rm_years_loss=(a.margin < 0).mean(),
                         share_rm_years_loss_trend_adj=(a.margin_trend_adj < 0).mean(),
                         margin_2021=a[a.year == 2021].margin.mean(),
                         beat_canola_wheat_blocks=(diff > 0).mean() if s != "canola_wheat" else np.nan,
                         beat_canola_wheat_blocks_trend_adj=(diffa > 0).mean() if s != "canola_wheat" else np.nan,
                         mean_gain_vs_canola_wheat_2yr=diff.mean() if s != "canola_wheat" else np.nan,
                         beat_guide_top_blocks_trend_adj=((b.margin_trend_adj - refg.margin_trend_adj).dropna() > 0).mean()
                         if s != "guide_top" else np.nan,
                         tie_guide_top_blocks=((b.margin_trend_adj - refg.margin_trend_adj).dropna().abs() < 1e-6).mean()
                         if s != "guide_top" else np.nan,
                         n_blocks=len(b)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ sanity checks
def sanity_checks(recs, bt):
    out = []
    for rm in sorted(_fc()[0].rm.unique()):
        sp = _sampler(rm)
        Y, _ = joint_yields(rm)
        out.append(("no negative yields in samples", rm, bool((Y >= 0).all())))
        for c in sp["crops"]:
            e = econ(rm, c)
            unit_ok = e["unit"] == UNITS[c]
            out.append((f"unit match {c} ({UNITS[c]} vs guide {e['unit']})", rm, unit_ok))
    out.append(("recommendations: no rule violations", "all",
                bool(all(not violations([None if h == "?" else h for h in r.history.split("-")] + [r.crop_2027, r.crop_2028])
                         for r in recs.itertuples()))))
    rec_bt = bt[bt.strategy.isin(list(PROFILES) + ["random", "guide_top"])]
    out.append(("backtest recommender/random/guide_top: no rule violations", "all", bool((rec_bt.rule_violation == "").all())))
    out.append(("backtest: realized yields non-negative", "all", bool((bt["yield"].dropna() >= 0).all())))
    out.append(("backtest: no missing realized yields except canola-wheat baseline (canola not reported)", "all",
                bool(bt[bt.strategy != "canola_wheat"]["yield"].notna().all())))
    return pd.DataFrame(out, columns=["check", "rm", "passed"])


def unit_table():
    rows = []
    for rm in sorted(_fc()[0].rm.unique()):
        sp = _sampler(rm)
        for i, c in enumerate(sp["crops"]):
            e = econ(rm, c)
            rows.append(dict(rm=rm, soil_zone=zone(rm), crop=c, unit=UNITS[c], guide_unit=e["unit"],
                             trend_2027=sp["t27"][i], guide_target_yield_p80=e["guide_yield"],
                             guide_avg_yield=e["guide_avg_yield"], ratio_trend_to_guide_avg=sp["t27"][i] / e["guide_avg_yield"]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ charts
def _style(plt):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "savefig.facecolor": SURF, "text.parse_math": False})


def charts(mix159, ins, bt_sum, rm=159):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    # 1. frontier
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    p = mix159["points"]
    ax.scatter(p.cvar10, p.exp_margin, s=6, color="#d6d5cf", label="all mixes (10% steps, up to 3 crops)")
    fr = mix159["frontier"]
    ax.plot(fr.cvar10, fr.exp_margin, color=BLUE, lw=2, label="efficient frontier")
    sing = p[p.n_crops == 1]
    for r in sing.itertuples():
        c = r.mix.split("% ")[1]
        ax.scatter(r.cvar10, r.exp_margin, s=70, color=CROP_COLOR[c], edgecolor="white", lw=1.5, zorder=3)
        ax.annotate(c, (r.cvar10, r.exp_margin), xytext=(6, -3), textcoords="offset points", fontsize=8, color=INK2)
    s = mix159["summary"]
    rec = s[s.option.str.startswith("max expected") & s.option.str.contains("25%")].iloc[0]
    ax.scatter(rec.cvar10, rec.exp_margin, marker="*", s=260, color=ORANGE, edgecolor="white", zorder=4,
               label=f"pick (25% steps): {rec.mix}")
    best_c = s[s.option == "best single crop (CVaR10)"].iloc[0]
    ax.axvline(best_c.cvar10, color=GRAY, ls=":", lw=1)
    ax.set_xlabel("CVaR10: average margin in the worst 10% of 2027 outcomes ($/ac)")
    ax.set_ylabel("Expected 2027 margin over variable cost ($/ac)")
    ax.set_title(f"RM {rm} {rm_name(rm)}, after canola: crop mix vs downside", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "chart_frontier_rm159.png", dpi=150); plt.close(fig)

    # 2. margin distribution per crop
    sp = _sampler(rm)
    Y, _ = joint_yields(rm)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    data, labs, cols = [], [], []
    order = sorted(sp["crops"], key=lambda c: -margin(rm, c, "Canola", Y[:, sp["crops"].index(c)]).mean())
    for c in order:
        i = sp["crops"].index(c)
        m = margin(rm, c, "Canola", Y[:, i])
        data.append(m); labs.append(c + (" (on canola: not allowed, -15%)" if c == "Canola" else "")); cols.append(CROP_COLOR[c])
    for k, (m, c) in enumerate(zip(data, cols)):
        q = np.quantile(m, [0.05, 0.1, 0.5, 0.9, 0.95])
        ax.plot([q[0], q[4]], [k, k], color=c, lw=2, alpha=0.5, solid_capstyle="round")
        ax.plot([q[1], q[3]], [k, k], color=c, lw=8, solid_capstyle="round")
        ax.plot(q[2], k, "o", ms=8, color="white", mec=c, mew=2)
        ax.plot(cvar(m), k, "|", ms=14, color=INK, mew=2)
        r21 = margin(rm, order[k], "Canola", sp["t27"][sp["crops"].index(order[k])] * (1 + sp["rep2021"][sp["crops"].index(order[k])]))
        ax.plot(r21, k, "v", ms=8, color=RED, mec="white")
        ax.text(ax.get_xlim()[1] if False else 520, k, f"E {_money(m.mean())}  CVaR10 {_money(cvar(m))}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=9); ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlim(right=760)
    ax.set_xlabel("2027 margin over variable cost ($/ac): bar P10-P90, line P5-P95, dot median")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="|", ls="", color=INK, ms=12, mew=2, label="CVaR10 (worst-10% average)"),
                       Line2D([], [], marker="v", ls="", color=RED, ms=8, label="2021 replay")],
              frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=2)
    ax.set_title(f"RM {rm} {rm_name(rm)}: 2027 margin per crop, field was canola in 2026", loc="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(OUT / "chart_margin_rm159.png", dpi=150); plt.close(fig)

    # 3. insurance
    sub = ins[ins.rm == rm]
    top = sub[sub.coverage == 0].sort_values("exp_margin", ascending=False).crop.head(4).tolist()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for c in top:
        d = sub[sub.crop == c]
        for ax, col in zip(axs, ["exp_margin", "cvar10"]):
            ax.plot(d.coverage * 100, d[col], "-o", color=CROP_COLOR[c], lw=2, ms=6, label=c)
    axs[0].set_title("Expected 2027 margin ($/ac)", loc="left", fontsize=10, color=INK)
    axs[1].set_title("CVaR10: worst-10% average ($/ac)", loc="left", fontsize=10, color=INK)
    for ax in axs:
        ax.set_xticks([0, 50, 60, 70, 80]); ax.set_xticklabels(["none", "50%", "60%", "70%", "80%"])
        ax.set_xlabel("Coverage level (share of probable yield)")
    axs[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"RM {rm} {rm_name(rm)}: effect of crop insurance (approximate SCIC-style, model premiums)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(OUT / "chart_insurance_rm159.png", dpi=150); plt.close(fig)

    # 4. backtest
    b = bt_sum.set_index("strategy")
    names = {"max_profit": "Max profit", "balanced": "Balanced", "steady_income": "Steady income",
             "canola_wheat": "Canola-wheat", "guide_top": "Guide's top crop", "random": "Random allowed"}
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, title in zip(axs, ["avg_margin_trend_adj", "p10_rm_year_trend_adj", "worst_year_avg_trend_adj"],
                              ["Average margin per RM-year", "P10 of RM-year margins", "Worst year (avg over 10 RMs)"]):
        vals = b[col].reindex(names)
        cols = [BLUE if k in PROFILES else GRAY for k in names]
        ax.barh(range(len(vals)), vals.values, color=cols, height=0.6)
        for i, v in enumerate(vals.values):
            ax.text(v + (4 if v >= 0 else -4), i, _money(v), va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK2)
        ax.set_yticks(range(len(vals))); ax.set_yticklabels(names.values() if ax is axs[0] else [""] * len(vals), fontsize=9)
        ax.invert_yaxis(); ax.axvline(0, color=INK, lw=1)
        lo, hi = min(0, vals.min()), max(0, vals.max())
        ax.set_xlim(lo - 0.25 * (hi - lo) * (lo < 0), hi + 0.25 * (hi - lo) * (hi > 0))
        ax.set_title(title + " ($/ac)", loc="left", fontsize=10, color=INK)
    fig.suptitle("Backtest 1992-2025, 10 RMs, past-only information, 2026 prices; yields trend-adjusted to 2025 level",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(OUT / "chart_backtest.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rms = sorted(_fc()[0].rm.unique())
    recs, allseq = [], []
    for rm in rms:
        for prev in ("Canola", "Spring Wheat"):
            d = score_sequences(rm, (None, prev))
            d.insert(3, "previous_crop", prev)
            allseq.append(d)
            for p in PROFILES:
                r = recommend(rm, prev, p)
                check_reasons(r, d)
                r.insert(2, "previous_crop", prev)
                recs.append(r)
    recs = pd.concat(recs, ignore_index=True)
    recs.to_csv(OUT / "recommendations.csv", index=False, float_format="%.2f")
    pd.concat(allseq).to_csv(OUT / "all_sequences.csv", index=False, float_format="%.2f")
    # sensitivity: no soft rotation effects
    noeff = pd.concat([recommend(rm, "Canola", p, effects=False).assign(previous_crop="Canola")
                       for rm in rms for p in PROFILES], ignore_index=True)
    noeff.to_csv(OUT / "recommendations_no_rotation_effects.csv", index=False, float_format="%.2f")

    mixes = {rm: crop_mix(rm, "Canola") for rm in rms}
    pd.concat([m["summary"] for m in mixes.values()]).to_csv(OUT / "crop_mix.csv", index=False, float_format="%.3f")
    pd.concat([m["frontier"] for m in mixes.values()]).to_csv(OUT / "crop_mix_frontier.csv", index=False, float_format="%.3f")

    ins = pd.concat([insurance_table(rm, c) for rm in rms for c in rm_crops(rm)], ignore_index=True)
    ins.to_csv(OUT / "insurance.csv", index=False, float_format="%.3f")

    alerts_table().to_csv(OUT / "alerts.csv", index=False, float_format="%.3f")

    bt = backtest(effects=True)
    bt0 = backtest(effects=False)
    bt_all = pd.concat([bt, bt0], ignore_index=True)
    bt_all.to_csv(OUT / "backtest.csv", index=False, float_format="%.2f")
    s1 = summarize_backtest(bt).assign(rotation_effects=True)
    s0 = summarize_backtest(bt0).assign(rotation_effects=False)
    pd.concat([s1, s0]).to_csv(OUT / "backtest_summary.csv", index=False, float_format="%.3f")

    sc = sanity_checks(recs, bt)
    sc.to_csv(OUT / "sanity_checks.csv", index=False)
    unit_table().to_csv(OUT / "units_check.csv", index=False, float_format="%.2f")
    charts(mixes[159], ins, s1)

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 400)
    print(recs[(recs.rm == 159)][["previous_crop", "profile", "rank", "sequence", "exp_margin_2yr", "cvar10_2yr",
                                  "p_loss_2yr", "bad2021_margin_2yr", "bad2021_model_margin_2yr"]])
    print(recs[(recs.rm == 159)][["previous_crop", "profile", "rank", "rotation", "risk_level", "bad_year_like_2021", "reason", "loss_control"]].to_string())
    print(recs[(recs.rm == 159) & (recs.previous_crop == "Canola")].details_reason.to_string())
    print(mixes[159]["summary"][["option", "mix", "exp_margin", "cvar10", "p_loss", "bad2021_margin"]])
    print(ins[ins.rm == 159].round(1).to_string())
    print(s1.round(1).T); print(s0.round(1).T)
    print(sc[~sc.passed])


if __name__ == "__main__":
    main()
