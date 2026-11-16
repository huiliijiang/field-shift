"""Check that the static demo (02_demo/canada/canada-core.mjs) selects the same tied group, standouts and usual-plan
numbers as the Python selection layer (app/data.py::options) for a spread of inputs. Needs node. Run after export_demo.py."""
import itertools, json, pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import data as D  # noqa: E402
DEMO = ROOT.parents[1] / "02_demo" / "canada"
PROF = {"Best average": "best_average", "Balanced": "balanced", "Most reliable": "most_reliable"}
JS = """
import {compute} from '%s/canada-core.mjs'; import fs from 'fs';
const data = JSON.parse(fs.readFileSync('%s/data.json'));
const cases = JSON.parse(fs.readFileSync(0)); const out = [];
for (const s of cases) { const r = compute(data, s);
  out.push({n: r.allowed.length, group: r.group.map(x => x.c1 + '>' + x.c2).sort(), picks: r.picks.map(p => p.r.c1 + '>' + p.r.c2 + ':' + p.labels.join(' · ')),
            usual: r.usual ? [r.usual.avg, r.usual.k21, r.usual.m, r.usual.rule] : null, usualIn: !!r.usualIn}); }
console.log(JSON.stringify(out));
""" % (DEMO, DEMO)
crops = D.PREVIOUS_CROPS
cases, labels = [], D.rm_points().set_index("rm").label
combos = itertools.product([159, 5, 461], crops, [None, "Canola", "Peas"], ["Balanced", "Most reliable"],
                           [(), ("Lentils", "Peas")], [("Canola", "Spring Wheat"), ("Lentils", "Peas"), None])
for rm, prev, before, pri, exc, plan in combos:
    cases.append(dict(rm=rm, prev=prev, before=before or "", priority=PROF[pri], blocked=list(exc), unchecked=[],
                      plan1=plan[0] if plan else "", plan2=plan[1] if plan else "", _pri=pri))
js = json.loads(subprocess.run(["node", "--input-type=module", "-e", JS], input=json.dumps(cases), capture_output=True,
                               text=True, check=True).stdout)
bad = 0
for c, j in zip(cases, js):
    plan = (c["plan1"], c["plan2"]) if c["plan1"] else None
    o = D.options(c["rm"], labels[c["rm"]], c["prev"], c["_pri"], crop_before=c["before"] or None,
                  exclude=tuple(c["blocked"]), plan=plan)
    if o is None:
        ok = j["n"] == 0
    else:
        g = o["full"][o["full"]["Tied group"] == "✓"].Rotation.tolist()
        ok = j["n"] == o["n_options"] and len(j["group"]) == o["n_group"] and j["usualIn"] == bool(o["usual_in_group"])
        if o["usual"] is not None:
            u = o["usual"]
            ok &= abs(u.avg_rel - j["usual"][0]) < 1e-3 and abs(u.kept_2021 - j["usual"][1]) < 1e-3 and \
                abs(u.margin_yr - j["usual"][2]) < 1e-2 and (u.rule_issue or "") == j["usual"][3]
        top = o["top"]; ok &= j["picks"][0].startswith(f"{top.crop_2027}>{top.crop_2028}:")
    if not ok:
        bad += 1; print("MISMATCH", c, j, None if o is None else (o["n_options"], o["n_group"]))
print(f"{len(cases)} cases, {bad} mismatches")
sys.exit(1 if bad else 0)
