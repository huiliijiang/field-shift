// Selection logic for the Saskatchewan page. No modelling here: every number comes from data.json,
// written by 04_reproduce/saskatchewan/scripts/export_demo.py. This mirrors app/data.py::options().
export const PRIORITY = {best_average: 'Best average', balanced: 'Balanced', most_reliable: 'Most reliable'};

export const defaults = () => ({step: 1, rm: 159, prev: 'Spring Wheat', before: '', plan1: 'Canola', plan2: 'Spring Wheat',
  blocked: [], unchecked: [], priority: 'balanced'});

export function sanitize(raw, data) {
  const d = defaults(), crops = data.crops;
  if (!raw || typeof raw !== 'object') return d;
  if (data.rms.some(r => r.rm === raw.rm)) d.rm = raw.rm;
  if (crops.includes(raw.prev)) d.prev = raw.prev;
  for (const k of ['before', 'plan1', 'plan2']) if (raw[k] === '' || crops.includes(raw[k])) d[k] = raw[k];
  for (const k of ['blocked', 'unchecked']) if (Array.isArray(raw[k])) d[k] = [...new Set(raw[k].filter(c => crops.includes(c)))];
  d.unchecked = d.unchecked.filter(c => !d.blocked.includes(c));
  if (raw.priority in PRIORITY) d.priority = raw.priority;
  if (Number.isInteger(raw.step) && raw.step >= 1 && raw.step <= 4) d.step = raw.step;
  return d;
}

// low / medium / high, same thresholds as scripts/recommend_yield.py::risk()
export const risk = r => r.pfail <= 0.25 && r.cvar >= 0.70 ? 'low' : (r.pfail > 0.45 || r.cvar < 0.55 ? 'high' : 'medium');

export function compute(data, s) {
  const nice = c => data.nice[c] || c, w = data.profiles[s.priority];
  // a 2025 -> 2026 history that already breaks a rule is treated as "2025 unknown", as in the Streamlit app
  const historyIssue = !!(s.before && data.rules[`${s.before}|${s.prev}`]);
  const b = s.before && !historyIssue ? s.before : '-';
  const all = data.rotations[s.rm][s.prev].map(r => ({...r, rotation: `${nice(r.c1)} → ${nice(r.c2)}`,
    score: (1 - w) * r.avg + w * r.cvar, rule: data.rules[`${b}|${s.prev}|${r.c1}|${r.c2}`] || '',
    blocked: [r.c1, r.c2].filter(c => s.blocked.includes(c)), unchecked: [...new Set([r.c1, r.c2])].filter(c => s.unchecked.includes(c))}));
  const allowed = all.filter(r => !r.rule && !r.blocked.length).sort((x, y) => y.score - x.score);
  if (!allowed.length) return {all, allowed, group: [], picks: [], usual: null, historyIssue};
  const best = allowed[0].score, group = allowed.filter(r => r.score >= best - data.tie);
  const top = (key) => [...group].sort((x, y) => y[key] - x[key] || y.score - x.score)[0];
  const picks = [];
  for (const [label, key] of [['Best in 2021', 'k21'], ['Highest $/ac', 'm']]) {
    const r = top(key), hit = picks.find(p => p.r === r);
    if (hit) hit.labels.push(label); else picks.push({r, labels: [label]});
  }
  const usual = s.plan1 && s.plan2 ? all.find(r => r.c1 === s.plan1 && r.c2 === s.plan2) : null;
  const usualIn = !!usual && group.includes(usual);
  return {all, allowed, group, picks, usual, usualIn, best, historyIssue};
}
