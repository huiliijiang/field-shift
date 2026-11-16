// Canada side of Field Shift (Saskatchewan): a four-step planner filled with the replay of 44 past summers from
// 04_reproduce/saskatchewan. Selection logic lives in canada-core.mjs; numbers come from data.json.
import {defaults, sanitize, compute, risk, PRIORITY} from './canada-core.mjs';

const esc = x => String(x ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));

const KEY = 'fieldshift.prairie.v1';
const $ = s => document.querySelector(s);
let data, s = defaults(), stored = true;

const nice = c => data.nice[c] || c;
const pct = x => `${Math.round(x * 100)}%`;
const money = x => x < 0 ? `-$${Math.round(-x).toLocaleString()}` : `$${Math.round(x).toLocaleString()}`;
const amt = (x, c) => `${Math.round(x).toLocaleString()} ${data.units[c]}`;
const rmNow = () => data.rms.find(r => r.rm === s.rm);
const kind = c => data.pulses.includes(c) ? 'pulse' : data.cereals.includes(c) ? 'cereal' : 'oilseed';
const RISK = {low: 'Low', medium: 'Medium', high: 'High'};
const tip = t => `data-tip="${esc(t)}"`;

function save() { try { localStorage.setItem(KEY, JSON.stringify(s)); stored = true; } catch { stored = false; } $('#save-status').textContent = stored ? 'Saved on this browser' : 'Not saved'; }
const cropOptions = (sel, empty) => (empty ? `<option value="" ${sel === '' ? 'selected' : ''}>${empty}</option>` : '') + data.crops.map(c => `<option value="${c}" ${sel === c ? 'selected' : ''}>${nice(c)}</option>`).join('');
const field = (key, label, html, full = false) => `<div class="${full ? 'full' : ''}"><label for="${key}">${label}</label>${html}</div>`;
const select = (key, opts) => `<select id="${key}" name="${key}">${opts}</select>`;
const chips = (key, taken = []) => `<div class="chips" role="group">${data.crops.map(c => `<label><input type="checkbox" name="${key}" value="${c}" ${s[key].includes(c) ? 'checked' : ''} ${taken.includes(c) ? 'disabled' : ''}>${nice(c)}</label>`).join('')}</div>`;
const steps = () => `<ol class="steps">${['Your field', 'What matters', 'Explore options', 'Review together'].map((n, i) => `<li><button class="step ${s.step === i + 1 ? 'current' : ''}" data-step="${i + 1}"><span class="number">${i + 1}</span>${n}</button></li>`).join('')}</ol>`;
const nav = (back, next) => `<div class="actions">${back ? `<button class="button secondary" data-step="${s.step - 1}">← Back</button>` : '<span></span>'}${next ? `<button class="button primary" data-step="${s.step + 1}">${next} →</button>` : ''}</div>`;
const head = (title, lead) => `<div class="intro"><div class="eyebrow">STEP ${s.step} / 4 · ${esc(rmNow().label.toUpperCase())}, SASKATCHEWAN</div><h1>${title}</h1><p class="lead">${lead}</p></div>`;
const stat = (n, label) => `<div><div class="stat-number">${n}</div><div class="stat-label">${label}</div></div>`;

// ---- graphics ---------------------------------------------------------------------------------------------
// Southern Saskatchewan, 49–54°N. Plain equirectangular with the longitude squeezed for ~52°N.
function map() {
  const W = 300, H = 250, X = lon => 14 + (lon + 110) / 8.64 * (W - 28), Y = lat => 14 + (54 - lat) / 5 * (H - 28);
  const towns = [['Regina', 50.45, -104.61], ['Saskatoon', 52.13, -106.67]];
  return `<svg class="map" viewBox="0 0 ${W} ${H}" role="img" aria-label="Map of the ten pilot rural municipalities in southern Saskatchewan">
  <rect x="14" y="14" width="${W - 28}" height="${H - 28}" rx="6" class="land"/>
  <text x="20" y="${H - 20}" class="map-note">49°N · US border</text><text x="20" y="28" class="map-note">54°N</text>
  ${data.rms.map(r => `<g class="pin ${r.rm === s.rm ? 'on' : ''}" data-rm="${r.rm}" ${tip(`${r.label} · ${r.zone} soil zone`)}><circle cx="${X(r.lon)}" cy="${Y(r.lat)}" r="12" class="hit"/><circle cx="${X(r.lon)}" cy="${Y(r.lat)}" r="${r.rm === s.rm ? 7 : 5}" class="dot"/></g>`).join('')}
  ${(() => { const r = rmNow(), x = X(r.lon), left = x > W - 110; return `<text x="${x + (left ? -12 : 12)}" y="${Y(r.lat) + 4}" text-anchor="${left ? 'end' : 'start'}" class="map-label">${esc(r.label.replace(/ \(RM.*/, ''))}</text>`; })()}
  </svg>`;
}

// 44 summers: district yield as a share of normal, one column per summer, growing from the 100% baseline.
function history() {
  const h = data.history[s.rm], W = 380, H = 160, L = 46, base = 70, bw = (W - L) / h.length, scale = 55 / 0.6;
  const ht = d => Math.max(1.5, Math.abs(Math.max(-0.6, Math.min(0.6, d.rel - 1))) * scale);
  const bar = (d, i) => { const up = d.rel >= 1;
    return `<rect x="${(L + i * bw + 1).toFixed(1)}" y="${(up ? base - ht(d) : base).toFixed(1)}" width="${(bw - 2).toFixed(1)}" height="${ht(d).toFixed(1)}" rx="2" class="${up ? 'good' : 'bad'} ${d.y === 2021 ? 'mark' : ''}" ${tip(`${d.y}: ${pct(d.rel)} of a normal crop · ${d.hot} days above 30 °C at flowering · ${d.rain} mm May–June rain`)}/>`; };
  const i21 = h.findIndex(d => d.y === 2021), d21 = h[i21];
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="District yield in each of 44 summers as a share of a normal crop">
  <line x1="${L}" x2="${W}" y1="${base}" y2="${base}" class="axis"/><text x="${L - 6}" y="${base + 3}" text-anchor="end" class="tick">normal</text>
  ${h.map(bar).join('')}
  ${d21 ? `<text x="${(L + (i21 + 1) * bw).toFixed(1)}" y="${(base + ht(d21) + 13).toFixed(1)}" text-anchor="end" class="callout-label">2021 · ${pct(d21.rel)}</text>` : ''}
  <text x="${L}" y="${H - 3}" class="tick">${h[0].y}</text><text x="${W}" y="${H - 3}" text-anchor="end" class="tick">${h.at(-1).y}</text></svg>`;
}

// Every allowed rotation: what it kept in a 2021-like drought (x) against $/ac a year (y).
function scatter(res, cards) {
  const pts = res.allowed, W = 620, H = 300, L = 46, B = 36, T = 14, Rr = 14;
  const xs = pts.map(r => r.k21), ys = pts.map(r => r.m);
  const x0 = Math.floor(Math.min(...xs) * 10) / 10, x1 = Math.ceil(Math.max(...xs) * 10) / 10 || 1;
  const y0 = Math.floor(Math.min(0, ...ys) / 50) * 50, y1 = Math.ceil(Math.max(...ys) / 50) * 50 || 50;
  const X = v => L + (v - x0) / (x1 - x0 || 1) * (W - L - Rr), Y = v => H - B - (v - y0) / (y1 - y0 || 1) * (H - B - T);
  const xt = []; for (let v = x0; v <= x1 + 1e-9; v += (x1 - x0) > 0.5 ? 0.2 : 0.1) xt.push(v);
  const yt = []; for (let v = y0; v <= y1; v += (y1 - y0) > 300 ? 100 : 50) yt.push(v);
  const named = new Map(cards.map(([r, , , cls]) => [r, cls]));
  const dot = r => `<circle cx="${X(r.k21).toFixed(1)}" cy="${Y(r.m).toFixed(1)}" r="${named.has(r) ? 7 : 5}" class="${res.group.includes(r) ? 'tied' : 'other'} ${named.has(r) ? 'named' : ''}" ${tip(`${r.rotation}: ${pct(r.k21)} kept in a 2021-like drought · ${money(r.m)}/ac a year · ${pct(r.avg)} on average${res.group.includes(r) ? ' · tied group' : ''}`)}/>`;
  const label = ([r, id]) => { const x = X(r.k21), right = x > W - 150; return `<text x="${x + (right ? -11 : 11)}" y="${Y(r.m) + 4}" text-anchor="${right ? 'end' : 'start'}" class="pt-label">${id.startsWith('S0') ? 'Your plan: ' : ''}${r.rotation}</text>`; };
  return `<svg class="chart scatter" viewBox="0 0 ${W} ${H}" role="img" aria-label="Allowed rotations by yield kept in a 2021-like drought and dollars per acre">
  ${yt.map(v => `<line x1="${L}" x2="${W - Rr}" y1="${Y(v)}" y2="${Y(v)}" class="grid"/><text x="${L - 8}" y="${Y(v) + 4}" text-anchor="end" class="tick">${money(v)}</text>`).join('')}
  ${xt.map(v => `<text x="${X(v)}" y="${H - B + 16}" text-anchor="middle" class="tick">${pct(v)}</text>`).join('')}
  <text x="${(L + W - Rr) / 2}" y="${H - 4}" text-anchor="middle" class="axis-title">kept in a 2021-like drought (share of a normal crop)</text>
  <text x="${L}" y="${T - 2}" class="axis-title">$/ac a year</text>
  ${pts.filter(r => !res.group.includes(r)).map(dot).join('')}${pts.filter(r => res.group.includes(r) && !named.has(r)).map(dot).join('')}${[...named.keys()].filter(r => pts.includes(r)).map(dot).join('')}
  ${cards.filter(([r]) => pts.includes(r)).map(label).join('')}</svg>
  <div class="legend chart-legend"><span><i class="k tied"></i>Tied group (${res.group.length})</span><span><i class="k other"></i>Other allowed rotations (${pts.length - res.group.length})</span></div>`;
}

const hbar = (label, v, cls, text) => `<div class="hbar"><span class="hbar-label">${label}</span><span class="hbar-track"><i class="${cls}" style="width:${Math.max(2, Math.min(100, v / 1.2 * 100)).toFixed(1)}%"></i></span><span class="hbar-val">${text}</span></div>`;

// ---- steps ------------------------------------------------------------------------------------------------
function step1() {
  const rm = rmNow(), h = data.history[s.rm], bad = h.filter(d => d.rel < 0.8).length;
  return `<div class="layout"><section>${head('A plan that starts<br>with your district.', 'You choose next year’s crops in autumn, months before you know the summer.')}
  <div class="panel"><div class="panel-title"><h2>Your starting point</h2><span class="label-tag">FARMER INPUT</span></div><div class="form-grid">
  ${field('rm', 'Rural municipality', select('rm', data.rms.map(r => `<option value="${r.rm}" ${r.rm === s.rm ? 'selected' : ''}>${esc(r.label)}</option>`).join('')), true)}
  ${field('prev', 'Harvested in 2026', select('prev', cropOptions(s.prev)))}
  ${field('before', 'Crop in 2025', select('before', cropOptions(s.before, 'Not sure')))}
  ${field('plan1', 'Usual plan: 2027', select('plan1', cropOptions(s.plan1, 'No usual plan')))}
  ${field('plan2', 'Usual plan: 2028', select('plan2', cropOptions(s.plan2, 'No usual plan')))}
  </div></div>${nav(false, 'Set limits & priorities')}</section>
  <aside><div class="panel map-card"><div class="map-top"><div class="eyebrow">REGIONAL CONTEXT</div><h2>${esc(rm.label)}</h2><div class="location"><span class="dot"></span>${esc(rm.zone)} soil zone · Saskatchewan</div></div>${map()}<div class="map-caption">Ten pilot districts. Click a dot to switch. Results are district averages, not your field.</div></div>
  <div class="panel"><div class="subhead">44 SUMMERS HERE · YIELD VS A NORMAL CROP</div>${history()}<div class="context-data top-gap">${stat(`${bad} of ${h.length}`, 'summers more than 20% below normal')}${stat(pct(Math.min(...h.map(d => d.rel))), `worst summer (${h.reduce((a, b) => a.rel < b.rel ? a : b).y})`)}</div>
  <p class="hint top-gap">NASA POWER · MODIS NDVI · Saskatchewan RM yields. History, not a forecast.</p></div></aside></div>`;
}

function step2() {
  return `<div class="layout"><section>${head('What needs to work<br>on your farm?', 'Unknown is a useful answer.')}
  <div class="panel"><h2>Practical limits</h2>
  <div class="top-gap"><span class="field-label">Crops I can’t grow</span>${chips('blocked')}</div>
  <div class="top-gap"><span class="field-label">Crops I haven’t checked yet <span class="optional">buyer, seed, equipment</span></span>${chips('unchecked', s.blocked)}</div></div>
  <div class="panel"><h2>How much should bad years count?</h2><div class="seg" role="radiogroup">${Object.entries(PRIORITY).map(([k, v]) => `<label><input type="radio" name="priority" value="${k}" ${s.priority === k ? 'checked' : ''}>${v}</label>`).join('')}</div>
  <p class="hint top-gap">${{best_average: 'Only the average of 44 summers counts.', balanced: 'Half the weight on the worst 10% of outcomes.', most_reliable: '80% of the weight on the worst 10% of outcomes.'}[s.priority]}</p></div>
  ${nav(true, 'Explore options')}</section>
  <aside><div class="panel"><div class="subhead">YOU SET THE DIRECTION</div><ul class="help-list"><li><strong>Can’t grow:</strong> taken out.</li><li><strong>Not checked:</strong> kept, and becomes a question.</li><li><strong>Rotation rules</strong> are applied for you.</li><li><strong>Prices</strong> never decide; $/ac is shown beside yield.</li></ul></div></aside></div>`;
}

function shown(res) {
  const out = [];
  if (res.usual) out.push([res.usual, 'S0 · YOUR USUAL PLAN', '', 's0']);
  res.picks.forEach((p, i) => { if (p.r !== res.usual) out.push([p.r, `S${out.length} · ${p.labels.join(' · ').toUpperCase()}`, '', i ? 's2' : 's1']); else out[0][1] += ` · ALSO ${p.labels.join(' · ').toUpperCase()}`; });
  return out;
}

function headline(res) {
  const g = res.group, u = res.usual;
  if (!g.length) return '<div class="focus-strip">No rotation is left after the rules and your limits. Remove a limit in step 2.</div>';
  const lo = Math.min(...g.map(x => x.avg)), hi = Math.max(...g.map(x => x.avg)), k = g.map(x => x.k21), m = g.map(x => x.m);
  const plan = !u ? '' : u.rule ? `Your usual plan breaks a rotation guideline, so it is not in the group.` : u.blocked.length ? 'Your usual plan uses a crop you can’t grow.' : res.usualIn ? 'Your usual plan is one of them.' : `Your usual plan did a little worse on average (${pct(u.avg)}).`;
  return `<div class="focus-strip"><div class="tiles">
  ${stat(g.length, g.length === 1 ? 'rotation held up best' : 'rotations held up about the same')}
  ${stat(lo === hi ? pct(lo) : `${pct(lo)}–${pct(hi)}`, 'of a normal crop on average')}
  ${stat(`${pct(Math.min(...k))}–${pct(Math.max(...k))}`, 'kept in a 2021-like drought')}
  ${stat(`${money(Math.min(...m))}–${money(Math.max(...m))}`, 'per acre a year, 2026 prices')}</div>
  <p class="no-margin top-gap"><strong>${plan}</strong> On average they are tied; the drought year and the dollars are where they differ.</p></div>`;
}

function effect(r) {
  const parts = [[r.e1, r.c1, s.prev], [r.e2, r.c2, r.c1]].filter(([e]) => Math.abs(e) >= 0.005).map(([e, c, after]) => `${nice(c).toLowerCase()} on ${nice(after).toLowerCase()} stubble ${e >= 0 ? '+' : ''}${Math.round(e * 100)}%`);
  return parts.length ? `Previous-crop effect: ${parts.join('; ')} (Manitoba fields, MASC 2011–20).` : 'No measurable previous-crop effect for these two years.';
}

function card(r, id, _t, cls, res) {
  const inG = res.group.includes(r), status = r.rule ? ['blocked', `Breaks a guideline: ${esc(r.rule)}`] : r.blocked.length ? ['blocked', `You can’t grow ${r.blocked.map(nice).join(', ')}`] : r.unchecked.length ? ['', `To check: ${r.unchecked.map(nice).join(', ')}`] : ['', inG ? 'In the tied group' : 'Outside the tied group'];
  return `<article class="panel strategy ${cls}"><div class="id">${id}</div><h2>${r.rotation}</h2><span class="status ${status[0]}">${status[1]}</span>
  <div class="timeline four"><div class="season ${kind(s.prev)}"><b>2026</b>${nice(s.prev)}</div><div class="season ${kind(r.c1)}"><b>2027</b>${nice(r.c1)}</div><div class="season ${kind(r.c2)}"><b>2028</b>${nice(r.c2)}</div><div class="season"><b>risk</b>${RISK[risk(r)]}</div></div>
  <div class="metric"><div class="metric-label">YIELD · SHARE OF A NORMAL CROP</div>${hbar('Average', r.avg, 'avg', pct(r.avg))}${hbar('2021-like', r.k21, 'dry', pct(r.k21))}</div>
  <div class="metric"><div class="metric-label">MARGIN · 2026 GUIDE PRICES</div><div class="pair"><div><span class="big">${money(r.m)}</span><span class="stat-label">/ac a year</span></div><div><span class="big">${money(r.m21)}</span><span class="stat-label">/ac in a 2021-like year</span></div></div></div>
  <div class="metric"><div class="metric-label">DOWNSIDE</div><div class="pair"><div><span class="big">${pct(r.pfail)}</span><span class="stat-label">chance of a bad year</span></div><div><span class="big">${pct(r.cvar)}</span><span class="stat-label">worst 10% of outcomes</span></div></div></div>
  <details><summary>Details & open questions</summary><p>Normal year: ${nice(r.c1).toLowerCase()} ~${amt(r.n1, r.c1)}, ${nice(r.c2).toLowerCase()} ~${amt(r.n2, r.c2)}. 2021-like: ~${amt(r.d1, r.c1)} and ~${amt(r.d2, r.c2)}.</p><p>${effect(r)}</p><p>${r.rule ? 'Past yields don’t show the disease and weed build-up a broken rule causes, so these numbers look better than they would be.' : inG ? `Within 2 points of the best score, like the other ${res.group.length - 1} in the tied group.` : r.blocked.length ? 'Shown for comparison only.' : `${Math.round((res.best - r.score) * 100)} points of a normal crop below the best score.`}</p><ul><li>District average, not your field.</li><li>2021 is history, not protection.</li>${r.unchecked.map(c => `<li>${nice(c)}: buyer, seed and equipment to confirm.</li>`).join('')}</ul></details></article>`;
}

// The tied group only (plus the usual plan if it is outside): the other rotations are the gray dots in the scatter.
function table(res) {
  const rows = [...res.group].sort((a, b) => b.k21 - a.k21);
  if (res.usual && !rows.includes(res.usual)) rows.push(res.usual);
  const mini = (v, cls) => `<span class="mini"><i class="${cls}" style="width:${Math.max(2, Math.min(100, v / 1.2 * 100)).toFixed(0)}%"></i></span>`;
  return `<div class="table-wrap"><table class="data"><thead><tr><th>ROTATION</th><th class="l">AVERAGE YEAR</th><th class="l">2021-LIKE DROUGHT</th><th>$/AC A YEAR</th></tr></thead><tbody>${rows.map(r => `<tr><td>${r.rotation}${r === res.usual ? '<span class="tag">your plan</span>' : ''}${r.rule ? '<span class="tag stop">breaks a guideline</span>' : !res.group.includes(r) ? '<span class="tag warn">outside the group</span>' : ''}${r.unchecked.length ? '<span class="tag warn">to check</span>' : ''}</td><td class="l">${mini(r.avg, 'avg')}${pct(r.avg)}</td><td class="l">${mini(r.k21, 'dry')}${pct(r.k21)}</td><td>${money(r.m)}</td></tr>`).join('')}</tbody></table></div>`;
}

function step3() {
  const res = compute(data, s), cards = shown(res);
  return `${head('Tied on average.<br>Different when it’s dry.', 'Every allowed two-year rotation, replayed through 44 past summers. No ranking.')}
  ${res.historyIssue ? `<div class="callout">Your 2025 → 2026 crops already break a rotation rule, so 2025 is treated as unknown.</div>` : ''}
  ${headline(res)}
  ${res.allowed.length ? `<section class="panel"><div class="subhead">WHERE THEY DIFFER · EACH DOT IS ONE ROTATION</div>${scatter(res, cards)}</section>` : ''}
  <div class="strategy-grid ${cards.length === 2 ? 'two' : ''} top-gap">${cards.map(c => card(c[0], c[1], c[2], c[3], res)).join('')}</div>
  <section class="panel source-panel"><details><summary>The ${res.group.length} tied rotations, side by side</summary><p class="hint">Sorted by what they kept in a 2021-like drought. The other ${res.allowed.length - res.group.length} allowed rotations are the gray dots above.</p>${table(res)}</details></section>
  <section class="panel source-panel"><details><summary>Evidence & limits</summary><ul><li><strong>No skill before planting.</strong> With the summer known, the model explains about 59% of year-to-year swings; before planting, nothing beats “assume a normal year”.</li><li><strong>Backtest on 6 districts never used for tuning:</strong> in 2021 the advice kept 61% of a normal crop vs 50% for canola → wheat; its worst year was lower (39% vs 50%).</li><li><strong>The order inside the tied group is not reliable,</strong> so options are labelled, not ranked.</li><li><strong>2021 is history, not protection:</strong> the rotation that held up best in earlier droughts did not keep more in the next one.</li></ul></details></section>
  ${nav(true, 'Prepare advisor review')}`;
}

function step4() {
  const res = compute(data, s), cards = shown(res), qs = [];
  for (const [r] of cards) for (const c of r.unchecked) qs.push(`${nice(c)} (in ${r.rotation}): buyer, seed and equipment?`);
  if (res.usual?.rule) qs.push(`Your usual plan breaks a guideline (${res.usual.rule}). What pressure has it caused?`);
  if (s.before === '' || res.historyIssue) qs.push('Confirm the 2025 crop.');
  qs.push('How does this field differ from the district average?', 'Which insurance coverage fits this downside?', 'Do current prices change the $/ac order?');
  const line = (a, b) => `<div class="summary-line"><span>${a}</span><strong>${b}</strong></div>`;
  return `${head('Take it to someone<br>who knows your land.', 'One page for your agronomist.')}
  <div class="review-layout"><section><div class="panel"><h2>Your entries</h2>${line('District', esc(rmNow().label))}${line('2025 → 2026', `${s.before ? nice(s.before) : 'not entered'} → ${nice(s.prev)}`)}${line('Usual plan', res.usual ? res.usual.rotation : 'none')}${line('Can’t grow', s.blocked.map(nice).join(', ') || 'none')}${line('Not checked', s.unchecked.map(nice).join(', ') || 'none')}${line('Priority', PRIORITY[s.priority])}</div>
  <div class="panel"><h2>What 44 past summers show</h2>${headline(res)}<div class="table-wrap top-gap"><table class="data"><thead><tr><th>OPTION</th><th>AVG YIELD</th><th>2021-LIKE</th><th>$/AC/YR</th><th>BAD-YEAR CHANCE</th></tr></thead><tbody>${cards.map(([r, id]) => `<tr><td>${r.rotation}<span class="tag">${id.split(' · ').slice(1).join(' · ').toLowerCase()}</span></td><td>${pct(r.avg)}</td><td>${pct(r.k21)}</td><td>${money(r.m)}</td><td>${pct(r.pfail)}</td></tr>`).join('')}</tbody></table></div></div>
  <div class="panel"><h2>Questions to settle locally</h2><ul class="mini-list">${qs.map(q => `<li>${esc(q)}</li>`).join('')}</ul></div>
  <div class="panel"><h2>Watch during the season</h2><div class="table-wrap"><table class="data"><thead><tr><th>ALERT</th><th class="l">KNOWN BY</th><th class="l">FIRED HERE</th><th class="l">WHAT TO DO</th></tr></thead><tbody>${data.alerts[s.rm].map(a => `<tr><td>${esc(a.alert)}</td><td class="l">${esc(a.known_by)}</td><td class="l">${a.fired} of ${a.n}</td><td class="l wrap">${esc(a.action)}</td></tr>`).join('')}</tbody></table></div></div>
  <p class="disclaimer">District averages, not fields. Likelihoods from past summers, not a forecast or a recommendation. Margins use the 2026 Saskatchewan Crop Planning Guide.</p>${nav(true, '')}</section>
  <aside class="no-print"><div class="panel"><h2>Share</h2><div class="download-stack"><button class="button primary" id="print">Print / save as PDF</button><button class="button secondary" id="reset">Start a new plan</button></div><p class="hint top-gap">Your entries stay in this browser.</p></div></aside></div>`;
}

function render() {
  $('#app').innerHTML = `<main class="workspace" id="main">${steps()}${[step1, step2, step3, step4][s.step - 1]()}</main>`;
  save();
}

// ---- events -----------------------------------------------------------------------------------------------
const tipBox = Object.assign(document.createElement('div'), {className: 'tip', hidden: true});
document.body.append(tipBox);
document.addEventListener('mousemove', e => {
  const el = e.target.closest?.('[data-tip]');
  if (!el) { tipBox.hidden = true; return; }
  tipBox.textContent = el.dataset.tip; tipBox.hidden = false;
  const w = tipBox.offsetWidth;
  tipBox.style.left = `${Math.max(8, Math.min(innerWidth - w - 8, e.clientX - w / 2))}px`;
  tipBox.style.top = `${e.clientY - tipBox.offsetHeight - 14 < 8 ? e.clientY + 18 : e.clientY - tipBox.offsetHeight - 14}px`;
});
document.addEventListener('click', e => {
  const b = e.target.closest('[data-step]'), pin = e.target.closest('[data-rm]');
  if (b) { s.step = Number(b.dataset.step); render(); scrollTo(0, 0); }
  else if (pin) { s.rm = Number(pin.dataset.rm); tipBox.hidden = true; render(); }
  else if (e.target.id === 'print') print();
  else if (e.target.id === 'reset') { s = defaults(); render(); scrollTo(0, 0); }
});
document.addEventListener('change', e => {
  const el = e.target, k = el.name;
  if (!k) return;
  if (el.type === 'checkbox') {
    s[k] = el.checked ? [...s[k], el.value] : s[k].filter(c => c !== el.value);
    if (k === 'blocked') s.unchecked = s.unchecked.filter(c => !s.blocked.includes(c));
  } else s[k] = k === 'rm' ? Number(el.value) : el.value;
  if ((k === 'plan1' || k === 'plan2') && el.value === '') s.plan1 = s.plan2 = '';
  render();
});

try {
  data = await (await fetch('data.json')).json();
  try { const saved = localStorage.getItem(KEY); if (saved) s = sanitize(JSON.parse(saved), data); } catch { stored = false; }
  render();
} catch (err) {
  $('#app').innerHTML = `<main class="workspace"><p class="error">Could not load data.json. Serve this folder over http (python3 -m http.server) instead of opening the file directly.</p></main>`;
}
