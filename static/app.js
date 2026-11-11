// Project page of PROPS: the two leaderboard tables and the scatter plot, drawn from the data in index.html.
(function () {
"use strict";
const D = JSON.parse(document.getElementById("props-data").textContent);
const M = Object.fromEntries(D.metrics.map(m => [m.k, m]));
const GCOL = {Recall: "#4A6C94", Resist: "#A8693F", Reuse: "#557F62", Diagnostic: "#6B6B6B", "": "#4A7AA8"};
const KCOL = {open: "#4A7AA8", closed: "#5A5A5A", ours: "#C2185B"};
const S = {sort: {full: ["Overall", false], phase_a: ["Overall", false]}, hover: null};
const $ = s => document.querySelector(s), $$ = s => Array.from(document.querySelectorAll(s));
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"})[c]);
const f1 = v => v == null ? "–" : v.toFixed(1);
const byName = (split, n) => D.splits[split].find(r => r.name === n);
// language: the page holds both; html.zh shows the Chinese copy, and the parts drawn here use D.ui of that language
let LANG = document.documentElement.classList.contains("zh") ? "zh" : "en";
const U = () => D.ui[LANG], Z = (o, k) => (LANG === "zh" && o[k + "_zh"]) || o[k];
const fmt = (s, v) => s.replace(/\{(\w+)\}/g, (_, k) => v[k]);
const hex = (h, a) => `rgba(${parseInt(h.slice(1, 3), 16)},${parseInt(h.slice(3, 5), 16)},${parseInt(h.slice(5, 7), 16)},${a})`;

// tooltip: any element with data-tip, plus the tips of table cells and points
const tip = $("#tip");
function showTip(h, e) {
  tip.innerHTML = h; tip.style.display = "block";
  const w = tip.offsetWidth, ht = tip.offsetHeight;
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + w > innerWidth - 8) x = e.clientX - w - 14;
  if (y + ht > innerHeight - 8) y = e.clientY - ht - 14;
  tip.style.left = Math.max(4, x) + "px"; tip.style.top = Math.max(4, y) + "px";
}
const hideTip = () => { tip.style.display = "none"; };
document.addEventListener("mousemove", e => {
  const t = e.target.closest && e.target.closest("[data-tip]");
  if (t) showTip((LANG === "zh" && t.dataset.tipZh) || t.dataset.tip, e); else if (!e.target.closest("[data-rich]")) hideTip();
});

function rowTip(split, r, k) {
  const g = D.groups.map(([g]) => `<tr><td style="color:${GCOL[g]}">${g}</td><td>${f1(r.v[g])}</td></tr>`).join("");
  const b = r.base && byName(split, r.base);
  let h = `<div class="t">${esc(r.name)}${r.closed ? " †" : ""}</div><div class="m">${esc(Z(r, "kind"))}</div>` +
    `<table><tr><td><b>Overall</b></td><td><b>${f1(r.v.Overall)}</b></td></tr>${g}` +
    `<tr><td>Follow</td><td>${f1(r.v.Follow)}</td></tr><tr><td>Exec</td><td>${f1(r.v.Exec)}</td></tr></table>` +
    `<div class="m">${fmt(U().ci, {a: f1(r.ci[0]), b: f1(r.ci[1])})}</div>`;
  if (k && !["Overall", "Follow", "Exec", ...D.groups.map(g => g[0])].includes(k))
    h += `<div>${esc(k)}: <b>${f1(r.v[k])}</b></div>`;
  if (b) { const kk = k || "Overall", d = r.v[kk] - b.v[kk];
    h += `<div>${esc(fmt(U().of, {k: kk, b: b.name, v: f1(b.v[kk]), d: (d >= 0 ? "+" : "") + d.toFixed(1)}))}</div>`; }
  return h;
}

// markers of equal area, as in the figures of the paper
function marker(shape, x, y, r, fill, stroke, sw) {
  const a = `fill="${fill}" stroke="${stroke || "#fff"}" stroke-width="${sw || 1}"`;
  if (shape === "o") return `<circle cx="${x}" cy="${y}" r="${r}" ${a}/>`;
  if (shape === "s") { const s = r * Math.sqrt(Math.PI); return `<rect x="${x - s / 2}" y="${y - s / 2}" width="${s}" height="${s}" ${a}/>`; }
  if (shape === "d") { const d = r * Math.sqrt(Math.PI / 2);
    return `<path d="M${x} ${y - d}L${x + d} ${y}L${x} ${y + d}L${x - d} ${y}Z" ${a}/>`; }
  const c = r * Math.sqrt(4 * Math.PI / (3 * Math.sqrt(3)));
  const p = [0, 2, 4].map(i => `${x + c * Math.sin(i * Math.PI / 3)} ${y - c * Math.cos(i * Math.PI / 3)}`);
  return `<path d="M${p.join("L")}Z" ${a}/>`;
}
const mini = r => `<svg width="12" height="12" viewBox="0 0 12 12">${marker(r.shape, 6, 6, 4.3, KCOL[r.color], "none")}</svg>`;

// ---- leaderboard: Overall, each group mean with its subtypes, then the diagnostics ----
const COLS = [{k: "Overall", g: ""}].concat(...D.groups.map(([g, subs]) => [{k: g, g}, ...subs.map(k => ({k, g}))]),
  [{k: "Follow", g: "Diagnostic"}, {k: "Exec", g: "Diagnostic"}]);
function th(split, k, extra) {
  const m = M[k], [key, asc] = S.sort[split], on = key === k;
  return `<th data-k="${k}" class="${on ? "on" : ""}" data-tip="${esc("<b>" + k + "</b><br>" + Z(m, "desc"))}" ` +
    `style="color:${GCOL[m.group]}" ${extra || ""}>${k === m.group ? U().mean : m.short}${on ? (asc ? " ▴" : " ▾") : ""}</th>`;
}
function drawTable(split) {
  const [key, asc] = S.sort[split], sg = asc ? 1 : -1, val = r => r.v[key] ?? -1;
  const R = D.splits[split].slice().sort((a, b) => sg * (val(a) - val(b)) || b.v.Overall - a.v.Overall);
  let h = `<thead><tr class="r1"><th class="l" rowspan="2">#</th><th class="l" rowspan="2">${U().system}</th>` +
    `<th class="l" rowspan="2">${U().type}</th>` + th(split, "Overall", 'rowspan="2"');
  for (const [g, subs] of D.groups)
    h += `<th class="gh" colspan="${subs.length + 1}" style="--g:${GCOL[g]}" data-tip="${esc(Z(M[g], "desc"))}">${g}</th>`;
  h += `<th class="gh" colspan="2" style="--g:${GCOL.Diagnostic}" data-tip="${esc(U().diag_tip)}">` +
    `${U().diag}</th></tr><tr class="r2">` + COLS.slice(1).map(c => th(split, c.k)).join("") + "</tr></thead><tbody>";
  for (const r of R) {
    const rank = 1 + R.filter(o => val(o) > val(r)).length, b = r.base && byName(split, r.base);
    h += `<tr data-n="${esc(r.name)}" class="${r.ours ? "ours" : ""}${S.hover === r.name ? " hl" : ""}">` +
      `<td class="rk l">${rank}</td><td class="nm l" data-rich="1">${esc(r.name)}${r.closed ? " †" : ""}</td>` +
      `<td class="l"><span class="kind">${mini(r)}${esc(Z(r, "kind"))}</span></td>`;
    for (const c of COLS) {
      const v = r.v[c.k], bg = v != null ? `background:${hex(GCOL[c.g], 0.04 + 0.3 * v / 100)}` : "";
      if (c.k === "Overall") {
        const d = b ? r.v.Overall - b.v.Overall : null;
        h += `<td class="ov" data-k="Overall" data-rich="1" style="${bg}">${f1(v)}` +
          (b ? `<span class="delta">${d >= 0 ? "+" : ""}${d.toFixed(1)}</span>` : "") + `<span class="ci">` +
          `<i style="left:${r.ci[0]}%;width:${r.ci[1] - r.ci[0]}%"></i><b style="left:${v}%"></b></span></td>`;
      } else h += `<td data-k="${c.k}" data-rich="1" style="${bg}">${f1(v)}</td>`;
    }
    h += "</tr>";
  }
  $(`#lb_${split}`).innerHTML = h + "</tbody>";
}
function setHover(n) {
  if (S.hover === n) return;
  S.hover = n;
  $$("table.board tbody tr").forEach(tr => tr.classList.toggle("hl", tr.dataset.n === n));
  $$("#sc .pt").forEach(g => g.classList.toggle("hot", g.dataset.n === n));
}
for (const split of Object.keys(D.splits)) {
  const el = $(`#lb_${split}`);
  el.addEventListener("click", e => {
    const t = e.target.closest("th[data-k]");
    if (!t) return;
    const k = t.dataset.k, [key, asc] = S.sort[split];
    S.sort[split] = [k, key === k ? !asc : false]; drawTable(split);
  });
  el.addEventListener("mousemove", e => {
    const tr = e.target.closest("tbody tr[data-n]"), td = e.target.closest("td[data-rich]");
    setHover(tr ? tr.dataset.n : null);
    if (tr && td) showTip(rowTip(split, byName(split, tr.dataset.n), td.dataset.k), e);
  });
  el.addEventListener("mouseleave", () => { setHover(null); hideTip(); });
}
// ---- scatter plot: Overall against Exec for every system on phase A, as in the paper ----
function hull(pts) {
  pts = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]), lo = [], up = [];
  for (const p of pts) { while (lo.length > 1 && cr(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); }
  for (const p of pts.reverse()) { while (up.length > 1 && cr(up[up.length - 2], up[up.length - 1], p) <= 0) up.pop(); up.push(p); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}
function region(P, pad) {
  const pts = [];
  for (const p of P) for (let t = 0; t < 24; t++) pts.push([p.x + pad * Math.cos(t * Math.PI / 12), p.y + pad * Math.sin(t * Math.PI / 12)]);
  return "M" + hull(pts).map(p => p[0].toFixed(1) + " " + p[1].toFixed(1)).join("L") + "Z";
}
const star = (x, y, r) => "M" + Array.from({length: 10}, (_, i) => { const q = i % 2 ? 0.42 * r : r;
  return `${x + q * Math.sin(i * Math.PI / 5)} ${y - q * Math.cos(i * Math.PI / 5)}`; }).join("L") + "Z";
const cross = (x, y) => `M${x - 5} ${y - 5}L${x + 5} ${y + 5}M${x - 5} ${y + 5}L${x + 5} ${y - 5}`;
function drawScatter() {
  const split = "phase_a", W = 860, H = 540, L = 58, R = 14, T = 12, B = 48, LO = -3, HI = 103;
  const X = v => L + (W - L - R) * (v - LO) / (HI - LO), Y = v => H - B - (H - T - B) * (v - LO) / (HI - LO);
  const P = D.splits[split].map(r => ({r, x: X(r.v.Exec), y: Y(r.v.Overall)}));
  let s = `<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" ` +
    `orient="auto"><path d="M0 0L10 5L0 10Z" fill="${KCOL.ours}"/></marker></defs>`;
  for (const v of [0, 20, 40, 60, 80, 100])
    s += `<line x1="${X(v)}" y1="${Y(LO)}" x2="${X(v)}" y2="${Y(HI)}" stroke="#eceef1"/>` +
      `<line x1="${X(LO)}" y1="${Y(v)}" x2="${X(HI)}" y2="${Y(v)}" stroke="#eceef1"/>` +
      `<text x="${X(v)}" y="${Y(LO) + 17}" font-size="11" fill="#777" text-anchor="middle">${v}</text>` +
      `<text x="${X(LO) - 7}" y="${Y(v) + 4}" font-size="11" fill="#777" text-anchor="end">${v}</text>`;
  s += `<rect x="${X(LO)}" y="${Y(HI)}" width="${X(HI) - X(LO)}" height="${Y(LO) - Y(HI)}" fill="none" stroke="#c8ccd2"/>` +
    `<text x="${(X(LO) + X(HI)) / 2}" y="${H - 9}" font-size="12.5" fill="#333" text-anchor="middle">${esc(Z(M.Exec, "axis"))}</text>` +
    `<text transform="translate(16 ${(Y(LO) + Y(HI)) / 2}) rotate(-90)" font-size="12.5" fill="#333" ` +
    `text-anchor="middle">${esc(Z(M.Overall, "axis"))}</text>`;
  const open = P.filter(p => p.r.color === "open"), closed = P.filter(p => p.r.color === "closed");
  const rng = Q => fmt(U().range, {a: f1(Math.min(...Q.map(p => p.r.v.Overall))), b: f1(Math.max(...Q.map(p => p.r.v.Overall)))});
  s += `<path d="${region(open, 13)}" fill="${hex(KCOL.open, 0.1)}" stroke="${hex(KCOL.open, 0.5)}" stroke-dasharray="4 3"/>` +
    `<path d="${region(closed, 13)}" fill="${hex(KCOL.closed, 0.1)}" stroke="${hex(KCOL.closed, 0.5)}" stroke-dasharray="4 3"/>`;
  const cx = Math.min(...closed.map(p => p.x)) - 22;
  const ox = X(28), oy = Y(24), obs = P.map(p => [p.x - 8, p.y - 8, p.x + 8, p.y + 8]);
  obs.push([L, T, X(75), Y(78)]);  // the legend, in the empty top-left corner
  obs.push([ox - 2, oy - 13, ox + 150, oy + 19], [cx - 160, Y(64) - 13, cx + 2, Y(64) + 19]);
  s += `<text x="${ox}" y="${oy}" font-size="12" fill="${KCOL.open}"><tspan font-weight="600">` +
    `${fmt(U().open, {n: open.length})}</tspan><tspan x="${ox}" dy="15">${rng(open)}</tspan></text>` +
    `<text x="${cx}" y="${Y(64)}" font-size="12" fill="${KCOL.closed}" text-anchor="end"><tspan font-weight="600">` +
    `${fmt(U().closed, {n: closed.length})}</tspan><tspan x="${cx}" dy="15">${rng(closed)}</tspan></text>`;
  const ref = D.refs[split], g = ref.gt, gx = X(g.v.Exec), gy = Y(g.v.Overall);
  s += `<path d="${star(gx, gy, 10)}" fill="#1f1f1f" stroke="#fff" data-tip="${esc(`<b>${Z(g, "name")}</b><br>Exec ${f1(g.v.Exec)} · ` +
    `Overall ${f1(g.v.Overall)}`)}"/><text x="${gx - 15}" y="${gy + 4}" font-size="11.5" font-weight="600" ` +
    `text-anchor="end">${U().gt}</text>`;
  obs.push([gx - 100, gy - 11, gx + 11, gy + 11]);
  for (const c of ref.ctrl) {
    const x = X(c.v.Exec), y = Y(c.v.Overall);
    s += `<g data-tip="${esc(`<b>${Z(c, "name")}</b><br>Exec ${f1(c.v.Exec)} · Overall ${f1(c.v.Overall)}`)}">` +
      `<rect x="${x - 7}" y="${y - 7}" width="14" height="14" fill="transparent"/>` +
      `<path d="${cross(x, y)}" stroke="#1f1f1f" stroke-width="1.6"/></g>`;
    obs.push([x - 7, y - 7, x + 7, y + 7]);
  }
  for (const p of P) {
    const q = p.r.base && P.find(o => o.r.name === p.r.base);
    if (!q) continue;
    const d = Math.hypot(p.x - q.x, p.y - q.y), ux = (p.x - q.x) / d, uy = (p.y - q.y) / d;
    if (d > 20) s += `<line x1="${q.x + 8 * ux}" y1="${q.y + 8 * uy}" x2="${p.x - 9 * ux}" y2="${p.y - 9 * uy}" ` +
      `stroke="${KCOL.ours}" stroke-width="1.3" stroke-opacity=".75" marker-end="url(#ah)"/>`;
  }
  // a few names next to their point, at the nearest free place; hover shows every other one
  const hit = (b, o) => b[0] < o[2] && b[2] > o[0] && b[1] < o[3] && b[3] > o[1], placed = [];
  const DIRS = [[1, 0], [-1, 0], [0, -1], [0, 1], [1, -1], [-1, -1], [1, 1], [-1, 1]];
  for (const n of D.labels) {
    const p = P.find(o => o.r.name === n);
    if (!p) continue;
    const t = `${n}${p.r.closed ? " †" : ""} ${f1(p.r.v.Overall)}`, w = t.length * 6.4, h = 13;
    let b = null;
    for (const d of [10, 22, 36]) {
      for (const [ax, ay] of DIRS) {
        const bx = ax > 0 ? p.x + d : ax < 0 ? p.x - d - w : p.x - w / 2, by = ay < 0 ? p.y - d - h : ay > 0 ? p.y + d : p.y - h / 2;
        const c = [bx, by, bx + w, by + h];
        if (bx < L + 4 || bx + w > W - R - 4 || by < T + 4 || by + h > Y(LO) - 4) continue;
        if (obs.some(o => hit(c, o)) || placed.some(o => hit(c, o))) continue;
        b = c; break;
      }
      if (b) break;
    }
    if (!b) continue;
    placed.push(b);
    const nx = Math.min(Math.max(p.x, b[0]), b[2]), ny = Math.min(Math.max(p.y, b[1]), b[3]);
    if (Math.hypot(nx - p.x, ny - p.y) > 14) s += `<line x1="${p.x}" y1="${p.y}" x2="${nx}" y2="${ny}" stroke="#9aa0a8" stroke-width=".8"/>`;
    s += `<text x="${b[0]}" y="${b[3] - 3}" font-size="11.5" pointer-events="none" ` +
      `fill="${p.r.ours ? KCOL.ours : "#333"}">${esc(t)}</text>`;
  }
  for (const p of P)
    s += `<g class="pt" data-rich="1" data-n="${esc(p.r.name)}">${marker(p.r.shape, p.x, p.y, 6.4, KCOL[p.r.color])}</g>`;
  // legend: three columns with one icon slot and one text start each, spaced after their text is measured; the
  // triangle and the star are moved down so that their boxes, not their centroids, sit on the middle of the text
  const LX = L + 10, LY = T + 10, ic = (x, y, k) => [
    marker("o", x, y, 5.6, KCOL.open), marker("s", x, y, 5.6, KCOL.open), marker("d", x, y, 5.6, KCOL.open),
    marker("t", x, y + 2.2, 5.6, KCOL.open), marker("o", x, y, 5.6, KCOL.open), marker("o", x, y, 5.6, KCOL.closed),
    marker("o", x, y, 5.6, KCOL.ours), `<path d="${star(x, y + 0.8, 8)}" fill="#1f1f1f"/>`,
    `<path d="${cross(x, y)}" stroke="#1f1f1f" stroke-width="1.6"/>`,
    `<line x1="${x - 8}" y1="${y}" x2="${x + 8}" y2="${y}" stroke="${KCOL.ours}" stroke-width="1.3" marker-end="url(#ah)"/>`][k];
  let k0 = 0;
  const LEG = U().legend.map(col => col.map(t => [k0++, t]));
  s += `<g id="lg"><rect id="lgbox" x="${LX}" y="${LY}" width="10" height="${4 * 19 + 10}" rx="6" fill="#fff" ` +
    `fill-opacity=".92" stroke="#dfe2e6"/>` + LEG.map(col => `<g class="lgc">` + col.map(([k, t], i) => {
      const y = LY + 14 + i * 19;
      return ic(9, y, k) + `<text x="22" y="${y + 4}" font-size="11.5" fill="#333">${esc(t)}</text>`;
    }).join("") + "</g>").join("") + "</g>";
  $("#sc").innerHTML = s;
  let lx = LX + 8;
  for (const c of $$("#sc .lgc")) {
    c.setAttribute("transform", `translate(${lx} 0)`);
    const b = c.getBBox(); lx += (b.width ? b.x + b.width : 160) + 22;
  }
  $("#lgbox").setAttribute("width", lx - 22 + 10 - LX);
}
$("#sc").addEventListener("mousemove", e => {
  const t = e.target.closest(".pt");
  setHover(t ? t.dataset.n : null);
  if (t) showTip(rowTip("phase_a", byName("phase_a", t.dataset.n)), e);
});
$("#sc").addEventListener("mouseleave", () => { setHover(null); hideTip(); });

// ---- best system of each subtype, shown on the cards ----
function drawBest() {
  for (const el of $$(".sub[data-k]")) {
    const k = el.dataset.k, best = sp => D.splits[sp].slice().sort((x, y) => y.v[k] - x.v[k])[0];
    const f = best("full"), a = best("phase_a");
    el.querySelector(".best").textContent = fmt(U().best, {f: f.name, fv: f1(f.v[k]), a: a.name, av: f1(a.v[k])});
  }
}

// ---- page: figure zoom, copy buttons, section highlight in the menu ----
const box = $("#lightbox");
document.addEventListener("click", e => {
  const img = e.target.closest("figure img");
  if (img) { box.querySelector("img").src = img.src; box.classList.add("on"); }
  else if (e.target.closest("#lightbox")) box.classList.remove("on");
});
document.addEventListener("keydown", e => { if (e.key === "Escape") box.classList.remove("on"); });
$$(".code button").forEach(b => b.addEventListener("click", () => {
  navigator.clipboard.writeText(b.parentNode.querySelector("pre").textContent).then(() => {
    const old = b.innerHTML; b.textContent = U().copied; setTimeout(() => { b.innerHTML = old; }, 1400); });
}));
const links = $$("nav.top a[href^='#']");
const spy = new IntersectionObserver(es => es.forEach(en => { if (en.isIntersecting)
  links.forEach(a => a.classList.toggle("on", a.getAttribute("href") === "#" + en.target.id)); }),
  {rootMargin: "-45% 0px -50% 0px"});
$$("main section[id]").forEach(s => spy.observe(s));
function drawAll() { Object.keys(D.splits).forEach(drawTable); drawScatter(); drawBest(); }
$("#lang").addEventListener("click", () => {
  LANG = LANG === "zh" ? "en" : "zh";
  document.documentElement.classList.toggle("zh", LANG === "zh");
  document.documentElement.lang = LANG === "zh" ? "zh-CN" : "en";
  try { localStorage.setItem("props-lang", LANG); } catch (e) {}
  hideTip(); drawAll();
});
document.documentElement.lang = LANG === "zh" ? "zh-CN" : "en";
drawAll();
})();
