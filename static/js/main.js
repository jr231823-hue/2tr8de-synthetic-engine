/* 2TR8DE Synthetic Lab — frontend */

"use strict";

// ── Chart.js global defaults ──────────────────────────────────────────────────
Chart.defaults.color = "#8b949e";
Chart.defaults.borderColor = "#21262d";
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 11;

const GOLD  = "#f0c040";
const CYAN  = "#00d4ff";
const RED   = "#e74c3c";
const GREEN = "#2ecc71";
const CAT_COLORS = { Good: GOLD, Medium: CYAN, Bad: RED };

// ── state ──────────────────────────────────────────────────────────────────────
const charts = {};  // keyed by canvas id
let busy = false;   // prevent double-submission while a request is in flight

// bulk state
let allStrategies  = [];
let currentFilter  = "all";
let sortCol        = "score";
let sortDir        = "desc";
let modalMcChart   = null;  // Deep Dive modal chart instance

// ── DOM refs ──────────────────────────────────────────────────────────────────
const dropzone      = document.getElementById("dropzone");
const fileInput     = document.getElementById("file-input");
const progressWrap  = document.getElementById("progress-wrap");
const progressFill  = document.getElementById("progress-fill");
const progressLabel = document.getElementById("progress-label");
const errorBox      = document.getElementById("error-box");
const errorMsg      = document.getElementById("error-msg");
const results       = document.getElementById("results");
const uploadSection = document.getElementById("upload-section");

// ── drag & drop ────────────────────────────────────────────────────────────────
dropzone.addEventListener("dragover", e => {
  e.preventDefault();
  dropzone.classList.add("drag-over");
});

// Only remove highlight when the cursor actually leaves the dropzone boundary,
// not when it moves over a child element inside it.
dropzone.addEventListener("dragleave", e => {
  if (!dropzone.contains(e.relatedTarget)) {
    dropzone.classList.remove("drag-over");
  }
});

dropzone.addEventListener("drop", e => {
  e.preventDefault();
  dropzone.classList.remove("drag-over");
  const files = [...e.dataTransfer.files].filter(f => f.name.toLowerCase().endsWith(".csv"));
  if (files.length) handleFiles(files);
});

// The label inside the dropzone has no `for` attribute, so a click anywhere in
// the dropzone reaches only this handler — no double-trigger.
dropzone.addEventListener("click", () => {
  if (!busy) fileInput.click();
});

fileInput.addEventListener("change", e => {
  const files = [...e.target.files].filter(f => f.name.toLowerCase().endsWith(".csv"));
  // Clear immediately so re-selecting the same files later fires `change` again.
  fileInput.value = "";
  if (files.length) handleFiles(files);
});

document.getElementById("btn-reset").addEventListener("click", resetUI);
document.getElementById("btn-export").addEventListener("click", exportAB);
document.querySelectorAll(".filter-btn").forEach(btn => {
  btn.addEventListener("click", () => filterBy(btn.dataset.filter));
});
document.querySelectorAll(".th-sort").forEach(th => {
  th.addEventListener("click", () => sortBy(th.dataset.col));
});

// ── upload & analyse ──────────────────────────────────────────────────────────
async function handleFiles(files) {
  if (!files.length) return;

  busy = true;
  hideError();

  const isBulk = files.length > 1;
  showProgress(5, isBulk ? `Loading ${files.length} strategy files…` : "Uploading file…");

  const form = new FormData();
  for (const f of files) form.append("file", f);

  let resp;
  try {
    showProgress(20, isBulk ? "Parsing trade logs…" : "Analysing statistical properties…");
    resp = await fetch("/api/analyse", { method: "POST", body: form });
  } catch (err) {
    showError("Network error: " + err.message);
    hideProgress();
    busy = false;
    return;
  }

  showProgress(55, isBulk ? "Scoring all strategies…" : "Generating 1,000 synthetic equity curves…");
  await tick(300);
  showProgress(80, "Computing Robustness Score…");
  await tick(250);
  showProgress(95, "Rendering charts…");
  await tick(200);

  let data;
  try {
    data = await resp.json();
  } catch {
    showError("Invalid response from server.");
    hideProgress();
    busy = false;
    return;
  }

  if (!data.ok) {
    showError(data.error || "Unknown error.");
    hideProgress();
    busy = false;
    return;
  }

  showProgress(100, "Done");
  await tick(400);
  hideProgress();
  busy = false;

  renderResults(data);
}

// ── verdict banner ────────────────────────────────────────────────────────────
const VERDICT = {
  A: { cls: "verdict-deploy",   icon: "✅", text: "READY TO DEPLOY TO DEMO" },
  B: { cls: "verdict-deploy",   icon: "✅", text: "READY TO DEPLOY TO DEMO" },
  C: { cls: "verdict-optimize", icon: "⚠️", text: "OPTIMIZE IN SQX BEFORE DEMO" },
  D: { cls: "verdict-discard",  icon: "❌", text: "DISCARD — GENERATE NEW STRATEGY" },
  F: { cls: "verdict-discard",  icon: "❌", text: "DISCARD — GENERATE NEW STRATEGY" },
};

function renderVerdict(scoreOrGrade) {
  const grade = typeof scoreOrGrade === "object" ? scoreOrGrade.grade : scoreOrGrade;
  const v      = VERDICT[grade] || VERDICT.F;
  const banner = document.getElementById("verdict-banner");
  banner.className = "verdict-banner " + v.cls;
  document.getElementById("verdict-icon").textContent = v.icon;
  document.getElementById("verdict-text").textContent =
    (typeof scoreOrGrade === "object" && scoreOrGrade.recommendation)
      ? scoreOrGrade.recommendation
      : v.text;
}

// ── render (dispatch on mode) ─────────────────────────────────────────────────
function renderResults(data) {
  uploadSection.style.display = "none";
  results.style.display       = "block";
  results.scrollIntoView({ behavior: "smooth" });

  renderVerdict(data.score);

  if (data.mode === "bulk") {
    document.getElementById("bulk-view").style.display   = "block";
    document.getElementById("single-view").style.display = "none";
    renderBulkResults(data);
  } else {
    document.getElementById("single-view").style.display = "block";
    document.getElementById("bulk-view").style.display   = "none";
    const { score, charts: cd, meta } = data;
    renderScoreBanner(score, meta);
    renderEquityChart(cd.equity);
    renderDonut(cd.donut);
    renderHistogram("chart-profit", cd.histograms.net_profit, "Net Profit", GOLD);
    renderHistogram("chart-dd",     cd.histograms.drawdown,   "Drawdown",   RED);
    renderHistogram("chart-wr",     cd.histograms.win_rate,   "Win Rate %", CYAN);
    renderHistogram("chart-sqn",    cd.histograms.sqn_score,  "SQN Score",  CYAN);
    renderScatter(cd.scatter);
    renderComparison(cd.comparison);
  }
}

// ── bulk results ──────────────────────────────────────────────────────────────
function renderBulkResults(data) {
  const { bulk, charts: cd } = data;
  const { strategies, summary, selection, executive } = bulk;
  window.currentSelection = selection || null;
  renderExecutiveSummary(executive || {}, selection || {});

  allStrategies = strategies;
  currentFilter = "visible";
  sortCol       = "score";
  sortDir       = "desc";

  // summary bar
  document.getElementById("bsb-total").textContent    = summary.total;
  const bsbSelected = document.getElementById("bsb-selected");
  if (bsbSelected) bsbSelected.textContent = summary.selected || 0;
  document.getElementById("bsb-deploy").textContent   = summary.deploy;
  document.getElementById("bsb-optimize").textContent = summary.optimize;
  document.getElementById("bsb-discard").textContent  = summary.discard;

  // filter counts
  document.getElementById("fcnt-all").textContent      = summary.total;
  const fcntSelected = document.getElementById("fcnt-selected");
  if (fcntSelected) fcntSelected.textContent = summary.selected || 0;
  document.getElementById("fcnt-deploy").textContent   = summary.deploy;
  document.getElementById("fcnt-optimize").textContent = summary.optimize;
  document.getElementById("fcnt-discard").textContent  = summary.discard;

  renderBulkTable();

  // bulk charts
  renderHistogram("bulk-chart-profit", cd.histograms.net_profit, "Net Profit", GOLD);
  renderHistogram("bulk-chart-dd",     cd.histograms.drawdown,   "Drawdown",   RED);
  renderHistogram("bulk-chart-wr",     cd.histograms.win_rate,   "Win Rate %", CYAN);
  renderHistogram("bulk-chart-sqn",    cd.histograms.sqn_score,  "SQN Score",  CYAN);
  renderScatterCanvas("bulk-chart-scatter", cd.scatter);
  renderDonutCanvas("bulk-chart-donut", cd.donut);
}


function renderExecutiveSummary(exec, selection) {
  const setText = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val ?? "—"; };
  setText("exec-verdict", exec.verdict || "PORTFOLIO ANALYSED");
  setText("exec-next", exec.next_move || (selection.message || "Review selected strategies before taking action."));
  setText("exec-risk", exec.risk_level || "—");
  setText("exec-edge", exec.edge_quality || "—");
  setText("exec-best", exec.best_candidate || "—");
  setText("exec-selected", `${exec.selected ?? 0} / ${exec.total ?? 0}`);

  const panel = document.getElementById("executive-panel");
  if (panel) {
    panel.classList.remove("risk-high", "risk-medium", "risk-low", "risk-controlled");
    const risk = String(exec.risk_level || "").toLowerCase();
    if (risk.includes("high")) panel.classList.add("risk-high");
    else if (risk.includes("medium")) panel.classList.add("risk-medium");
    else if (risk.includes("controlled")) panel.classList.add("risk-controlled");
    else if (risk.includes("low")) panel.classList.add("risk-low");
  }

  const bullets = document.getElementById("exec-bullets");
  if (bullets) {
    const items = Array.isArray(exec.bullets) ? exec.bullets : [];
    bullets.innerHTML = items.slice(0, 5).map(x => `<span>• ${escHtml(String(x))}</span>`).join("");
  }
}

function statusBadge(s) {
  const badge = String(s.badge || s.decision || "DISCARD").replace(/_/g, " ");
  const cls =
    s.selected ? "badge-selected" :
    badge.includes("DEMO") ? "badge-demo" :
    badge.includes("OPTIMIZE") || badge.includes("WATCH") ? "badge-watch" :
    "badge-discard";
  return `<span class="status-badge ${cls}">${escHtml(badge)}</span>`;
}

function renderBulkTable() {
  let rows = allStrategies.slice();

  // filter
  if (currentFilter === "visible") {
    rows = rows.filter(s => !(s.grade === "D" || s.grade === "F"));
  } else if (currentFilter === "selected") {
    rows = rows.filter(s => s.selected === true);
  } else if (currentFilter === "deploy") {
    rows = rows.filter(s => s.grade === "A" || s.grade === "B");
  } else if (currentFilter === "optimize") {
    rows = rows.filter(s => s.grade === "C");
  } else if (currentFilter === "discard") {
    rows = rows.filter(s => s.grade === "D" || s.grade === "F");
  }

  // sort
  rows.sort((a, b) => {
    let av = a[sortCol], bv = b[sortCol];
    if (typeof av === "string") av = av.toLowerCase();
    if (typeof bv === "string") bv = bv.toLowerCase();
    if (av < bv) return sortDir === "asc" ? -1 :  1;
    if (av > bv) return sortDir === "asc" ?  1 : -1;
    return 0;
  });

  // update header arrows
  document.querySelectorAll(".th-sort").forEach(th => {
    th.classList.remove("sort-active");
    const arrow = th.querySelector(".sort-arrow");
    if (arrow) arrow.textContent = "↕";
  });
  const activeHeader = document.querySelector(`.th-sort[data-col="${sortCol}"]`);
  if (activeHeader) {
    activeHeader.classList.add("sort-active");
    const arrow = activeHeader.querySelector(".sort-arrow");
    if (arrow) arrow.textContent = sortDir === "asc" ? "↑" : "↓";
  }

  // build rows
  const tbody = document.getElementById("bulk-tbody");
  tbody.innerHTML = "";
  rows.forEach((s, i) => {
    const rowClass =
      (s.grade === "A" || s.grade === "B") ? "row-ab" :
      s.grade === "C"                       ? "row-c"  : "row-df";
    const catClass =
      s.category === "Good"   ? "cat-good"   :
      s.category === "Medium" ? "cat-medium" : "cat-bad";
    const gradeClass = "tbl-grade tbl-grade-" + s.grade;

    const tr = document.createElement("tr");
    tr.className = rowClass;
    tr.addEventListener("dblclick", () => openStrategyModal(s));
    tr.addEventListener("click", (ev) => {
      if (ev.target && ev.target.closest && ev.target.closest(".btn-details")) {
        openStrategyModal(s);
      }
    });
    tr.title = "Double-click row or press Details to open Monte Carlo deep dive";
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td class="td-id">${s.selected ? "⭐ " : ""}${escHtml(String(s.id))}</td>
      <td>${s.score.toFixed(1)}</td>
      <td><span class="${gradeClass}">${s.grade}</span></td>
      <td>${statusBadge(s)}</td>
      <td>${fmtNum(s.net_profit)}</td>
      <td>${fmtNum(s.drawdown)}</td>
      <td>${s.win_rate.toFixed(1)}%</td>
      <td>${s.sqn_score.toFixed(2)}</td>
      <td>${s.ret_dd.toFixed(2)}</td>
      <td><span class="cat-chip ${catClass}">${escHtml(s.category)}</span></td>
      <td class="td-reason" title="${escHtml(String(s.selection_reason || s.explanation || ''))}">${escHtml(String(s.selection_reason || s.explanation || '—'))}</td>
      <td><button class="btn-details" type="button">Deep Dive</button></td>`;
    tbody.appendChild(tr);
  });
}


function openStrategyModal(s) {
  const modal = document.getElementById("strategy-modal");
  if (!modal) return;
  const detail = s.detail || {};
  const mc = detail.mc || {};
  const monkey = detail.monkey || {};

  document.getElementById("modal-title").textContent = String(s.id || "Strategy");
  document.getElementById("modal-headline").textContent = detail.headline || s.selection_reason || s.explanation || "No explanation available.";
  document.getElementById("modal-grade").textContent = s.grade || "—";
  document.getElementById("modal-score").textContent = Number(s.score || 0).toFixed(1);
  document.getElementById("modal-recommendation").textContent = detail.recommendation || s.recommendation || "REVIEW";
  document.getElementById("modal-profit").textContent = fmtNum(s.net_profit || 0);
  document.getElementById("modal-dd").textContent = fmtNum(s.drawdown || 0);
  document.getElementById("modal-sqn").textContent = Number(s.sqn_score || 0).toFixed(2);
  document.getElementById("modal-retdd").textContent = Number(s.ret_dd || 0).toFixed(2);
  document.getElementById("modal-survival").textContent = (mc.survival_pct ?? 0).toFixed ? `${mc.survival_pct.toFixed(1)}%` : `${mc.survival_pct || 0}%`;
  document.getElementById("modal-p05").textContent = fmtNum(mc.p05_final || 0);
  document.getElementById("modal-median").textContent = fmtNum(mc.median_final || 0);

  const monkeyVerdict = document.getElementById("modal-monkey-verdict");
  const monkeyNote = document.getElementById("modal-monkey-note");
  const monkeyScore = document.getElementById("modal-monkey-score");
  const monkeyBeats = document.getElementById("modal-monkey-beats");
  const monkeyEdge = document.getElementById("modal-monkey-edge");
  const monkeyMedian = document.getElementById("modal-monkey-median");

  if (monkeyVerdict) {
    monkeyVerdict.textContent = monkey.verdict || s.monkey_verdict || "NOT TESTED";
    monkeyVerdict.className = "monkey-verdict " + (
      String(monkey.verdict || s.monkey_verdict || "").includes("REAL EDGE") ? "monkey-pass" :
      String(monkey.verdict || s.monkey_verdict || "").includes("CONFIRMATION") ? "monkey-warn" :
      "monkey-fail"
    );
  }
  if (monkeyNote) monkeyNote.textContent = monkey.note || "Compares this strategy against randomized monkey baselines.";
  if (monkeyScore) monkeyScore.textContent = monkey.score != null ? Number(monkey.score).toFixed(1) : "—";
  if (monkeyBeats) monkeyBeats.textContent = monkey.strategy_beats_pct != null ? `${Number(monkey.strategy_beats_pct).toFixed(1)}%` : "—";
  if (monkeyEdge) monkeyEdge.textContent = monkey.edge_pct != null ? `${Number(monkey.edge_pct).toFixed(1)}%` : "—";
  if (monkeyMedian) monkeyMedian.textContent = monkey.monkey_median_ret_dd != null ? Number(monkey.monkey_median_ret_dd).toFixed(2) : "—";

  const checks = Array.isArray(detail.checks) ? detail.checks : [];
  document.getElementById("modal-checks").innerHTML = checks.map(c => {
    const status = String(c.status || "warning").toLowerCase();
    const icon = status === "pass" ? "✅" : status === "fail" ? "❌" : "⚠️";
    return `<div class="check-item check-${status}">
      <div class="check-top"><strong>${icon} ${escHtml(c.name || "Check")}</strong><span>${escHtml(String(c.value ?? "—"))}</span></div>
      <p>${escHtml(c.note || "")}</p>
    </div>`;
  }).join("") || `<div class="check-item check-warning"><p>No detailed checks available.</p></div>`;

  renderModalMcChart(mc);
  modal.classList.add("open");
  modal.setAttribute("aria-hidden", "false");
}

function closeStrategyModal() {
  const modal = document.getElementById("strategy-modal");
  if (!modal) return;
  modal.classList.remove("open");
  modal.setAttribute("aria-hidden", "true");
}

function renderModalMcChart(mc) {
  const canvas = document.getElementById("modal-mc-chart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (modalMcChart) modalMcChart.destroy();
  const labels = mc.labels || [];
  const curves = mc.curves || [];
  const datasets = curves.map((curve, idx) => ({
    data: curve,
    borderColor: idx % 5 === 0 ? "rgba(240,192,64,0.55)" : "rgba(0,212,255,0.18)",
    borderWidth: idx % 5 === 0 ? 1.4 : 1,
    pointRadius: 0,
    tension: 0.28,
    fill: false,
  }));
  modalMcChart = new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: {
        x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { display: false } },
        y: { grid: { color: "rgba(255,255,255,0.08)" }, ticks: { color: "#8b949e" } },
      },
    },
  });
}


function sortBy(col) {
  if (sortCol === col) {
    sortDir = sortDir === "asc" ? "desc" : "asc";
  } else {
    sortCol = col;
    sortDir = col === "score" || col === "rank" ? "desc" : "asc";
  }
  renderBulkTable();
}

function filterBy(filter) {
  currentFilter = filter;
  document.querySelectorAll(".filter-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.filter === filter);
  });
  renderBulkTable();
}

function exportAB() {
  const ab = allStrategies.filter(s => s.selected === true);
  if (!ab.length) { alert("No auto-selected strategies to export."); return; }

  const headers = ["Selection_Rank","Strategy_ID","Score","Grade","Net_Profit","Drawdown",
                   "Win_Rate","SQN_Score","Ret_DD_Ratio","Category","Decision","Reason"];
  const lines = [headers.join(",")];
  ab.forEach((s, i) => {
    lines.push([
      i + 1,
      `"${String(s.id).replace(/"/g,'""')}"`,
      s.score.toFixed(1),
      s.grade,
      s.net_profit.toFixed(2),
      s.drawdown.toFixed(2),
      s.win_rate.toFixed(2),
      s.sqn_score.toFixed(2),
      s.ret_dd.toFixed(2),
      `"${String(s.category).replace(/"/g,'""')}"`,
      `"${String(s.decision || "").replace(/"/g,'""')}"`,
      `"${String(s.selection_reason || "").replace(/"/g,'""')}"`
    ].join(","));
  });

  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = "auto_selected_strategies.csv";
  a.click();
  URL.revokeObjectURL(url);
}

// scatter/donut with custom canvas id
function renderScatterCanvas(canvasId, scatter) {
  const datasets = Object.entries(scatter).map(([cat, pts]) => ({
    label: cat,
    data: pts.x.map((x, i) => ({ x, y: pts.y[i] })),
    backgroundColor: hexAlpha(CAT_COLORS[cat], 0.55),
    borderColor:     hexAlpha(CAT_COLORS[cat], 0.8),
    borderWidth: 1,
    pointRadius: 3.5,
    pointHoverRadius: 6,
  }));
  destroyChart(canvasId);
  charts[canvasId] = new Chart(document.getElementById(canvasId), {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top", labels: { color: "#8b949e", padding: 14 } },
        tooltip: { callbacks: { label: ctx => ` SQN ${ctx.parsed.x.toFixed(2)} · P&L $${ctx.parsed.y.toLocaleString()}` } }
      },
      scales: {
        x: { title: { display: true, text: "SQN Score", color: "#8b949e" }, ticks: { color: "#484f58" }, grid: { color: "rgba(255,255,255,0.04)" } },
        y: { title: { display: true, text: "Net Profit ($)", color: "#8b949e" }, ticks: { color: "#484f58", callback: v => "$" + v.toLocaleString() }, grid: { color: "rgba(255,255,255,0.04)" } }
      }
    }
  });
}

function renderDonutCanvas(canvasId, d) {
  destroyChart(canvasId);
  charts[canvasId] = new Chart(document.getElementById(canvasId), {
    type: "doughnut",
    data: {
      labels: d.labels,
      datasets: [{ data: d.values, backgroundColor: [GOLD, CYAN, RED], borderColor: "#161b22", borderWidth: 3, hoverOffset: 8 }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "60%",
      plugins: {
        legend: { position: "bottom", labels: { padding: 16, color: "#c9d1d9" } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.parsed} (${(ctx.parsed/d.values.reduce((a,b)=>a+b,0)*100).toFixed(1)}%)` } }
      },
      animation: { animateRotate: true, duration: 900 },
    }
  });
}

// ── score banner ──────────────────────────────────────────────────────────────
function renderScoreBanner(score, meta) {
  document.getElementById("score-number").textContent = score.composite.toFixed(1);
  const gradeEl = document.getElementById("score-grade");
  gradeEl.textContent = score.grade;
  gradeEl.className = "score-grade grade-" + score.grade;
  document.getElementById("score-file").textContent = meta.filename;
  const expEl = document.getElementById("score-explanation");
  if (expEl) expEl.textContent = score.explanation || "";

  // score ring (doughnut)
  const pct = score.composite / 100;
  const gradeColor = { A: GREEN, B: CYAN, C: GOLD, D: "#e08020", F: RED }[score.grade] || GOLD;
  destroyChart("score-ring");
  charts["score-ring"] = new Chart(document.getElementById("score-ring"), {
    type: "doughnut",
    data: {
      datasets: [{
        data: [pct, 1 - pct],
        backgroundColor: [gradeColor, "rgba(255,255,255,0.05)"],
        borderWidth: 0,
        circumference: 270,
        rotation: 225,
      }]
    },
    options: {
      cutout: "76%",
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      animation: { duration: 1200, easing: "easeInOutQuart" },
    }
  });

  // component bars
  const compLabels = {
    sqn_quality:                  "SQN Quality",
    ret_dd_quality:               "Return / DD Quality",
    drawdown_control:             "Drawdown Control",
    profitability_consistency:    "Profitability Consistency",
    monte_carlo_survival:         "Monte Carlo Survival",
  };
  const compWrap = document.getElementById("score-components");
  compWrap.innerHTML = "";
  for (const [key, label] of Object.entries(compLabels)) {
    const val = score.components[key] ?? 0;
    const row = document.createElement("div");
    row.className = "comp-row";
    row.innerHTML = `
      <span class="comp-label">${label}</span>
      <div class="comp-track"><div class="comp-fill" style="width:0%" data-target="${val}"></div></div>
      <span class="comp-val">${val.toFixed(0)}</span>`;
    compWrap.appendChild(row);
  }
  // animate bars after paint
  requestAnimationFrame(() => {
    document.querySelectorAll(".comp-fill").forEach(el => {
      el.style.width = el.dataset.target + "%";
    });
  });

  // meta
  const { avg_sqn, positive_pct, median_ret_dd, survival_pct } = score.stats;
  const metaItems = [
    [avg_sqn.toFixed(2),    "Avg SQN Score"],
    [positive_pct.toFixed(0) + "%", "Profitable Paths"],
    [median_ret_dd.toFixed(2), "Median Ret/DD"],
    [(survival_pct ?? positive_pct).toFixed(0) + "%", "Monte Carlo Survival"],
  ];
  const metaWrap = document.getElementById("score-meta");
  metaWrap.innerHTML = metaItems.map(([v, k]) =>
    `<div class="meta-item"><div class="meta-val">${v}</div><div class="meta-key">${k}</div></div>`
  ).join("");
}

// ── equity fan chart ──────────────────────────────────────────────────────────
function renderEquityChart(eq) {
  destroyChart("chart-equity");
  charts["chart-equity"] = new Chart(document.getElementById("chart-equity"), {
    type: "line",
    data: { labels: eq.labels, datasets: eq.datasets },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          labels: {
            filter: item => item.text !== undefined,
            color: "#8b949e",
            boxWidth: 20,
            padding: 14,
          }
        },
        tooltip: { enabled: false },
      },
      scales: {
        x: {
          ticks: { maxTicksLimit: 10, color: "#484f58" },
          grid:  { color: "rgba(255,255,255,0.04)" },
          title: { display: true, text: "Trade #", color: "#8b949e" },
        },
        y: {
          ticks: { color: "#484f58", callback: v => "$" + v.toLocaleString() },
          grid:  { color: "rgba(255,255,255,0.04)" },
          title: { display: true, text: "Cumulative P&L", color: "#8b949e" },
        }
      }
    }
  });
}

// ── donut ──────────────────────────────────────────────────────────────────────
function renderDonut(d) {
  destroyChart("chart-donut");
  charts["chart-donut"] = new Chart(document.getElementById("chart-donut"), {
    type: "doughnut",
    data: {
      labels: d.labels,
      datasets: [{
        data: d.values,
        backgroundColor: [GOLD, CYAN, RED],
        borderColor: "#161b22",
        borderWidth: 3,
        hoverOffset: 8,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "60%",
      plugins: {
        legend: { position: "bottom", labels: { padding: 16, color: "#c9d1d9" } },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${ctx.parsed} (${(ctx.parsed/d.values.reduce((a,b)=>a+b,0)*100).toFixed(1)}%)`
          }
        }
      },
      animation: { animateRotate: true, duration: 900 },
    }
  });
}

// ── histogram ─────────────────────────────────────────────────────────────────
function renderHistogram(canvasId, h, label, color) {
  destroyChart(canvasId);
  charts[canvasId] = new Chart(document.getElementById(canvasId), {
    type: "bar",
    data: {
      labels: h.x.map(v => v.toFixed(1)),
      datasets: [{
        label,
        data: h.y,
        backgroundColor: hexAlpha(color, 0.55),
        borderColor: color,
        borderWidth: 1,
        borderRadius: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { maxTicksLimit: 7, color: "#484f58" }, grid: { display: false } },
        y: { ticks: { color: "#484f58" }, grid: { color: "rgba(255,255,255,0.04)" } },
      }
    }
  });
}

// ── scatter ───────────────────────────────────────────────────────────────────
function renderScatter(scatter) {
  const datasets = Object.entries(scatter).map(([cat, pts]) => ({
    label: cat,
    data: pts.x.map((x, i) => ({ x, y: pts.y[i] })),
    backgroundColor: hexAlpha(CAT_COLORS[cat], 0.55),
    borderColor:     hexAlpha(CAT_COLORS[cat], 0.8),
    borderWidth: 1,
    pointRadius: 3.5,
    pointHoverRadius: 6,
  }));

  destroyChart("chart-scatter");
  charts["chart-scatter"] = new Chart(document.getElementById("chart-scatter"), {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top", labels: { color: "#8b949e", padding: 14 } },
        tooltip: {
          callbacks: {
            label: ctx => ` SQN ${ctx.parsed.x.toFixed(2)} · P&L $${ctx.parsed.y.toLocaleString()}`
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: "SQN Score", color: "#8b949e" },
          ticks: { color: "#484f58" },
          grid:  { color: "rgba(255,255,255,0.04)" },
        },
        y: {
          title: { display: true, text: "Net Profit ($)", color: "#8b949e" },
          ticks: { color: "#484f58", callback: v => "$" + v.toLocaleString() },
          grid:  { color: "rgba(255,255,255,0.04)" },
        }
      }
    }
  });
}

// ── comparison table ──────────────────────────────────────────────────────────
function renderComparison(comp) {
  const labels = {
    Net_Profit:   "Net Profit ($)",
    Drawdown:     "Drawdown ($)",
    Win_Rate:     "Win Rate (%)",
    SQN_Score:    "SQN Score",
    Ret_DD_Ratio: "Ret/DD Ratio",
  };
  const tbody = document.getElementById("comparison-body");
  tbody.innerHTML = "";
  for (const [col, row] of Object.entries(comp)) {
    const hasInput = row.input_mean !== null && row.input_mean !== undefined;
    const drift = hasInput
      ? Math.abs(row.synth_mean - row.input_mean) / (Math.abs(row.input_mean) + 1e-6) * 100
      : null;
    const driftClass = drift === null ? "" : drift < 5 ? "drift-low" : drift < 15 ? "drift-mid" : "drift-high";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${labels[col] || col}</td>
      <td>${hasInput ? fmt(row.input_mean) : "—"}</td>
      <td>${hasInput ? fmt(row.input_std)  : "—"}</td>
      <td>${fmt(row.synth_mean)}</td>
      <td>${fmt(row.synth_std)}</td>
      <td class="${driftClass}">${drift !== null ? drift.toFixed(1) + "%" : "—"}</td>`;
    tbody.appendChild(tr);
  }
}

// ── helpers ───────────────────────────────────────────────────────────────────
function fmtNum(n) {
  if (n === null || n === undefined || isNaN(n)) return "—";
  if (Math.abs(n) >= 1000) return "$" + n.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return "$" + n.toFixed(2);
}

function escHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function fmt(n) {
  if (Math.abs(n) >= 1000) return "$" + n.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return n.toFixed(2);
}

function hexAlpha(hex, alpha) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function destroyChart(id) {
  if (charts[id]) { charts[id].destroy(); delete charts[id]; }
}

function showProgress(pct, label) {
  progressWrap.style.display = "block";
  progressFill.style.width   = pct + "%";
  progressLabel.textContent  = label;
}

function hideProgress() { progressWrap.style.display = "none"; }

function showError(msg) {
  errorBox.style.display = "flex";
  errorMsg.textContent = msg;
}

function hideError() { errorBox.style.display = "none"; }

function tick(ms) { return new Promise(r => setTimeout(r, ms)); }

function resetUI() {
  busy                        = false;
  results.style.display       = "none";
  uploadSection.style.display = "flex";
  fileInput.value             = "";
  allStrategies               = [];
  currentFilter               = "all";
  document.getElementById("bulk-view").style.display   = "none";
  document.getElementById("single-view").style.display = "none";
  uploadSection.scrollIntoView({ behavior: "smooth" });
  Object.keys(charts).forEach(destroyChart);
}

// ── Modal close fix ───────────────────────────────────────────────────────────
// Works even if the modal HTML is loaded after this script.
document.addEventListener("click", function (e) {
  if (
    e.target.closest("#modal-close") ||
    e.target.closest(".modal-close") ||
    e.target.closest("#modal-backdrop") ||
    e.target.closest(".modal-backdrop")
  ) {
    e.preventDefault();
    closeStrategyModal();
  }
});

document.addEventListener("keydown", function (e) {
  if (e.key === "Escape") {
    closeStrategyModal();
  }
});
