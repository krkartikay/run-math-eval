#!/usr/bin/env python3
import argparse
import json
import sqlite3
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_DB_PATH = Path(__file__).resolve().parent / "traces.sqlite3"


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trace Viewer</title>
  <style>
    :root {
      --bg: #f5efe6;
      --panel: rgba(255, 252, 246, 0.92);
      --panel-strong: #fffaf1;
      --ink: #211c18;
      --muted: #6d655c;
      --line: rgba(33, 28, 24, 0.12);
      --accent: #b85c38;
      --accent-2: #275d63;
      --shadow: 0 20px 50px rgba(43, 30, 19, 0.12);
      --radius: 22px;
      --mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      --sans: "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(184, 92, 56, 0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(39, 93, 99, 0.15), transparent 26%),
        linear-gradient(180deg, #f7f1e8 0%, #f2ebe1 45%, #efe7dc 100%);
      min-height: 100vh;
    }

    .shell {
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 18px;
      padding: 22px;
      min-height: 100vh;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }

    .sidebar {
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .hero, .trace-header {
      padding: 22px 22px 14px;
      border-bottom: 1px solid var(--line);
    }

    h1, h2, h3, p { margin: 0; }

    h1 {
      font-size: 30px;
      line-height: 1;
      letter-spacing: -0.03em;
    }

    .subtle {
      color: var(--muted);
      margin-top: 10px;
      line-height: 1.45;
    }

    .stats {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 18px;
    }

    .stat {
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(255,255,255,0.55);
      border: 1px solid rgba(33, 28, 24, 0.08);
    }

    .stat strong {
      display: block;
      font-size: 22px;
      letter-spacing: -0.03em;
    }

    .stat span {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .toolbar {
      display: flex;
      gap: 10px;
      padding: 14px 22px;
      border-bottom: 1px solid var(--line);
    }

    input, select {
      width: 100%;
      border: 1px solid rgba(33, 28, 24, 0.12);
      background: rgba(255,255,255,0.85);
      border-radius: 12px;
      padding: 11px 13px;
      color: var(--ink);
      outline: none;
    }

    input:focus, select:focus {
      border-color: rgba(184, 92, 56, 0.45);
      box-shadow: 0 0 0 4px rgba(184, 92, 56, 0.09);
    }

    .trace-list {
      overflow: auto;
      padding: 10px;
    }

    .trace-item {
      border-radius: 18px;
      padding: 14px;
      border: 1px solid transparent;
      cursor: pointer;
      transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
      margin-bottom: 8px;
      background: transparent;
    }

    .trace-item:hover, .trace-item.active {
      background: rgba(255,255,255,0.72);
      border-color: rgba(184, 92, 56, 0.18);
      transform: translateY(-1px);
    }

    .trace-item .title {
      font-weight: 700;
      letter-spacing: -0.02em;
      font-size: 14px;
      margin-bottom: 6px;
    }

    .trace-item .prompt {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
      margin-bottom: 10px;
    }

    .meta-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .pill {
      border-radius: 999px;
      padding: 5px 9px;
      background: rgba(39, 93, 99, 0.1);
      color: var(--accent-2);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }

    .main {
      display: grid;
      grid-template-rows: auto auto 1fr;
      min-width: 0;
    }

    .trace-title {
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 12px;
    }

    .trace-title h2 {
      font-size: 28px;
      letter-spacing: -0.03em;
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      padding: 16px 22px 22px;
    }

    .summary-card {
      background: rgba(255,255,255,0.56);
      border: 1px solid rgba(33, 28, 24, 0.08);
      border-radius: 18px;
      padding: 14px;
    }

    .summary-card strong {
      display: block;
      font-size: 24px;
      margin-top: 6px;
    }

    .content {
      display: grid;
      grid-template-columns: minmax(360px, 0.9fr) minmax(320px, 1.1fr);
      gap: 18px;
      min-height: 0;
      padding: 0 0 22px;
    }

    .graph-panel, .detail-panel {
      margin: 0 22px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    .section-head {
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .graph-wrap {
      padding: 18px;
      overflow: auto;
      min-height: 380px;
      background:
        linear-gradient(rgba(33, 28, 24, 0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(33, 28, 24, 0.04) 1px, transparent 1px);
      background-size: 28px 28px;
    }

    svg {
      width: 100%;
      min-height: 420px;
      overflow: visible;
    }

    .detail-body {
      overflow: auto;
      padding: 18px;
      display: grid;
      gap: 16px;
    }

    .detail-card {
      border-radius: 18px;
      background: rgba(255,255,255,0.62);
      border: 1px solid rgba(33, 28, 24, 0.08);
      overflow: hidden;
    }

    .detail-card header {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.5);
    }

    .detail-card .body {
      padding: 16px;
    }

    .kv {
      display: grid;
      grid-template-columns: 130px 1fr;
      gap: 10px;
      font-size: 13px;
      line-height: 1.45;
      margin-bottom: 8px;
    }

    .kv .key {
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-size: 11px;
      font-weight: 700;
    }

    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.55;
      background: #1d1a17;
      color: #f8efe2;
      border-radius: 14px;
      padding: 14px;
      overflow: auto;
    }

    .empty {
      padding: 32px 22px;
      color: var(--muted);
    }

    .node-label {
      font-size: 12px;
      font-weight: 700;
      fill: #211c18;
    }

    .edge-label {
      font-size: 10px;
      fill: #6d655c;
    }

    .node {
      cursor: pointer;
    }

    .node.selected rect {
      stroke: #b85c38;
      stroke-width: 3;
    }

    .footer-note {
      padding: 0 22px 22px;
      color: var(--muted);
      font-size: 12px;
    }

    @media (max-width: 1100px) {
      .shell, .content {
        grid-template-columns: 1fr;
      }

      .main {
        min-height: auto;
      }

      .summary-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }

    @media (max-width: 720px) {
      .shell {
        padding: 14px;
      }

      .stats, .summary-grid {
        grid-template-columns: 1fr;
      }

      .toolbar {
        flex-direction: column;
      }

      .graph-panel, .detail-panel {
        margin: 0 14px;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="panel sidebar">
      <div class="hero">
        <h1>Trace Atlas</h1>
        <p class="subtle">A single-file viewer for SQLite traces. Filter by task or run, then inspect the execution graph and raw payloads.</p>
        <div class="stats">
          <div class="stat"><span>Traces</span><strong id="totalTraces">0</strong></div>
          <div class="stat"><span>Nodes</span><strong id="totalNodes">0</strong></div>
          <div class="stat"><span>Edges</span><strong id="totalEdges">0</strong></div>
        </div>
      </div>
      <div class="toolbar">
        <input id="searchBox" placeholder="Search prompt, trace, task">
      </div>
      <div class="toolbar" style="padding-top:0">
        <select id="runFilter"><option value="">All runs</option></select>
      </div>
      <div id="traceList" class="trace-list"></div>
    </aside>
    <main class="main panel">
      <div class="trace-header">
        <div class="trace-title">
          <div>
            <h2 id="traceTitle">Select a trace</h2>
            <p class="subtle" id="traceSubtitle">The graph and details will appear here.</p>
          </div>
          <div class="meta-row" id="traceMeta"></div>
        </div>
      </div>
      <div class="summary-grid" id="summaryGrid"></div>
      <div class="content">
        <section class="panel graph-panel">
          <div class="section-head">
            <h3>Graph</h3>
            <span class="subtle" id="graphCaption">Node flow</span>
          </div>
          <div class="graph-wrap">
            <svg id="graphSvg" viewBox="0 0 1200 720" preserveAspectRatio="xMinYMin meet"></svg>
          </div>
        </section>
        <section class="panel detail-panel">
          <div class="section-head">
            <h3>Details</h3>
            <span class="subtle" id="detailCaption">Select a node</span>
          </div>
          <div class="detail-body" id="detailBody">
            <div class="empty">Trace metadata, eval target, and node details will show here.</div>
          </div>
        </section>
      </div>
      <div class="footer-note">Tip: response nodes are dark, tool outputs are light, and edges are labeled with the recorded relation.</div>
    </main>
  </div>
  <script>
    const state = {
      summary: null,
      traces: [],
      currentTraceId: null,
      currentTrace: null,
      selectedNodeId: null,
    };

    const el = (id) => document.getElementById(id);

    async function loadSummary() {
      const response = await fetch('/api/summary');
      state.summary = await response.json();
      state.traces = state.summary.traces;
      renderSidebar();
      renderRunOptions();
      renderTotals();
      if (state.traces.length) {
        await selectTrace(state.traces[0].id);
      }
    }

    function renderTotals() {
      el('totalTraces').textContent = state.summary.counts.traces.toLocaleString();
      el('totalNodes').textContent = state.summary.counts.nodes.toLocaleString();
      el('totalEdges').textContent = state.summary.counts.edges.toLocaleString();
    }

    function renderRunOptions() {
      const select = el('runFilter');
      const runs = state.summary.runs || [];
      select.innerHTML = '<option value="">All runs</option>' + runs.map(
        (run) => `<option value="${escapeHtml(run.eval_run_id)}">${escapeHtml(run.eval_run_id)} (${run.trace_count})</option>`
      ).join('');
    }

    function getFilteredTraces() {
      const query = el('searchBox').value.trim().toLowerCase();
      const runFilter = el('runFilter').value;
      return state.traces.filter((trace) => {
        if (runFilter && trace.eval_run_id !== runFilter) return false;
        if (!query) return true;
        const haystack = [
          trace.id,
          trace.eval_run_id,
          trace.task_name,
          trace.prompt_preview,
          trace.expected_answer,
        ].filter(Boolean).join(' ').toLowerCase();
        return haystack.includes(query);
      });
    }

    function renderSidebar() {
      const traces = getFilteredTraces();
      const container = el('traceList');
      if (!traces.length) {
        container.innerHTML = '<div class="empty">No traces match the current filter.</div>';
        return;
      }
      container.innerHTML = traces.map((trace) => {
        const active = trace.id === state.currentTraceId ? 'active' : '';
        return `
          <div class="trace-item ${active}" data-trace-id="${escapeHtml(trace.id)}">
            <div class="title">${escapeHtml(trace.task_name || shortId(trace.id))}</div>
            <div class="prompt">${escapeHtml(trace.prompt_preview || 'No eval target prompt recorded.')}</div>
            <div class="meta-row">
              <span class="pill">${trace.node_count} nodes</span>
              <span class="pill">${trace.edge_count} edges</span>
              ${trace.eval_run_id ? `<span class="pill">${escapeHtml(shortId(trace.eval_run_id))}</span>` : ''}
            </div>
          </div>
        `;
      }).join('');
      for (const item of container.querySelectorAll('.trace-item')) {
        item.addEventListener('click', () => selectTrace(item.dataset.traceId));
      }
    }

    async function selectTrace(traceId) {
      state.currentTraceId = traceId;
      state.selectedNodeId = null;
      renderSidebar();
      const response = await fetch(`/api/trace/${encodeURIComponent(traceId)}`);
      state.currentTrace = await response.json();
      const firstNode = state.currentTrace.nodes[0];
      state.selectedNodeId = firstNode ? firstNode.id : null;
      renderTrace();
    }

    function renderTrace() {
      const trace = state.currentTrace;
      if (!trace) return;
      el('traceTitle').textContent = trace.eval_target?.task_name || shortId(trace.id);
      el('traceSubtitle').textContent = trace.eval_target?.prompt_text?.trim() || 'No eval target prompt recorded.';
      el('traceMeta').innerHTML = [
        trace.eval_run_id ? `<span class="pill">${escapeHtml(shortId(trace.eval_run_id))}</span>` : '',
        `<span class="pill">${trace.nodes.length} nodes</span>`,
        `<span class="pill">${trace.edges.length} edges</span>`,
        `<span class="pill">${escapeHtml(trace.created_at || '')}</span>`,
      ].join('');
      renderSummaryGrid(trace);
      renderGraph(trace);
      renderDetails(trace);
    }

    function renderSummaryGrid(trace) {
      const counts = trace.kind_counts || {};
      const cards = [
        ['Problem nodes', counts.problem || 0],
        ['Response nodes', counts.response || 0],
        ['Tool outputs', counts.tool_output || 0],
        ['Expected answer', trace.eval_target?.expected_answer || 'N/A'],
      ];
      el('summaryGrid').innerHTML = cards.map(([label, value]) => `
        <div class="summary-card">
          <div class="key">${escapeHtml(label)}</div>
          <strong>${escapeHtml(String(value))}</strong>
        </div>
      `).join('');
    }

    function renderGraph(trace) {
      const svg = el('graphSvg');
      const layerOrder = ['problem', 'response', 'tool_output'];
      const nodes = [...trace.nodes];
      const indexById = Object.fromEntries(nodes.map((node, index) => [node.id, index]));
      const incoming = Object.fromEntries(nodes.map((node) => [node.id, 0]));
      for (const edge of trace.edges) {
        incoming[edge.target_node_id] = (incoming[edge.target_node_id] || 0) + 1;
      }

      const columns = new Map();
      for (const kind of layerOrder) columns.set(kind, []);
      for (const node of nodes) {
        if (!columns.has(node.kind)) columns.set(node.kind, []);
        columns.get(node.kind).push(node);
      }
      for (const column of columns.values()) {
        column.sort((a, b) => {
          const ia = incoming[a.id] || 0;
          const ib = incoming[b.id] || 0;
          if (ia !== ib) return ia - ib;
          return (a.created_at || '').localeCompare(b.created_at || '');
        });
      }

      const orderedKinds = [...columns.keys()];
      const xStep = 330;
      const yStep = 110;
      const nodeW = 220;
      const nodeH = 64;
      const positions = {};
      let maxRows = 1;
      orderedKinds.forEach((kind, colIndex) => {
        const group = columns.get(kind) || [];
        maxRows = Math.max(maxRows, group.length);
        group.forEach((node, rowIndex) => {
          positions[node.id] = {
            x: 70 + colIndex * xStep,
            y: 60 + rowIndex * yStep,
          };
        });
      });

      const width = 160 + Math.max(1, orderedKinds.length) * xStep;
      const height = 160 + maxRows * yStep;
      svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

      const defs = `
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L10,3 L0,6 z" fill="#6d655c"></path>
          </marker>
        </defs>
      `;

      const edges = trace.edges.map((edge) => {
        const source = positions[edge.source_node_id];
        const target = positions[edge.target_node_id];
        if (!source || !target) return '';
        const x1 = source.x + nodeW;
        const y1 = source.y + nodeH / 2;
        const x2 = target.x;
        const y2 = target.y + nodeH / 2;
        const mx = (x1 + x2) / 2;
        const curve = Math.max(40, Math.abs(x2 - x1) / 2);
        return `
          <path d="M ${x1} ${y1} C ${x1 + curve} ${y1}, ${x2 - curve} ${y2}, ${x2} ${y2}"
            fill="none" stroke="#6d655c" stroke-width="1.5" marker-end="url(#arrow)" opacity="0.75"></path>
          <text class="edge-label" x="${mx}" y="${(y1 + y2) / 2 - 8}" text-anchor="middle">${escapeHtml(edge.relation)}</text>
        `;
      }).join('');

      const nodesMarkup = nodes.map((node) => {
        const pos = positions[node.id];
        const selected = node.id === state.selectedNodeId ? 'selected' : '';
        const fill = node.color === 'black' ? '#2a241f' : '#fff8ef';
        const stroke = node.color === 'black' ? '#2a241f' : '#d9cbbb';
        const textFill = node.color === 'black' ? '#fff5ea' : '#211c18';
        const label = truncate(`${node.kind} · ${node.id.slice(-6)}`, 22);
        const preview = truncate(node.preview || '', 44);
        return `
          <g class="node ${selected}" data-node-id="${escapeHtml(node.id)}">
            <rect x="${pos.x}" y="${pos.y}" rx="18" ry="18" width="${nodeW}" height="${nodeH}"
              fill="${fill}" stroke="${stroke}" stroke-width="2"></rect>
            <text class="node-label" x="${pos.x + 14}" y="${pos.y + 24}" fill="${textFill}">${escapeHtml(label)}</text>
            <text x="${pos.x + 14}" y="${pos.y + 45}" fill="${textFill}" opacity="0.75" font-size="11">${escapeHtml(preview)}</text>
          </g>
        `;
      }).join('');

      svg.innerHTML = defs + edges + nodesMarkup;
      for (const nodeEl of svg.querySelectorAll('.node')) {
        nodeEl.addEventListener('click', () => {
          state.selectedNodeId = nodeEl.dataset.nodeId;
          renderGraph(trace);
          renderDetails(trace);
        });
      }
      el('graphCaption').textContent = `${trace.edges.length} connections across ${orderedKinds.length} columns`;
    }

    function renderDetails(trace) {
      const node = trace.nodes.find((entry) => entry.id === state.selectedNodeId) || null;
      const parts = [];
      parts.push(renderTraceCard(trace));
      if (trace.eval_target) parts.push(renderEvalTargetCard(trace.eval_target));
      if (node) parts.push(renderNodeCard(node));
      el('detailBody').innerHTML = parts.join('');
      el('detailCaption').textContent = node ? `${node.kind} · ${shortId(node.id)}` : 'Select a node';
    }

    function renderTraceCard(trace) {
      return `
        <section class="detail-card">
          <header><strong>Trace</strong></header>
          <div class="body">
            ${kv('Trace ID', trace.id)}
            ${kv('Eval run', trace.eval_run_id || 'None')}
            ${kv('Created', trace.created_at || 'Unknown')}
            ${kv('Metadata', '')}
            <pre>${escapeHtml(formatJson(trace.metadata))}</pre>
          </div>
        </section>
      `;
    }

    function renderEvalTargetCard(target) {
      return `
        <section class="detail-card">
          <header><strong>Eval Target</strong></header>
          <div class="body">
            ${kv('Task', target.task_name || 'None')}
            ${kv('Doc ID', target.doc_id == null ? 'None' : String(target.doc_id))}
            ${kv('Expected', target.expected_answer || 'None')}
            ${kv('Prompt', '')}
            <pre>${escapeHtml(target.prompt_text || '')}</pre>
            ${kv('Doc JSON', '')}
            <pre>${escapeHtml(formatJson(target.doc))}</pre>
          </div>
        </section>
      `;
    }

    function renderNodeCard(node) {
      return `
        <section class="detail-card">
          <header><strong>Node</strong></header>
          <div class="body">
            ${kv('Node ID', node.id)}
            ${kv('Kind', node.kind)}
            ${kv('Color', node.color)}
            ${kv('Created', node.created_at || 'Unknown')}
            ${kv('Content', '')}
            <pre>${escapeHtml(formatJson(node.content))}</pre>
            ${kv('Metadata', '')}
            <pre>${escapeHtml(formatJson(node.metadata))}</pre>
          </div>
        </section>
      `;
    }

    function kv(key, value) {
      return `<div class="kv"><div class="key">${escapeHtml(key)}</div><div>${escapeHtml(value)}</div></div>`;
    }

    function shortId(value) {
      if (!value) return '';
      return value.length > 18 ? `${value.slice(0, 10)}…${value.slice(-6)}` : value;
    }

    function truncate(value, limit) {
      if (!value) return '';
      return value.length > limit ? `${value.slice(0, limit - 1)}…` : value;
    }

    function formatJson(value) {
      return JSON.stringify(value, null, 2);
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }

    el('searchBox').addEventListener('input', renderSidebar);
    el('runFilter').addEventListener('change', renderSidebar);
    loadSummary().catch((error) => {
      el('traceList').innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`;
    });
  </script>
</body>
</html>
"""


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def parse_json(value: str | None):
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def preview_content(value, limit: int = 120) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=True)
    compact = " ".join(text.split())
    return compact[: limit - 1] + "…" if len(compact) > limit else compact


def fetch_summary(db_path: Path) -> dict:
    with connect(db_path) as conn:
        counts = {
            "traces": conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0],
            "nodes": conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
            "edges": conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
        }
        trace_rows = conn.execute(
            """
            SELECT
                t.id,
                t.eval_run_id,
                t.created_at,
                t.metadata_json,
                COALESCE(COUNT(DISTINCT n.id), 0) AS node_count,
                COALESCE(COUNT(DISTINCT e.id), 0) AS edge_count,
                et.task_name,
                et.expected_answer,
                et.prompt_text
            FROM traces AS t
            LEFT JOIN nodes AS n ON n.trace_id = t.id
            LEFT JOIN edges AS e ON e.trace_id = t.id
            LEFT JOIN eval_targets AS et
              ON et.trace_id = t.id
             AND et.id = (
                SELECT et2.id
                FROM eval_targets AS et2
                WHERE et2.trace_id = t.id
                ORDER BY et2.created_at DESC, et2.id DESC
                LIMIT 1
             )
            GROUP BY t.id
            ORDER BY t.created_at DESC, t.id DESC
            """
        ).fetchall()
        runs = conn.execute(
            """
            SELECT eval_run_id, COUNT(*) AS trace_count
            FROM traces
            WHERE eval_run_id IS NOT NULL
            GROUP BY eval_run_id
            ORDER BY MAX(created_at) DESC
            """
        ).fetchall()

    traces = []
    for row in trace_rows:
        traces.append(
            {
                "id": row["id"],
                "eval_run_id": row["eval_run_id"],
                "created_at": row["created_at"],
                "task_name": row["task_name"],
                "expected_answer": row["expected_answer"],
                "prompt_preview": preview_content(row["prompt_text"], limit=170),
                "node_count": row["node_count"],
                "edge_count": row["edge_count"],
                "metadata": parse_json(row["metadata_json"]),
            }
        )
    return {
        "counts": counts,
        "traces": traces,
        "runs": [dict(row) for row in runs],
    }


def fetch_trace(db_path: Path, trace_id: str) -> dict | None:
    with connect(db_path) as conn:
        trace_row = conn.execute(
            "SELECT id, eval_run_id, created_at, metadata_json FROM traces WHERE id = ?",
            (trace_id,),
        ).fetchone()
        if trace_row is None:
            return None

        node_rows = conn.execute(
            """
            SELECT id, kind, color, content_json, metadata_json, created_at
            FROM nodes
            WHERE trace_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (trace_id,),
        ).fetchall()
        edge_rows = conn.execute(
            """
            SELECT id, source_node_id, target_node_id, relation, metadata_json, created_at
            FROM edges
            WHERE trace_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (trace_id,),
        ).fetchall()
        eval_target_row = conn.execute(
            """
            SELECT task_name, doc_id, prompt_text, expected_answer, doc_json, metadata_json, created_at
            FROM eval_targets
            WHERE trace_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (trace_id,),
        ).fetchone()

    nodes = []
    kind_counts = Counter()
    for row in node_rows:
        content = parse_json(row["content_json"])
        metadata = parse_json(row["metadata_json"])
        kind_counts[row["kind"]] += 1
        nodes.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "color": row["color"],
                "content": content,
                "metadata": metadata,
                "created_at": row["created_at"],
                "preview": preview_content(content, limit=52),
            }
        )

    edges = [
        {
            "id": row["id"],
            "source_node_id": row["source_node_id"],
            "target_node_id": row["target_node_id"],
            "relation": row["relation"],
            "metadata": parse_json(row["metadata_json"]),
            "created_at": row["created_at"],
        }
        for row in edge_rows
    ]

    eval_target = None
    if eval_target_row is not None:
        eval_target = {
            "task_name": eval_target_row["task_name"],
            "doc_id": eval_target_row["doc_id"],
            "prompt_text": eval_target_row["prompt_text"],
            "expected_answer": eval_target_row["expected_answer"],
            "doc": parse_json(eval_target_row["doc_json"]),
            "metadata": parse_json(eval_target_row["metadata_json"]),
            "created_at": eval_target_row["created_at"],
        }

    return {
        "id": trace_row["id"],
        "eval_run_id": trace_row["eval_run_id"],
        "created_at": trace_row["created_at"],
        "metadata": parse_json(trace_row["metadata_json"]),
        "kind_counts": dict(kind_counts),
        "nodes": nodes,
        "edges": edges,
        "eval_target": eval_target,
    }


class TraceViewerHandler(BaseHTTPRequestHandler):
    db_path: Path

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML)
            return
        if parsed.path == "/api/summary":
            self._send_json(fetch_summary(self.db_path))
            return
        if parsed.path.startswith("/api/trace/"):
            trace_id = parsed.path.removeprefix("/api/trace/")
            payload = fetch_trace(self.db_path, trace_id)
            if payload is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Trace not found")
                return
            self._send_json(payload)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args) -> None:
        return

    def _send_html(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def build_handler(db_path: Path):
    class Handler(TraceViewerHandler):
        pass

    Handler.db_path = db_path
    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a single-file web UI for exploring trace data in SQLite."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite file")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = args.db.resolve()
    if not db_path.exists():
        raise SystemExit(f"SQLite file not found: {db_path}")
    handler = build_handler(db_path)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    location = f"http://{args.host}:{args.port}"
    print(f"Serving trace viewer for {db_path} at {location}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
