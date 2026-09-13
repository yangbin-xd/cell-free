/* Interactive network map for the cell-free digital twin.
 *
 * Runs inside a Streamlit custom-component iframe.  Python sends the Plotly
 * figure built by app._net_fig plus plain-list geometry (positions, AP
 * locations, association A, power P, the 2500 ray-traced grid points).  This
 * script owns UE positions while the user drags them or Brownian motion is
 * on, redraws the position-dependent traces locally, and reports positions
 * back to Python at a throttled, ack-gated rate so the GNN / true-value bars
 * can follow.
 *
 * Authority protocol: Python bumps `epoch` whenever it sets positions itself
 * (sample load, reset, agent move_ue).  We adopt Python's positions only when
 * the epoch changes; otherwise our animated positions win.
 *
 * Region rule (mirrors dyn_users.clamp_to_region / brownian_step): a UE may
 * only be within r_max metres of some grid point.
 */
(function () {
  "use strict";

  const gd = document.getElementById("gd");

  // ── state ────────────────────────────────────────────────────────────────
  let epoch = null, seq = 0;
  let pos = [], apLoc = [], A = [], P = [], grid = [];
  let added = [], removed = [], changed = [];
  let rMax = 3.0, sigma = 1.0, tickMs = 100, emitMs = 300;
  let moving = false, timer = null;
  let fig = null, height = 540, selUe = null, selAp = null;
  let dragIdx = null, downPx = null, dragMoved = false;
  let lastEmit = 0, emitAt = 0, lastRtt = 0, renderSinceEmit = true;
  let plotted = false;

  // ── Streamlit iframe protocol ────────────────────────────────────────────
  function send(type, payload) {
    window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type: type }, payload || {}), "*");
  }
  // index.html may already have sent componentReady (early, before Plotly
  // loaded) and parked the first render args in window.__ueMapEarly.pending;
  // from here on this listener owns render messages.
  const early = window.__ueMapEarly || null;
  window.__ueMapReady = true;
  window.addEventListener("message", function (ev) {
    if (ev.data && ev.data.type === "streamlit:render") onRender(ev.data.args || {});
  });
  if (!early) send("streamlit:componentReady", { apiVersion: 1 });

  // ── geometry helpers (must match dyn_users.py) ───────────────────────────
  function nearestDist(x, y) {
    let best = Infinity;
    for (let i = 0; i < grid.length; i++) {
      const dx = grid[i][0] - x, dy = grid[i][1] - y;
      const d2 = dx * dx + dy * dy;
      if (d2 < best) best = d2;
    }
    return Math.sqrt(best);
  }
  function nearestPoint(x, y) {
    let best = Infinity, bi = 0;
    for (let i = 0; i < grid.length; i++) {
      const dx = grid[i][0] - x, dy = grid[i][1] - y;
      const d2 = dx * dx + dy * dy;
      if (d2 < best) { best = d2; bi = i; }
    }
    return [grid[bi][0], grid[bi][1], Math.sqrt(best)];
  }
  function clampToRegion(x, y) {
    if (!grid.length) return [x, y];
    const np = nearestPoint(x, y);
    if (np[2] <= rMax) return [x, y];
    const s = rMax / np[2];
    return [np[0] + (x - np[0]) * s, np[1] + (y - np[1]) * s];
  }
  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  // ── trace rebuild from local geometry ────────────────────────────────────
  function key(l, k) { return l + "," + k; }
  function toSet(list) { const s = new Set(); (list || []).forEach(p => s.add(key(p[0], p[1]))); return s; }

  function buildData() {
    const data = JSON.parse(JSON.stringify(fig.data));
    const addS = toSet(added), remS = toSet(removed), chgS = toSet(changed);
    const ex = [], ey = [], lx = [], ly = [], lt = [];
    const ax = [], ay = [], rx = [], ry = [], cx = [], cy = [];
    const apN = apLoc.length, ueN = pos.length;
    for (let l = 0; l < apN; l++) {
      for (let k = 0; k < ueN; k++) {
        const on = A[l] && A[l][k] > 0.5;
        const x1 = apLoc[l][0], y1 = apLoc[l][1], x2 = pos[k][0], y2 = pos[k][1];
        if (on) {
          if (addS.has(key(l, k)))      { ax.push(x1, x2, null); ay.push(y1, y2, null); }
          else if (chgS.has(key(l, k))) { cx.push(x1, x2, null); cy.push(y1, y2, null); }
          else                          { ex.push(x1, x2, null); ey.push(y1, y2, null); }
          lx.push((x1 + x2) / 2); ly.push((y1 + y2) / 2);
          lt.push((P[l] ? P[l][k] : 0).toFixed(2));
        } else if (remS.has(key(l, k))) {
          rx.push(x1, x2, null); ry.push(y1, y2, null);
        }
      }
    }
    const setXY = (i, x, y) => { if (data[i]) { data[i].x = x; data[i].y = y; } };
    setXY(0, ex, ey);
    setXY(1, lx, ly); if (data[1]) data[1].text = lt;
    setXY(2, apLoc.map(p => p[0]), apLoc.map(p => p[1]));
    setXY(3, pos.map(p => p[0]), pos.map(p => p[1]));
    setXY(5, ax, ay);
    setXY(6, rx, ry);
    setXY(7, cx, cy);
    // Trace 8 – the measured region (2500 ray-traced points), drawn once per render
    data.push({
      type: "scatter", mode: "markers",
      x: grid.map(p => p[0]), y: grid.map(p => p[1]),
      marker: { size: 2, color: "rgba(120,140,200,0.18)" },
      hoverinfo: "skip", showlegend: false, name: "measured",
    });
    // put the region behind everything else
    data.unshift(data.pop());
    return data;
  }

  function buildLayout() {
    const layout = JSON.parse(JSON.stringify(fig.layout || {}));
    layout.dragmode = false;
    layout.height = height;
    layout.margin = layout.margin || {};
    if (grid.length) {
      // Freeze the axes to the measured region so moving UEs don't make the map "breathe".
      let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
      for (const p of grid) { xmin = Math.min(xmin, p[0]); xmax = Math.max(xmax, p[0]); ymin = Math.min(ymin, p[1]); ymax = Math.max(ymax, p[1]); }
      for (const p of apLoc) { xmin = Math.min(xmin, p[0]); xmax = Math.max(xmax, p[0]); ymin = Math.min(ymin, p[1]); ymax = Math.max(ymax, p[1]); }
      const pad = 12;
      layout.xaxis = Object.assign({}, layout.xaxis, { autorange: false, range: [xmax + pad, xmin - pad] });
      layout.yaxis = Object.assign({}, layout.yaxis, { autorange: false, range: [ymin - pad, ymax + pad] });
    }
    // Move the selected-UE annotation with the UE
    (layout.annotations || []).forEach(a => {
      const m = /^<b>UE (\d+)<\/b>/.exec(a.text || "");
      if (m) { const k = parseInt(m[1], 10); if (pos[k]) { a.x = pos[k][0]; a.y = pos[k][1]; } }
    });
    return layout;
  }

  function redraw() {
    if (!fig) return;
    Plotly.react(gd, buildData(), buildLayout(), { displayModeBar: false, responsive: true });
    if (!plotted) { plotted = true; send("streamlit:setFrameHeight", { height: height + 8 }); }
  }

  // ── emit to Python ───────────────────────────────────────────────────────
  function emit(event, force) {
    const now = performance.now();
    if (!force) {
      const eff = Math.max(emitMs, 1.2 * lastRtt);
      if (now - lastEmit < eff) return;
      if (!renderSinceEmit && now - lastEmit < 1500) return;
    }
    seq += 1; lastEmit = now; emitAt = now; renderSinceEmit = false;
    send("streamlit:setComponentValue", {
      dataType: "json",
      value: { epoch: epoch, seq: seq, positions: pos.map(p => [p[0], p[1]]), moving: moving, event: event || null },
    });
  }

  // ── Brownian motion ──────────────────────────────────────────────────────
  function tick() {
    if (dragIdx !== null) return;           // don't fight the user's hand
    for (let k = 0; k < pos.length; k++) {
      for (let t = 0; t < 3; t++) {
        const nx = pos[k][0] + sigma * randn(), ny = pos[k][1] + sigma * randn();
        if (nearestDist(nx, ny) <= rMax) { pos[k] = [nx, ny]; break; }
      }
    }
    redraw();
    emit(null, false);
  }
  function setMoving(on) {
    const wasRunning = timer !== null;
    moving = on;                              // set first: the stop emit reports moving=false
    if (on && !wasRunning) { timer = setInterval(tick, tickMs); }
    if (!on && wasRunning) { clearInterval(timer); timer = null; emit(null, true); }
  }

  // ── pointer handling (drag / click) ──────────────────────────────────────
  function plotPx(ev) {
    const fl = gd._fullLayout; if (!fl) return null;
    const r = gd.getBoundingClientRect();
    return [ev.clientX - r.left - fl._size.l, ev.clientY - r.top - fl._size.t];
  }
  function pxToData(px) { const fl = gd._fullLayout; return [fl.xaxis.p2d(px[0]), fl.yaxis.p2d(px[1])]; }
  function dataToPx(x, y) { const fl = gd._fullLayout; return [fl.xaxis.d2p(x), fl.yaxis.d2p(y)]; }
  function hit(px, pts, radius) {
    let bi = null, bd = radius * radius;
    for (let i = 0; i < pts.length; i++) {
      const q = dataToPx(pts[i][0], pts[i][1]);
      const d2 = (q[0] - px[0]) ** 2 + (q[1] - px[1]) ** 2;
      if (d2 < bd) { bd = d2; bi = i; }
    }
    return bi;
  }
  function insidePlot(px) {
    const fl = gd._fullLayout;
    return px[0] >= 0 && px[1] >= 0 && px[0] <= fl._size.w && px[1] <= fl._size.h;
  }

  gd.addEventListener("pointerdown", function (ev) {
    if (!gd._fullLayout || ev.button !== 0) return;
    const px = plotPx(ev); if (!px || !insidePlot(px)) return;
    downPx = px; dragMoved = false;
    const k = hit(px, pos, 12);
    if (k !== null) {
      dragIdx = k;
      try { gd.setPointerCapture(ev.pointerId); } catch (e) { /* ignore */ }
      ev.preventDefault(); ev.stopPropagation();
    }
  }, true);

  gd.addEventListener("pointermove", function (ev) {
    if (dragIdx === null) return;
    const px = plotPx(ev); if (!px) return;
    if (!dragMoved && Math.hypot(px[0] - downPx[0], px[1] - downPx[1]) < 4) return;
    dragMoved = true;
    const d = pxToData(px);
    pos[dragIdx] = clampToRegion(d[0], d[1]);
    redraw();
    emit(null, false);
    ev.preventDefault(); ev.stopPropagation();
  }, true);

  function finishPointer(ev) {
    if (!gd._fullLayout || downPx === null) return;
    const px = plotPx(ev) || downPx;
    const wasDrag = dragIdx !== null && dragMoved;
    const idx = dragIdx;
    dragIdx = null; downPx = null;
    try { gd.releasePointerCapture(ev.pointerId); } catch (e) { /* ignore */ }
    if (wasDrag) { emit(null, true); ev.preventDefault(); ev.stopPropagation(); return; }
    // click: UE, then AP, then empty area
    const k = idx !== null ? idx : hit(px, pos, 12);
    if (k !== null) { emit({ type: "select_ue", index: k }, true); return; }
    const l = hit(px, apLoc, 14);
    if (l !== null) { emit({ type: "click_ap", index: l }, true); return; }
    if (insidePlot(px) && (selUe !== null || selAp !== null)) emit({ type: "deselect", index: null }, true);
  }
  gd.addEventListener("pointerup", finishPointer, true);
  gd.addEventListener("pointercancel", function () { dragIdx = null; downPx = null; }, true);

  // ── render from Python ───────────────────────────────────────────────────
  function onRender(args) {
    if (!renderSinceEmit) { lastRtt = performance.now() - emitAt; renderSinceEmit = true; }
    fig = args.fig || fig;
    apLoc = args.ap_loc || apLoc;
    A = args.A || A; P = args.P || P;
    grid = args.grid || grid;
    added = args.added || []; removed = args.removed || []; changed = args.changed || [];
    rMax = args.r_max != null ? args.r_max : rMax;
    sigma = args.sigma != null ? args.sigma : sigma;
    const newTick = args.tick_ms || tickMs;
    if (timer && newTick !== tickMs) { clearInterval(timer); timer = setInterval(tick, newTick); }
    tickMs = newTick; emitMs = args.emit_ms || emitMs;
    height = args.height || height;
    selUe = args.selected_ue == null ? null : args.selected_ue;
    selAp = args.selected_ap == null ? null : args.selected_ap;
    const incoming = args.positions || [];
    if (epoch === null || args.epoch !== epoch || incoming.length !== pos.length) {
      epoch = args.epoch;
      pos = incoming.map(p => [p[0], p[1]]);
    }
    setMoving(!!args.moving);
    redraw();
  }

  // Test hook (used by tests/test_ue_map_js.py under QuickJS; harmless in browsers)
  window.__ueMapTest = {
    clampToRegion: clampToRegion, nearestDist: nearestDist, buildData: buildData,
    buildLayout: buildLayout, tick: tick, emit: emit, onRender: onRender,
    state: function () { return { epoch: epoch, seq: seq, pos: pos, moving: moving, timer: timer !== null }; },
  };
  if (early && early.pending) { const a = early.pending; early.pending = null; onRender(a); }
})();
