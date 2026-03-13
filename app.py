"""Radio Map-Enabled Digital Twin — Streamlit Web Application"""

import os
import streamlit as st

st.set_page_config(
    page_title="Radio Map Digital Twin",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Project imports ──────────────────────────────────────────────────────────
# data/CSI.npy is included in the GitHub repo (88 MB, under GitHub's 100 MB limit)
import numpy as np
import torch

# Ensure all torch.load calls default to CPU (for HF Spaces / CPU-only environments)
_torch_load_orig = torch.load
def _torch_load_cpu(f, *args, **kwargs):
    kwargs.setdefault('map_location', 'cpu')
    return _torch_load_orig(f, *args, **kwargs)
torch.load = _torch_load_cpu
import plotly.graph_objects as go

from main      import BS_loc
from generate  import cf_test, UE_test, loc_mean, loc_std
from process   import (
    AP_num_test, UE_num_test, loc_test_norm,
    A_test, P_test, signal_test, interf_test, rate_test,
    signal_mean, signal_std, interf_mean, interf_std, rate_mean, rate_std,
)
from signal_map import SignalModel
from interf_map import InterfModel
from rate_map   import RateModel

# ─── Cached model loading ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading GNN models …")
def _load_models():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load(model, path):
        if os.path.exists(path):
            model.load_state_dict(
                torch.load(path, weights_only=True, map_location=dev))
        model.eval()
        return model

    sm  = _load(SignalModel().to(dev), "model/signal_map.pth")
    im  = _load(InterfModel().to(dev), "model/interf_map.pth")
    rms = {
        snr: _load(RateModel(snr=snr).to(dev), f"model/rate_map_{snr}dB.pth")
        for snr in range(0, 31, 5)
    }
    return sm, im, rms, dev

signal_model, interf_model, rate_models, _device = _load_models()

# Stable references — never mutate cf_test globally
_CSI_data    = cf_test.CSI       # (L, N_test, Nc, Nt, Nr)
_BS_2d       = BS_loc[:, :2]     # (12, 2)
_UE_test_2d  = UE_test[:, :2]   # (N_test, 2)
_N_TEST      = len(AP_num_test)

# ─── Colour palette ───────────────────────────────────────────────────────────
BG    = "#1a1a2e"
AX    = "#0f0f1f"
GRID  = "#2a2a4a"
C_AP  = "#F65314"
C_UE  = "#7CBB00"
C_SEL = "#FFD700"
C_EDGE= "#4a9eff"
C_TRUE= "#FF6E40"
C_PRED= "#40C4FF"

# ─── Pure computation helpers ─────────────────────────────────────────────────
def _renorm(A: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Normalise raw weights per AP row."""
    P = np.zeros_like(A, dtype=np.float32)
    for l in range(A.shape[0]):
        w = raw[l] * A[l]
        s = w.sum()
        if s > 0:
            P[l] = w / s
    return P


def _recompute_true(ue_csi, ap_orig, A, P, snr_val):
    """Vectorised true SINR / rate computation (no global mutation)."""
    L   = _CSI_data.shape[0]
    K   = ue_csi.shape[1]
    Nc  = ue_csi.shape[2]

    A_full = np.zeros((L, K), dtype=np.float32)
    P_full = np.zeros((L, K), dtype=np.float32)
    for i, la in enumerate(ap_orig):
        A_full[la] = A[i]
        P_full[la] = P[i]

    # h: (L, K, Nc, Nt)  — squeeze Nr=1
    h       = ue_csi[..., 0]
    h_norms = np.linalg.norm(h, axis=-1)              # (L, K, Nc)

    # ── Signal power (MRT) ────────────────────────────────────────
    # signal[k,nc] = (sum_l sqrt(P[l,k])*A[l,k]*||h[l,k,nc]||)^2
    contrib    = (np.sqrt(P_full) * A_full)[:, :, None] * h_norms  # (L, K, Nc)
    sig_amp    = contrib.sum(axis=0)                   # (K, Nc)
    sig_pow    = sig_amp ** 2                          # (K, Nc)

    # ── Interference power ────────────────────────────────────────
    # w[l,m,nc] = h[l,m,nc] / ||h[l,m,nc]||
    w          = h / (h_norms[..., None] + 1e-30)     # (L, K, Nc, Nt)
    # cross[l,k,m,nc] = h[l,k,nc].conj() @ w[l,m,nc]
    cross      = np.einsum('lknt,lmnt->lkmn', h.conj(), w)  # (L, K, K, Nc)
    PA         = np.sqrt(P_full) * A_full              # (L, K)
    # interf_complex[k,m,nc] = sum_l PA[l,m] * cross[l,k,m,nc]
    ic         = np.einsum('lm,lkmn->kmn', PA, cross)  # (K, K, Nc)
    k_idx      = np.arange(K)
    ic[k_idx, k_idx, :] = 0                            # zero self-term
    int_pow    = (np.abs(ic) ** 2).sum(axis=1)          # (K, Nc)

    # ── Rates ─────────────────────────────────────────────────────
    noise  = 10 ** ((-87 - snr_val) / 10)
    SINR   = sig_pow / (int_pow + noise)               # (K, Nc)
    rate   = np.mean(np.log2(1 + SINR), axis=1)       # (K,)

    sig_db = 10 * np.log10(sig_pow.mean(axis=1) + 1e-300)
    int_db = 10 * np.log10(int_pow.mean(axis=1) + 1e-300)

    unserved = np.where(A_full.sum(axis=0) < 0.5)[0]
    rate[unserved] = 0.0
    # sig_db stays -inf for unserved → displayed as gap in bar chart

    return sig_db, int_db, rate


def _predict(loc_norm, ap_num, ue_num, A, P, snr_val):
    """GNN inference → (pred_signal, pred_interf, pred_rate)."""
    A_t = torch.zeros(12, 30, dtype=torch.int32,    device=_device)
    P_t = torch.zeros(12, 30, dtype=torch.float32,  device=_device)
    A_t[:ap_num, :ue_num] = torch.from_numpy(A.astype(np.int32))
    P_t[:ap_num, :ue_num] = torch.from_numpy(P)
    loc = loc_norm.to(_device)

    with torch.no_grad():
        ei, ea = signal_model.create_edges(loc, ap_num, ue_num, A_t, P_t)
        ps = (signal_model(loc, ei, ea) * signal_std + signal_mean).cpu().numpy()

        e1i, e1a, e2i, e2a = interf_model.create_edges(loc, ap_num, ue_num, A_t, P_t)
        pi = (interf_model(loc, e1i, e1a, e2i, e2a) * interf_std + interf_mean).cpu().numpy()

        pr = (rate_models[snr_val](loc, ap_num, ue_num, A_t, P_t)
              * rate_std + rate_mean).cpu().numpy()

    unserved = np.where(A.sum(axis=0) < 0.5)[0]
    ps[unserved] = 0.0
    pr[unserved] = 0.0

    return ps[:ue_num], pi[:ue_num], pr[:ue_num]


# ─── Session-state helpers ────────────────────────────────────────────────────
def _load_sample(idx: int):
    ss = st.session_state
    ss.test_idx    = idx
    ss.selected_ue = None
    ss.is_modified = False

    ap_num = int(AP_num_test[idx].item())
    ue_num = int(UE_num_test[idx].item())
    xy     = loc_test_norm[idx].numpy() * loc_std + loc_mean
    ap_loc = xy[:ap_num]
    ue_loc = xy[ap_num: ap_num + ue_num]

    A = A_test[idx, :ap_num, :ue_num].numpy().copy().astype(np.float32)
    P = P_test[idx, :ap_num, :ue_num].numpy().copy().astype(np.float32)

    ap_orig = np.array([
        np.argmin(np.sum((_BS_2d    - loc) ** 2, axis=1)) for loc in ap_loc])
    ue_orig = np.array([
        np.argmin(np.sum((_UE_test_2d - loc) ** 2, axis=1)) for loc in ue_loc])

    snr_val = ss.get("snr", 15)

    ss.ap_num  = ap_num;   ss.ue_num  = ue_num
    ss.ap_loc  = ap_loc;   ss.ue_loc  = ue_loc
    ss.A       = A;        ss.P       = P
    ss.raw_w   = P.copy()
    ss.ap_orig = ap_orig
    ss.ue_csi  = _CSI_data[:, ue_orig, :]   # (L, K, Nc, Nt, Nr)
    ss.loc_norm = loc_test_norm[idx]

    ss.true_signal = signal_test[idx, :ue_num].numpy()
    ss.true_interf = interf_test[idx, :ue_num].numpy()
    ss.true_rate   = rate_test[snr_val // 5, idx, :ue_num].numpy()

    ps, pi, pr = _predict(ss.loc_norm, ap_num, ue_num, A, P, snr_val)
    ss.pred_signal = ps;  ss.pred_interf = pi;  ss.pred_rate = pr


def _refresh(ss):
    """Recompute true + predicted values after A/P change."""
    sig, intr, rate = _recompute_true(
        ss.ue_csi, ss.ap_orig, ss.A, ss.P, ss.snr)
    ss.true_signal = sig;  ss.true_interf = intr;  ss.true_rate = rate

    ps, pi, pr = _predict(
        ss.loc_norm, ss.ap_num, ss.ue_num, ss.A, ss.P, ss.snr)
    ss.pred_signal = ps;  ss.pred_interf = pi;  ss.pred_rate = pr


# ─── Plotly figure builders ───────────────────────────────────────────────────
# Trace indices in the network figure are FIXED:
#   0 = edge lines   1 = power labels   2 = APs   3 = UEs
AP_TRACE = 2
UE_TRACE = 3

def _net_fig(ss):
    ap_loc, ue_loc = ss.ap_loc, ss.ue_loc
    A, P           = ss.A, ss.P
    ap_num, ue_num = ss.ap_num, ss.ue_num
    sel            = ss.selected_ue

    fig = go.Figure()

    # Trace 0 – edges (single trace, None separators)
    ex, ey, lx, ly, lt = [], [], [], [], []
    for l in range(ap_num):
        for k in range(ue_num):
            if A[l, k] > 0.5:
                pw = float(P[l, k])
                x1, y1 = ap_loc[l];  x2, y2 = ue_loc[k]
                ex += [x1, x2, None];  ey += [y1, y2, None]
                lx.append((x1 + x2) / 2);  ly.append((y1 + y2) / 2)
                lt.append(f"{pw:.2f}")

    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(color=C_EDGE, width=1.8), opacity=0.55,
        hoverinfo="skip", showlegend=False,
    ))  # trace 0

    # Trace 1 – power labels
    fig.add_trace(go.Scatter(
        x=lx, y=ly, mode="text", text=lt,
        textfont=dict(color="#99ccff", size=8),
        hoverinfo="skip", showlegend=False,
    ))  # trace 1

    # Trace 2 – APs
    fig.add_trace(go.Scatter(
        x=ap_loc[:, 0], y=ap_loc[:, 1],
        mode="markers+text",
        marker=dict(symbol="triangle-up", size=16, color=C_AP,
                    line=dict(color="white", width=1)),
        text=[f"AP{i}" for i in range(ap_num)],
        textposition="top center",
        textfont=dict(color="#ffaa88", size=10),
        customdata=list(range(ap_num)),
        name="AP",
        hovertemplate="AP %{customdata}<extra></extra>",
    ))  # trace 2

    # Trace 3 – UEs
    colors = [C_SEL if k == sel else C_UE for k in range(ue_num)]
    sizes  = [14    if k == sel else 10   for k in range(ue_num)]
    fig.add_trace(go.Scatter(
        x=ue_loc[:, 0], y=ue_loc[:, 1],
        mode="markers+text",
        marker=dict(symbol="circle", size=sizes, color=colors,
                    line=dict(color="white", width=0.6)),
        text=[str(k) for k in range(ue_num)],
        textposition="bottom center",
        textfont=dict(color="#aaffaa", size=9),
        customdata=list(range(ue_num)),
        name="UE",
        hovertemplate="UE %{customdata}<extra></extra>",
    ))  # trace 3

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=AX,
        xaxis=dict(autorange="reversed", scaleanchor="y", scaleratio=1,
                   gridcolor=GRID, title="X (m)", color="white"),
        yaxis=dict(gridcolor=GRID, title="Y (m)", color="white"),
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
                    x=0.01, y=0.99),
        margin=dict(l=55, r=15, t=45, b=50),
        clickmode="event+select",
        dragmode=False,
        height=710,
        title=dict(text="Cell-Free Network Topology",
                   font=dict(color="white", size=13), x=0.5),
    )
    return fig


def _fit_yrange(true_v, pred_v):
    finite = [v for v in list(true_v) + list(pred_v) if np.isfinite(v)]
    if not finite:
        return {}
    lo, hi = min(finite), max(finite)
    margin = max((hi - lo) * 0.15, 0.5)
    return {"range": [lo - margin, hi + margin]}


def _bar_fig(true_v, pred_v, ylabel, title, ue_num, sel, fit_range=False):
    def _clean(arr):
        return [float(v) if np.isfinite(v) else None for v in arr]

    x  = list(range(ue_num))
    tc = [C_SEL if k == sel else C_TRUE for k in range(ue_num)]
    pc = [C_SEL if k == sel else C_PRED for k in range(ue_num)]

    fig = go.Figure([
        go.Bar(x=x, y=_clean(true_v),  name="True",      marker_color=tc, opacity=0.9),
        go.Bar(x=x, y=_clean(pred_v),  name="Predicted", marker_color=pc, opacity=0.9),
    ])
    fig.update_layout(
        barmode="group",
        title=dict(text=title, font=dict(color="white", size=11), x=0.5),
        yaxis=dict(title=dict(text=ylabel, font=dict(size=10)),
                   color="white", gridcolor=GRID,
                   **(_fit_yrange(true_v, pred_v) if fit_range else {})),
        xaxis=dict(title="UE", color="white", gridcolor=GRID,
                   tickmode="linear", tick0=0, dtick=max(1, ue_num // 12)),
        paper_bgcolor=BG, plot_bgcolor=AX,
        font=dict(color="white", size=9),
        legend=dict(bgcolor="rgba(17,17,34,0.6)", font=dict(size=9, color="white"),
                    bordercolor=GRID, borderwidth=1),
        margin=dict(l=55, r=10, t=42, b=35),
        height=226,
    )
    return fig


# ─── Initialise session state ─────────────────────────────────────────────────
if "snr" not in st.session_state:
    st.session_state.snr = 15
    _load_sample(0)

ss = st.session_state

# ─── Title ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h3 style='text-align:center;color:white;margin-bottom:6px'>"
    "Radio Map-Enabled Digital Twin &mdash; Interactive Demo</h3>",
    unsafe_allow_html=True,
)

with st.expander("📖 How to use", expanded=True):
    st.markdown("""
**Select a UE**  &nbsp;→&nbsp; Click any green circle on the network graph. It turns gold when selected.

**Deselect** &nbsp;→&nbsp; Click the selected UE again, or click an empty area.

**Toggle AP connection** &nbsp;→&nbsp; First select a UE, then click an AP (triangle). If the AP is already connected it will be disconnected (and vice versa). Note: the last serving AP cannot be removed.

**Adjust power** &nbsp;→&nbsp; Select a UE — power weight sliders for each serving AP appear at the bottom of the page. Drag a slider to redistribute power; values are auto-normalised per AP.

**Browse samples** &nbsp;→&nbsp; Use **< Prev** / **Next >** to switch between test scenarios.

**Reset** &nbsp;→&nbsp; Restores the current sample to its original topology and power.

**SNR** &nbsp;→&nbsp; Use the SNR slider (bottom right) to change the noise level (0 – 30 dB). The rate chart updates instantly.
""")

# ─── Main layout ──────────────────────────────────────────────────────────────
col_l, col_r = st.columns([1.6, 1.0])

with col_l:
    ev = st.plotly_chart(
        _net_fig(ss), use_container_width=True,
        key="net", on_select="rerun",
    )

    # Handle click events
    pts = ev.selection.points if ev and ev.selection else []
    if pts:
        pt    = pts[0]
        curve = pt["curve_number"]
        pidx  = pt["point_number"]

        if curve == UE_TRACE:
            k = pidx
            ss.selected_ue = k if ss.selected_ue != k else None
            st.rerun()

        elif curve == AP_TRACE and ss.selected_ue is not None:
            l, k = pidx, ss.selected_ue
            # Guard: cannot disconnect last serving AP
            if ss.A[l, k] > 0.5 and ss.A[:, k].sum() <= 1:
                pass
            else:
                ss.A[l, k]    = 1.0 - ss.A[l, k]
                ss.raw_w[l, k] = 1.0 if ss.A[l, k] > 0.5 else 0.0
                ss.P           = _renorm(ss.A, ss.raw_w)
                ss.is_modified = True
                _refresh(ss)
                st.rerun()

with col_r:
    st.plotly_chart(
        _bar_fig(ss.true_signal, ss.pred_signal,
                 "dBW", "Signal Power", ss.ue_num, ss.selected_ue,
                 fit_range=True),
        use_container_width=True, key="b_sig",
    )
    st.plotly_chart(
        _bar_fig(ss.true_interf, ss.pred_interf,
                 "dBW", "Interference Power", ss.ue_num, ss.selected_ue,
                 fit_range=True),
        use_container_width=True, key="b_int",
    )
    st.plotly_chart(
        _bar_fig(ss.true_rate, ss.pred_rate,
                 "bits/s/Hz", f"Achievable Rate  (SNR = {ss.snr} dB)",
                 ss.ue_num, ss.selected_ue),
        use_container_width=True, key="b_rate",
    )

# ─── Controls ─────────────────────────────────────────────────────────────────
st.divider()
c1, c2, c3, c4 = st.columns([1, 1, 1, 4])

with c1:
    if st.button("< Prev", use_container_width=True,
                 disabled=ss.test_idx == 0):
        _load_sample(ss.test_idx - 1)
        st.rerun()

with c2:
    if st.button("Next >", use_container_width=True,
                 disabled=ss.test_idx >= _N_TEST - 1):
        _load_sample(ss.test_idx + 1)
        st.rerun()

with c3:
    if st.button("Reset", use_container_width=True):
        _load_sample(ss.test_idx)
        st.rerun()

with c4:
    new_snr = st.select_slider(
        "SNR (dB)", options=list(range(0, 31, 5)), value=ss.snr,
    )
    if new_snr != ss.snr:
        ss.snr = new_snr
        if ss.is_modified:
            _refresh(ss)          # user changed topology → full recompute
        else:
            # original topology → use precomputed dataset values
            ss.true_rate = rate_test[new_snr // 5, ss.test_idx,
                                     :ss.ue_num].numpy()
            _, _, pr = _predict(ss.loc_norm, ss.ap_num, ss.ue_num,
                                ss.A, ss.P, new_snr)
            ss.pred_rate = pr
        st.rerun()

st.caption(
    f"Sample #{ss.test_idx}  |  APs: {ss.ap_num}  |  UEs: {ss.ue_num}  "
    f"|  Connections: {int(ss.A.sum())}"
)

# ─── Power sliders (shown when a UE is selected) ──────────────────────────────
if ss.selected_ue is not None:
    k       = ss.selected_ue
    serving = [l for l in range(ss.ap_num) if ss.A[l, k] > 0.5]
    if serving:
        st.markdown(
            f"**Power weights for UE {k}** "
            f"(raw weights, auto-normalised per AP):"
        )
        cols    = st.columns(len(serving))
        changed = False
        for col, l in zip(cols, serving):
            w_new = col.slider(
                f"AP {l}",
                min_value=0.05, max_value=5.0,
                value=float(ss.raw_w[l, k]),
                step=0.05,
                key=f"pw_{l}_{k}_{ss.test_idx}",
            )
            if abs(w_new - ss.raw_w[l, k]) > 1e-4:
                ss.raw_w[l, k] = w_new
                changed        = True
        if changed:
            ss.P           = _renorm(ss.A, ss.raw_w)
            ss.is_modified = True
            _refresh(ss)
            st.rerun()
