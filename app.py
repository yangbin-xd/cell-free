"""Radio Map-Enabled Digital Twin — Streamlit Web Application"""

import os
import re
import json
import streamlit as st
import streamlit.components.v1 as components

# Promote .streamlit/secrets.toml entries into os.environ. Parsed directly
# rather than via st.secrets — st.secrets renders a "No secrets found" error
# to the page when the file is absent (HF Space case), which would count as
# the first Streamlit command and break st.set_page_config below.
# Does not overwrite env vars already set externally (HF repo secrets, systemd).
_secrets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             ".streamlit", "secrets.toml")
if os.path.isfile(_secrets_path):
    try:
        with open(_secrets_path) as _f:
            for _line in _f:
                _m = re.match(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"([^"]*)"', _line)
                if _m:
                    os.environ.setdefault(_m.group(1), _m.group(2))
    except Exception:
        pass

from agent import (detect_optimization_intent, parse_goal,
                   format_report as opt_format_report,
                   build_unified_prompt, parse_unified_response,
                   all_ue_metric_query, format_all_ue_metric)
from power_only import power_optimize  # noqa: F401
from joint_optimize import joint_optimize  # noqa: F401
import closed_loop as _cl
try:                      # private analytics module -- not in the public repo
    from visits import log_visit_once, log_query
except ImportError:       # public checkout: logging disabled
    def log_visit_once(*a, **k): pass
    def log_query(*a, **k): pass

st.set_page_config(
    page_title="AI Agent for Resource Allocation in Wireless Networks",
    layout="wide",
    initial_sidebar_state="collapsed",
)
log_visit_once()

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 0.5rem; }

    /* ── Mobile (≤768px): stack columns, keep desktop untouched ── */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            padding-top: 3.5rem !important;  /* clear Streamlit's fixed top header */
        }
        /* Title: shrink + ensure breathing room from header */
        .block-container h1 {
            font-size: 1.25rem !important;
            line-height: 1.3 !important;
            margin-top: 0.25rem !important;
            padding: 0 0.25rem !important;
        }
        /* Force every st.columns row to wrap to a single column */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.4rem !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
        [data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 1 1 100% !important;
            min-width: 100% !important;
            width: 100% !important;
        }
        /* Buttons compact + full-width tap targets */
        .stButton > button {
            padding: 0.45rem 0.6rem !important;
            font-size: 0.92rem !important;
            min-height: 2.4rem !important;
        }
        /* Inputs / form fields */
        .stTextInput input, .stTextArea textarea {
            font-size: 0.95rem !important;
        }
        /* Chat message bubbles */
        [data-testid="stChatMessage"] {
            padding: 0.55rem 0.7rem !important;
        }
        /* Plotly figures should be full width */
        .js-plotly-plot, .plot-container { max-width: 100% !important; }
        /* Reduce header sizes a touch */
        h1 { font-size: 1.4rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1.05rem !important; }
        /* Caption / markdown sizing */
        [data-testid="stCaptionContainer"] { font-size: 0.8rem !important; }
        /* Sliders take full width */
        .stSlider { padding: 0 0.2rem !important; }
    }
</style>
""", unsafe_allow_html=True)

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
        # Fail loudly: a missing checkpoint must not silently fall back to
        # untrained weights (same reasoning as demo._load).
        if not os.path.exists(path):
            raise FileNotFoundError(f"checkpoint not found: {path!r}")
        model.load_state_dict(
            torch.load(path, weights_only=True, map_location=dev))
        model.eval()
        return model

    # Proposed model = the ablation campaign's seed-0 full model (Table I
    # "Proposed"; see demo.py). Only a 15 dB rate head exists and it is the
    # right one at every SNR: snr enters RateModel solely through
    # `noise_dB = -87 - self.snr`, so no parameter is SNR-specific.
    stem = os.environ.get("CF_MODEL_STEM", "main_seed0")
    sig_ckpt, int_ckpt = f"model/signal_{stem}.pth", f"model/interf_{stem}.pth"
    rate_ckpt = f"model/rate_{stem}_15dB.pth"
    sm  = _load(SignalModel().to(dev), sig_ckpt)
    im  = _load(InterfModel().to(dev), int_ckpt)
    rms = {
        snr: _load(RateModel(signal_model_path=sig_ckpt,
                             interf_model_path=int_ckpt,
                             snr=snr).to(dev), rate_ckpt)
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

MAX_AP_PER_UE = 4   # each UE may only connect to its N nearest APs

# ─── Pure computation helpers ─────────────────────────────────────────────────
def _nearest_aps(ap_loc: np.ndarray, ue_loc: np.ndarray,
                 k: int, n: int = MAX_AP_PER_UE) -> set:
    """Return set of n nearest AP indices for UE k."""
    dists = np.sum((ap_loc - ue_loc[k]) ** 2, axis=1)
    return set(np.argsort(dists)[:n].tolist())


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


def _predict_rate_only(loc_norm, ap_num, ue_num, A, P, snr_val):
    """Lightweight oracle for optimizer: only calls RateModel (skips Signal/Interf)."""
    A_t = torch.zeros(12, 30, dtype=torch.int32,   device=_device)
    P_t = torch.zeros(12, 30, dtype=torch.float32, device=_device)
    A_t[:ap_num, :ue_num] = torch.from_numpy(A.astype(np.int32))
    P_t[:ap_num, :ue_num] = torch.from_numpy(P)
    loc = loc_norm.to(_device)
    with torch.no_grad():
        pr = (rate_models[snr_val](loc, ap_num, ue_num, A_t, P_t)
              * rate_std + rate_mean).cpu().numpy()
    unserved = np.where(A.sum(axis=0) < 0.5)[0]
    pr[unserved] = 0.0
    return None, None, pr[:ue_num]


# ─── Session-state helpers ────────────────────────────────────────────────────
def _load_sample(idx: int):
    ss = st.session_state
    ss.test_idx    = idx
    ss.selected_ue = None
    ss.selected_ap = None
    ss.is_modified = False
    ss.changed_links = set(); ss.added_links = set(); ss.removed_links = set()
    ss.opt_targets   = []
    ss._chat_history = []

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
    # Immutable original-allocation snapshot — dual-baseline report uses it to
    # show gains vs the original when an optimization starts from an
    # already-modified state (never mutated; reset reloads the whole sample).
    ss.A0      = A.copy(); ss.P0      = P.copy()
    ss.ap_orig = ap_orig
    ss.ue_csi  = _CSI_data[:, ue_orig, :]   # (L, K, Nc, Nt, Nr)
    ss.loc_norm = loc_test_norm[idx]

    # Compute true β from CSI for DNN optimizer (avoids distance approximation)
    h = ss.ue_csi[..., 0]                              # (L_full, K, Nc, Nt)
    h_norms = np.linalg.norm(h, axis=-1)                # (L_full, K, Nc)
    beta_full = np.mean(h_norms ** 2, axis=2)           # (L_full, K)
    ss.beta = beta_full[ap_orig]                        # (ap_num, K)

    ss.true_signal = signal_test[idx, :ue_num].numpy()
    ss.true_interf = interf_test[idx, :ue_num].numpy()
    ss.true_rate   = rate_test[snr_val // 5, idx, :ue_num].numpy()

    ps, pi, pr = _predict(ss.loc_norm, ap_num, ue_num, A, P, snr_val)
    ss.pred_signal = ps;  ss.pred_interf = pi;  ss.pred_rate = pr

    # Clear prev snapshot — no delta to show on fresh load
    ss._delta_locked = False
    ss.prev_true_signal = None
    ss.prev_true_interf = None
    ss.prev_true_rate   = None
    ss.prev_pred_signal = None
    ss.prev_pred_interf = None
    ss.prev_pred_rate   = None


def _snapshot_before(ss):
    """Save current values as 'prev' for before/after delta display."""
    if getattr(ss, '_delta_locked', False):
        return                                    # optimization delta protected
    ss.prev_true_signal = ss.true_signal.copy()
    ss.prev_true_interf = ss.true_interf.copy()
    ss.prev_true_rate   = ss.true_rate.copy()
    ss.prev_pred_signal = ss.pred_signal.copy()
    ss.prev_pred_interf = ss.pred_interf.copy()
    ss.prev_pred_rate   = ss.pred_rate.copy()


def _refresh(ss):
    """Recompute true + predicted values after A/P change."""
    _snapshot_before(ss)
    sig, intr, rate = _recompute_true(
        ss.ue_csi, ss.ap_orig, ss.A, ss.P, ss.snr)
    ss.true_signal = sig;  ss.true_interf = intr;  ss.true_rate = rate

    ps, pi, pr = _predict(
        ss.loc_norm, ss.ap_num, ss.ue_num, ss.A, ss.P, ss.snr)
    ss.pred_signal = ps;  ss.pred_interf = pi;  ss.pred_rate = pr


def _make_ss_snap(ss) -> dict:
    """Pure-dict snapshot of session state for optimizer (no Streamlit refs)."""
    return {
        "A":        ss.A.copy(),
        "raw_w":    ss.raw_w.copy(),
        "P":        ss.P.copy(),
        "ap_num":   ss.ap_num,
        "ue_num":   ss.ue_num,
        "ap_loc":   ss.ap_loc,
        "ue_loc":   ss.ue_loc,
        "true_rate": ss.true_rate[:ss.ue_num].copy(),
        "loc_norm": ss.loc_norm,
        "snr":      ss.snr,
        "beta":     getattr(ss, "beta", None),
    }


def _resolve_pronouns(instruction: str, ss) -> str:
    """If instruction has pronouns (它/他/她/it/this UE) but no explicit UEn,
    look at recent chat history to find the last mentioned UE index and append it."""
    t = instruction.lower()
    has_ue = bool(re.search(r'ue\s*\d+', t, re.IGNORECASE))
    has_pronoun = bool(re.search(r'\b(it|this\s+ue|that\s+ue)\b|[它他她]', t))
    if has_ue or not has_pronoun:
        return instruction
    # Search chat history for last UE mention
    history = ss.get("_chat_history") or []
    for entry in reversed(history):
        m = re.search(r'UE\s*(\d+)', entry.get("assistant", "") + " " + entry.get("user", ""))
        if m:
            k = int(m.group(1))
            return instruction + f"（UE{k}）"
    return instruction


def _goal_summary(goal: dict, zh: bool, ue_num: int = 0) -> str:
    """Return a concise goal description using three-level objective hierarchy.

    Always derived from the goal dict so the summary stays consistent with
    the per-UE sections in ``format_report``. The LLM's ``summary`` field
    is ignored — it hallucinated priority swaps when two boost groups
    existed without a global objective.
    """

    g = goal.get("type", "max_sum_rate")
    t = goal.get("target_ues", [])
    p = goal.get("protected_ues", [])
    mult = goal.get("target_multiplier")
    p_floor = goal.get("protected_floor", 1.0)
    c_floor = goal.get("constraint_floor", 0.0)
    _all_targeted = t and ue_num > 0 and len(t) >= ue_num

    _has_global = g in ("max_sum_rate", "fairness", "max_min")
    parts = []

    # ── Determine hierarchy labels ──
    def _ue_list_str(ues):
        if len(ues) <= 5:
            return "UE" + ",".join(map(str, ues))
        # Summarise patterns
        evens = sorted(k for k in ues if k % 2 == 0)
        odds  = sorted(k for k in ues if k % 2 == 1)
        all_evens = evens == sorted(k for k in range(ue_num) if k % 2 == 0 and k < ue_num)
        all_odds  = odds  == sorted(k for k in range(ue_num) if k % 2 == 1 and k < ue_num)
        if all_evens and not odds:
            return "偶数ID用户" if zh else "even-ID UEs"
        if all_odds and not evens:
            return "奇数ID用户" if zh else "odd-ID UEs"
        return f"{len(ues)}个用户" if zh else f"{len(ues)} UEs"

    # ── Primary ──
    _um = goal.get("ue_multipliers", {})
    if _has_global and not (g == "max_sum_rate" and not t and not p and mult is None):
        # Global objective is primary
        if g == "fairness":
            parts.append("主要: 比例公平" if zh else "primary: proportional fairness")
        elif g == "max_min":
            # When a numeric min-rate floor is set, the instruction is really
            # "raise the worst UE to X" — one objective, not two. Skip the
            # generic "maximize worst UE" so the floor line below becomes the
            # sole primary ("主要: 最差用户速率≥X"), instead of a redundant
            # "最大化最差用户 + 最差用户速率≥X" pair describing the same UE.
            _mrf = goal.get("min_rate_floor")
            if not (_mrf is not None and _mrf > 0):
                parts.append("主要: 最大化最差用户" if zh else "primary: maximize worst UE")
        else:
            parts.append("主要: 最大化总速率" if zh else "primary: maximize sum rate")
    elif _um:
        pass  # ue_multipliers groups will be added in the next section
    elif _all_targeted:
        _pct = round((mult - 1) * 100) if mult else 0
        if mult and 0 < _pct < 200:
            parts.append(
                f"主要: 所有用户提升{_pct}%" if zh else f"primary: all UEs +{_pct}%")
        elif mult:
            parts.append(
                f"主要: 所有用户提升{mult}倍" if zh else f"primary: all UEs {mult}x")
        else:
            parts.append("主要: 提升所有用户" if zh else "primary: maximize all UEs")
    elif t:
        _tstr = _ue_list_str(t)
        if mult:
            _pct = round((mult - 1) * 100)
            if 0 < _pct < 200:
                parts.append(
                    f"主要: {_tstr}提升{_pct}%" if zh else f"primary: {_tstr} +{_pct}%")
            else:
                parts.append(
                    f"主要: {_tstr}提升{mult}倍" if zh else f"primary: {_tstr} {mult}x")
        else:
            parts.append(
                f"主要: 提升{_tstr}" if zh else f"primary: boost {_tstr}")
    else:
        parts.append("主要: 最大化总速率" if zh else "primary: maximize sum rate")

    # ── Secondary type (second global objective) ──
    sec_type = goal.get("secondary_type")
    if sec_type:
        _n_sec = len(parts) - 1
        _lbl = ("次要" if _n_sec == 0 else "补充")
        _lbl_en = ("secondary" if _n_sec == 0 else "supplementary")
        _sec_names = {
            "fairness": ("尽可能公平", "maximize fairness"),
            "max_sum_rate": ("最大化总速率", "maximize sum rate"),
            "max_min": ("最大化最差用户", "maximize worst UE"),
        }
        _sz, _se = _sec_names.get(sec_type, (sec_type, sec_type))
        parts.append(f"{_lbl}: {_sz}" if zh else f"{_lbl_en}: {_se}")

    # ── UE boost groups (from ue_multipliers or legacy target/protected) ──
    def _next_label():
        n = len(parts)
        if n == 0: return ("主要", "primary")
        if n == 1: return ("次要", "secondary")
        return ("补充", "supplementary")

    _um = goal.get("ue_multipliers", {})
    if _um:
        from collections import defaultdict
        # Build groups in the same priority order as format_report's
        # display: target_ues first (primary), then protected_ues
        # (secondary), then any remaining UEs in _um (supplementary).
        _t_set = set(goal.get("target_ues") or [])
        _p_set = set(goal.get("protected_ues") or [])
        _bare_set = set(goal.get("bare_targets") or [])
        _ordered_groups = []  # list of (multiplier, [ue_indices], is_bare)

        def _collect(ue_set):
            # Split into bare and explicit sub-groups so they render
            # differently (bare → "提升UEx" no %).
            _bare_list = sorted(k for k in ue_set if k in _bare_set and k in _um)
            _explicit = defaultdict(list)
            for k, m in _um.items():
                if k < ue_num and k in ue_set and k not in _bare_set:
                    _explicit[m].append(k)
            _out = []
            if _bare_list:
                _out.append((None, _bare_list, True))  # marker: bare
            for _mv, _klist in sorted(_explicit.items(), key=lambda x: -x[0]):
                _out.append((_mv, _klist, False))
            return _out

        _ordered_groups.extend(_collect(_t_set))
        _ordered_groups.extend(_collect(_p_set - _t_set))
        _rest_set = set(_um) - _t_set - _p_set
        _ordered_groups.extend(_collect(_rest_set))

        for _mv, _klist, _is_bare in _ordered_groups:
            _ustr = _ue_list_str(sorted(_klist))
            _lbl, _lbl_en = _next_label()
            if _is_bare:
                parts.append(
                    f"{_lbl}: 提升{_ustr}" if zh else f"{_lbl_en}: boost {_ustr}")
                continue
            _pct = round((_mv - 1) * 100)
            if _mv > 1.0 and 0 < _pct < 200:
                parts.append(
                    f"{_lbl}: {_ustr}提升{_pct}%" if zh else f"{_lbl_en}: {_ustr} +{_pct}%")
            elif _mv > 1.0:
                parts.append(
                    f"{_lbl}: {_ustr}提升{_mv}倍" if zh else f"{_lbl_en}: {_ustr} {_mv}x")
            elif _mv >= 1.0:
                parts.append(
                    f"{_lbl}: {_ustr}不降低" if zh else f"{_lbl_en}: {_ustr} no degradation")
            else:
                _drop = round((1 - _mv) * 100)
                parts.append(
                    f"{_lbl}: {_ustr}降低≤{_drop}%" if zh else f"{_lbl_en}: {_ustr} drop ≤{_drop}%")
    elif not _has_global:
        # Legacy (no global): targets already added as Primary above, only add protected
        if p:
            _pstr = _ue_list_str(p)
            _lbl, _lbl_en = _next_label()
            if p_floor > 1.0:
                _pct = round((p_floor - 1) * 100)
                if 0 < _pct < 200:
                    parts.append(
                        f"{_lbl}: {_pstr}提升{_pct}%" if zh else f"{_lbl_en}: {_pstr} +{_pct}%")
                else:
                    parts.append(
                        f"{_lbl}: {_pstr}提升{p_floor}倍" if zh else f"{_lbl_en}: {_pstr} {p_floor}x")
            elif p_floor >= 1.0:
                parts.append(
                    f"{_lbl}: {_pstr}不降低" if zh else f"{_lbl_en}: {_pstr} no degradation")
            else:
                p_drop = round((1 - p_floor) * 100)
                parts.append(
                    f"{_lbl}: {_pstr}降低≤{p_drop}%" if zh else f"{_lbl_en}: {_pstr} drop ≤{p_drop}%")
    else:
        # Legacy: global objective + target/protect as secondary constraints
        if t and not _all_targeted:
            _tstr = _ue_list_str(t)
            _lbl, _lbl_en = _next_label()
            if mult:
                _pct = round((mult - 1) * 100)
                if 0 < _pct < 200:
                    parts.append(
                        f"{_lbl}: {_tstr}提升{_pct}%" if zh else f"{_lbl_en}: {_tstr} +{_pct}%")
                else:
                    parts.append(
                        f"{_lbl}: {_tstr}提升{mult}倍" if zh else f"{_lbl_en}: {_tstr} {mult}x")
            else:
                parts.append(
                    f"{_lbl}: 提升{_tstr}" if zh else f"{_lbl_en}: boost {_tstr}")
        if p:
            _pstr = _ue_list_str(p)
            _lbl, _lbl_en = _next_label()
            if p_floor > 1.0:
                _pct = round((p_floor - 1) * 100)
                if 0 < _pct < 200:
                    parts.append(
                        f"{_lbl}: {_pstr}提升{_pct}%" if zh else f"{_lbl_en}: {_pstr} +{_pct}%")
                else:
                    parts.append(
                        f"{_lbl}: {_pstr}提升{p_floor}倍" if zh else f"{_lbl_en}: {_pstr} {p_floor}x")
            elif p_floor >= 1.0:
                parts.append(
                    f"{_lbl}: {_pstr}不降低" if zh else f"{_lbl_en}: {_pstr} no degradation")
            else:
                p_drop = round((1 - p_floor) * 100)
                parts.append(
                    f"{_lbl}: {_pstr}降低≤{p_drop}%" if zh else f"{_lbl_en}: {_pstr} drop ≤{p_drop}%")

    # Per-UE absolute floor
    _uaf = goal.get("ue_abs_floor") or {}
    _uaf_covered = set(_um.keys()) if _um else set(t) | set(p)
    _uaf_ues = sorted(k for k in _uaf if k not in _uaf_covered)
    if _uaf_ues:
        _ustr = _ue_list_str(_uaf_ues)
        _lbl, _lbl_en = _next_label()
        # Group by floor value
        _floor_vals = set(_uaf[k] for k in _uaf_ues)
        if len(_floor_vals) == 1:
            _fv = next(iter(_floor_vals))
            parts.append(
                f"{_lbl}: {_ustr}速率≥{_fv} bps/Hz" if zh else f"{_lbl_en}: {_ustr} rate ≥{_fv} bps/Hz")
        else:
            _descs = []
            for k in _uaf_ues:
                _descs.append(f"UE{k}≥{_uaf[k]}")
            parts.append(f"{_lbl}: {', '.join(_descs)} bps/Hz" if zh else f"{_lbl_en}: {', '.join(_descs)} bps/Hz")
        _uaf_covered = _uaf_covered | set(_uaf_ues)

    # Absolute minimum rate floor (global worst UE)
    mrf = goal.get("min_rate_floor")
    if mrf is not None and mrf > 0:
        _lbl, _lbl_en = _next_label()
        parts.append(
            f"{_lbl}: 最差用户速率≥{mrf}" if zh else f"{_lbl_en}: worst UE rate ≥{mrf}")

    # Others constraint — label follows _next_label() so it stays in sync
    # with format_report (which counts already-rendered sections).
    _all_covered = set(_um.keys()) if _um else set(t) | set(p)
    _all_covered |= set(_uaf.keys())
    _has_others = ue_num > 0 and len(_all_covered) < ue_num
    # NOTE: do NOT also require _all_covered to be non-empty. For a global
    # objective (max_min / max_sum / fairness) with no named UE group,
    # _all_covered is empty but "其他用户不降低" is still meaningful (others =
    # everyone except the implicit worst/optimised UE). The old `(_all_covered)`
    # truthiness guard silently dropped constraint_floor for max_min inputs
    # like "提升最差用户，不能降低其他用户".
    if _has_others and c_floor > 0:
        c_drop = round((1 - c_floor) * 100)
        _lbl, _lbl_en = _next_label()
        if c_floor >= 1.0:
            parts.append(f"{_lbl}: 其他用户不降低" if zh else f"{_lbl_en}: others no degradation")
        else:
            parts.append(
                f"{_lbl}: 其他用户降低≤{c_drop}%" if zh else f"{_lbl_en}: others drop ≤{c_drop}%")

    return "，".join(parts) if zh else ", ".join(parts)


def _run_grad_optimizer(optmode, gtype, snap, goal, pred_before, ss):
    """Dispatch to Joint Optimize or Power Only.

    Both modes go through the same goal dict → GNN gradient pipeline; the
    only difference is whether the association matrix A is optimised
    jointly (Joint Optimize) or kept fixed (Power Only).
    """
    import time as _time
    _init_P = (ss.P[:snap["ap_num"], :snap["ue_num"]].copy()
               if ss.is_modified else None)
    _common = dict(
        objective=gtype,
        target_ues=goal.get("target_ues", []),
        constraint_floor=goal.get("constraint_floor", 0.0),
        protected_floor=goal.get("protected_floor", 1.0),
        target_multiplier=goal.get("target_multiplier"),
        protected_ues=goal.get("protected_ues", []),
        secondary_type=goal.get("secondary_type"),
        baseline_rates=pred_before,
        min_rate_floor=goal.get("min_rate_floor"),
        ue_multipliers=goal.get("ue_multipliers"),
        ue_abs_floor=goal.get("ue_abs_floor"),
        bare_targets=goal.get("bare_targets") or [],
        secondary_weight_scale=goal.get("secondary_weight_scale", 1.0),
        init_P=_init_P,
    )

    _t0 = _time.time()
    if optmode == "Joint Optimize":
        A_opt, opt_P, _, steps = joint_optimize(
            rate_models[snap["snr"]], snap["loc_norm"],
            snap["ap_num"], snap["ue_num"],
            snap["A"], snap["snr"], _device,
            n_joint_iters=20, n_refine_iters=100,
            lr_a=0.1, lr_p=0.05, **_common)
        A_best = A_opt
    else:  # "Power Only" — A kept fixed, only power is tuned
        opt_P, _, steps = power_optimize(
            rate_models[snap["snr"]], snap["loc_norm"],
            snap["ap_num"], snap["ue_num"],
            snap["A"], snap["snr"], _device,
            n_iters=100, lr=0.05, **_common)
        A_best = snap["A"].copy()

    return dict(A_best=A_best, w_best=opt_P,
                elapsed=_time.time() - _t0, n_evals=steps,
                is_gradient=True)


def _loop_better(a: dict, b: dict) -> bool:
    """Closed-loop 'best so far' ordering: satisfied > higher score > less gap."""
    return ((a["satisfied"], a["score"], -a["total_gap"])
            > (b["satisfied"], b["score"], -b["total_gap"]))


def _format_loop_trace(trace: list, best_it: int, zh: bool) -> str:
    """Render the closed-loop validation trace appended to the report."""
    n = len(trace)
    head = ("🔁 闭环验证(数字孪生): 共 {n} 轮" if zh
            else "🔁 Closed-loop validation (digital twin): {n} round(s)").format(n=n)
    rows = []
    for it, vr in trace:
        tag = it + 1
        if vr["satisfied"]:
            body = "全部达标 ✓" if zh else "all requirements met ✓"
        else:
            body = _cl.format_failures(vr["failures"], zh)
            body += "  → 重生损失再试" if zh else "  → regenerate loss & retry"
        mark = ("（采用）" if zh else " (adopted)") if it == best_it else ""
        rows.append((f"&emsp;第{tag}轮: {body}{mark}" if zh
                     else f"&emsp;round {tag}: {body}{mark}"))
    inner = "<br>".join([head] + rows)
    lbl = "闭环优化过程" if zh else "Closed-loop trace"
    return (f'<details><summary>{lbl}</summary>'
            f'<div style="background:#f7f7fb;padding:8px 12px;border-radius:6px;'
            f'margin:6px 0;font-size:0.9em">{inner}</div></details>')


def _optimize_and_report(instruction: str, goal: dict, zh: bool, ss):
    """Shared closed-loop optimisation + report for both entry points.

    Optimise → validate on the GNN digital twin (pred_after) → if a hard
    requirement is unmet, regenerate the loss knobs (LLM, heuristic fallback)
    and re-optimise from the same baseline. Keeps the best-scoring round and
    commits only that one to the session state. See ``closed_loop.py``.
    """
    snap = _make_ss_snap(ss)
    ap_n, ue_n, snr = snap["ap_num"], snap["ue_num"], snap["snr"]

    A_before    = ss.A.copy()
    w_before    = ss.raw_w.copy()
    rate_before = ss.true_rate[:ss.ue_num].copy()
    pred_before = ss.pred_rate[:ss.ue_num].copy()

    # Dual baseline: when starting from an already-modified/optimized state,
    # also fetch the ORIGINAL allocation's rates so the report shows the true
    # gain (vs 原始分配) and avoids a coin-flip ✔/❌ on a converged point.
    orig_true = orig_pred = None
    if ss.is_modified and ss.get("A0") is not None:
        _, _, orig_pred = _predict_rate_only(snap["loc_norm"], ap_n, ue_n,
                                             ss.A0, ss.P0, snr)
        orig_true = rate_test[snr // 5, ss.test_idx, :ue_n].numpy()

    # Captured starting point so every loop iteration cold-starts identically
    # (only the regenerated loss differs between rounds).
    init_is_modified = ss.is_modified
    init_P           = ss.P.copy()
    snap_A0          = snap["A"].copy()

    _gtype   = goal.get("type", "max_sum_rate")
    _optmode = ss.get("optimizer_mode", "Joint Optimize")

    use_loop  = _cl.has_hard_requirement(goal)
    max_iters = _cl.MAX_ITERS if use_loop else 1

    # Live "thinking" step sink (set by the UI's st.status panel; may be None).
    _emit = getattr(ss, "_progress_cb", None)
    def emit(msg):
        if _emit:
            try:
                _emit(msg)
            except Exception:
                pass

    _intent = _goal_summary(goal, zh, ue_num=ue_n)
    if _intent:
        emit(("🎯 解析意图: " if zh else "🎯 Intent: ") + _intent)
    if use_loop:
        emit(("🔁 启用闭环验证(数字孪生),最多 %d 轮" % max_iters) if zh
             else "🔁 Closed-loop validation on (digital twin), up to %d rounds" % max_iters)

    cur_goal = goal
    best = None
    best_it = 0
    trace = []
    total_elapsed = 0.0
    total_steps = 0

    for it in range(max_iters):
        # Reset the optimiser's starting point for a clean per-round comparison.
        ss.is_modified = init_is_modified
        ss.P           = init_P.copy()
        snap["A"]      = snap_A0.copy()

        emit((f"⚙️ 第 {it+1} 轮:{_optmode} 优化中…" if zh
              else f"⚙️ Round {it+1}: optimizing ({_optmode})…"))
        result = _run_grad_optimizer(_optmode, _gtype, snap, cur_goal,
                                     pred_before, ss)
        total_elapsed += result["elapsed"]
        total_steps   += result["n_evals"]

        A_cand = result["A_best"]
        w_cand = result["w_best"]
        P_cand = _renorm(A_cand, w_cand)
        _, _, pred_after = _predict_rate_only(snap["loc_norm"], ap_n, ue_n,
                                              A_cand, P_cand, snr)
        vr = _cl.check_requirements(cur_goal, pred_before, pred_after, ue_n)
        trace.append((it, vr))

        if use_loop:
            if vr["satisfied"]:
                emit((f"✅ 数字孪生验证: 全部达标 ✓ ({result['elapsed']:.1f}s)" if zh
                      else f"✅ Digital-twin check: all met ✓ ({result['elapsed']:.1f}s)"))
            else:
                emit((f"❌ 数字孪生验证: {_cl.format_failures(vr['failures'], zh)}"
                      if zh else
                      f"❌ Digital-twin check: {_cl.format_failures(vr['failures'], zh)}"))

        cand = dict(A=A_cand, w=w_cand, vr=vr,
                    is_gradient=result.get("is_gradient", False))
        if best is None or _loop_better(vr, best["vr"]):
            best, best_it = cand, it

        if vr["satisfied"] or it == max_iters - 1:
            break

        # Regenerate the loss knobs: LLM first, deterministic heuristic fallback.
        emit(("🧠 重新生成损失函数…" if zh else "🧠 Regenerating loss function…"))
        revised = None
        _via = "heuristic"
        try:
            raw = _call_llm(_cl.build_revision_prompt(cur_goal, vr, zh))
            if raw:
                adj = _cl.parse_adjustments(raw)
                if adj is not None:
                    revised = _cl.apply_adjustments(cur_goal, adj, ue_n)
                    _via = "LLM"
        except Exception:
            revised = None
        if revised is None:
            revised = _cl.heuristic_reweight(cur_goal, vr, ue_n)
            _via = "heuristic"
        emit((f"   → 已调整损失权重 ({'LLM' if _via=='LLM' else '启发式'})" if zh
              else f"   → loss reweighted ({_via})"))
        cur_goal = revised

    if use_loop and not best["vr"]["satisfied"]:
        emit((f"📊 未完全达标,采用最接近的第 {best_it+1} 轮结果" if zh
              else f"📊 Not fully met — adopting closest round {best_it+1}"))
    elif use_loop:
        emit((f"📊 采用第 {best_it+1} 轮结果,生成报告" if zh
              else f"📊 Adopting round {best_it+1}, building report"))

    # Commit the best round to the session state and refresh once.
    snap["A"]      = snap_A0
    ss.A           = best["A"]
    ss.raw_w       = best["w"]
    ss.P           = _renorm(ss.A, ss.raw_w)
    ss.is_modified = True
    ss._delta_locked = False
    _refresh(ss)
    ss._delta_locked = True

    rate_after = ss.true_rate[:ss.ue_num].copy()
    pred_after = ss.pred_rate[:ss.ue_num].copy()
    A_opt, w_opt, P_opt = ss.A.copy(), ss.raw_w.copy(), ss.P.copy()

    _targets = [k for k in goal.get("target_ues", []) if k < ss.ue_num]
    added_links   = {(l, k) for l in range(ss.ap_num) for k in range(ss.ue_num)
                     if A_before[l, k] < 0.5 and ss.A[l, k] > 0.5}
    removed_links = {(l, k) for l in range(ss.ap_num) for k in range(ss.ue_num)
                     if A_before[l, k] > 0.5 and ss.A[l, k] < 0.5}
    ss.added_links   = added_links
    ss.removed_links = removed_links
    ss.changed_links = added_links | removed_links
    if _gtype == "targeted":
        ss.opt_targets = _targets
        if _targets:
            ss.selected_ue = _targets[0]
    elif _gtype == "max_min":
        worst = int(np.argmin(rate_before))
        ss.opt_targets = [worst]
        ss.selected_ue = worst
    else:
        ss.opt_targets = []
        ss.selected_ue = None

    P_before = _renorm(A_before, w_before)
    goal["_summary"] = _goal_summary(goal, zh, ue_num=ue_n)
    goal["_is_gradient"] = best.get("is_gradient", False)
    report = opt_format_report(
        goal, rate_before, rate_after,
        A_before, A_opt, P_before, P_opt,
        total_elapsed, total_steps, zh,
        pred_before=pred_before, pred_after=pred_after,
        orig_before=orig_true, orig_pred_before=orig_pred,
    )
    # (Closed-loop trace intentionally NOT appended to the persisted report —
    # the live "thinking" panel shows progress while running; the final report
    # stays clean. _format_loop_trace is kept for potential debugging use.)
    _add_to_chat_history(instruction, report, ss)
    return [f"💬 {report}"], True


def _run_optimization(instruction: str, ss):
    """Goal-directed optimizer: parse intent → search A/P → report.
    Returns None if LLM determines this is not an optimization request."""
    zh          = _is_chinese(instruction)
    resolved    = _resolve_pronouns(instruction, ss)
    snap = _make_ss_snap(ss)

    # Quick regex pre-filter: skip obviously non-optimization inputs
    if not detect_optimization_intent(instruction):
        return None

    goal = parse_goal(resolved, snap, _call_llm)
    with open("/tmp/debug_goal.txt", "a") as _df:
        _df.write(f"{goal}\n")
    if goal is None:
        # LLM says not optimization — use default since regex already confirmed
        goal = {"type": "max_sum_rate", "target_ues": [], "alpha": 1.0,
                "constraint_floor": 0.0, "raw_text": instruction, "zh": zh}

    return _optimize_and_report(instruction, goal, zh, ss)


def _run_optimization_with_goal(instruction: str, goal: dict, ss):
    """Run optimizer with a pre-parsed goal dict (from unified LLM)."""
    print(f"[DEBUG goal] {goal}")
    zh = goal.get("zh", _is_chinese(instruction))
    return _optimize_and_report(instruction, goal, zh, ss)


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
    sel_ap         = getattr(ss, "selected_ap", None)

    fig = go.Figure()

    # Build selected-UE data annotation
    _annotations = []
    if sel is not None and sel < ue_num:
        _ts = getattr(ss, "true_signal", None)
        _ps = getattr(ss, "pred_signal", None)
        _ti = getattr(ss, "true_interf", None)
        _pi = getattr(ss, "pred_interf", None)
        _tr = getattr(ss, "true_rate",   None)
        _pr = getattr(ss, "pred_rate",   None)
        if all(v is not None for v in [_ts, _ps, _ti, _pi, _tr, _pr]):
            _label = (
                f"<b>UE {sel}</b><br>"
                f"Signal:  {_ts[sel]:.1f} / {_ps[sel]:.1f} dBW<br>"
                f"Interf:  {_ti[sel]:.1f} / {_pi[sel]:.1f} dBW<br>"
                f"Rate:    {_tr[sel]:.2f} / {_pr[sel]:.2f} bps/Hz"
            )
            ux, uy = float(ue_loc[sel, 0]), float(ue_loc[sel, 1])
            _annotations.append(dict(
                x=ux, y=uy, xref="x", yref="y",
                text=_label,
                showarrow=True, arrowhead=2, arrowcolor=C_SEL,
                ax=60, ay=-60,
                bgcolor="rgba(26,26,46,0.88)",
                bordercolor=C_SEL, borderwidth=1.5,
                font=dict(color="white", size=10),
                align="left",
            ))

    # Build selected-AP annotation (served UEs + power allocation)
    if sel_ap is not None and sel_ap < ap_num:
        _served = [k for k in range(ue_num) if A[sel_ap, k] > 0.5]
        if _served:
            _ue_lines = "<br>".join(
                f"  UE {k}: {float(P[sel_ap, k]):.3f}" for k in _served
            )
            _total = sum(float(P[sel_ap, k]) for k in _served)
            _label_ap = (
                f"<b>AP {sel_ap}</b><br>"
                f"Serves {len(_served)} UE(s):<br>"
                f"{_ue_lines}<br>"
                f"Total power: {_total:.3f}"
            )
        else:
            _label_ap = f"<b>AP {sel_ap}</b><br>(idle, serves no UE)"
        ax_, ay_ = float(ap_loc[sel_ap, 0]), float(ap_loc[sel_ap, 1])
        _annotations.append(dict(
            x=ax_, y=ay_, xref="x", yref="y",
            text=_label_ap,
            showarrow=True, arrowhead=2, arrowcolor=C_SEL,
            ax=60, ay=60,
            bgcolor="rgba(26,26,46,0.88)",
            bordercolor=C_SEL, borderwidth=1.5,
            font=dict(color="white", size=10),
            align="left",
        ))

    # Trace 0 – edges (single trace, None separators)
    added   = getattr(ss, "added_links",   set())
    removed = getattr(ss, "removed_links", set())
    changed = getattr(ss, "changed_links", set())
    ex, ey, lx, ly, lt = [], [], [], [], []
    ax_e, ay_e = [], []   # added edges (yellow solid)
    rx_e, ry_e = [], []   # removed edges (yellow dashed)
    cx_e, cy_e = [], []   # changed edges (yellow, not added/removed)
    for l in range(ap_num):
        for k in range(ue_num):
            if A[l, k] > 0.5:
                pw = float(P[l, k])
                x1, y1 = ap_loc[l];  x2, y2 = ue_loc[k]
                if (l, k) in added:
                    ax_e += [x1, x2, None];  ay_e += [y1, y2, None]
                elif (l, k) in changed:
                    cx_e += [x1, x2, None];  cy_e += [y1, y2, None]
                else:
                    ex += [x1, x2, None];  ey += [y1, y2, None]
                lx.append((x1 + x2) / 2);  ly.append((y1 + y2) / 2)
                lt.append(f"{pw:.2f}")
            elif (l, k) in removed:
                x1, y1 = ap_loc[l];  x2, y2 = ue_loc[k]
                rx_e += [x1, x2, None];  ry_e += [y1, y2, None]

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
    ap_colors = [C_SEL if l == sel_ap else C_AP for l in range(ap_num)]
    ap_sizes  = [22    if l == sel_ap else 16   for l in range(ap_num)]
    fig.add_trace(go.Scatter(
        x=ap_loc[:, 0], y=ap_loc[:, 1],
        mode="markers+text",
        marker=dict(symbol="triangle-up", size=ap_sizes, color=ap_colors,
                    line=dict(color="white", width=1)),
        text=[f"AP{i}" for i in range(ap_num)],
        textposition="top center",
        textfont=dict(color="#ffaa88", size=10),
        customdata=list(range(ap_num)),
        name="AP",
        hovertemplate="AP %{customdata}<extra></extra>",
    ))  # trace 2

    # Trace 3 – UEs (showlegend=False: legend entry is handled by trace 4)
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
        showlegend=False,
        hovertemplate="UE %{customdata}<extra></extra>",
    ))  # trace 3

    # Trace 4 – legend-only UE entry (always green, not clickable)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="circle", size=10, color=C_UE,
                    line=dict(color="white", width=0.6)),
        name="UE", showlegend=True, hoverinfo="skip",
    ))  # trace 4

    # Trace 5 – added edges (yellow solid)
    fig.add_trace(go.Scatter(
        x=ax_e, y=ay_e, mode="lines",
        line=dict(color="#FFD700", width=2.5), opacity=0.9,
        hoverinfo="skip", showlegend=False,
    ))  # trace 5

    # Trace 6 – removed edges (yellow dashed)
    fig.add_trace(go.Scatter(
        x=rx_e, y=ry_e, mode="lines",
        line=dict(color="#FFD700", width=2.0, dash="dash"), opacity=0.85,
        hoverinfo="skip", showlegend=False,
    ))  # trace 6

    # Trace 7 – changed edges (yellow solid, e.g. power weight changes)
    fig.add_trace(go.Scatter(
        x=cx_e, y=cy_e, mode="lines",
        line=dict(color="#FFD700", width=2.5), opacity=0.9,
        hoverinfo="skip", showlegend=False,
    ))  # trace 7

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=AX,
        xaxis=dict(autorange="reversed", scaleanchor="y", scaleratio=1,
                   gridcolor=GRID, title="X (m)", color="white"),
        yaxis=dict(gridcolor=GRID, title="Y (m)", color="white"),
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
                    x=0.01, y=0.99),
        margin=dict(l=45, r=10, t=35, b=40),
        clickmode="event+select",
        dragmode=False,
        height=550,
        title=dict(text="Cell-Free Network Topology",
                   font=dict(color="white", size=13), x=0.5, xanchor="center"),
        annotations=_annotations,
    )
    return fig


def _fit_yrange(true_v, pred_v):
    finite = [v for v in list(true_v) + list(pred_v) if np.isfinite(v)]
    if not finite:
        return {}
    lo, hi = min(finite), max(finite)
    margin = max((hi - lo) * 0.15, 0.5)
    return {"range": [lo - margin, hi + margin]}


def _bar_fig(true_v, pred_v, ylabel, title, ue_num, sel, fit_range=False,
             highlights=None, orig_true=None, orig_pred=None,
             inc_solid=True):
    def _clean(arr):
        return [float(v) if np.isfinite(v) else None for v in arr]

    def _rgba(hex_color, alpha):
        """Convert '#RRGGBB' to 'rgba(r,g,b,alpha)'."""
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    _hl = set(highlights or [])
    x   = list(range(ue_num))
    if sel is not None:
        tc = [C_SEL if k == sel else C_TRUE for k in range(ue_num)]
        pc = [C_SEL if k == sel else C_PRED for k in range(ue_num)]
    else:
        tc = ["#FFD700" if k in _hl else C_TRUE for k in range(ue_num)]
        pc = ["#FFD700" if k in _hl else C_PRED for k in range(ue_num)]

    has_delta = (orig_true is not None and orig_pred is not None)

    if has_delta:
        # Main bars show min(current, original) — the stable base portion
        true_y = [min(float(c), float(o)) if np.isfinite(c) and np.isfinite(o)
                  else float(c) for c, o in zip(true_v, orig_true)]
        pred_y = [min(float(c), float(o)) if np.isfinite(c) and np.isfinite(o)
                  else float(c) for c, o in zip(pred_v, orig_pred)]
    else:
        true_y = _clean(true_v)
        pred_y = _clean(pred_v)

    traces = [
        go.Bar(x=x, y=true_y, name="True", marker_color=tc, opacity=0.9,
               showlegend=False, offsetgroup="true"),
        go.Bar(x=x, y=pred_y, name="Predicted", marker_color=pc, opacity=0.9,
               showlegend=False, offsetgroup="pred"),
        go.Scatter(x=[None], y=[None], mode="markers", name="True",
                   marker=dict(color=C_TRUE, size=10, symbol="square")),
        go.Scatter(x=[None], y=[None], mode="markers", name="Predicted",
                   marker=dict(color=C_PRED, size=10, symbol="square")),
    ]

    if has_delta:
        t_delta = [float(c) - float(o) for c, o in zip(true_v, orig_true)]
        p_delta = [float(c) - float(o) for c, o in zip(pred_v, orig_pred)]

        # Styles: solid fill vs outline-only
        def _solid(colors):
            return dict(color=colors, line=dict(color=colors, width=1))
        def _outline(colors):
            return dict(color=AX, line=dict(color=colors, width=2))

        _inc_style = _solid if inc_solid else _outline
        _dec_style = _outline if inc_solid else _solid

        # ── Increase portion (above base bar) ──
        t_inc_y    = [max(0, d) for d in t_delta]
        t_inc_base = [float(o) for o in orig_true]
        p_inc_y    = [max(0, d) for d in p_delta]
        p_inc_base = [float(o) for o in orig_pred]
        traces.append(go.Bar(
            x=x, y=t_inc_y, base=t_inc_base, offsetgroup="true",
            marker=_inc_style(tc),
            opacity=0.9, showlegend=False, hoverinfo="skip",
        ))
        traces.append(go.Bar(
            x=x, y=p_inc_y, base=p_inc_base, offsetgroup="pred",
            marker=_inc_style(pc),
            opacity=0.9, showlegend=False, hoverinfo="skip",
        ))

        # ── Decrease portion (above current bar) ──
        t_dec_y    = [max(0, -d) for d in t_delta]
        t_dec_base = [float(c) for c in true_v]
        p_dec_y    = [max(0, -d) for d in p_delta]
        p_dec_base = [float(c) for c in pred_v]
        traces.append(go.Bar(
            x=x, y=t_dec_y, base=t_dec_base, offsetgroup="true",
            marker=_dec_style(tc),
            opacity=0.9, showlegend=False, hoverinfo="skip",
        ))
        traces.append(go.Bar(
            x=x, y=p_dec_y, base=p_dec_base, offsetgroup="pred",
            marker=_dec_style(pc),
            opacity=0.9, showlegend=False, hoverinfo="skip",
        ))

    fig = go.Figure(traces)

    # Compute y-range considering both current and original values
    _yr_kwargs = {}
    if fit_range:
        if has_delta:
            all_v = np.concatenate([true_v, pred_v, orig_true, orig_pred])
            vmin = float(np.nanmin(all_v)); vmax = float(np.nanmax(all_v))
            margin = (vmax - vmin) * 0.1 + 0.5
            _yr_kwargs = {"range": [vmin - margin, vmax + margin]}
        else:
            _yr_kwargs = _fit_yrange(true_v, pred_v)

    fig.update_layout(
        barmode="group",
        title=dict(text=title, font=dict(color="white", size=11), x=0.5, xanchor="center"),
        yaxis=dict(title=dict(text=ylabel, font=dict(size=10)),
                   color="white", gridcolor=GRID, **_yr_kwargs),
        xaxis=dict(title="UE", color="white", gridcolor=GRID,
                   tickmode="linear", tick0=0, dtick=max(1, ue_num // 12)),
        paper_bgcolor=BG, plot_bgcolor=AX,
        font=dict(color="white", size=9),
        legend=dict(bgcolor="rgba(17,17,34,0.6)", font=dict(size=9, color="white"),
                    bordercolor=GRID, borderwidth=1),
        margin=dict(l=45, r=8, t=36, b=28),
        height=172,
    )
    return fig


# ─── Agent helpers ────────────────────────────────────────────────────────────

def _build_network_state(ss) -> str:
    """Summarise current network metrics for LLM context."""
    import numpy as _np
    rates = ss.true_rate[:ss.ue_num]
    avg_r = float(_np.mean(rates)); tot_r = float(_np.sum(rates))
    best_k = int(_np.argmax(rates)); worst_k = int(_np.argmin(rates))
    lines = [
        f"Current network state (SNR={ss.snr} dB, {ss.ap_num} APs, {ss.ue_num} UEs):",
        f"  [Summary] avg rate={avg_r:.2f} bps/Hz, total rate={tot_r:.2f} bps/Hz, "
        f"best UE{best_k}({rates[best_k]:.2f}), worst UE{worst_k}({rates[worst_k]:.2f})",
    ]
    for k in range(ss.ue_num):
        serving = [l for l in range(ss.ap_num) if ss.A[l, k] > 0.5]
        weights = [f"{ss.raw_w[l,k]:.2f}" for l in serving]
        ts = float(ss.true_signal[k]); ps = float(ss.pred_signal[k])
        ti = float(ss.true_interf[k]); pi = float(ss.pred_interf[k])
        tr = float(ss.true_rate[k]);   pr = float(ss.pred_rate[k])
        lines.append(
            f"  UE{k}: signal={ts:.1f}/{ps:.1f}dBW, interf={ti:.1f}/{pi:.1f}dBW, "
            f"rate={tr:.2f}/{pr:.2f}bps/Hz | APs={serving} weights={weights}"
        )
    return "\n".join(lines)


def _history_for_llm(entry_text: str) -> str:
    """Compact a stored assistant message for LLM prompt history: replace bulky
    HTML tables (per-UE heat-tables) with a [table] placeholder. The full HTML
    MUST stay in _chat_history itself — the chat panel renders straight from it
    (stripping at store time showed literal "[table]" to the user, 2026-07-03).
    """
    if isinstance(entry_text, str) and "<table" in entry_text:
        entry_text = re.sub(r'<table.*?</table>', '[table]', entry_text,
                            flags=re.DOTALL)
    return entry_text


def _add_to_chat_history(user_msg, assistant_msg, ss):
    if not ss.get("_chat_history"):
        ss._chat_history = []
    ss._chat_history.append({"user": user_msg, "assistant": assistant_msg})
    if len(ss._chat_history) > 10:
        ss._chat_history = ss._chat_history[-10:]


def _is_chinese(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def _regex_answer(text, ss):
    """Answer simple factual questions from session state. Returns string or None."""
    t = (text.lower()
         .replace('，', ',').replace('。', '').replace('：', ':')
         .replace('、', '').replace(' ', ''))
    zh = _is_chinese(text)
    ue_m = re.search(r'ue(\d+)', t)

    # Optimization/command keywords: bypass regex Q&A, let LLM/parser handle
    if re.search(r'(optimize|最大化|最优化|优化|boost|maximize|minimize|improve|提升|提高|改善|增强|设置|set\s)', t):
        return None
    # Also defer to optimizer if full intent detection says yes
    if detect_optimization_intent(text):
        return None

    # "How to improve" questions must be checked BEFORE network-wide SE,
    # because they also contain keywords like "网络" + "频谱效率".
    if re.search(r'(怎么|如何|怎样|how\s*to|how\s*can|what.{0,6}do).{0,20}(提升|提高|改善|优化|增加|增强|improve|boost|increase|optimize)', t) or \
       re.search(r'(提升|提高|改善|优化).{0,20}(频谱效率|rate|速率|吞吐|throughput)', t):
        return None  # Let LLM handle advice questions

    # All-UE metric listing ("每个/所有UE的速率/干扰/信号", "all UE rates/…").
    # Must run BEFORE the network-wide summary branch below, which also matches
    # "当前…速率" and would otherwise swallow the query with just avg/best/worst.
    _metric = all_ue_metric_query(text)
    if _metric:
        _tv, _pv = {"rate":   (ss.true_rate,   ss.pred_rate),
                    "interf": (ss.true_interf, ss.pred_interf),
                    "signal": (ss.true_signal, ss.pred_signal)}[_metric]
        return format_all_ue_metric(_tv, _pv, ss.ue_num, zh, metric=_metric)

    # Network-wide spectral efficiency / throughput / rate
    # Skip if the query references a specific UE (e.g. "UE3现在的速率")
    if not re.search(r'ue\d+', t) and \
       (re.search(r'(整体|总体|整个网络|网络|overall|total|average|avg|mean|平均|现在|当前|目前|这个网络|这网络).{0,12}(频谱效率|spectral|rate|速率|吞吐|throughput)', t) or
        re.search(r'(频谱效率|spectral|throughput|吞吐).{0,12}(整体|总体|整个|网络|overall|total|average|avg|mean|平均|现在|当前|目前)', t) or
        re.search(r'^(频谱效率|spectralefficiency|throughput)$', t)):
        rates = ss.true_rate[:ss.ue_num]
        avg = float(np.mean(rates)); total = float(np.sum(rates))
        best_k = int(np.argmax(rates)); worst_k = int(np.argmin(rates))
        if zh:
            return (f"当前网络频谱效率：均值 {avg:.2f} bits/s/Hz，"
                    f"所有 {ss.ue_num} 个 UE 之和 {total:.2f} bits/s/Hz。"
                    f"最高：UE{best_k}（{rates[best_k]:.2f}），最低：UE{worst_k}（{rates[worst_k]:.2f}）。")
        return (f"Network spectral efficiency: avg {avg:.2f} bits/s/Hz, "
                f"sum {total:.2f} bits/s/Hz across {ss.ue_num} UEs. "
                f"Best: UE{best_k} ({rates[best_k]:.2f}), Worst: UE{worst_k} ({rates[worst_k]:.2f}).")

    # Best / worst UE by rate
    if re.search(r'(highest|best|max|最高|最大|最好).{0,10}(rate|频谱效率|速率)|(rate|频谱效率|速率).{0,10}(highest|best|max|最高)', t):
        k = int(np.argmax(ss.true_rate[:ss.ue_num]))
        if zh:
            return f"UE{k} 的真实可达速率最高，为 {ss.true_rate[k]:.2f} bits/s/Hz。"
        return f"UE{k} has the highest true achievable rate: {ss.true_rate[k]:.2f} bits/s/Hz."
    if re.search(r'(lowest|worst|min|最低|最差|最小).{0,10}(rate|频谱效率|速率)|(rate|频谱效率|速率).{0,10}(lowest|worst|min|最低)', t):
        k = int(np.argmin(ss.true_rate[:ss.ue_num]))
        if zh:
            return f"UE{k} 的真实可达速率最低，为 {ss.true_rate[k]:.2f} bits/s/Hz。"
        return f"UE{k} has the lowest true achievable rate: {ss.true_rate[k]:.2f} bits/s/Hz."

    # Per-UE queries
    if ue_m:
        k = int(ue_m.group(1))
        if 0 <= k < ss.ue_num:
            if re.search(r'rate|速率|吞吐|频谱效率', t):
                if zh:
                    return (f"UE{k} — 真实速率：{ss.true_rate[k]:.2f} bits/s/Hz，"
                            f"预测：{ss.pred_rate[k]:.2f} bits/s/Hz。")
                return (f"UE{k} — true rate: {ss.true_rate[k]:.2f} bits/s/Hz, "
                        f"predicted: {ss.pred_rate[k]:.2f} bits/s/Hz.")
            if re.search(r'signal|信号', t):
                if zh:
                    return (f"UE{k} — 真实信号：{ss.true_signal[k]:.1f} dBW，"
                            f"预测：{ss.pred_signal[k]:.1f} dBW。")
                return (f"UE{k} — true signal: {ss.true_signal[k]:.1f} dBW, "
                        f"predicted: {ss.pred_signal[k]:.1f} dBW.")
            if re.search(r'interf|干扰', t):
                if zh:
                    return (f"UE{k} — 真实干扰：{ss.true_interf[k]:.1f} dBW，"
                            f"预测：{ss.pred_interf[k]:.1f} dBW。")
                return (f"UE{k} — true interference: {ss.true_interf[k]:.1f} dBW, "
                        f"predicted: {ss.pred_interf[k]:.1f} dBW.")
            if re.search(r'(how\s*many|几个|多少).{0,10}(ap|基站)|(ap|基站).{0,10}(how\s*many|几个|多少)', t):
                aps = [l for l in range(ss.ap_num) if ss.A[l, k] > 0.5]
                if zh:
                    return f"UE{k} 当前连接了 {len(aps)} 个 AP：{aps}。"
                return f"UE{k} is connected to {len(aps)} AP(s): {aps}."

    # General queries
    if re.search(r'(how\s*many|多少|几个).{0,6}(ue|用户)', t):
        return (f"当前场景共有 {ss.ue_num} 个 UE。" if zh
                else f"There are {ss.ue_num} UEs in this scenario.")
    if re.search(r'(how\s*many|多少|几个).{0,6}(ap|基站)', t):
        return (f"当前场景共有 {ss.ap_num} 个 AP。" if zh
                else f"There are {ss.ap_num} APs in this scenario.")
    if re.search(r'\bsnr\b|信噪比', t) and not re.search(r'设置|set|=|\d', t):
        return (f"当前 SNR 为 {ss.snr} dB。" if zh
                else f"The current SNR is {ss.snr} dB.")

    return None


# ─── Agent: rule-based instruction parser (no API key required) ───────────────


def _execute_agent_tool(name: str, inp: dict, ss):
    """Execute one agent tool call.  Returns (log_message, modified_topology).
    Side effect: appends affected UE indices to ss._cmd_ues for highlighting."""
    _cmd_ues = getattr(ss, "_cmd_ues", [])

    if name == "select_ue":
        k = int(inp["ue_index"])
        if not (0 <= k < ss.ue_num):
            return f"❌ UE index {k} out of range (0–{ss.ue_num - 1})", False
        ss.selected_ue = k
        _cmd_ues.append(k); ss._cmd_ues = _cmd_ues
        return f"✅ Selected UE{k}", False

    if name == "set_connection":
        l, k, conn = int(inp["ap_index"]), int(inp["ue_index"]), bool(inp["connected"])
        if not (0 <= l < ss.ap_num and 0 <= k < ss.ue_num):
            return "❌ AP or UE index out of range", False
        _cmd_ues.append(k); ss._cmd_ues = _cmd_ues
        if not conn and ss.A[:, k].sum() <= 1:
            return f"❌ AP{l} is the last serving AP for UE{k}, cannot disconnect", False
        if conn and ss.A[l, k] > 0.5:
            return f"ℹ️ AP{l} is already connected to UE{k}", False
        if not conn and ss.A[l, k] < 0.5:
            return f"ℹ️ AP{l} is not connected to UE{k}, no need to disconnect", False
        if conn and l not in _nearest_aps(ss.ap_loc, ss.ue_loc, k):
            return f"❌ AP{l} is not among the {MAX_AP_PER_UE} nearest APs for UE{k}", False
        ss.A[l, k]     = 1.0 if conn else 0.0
        ss.raw_w[l, k] = 1.0 if conn else 0.0
        ss.P           = _renorm(ss.A, ss.raw_w)
        ss.is_modified = True
        cl = getattr(ss, "changed_links", set())
        cl.add((l, k)); ss.changed_links = cl
        if conn:
            al = getattr(ss, "added_links", set())
            al.add((l, k)); ss.added_links = al
            # remove from removed if re-connecting a previously removed link
            rl = getattr(ss, "removed_links", set())
            rl.discard((l, k)); ss.removed_links = rl
        else:
            rl = getattr(ss, "removed_links", set())
            rl.add((l, k)); ss.removed_links = rl
            # remove from added if disconnecting a previously added link
            al = getattr(ss, "added_links", set())
            al.discard((l, k)); ss.added_links = al
        verb = "Connected" if conn else "Disconnected"
        return f"✅ {verb} AP{l} ↔ UE{k}", True

    if name == "set_power_weight":
        l, k, w = int(inp["ap_index"]), int(inp["ue_index"]), float(inp["weight"])
        if not (0 <= l < ss.ap_num and 0 <= k < ss.ue_num):
            return "❌ AP or UE index out of range", False
        _cmd_ues.append(k); ss._cmd_ues = _cmd_ues
        if ss.A[l, k] < 0.5:
            return f"❌ AP{l} is not connected to UE{k}", False
        ss.raw_w[l, k] = float(np.clip(w, 0.05, 1.0))
        ss.P           = _renorm(ss.A, ss.raw_w)
        ss.is_modified = True
        ss.selected_ue = k
        cl = getattr(ss, "changed_links", set())
        cl.add((l, k)); ss.changed_links = cl
        return f"✅ Power weight AP{l}→UE{k} set to {ss.raw_w[l,k]:.2f}", True

    if name == "connect_nearest_aps":
        k = int(inp["ue_index"])
        n = int(inp["n"])
        if not (0 <= k < ss.ue_num):
            return f"❌ UE index {k} out of range", False
        _cmd_ues.append(k); ss._cmd_ues = _cmd_ues
        n = max(1, min(n, ss.ap_num))
        dists   = np.sum((ss.ap_loc - ss.ue_loc[k]) ** 2, axis=1)
        nearest = np.argsort(dists)[:n]
        ss.A[:, k]     = 0.0
        ss.raw_w[:, k] = 0.0
        for l in nearest:
            ss.A[l, k]     = 1.0
            ss.raw_w[l, k] = 1.0
        ss.P           = _renorm(ss.A, ss.raw_w)
        ss.is_modified = True
        cl = getattr(ss, "changed_links", set())
        for l in nearest:
            cl.add((l, k))
        ss.changed_links = cl
        return f"✅ UE{k} connected to {n} nearest APs: {list(nearest)}", True

    if name == "reset_to_original":
        _load_sample(ss.test_idx)
        ss.changed_links = set(); ss.added_links = set(); ss.removed_links = set()
        return "✅ Reset to original topology", False

    return f"❌ 未知工具：{name}", False


def _parse_instruction(text: str, ss):
    """Regex fallback parser — used when Claude CLI is unavailable."""
    t = (text.lower()
         .replace('，', ',').replace('。', '').replace('：', ':')
         .replace('、', '').replace(' ', ''))
    logs = []
    needs_refresh = False

    def run(name, inp):
        nonlocal needs_refresh
        msg, mod = _execute_agent_tool(name, inp, ss)
        logs.append(msg)
        if mod:
            needs_refresh = True

    if re.search(r'重置|reset', t):
        run("reset_to_original", {})
        return logs, needs_refresh

    m = re.search(r'snr[^\d]*(\d+)', t) or re.search(r'信噪比[^\d]*(\d+)', t)
    if m:
        val = int(m.group(1))
        if val in range(0, 31, 5):
            ss.snr = val
            ss.true_rate = rate_test[val // 5, ss.test_idx, :ss.ue_num].numpy()
            logs.append(f"✅ SNR 设置为 {val} dB")
        else:
            logs.append(f"❌ SNR 必须是 0/5/10/15/20/25/30 dB 之一")
        return logs, False

    m = re.search(r'选[择中]?ue(\d+)', t) or re.search(r'(?:select|choose)(?:ue|u\.?e\.?)(\d+)', t)
    if m:
        run("select_ue", {"ue_index": int(m.group(1))})

    m = (re.search(r'ue(\d+).{0,6}最近.{0,4}(\d+).{0,4}ap', t) or
         re.search(r'ue(\d+).{0,10}nearest(\d+).{0,6}ap', t) or
         re.search(r'(?:give|connect|assign)\s+ue\s*(\d+)\s+(?:its\s+)?(\d+)\s+(?:nearest|closest)', text.lower()) or
         re.search(r'ue\s*(\d+)\s+(\d+)\s+(?:nearest|closest)', text.lower()))
    if m:
        run("connect_nearest_aps", {"ue_index": int(m.group(1)), "n": int(m.group(2))})
    else:
        m = (re.search(r'断开ap(\d+).{0,3}ue(\d+)', t) or
             re.search(r'disconnectap(\d+).{0,8}ue(\d+)', t))
        if m:
            run("set_connection", {"ap_index": int(m.group(1)), "ue_index": int(m.group(2)), "connected": False})
        else:
            m = (re.search(r'断开ue(\d+).{0,3}ap(\d+)', t) or
                 re.search(r'disconnectue(\d+).{0,8}ap(\d+)', t))
            if m:
                run("set_connection", {"ap_index": int(m.group(2)), "ue_index": int(m.group(1)), "connected": False})
        m = (re.search(r'连接ap(\d+).{0,3}ue(\d+)', t) or
             re.search(r'connectap(\d+).{0,8}ue(\d+)', t) or
             re.search(r'connect\s+ap\s*(\d+)\s+to\s+ue\s*(\d+)', text.lower()) or
             re.search(r'add\s+ap\s*(\d+)\s+to\s+ue\s*(\d+)', text.lower()))
        if m:
            run("set_connection", {"ap_index": int(m.group(1)), "ue_index": int(m.group(2)), "connected": True})
        else:
            m = (re.search(r'连接ue(\d+).{0,3}ap(\d+)', t) or
                 re.search(r'connectue(\d+).{0,8}ap(\d+)', t) or
                 re.search(r'connect\s+ue\s*(\d+)\s+to\s+ap\s*(\d+)', text.lower()))
            if m:
                run("set_connection", {"ap_index": int(m.group(2)), "ue_index": int(m.group(1)), "connected": True})

    ap_m = re.search(r'ap(\d+)', t)
    ue_m = re.search(r'ue(\d+)', t)
    if ap_m and ue_m and re.search(r'功率|power|weight', t):
        l, k = int(ap_m.group(1)), int(ue_m.group(1))
        if re.search(r'最大|max(?:imum)?', t):
            run("set_power_weight", {"ap_index": l, "ue_index": k, "weight": 1.0})
        elif re.search(r'最小|min(?:imum)?', t):
            run("set_power_weight", {"ap_index": l, "ue_index": k, "weight": 0.05})
        else:
            wm = (re.search(r'(?:功率|power|weight|调到|设为|设到|to)[^\d]*(\d+\.?\d*)', t) or
                  re.search(r'(\d+\.?\d*)(?:[^\d]{0,4})(?:功率|power)', t))
            if wm:
                run("set_power_weight", {"ap_index": l, "ue_index": k, "weight": float(wm.group(1))})

    if not logs:
        logs.append("❓ Command not recognised.")
    return logs, needs_refresh


def _build_llm_prompt(instruction: str, ss) -> str:
    network_state = _build_network_state(ss)

    # Embed recent conversation history
    chat_hist = ss.get("_chat_history") or []
    history_text = ""
    if chat_hist:
        lines = []
        for entry in chat_hist[-3:]:
            lines.append(f"User: {entry['user']}")
            lines.append(f"Assistant: {_history_for_llm(entry['assistant'])}")
        history_text = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

    return f"""You are an intelligent assistant for a cell-free massive MIMO network digital twin.
You can execute network control commands OR answer questions about the current network state.

{network_state}

{history_text}INSTRUCTIONS:
- Always reply in the same language as the user (Chinese if Chinese, English if English).
- If the user input is a GREETING or CASUAL CHAT (hello, 你好, good morning, etc.):
  Return exactly: ANSWER: <friendly short reply, then briefly mention what you can do>
- If the user input is a COMMAND (select UE, change connections, adjust power, reset):
  Return ONLY a JSON array of actions. No explanation, no markdown fences.
- If the user input is a QUESTION about the network:
  Return exactly: ANSWER: <concise answer in 1-2 sentences>

Available command actions:
  {{"action":"select_ue","ue_index":N}}
  {{"action":"set_connection","ap_index":N,"ue_index":M,"connected":true|false}}
  {{"action":"connect_nearest_aps","ue_index":N,"n":M}}
  {{"action":"set_power_weight","ap_index":N,"ue_index":M,"weight":W}}   (W: 0.05–1.0)
  {{"action":"reset"}}

Command examples:
"choose UE3"                            → [{{"action":"select_ue","ue_index":3}}]
"cut the link between AP9 and UE25"    → [{{"action":"set_connection","ap_index":9,"ue_index":25,"connected":false}}]
"give UE0 its 3 closest base stations" → [{{"action":"connect_nearest_aps","ue_index":0,"n":3}}]
"boost AP4's power to UE14 to max"     → [{{"action":"set_power_weight","ap_index":4,"ue_index":14,"weight":1.0}}]
"restore everything"                    → [{{"action":"reset"}}]

Question examples:
"which UE has the highest rate?"        → ANSWER: UE3 has the highest true rate at 4.21 bits/s/Hz.
"how many APs serve UE2?"              → ANSWER: UE2 is currently connected to 3 APs (AP0, AP4, AP7).
"UE5的信号功率是多少？"                  → ANSWER: UE5 — true signal: -87.3 dBW, predicted: -88.1 dBW.

User: {instruction}"""


def _call_llm(prompt: str):
    """Return LLM response text.
    0. DeepSeek via Tencent LKEAP — domestic-accessible (set DeepSeek_API_KEY)
    1. Groq API            — free tier, fast (set GROQ_API_KEY)
    2. Anthropic API       — fallback (set ANTHROPIC_API_KEY)
    Returns None if unavailable → caller uses regex parser.
    """
    # Track the last error across backends; cleared on the first success so the
    # UI only flags errors when *all* backends failed.
    _accumulated_err = None

    import requests as _req

    def _ok(label, text):
        st.session_state["_last_llm_backend"] = label
        st.session_state["_last_llm_error"] = None
        return text

    # Backend 0: DeepSeek via Tencent Cloud TokenHub (OpenAI-compatible, domestic).
    # Listed first because it's reachable from CN IDC where Groq/Anthropic are
    # blocked. Intent parsing fits comfortably within deepseek-v4-flash.
    ds_key = os.environ.get("DeepSeek_API_KEY")
    if ds_key:
        try:
            r = _req.post(
                "https://tokenhub.tencentmaas.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {ds_key}",
                         "Content-Type": "application/json"},
                json={"model": "deepseek-v4-flash",
                      "messages": [{"role": "user", "content": prompt}],
                      # deepseek-v4-flash is a REASONING model: completion =
                      # reasoning_tokens + content. The unified parse prompt is
                      # ~3k tokens and reasoning alone can run 300-560 tokens,
                      # so 512 intermittently truncates the JSON (finish=length,
                      # empty content) → regex fallback → wrong "max_sum_rate".
                      # 2048 leaves ample headroom; the model stops early anyway.
                      "max_tokens": 2048,
                      "stream": False},
                timeout=30,
            )
            r.raise_for_status()
            return _ok("DeepSeek · deepseek-v4-flash (Tencent TokenHub)",
                       r.json()["choices"][0]["message"]["content"].strip())
        except _req.HTTPError:
            _accumulated_err = f"DeepSeek HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            _accumulated_err = f"DeepSeek {type(e).__name__}: {e}"

    # Backend 1: Groq (free tier — llama-3.3-70b-versatile, via requests)
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        try:
            r = _req.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}",
                         "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile",
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 512},
                timeout=30,
            )
            r.raise_for_status()
            return _ok("Groq · llama-3.3-70b-versatile",
                       r.json()["choices"][0]["message"]["content"].strip())
        except _req.HTTPError:
            _accumulated_err = f"Groq HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            _accumulated_err = f"Groq {type(e).__name__}: {e}"

    # Backend 2: Anthropic (claude-haiku-4-5, via requests)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            r = _req.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key,
                         "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": "claude-haiku-4-5",
                      "max_tokens": 512,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=30,
            )
            r.raise_for_status()
            return _ok("Anthropic · claude-haiku-4-5",
                       r.json()["content"][0]["text"].strip())
        except _req.HTTPError:
            _accumulated_err = f"Anthropic HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            _accumulated_err = f"Anthropic {type(e).__name__}: {e}"

    st.session_state["_last_llm_backend"] = None
    st.session_state["_last_llm_error"] = _accumulated_err
    return None


def _answer_post_query(query: str, ss, zh: bool) -> str:
    """Answer the trailing question of a compound "optimize + query" instruction.

    Runs AFTER the optimizer committed + _refresh, so ss reflects the optimized
    state. Regex Q&A first (free, fast); LLM fallback with the fresh network
    state; graceful notice if both fail (report already shown, never lost).

    NO history side-effect here: the extracted query is NOT something the user
    typed as its own message, so it must never become a "user" bubble. The
    caller merges the returned answer into the SAME history entry as the
    optimization report (user feedback 2026-07-03).
    """
    ans = _regex_answer(query, ss)
    if not ans:
        raw = _call_llm(build_unified_prompt(query, _build_network_state(ss)))
        if raw:
            r = parse_unified_response(raw, ss.ue_num, raw_text=query)
            if r and r.get("action") in ("chat", "query"):
                ans = r.get("response") or None
    if ans:
        return f"💬 {ans}"
    return (f"💬 （查询部分未能解析：{query}）" if zh
            else f"💬 (could not answer the query part: {query})")


def _llm_parse_instruction(instruction: str, ss):
    """Single LLM call handles all inputs: routing + parsing + answering."""
    # Split multi-line input into separate instructions
    lines = [l.strip() for l in instruction.split("\n") if l.strip()]
    if len(lines) > 1:
        all_msgs, any_refresh = [], False
        for line in lines:
            msgs, refresh = _llm_parse_instruction(line, ss)
            if msgs:
                all_msgs.extend(msgs)
            any_refresh = any_refresh or refresh
        return all_msgs, any_refresh

    zh = _is_chinese(instruction)

    # ── Build unified prompt and call LLM ──
    network_state = _build_network_state(ss)
    chat_hist = ss.get("_chat_history") or []
    history_text = ""
    if chat_hist:
        h_lines = []
        for entry in chat_hist[-3:]:
            h_lines.append(f"User: {entry['user']}")
            h_lines.append(f"Assistant: {_history_for_llm(entry['assistant'])}")
        history_text = "Recent conversation:\n" + "\n".join(h_lines) + "\n\n"

    prompt = build_unified_prompt(instruction, network_state, history_text)
    raw = _call_llm(prompt)
    with open("/tmp/llm_debug.log", "a") as _f:
        _f.write(f"instruction={repr(instruction)}\nraw={repr(raw)[:300]}\n---\n")

    if raw:
        result = parse_unified_response(raw, ss.ue_num, raw_text=instruction)
        if result:
            action = result.get("action")

            # ── Chat / Query → direct text response ──
            if action in ("chat", "query"):
                answer = result.get("response", "")
                _add_to_chat_history(instruction, answer, ss)
                # Auto-select UE: check instruction first, then answer
                _ue_m = (re.search(r'[Uu][Ee]\s*(\d+)', instruction)
                         or re.search(r'用户\s*(\d+)', instruction)
                         or re.search(r'[Uu][Ee]\s*(\d+)', answer)
                         or re.search(r'用户\s*(\d+)', answer))
                if _ue_m:
                    _k = int(_ue_m.group(1))
                    if 0 <= _k < ss.ue_num:
                        ss.selected_ue = _k
                return [f"💬 {answer}"], False

            # ── Command → execute actions ──
            if action == "command":
                actions = result.get("actions", [])
                logs, needs_refresh = [], False
                action_map = {
                    "select_ue":           lambda a: ("select_ue",           {"ue_index": a["ue_index"]}),
                    "set_connection":      lambda a: ("set_connection",      {"ap_index": a["ap_index"], "ue_index": a["ue_index"], "connected": a["connected"]}),
                    "connect_nearest_aps": lambda a: ("connect_nearest_aps", {"ue_index": a["ue_index"], "n": a["n"]}),
                    "set_power_weight":    lambda a: ("set_power_weight",    {"ap_index": a["ap_index"], "ue_index": a["ue_index"], "weight": a["weight"]}),
                    "reset":               lambda a: ("reset_to_original",   {}),
                }
                # Handle nav/SNR commands from LLM
                for act in actions:
                    name = act.get("action", "")
                    if name == "next_topology":
                        if ss.test_idx < _N_TEST - 1:
                            _load_sample(ss.test_idx + 1)
                            logs.append(f"✅ Switched to topology {ss.test_idx}")
                        else:
                            logs.append("❌ Already at the last topology")
                    elif name == "prev_topology":
                        if ss.test_idx > 0:
                            _load_sample(ss.test_idx - 1)
                            logs.append(f"✅ Switched to topology {ss.test_idx}")
                        else:
                            logs.append("❌ Already at the first topology")
                    elif name == "set_snr":
                        val = int(act.get("value", -1))
                        if val in range(0, 31, 5):
                            ss.snr = val
                            ss.true_rate = rate_test[val // 5, ss.test_idx, :ss.ue_num].numpy()
                            logs.append(f"✅ SNR set to {val} dB")
                            needs_refresh = True
                        else:
                            logs.append(f"❌ SNR must be 0/5/10/15/20/25/30 dB, got {val}")
                    elif name in action_map:
                        tool_name, tool_inp = action_map[name](act)
                        msg, mod = _execute_agent_tool(tool_name, tool_inp, ss)
                        logs.append(msg)
                        if mod:
                            needs_refresh = True
                    else:
                        logs.append(f"❌ Unknown action: {name}")
                _add_to_chat_history(instruction, " | ".join(logs), ss)
                return logs, needs_refresh

            # ── Optimization → run optimizer ──
            if action == "optimization":
                # Build goal dict from unified response
                goal = {k: v for k, v in result.items() if k != "action"}
                # Compound "optimize + query": pop the trailing question so it
                # never reaches the optimizer; answered after it finishes.
                post_query = goal.pop("post_query", None)
                goal["raw_text"] = instruction
                goal["zh"] = zh
                # Switch to power-only when user explicitly requests no topology change
                if goal.pop("power_only", False):
                    ss.optimizer_mode = "Power Only"
                    print(f"[DEBUG] power_only=True → optimizer_mode={ss.optimizer_mode}")
                else:
                    ss.optimizer_mode = "Joint Optimize"
                msgs, needs_refresh = _run_optimization_with_goal(instruction, goal, ss)
                if post_query:
                    ans_msg = _answer_post_query(post_query, ss, zh)
                    # Merge into the report's history entry: the chat panel
                    # renders from _chat_history, and the extracted query must
                    # not appear as a "user" bubble the user never typed.
                    hist = ss.get("_chat_history")
                    if hist and hist[-1]["user"] == instruction:
                        hist[-1]["assistant"] += "\n\n" + ans_msg
                    else:  # report entry missing (unexpected) — attach to the
                        # real user input, never to the extracted query text
                        _add_to_chat_history(instruction, ans_msg, ss)
                    msgs = list(msgs) + [ans_msg]
                return msgs, needs_refresh

    # ── Fallback: regex Q&A → regex commands → friendly default ──
    regex_ans = _regex_answer(instruction, ss)
    if regex_ans:
        _add_to_chat_history(instruction, regex_ans, ss)
        return [f"💬 {regex_ans}"], False

    _opt_result = _run_optimization(instruction, ss)
    if _opt_result is not None:
        return _opt_result

    logs, needs_refresh = _parse_instruction(instruction, ss)
    if logs[0].startswith("❓"):
        answer = ('你好！我是无线网络数字孪生助手，你可以输入优化指令（如「提升用户3的速率」）或提问（如「当前网络状态」）。'
                  if zh else
                  "Hello! I'm the wireless network digital twin assistant. "
                  "You can enter optimization commands (e.g. 'boost UE3 rate') "
                  "or ask questions (e.g. 'current network status').")
        _add_to_chat_history(instruction, answer, ss)
        return [f"💬 {answer}"], False
    _add_to_chat_history(instruction, " | ".join(logs), ss)
    return logs, needs_refresh


# ─── Initialise session state ─────────────────────────────────────────────────
if "snr" not in st.session_state:
    st.session_state.snr = 15
    _load_sample(0)
if "changed_links" not in st.session_state:
    st.session_state.changed_links = set()
if "added_links" not in st.session_state:
    st.session_state.added_links = set()
if "removed_links" not in st.session_state:
    st.session_state.removed_links = set()

ss = st.session_state

# ─── Title ────────────────────────────────────────────────────────────────────
import pathlib as _pl
import re as _re
_title_svg_path = _pl.Path(__file__).parent / "title.svg"
if _title_svg_path.is_file():
    _title_svg = _title_svg_path.read_text()
    _title_svg = _re.sub(r'\s(width|height)="[^"]*"', "", _title_svg, count=2)
    _title_html = (
        "<div style='display:flex;justify-content:center;align-items:center;"
        "width:100%;pointer-events:none'>"
        f"<div id='radiomap-title-svg' style='max-width:520px;width:100%'>{_title_svg}</div>"
        "</div>"
    )
else:
    _title_html = (
        "<h1 style='text-align:center;margin:0;font-size:1.8rem'>"
        "AI Agent for Resource Allocation in Wireless Networks</h1>"
    )
st.markdown(
    "<style>"
    "div[data-testid='stElementContainer']:has(iframe[height='0']){"
    "height:0!important;min-height:0!important;margin:0!important;"
    "padding:0!important;overflow:hidden;}"
    "@media (max-width:768px){#radiomap-email-link{display:none!important;}"
    "#radiomap-title-svg{width:80%!important;}}"
    "</style>"
    "<div style='position:relative;margin-bottom:0;padding-top:0.5rem'>"
    f"{_title_html}"
    "<a id='radiomap-email-link' "
    "style='position:absolute;right:0;top:50%;transform:translateY(-50%);"
    "font-size:0.9rem;text-decoration:none;color:#888;cursor:pointer;"
    "user-select:none' "
    "title='Click to copy'>"
    "📧 binyang_2020@163.com</a>"
    "</div>",
    unsafe_allow_html=True,
)
components.html(
    """
    <script>
      (function() {
        function bind() {
          var doc, nav;
          try { doc = window.parent.document; nav = window.parent.navigator; }
          catch (e) { return; }                // cross-origin: degrade quietly
          var link = doc.getElementById('radiomap-email-link');
          if (!link || link.dataset.bound === '1') return;
          link.dataset.bound = '1';
          link.removeAttribute('href');        // guarantee it never navigates
          var orig = link.innerHTML;
          link.addEventListener('click', function(ev) {
            ev.preventDefault();               // no redirect, ever
            ev.stopPropagation();
            var text = 'binyang_2020@163.com';
            var done = function() {
              link.innerHTML = '✓ Copied';
              link.style.color = '#4caf50';
              setTimeout(function() {
                link.innerHTML = orig;
                link.style.color = '#888';
              }, 1500);
            };
            var fallback = function() {
              try {
                var ta = doc.createElement('textarea');
                ta.value = text;
                ta.style.position = 'fixed';
                ta.style.left = '-9999px';
                doc.body.appendChild(ta);
                ta.select();
                doc.execCommand('copy');
                doc.body.removeChild(ta);
                done();
              } catch (e) {}
            };
            try {
              if (nav.clipboard && nav.clipboard.writeText) {
                nav.clipboard.writeText(text).then(done).catch(fallback);
              } else {
                fallback();
              }
            } catch (e) { fallback(); }
          });
        }
        bind();
        // Streamlit reruns replace the <a>; keep re-binding the fresh element.
        setInterval(bind, 600);
      })();
    </script>
    """,
    height=0,
)

# ─── AI Agent ─────────────────────────────────────────────────────────────────
# Always show the greeting + preset commands at the top of the chat,
# then any conversation history below it.
chat_hist = ss.get("_chat_history") or []
with st.chat_message("assistant"):
    st.markdown(
        "Hello, I am **Wireless Network Agent**. I can help you optimize "
        "access point (AP) and user equipment (UE) association, and power allocation "
        "in the following cell-free network."
    )
    st.markdown(
        "<div style='margin-bottom:-0.75rem'>You can manage the network through "
        "natural-language commands. To get started, click one of the example "
        "commands below.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("""
<style>
div[class*="st-key-preset_cmd_"] {
    margin: 0 !important;
    padding: 0 !important;
}
div[class*="st-key-preset_cmd_"] button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
    height: auto !important;
    line-height: 1.55 !important;
    width: auto !important;
    color: inherit !important;
    font-weight: normal !important;
    justify-content: flex-start !important;
    text-align: left !important;
}
div[class*="st-key-preset_cmd_"] button p {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.55 !important;
}
div[class*="st-key-preset_cmd_"] button:hover { color: #4a9eff !important; }
div[class*="st-key-preset_cmd_"] button:focus { box-shadow: none !important; }
[data-testid="stChatMessage"] [data-testid="stVerticalBlock"] {
    gap: 0.75rem !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdown"] p {
    line-height: 1.55 !important;
    margin-bottom: 0.75rem !important;
}
</style>
""", unsafe_allow_html=True)
    _PRESET_GROUPS = [
        ("Chat", [
            "what is cell-free",
            "introduce this network",
            "current connection rules",
        ]),
        ("Query", [
            "show UE3 metrics",
            "which UEs does AP1 serve",
            "AP7 power allocation to UE5",
        ]),
        ("Action", [
            "connect AP4 to UE12",
            "give UE3 its 2 nearest APs",
            "set AP4 power to UE14 to max",
        ]),
        ("Optimization", [
            "maximum sum throughput",
            "improve user fairness",
            "improve the worst UE while not change topology",
        ]),
    ]
    for _gi, (_gname, _gcmds) in enumerate(_PRESET_GROUPS):
        _cols = st.columns([1.2, 2.5, 2.5, 3.2])
        with _cols[0]:
            st.markdown(f"**{_gi + 1}\\. {_gname}**")
        for _ci, _cmd in enumerate(_gcmds):
            with _cols[_ci + 1]:
                if st.button(f"• `{_cmd}`", key=f"preset_cmd_{_gi}_{_ci}"):
                    ss._preset_pending = _cmd
    st.markdown("\nOr type your own request below.")

# Conversation history (rendered below the persistent greeting card)
for entry in chat_hist[-6:]:
    with st.chat_message("user"):
        st.write(entry["user"])
    with st.chat_message("assistant"):
        st.markdown(entry["assistant"], unsafe_allow_html=True)

with st.form("agent_form", clear_on_submit=True):
    agent_col1, agent_col2 = st.columns([5, 1])
    with agent_col1:
        agent_instruction = st.text_input(
            label="command",
            placeholder="English or Chinese both work",
            label_visibility="collapsed",
        )
    with agent_col2:
        agent_run = st.form_submit_button("Enter", use_container_width=True, type="primary")

# A preset button click on the greeting card behaves like submitting that text.
_preset_pending = ss.pop("_preset_pending", None)
if _preset_pending:
    agent_instruction = _preset_pending
    agent_run = True

_backend = ss.get("_last_llm_backend")
if _backend:
    st.caption(f"🤖 Last response via **{_backend}**")
_llm_err = ss.get("_last_llm_error")
if _llm_err:
    st.caption(f"⚠️ {_llm_err}")

# Reserve a slot for the "thinking" panel directly under the dialog box so it
# renders ABOVE Mouse Controls. Mouse Controls is drawn before the blocking LLM
# call, so it stays visible during thinking and sits below the thinking panel.
_think_slot = st.container()

with st.expander("🖱️ Mouse Controls"):
    st.markdown("""
**Select a UE** &nbsp;→&nbsp; Click any green circle on the network graph. The circle turns gold when selected. Click again or click empty area to deselect.

**Toggle AP connection** &nbsp;→&nbsp; Select a UE first, then click an AP (triangle) to connect or disconnect it. The last serving AP cannot be removed.

**Adjust power** &nbsp;→&nbsp; Select a UE — power weight sliders for each serving AP appear at the bottom of the page.

**Browse samples** &nbsp;→&nbsp; Use **< Prev** / **Next >** to switch between test scenarios.

**Reset** &nbsp;→&nbsp; Restores the current sample to its original topology and power.

**SNR** &nbsp;→&nbsp; Use the SNR slider (bottom right) to change the noise level (0 – 30 dB).
""")

if agent_run:
    if not agent_instruction.strip():
        st.warning("Please enter a command.")
    else:
        log_query(agent_instruction)
        ss.changed_links = set(); ss.added_links = set(); ss.removed_links = set()
        _is_opt = detect_optimization_intent(agent_instruction)
        if _is_opt:
            spinner_text = (
                "正在优化网络，稍等…"
                if _is_chinese(agent_instruction)
                else "Optimizing network…"
            )
        else:
            spinner_text = "Thinking…"
        ss._cmd_ues = []  # collect affected UEs during command execution
        _zh_status = _is_chinese(agent_instruction)
        # Live "thinking" panel: shows ONLY the current step. Each step
        # overwrites the status label in place (no accumulation) — once a
        # step is done it disappears, replaced by the next one. Rendered into
        # _think_slot so it appears under the dialog box and ABOVE Mouse Controls.
        with _think_slot:
            with st.status(spinner_text, expanded=False) as _status:
                def _on_step(msg):
                    _status.update(label=msg)
                ss._progress_cb = _on_step
                _on_step("🔎 " + ("解析指令…" if _zh_status else "Parsing request…"))
                try:
                    _, needs_refresh = _llm_parse_instruction(agent_instruction, ss)
                finally:
                    ss._progress_cb = None
                _status.update(
                    label=("✅ 思考完成" if _zh_status else "✅ Done thinking"),
                    state="complete", expanded=False)
        # Highlight affected UEs in bar charts and select last mentioned UE in topology
        if not _is_opt and getattr(ss, "_cmd_ues", []):
            ss.opt_targets = sorted(set(ss._cmd_ues))
            ss.selected_ue = ss._cmd_ues[-1]
        if needs_refresh and not _is_opt:
            # _run_optimization already called _refresh internally
            ss._delta_locked = False
            _refresh(ss)
        st.rerun()

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
        ss.changed_links = set(); ss.added_links = set(); ss.removed_links = set()
        pt    = pts[0]
        curve = pt["curve_number"]
        pidx  = pt["point_number"]

        if curve == UE_TRACE:
            k = pidx
            ss.selected_ap = None  # mutual exclusion: UE selection clears AP selection
            ss.selected_ue = k if ss.selected_ue != k else None
            st.rerun()

        elif curve == AP_TRACE and ss.selected_ue is None:
            # No UE selected → toggle AP inspection (show served UEs + power)
            l = pidx
            ss.selected_ap = l if ss.selected_ap != l else None
            st.rerun()

        elif curve == AP_TRACE and ss.selected_ue is not None:
            l, k = pidx, ss.selected_ue
            # Guard: cannot disconnect last serving AP
            if ss.A[l, k] > 0.5 and ss.A[:, k].sum() <= 1:
                pass
            # Guard: can only connect to nearest MAX_AP_PER_UE APs
            elif ss.A[l, k] < 0.5 and l not in _nearest_aps(ss.ap_loc, ss.ue_loc, k):
                pass
            else:
                was_connected = ss.A[l, k] > 0.5
                ss.A[l, k]    = 1.0 - ss.A[l, k]
                ss.raw_w[l, k] = 1.0 if ss.A[l, k] > 0.5 else 0.0
                ss.P           = _renorm(ss.A, ss.raw_w)
                ss.is_modified = True
                cl = getattr(ss, "changed_links", set())
                cl.add((l, k)); ss.changed_links = cl
                if was_connected:
                    rl = getattr(ss, "removed_links", set())
                    rl.add((l, k)); ss.removed_links = rl
                else:
                    al = getattr(ss, "added_links", set())
                    al.add((l, k)); ss.added_links = al
                ss._delta_locked = False
                _refresh(ss)
                st.rerun()

with col_r:
    _hl = getattr(ss, "opt_targets", [])
    _prev_ts = getattr(ss, "prev_true_signal", None)
    _prev_ps = getattr(ss, "prev_pred_signal", None)
    _prev_ti = getattr(ss, "prev_true_interf", None)
    _prev_pi = getattr(ss, "prev_pred_interf", None)
    _prev_tr = getattr(ss, "prev_true_rate",   None)
    _prev_pr = getattr(ss, "prev_pred_rate",   None)
    st.plotly_chart(
        _bar_fig(ss.true_signal, ss.pred_signal,
                 "dBW", "Signal Power", ss.ue_num, ss.selected_ue,
                 fit_range=True, highlights=_hl,
                 orig_true=_prev_ts, orig_pred=_prev_ps, inc_solid=False),
        use_container_width=True, key="b_sig",
    )
    st.plotly_chart(
        _bar_fig(ss.true_interf, ss.pred_interf,
                 "dBW", "Interference Power", ss.ue_num, ss.selected_ue,
                 fit_range=True, highlights=_hl,
                 orig_true=_prev_ti, orig_pred=_prev_pi, inc_solid=False),
        use_container_width=True, key="b_int",
    )
    st.plotly_chart(
        _bar_fig(ss.true_rate, ss.pred_rate,
                 "bits/s/Hz", f"Achievable Rate  (SNR = {ss.snr} dB)",
                 ss.ue_num, ss.selected_ue, highlights=_hl,
                 orig_true=_prev_tr, orig_pred=_prev_pr, inc_solid=True),
        use_container_width=True, key="b_rate",
    )

_snr_opts = list(range(0, 31, 5))
_snr_idx  = _snr_opts.index(ss.snr) if ss.snr in _snr_opts else 3

with col_l:
    # ─── Controls (left column: Prev/Next/Reset + Power) ─────────────────────
    if "optimizer_mode" not in ss:
        ss.optimizer_mode = "Joint Optimize"

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("◀ Prev", use_container_width=True,
                     disabled=ss.test_idx == 0):
            _load_sample(ss.test_idx - 1)
            st.rerun()
    with c2:
        if st.button("Next ▶", use_container_width=True,
                     disabled=ss.test_idx >= _N_TEST - 1):
            _load_sample(ss.test_idx + 1)
            st.rerun()
    with c3:
        if st.button("Reset", use_container_width=True):
            _load_sample(ss.test_idx)
            st.rerun()

    st.caption(
        f"Sample #{ss.test_idx}  |  APs: {ss.ap_num}  |  UEs: {ss.ue_num}  "
        f"|  Connections: {int(ss.A.sum())}"
    )

with col_r:
    # ─── SNR controls (right column, aligned with bar charts) ─────────────────
    s1, s2, s3 = st.columns(3)
    with s1:
        if st.button("SNR −", use_container_width=True,
                     disabled=_snr_idx == 0):
            new_snr = _snr_opts[_snr_idx - 1]
            ss.snr = new_snr
            ss._delta_locked = False
            if ss.is_modified:
                _refresh(ss)
            else:
                _snapshot_before(ss)
                ss.true_rate = rate_test[new_snr // 5, ss.test_idx, :ss.ue_num].numpy()
                ps, pi, pr = _predict(ss.loc_norm, ss.ap_num, ss.ue_num, ss.A, ss.P, new_snr)
                ss.pred_signal = ps; ss.pred_interf = pi; ss.pred_rate = pr
            st.rerun()
    with s2:
        st.markdown(
            f"<div style='text-align:center;padding:6px 0;font-weight:600'>"
            f"SNR {ss.snr} dB</div>",
            unsafe_allow_html=True)
    with s3:
        if st.button("SNR +", use_container_width=True,
                     disabled=_snr_idx >= len(_snr_opts) - 1):
            new_snr = _snr_opts[_snr_idx + 1]
            ss.snr = new_snr
            ss._delta_locked = False
            if ss.is_modified:
                _refresh(ss)
            else:
                _snapshot_before(ss)
                ss.true_rate = rate_test[new_snr // 5, ss.test_idx, :ss.ue_num].numpy()
                ps, pi, pr = _predict(ss.loc_norm, ss.ap_num, ss.ue_num, ss.A, ss.P, new_snr)
                ss.pred_signal = ps; ss.pred_interf = pi; ss.pred_rate = pr
            st.rerun()

# ─── Power sliders (full width, shown when a UE is selected) ─────────────────
if ss.selected_ue is not None:
    k       = ss.selected_ue
    serving = [l for l in range(ss.ap_num) if ss.A[l, k] > 0.5]
    if serving:
        st.markdown(f"**Power for UE {k}** (normalised, sums to 1 per AP):")
        cols    = st.columns(len(serving))
        changed = False
        _step = 0.05
        for col, l in zip(cols, serving):
            cur_p = round(float(ss.P[l, k]) / _step) * _step
            key   = f"pw_{l}_{k}_{ss.test_idx}_{cur_p:.2f}"
            w_new = col.slider(
                f"AP {l}",
                min_value=0.00, max_value=1.0,
                value=round(cur_p, 2),
                step=_step,
                key=key,
            )
            if abs(w_new - cur_p) > 1e-4:
                other_sum = sum(
                    ss.raw_w[l, k2] for k2 in range(ss.ue_num)
                    if k2 != k and ss.A[l, k2] > 0.5
                )
                if other_sum < 1e-9 or w_new >= 1.0 - 1e-6:
                    ss.raw_w[l, k] = 1.0
                else:
                    ss.raw_w[l, k] = max(w_new * other_sum / (1.0 - w_new), 0.01)
                changed = True
        if changed:
            ss.P           = _renorm(ss.A, ss.raw_w)
            ss.is_modified = True
            ss._delta_locked = False
            _refresh(ss)
            st.rerun()

# ─── AP metrics (full width, shown when an AP is selected) ───────────────────
if getattr(ss, "selected_ap", None) is not None and ss.selected_ue is None:
    l = ss.selected_ap
    served = [k for k in range(ss.ue_num) if ss.A[l, k] > 0.5]
    if served:
        powers = [float(ss.P[l, k]) for k in served]
        total_p = sum(powers)
        st.markdown(
            f"**AP {l} serves {len(served)} UE(s)** — power allocation "
            f"(normalised, sums to {total_p:.3f} ≤ 1):"
        )
        cols = st.columns(len(served))
        for col, k, p in zip(cols, served, powers):
            col.metric(label=f"UE {k}", value=f"{p:.3f}")
    else:
        st.markdown(f"**AP {l}** is idle — no UEs are currently associated.")

