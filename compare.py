"""compare.py -- Compare 2 proposed methods + 4 baselines on test scenarios.

Proposed (GNN-surrogate gradient):
  - Leaky STE         (joint_optimize.joint_optimize)
  - Power-Only        (power_only.power_optimize)

Baselines:
  - original          (no optimisation, baseline A/P)
  - WMMSE-ADMM        (β oracle + ADMM QCQP)
  - APG               (Adam + projection, β oracle)

All results are evaluated against true CSI via recompute_true().

Usage:
    python compare.py                    # defaults: 50 scenarios, SNR=15
    python compare.py --n 200 --snr 15
    python compare.py --snr_sweep        # sweep 0-30 dB
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

import numpy as np
import torch

# The per-scenario graphs are tiny (<=12 APs, <=30 UEs), so every operator in
# the surrogate is dominated by thread-synchronisation rather than arithmetic.
# Measured on one scenario: 22.6 ms/step at the 8-thread default vs 6.1 ms at
# 2 threads -- a 3.7x penalty for using more cores. A later 6-scenario sweep
# put 1 thread another 1.4x ahead of 2 (STE 0.71s -> 0.48s), so 1 is the
# default. Override with CF_THREADS.
torch.set_num_threads(int(os.environ.get("CF_THREADS", "1")))

# Force CPU loading
_torch_load_orig = torch.load
def _torch_load_cpu(f, *a, **kw):
    kw.setdefault('map_location', 'cpu')
    return _torch_load_orig(f, *a, **kw)
torch.load = _torch_load_cpu

from main import BS_loc
from generate import cf_test, UE_test, loc_mean, loc_std
from process import (AP_num_test, UE_num_test, loc_test_norm,
                     A_test, P_test, rate_test)

_CSI_data   = cf_test.CSI
_BS_2d      = BS_loc[:, :2]
_UE_test_2d = UE_test[:, :2]
_loc_std  = loc_std.numpy()  if hasattr(loc_std,  'numpy') else np.asarray(loc_std)
_loc_mean = loc_mean.numpy() if hasattr(loc_mean, 'numpy') else np.asarray(loc_mean)
_device = torch.device('cpu')

# ── Load rate models ──────────────────────────────────────────────────────
# The radio map used here must be the SAME checkpoint the accuracy tables
# report as "Proposed" (result/se_snr_table.md, main_seed0), otherwise
# the two sets of results describe different models.
#
# Only a 15 dB rate head exists for this stem, and that is sufficient: no
# parameter of RateModel depends on the receive SNR -- it enters solely through
# noise_dB = -87 - self.snr in the closed-form SINR step -- so one fine-tuned
# checkpoint is evaluated at every SNR. This is the same property
# ablation/snr_eval.py relies on.
from rate_map import RateModel
MODEL_STEM = os.environ.get("CF_MODEL_STEM", "main_seed0")
_FIT_SNR = 15
_sig_path  = f"model/signal_{MODEL_STEM}.pth"
_itf_path  = f"model/interf_{MODEL_STEM}.pth"
_rate_path = f"model/rate_{MODEL_STEM}_{_FIT_SNR}dB.pth"
for _p in (_sig_path, _itf_path, _rate_path):
    if not os.path.exists(_p):
        raise SystemExit(f"checkpoint missing: {_p}")
rate_models = {}
for _snr in range(0, 31, 5):
    _m = RateModel(signal_model_path=_sig_path, interf_model_path=_itf_path,
                   snr=_snr)
    _m.load_state_dict(torch.load(_rate_path, map_location='cpu'))
    _m.eval()
    rate_models[_snr] = _m
print(f"[compare] radio map = {MODEL_STEM} (rate head fine-tuned at "
      f"{_FIT_SNR} dB, reused across SNRs)")

# ── Import optimisers ─────────────────────────────────────────────────────
from baseline.apg_optimizer import apg_optimize

try:
    from joint_optimize import joint_optimize
    from power_only import power_optimize
    _JO_AVAILABLE = True
except (ImportError, AttributeError):
    _JO_AVAILABLE = False

try:
    from power_only import power_optimize  # noqa: F811
    _PO_AVAILABLE = True
except (ImportError, AttributeError):
    _PO_AVAILABLE = False

try:
    from baseline.wmmse_admm_optimizer import wmmse_admm_optimize
    _WMMSE_ADMM_AVAILABLE = True
except ImportError:
    _WMMSE_ADMM_AVAILABLE = False

try:
    from baseline.socp_optimizer import socp_max_min_downlink
    _SOCP_AVAILABLE = True
except ImportError:
    _SOCP_AVAILABLE = False


# ── True rate computation ─────────────────────────────────────────────────

def recompute_true(ue_csi, ap_orig, A, P, snr_val):
    """Vectorised true MRT SINR / rate (same as app.py _recompute_true)."""
    L = _CSI_data.shape[0]
    K = ue_csi.shape[1]
    A_full = np.zeros((L, K), dtype=np.float32)
    P_full = np.zeros((L, K), dtype=np.float32)
    for i, la in enumerate(ap_orig):
        A_full[la] = A[i]
        P_full[la] = P[i]
    h = ue_csi[..., 0]
    h_norms = np.linalg.norm(h, axis=-1)
    contrib = (np.sqrt(P_full) * A_full)[:, :, None] * h_norms
    sig_pow = contrib.sum(axis=0) ** 2
    w = h / (h_norms[..., None] + 1e-30)
    cross = np.einsum('lknt,lmnt->lkmn', h.conj(), w)
    PA = np.sqrt(P_full) * A_full
    ic = np.einsum('lm,lkmn->kmn', PA, cross)
    k_idx = np.arange(K)
    ic[k_idx, k_idx, :] = 0
    int_pow = (np.abs(ic) ** 2).sum(axis=1)
    noise = 10 ** ((-87 - snr_val) / 10)
    SINR = sig_pow / (int_pow + noise)
    rate = np.mean(np.log2(1 + SINR), axis=1)
    unserved = np.where(A_full.sum(axis=0) < 0.5)[0]
    rate[unserved] = 0.0
    return rate


def _renorm(A, raw_w):
    """Normalise power per AP row (same as app.py)."""
    P = np.zeros_like(raw_w, dtype=np.float32)
    for l in range(A.shape[0]):
        conn = A[l] > 0.5
        row = raw_w[l] * conn
        s = row.sum()
        P[l] = (row / s) if s > 1e-12 else conn / (conn.sum() + 1e-9)
    return P


# ── Scenario loader ───────────────────────────────────────────────────────

def load_scenario(idx):
    """Load test scenario by index, return dict with all data."""
    ap_num = int(AP_num_test[idx].item())
    ue_num = int(UE_num_test[idx].item())
    xy = loc_test_norm[idx].numpy() * _loc_std + _loc_mean
    ap_loc = xy[:ap_num]
    ue_loc = xy[ap_num:ap_num + ue_num]

    A = A_test[idx, :ap_num, :ue_num].numpy().copy().astype(np.float32)
    P = P_test[idx, :ap_num, :ue_num].numpy().copy().astype(np.float32)

    ap_orig = np.array([
        np.argmin(np.sum((_BS_2d - loc) ** 2, axis=1)) for loc in ap_loc])
    ue_orig = np.array([
        np.argmin(np.sum((_UE_test_2d - loc) ** 2, axis=1)) for loc in ue_loc])

    ue_csi = _CSI_data[:, ue_orig, :]
    loc_norm = loc_test_norm[idx]

    return {
        "idx": idx,
        "ap_num": ap_num,
        "ue_num": ue_num,
        "A": A,
        "P": P,
        "ap_orig": ap_orig,
        "ue_orig": ue_orig,
        "ue_csi": ue_csi,
        "loc_norm": loc_norm,
    }


# ── Run one optimiser ─────────────────────────────────────────────────────

def run_apg(sc, snr, objective="max_sum_rate"):
    result = apg_optimize(
        sc["ue_csi"], sc["ap_orig"], sc["A"], sc["P"],
        snr, sc["ap_num"], sc["ue_num"],
        objective=objective, max_iters=200, refine_iters=50)
    true_rate = recompute_true(
        sc["ue_csi"], sc["ap_orig"], result["A_best"],
        _renorm(result["A_best"], result["w_best"]), snr)
    return true_rate, result["elapsed"]


def run_joint_optimize(sc, snr, objective="max_sum_rate"):
    if not _JO_AVAILABLE or snr not in rate_models:
        return None, 0
    A_opt, P_opt, _, steps = joint_optimize(
        rate_models[snr], sc["loc_norm"],
        sc["ap_num"], sc["ue_num"], sc["A"], snr, _device,
        n_joint_iters=20, n_refine_iters=100,
        lr_a=0.1, lr_p=0.05, objective=objective)
    P_norm = _renorm(A_opt, P_opt)
    true_rate = recompute_true(
        sc["ue_csi"], sc["ap_orig"], A_opt, P_norm, snr)
    return true_rate, 0


def run_power_only(sc, snr, objective="max_sum_rate"):
    """Power-only surrogate gradient (fixed topology A)."""
    if not _PO_AVAILABLE or snr not in rate_models:
        return None, 0
    P_opt, _, steps = power_optimize(
        rate_models[snr], sc["loc_norm"],
        sc["ap_num"], sc["ue_num"], sc["A"], snr, _device,
        n_iters=100, lr=0.05, objective=objective)
    P_norm = _renorm(sc["A"], P_opt)
    true_rate = recompute_true(
        sc["ue_csi"], sc["ap_orig"], sc["A"], P_norm, snr)
    return true_rate, 0


def run_wmmse_beta(sc, snr, objective="max_sum_rate"):
    """WMMSE-ADMM using only beta (large-scale fading), no instantaneous CSI."""
    if not _WMMSE_ADMM_AVAILABLE:
        return None, 0
    from baseline.wmmse_admm_optimizer import _wmmse_admm_iterate, _compute_channel_stats_beta
    import time as _t

    t0 = _t.time()
    ap_num, ue_num = sc["ap_num"], sc["ue_num"]
    L_full = sc["ue_csi"].shape[0]
    K = ue_num

    # Extract beta
    h = sc["ue_csi"][..., 0]
    h_norms = np.linalg.norm(h, axis=-1)
    beta = np.mean(h_norms ** 2, axis=2)  # (L_full, K)

    # Map local A/P -> global
    A_global = np.zeros((L_full, K), dtype=np.float32)
    P_global = np.zeros((L_full, K), dtype=np.float32)
    for i, la in enumerate(sc["ap_orig"]):
        A_global[la] = sc["A"][i]
        P_global[la] = sc["P"][i]

    sigma2 = 10 ** ((-87 - snr) / 10.0)
    P_max = 1.0

    # Beta-only channel statistics
    a, B = _compute_channel_stats_beta(beta[:, :K])

    A_mask = (A_global.T > 0.5)
    mu_init = np.sqrt(P_global.T * P_max + 1e-12) * A_mask

    rho, obj, iters = _wmmse_admm_iterate(
        a, B, K, L_full, A_mask, sigma2, P_max,
        mu_init, objective, 50, 1e-4)

    P_opt = np.zeros((ap_num, ue_num), dtype=np.float32)
    for i, la in enumerate(sc["ap_orig"]):
        row = rho[:, la]
        s = row.sum()
        P_opt[i] = (row / s) if s > 1e-12 else sc["P"][i]

    P_norm = _renorm(sc["A"], P_opt)
    true_rate = recompute_true(
        sc["ue_csi"], sc["ap_orig"], sc["A"], P_norm, snr)
    return true_rate, _t.time() - t0


def run_socp(sc, snr, objective="max_min"):
    """β-oracle max-min downlink power control (bisection + SOCP).

    Only defined for max_min. For other objectives, returns None.
    """
    if not _SOCP_AVAILABLE or objective != "max_min":
        return None, 0
    import time as _t

    t0 = _t.time()
    ap_num, ue_num = sc["ap_num"], sc["ue_num"]
    L_full = sc["ue_csi"].shape[0]
    K = ue_num

    h = sc["ue_csi"][..., 0]
    h_norms = np.linalg.norm(h, axis=-1)
    beta = np.mean(h_norms ** 2, axis=2)

    A_global = np.zeros((L_full, K), dtype=np.float32)
    for i, la in enumerate(sc["ap_orig"]):
        A_global[la] = sc["A"][i]

    rho_d = 10 ** ((87 + snr) / 10.0)

    P_global, _t_star = socp_max_min_downlink(
        beta[:, :K], A_global, rho_d)

    P_local = np.zeros_like(sc["P"])
    for i, la in enumerate(sc["ap_orig"]):
        P_local[i] = P_global[la]

    P_norm = _renorm(sc["A"], P_local)
    true_rate = recompute_true(
        sc["ue_csi"], sc["ap_orig"], sc["A"], P_norm, snr)
    return true_rate, _t.time() - t0


# ── Main benchmark ────────────────────────────────────────────────────────

def _log_fairness(rate):
    """Sum of log2(rate_k) — proportional fairness metric."""
    r = np.maximum(rate, 1e-12)
    return float(np.sum(np.log2(r)))


def benchmark(n_scenarios, snr, verbose=True, objective="max_sum_rate"):
    """Run all optimisers on n_scenarios at given SNR.

    Parameters
    ----------
    objective : str -- "max_sum_rate" or "fairness"
    """
    _empty = lambda: {"sum_rate": [], "min_rate": [], "fairness": [], "time": []}
    results = {
        "original":       _empty(),
        "APG":            _empty(),
        "Leaky STE":      _empty(),
        "Power-Only":     _empty(),
        "WMMSE-ADMM":     _empty(),
        "SOCP":        _empty(),
    }

    def _record(name, rate, t):
        results[name]["sum_rate"].append(float(rate.sum()))
        results[name]["min_rate"].append(float(rate.min()))
        results[name]["fairness"].append(_log_fairness(rate))
        results[name]["time"].append(t)

    for idx in range(n_scenarios):
        sc = load_scenario(idx)

        # Original (no optimisation)
        rate_orig = recompute_true(sc["ue_csi"], sc["ap_orig"],
                                   sc["A"], sc["P"], snr)
        _record("original", rate_orig, 0.0)

        # APG (beta-based closed-form)
        t0 = time.time()
        rate_apg, _ = run_apg(sc, snr, objective)
        _record("APG", rate_apg, time.time() - t0)

        # Joint Optimize (leaky STE) — joint A+P
        t0 = time.time()
        rate_jo, _ = run_joint_optimize(sc, snr, objective)
        if rate_jo is not None:
            _record("Leaky STE", rate_jo, time.time() - t0)

        # Power-Only (surrogate gradient, fixed A)
        t0 = time.time()
        rate_po, _ = run_power_only(sc, snr, objective)
        if rate_po is not None:
            _record("Power-Only", rate_po, time.time() - t0)

        # Under max_min the β-oracle slot is filled by SOCP, not WMMSE-ADMM.
        # WMMSE has no closed-form ω for the non-smooth min operator (see
        # §9.8), so running it here would mean optimising a fairness proxy
        # while the column is labelled max-min -- the comparison would not be
        # against a max-min algorithm at all. SOCP (bisection + second-order
        # cone program) solves the max-min problem itself and is the
        # convex-optimal reference for this objective.
        if objective == "max_min":
            t0 = time.time()
            rate_socp, _ = run_socp(sc, snr, objective)
            if rate_socp is not None:
                _record("SOCP", rate_socp, time.time() - t0)
        else:
            _wm_obj = "fairness" if objective == "fairness" else "max_sum_rate"
            t0 = time.time()
            rate_wb, _ = run_wmmse_beta(sc, snr, _wm_obj)
            if rate_wb is not None:
                _record("WMMSE-ADMM", rate_wb, time.time() - t0)
            rate_socp = None

        if verbose and (idx < 5 or idx % 10 == 0):
            # Progress line reports the optimised metric, not always the sum.
            if objective == "max_min":
                _v = lambda r: r[r > 0].min() if r is not None and (r > 0).any() else None
                _f = "{:.2f}"
            elif objective == "fairness":
                _v = lambda r: _log_fairness(r) if r is not None else None
                _f = "{:.1f}"
            else:
                _v = lambda r: r.sum() if r is not None else None
                _f = "{:.1f}"
            _s = lambda r: _f.format(_v(r)) if _v(r) is not None else "--"
            # The β-oracle slot holds SOCP under max_min and WMMSE-ADMM otherwise.
            _oracle = (f"SOCP={_s(rate_socp)}" if objective == "max_min"
                       else f"WMb={_s(rate_wb)}")
            print(f"  [{idx:3d}] orig={_s(rate_orig)}  APG={_s(rate_apg)}  "
                  f"STE={_s(rate_jo)}  "
                  f"PO={_s(rate_po)}  {_oracle}")

    return results


# Each objective is scored on the quantity it actually optimises. Reporting a
# method's min-rate while it was maximising sum-rate compares it against a
# target it was never given, and the resulting "loss" is an artefact of the
# objective rather than of the method. All four metrics are still stored by
# save_results(); only the report is restricted.
PRIMARY_METRIC = {
    "max_sum_rate": ("sum_rate", "sum rate",     "{:7.2f}", "bit/s/Hz"),
    "max_min":      ("min_rate", "min rate",     "{:7.3f}", "bit/s/Hz"),
    "fairness":     ("fairness", "PF utility",   "{:7.2f}", ""),
}

_METHODS = ["original", "APG", "Leaky STE",
            "Power-Only", "WMMSE-ADMM", "SOCP"]


def print_summary(results, snr, n, objective="max_sum_rate"):
    """Print summary statistics for the optimised objective only."""
    key, label, num_fmt, unit = PRIMARY_METRIC.get(
        objective, ("sum_rate", "sum rate", "{:7.2f}", "bit/s/Hz"))

    print(f"\n{'='*70}")
    print(f"  {n} Scenarios, SNR = {snr} dB, objective = {objective}")
    print(f"{'='*70}")
    header = f"{label} ({unit})" if unit else label
    print(f"  {'Method':<12s}  {header:>20s}  {'time':>8s}")
    print("  " + "-" * 46)
    for name in _METHODS:
        data = results.get(name)
        if data is None or not data.get(key):
            continue
        val = np.array(data[key]).mean()
        tm = np.array(data["time"]).mean()
        print(f"  {name:<12s}  {num_fmt.format(val):>20s}  {tm:7.2f}s")

    # Win rate vs original, on the optimised metric
    orig = np.array(results["original"][key])
    print(f"\n  Win rate vs Original ({label}):")
    for name in _METHODS[1:]:
        if not results[name].get(key):
            continue
        v = np.array(results[name][key])
        wins = (v > orig).sum()
        print(f"    {name:<12s}: {wins}/{len(v)}"
              f"  (avg gain: {(v.mean()/orig.mean() - 1)*100:+.1f}%)")

    # Head-to-head against the strongest baseline, on the same metric
    if results["Leaky STE"].get(key) and results["APG"].get(key):
        ste = np.array(results["Leaky STE"][key])
        apg = np.array(results["APG"][key])
        n_cmp = min(len(ste), len(apg))
        wins = (ste[:n_cmp] > apg[:n_cmp]).sum()
        print(f"\n  Leaky STE vs APG ({label}): {wins}/{n_cmp} wins"
              f"  (avg gain: {(ste[:n_cmp].mean()/apg[:n_cmp].mean() - 1)*100:+.1f}%)")


def save_results(results, snr, n, objective="max_sum_rate", outdir="result"):
    """Save per-scenario arrays to .npz, one file per method per objective.

    Each .npz contains 4 arrays: sum_rate, min_rate, fairness, time
    (all length-N per-scenario lists). Layout:
        result/{method_tag}/{obj_short}.npz
    """
    _obj_short = {
        "max_sum_rate": "max_sum",
        "fairness":     "fairness",
        "max_min":      "max_min",
    }.get(objective, objective)

    for name, data in results.items():
        if not data["sum_rate"]:
            continue
        tag = name.lower().replace(" ", "_").replace("-", "_")
        method_dir = os.path.join(outdir, tag)
        os.makedirs(method_dir, exist_ok=True)
        np.savez(os.path.join(method_dir, f"{_obj_short}.npz"),
                 sum_rate=np.array(data["sum_rate"]),
                 min_rate=np.array(data["min_rate"]),
                 fairness=np.array(data["fairness"]),
                 time=np.array(data["time"]))
    print(f"\n  Results saved to {outdir}/<method>/{_obj_short}.npz "
          f"(per-scenario sum_rate/min_rate/fairness/time)")


# ── SNR curve (all objectives × all SNRs × all methods) ──────────────────

def snr_curve(n_scenarios=1, snr_values=None,
              objectives=("max_sum_rate", "fairness", "max_min"),
              outdir="result", verbose=True):
    """Run all methods at every SNR value for all objectives.

    Output: ``result/snrcurve_{method}_{obj_short}.npz`` with keys
        snr_values, sum_rate, min_rate, fairness, time
    where each metric is shaped (n_snr, n_scenarios) — preserving the
    per-scenario data so plot_snr.py can show mean ± std.
    """
    if snr_values is None:
        snr_values = list(range(0, 31, 5))
    snr_list = list(snr_values)

    _obj_short_map = {
        "max_sum_rate": "max_sum",
        "fairness":     "fairness",
        "max_min":      "max_min",
    }
    os.makedirs(outdir, exist_ok=True)

    for objective in objectives:
        if verbose:
            print(f"\n{'='*60}\n  Objective: {objective}\n{'='*60}")
        per_method = {}   # method_name -> list of (snr, results dict)

        for snr in snr_list:
            if verbose:
                print(f"\n--- SNR = {snr} dB ---")
            res = benchmark(n_scenarios, snr, objective=objective,
                            verbose=False)
            for name, data in res.items():
                if not data["sum_rate"]:
                    continue
                if name not in per_method:
                    per_method[name] = {
                        "snr_values": [],
                        "sum_rate":   [],
                        "min_rate":   [],
                        "fairness":   [],
                        "time":       [],
                    }
                per_method[name]["snr_values"].append(snr)
                per_method[name]["sum_rate"].append(data["sum_rate"])
                per_method[name]["min_rate"].append(data["min_rate"])
                per_method[name]["fairness"].append(data["fairness"])
                per_method[name]["time"].append(data["time"])
            if verbose:
                line = "  " + "  ".join(
                    f"{n[:8]:<8s}={np.mean(d['sum_rate']):5.1f}/{np.mean(d['min_rate']):.2f}"
                    for n, d in res.items() if d['sum_rate'])
                print(line)

        # Save one file per method for this objective
        obj_short = _obj_short_map.get(objective, objective)
        for name, d in per_method.items():
            tag = name.lower().replace(" ", "_").replace("-", "_")
            method_dir = os.path.join(outdir, tag)
            os.makedirs(method_dir, exist_ok=True)
            np.savez(
                os.path.join(method_dir, f"snrcurve_{obj_short}.npz"),
                snr_values=np.array(d["snr_values"]),
                sum_rate=np.array(d["sum_rate"]),
                min_rate=np.array(d["min_rate"]),
                fairness=np.array(d["fairness"]),
                time=np.array(d["time"]),
            )
        if verbose:
            print(f"  Saved {outdir}/<method>/snrcurve_{obj_short}.npz")


# ── SNR curve: single-SNR variant (for parallel deployment) ──────────────

def snr_curve_one(snr_value, n_scenarios=1,
                  objectives=("max_sum_rate", "fairness", "max_min"),
                  outdir="result", verbose=True):
    """Run snr_curve at exactly one SNR, save to per-SNR temp files.

    Output: ``result/<method>/snrcurve_<obj>_snr{SNR}.npz`` with same key
    layout as the canonical file but with shape (1, n_scenarios) on the
    metric arrays. Use ``merge_snr_curves`` after all parallel jobs to
    consolidate into ``snrcurve_<obj>.npz``.
    """
    _obj_short_map = {
        "max_sum_rate": "max_sum",
        "fairness":     "fairness",
        "max_min":      "max_min",
    }
    os.makedirs(outdir, exist_ok=True)

    for objective in objectives:
        if verbose:
            print(f"\n--- [{objective}] SNR={snr_value} dB ---")
        res = benchmark(n_scenarios, snr_value, objective=objective,
                        verbose=False)
        obj_short = _obj_short_map.get(objective, objective)

        for name, data in res.items():
            if not data["sum_rate"]:
                continue
            tag = name.lower().replace(" ", "_").replace("-", "_")
            method_dir = os.path.join(outdir, tag)
            os.makedirs(method_dir, exist_ok=True)
            np.savez(
                os.path.join(method_dir,
                             f"snrcurve_{obj_short}_snr{snr_value}.npz"),
                snr_values=np.array([snr_value]),
                sum_rate=np.array([data["sum_rate"]]),
                min_rate=np.array([data["min_rate"]]),
                fairness=np.array([data["fairness"]]),
                time=np.array([data["time"]]),
            )
        if verbose:
            print(f"  Saved snrcurve_{obj_short}_snr{snr_value}.npz")


def extract_single_snr(snr_value, outdir="result", verbose=True):
    """Extract one SNR slice from snrcurve_<obj>.npz and save as <obj>.npz.

    This eliminates the need for a separate Phase 1 single-SNR benchmark
    when Phase 2 (snr_curve) already covers that SNR. After running
    snr_curve at all SNRs in [0..30] and merging, call this with
    snr_value=15 to materialise the single-SNR files that the
    cdf/ue_count/scatter/time plot scripts read.
    """
    import glob
    obj_shorts = ["max_sum", "fairness", "max_min"]

    n_extracted = 0
    for method_tag in sorted(os.listdir(outdir)):
        full_dir = os.path.join(outdir, method_tag)
        if not os.path.isdir(full_dir):
            continue

        for obj_short in obj_shorts:
            src = os.path.join(full_dir, f"snrcurve_{obj_short}.npz")
            if not os.path.exists(src):
                continue
            d = np.load(src)
            snrs = d["snr_values"]
            if snr_value not in snrs:
                continue
            i = int(np.where(snrs == snr_value)[0][0])

            dst = os.path.join(full_dir, f"{obj_short}.npz")
            np.savez(
                dst,
                sum_rate=d["sum_rate"][i],
                min_rate=d["min_rate"][i],
                fairness=d["fairness"][i],
                time=d["time"][i],
            )
            n_extracted += 1
            if verbose:
                print(f"  extracted SNR={snr_value} → {dst}")

    if verbose:
        print(f"  total: {n_extracted} files extracted at SNR={snr_value}")


def merge_snr_curves(outdir="result", clean=False, verbose=True):
    """Incrementally merge per-SNR temp files into snrcurve_<obj>.npz.

    Behaviour:
      1. If a canonical ``snrcurve_<obj>.npz`` already exists, its
         per-SNR rows are preserved.
      2. Any ``snrcurve_<obj>_snr<X>.npz`` temp files are layered on
         top — for SNRs that already existed in the canonical, the
         temp file's row replaces the canonical row (newer data wins).
      3. The result is sorted by SNR and written back as the canonical.
      4. If ``clean=True``, the temp files are deleted after merging.

    Constraint: all rows (existing canonical + new temp files) must
    have the same n_scenarios. A mismatch indicates the user is mixing
    runs at different N — the function refuses to merge in that case
    and tells the user to delete the canonical first.
    """
    import glob

    obj_shorts = ["max_sum", "fairness", "max_min"]

    for method_tag in sorted(os.listdir(outdir)):
        full_dir = os.path.join(outdir, method_tag)
        if not os.path.isdir(full_dir):
            continue

        for obj_short in obj_shorts:
            canon_path = os.path.join(full_dir, f"snrcurve_{obj_short}.npz")
            pattern = os.path.join(full_dir,
                                   f"snrcurve_{obj_short}_snr*.npz")
            temp_files = sorted(glob.glob(pattern))

            # Stage 1: load existing canonical into a dict keyed by SNR
            snr_data = {}
            if os.path.exists(canon_path):
                d = np.load(canon_path)
                for i, snr in enumerate(d["snr_values"]):
                    snr_data[int(snr)] = (
                        d["sum_rate"][i], d["min_rate"][i],
                        d["fairness"][i], d["time"][i],
                    )

            # Stage 2: overlay temp files (newer data wins for same SNR)
            for f in temp_files:
                d = np.load(f)
                snr = int(d["snr_values"][0])
                snr_data[snr] = (
                    d["sum_rate"][0], d["min_rate"][0],
                    d["fairness"][0], d["time"][0],
                )

            if not snr_data:
                continue

            # Stage 3: N consistency check
            n_per_snr = {s: len(snr_data[s][0]) for s in snr_data}
            unique_n = set(n_per_snr.values())
            if len(unique_n) > 1:
                print(f"  WARN: {canon_path} has inconsistent n_scenarios "
                      f"across SNRs: {n_per_snr}")
                print(f"        delete the file and re-merge with consistent N. "
                      f"skipping.")
                continue

            # Stage 4: sort by SNR and write back
            sorted_snrs = sorted(snr_data.keys())
            np.savez(
                canon_path,
                snr_values=np.array(sorted_snrs),
                sum_rate=np.array([snr_data[s][0] for s in sorted_snrs]),
                min_rate=np.array([snr_data[s][1] for s in sorted_snrs]),
                fairness=np.array([snr_data[s][2] for s in sorted_snrs]),
                time=np.array([snr_data[s][3] for s in sorted_snrs]),
            )
            if verbose:
                added = len(temp_files)
                total = len(sorted_snrs)
                print(f"  merged +{added} temp → {total} SNRs {sorted_snrs} "
                      f"→ {canon_path}")

            # Stage 5: optionally clean temp files
            if clean:
                for f in temp_files:
                    os.remove(f)


# ── SNR sweep (legacy: max_sum_rate only, used by plot_train.py) ──────────

def snr_sweep(n_scenarios, snr_list=None, verbose=True):
    """Run benchmark across multiple SNR values."""
    if snr_list is None:
        snr_list = list(range(0, 31, 5))

    sweep = {}
    for snr in snr_list:
        print(f"\n--- SNR = {snr} dB ---")
        res = benchmark(n_scenarios, snr, verbose=False)
        sweep[snr] = res
        print_summary(res, snr, n_scenarios)
        save_results(res, snr, n_scenarios)

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SNR Sweep Summary ({n_scenarios} scenarios)")
    print(f"{'='*70}")
    header = f"  {'SNR':>4s}"
    for name in ["original", "WMMSE-ADMM", "APG", "Power-Only", "Leaky STE"]:
        header += f"  {name:>12s}"
    print(header)
    print("  " + "-" * 60)
    for snr in snr_list:
        row = f"  {snr:>3d}dB"
        for name in ["original", "WMMSE-ADMM", "APG", "Power-Only", "Leaky STE"]:
            data = sweep[snr][name]["sum_rate"]
            if data:
                row += f"  {np.mean(data):>12.2f}"
            else:
                row += f"  {'N/A':>12s}"
        print(row)

    return sweep


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark optimisers")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of test scenarios")
    parser.add_argument("--snr", type=int, default=15,
                        help="SNR in dB (single-point mode)")
    parser.add_argument("--snr_sweep", action="store_true",
                        help="Sweep SNR from 0 to 30 dB (legacy, max_sum only)")
    parser.add_argument("--snr_curve", action="store_true",
                        help="Run all 3 objectives × all SNRs × all methods, "
                             "save snrcurve_*.npz to result/")
    parser.add_argument("--snr_curve_one", type=int, default=None,
                        metavar="SNR",
                        help="Run snr_curve at one SNR only (for parallel "
                             "deployment). Saves per-SNR temp files; run "
                             "--merge_snr_curves after all parallel jobs.")
    parser.add_argument("--merge_snr_curves", action="store_true",
                        help="Merge per-SNR temp files (from --snr_curve_one) "
                             "into canonical snrcurve_<obj>.npz")
    parser.add_argument("--clean_temp", action="store_true",
                        help="With --merge_snr_curves, delete per-SNR temp "
                             "files after merging")
    parser.add_argument("--extract_single_snr", type=int, default=None,
                        metavar="SNR",
                        help="Extract one SNR slice from snrcurve_<obj>.npz "
                             "and save as <obj>.npz (replaces Phase 1)")
    parser.add_argument("--objective", type=str, default="max_sum_rate",
                        choices=["max_sum_rate", "fairness", "max_min"],
                        help="Optimisation objective")
    args = parser.parse_args()

    if args.extract_single_snr is not None:
        extract_single_snr(args.extract_single_snr)
    elif args.merge_snr_curves:
        merge_snr_curves(clean=args.clean_temp)
    elif args.snr_curve_one is not None:
        snr_curve_one(args.snr_curve_one, n_scenarios=args.n)
    elif args.snr_curve:
        snr_curve(n_scenarios=args.n)
    elif args.snr_sweep:
        snr_sweep(args.n)
    else:
        results = benchmark(args.n, args.snr, objective=args.objective)
        print_summary(results, args.snr, args.n, args.objective)
        save_results(results, args.snr, args.n, args.objective)
