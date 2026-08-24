"""joint_optimize.py — Joint AP selection + power allocation via surrogate gradient.

Uses Sigmoid continuous relaxation for the discrete association matrix A,
combined with differentiable per-AP softmax power allocation. Temperature
annealing gradually sharpens A_soft toward binary values.

Usage:
    from joint_optimize import joint_optimize
    A_opt, P_opt, rates, steps = joint_optimize(
        rate_model, loc_norm, ap_num, ue_num, A_np, snr, device, ...)
"""

import os

import numpy as np
import torch
import torch.nn as nn


N_NEAREST = 4   # max APs per UE (from rl_env.py)
MAX_UE_PER_AP = 8  # max UEs per AP (from main.py: max_k=8)

# Phase-0 (power-informed init) early-stop knobs. Phase 0 only seeds the
# logits for Phase 1, so it needs far less precision than the loop that
# follows it -- EXCEPT under max_min, where which basin Phase 1 lands in is
# decided by the init: relaxing to 1e-2 there costs -16% true min-rate, while
# max_sum/fairness change within noise (ablation/ste_speed_ab.py, 40 paired
# scenarios). Module-level so runtime experiments can sweep them.
P0_MAX_ITERS = 100
P0_TOL = 1e-2         # relative single-step loss change treated as converged
P0_TOL_MAXMIN = 1e-3  # max_min keeps the tight tolerance (init-sensitive)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _masked_softmax_rows(w_logits, mask):
    """Per-row masked softmax over the last dim (fully vectorised).

    Replaces `for l in range(ap_num): softmax over valid UEs` Python loops.
    Rows with no True entry in `mask` are filled with zeros (matching the
    `if valid_l.sum() == 0: continue` branch of the legacy code).

    w_logits : (L, K) float tensor
    mask     : (L, K) bool tensor
    returns  : (L, K) float tensor, each row sums to 1 over True positions
               (or all zeros if the row is fully masked).
    """
    neg_inf = torch.finfo(w_logits.dtype).min
    w_masked = w_logits.masked_fill(~mask, neg_inf)
    sm = torch.softmax(w_masked, dim=1)
    return torch.where(mask.any(dim=1, keepdim=True),
                       sm, torch.zeros_like(sm))


def _compute_valid_mask(loc, ap_num, ue_num, n_nearest=N_NEAREST):
    """Bool mask (ap_num, ue_num): True if AP l is among n_nearest for UE k."""
    x_ap = loc[:ap_num]
    x_ue = loc[ap_num:ap_num + ue_num]
    dist = torch.cdist(x_ue, x_ap)  # (K, L)
    _, top_idx = dist.topk(n_nearest, dim=1, largest=False)  # (K, n_nearest)
    mask = torch.zeros(ap_num, ue_num, dtype=torch.bool, device=loc.device)
    for k in range(ue_num):
        mask[top_idx[k], k] = True
    return mask


def _build_geom_cache(loc, ap_num, ue_num, valid_mask):
    """Precompute static geometry tensors used by the soft-edge builders.

    Everything here depends only on node locations and valid_mask (both
    static across Phase 0/1/3), so it can be computed once per scenario
    and reused every optimisation step — the hot loop then only touches
    the per-edge power column.

    Layout of the returned dict is intentionally flat so the cache-aware
    edge builders can gather with a single fancy-index op.
    """
    device = loc.device
    x_ap = loc[:ap_num]
    x_ue = loc[ap_num:ap_num + ue_num]

    delta = x_ap[:, None, :] - x_ue[None, :, :]        # (L, K, 2)
    dist2 = (delta * delta).sum(-1)                    # (L, K)
    dist  = dist2.sqrt()
    unit  = delta / (dist[..., None] + 1e-8)           # (L, K, 2)

    # Signal / interf-edge1: one edge per valid (AP, UE) pair.
    # `nonzero(as_tuple=True)` on a 2D bool tensor returns row-major indices
    # (i.e. sorted by ap*ue_num+ue), which matches create_edges' explicit
    # argsort key — so the resulting edge ordering is identical.
    ap_s, ue_s = valid_mask.nonzero(as_tuple=True)     # (E,)
    sig_edge_index = torch.stack([ap_s, ue_s], dim=0)  # (2, E)  [AP→UE]
    i1_edge_index  = torch.stack([ue_s, ap_s], dim=0)  # (2, E)  [UE→AP]

    d2_s   = dist2[ap_s, ue_s]
    un_s_x = unit[ap_s, ue_s, 0]
    un_s_y = unit[ap_s, ue_s, 1]

    # Interf edge2: top-L nearest APs for each UE (L=8 by convention).
    top_l = min(8, ap_num)
    _, top_ap_idx = torch.topk(dist, k=top_l, dim=0, largest=False)  # (top_l,K)
    ap_i = top_ap_idx.t().reshape(-1)                                # (K*top_l,)
    ue_i = torch.arange(ue_num, device=device).unsqueeze(1)\
                .expand(-1, top_l).reshape(-1)
    i2_edge_index = torch.stack([ap_i, ue_i], dim=0)
    d2_i   = dist2[ap_i, ue_i]
    un_i_x = unit[ap_i, ue_i, 0]
    un_i_y = unit[ap_i, ue_i, 1]

    return {
        "ap_s": ap_s, "ue_s": ue_s,
        "ap_i": ap_i, "ue_i": ue_i,
        "sig_edge_index": sig_edge_index,
        "i1_edge_index":  i1_edge_index,
        "i2_edge_index":  i2_edge_index,
        "d2_s": d2_s, "un_s_x": un_s_x, "un_s_y": un_s_y,
        "d2_i": d2_i, "un_i_x": un_i_x, "un_i_y": un_i_y,
    }


def _build_soft_P(w_logits, a_logits, valid_mask, ap_num, ue_num, tau,
                   use_ste=True, leak=0.01):
    """Convert learnable logits to soft A and effective power matrix.

    When use_ste=True, applies Leaky Straight-Through Estimator:
    ON edges → 1.0 in forward, OFF edges → leak (not 0).
    Backward uses sigmoid gradients for all edges.

    Returns:
        A_soft : Tensor (ap_num, ue_num) in [0, 1] — for constraint penalties
        P_eff  : Tensor (ap_num, ue_num), effective power passed to GNN
    """
    A_soft = torch.sigmoid(a_logits[:ap_num, :ue_num] / tau)
    A_soft = A_soft * valid_mask.float()

    if use_ste:
        # Leaky STE: ON→1.0, OFF→leak (not 0, so gradient survives)
        A_hard_leaky = torch.where(A_soft > 0.5,
                                   torch.ones_like(A_soft),
                                   torch.full_like(A_soft, leak))
        A_used = A_hard_leaky - A_soft.detach() + A_soft
    else:
        A_used = A_soft

    # Per-AP softmax over ALL valid UEs (so OFF edges get non-zero P_raw)
    # Vectorised via _masked_softmax_rows.
    P_raw = _masked_softmax_rows(w_logits[:ap_num, :ue_num], valid_mask)

    P_eff = A_used * P_raw
    return A_soft, P_eff


def _create_soft_signal_edges(loc, ap_num, ue_num, P_eff, valid_mask,
                               cache=None):
    """Build signal edges (AP->UE) for all valid pairs.

    Convention matches signal_map.py:85 — edge_index = [ap_idx, ue_idx].
    When `cache` is provided, reuse the precomputed edge_index and
    geometry attributes; only the per-edge power column is recomputed.
    """
    if cache is not None:
        power = P_eff[cache["ap_s"], cache["ue_s"]]
        edge_attr = torch.stack(
            [power, cache["d2_s"], cache["un_s_x"], cache["un_s_y"]],
            dim=1)
        return cache["sig_edge_index"], edge_attr

    device = loc.device
    x_ap = loc[:ap_num]
    x_ue = loc[ap_num:ap_num + ue_num]

    delta = x_ap[:, None, :] - x_ue[None, :, :]  # (L, K, 2)
    dist2 = (delta ** 2).sum(-1)                    # (L, K)
    dist = dist2.sqrt()
    unit = delta / (dist[..., None] + 1e-8)        # (L, K, 2)

    ap_idx, ue_idx = valid_mask.nonzero(as_tuple=True)

    power = P_eff[ap_idx, ue_idx]
    d2 = dist2[ap_idx, ue_idx]
    un = unit[ap_idx, ue_idx]

    edge_index = torch.stack([ap_idx, ue_idx], dim=0).to(device)
    edge_attr = torch.stack([power, d2, un[:, 0], un[:, 1]], dim=1)
    return edge_index, edge_attr


def _create_soft_interf_edges(loc, ap_num, ue_num, P_eff, valid_mask,
                                cache=None):
    """Build interference edges: edge1 (UE->AP served) + edge2 (AP->UE interf).

    Convention matches interf_map.py:103,123.
    When `cache` is provided, reuse the precomputed edge_index and
    geometry attributes; only the per-edge power column is recomputed.
    """
    if cache is not None:
        # Edge1: same (ap_s, ue_s) pairs reversed, power = P_eff
        power_s = P_eff[cache["ap_s"], cache["ue_s"]]
        e1_attr = torch.stack(
            [power_s, cache["d2_s"], cache["un_s_x"], cache["un_s_y"]],
            dim=1)
        # Edge2: top-8 nearest AP per UE, power = 1 - P_eff
        power_i = 1.0 - P_eff[cache["ap_i"], cache["ue_i"]]
        e2_attr = torch.stack(
            [power_i, cache["d2_i"], cache["un_i_x"], cache["un_i_y"]],
            dim=1)
        return (cache["i1_edge_index"], e1_attr,
                cache["i2_edge_index"], e2_attr)

    device = loc.device
    x_ap = loc[:ap_num]
    x_ue = loc[ap_num:ap_num + ue_num]

    delta = x_ap[:, None, :] - x_ue[None, :, :]  # (L, K, 2)
    dist2 = (delta ** 2).sum(-1)                    # (L, K)
    dist = dist2.sqrt()
    unit = delta / (dist[..., None] + 1e-8)

    # Edge1 (UE -> AP): same valid pairs, reversed direction
    ap_idx, ue_idx = valid_mask.nonzero(as_tuple=True)
    power_s = P_eff[ap_idx, ue_idx]
    d2_s = dist2[ap_idx, ue_idx]
    un_s = unit[ap_idx, ue_idx]

    e1_index = torch.stack([ue_idx, ap_idx], dim=0).to(device)
    e1_attr = torch.stack([power_s, d2_s, un_s[:, 0], un_s[:, 1]], dim=1)

    # Edge2 (AP -> UE): top-8 nearest APs per UE, interference power
    top_l = min(8, ap_num)
    _, top_ap_idx = torch.topk(dist, k=top_l, dim=0, largest=False)  # (top_l, K)

    ap_idx_i = top_ap_idx.t().reshape(-1)  # (K * top_l,)
    ue_idx_i = torch.arange(ue_num, device=device).unsqueeze(1).repeat(
        1, top_l).reshape(-1)

    # Interference power = 1 - effective_power (no hard filter)
    power_i = 1.0 - P_eff[ap_idx_i, ue_idx_i]
    d2_i = dist2[ap_idx_i, ue_idx_i]
    un_i = unit[ap_idx_i, ue_idx_i]

    e2_index = torch.stack([ap_idx_i, ue_idx_i], dim=0).to(device)
    e2_attr = torch.stack([power_i, d2_i, un_i[:, 0], un_i[:, 1]], dim=1)

    return e1_index, e1_attr, e2_index, e2_attr


def _soft_rate_forward(rate_model, loc, ap_num, ue_num, A_soft, P_eff,
                       valid_mask, cache=None):
    """Forward pass through RateModel with pre-built soft edges.

    Builds Leaky STE edges and passes them to rate_model.forward(edges=...),
    bypassing the hard-threshold create_edges.

    When `cache` (from `_build_geom_cache`) is provided, edge_index and the
    geometry attribute columns are reused — only the power column is
    recomputed each step.

    Returns: pred_rate Tensor (ue_num,) — predicted rate per UE (not normalized).
    """
    sig_ei, sig_ea = _create_soft_signal_edges(
        loc, ap_num, ue_num, P_eff, valid_mask, cache=cache)
    i1_ei, i1_ea, i2_ei, i2_ea = _create_soft_interf_edges(
        loc, ap_num, ue_num, P_eff, valid_mask, cache=cache)

    edges = {
        'signal':  (sig_ei, sig_ea),
        'interf1': (i1_ei, i1_ea),
        'interf2': (i2_ei, i2_ea),
    }

    pred_rate_norm = rate_model(loc, ap_num, ue_num, None, None, edges=edges)
    pred_rate = pred_rate_norm * rate_model.rate_std + rate_model.rate_mean
    return pred_rate


def _ap_constraints(A_soft, valid_mask, ue_num,
                    max_conn_per_ue=N_NEAREST, min_conn_per_ue=1,
                    max_conn_per_ap=MAX_UE_PER_AP, sparsity_w=0.1,
                    topo_reg=0.0):
    """Soft penalty for AP selection constraints."""
    conn_per_ue = A_soft[:, :ue_num].sum(dim=0)  # (K,)
    conn_per_ap = A_soft[:, :ue_num].sum(dim=1)  # (L,)

    # Max N_NEAREST APs per UE
    pen_max_ue = torch.relu(conn_per_ue - max_conn_per_ue).sum()
    # Min AP per UE
    pen_min_ue = torch.relu(min_conn_per_ue - conn_per_ue).sum()
    # Max MAX_UE_PER_AP UEs per AP
    pen_max_ap = torch.relu(conn_per_ap - max_conn_per_ap).sum()

    # Binary entropy: encourage A_soft toward 0 or 1
    a = A_soft[valid_mask].clamp(1e-6, 1 - 1e-6)
    pen_binary = -(a * a.log() + (1 - a) * (1 - a).log()).sum()

    penalty = 30.0 * (pen_max_ue + pen_min_ue) + sparsity_w * pen_binary

    # Topology density penalty: penalise deviating from target density
    # GNN trained at ~80% connection density (p=0.8 in generate.py)
    if topo_reg > 0:
        target_density = 0.8
        n_valid = valid_mask.sum().float()
        target_edges = target_density * n_valid
        actual_edges = A_soft[:, :ue_num][valid_mask[:, :ue_num]].sum()
        penalty = penalty + topo_reg * (actual_edges - target_edges) ** 2

    return penalty


# Smooth-min temperature. apg_optimizer had the same hardcoded 0.1 and there it
# was crippling (gradient collapsed onto one UE). Measured here it is inert: a
# constant tau of 0.1 / 0.3 / 0.5 / 1.0 gives 0.8972 / 0.8930 / 0.8953 / 0.8926
# min-rate over 20 scenarios -- a 0.005 spread. Spread-adaptive variants were
# non-monotone (0.25*spread -> 0.9553, 0.5*spread -> 0.7805), i.e. noise plus
# non-stationarity, so 0.1 stays. Do not "fix" this by analogy with APG.
TAU_MM = 0.1


def _smooth_min_loss(rate_masked):
    """-(smooth min of rate_masked), i.e. a loss to MINIMISE."""
    return TAU_MM * torch.logsumexp(
        -rate_masked.clamp(min=1e-9) / TAU_MM, dim=0)


# Optional monotone recalibration of the surrogate rate inside the fairness
# loss. The surrogate compresses the rate distribution (over-predicts weak
# UEs, under-predicts strong ones); log-utility curvature turns that into a
# systematic mis-weighting. A monotone map commutes with min, so max_min is
# untouched by design. Knots come from ablation/fit_rate_calibration.py,
# fitted on the training split's validation slice; enabled by pointing
# CF_RATE_CALIB at the knots file. Cost per step: one bucketize + gather.
_CALIB_CACHE = {}


def _load_calib():
    path = os.environ.get('CF_RATE_CALIB', '')
    if not path:
        return None
    if path not in _CALIB_CACHE:
        kx, ky = np.load(path)
        _CALIB_CACHE[path] = (torch.tensor(kx, dtype=torch.float32),
                              torch.tensor(ky, dtype=torch.float32))
    return _CALIB_CACHE[path]


def _apply_calib(rate_masked):
    """Piecewise-linear g(rate), linear extrapolation past the end knots.

    Zeros (unserved-UE slots) stay zero so the serving incentive of
    log(clamp(r)) is unchanged."""
    calib = _load_calib()
    if calib is None:
        return rate_masked
    kx, ky = calib
    idx = torch.bucketize(rate_masked.detach(), kx).clamp(1, len(kx) - 1)
    x0, x1 = kx[idx - 1], kx[idx]
    y0, y1 = ky[idx - 1], ky[idx]
    w = (rate_masked - x0) / (x1 - x0 + 1e-12)
    g = y0 + w * (y1 - y0)
    return torch.where(rate_masked > 0, g, rate_masked)


def _compute_loss(rate_masked, objective, ue_num,
                  target_mask=None, protect_mask=None,
                  target_floor_t=None, floor_t=None,
                  _decrease_target=False,
                  secondary_type=None, min_rate_floor=None,
                  ue_abs_floor=None, baseline_t=None,
                  primary_mask=None, bare_mask=None,
                  secondary_weight_scale=1.0):
    """Compute objective loss + constraint penalties.

    ``secondary_weight_scale`` (default 1.0 → unchanged) multiplies every
    *below-primary* penalty term — the secondary global objective, the
    min-rate floor, per-UE absolute floors, and the supplementary protection.
    The closed loop raises it when the primary objective is already met but a
    secondary / floor requirement is not, so the optimiser weights those
    harder without touching the primary objective's own loss.

    Shares its mathematical form with ``power_only.power_optimize``.

    ``primary_mask`` marks UEs in the UI's "主要目标" group. They get a
    ``_PRIMARY_WEIGHT`` multiplier so the optimiser pulls harder on them
    than on "次要目标" UEs (which share the same ``target_mask`` but are
    only protection-style constraints).

    ``bare_mask`` marks UEs the user asked to "boost" without giving a
    specific multiplier. For these UEs the loss is an unbounded
    ``-rate / baseline`` (pure maximise); the stretched / above /
    target_gap logic is skipped. Upper limits come from the other
    constraints (constraint_floor / min_rate_floor / ue_abs_floor)
    rather than from a hard-coded internal multiplier.
    """
    _PRIMARY_WEIGHT = 3.0  # how much stronger 主要目标 is vs 次要目标

    # Build a per-UE weight vector for a given boolean sub-mask (the
    # mask that we've already used to slice target_mask or similar).
    def _weights_for(sub_mask_on_target):
        if primary_mask is None or target_mask is None:
            return None
        _pri_sub = primary_mask[target_mask][sub_mask_on_target]
        return torch.where(
            _pri_sub,
            torch.tensor(_PRIMARY_WEIGHT, device=rate_masked.device),
            torch.tensor(1.0, device=rate_masked.device))

    # Primary objective
    if objective == "targeted" and target_mask is not None:
        if _decrease_target:
            loss = torch.sum(rate_masked[target_mask])
        elif target_floor_t is not None and baseline_t is not None:
            # Split target_mask into "concrete" (has an explicit multiplier)
            # and "bare" (user said "提升" without a number).
            if bare_mask is not None:
                _bm_on_tgt = bare_mask[target_mask]
                _cm_on_tgt = ~_bm_on_tgt
            else:
                _bm_on_tgt = None
                _cm_on_tgt = torch.ones_like(rate_masked[target_mask], dtype=torch.bool)

            _scale = baseline_t[target_mask].clamp(min=1e-3)
            _loss_terms = []

            # Concrete-target part: aim for 2× requested increase with
            # stretched = 2*target_floor - baseline.
            if _cm_on_tgt.any():
                stretched = 2.0 * target_floor_t - baseline_t
                _rate_c   = rate_masked[target_mask][_cm_on_tgt]
                _str_c    = stretched[target_mask][_cm_on_tgt]
                _scale_c  = _scale[_cm_on_tgt]
                # Below stretched: push up (relu penalty)
                below = torch.relu(_str_c - _rate_c) / _scale_c
                # Above stretched: gently push down
                above = torch.relu(_rate_c - _str_c) * 0.5 / _scale_c
                _wv_c = _weights_for(_cm_on_tgt)
                if _wv_c is not None:
                    below = below * _wv_c
                    above = above * _wv_c
                _loss_terms.append(torch.sum(below) + torch.sum(above))

            # Bare-target part: unbounded maximise -rate/baseline.
            # Upper limit will come from protect_mask / min_rate_floor
            # / ue_abs_floor constraints elsewhere.
            if _bm_on_tgt is not None and _bm_on_tgt.any():
                _rate_b  = rate_masked[target_mask][_bm_on_tgt]
                _scale_b = _scale[_bm_on_tgt]
                bare_loss = -_rate_b / _scale_b
                _wv_b = _weights_for(_bm_on_tgt)
                if _wv_b is not None:
                    bare_loss = bare_loss * _wv_b
                _loss_terms.append(torch.sum(bare_loss))

            loss = sum(_loss_terms) if _loss_terms \
                   else torch.tensor(0.0, device=rate_masked.device)
        elif target_floor_t is not None:
            loss = torch.sum(torch.relu(target_floor_t[target_mask] - rate_masked[target_mask]))
        else:
            loss = -torch.sum(rate_masked[target_mask])
    elif objective == "fairness":
        loss = -torch.sum(torch.log2(_apply_calib(rate_masked).clamp(min=1e-9)))
    elif objective == "max_min":
        loss = _smooth_min_loss(rate_masked)
    else:  # max_sum_rate
        loss = -torch.sum(rate_masked)

    # Secondary global objective  (scaled by secondary_weight_scale)
    if secondary_type == "fairness":
        loss = loss - 5.0 * secondary_weight_scale * torch.sum(
            torch.log2(rate_masked.clamp(min=1e-9)))
    elif secondary_type == "max_min":
        loss = loss + 5.0 * secondary_weight_scale * _smooth_min_loss(rate_masked)

    # Absolute minimum rate floor  (scaled by secondary_weight_scale)
    if min_rate_floor is not None and min_rate_floor > 0:
        gap = torch.relu(min_rate_floor - rate_masked.clamp(min=0))
        loss = loss + 100.0 * secondary_weight_scale * torch.sum(gap)

    # Per-UE absolute floor  (scaled by secondary_weight_scale)
    if ue_abs_floor:
        for k, v in ue_abs_floor.items():
            if k < ue_num:
                loss = loss + 20.0 * secondary_weight_scale * torch.relu(v - rate_masked[k])

    # Target/protection constraints
    penalty_secondary = 20.0
    penalty_supplementary = 5.0

    if target_mask is not None and target_floor_t is not None:
        # Concrete-target UEs only — bare targets are pure-maximise in the
        # primary branch above and have no explicit floor to penalise.
        if bare_mask is not None:
            _cm_on_tgt = ~bare_mask[target_mask]
        else:
            _cm_on_tgt = torch.ones_like(rate_masked[target_mask], dtype=torch.bool)
        if _cm_on_tgt.any():
            _rate_c = rate_masked[target_mask][_cm_on_tgt]
            _flr_c  = target_floor_t[target_mask][_cm_on_tgt]
            if _decrease_target:
                target_gap = torch.relu(_rate_c - _flr_c)
            else:
                target_gap = torch.relu(_flr_c - _rate_c)
            if baseline_t is not None:
                _scale_c = baseline_t[target_mask][_cm_on_tgt].clamp(min=1e-3)
                target_gap = target_gap / _scale_c
            _wv_c = _weights_for(_cm_on_tgt)
            if _wv_c is not None:
                target_gap = target_gap * _wv_c
            loss = loss + penalty_secondary * torch.sum(target_gap)

    if protect_mask is not None and floor_t is not None and protect_mask.any():
        shortfall = torch.relu(
            floor_t[protect_mask] - rate_masked[protect_mask])
        # Normalise by baseline so the supplementary penalty lives in the
        # same relative-gap space as primary / secondary, making the
        # hierarchy (primary 60 > secondary 20 > supplementary 5)
        # independent of each UE's absolute baseline.
        if baseline_t is not None:
            shortfall = shortfall / baseline_t[protect_mask].clamp(min=1e-3)
        loss = loss + penalty_supplementary * secondary_weight_scale * torch.sum(shortfall)

    return loss


# ── Discretization ───────────────────────────────────────────────────────────

def _discretize_A(a_logits, ap_num, ue_num, tau, valid_mask, A_orig=None,
                  min_conn_per_ue=1, threshold=0.5):
    """Round A_soft to binary and fix constraint violations.

    Enforces: each UE has >= min_conn_per_ue APs, each AP serves <= MAX_UE_PER_AP UEs.
    Uses iterative greedy pruning: repeatedly remove the weakest connection
    from the most overloaded AP until all constraints are satisfied.
    When removing, prefer newly-added connections over original ones to
    preserve baseline connectivity.
    Returns: A_hard ndarray (ap_num, ue_num), float32.
    """
    with torch.no_grad():
        A_soft = torch.sigmoid(a_logits[:ap_num, :ue_num] / tau)
        A_soft = A_soft * valid_mask.float()
        A_hard = (A_soft > threshold).float()

        # Track which connections are original vs newly added
        if A_orig is not None:
            is_original = torch.tensor(A_orig[:ap_num, :ue_num],
                                       dtype=torch.float32, device=A_soft.device) > 0.5
        else:
            is_original = torch.zeros_like(A_hard, dtype=torch.bool)

        # Per-UE fix: ensure each UE has >= min_conn_per_ue APs
        for k in range(ue_num):
            while A_hard[:, k].sum() < min_conn_per_ue:
                # Add the best unconnected valid AP
                candidates = A_soft[:, k] * valid_mask[:, k].float() * (1 - A_hard[:, k])
                if candidates.max() <= 0:
                    break
                A_hard[candidates.argmax(), k] = 1.0

        # Per-UE cap: at most N_NEAREST APs per UE. This was implicit while
        # valid_mask was the distance top-N_NEAREST set (only N_NEAREST
        # candidates existed); once the mask also carries the incoming topology
        # a UE can be offered more, and nothing else here bounds it. Keep the
        # N_NEAREST strongest by A_soft. Same omission that let APG emit
        # infeasible 5-6 AP assignments.
        for k in range(ue_num):
            served = (A_hard[:, k] > 0.5).nonzero(as_tuple=True)[0]
            if served.numel() > N_NEAREST:
                keep = served[A_soft[served, k].topk(N_NEAREST).indices]
                A_hard[:, k] = 0.0
                A_hard[keep, k] = 1.0

        # Per-AP fix: iteratively remove weakest from most overloaded AP
        # Prefer removing newly-added connections over original ones
        for _iter in range(200):
            ap_load = A_hard.sum(dim=1)  # (L,)
            worst_ap = ap_load.argmax().item()
            if ap_load[worst_ap] <= MAX_UE_PER_AP:
                break

            served = (A_hard[worst_ap, :] > 0.5).nonzero(as_tuple=True)[0]
            removable = [k.item() for k in served
                         if A_hard[:, k.item()].sum() > 1]
            if not removable:
                break

            # Prefer removing new connections; among same type, remove weakest
            new_removable = [k for k in removable if not is_original[worst_ap, k]]
            pool = new_removable if new_removable else removable

            soft_vals = A_soft[worst_ap, pool]
            weakest = pool[soft_vals.argmin().item()]
            A_hard[worst_ap, weakest] = 0.0

    return A_hard.cpu().numpy().astype(np.float32)


# ── Main entry point ────────────────────────────────────────────────────────

def joint_optimize(rate_model, loc_norm, ap_num, ue_num, A_np, snr,
                   device, n_joint_iters=20, n_refine_iters=20,
                   lr_a=0.1, lr_p=0.05, objective="max_sum_rate",
                   tau_start=1.0, tau_end=0.05,
                   target_ues=None, constraint_floor=0.9,
                   protected_floor=1.0, secondary_type=None,
                   target_multiplier=None, protected_ues=None,
                   baseline_rates=None, min_rate_floor=None,
                   ue_multipliers=None, ue_abs_floor=None,
                   bare_targets=None, init_P=None,
                   secondary_weight_scale=1.0,
                   n_starts=1, disc_thresholds=(0.5,)):
    """Jointly optimize AP selection (A) and power allocation (P).

    Phase 1: Joint gradient-based optimization with sigmoid relaxation.
             With n_starts > 1 the SAME n_joint_iters budget is split into
             n_starts independent anneals (compressed tau schedule), so the
             step count and wall-clock stay those of a single run.
    Phase 2: Discretize A_soft to binary A_hard. disc_thresholds with more
             than one entry rounds at each threshold and keeps the candidate
             that scores best on the surrogate objective.
    Phase 3: Power-only refinement with fixed A_hard.

    All (start, threshold) candidates are scored at the SAME final state --
    hard A, tau_end power, one no-grad forward each -- so the selection is
    comparable, unlike best-iterate tracking across the anneal (see the NOTE
    in Phase 1). n_starts=1 with a single threshold skips scoring entirely
    and is the legacy path, bit-identical to the previous behaviour.

    Returns:
        A_best  : ndarray (ap_num, ue_num) binary
        P_best  : ndarray (ap_num, ue_num) float, rows sum to 1
        rates   : ndarray (ue_num,) predicted rates
        steps   : int, total optimization steps taken
    """
    loc = loc_norm.to(device) if isinstance(loc_norm, torch.Tensor) \
        else torch.tensor(loc_norm, dtype=torch.float32, device=device)

    # Candidate edges: the N_NEAREST closest APs per UE, UNION the incoming
    # topology. The distance-based set alone excludes ~3.9 edges per scenario
    # that the reference topology actually uses (beta ordering is not distance
    # ordering once shadowing is in play), so without the union this optimiser
    # cannot even express the A it was handed, let alone improve on it. The
    # <= N_NEAREST-per-UE cap is enforced separately in _ap_constraints and
    # _discretize_A, so widening the candidate set does not widen the feasible
    # set -- it only stops us from discarding good edges before we start.
    valid_mask = _compute_valid_mask(loc, ap_num, ue_num)
    if A_np is not None:
        _a_in = torch.as_tensor(np.asarray(A_np) > 0.5, device=valid_mask.device)
        valid_mask = valid_mask | _a_in[:ap_num, :ue_num]

    # Precompute static geometry (distances, unit vectors, top-8 neighbours,
    # edge indices) once — shared across Phase 0 / Phase 1 / Phase 3.
    geom_cache = _build_geom_cache(loc, ap_num, ue_num, valid_mask)

    # ── Prepare target/protection constraints ─────────────────────────────
    # (moved before Phase 0 so it can use the same objective)
    if target_ues is None:
        target_ues = []
    if protected_ues is None:
        protected_ues = []
    if ue_multipliers is None:
        ue_multipliers = {}

    # Preserve the *original* target_ues (from the parser) before we
    # expand target_ues to cover every UE in ue_multipliers below. Only
    # the original ones are "primary" in the UI sense; the rest are
    # "secondary" (protection-style) UEs. Loss weighting uses this split.
    _primary_ues_orig = list(target_ues)

    if ue_multipliers:
        target_ues = sorted(k for k in ue_multipliers.keys() if k < ue_num)
    elif not target_ues and target_multiplier is not None:
        target_ues = list(range(ue_num))
        _primary_ues_orig = list(target_ues)  # all UEs are primary

    has_targets = len(target_ues) > 0
    target_mask = None
    protect_mask = None
    target_floor_t = None
    floor_t = None
    _decrease_target = False

    if has_targets and baseline_rates is not None:
        bl = baseline_rates[:ue_num]

        floor_vals = np.full(ue_num, constraint_floor, dtype=np.float32)
        floor_t = torch.tensor(bl * floor_vals, dtype=torch.float32,
                               device=device)

        target_floor_t = None
        if ue_multipliers:
            tf = np.zeros(ue_num, dtype=np.float32)
            for k, m in ue_multipliers.items():
                if k < ue_num:
                    tf[k] = bl[k] * m
            target_floor_t = torch.tensor(tf, dtype=torch.float32,
                                          device=device)
        elif target_multiplier is not None:
            tf = np.zeros(ue_num, dtype=np.float32)
            for k in target_ues:
                if k < ue_num:
                    tf[k] = bl[k] * target_multiplier
            target_floor_t = torch.tensor(tf, dtype=torch.float32,
                                          device=device)

        target_mask = torch.zeros(ue_num, dtype=torch.bool, device=device)
        for k in target_ues:
            if k < ue_num:
                target_mask[k] = True
        protect_mask = ~target_mask

        _decrease_target = (target_multiplier is not None
                            and target_multiplier < 1.0
                            and not ue_multipliers)
    else:
        has_targets = False

    # baseline_t for stretched target computation
    _baseline_t = None
    if has_targets and baseline_rates is not None:
        _baseline_t = torch.tensor(baseline_rates[:ue_num], dtype=torch.float32,
                                   device=device)

    # primary_mask: subset of target_mask that corresponds to the UI's
    # "primary" group (= user's original target_ues). These UEs get a
    # larger gradient weight in _compute_loss so "主要目标" actually
    # carries more force than "次要目标".
    _primary_mask_t = None
    if has_targets and _primary_ues_orig:
        _primary_mask_t = torch.zeros(ue_num, dtype=torch.bool, device=device)
        for k in _primary_ues_orig:
            if 0 <= k < ue_num:
                _primary_mask_t[k] = True

    # bare_mask: UEs where the user said "提升" without a specific %.
    # _compute_loss treats them as unbounded maximise (loss = -rate/baseline)
    # instead of pushing to a fixed stretched target.
    _bare_mask_t = None
    if has_targets and bare_targets:
        _bare_mask_t = torch.zeros(ue_num, dtype=torch.bool, device=device)
        for k in bare_targets:
            if 0 <= k < ue_num:
                _bare_mask_t[k] = True

    _loss_kwargs = dict(
        target_mask=target_mask if has_targets else None,
        protect_mask=protect_mask if has_targets else None,
        target_floor_t=target_floor_t,
        floor_t=floor_t,
        _decrease_target=_decrease_target,
        secondary_type=secondary_type,
        min_rate_floor=min_rate_floor,
        ue_abs_floor=ue_abs_floor,
        baseline_t=_baseline_t,
        primary_mask=_primary_mask_t,
        bare_mask=_bare_mask_t,
        secondary_weight_scale=secondary_weight_scale,
    )

    # ── Phase 0: Power-informed initialization ─────────────────────────
    # Run quick power optimization on full-connect topology (all valid edges).
    # Uses the SAME objective as Phase 1. Uses the soft-edge fast path
    # (edges= bypass via _soft_rate_forward + geom_cache) instead of the
    # original create_edges slow path — ~6× faster per iteration.
    A_full_t = valid_mask.float()

    w0 = torch.zeros(ap_num, ue_num, device=device, requires_grad=True)
    opt0 = torch.optim.Adam([w0], lr=0.05)
    _prev_loss0 = float("inf")
    _served0 = (A_full_t.sum(dim=0) > 0.5).float()
    for _i in range(P0_MAX_ITERS):
        opt0.zero_grad()
        # Vectorised per-AP masked softmax → soft power, full-connect A.
        P_raw0 = _masked_softmax_rows(w0, valid_mask)
        P_eff0 = A_full_t * P_raw0

        # _soft_rate_forward returns already-denormalised rate (bits/s/Hz);
        # same convention as Phase 1.
        pred0 = _soft_rate_forward(
            rate_model, loc, ap_num, ue_num,
            A_full_t, P_eff0, valid_mask, cache=geom_cache)
        rate0_masked = pred0[:ue_num] * _served0
        loss0 = _compute_loss(rate0_masked, objective, ue_num, **_loss_kwargs)
        loss0.backward()
        opt0.step()

        _cur0 = loss0.item()
        _p0_tol = P0_TOL_MAXMIN if objective == "max_min" else P0_TOL
        if _i > 0 and (_prev_loss0 - _cur0) / (abs(_cur0) + 1e-8) < _p0_tol:
            break
        _prev_loss0 = _cur0

    # w_logits_init: directly from Phase 0
    w_logits_init = (w0 * valid_mask.float()).detach()

    # a_logits_init: rank-based mapping (robust to any power distribution)
    # Top 80% power → positive (ON), bottom 20% → negative (OFF)
    with torch.no_grad():
        P0 = _masked_softmax_rows(w0, valid_mask)

        valid_vals = P0[valid_mask]
        # z-score normalization: robust to any power distribution shape
        mean_p = valid_vals.mean()
        std_p = valid_vals.std()
        a_logits_init = ((P0 - mean_p) / (std_p + 1e-6)).clamp(-2.0, 2.0)
        a_logits_init = a_logits_init * valid_mask.float()

    # ── Annealing ratios ────────────────────────────────────────────────
    tau_ratio = (tau_end / tau_start) if tau_start > 0 else 1.0
    leak_start, leak_end = 0.1, 0.01  # probe strength: strong→weak

    total_steps = 0

    from power_only import power_optimize

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: True joint optimization — single optimizer, single forward
    #          per step, updating both a_logits and w_logits simultaneously
    # ══════════════════════════════════════════════════════════════════════
    # NOTE: no best-iterate tracking WITHIN an anneal, deliberately. Raising
    # n_joint_iters makes the result worse (20/50/100/200 -> 0.882/0.866/0.848/
    # 0.852 true min-rate), which looks like the last iterate being a lottery --
    # but adding best-by-surrogate-objective tracking made it worse still
    # (-> 0.853/0.858/0.858/0.826). The objective is not comparable across
    # iterates because tau anneals 1.0 -> 0.05 underneath it: early high-tau
    # points score best on a fractional A and then discretise badly. The
    # (n_starts, disc_thresholds) selection below avoids exactly that trap by
    # scoring every candidate at the same final state: hard A, tau_end power.

    def _hard_score(A_hard_np_, P_soft_t):
        """Surrogate objective of a discretised candidate, higher = better."""
        with torch.no_grad():
            A_h = torch.as_tensor(A_hard_np_, dtype=torch.float32,
                                  device=device)
            P_h = A_h * P_soft_t[:ap_num, :ue_num]
            row = P_h.sum(dim=1, keepdim=True)
            P_h = torch.where(row > 1e-12, P_h / row, P_h)
            r = _soft_rate_forward(rate_model, loc, ap_num, ue_num,
                                   A_h, P_h, valid_mask, cache=geom_cache)
            r = r[:ue_num] * (A_h.sum(dim=0) > 0.5).float()
            if objective == "max_min":
                return float(r.min())
            if objective == "fairness":
                return float(torch.log2(r.clamp(min=1e-9)).sum())
            return float(r.sum())

    _min_conn_disc = 1
    n_starts = max(int(n_starts), 1)
    iters_per_start = max(n_joint_iters // n_starts, 1)
    _legacy = (n_starts == 1 and len(disc_thresholds) == 1)
    best = None  # (score, A_hard_np, P_ws)

    for _start in range(n_starts):
        if _start == 0:
            _a0, _w0 = a_logits_init, w_logits_init
        else:
            # deterministic perturbation of the Phase-0 init, valid edges only
            _g = torch.Generator().manual_seed(_start)
            _noise = lambda ref: (0.5 * torch.randn(ref.shape, generator=_g)
                                  .to(device) * valid_mask.float())
            _a0 = a_logits_init + _noise(a_logits_init)
            _w0 = w_logits_init + _noise(w_logits_init)

        _a = _a0.clone().detach().requires_grad_(True)
        _w = _w0.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([
            {"params": [_a], "lr": lr_a},
            {"params": [_w], "lr": lr_p},
        ])

        for i in range(iters_per_start):
            optimizer.zero_grad()

            _progress = i / max(iters_per_start - 1, 1)
            tau = tau_start * (tau_ratio ** _progress)
            leak = leak_start * (leak_end / leak_start) ** _progress

            A_soft, P_eff = _build_soft_P(
                _w, _a, valid_mask, ap_num, ue_num, tau, leak=leak)

            pred_rate = _soft_rate_forward(
                rate_model, loc, ap_num, ue_num, A_soft, P_eff, valid_mask,
                cache=geom_cache)
            conn_mask = (A_soft.sum(dim=0) > 0.01).float()
            rate_masked = pred_rate[:ue_num] * conn_mask

            loss = _compute_loss(rate_masked, objective, ue_num, **_loss_kwargs)
            # For max-min: only density regularisation, no per-UE floor
            _min_conn = 1
            _topo_reg = 3.0 if objective == "max_min" else 0.0
            loss = loss + _ap_constraints(
                A_soft, valid_mask, ue_num,
                min_conn_per_ue=_min_conn,
                topo_reg=_topo_reg)

            loss.backward()
            optimizer.step()
            total_steps += 1

        # ══════════════════════════════════════════════════════════════════
        # Phase 2: Discretize A (one candidate per threshold, best kept)
        # ══════════════════════════════════════════════════════════════════
        with torch.no_grad():
            _, P_ws = _build_soft_P(
                _w, _a, valid_mask, ap_num, ue_num, tau_end)

        for _th in disc_thresholds:
            A_hard_np = _discretize_A(_a, ap_num, ue_num, tau_end, valid_mask,
                                      A_orig=A_np,
                                      min_conn_per_ue=_min_conn_disc,
                                      threshold=_th)
            if _legacy:
                best = (0.0, A_hard_np, P_ws)
                break
            _s = _hard_score(A_hard_np, P_ws)
            if best is None or _s > best[0]:
                best = (_s, A_hard_np, P_ws)

    A_hard_np, P_ws = best[1], best[2]

    # ══════════════════════════════════════════════════════════════════════
    # Phase 3: Power-only refinement with fixed A_hard
    # ══════════════════════════════════════════════════════════════════════
    P_ref, rates_ref, ref_steps = power_optimize(
        rate_model, loc_norm, ap_num, ue_num, A_hard_np, snr,
        device, n_iters=n_refine_iters, lr=lr_p,
        objective=objective, init_P=P_ws.cpu().numpy(),
        # Pass original target_ues so power_optimize can rebuild primary_mask
        target_ues=_primary_ues_orig, constraint_floor=constraint_floor,
        protected_floor=protected_floor, secondary_type=secondary_type,
        target_multiplier=target_multiplier, protected_ues=protected_ues,
        baseline_rates=baseline_rates, min_rate_floor=min_rate_floor,
        ue_multipliers=ue_multipliers, ue_abs_floor=ue_abs_floor,
        bare_targets=bare_targets,
        geom_cache=geom_cache, valid_mask_in=valid_mask,
        secondary_weight_scale=secondary_weight_scale,
    )
    total_steps += ref_steps

    return A_hard_np, P_ref, rates_ref, total_steps
