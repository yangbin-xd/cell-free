"""power_only.py — Power allocation via GNN surrogate gradient descent.

Power-only optimiser for cell-free massive MIMO: keeps the AP-UE association
matrix ``A`` fixed and optimises the per-AP power allocation through a
differentiable masked softmax on top of learnable logits. Gradients flow
through a pretrained GNN rate surrogate (``RateModel``), and the objective
function supports sum-rate, proportional fairness, max-min, and targeted
per-UE intents together with secondary objectives and auxiliary floor
constraints.

See ``power_only_algorithm.tex`` for the full mathematical description.

Usage:

    from power_only import power_optimize
    P_opt, rates, steps = power_optimize(
        rate_model, loc_norm, ap_num, ue_num, A_np, snr, device,
        objective="max_sum_rate", n_iters=100, lr=0.05, ...)

The function serves two roles in the code base:
  1. A standalone "Power-Only" method (see ``compare.py``).
  2. Phase 3 of the Leaky STE joint optimiser (``joint_optimize.py``),
     which passes in a pre-built ``geom_cache`` + ``valid_mask_in`` to
     skip the per-call geometry rebuild.
"""

import numpy as np
import torch

# Early-stop knobs for the optimisation loop. Module-level so runtime
# experiments (ablation/ste_speed_ab.py) can sweep them without editing code.
# PLATEAU_TOL 1e-4 -> 1e-3 halves the wall-clock of both Power-Only and the
# Leaky STE Phase 3 with paired metric deltas inside noise on all three
# objectives (ablation/ste_speed_confirm.py, result/ste_speed_confirm_40.npz).
PATIENCE = 20      # iters without a new best before giving up (safety net)
PLATEAU_TOL = 1e-3  # relative single-step loss change treated as converged


def power_optimize(rate_model, loc_norm, ap_num, ue_num, A_np, snr,
                   device, n_iters=100, lr=0.05, objective="max_sum_rate",
                   target_ues=None, constraint_floor=0.9,
                   protected_floor=1.0, secondary_type=None,
                   target_multiplier=None, protected_ues=None,
                   baseline_rates=None, min_rate_floor=None,
                   ue_multipliers=None, ue_abs_floor=None,
                   bare_targets=None,
                   init_P=None, geom_cache=None, valid_mask_in=None,
                   secondary_weight_scale=1.0):
    """Power-only optimisation via surrogate gradient through RateModel.

    Keeps topology A fixed and optimises power weights w_logits through
    differentiable softmax + RateModel forward pass.

    If ``geom_cache`` and ``valid_mask_in`` are supplied (e.g. from
    ``joint_optimize`` Phase 3), they are reused — skipping redundant
    ``_compute_valid_mask`` + edge-index reconstruction on every step.

    Parameters
    ----------
    rate_model : RateModel
        Pretrained GNN rate surrogate. Must expose ``forward(loc, ap_num,
        ue_num, A, P, edges=...)`` and the ``rate_mean``/``rate_std``
        attributes consumed by ``_soft_rate_forward`` for denormalisation.
    loc_norm : Tensor | ndarray (L+K, 2)
        Normalised AP + UE 2-D coordinates.
    ap_num, ue_num : int
    A_np : ndarray (ap_num, ue_num)
        Fixed binary association matrix.
    snr : int
        SNR in dB (used only for consistency with the broader pipeline;
        the rate model already encodes its own SNR-specific head).
    device : torch.device
    n_iters : int, optional
        Maximum number of Adam iterations (default 100).
    lr : float, optional
        Adam learning rate (default 0.05).
    objective : {"max_sum_rate", "fairness", "max_min", "targeted"}
        Primary intent. See ``_compute_loss`` in ``joint_optimize.py``
        for the mathematical definitions.
    target_ues, protected_ues : list[int] | None
        Named UE groups for targeted/protection semantics.
    constraint_floor : float
        Allowed rate ratio for non-target UEs (0.9 = max 10% drop).
    target_multiplier : float | None
        Global boost/cut multiplier applied to every ``target_ues`` entry.
    ue_multipliers : dict[int, float] | None
        Per-UE multipliers (overrides ``target_multiplier`` when supplied).
    secondary_type : {"fairness", "max_min"} | None
        Optional secondary global objective.
    baseline_rates : ndarray (ue_num,) | None
        Baseline rates used to convert multipliers into absolute floors.
    min_rate_floor : float | None
        Hard floor on every UE's rate.
    ue_abs_floor : dict[int, float] | None
        Absolute per-UE floors (not ratio-based).
    init_P : ndarray | None
        Warm-start power matrix used as initial logits. If ``None``,
        zeros are used (uniform distribution).
    geom_cache : dict | None
        Optional precomputed geometry cache (see ``_build_geom_cache``).
    valid_mask_in : Tensor | None
        Optional precomputed valid-connection mask (see
        ``_compute_valid_mask``).

    Returns
    -------
    P_out : ndarray (ap_num, ue_num)
        Optimised power (rows sum to 1 on served UEs).
    rates_out : ndarray (ue_num,)
        Predicted rates at the best iterate (zeros if no step ran).
    steps : int
        Number of iterations actually executed.
    """
    from joint_optimize import (_compute_valid_mask, _compute_loss,
                                _soft_rate_forward, _masked_softmax_rows,
                                _build_geom_cache)

    loc = loc_norm.to(device) if isinstance(loc_norm, torch.Tensor) \
        else torch.tensor(loc_norm, dtype=torch.float32, device=device)

    if valid_mask_in is not None:
        valid_mask = valid_mask_in
    else:
        valid_mask = _compute_valid_mask(loc, ap_num, ue_num)
    if geom_cache is None:
        geom_cache = _build_geom_cache(loc, ap_num, ue_num, valid_mask)

    # Fixed association
    A_t = torch.tensor(A_np[:ap_num, :ue_num].astype(np.float32),
                       device=device)

    # Initialise w_logits
    if init_P is not None:
        w_logits = torch.tensor(init_P[:ap_num, :ue_num].astype(np.float32),
                                device=device).requires_grad_(True)
    else:
        w_logits = torch.zeros(ap_num, ue_num, device=device,
                               requires_grad=True)

    # ── Build constraint tensors ──────────────────────────────────────
    if target_ues is None:
        target_ues = []
    if protected_ues is None:
        protected_ues = []
    if ue_multipliers is None:
        ue_multipliers = {}

    # Save the *original* target_ues (what the parser labelled as
    # "主要目标") BEFORE we expand it to cover every UE in ue_multipliers.
    _primary_ues_orig = list(target_ues)

    if ue_multipliers:
        target_ues = sorted(k for k in ue_multipliers.keys() if k < ue_num)
    elif not target_ues and target_multiplier is not None:
        target_ues = list(range(ue_num))
        _primary_ues_orig = list(target_ues)

    has_targets = len(target_ues) > 0
    target_mask = protect_mask = target_floor_t = floor_t = None
    _decrease_target = False
    _baseline_t = None
    _primary_mask_t = None

    if has_targets and baseline_rates is not None:
        bl = baseline_rates[:ue_num]
        floor_vals = np.full(ue_num, constraint_floor, dtype=np.float32)
        floor_t = torch.tensor(bl * floor_vals, dtype=torch.float32,
                               device=device)
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
        _baseline_t = torch.tensor(bl, dtype=torch.float32, device=device)
        if _primary_ues_orig:
            _primary_mask_t = torch.zeros(ue_num, dtype=torch.bool, device=device)
            for k in _primary_ues_orig:
                if 0 <= k < ue_num:
                    _primary_mask_t[k] = True
    else:
        has_targets = False

    # bare_mask: UEs where user said "提升" with no specific %; treated
    # as unbounded maximise inside _compute_loss.
    _bare_mask_t = None
    if has_targets and bare_targets:
        _bare_mask_t = torch.zeros(ue_num, dtype=torch.bool, device=device)
        for k in bare_targets:
            if 0 <= k < ue_num:
                _bare_mask_t[k] = True

    loss_kw = dict(
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

    # ── Optimisation loop ─────────────────────────────────────────────
    optimizer = torch.optim.Adam([w_logits], lr=lr)
    best_loss = float('inf')
    best_w = w_logits.data.clone()
    best_rates = None
    _prev_loss = float('inf')
    _no_improve = 0
    # Patience-based safety net — defaults to never triggering unless
    # best_loss stalls for many consecutive iters. Kept conservative
    # because the strict plateau check below is the primary early stop.
    _PATIENCE = PATIENCE
    _PLATEAU_TOL = PLATEAU_TOL

    # Combined mask (A ∧ valid_mask) is fixed across iterations.
    pow_mask = (A_t > 0.5) & valid_mask
    conn_mask = (A_t.sum(dim=0) > 0.5).float()[:ue_num]  # static across iters

    i = 0
    for i in range(n_iters):
        optimizer.zero_grad()

        # Build P via per-AP masked softmax (fully vectorised)
        P_eff = _masked_softmax_rows(w_logits, pow_mask)

        pred_rate = _soft_rate_forward(
            rate_model, loc, ap_num, ue_num, A_t, P_eff, valid_mask,
            cache=geom_cache)
        rate_masked = pred_rate[:ue_num] * conn_mask

        loss = _compute_loss(rate_masked, objective, ue_num, **loss_kw)
        loss.backward()
        optimizer.step()

        cur = loss.item()
        if cur < best_loss:
            best_loss = cur
            best_w = w_logits.data.clone()
            best_rates = rate_masked.detach()
            _no_improve = 0
        else:
            _no_improve += 1

        # Plateau detection: exit when single-step relative loss change
        # falls below _PLATEAU_TOL. Primary early-stop criterion.
        if i > 5 and abs(_prev_loss - cur) / (abs(cur) + 1e-8) < _PLATEAU_TOL:
            break
        # Patience-based early stop: no best improvement for _PATIENCE
        # consecutive iters. Safe because we always return best_w.
        if _no_improve >= _PATIENCE and i > 5:
            break
        _prev_loss = cur

    # ── Recover best P ────────────────────────────────────────────────
    with torch.no_grad():
        P_out_t = _masked_softmax_rows(best_w, pow_mask)
    P_out = P_out_t.cpu().numpy().astype(np.float32)

    rates_out = best_rates.cpu().numpy() if best_rates is not None \
        else np.zeros(ue_num, dtype=np.float32)

    return P_out, rates_out, i + 1
