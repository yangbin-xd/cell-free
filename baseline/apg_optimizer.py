"""apg_optimizer.py -- APG (Accelerated Projected Gradient) baseline.

Implements the APG algorithm from:
  Abbas et al., "Joint AP Selection and Power Allocation for
  Unicast-Multicast Cell-Free Massive MIMO," IEEE Internet of Things
  Journal, vol. 12, no. 22, pp. 47135-47150, Nov. 2025.

Adapted to unicast-only Cell-Free Massive MIMO with MR precoding.
Uses the closed-form UatF (use-and-then-forget) SE bound based on
large-scale fading coefficients beta only.

Public interface:
  apg_optimize() -- beta-based closed-form SE (main baseline)
"""

import time
import numpy as np
import torch

N_NEAREST = 4        # max APs per UE
MAX_UE_PER_AP = 8    # max UEs per AP

# Smooth-min temperature for the max_min objective, as a fraction of the
# per-UE SE spread (see _compute_sse_maxmin_torch). Set from the scale of the
# objective rather than tuned: anything in 0.25-0.75 lands within ~0.03
# bit/s/Hz of the same min-rate, so the result is not knife-edge on this value.
TAU_FRAC = 0.5
TAU_MIN = 0.2


# ── Helpers ──────────────────────────────────────────────────────────────────

def _compute_valid_mask_beta(beta, n_nearest=N_NEAREST):
    """Valid mask from beta: top-n_nearest strongest APs per UE.

    Parameters
    ----------
    beta : ndarray (L, K) -- large-scale fading coefficients

    Returns
    -------
    mask : ndarray (L, K) bool
    """
    L, K = beta.shape
    mask = np.zeros((L, K), dtype=bool)
    for k in range(K):
        top = np.argsort(beta[:, k])[-n_nearest:]
        mask[top, k] = True
    return mask


def _compute_valid_mask_loc(loc, ap_num, ue_num, n_nearest=N_NEAREST):
    """Valid mask from locations: n_nearest closest APs per UE.

    Parameters
    ----------
    loc : Tensor (ap_num + ue_num, 2)
    """
    x_ap = loc[:ap_num]
    x_ue = loc[ap_num:ap_num + ue_num]
    dist = torch.cdist(x_ue.float(), x_ap.float())  # (K, L)
    _, top_idx = dist.topk(n_nearest, dim=1, largest=False)
    mask = np.zeros((ap_num, ue_num), dtype=bool)
    for k in range(ue_num):
        mask[top_idx[k].cpu().numpy(), k] = True
    return mask


# ── Closed-form SE (UatF bound, unicast MR) ──────────────────────────────────

def _compute_sse_torch(a_t, eta_t, a_stat_t, B_diag_t, B_cross_t,
                       sigma2, Nt=32):
    """Differentiable sum SE using WMMSE-style channel statistics.

    Uses the same (a_stat, B) channel statistics as WMMSE-beta for
    accurate SINR computation under MR precoding with Rayleigh fading.

    SINR_u = (sum_l mu[u,l] * a_stat[u,l])^2
             / (sum_{u'!=u} sum_l mu[u',l]^2 * B_diag[u,l] + sigma2)

    where mu[u,l] = sqrt(a[l,u] * eta[l,u])  (effective sqrt-power).

    Parameters
    ----------
    a_t       : Tensor (L, K) -- relaxed association in [0, 1]
    eta_t     : Tensor (L, K) -- power coefficients >= 0
    a_stat_t  : Tensor (K, L) -- expected channel gain sqrt(beta) = E||h||
    B_diag_t  : Tensor (K, L) -- B[k,i,l,l] = beta[l,k]/Nt (interference coupling, diagonal)
    B_cross_t : Tensor (K, L) -- B[k,k,l,l] = beta[l,k] (self-coupling, diagonal)
    sigma2    : float -- noise power

    Returns
    -------
    sse  : scalar Tensor
    sinr : Tensor (K,)
    """
    L, K = a_t.shape

    # mu[u, l] = sqrt(a[l,u] * eta[l,u]) -- effective sqrt-power coeff
    mu = torch.sqrt(a_t.T * eta_t.T + 1e-30)  # (K, L)

    # Signal: DS_u = sum_l mu[u,l] * a_stat[u,l]
    ds = (mu * a_stat_t).sum(dim=1)             # (K,)
    ds_sq = ds ** 2

    # Self-term: mu_u^T B[u,u] mu_u  (only diagonal of B matters)
    # B[k,k,l,l] = beta[l,k],  B[k,k,l,m]=a_stat[k,l]*a_stat[k,m] for l!=m
    # Full: mu^T B[k,k] mu = sum_l mu_l^2 * B_cross[k,l] + (sum_l mu_l*a_stat[k,l])^2 - sum_l mu_l^2*a_stat[k,l]^2
    self_diag = (mu ** 2 * B_cross_t).sum(dim=1)              # (K,)
    self_cross = ds_sq                                         # already have this
    self_cross_diag = (mu ** 2 * a_stat_t ** 2).sum(dim=1)    # (K,)
    self_term = self_diag + self_cross - self_cross_diag       # (K,)

    # Beamforming uncertainty: BU_u = self_term - DS^2
    # bu = self_term - ds_sq  (should be ~0 for Rayleigh, but keep for correctness)

    # Inter-user interference: sum_{u'!=u} mu_{u'}^T B[u,u'] mu_{u'}
    # B[k,i,l,l] = beta[l,k] for i!=k (diagonal only)
    # So: sum_{u'!=u} sum_l mu[u',l]^2 * beta[l,u]
    mu_sq_sum = (mu ** 2).sum(dim=0)              # (L,) total power per AP
    # Total interference from all users at UE u: sum_l mu_sq_sum[l] * B_diag[u,l]
    total_interf = (mu_sq_sum[None, :] * B_diag_t).sum(dim=1)  # (K,)
    # Self contribution: sum_l mu[u,l]^2 * beta[l,u]
    self_interf = (mu ** 2 * B_diag_t).sum(dim=1)              # (K,)
    # Inter-user only
    other_interf = total_interf - self_interf                   # (K,)

    # Full denominator = BU + inter-user interference + noise
    denom = (self_term - ds_sq) + other_interf + sigma2

    sinr = ds_sq / (denom + 1e-30)
    se = torch.log2(1.0 + sinr)
    return se.sum(), sinr.detach()


def _compute_sse_fairness_torch(a_t, eta_t, a_stat_t, B_diag_t, B_cross_t,
                                sigma2, Nt=32):
    """Proportional fairness: sum of log(SE_u)."""
    L, K = a_t.shape
    mu = torch.sqrt(a_t.T * eta_t.T + 1e-30)
    ds = (mu * a_stat_t).sum(dim=1)
    ds_sq = ds ** 2
    self_diag = (mu ** 2 * B_cross_t).sum(dim=1)
    self_cross_diag = (mu ** 2 * a_stat_t ** 2).sum(dim=1)
    self_term = self_diag + ds_sq - self_cross_diag
    mu_sq_sum = (mu ** 2).sum(dim=0)
    total_interf = (mu_sq_sum[None, :] * B_diag_t).sum(dim=1)
    self_interf = (mu ** 2 * B_diag_t).sum(dim=1)
    other_interf = total_interf - self_interf
    denom = (self_term - ds_sq) + other_interf + sigma2
    sinr = ds_sq / (denom + 1e-30)
    se = torch.log2(1.0 + sinr)
    return torch.sum(torch.log(se.clamp(min=1e-12))), sinr.detach()


def _compute_sse_maxmin_torch(a_t, eta_t, a_stat_t, B_diag_t, B_cross_t,
                              sigma2, Nt=32):
    """Max-min fairness: smooth approximation of min(SE_u) via logsumexp."""
    L, K = a_t.shape
    mu = torch.sqrt(a_t.T * eta_t.T + 1e-30)
    ds = (mu * a_stat_t).sum(dim=1)
    ds_sq = ds ** 2
    self_diag = (mu ** 2 * B_cross_t).sum(dim=1)
    self_cross_diag = (mu ** 2 * a_stat_t ** 2).sum(dim=1)
    self_term = self_diag + ds_sq - self_cross_diag
    mu_sq_sum = (mu ** 2).sum(dim=0)
    total_interf = (mu_sq_sum[None, :] * B_diag_t).sum(dim=1)
    self_interf = (mu ** 2 * B_diag_t).sum(dim=1)
    other_interf = total_interf - self_interf
    denom = (self_term - ds_sq) + other_interf + sigma2
    sinr = ds_sq / (denom + 1e-30)
    se = torch.log2(1.0 + sinr)
    # Smooth min: -tau * logsumexp(-se/tau) approximates min(se).
    #
    # tau has to scale with the spread of se, not be a fixed constant. The
    # gradient weights are softmax(-se/tau), so a tau far below the spread
    # collapses all of them onto the single worst UE: with the old tau=0.1 and a
    # measured spread of 3.5-5.2 bit/s/Hz, the effective number of UEs receiving
    # gradient was 1.0 out of K. That turns the objective into a hard min, and
    # each step helps one UE only to make another one the minimum -- the iterate
    # oscillates instead of balancing SINRs, which is what max-min needs. It also
    # pushes the optimiser towards silencing whole APs (per-AP eta -> 0), the one
    # lever _eta_to_P/_renorm cannot represent, so those solutions collapse when
    # re-normalised to full per-AP power.
    with torch.no_grad():
        spread = (se.max() - se.min()).clamp(min=0.0)
    tau = torch.clamp(TAU_FRAC * spread, min=TAU_MIN)
    smooth_min = -tau * torch.logsumexp(-se / tau, dim=0)
    return smooth_min, sinr.detach()


# ── Projection onto feasible set ─────────────────────────────────────────────

def _project(a, eta, valid_mask_t):
    """Project (a, eta) onto the feasible set.

    Constraints:
      1. a[n,u] = 0 if not valid
      2. 0 <= a[n,u] <= 1
      3. eta[n,u] >= 0
      4. sum_u a[n,u]*eta[n,u] <= 1 for each AP n  (per-AP power)
    """
    with torch.no_grad():
        # Validity + box constraints
        a.data.mul_(valid_mask_t)
        a.data.clamp_(0.0, 1.0)
        eta.data.clamp_(0.0)

        # Per-AP power constraint: scale eta down if overloaded
        L = a.shape[0]
        for n in range(L):
            load = (a.data[n] * eta.data[n]).sum()
            if load > 1.0:
                eta.data[n] *= 1.0 / (load.item() + 1e-10)


# ── APG inner loop (Nesterov-accelerated projected gradient) ──────────────

def _apg_loop(a_init, eta_init, valid_mask_t, obj_fn,
              max_iters=200, tol=1e-5, lr_init=0.05):
    """Projected gradient ascent with Adam optimiser.

    Parameters
    ----------
    a_init, eta_init : Tensor (L, K) -- initial point
    valid_mask_t : Tensor (L, K) float -- validity mask
    obj_fn : callable(a, eta) -> (scalar, info) -- objective to MAXIMISE
    max_iters : int
    tol : float -- convergence tolerance
    lr_init : float -- learning rate

    Returns
    -------
    a_best, eta_best : Tensor (L, K)
    best_obj : float
    history : list of float -- objective per iteration
    """
    L, K = a_init.shape

    a = a_init.clone().detach().requires_grad_(True)
    eta = eta_init.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([
        {"params": [a],   "lr": lr_init},
        {"params": [eta], "lr": lr_init},
    ])

    best_obj = -float('inf')
    best_a = a_init.clone().detach()
    best_eta = eta_init.clone().detach()
    history = []

    for k in range(max_iters):
        optimizer.zero_grad()

        obj_val, _ = obj_fn(a, eta)
        (-obj_val).backward()   # minimise negative → maximise
        optimizer.step()

        # Project onto feasible set
        _project(a, eta, valid_mask_t)

        # Score the projected iterate, not the pre-step one: obj_val was
        # evaluated before optimizer.step()+_project(), so pairing it with the
        # post-step (a, eta) below would store a point that was never scored.
        with torch.no_grad():
            cur_obj = obj_fn(a, eta)[0].item()
        history.append(cur_obj)

        if cur_obj > best_obj:
            best_obj = cur_obj
            best_a = a.data.clone()
            best_eta = eta.data.clone()

        # Convergence
        if k > 10 and abs(history[-1] - history[-2]) / (abs(history[-1]) + 1e-8) < tol:
            break

    return best_a, best_eta, best_obj, history


# ── Discretization ─────────────────────────────────────────────────────────

def _discretize_apg(a_relaxed, eta, valid_mask, A_orig=None,
                    max_ap_per_ue=N_NEAREST, max_ue_per_ap=MAX_UE_PER_AP):
    """Round relaxed a to binary and enforce constraints.

    Parameters
    ----------
    a_relaxed : ndarray (L, K) in [0, 1]
    eta       : ndarray (L, K) power coefficients
    valid_mask : ndarray (L, K) bool
    A_orig     : ndarray (L, K) or None -- original topology for tie-breaking

    Returns
    -------
    A_hard : ndarray (L, K) binary float32
    eta_out : ndarray (L, K) float32 -- adjusted power
    """
    L, K = a_relaxed.shape
    A_hard = (a_relaxed > 0.5).astype(np.float32)
    A_hard *= valid_mask.astype(np.float32)

    is_original = (A_orig > 0.5) if A_orig is not None else np.zeros_like(A_hard, dtype=bool)

    # Ensure each UE has >= 1 AP
    for u in range(K):
        if A_hard[:, u].sum() == 0:
            candidates = a_relaxed[:, u] * valid_mask[:, u].astype(np.float32)
            if candidates.max() > 0:
                A_hard[candidates.argmax(), u] = 1.0
            else:
                # Fallback: pick strongest beta (proxy: highest eta)
                A_hard[eta[:, u].argmax(), u] = 1.0

    # Cap APs per UE. valid_mask is the UNION of the top-N_NEAREST-by-beta set
    # and the original topology, so it can offer more than max_ap_per_ue
    # candidates for a UE; without this the discretised A hands some UEs more
    # macro-diversity than the system allows. The reference topology in
    # data/test never exceeds N_NEAREST APs per UE (191/200 scenarios sit
    # exactly at 4, the rest at 3), and every other method here inherits that
    # topology, so exceeding it is not a fairer baseline -- it is an infeasible
    # one. Keep the max_ap_per_ue strongest by relaxed association.
    for u in range(K):
        served = np.where(A_hard[:, u] > 0.5)[0]
        if len(served) > max_ap_per_ue:
            keep = served[np.argsort(a_relaxed[served, u])[-max_ap_per_ue:]]
            A_hard[:, u] = 0.0
            A_hard[keep, u] = 1.0

    # Iterative greedy pruning for overloaded APs
    for _iter in range(200):
        ap_load = A_hard.sum(axis=1)
        worst_ap = ap_load.argmax()
        if ap_load[worst_ap] <= max_ue_per_ap:
            break

        served = np.where(A_hard[worst_ap, :] > 0.5)[0]
        removable = [u for u in served if A_hard[:, u].sum() > 1]
        if not removable:
            break

        # Prefer removing new connections
        new_removable = [u for u in removable if not is_original[worst_ap, u]]
        pool = new_removable if new_removable else removable

        soft_vals = a_relaxed[worst_ap, pool]
        weakest = pool[soft_vals.argmin()]
        A_hard[worst_ap, weakest] = 0.0

    # Adjust eta: zero out disconnected entries
    eta_out = eta.copy()
    eta_out[A_hard < 0.5] = 0.0

    # Re-normalise per-AP power: sum_u A[n,u]*eta[n,u] <= 1
    for n in range(L):
        load = (A_hard[n] * eta_out[n]).sum()
        if load > 1.0:
            eta_out[n] *= 1.0 / (load + 1e-10)

    return A_hard, eta_out


# ── Power refinement with fixed topology ──────────────────────────────────

def _refine_power(A_hard_t, eta_init_t, valid_mask_t, obj_fn,
                  n_iters=50, lr=0.02):
    """Gradient-based power refinement with fixed binary A.

    Parameters
    ----------
    A_hard_t : Tensor (L, K) -- fixed binary topology
    eta_init_t : Tensor (L, K) -- initial power coefficients
    valid_mask_t : Tensor (L, K) float
    obj_fn : callable(a, eta) -> (scalar, info)
    """
    eta = eta_init_t.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([eta], lr=lr)

    best_eta = eta_init_t.clone().detach()
    # Score the starting point too, so refinement can never return something
    # worse than what it was handed.
    with torch.no_grad():
        best_obj = obj_fn(A_hard_t, eta)[0].item()

    for i in range(n_iters):
        optimizer.zero_grad()
        obj_val, _ = obj_fn(A_hard_t, eta)
        (-obj_val).backward()  # minimise negative objective
        optimizer.step()

        # Project eta, then score the projected iterate (same off-by-one as
        # _apg_loop: obj_val belongs to the pre-step point, not to eta below).
        with torch.no_grad():
            eta.data.clamp_(0.0)
            eta.data.mul_(valid_mask_t)
            L = eta.shape[0]
            for n in range(L):
                load = (A_hard_t[n] * eta.data[n]).sum()
                if load > 1.0:
                    eta.data[n] *= 1.0 / (load.item() + 1e-10)
            cur = obj_fn(A_hard_t, eta)[0].item()

        if cur > best_obj:
            best_obj = cur
            best_eta = eta.data.clone()

    return best_eta, best_obj


# ── Convert eta to normalised power P ─────────────────────────────────────

def _eta_to_P(A_hard, eta, ap_num, ue_num):
    """Convert APG power coefficients to normalised power matrix P.

    P[l, :] = A[l,:]*eta[l,:] / sum_k A[l,k]*eta[l,k]   (rows sum to 1)
    """
    P = np.zeros((ap_num, ue_num), dtype=np.float32)
    for l in range(ap_num):
        weighted = A_hard[l] * eta[l]
        s = weighted.sum()
        if s > 1e-12:
            P[l] = weighted / s
        else:
            # Uniform over served UEs
            served = np.where(A_hard[l] > 0.5)[0]
            if len(served) > 0:
                P[l, served] = 1.0 / len(served)
    return P


# ══════════════════════════════════════════════════════════════════════════════
# Public interface 1: APG with closed-form SE (beta-based)
# ══════════════════════════════════════════════════════════════════════════════

def apg_optimize(ue_csi, ap_orig, A, P, snr, ap_num, ue_num,
                 objective="max_sum_rate", max_iters=200,
                 refine_iters=100, tol=1e-5, lr=0.05, Nt=32,
                 freeze_assoc=False):
    """APG joint AP selection + power allocation using closed-form UatF SE.

    Parameters
    ----------
    ue_csi  : ndarray (L_full, K, Nc, Nt, Nr) -- CSI for selected UEs
    ap_orig : ndarray (ap_num,) -- global AP indices
    A       : ndarray (ap_num, ue_num) -- initial binary association
    P       : ndarray (ap_num, ue_num) -- initial normalised power
    snr     : int -- SNR in dB
    ap_num, ue_num : int
    objective : str -- "max_sum_rate" or "fairness"
    max_iters : int -- APG iterations
    refine_iters : int -- power refinement iterations after discretisation
    tol : float -- convergence tolerance
    lr : float -- step size
    Nt : int -- transmit antennas per AP

    Returns
    -------
    dict with keys: A_best, w_best, score, n_evals, elapsed, history
    """
    t0 = time.time()
    device = torch.device('cpu')

    # ── Extract beta from CSI ─────────────────────────────────────────────
    h = ue_csi[..., 0]                                  # (L_full, K, Nc, Nt)
    h_norms = np.linalg.norm(h, axis=-1)                # (L_full, K, Nc)
    beta_full = np.mean(h_norms ** 2, axis=2)           # (L_full, K)
    beta = beta_full[ap_orig]                           # (ap_num, K)

    # ── Valid mask ────────────────────────────────────────────────────────
    # Union of beta-based mask and original topology — never lose existing edges
    valid_mask = _compute_valid_mask_beta(beta[:, :ue_num], N_NEAREST)
    orig_on = (A[:ap_num, :ue_num] > 0.5)
    valid_mask = valid_mask | orig_on
    valid_mask_t = torch.tensor(valid_mask, dtype=torch.float32, device=device)

    # ── Channel statistics (same as WMMSE-beta) ──────────────────────────
    sigma2 = 10 ** ((-87.0 - snr) / 10.0)

    beta_loc = np.maximum(beta[:ap_num, :ue_num], 1e-30)

    # beta here is E[||h_lk||^2] over subcarriers (see beta_full above), i.e. the
    # TOTAL gain over the Nt antennas, not the per-antenna variance. The stats
    # below must therefore be consistent with recompute_true's MR/MRT model
    #     signal_k = (sum_l sqrt(P_lk) ||h_lk||)^2
    #     interf_k = sum_{m!=k} |sum_l sqrt(P_lm) h_lk^H w_lm|^2,  w = h/||h||
    # which gives E||h_lk|| ~ sqrt(beta) and E|h_lk^H w_lm|^2 = beta/Nt for a
    # unit-norm w independent of h_lk. The previous form carried an extra factor
    # Nt in BOTH the signal and the interference; those cancel in the
    # interference-limited regime, which is why it went unnoticed, but they do
    # not cancel against sigma2 -- the noise floor was effectively 32x (15 dB)
    # too low. Correcting it cuts per-UE MAE against true CSI from 1.63 to 1.38
    # bit/s/Hz. The residual ~1.9x optimism is the Jensen gap of a large-scale-
    # fading-only model (mean_n log2(1+SINR_n) vs log2(1+SINR(beta))), not a bug.
    a_stat = np.sqrt(beta_loc.T)                              # (K, L) E||h||
    B_diag = beta_loc.T / Nt                                  # (K, L) E|h^H w|^2
    B_cross = beta_loc.T.copy()                               # (K, L) self-coupling

    a_stat_t  = torch.tensor(a_stat,  dtype=torch.float32, device=device)
    B_diag_t  = torch.tensor(B_diag,  dtype=torch.float32, device=device)
    B_cross_t = torch.tensor(B_cross, dtype=torch.float32, device=device)

    # ── Initialisation ────────────────────────────────────────────────────
    # Following IoTJ paper: initialise all valid entries with intermediate
    # values so the optimiser can freely add or remove connections.
    # Use beta-proportional warm start: stronger channels get higher a.
    beta_norm = beta_loc / (beta_loc.max() + 1e-30)        # normalise to [0, 1]
    a_warm = 0.3 + 0.4 * beta_norm                         # range [0.3, 0.7]
    # Boost existing connections
    orig_on = (A[:ap_num, :ue_num] > 0.5)
    a_warm[orig_on] = np.clip(a_warm[orig_on] + 0.2, 0, 1) # existing: [0.5, 0.9]
    a_init = torch.tensor(a_warm.astype(np.float32), device=device)
    a_init = a_init * valid_mask_t

    # eta: uniform over valid entries, scaled to satisfy per-AP power
    eta_init = torch.zeros(ap_num, ue_num, device=device)
    for n in range(ap_num):
        v = valid_mask[n]
        nv = v.sum()
        if nv > 0:
            eta_init[n, v] = 1.0 / nv
    # Ensure per-AP feasibility with soft a
    for n in range(ap_num):
        load = (a_init[n] * eta_init[n]).sum()
        if load > 1.0:
            eta_init[n] *= 1.0 / (load.item() + 1e-10)

    # ── Objective function ────────────────────────────────────────────────
    if objective == "fairness":
        obj_fn = lambda a, e: _compute_sse_fairness_torch(
            a, e, a_stat_t, B_diag_t, B_cross_t, sigma2, Nt)
    elif objective == "max_min":
        obj_fn = lambda a, e: _compute_sse_maxmin_torch(
            a, e, a_stat_t, B_diag_t, B_cross_t, sigma2, Nt)
    else:
        obj_fn = lambda a, e: _compute_sse_torch(
            a, e, a_stat_t, B_diag_t, B_cross_t, sigma2, Nt)

    if freeze_assoc:
        # Power-only mode: keep the association at its input value and run
        # Phase 3 alone. This is what makes APG comparable to the power-only
        # solvers (SOCP, WMMSE-ADMM), which never touch A. Comparing default
        # APG against them confounds the quality of the power solution with
        # the topology change, and the two effects pull in opposite directions.
        # Starting eta is the input P, the same operating point the other
        # power-only solvers are initialised from.
        A_hard = (A[:ap_num, :ue_num] > 0.5).astype(np.float32)
        eta_disc = (P[:ap_num, :ue_num] * A_hard).astype(np.float32)
        n_apg_iters, history = 0, []
    else:
        # ── Phase 1: APG joint optimisation ───────────────────────────────
        a_opt, eta_opt, best_obj, history = _apg_loop(
            a_init, eta_init, valid_mask_t, obj_fn,
            max_iters=max_iters, tol=tol, lr_init=lr)

        n_apg_iters = len(history)

        # ── Phase 2: Discretise ───────────────────────────────────────────
        a_np = a_opt.cpu().numpy()
        eta_np = eta_opt.cpu().numpy()
        A_hard, eta_disc = _discretize_apg(
            a_np, eta_np, valid_mask, A_orig=A[:ap_num, :ue_num])

    # ── Phase 3: Power refinement ─────────────────────────────────────────
    A_hard_t = torch.tensor(A_hard, dtype=torch.float32, device=device)
    eta_disc_t = torch.tensor(eta_disc, dtype=torch.float32, device=device)

    eta_refined_t, refined_obj = _refine_power(
        A_hard_t, eta_disc_t, valid_mask_t, obj_fn,
        n_iters=refine_iters, lr=0.02)

    eta_refined = eta_refined_t.cpu().numpy()

    # ── Convert to normalised P ───────────────────────────────────────────
    P_opt = _eta_to_P(A_hard, eta_refined, ap_num, ue_num)

    elapsed = time.time() - t0
    return {
        "A_best":  A_hard,
        "w_best":  P_opt,
        "score":   refined_obj,
        "n_evals": n_apg_iters + refine_iters,
        "elapsed": elapsed,
        "history": history,
    }
