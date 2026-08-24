"""wmmse_admm_optimizer.py — WMMSE-ADMM power allocation for Cell-Free Massive MIMO.

Implements Algorithm 1 (WMMSE) from:
  Zaher et al., "Learning-Based Downlink Power Allocation in Cell-Free
  Massive MIMO Systems," IEEE Trans. Wireless Commun., vol. 22, no. 1,
  pp. 174–188, Jan. 2023.

Step 5 (QCQP sub-problem) is solved via ADMM as described in [9]:
  Björnson et al., "Making Cell-Free Massive MIMO Competitive With MMSE
  Processing and Centralized Implementation," IEEE Trans. Wireless Commun.,
  vol. 19, no. 1, pp. 77–90, Jan. 2020.

ADMM decomposes the QCQP into per-UE unconstrained updates + per-AP
power projection, connected by dual variables.

Uses MR (Maximum Ratio) precoding consistent with our system (main.py).
Channel statistics helpers (_compute_channel_stats, _compute_channel_stats_beta)
live in this file.
"""

import time
import numpy as np


# ── Channel statistics ───────────────────────────────────────────────────────

def _compute_channel_stats(CSI, A_global):
    """Compute a_k and B_ki from CSI data with MR precoding.

    Parameters
    ----------
    CSI : ndarray (L_full, K, Nc, Nt, Nr)   — Nr=1 for single-antenna UEs
    A_global : ndarray (L_full, K)           — binary connection matrix

    Returns
    -------
    a : ndarray (K, L_full)   — a[k, l] = E[||h_{kl}||]
    B : ndarray (K, K, L_full, L_full) — interference coupling matrix
    """
    L, K, Nc, Nt, Nr = CSI.shape
    h = CSI[..., 0]                                       # (L, K, Nc, Nt)
    h_norms = np.linalg.norm(h, axis=-1)                  # (L, K, Nc)
    w = h / (h_norms[..., None] + 1e-30)                  # (L, K, Nc, Nt)

    # a_{kl} = E[||h_{kl}||]  averaged over subcarriers
    a = np.mean(h_norms, axis=2).T                        # (K, L)

    # α[k,i,l,nc] = h_{kl,nc}^H w_{il,nc}
    # Use einsum: h.conj() is (L, K, Nc, Nt), w is (L, K, Nc, Nt)
    # cross[l, k, i, nc] = Σ_t conj(h[l,k,nc,t]) * w[l,i,nc,t]
    cross = np.einsum('lknt,lint->lkin', h.conj(), w)     # (L, K, K, Nc)
    # Rearrange to (K, K, L, Nc) for convenience
    alpha = cross.transpose(1, 2, 0, 3)                   # (K, K, L, Nc)

    # B[k,i,l,m] = Re( (1/Nc) Σ_{nc} α[k,i,l,nc] * conj(α[k,i,m,nc]) )
    # = Re( (1/Nc) α[k,i,l,:] @ conj(α[k,i,m,:]) )
    # Vectorised: for each (k,i), outer product over L dim averaged over Nc
    B = np.zeros((K, K, L, L))
    for nc in range(Nc):
        a_nc = alpha[:, :, :, nc]                         # (K, K, L) complex
        # B[k,i,l,m] += Re(a_nc[k,i,l] * conj(a_nc[k,i,m])) / Nc
        # = Re(outer(a_nc[k,i,:], conj(a_nc[k,i,:]))) / Nc for each (k,i)
        for k in range(K):
            for i in range(K):
                B[k, i] += np.real(
                    np.outer(a_nc[k, i], a_nc[k, i].conj())) / Nc

    return a, B


def _compute_channel_stats_beta(beta, Nt=32):
    """β-only channel stats for MR precoding (hardened SINR / UatF bound).

    Replaces _compute_channel_stats when only large-scale fading is available.

    β is the TOTAL gain over the Nt antennas, β_kl = E[||h_kl||²]. That is what
    every caller passes (all of them compute `np.mean(||h||**2, axis=subcarrier)`),
    so the statistics below are written in that convention. Under
    h_kl ~ CN(0, (β_kl/Nt) · I_Nt) and MR precoding w = h/||h||:

        a[k, l]          = sqrt(β_kl)          (E||h_kl||)
        B[k, i, l, l]    = β_kl / Nt           for k ≠ i  (E|h_kl^H w_il|², w ⊥ h)
        B[k, i, l, m]    = 0                   for k ≠ i, l ≠ m
        B[k, k, l, l]    = β_kl                (signal power: E[||h||²])
        B[k, k, l, m]    = a[k,l] · a[k,m]     for l ≠ m  (cross-AP coherent)

    This previously read β as the PER-ANTENNA gain (a = sqrt(β·Nt), B[k,i] = β,
    B[k,k] = β·Nt) while the callers passed the total — an extra factor Nt in both
    the signal and the interference. Those cancel in the interference-limited
    regime, which hid the mismatch, but not against σ²: the noise floor came out
    Nt = 32 (15 dB) too low. Correcting it cuts per-UE MAE against true CSI from
    1.63 to 1.38 bit/s/Hz. Solvers that also choose the topology are the most
    sensitive, since whether an AP is worth switching on is set by the noise
    floor. Residual ~1.9x optimism is the Jensen gap inherent to any
    large-scale-fading-only model, not a defect.

    Parameters
    ----------
    beta : (L, K) — large-scale fading coefficients, TOTAL gain E[||h||²]
    Nt   : int    — transmit antennas per AP

    Returns
    -------
    a : (K, L)           — expected channel gain
    B : (K, K, L, L)     — interference coupling matrix
    """
    L, K = beta.shape
    beta = np.maximum(np.asarray(beta, dtype=np.float64), 1e-30)

    a = np.sqrt(beta).T                              # (K, L)

    B = np.zeros((K, K, L, L))
    for k in range(K):
        # Self-coupling B[k,k]: rank-1 outer product + diagonal correction
        B[k, k] = np.outer(a[k], a[k])
        np.fill_diagonal(B[k, k], beta[:, k])        # E[||h||²] on diagonal
        # Cross-coupling B[k,i]: diagonal only
        for i in range(K):
            if i != k:
                np.fill_diagonal(B[k, i], beta[:, k] / Nt)

    return a, B


# ── ADMM solver for Step 5 QCQP ─────────────────────────────────────────────

def _admm_solve_qcqp(C, rhs, K, L, A_mask, P_max,
                      mu_warm, rho_admm=1.0, max_admm_iters=50, tol_admm=1e-5):
    """Solve the QCQP sub-problem via ADMM.

    Problem:
        min  Σ_j  [ μ_j^T C_j μ_j - 2 rhs_j^T μ_j ]
        s.t. Σ_k μ_{kl}^2 ≤ P_max,  l = 1,...,L
             μ_{kl} = 0  for A_mask[k,l] = False

    ADMM formulation (consensus):
        min  f(μ)  s.t. μ = z,  z ∈ constraint set
        Augmented Lagrangian: f(μ) + u^T(μ - z) + (ρ/2)||μ - z||^2

    Updates:
        1. μ-update (per-UE): (C_j + ρI) μ_j = rhs_j + ρ(z_j - u_j)
        2. z-update (per-AP projection): project onto Σ_k z_{kl}^2 ≤ P_max
        3. u-update: u = u + μ - z

    Parameters
    ----------
    C : (K, L, L)     — per-UE quadratic cost matrices
    rhs : (K, L)      — per-UE linear cost vectors (ω_j v_j a_j)
    K, L : int
    A_mask : (K, L)   — bool connection mask
    P_max : float     — per-AP power budget
    mu_warm : (K, L)  — warm start
    rho_admm : float  — ADMM penalty parameter
    max_admm_iters : int
    tol_admm : float  — primal residual tolerance

    Returns
    -------
    mu : (K, L)       — optimised √power coefficients
    """
    mu = mu_warm.copy()
    z = mu.copy()
    u = np.zeros((K, L))

    # Precompute (C_j + ρI)^{-1} for each UE j — these are fixed within ADMM
    C_inv = np.zeros((K, L, L))
    for j in range(K):
        M = C[j] + rho_admm * np.eye(L)
        M += 1e-10 * np.eye(L)  # regularise
        try:
            C_inv[j] = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            C_inv[j] = np.eye(L) / rho_admm

    for _it in range(max_admm_iters):
        mu_old = mu.copy()

        # ── Step 1: μ-update (per-UE, unconstrained) ────────────────────
        for j in range(K):
            b = rhs[j] + rho_admm * (z[j] - u[j])
            mu[j] = C_inv[j] @ b
            mu[j] *= A_mask[j]  # zero out non-connected

        # ── Step 2: z-update (per-AP projection onto power ball) ────────
        # For each AP l, project {mu[k,l] + u[k,l]} onto ||·||^2 ≤ P_max
        v_proj = mu + u  # (K, L)
        for l in range(L):
            col = v_proj[:, l].copy()
            col *= A_mask[:, l]  # only connected UEs
            power = np.sum(col ** 2)
            if power > P_max:
                # Scale down to satisfy constraint
                scale = np.sqrt(P_max / (power + 1e-30))
                z[:, l] = col * scale
            else:
                z[:, l] = col

        # ── Step 3: dual update ─────────────────────────────────────────
        u = u + mu - z

        # ── Convergence check (primal residual) ─────────────────────────
        primal_res = np.linalg.norm(mu - z)
        if primal_res < tol_admm:
            break

    return z  # z satisfies constraints


# ── WMMSE-ADMM iteration ────────────────────────────────────────────────────

def _wmmse_admm_iterate(a, B, K, L, A_mask, sigma2, P_max,
                        mu_init, objective, max_iters, tol,
                        admm_iters=50, rho_admm=1.0):
    """Run WMMSE iterations with ADMM for the QCQP sub-problem.

    Parameters
    ----------
    a : (K, L)          — expected channel gains
    B : (K, K, L, L)    — interference coupling
    A_mask : (K, L)     — bool, True where connection exists
    sigma2 : float      — noise power
    P_max  : float      — per-AP power budget
    mu_init : (K, L)    — initial √power coefficients
    objective : str     — "max_sum_rate" or "fairness"
    max_iters : int     — outer WMMSE iterations
    tol : float         — convergence tolerance
    admm_iters : int    — inner ADMM iterations per WMMSE step
    rho_admm : float    — ADMM penalty parameter

    Returns
    -------
    P_opt : (K, L)      — optimised ρ_{kl} = μ_{kl}^2
    obj   : float       — final objective value
    iters : int
    """
    mu = mu_init.copy()
    prev_obj = -np.inf

    for iteration in range(max_iters):
        # ── Steps 2-3: update v_k, e_k ────────────────────────────────
        v = np.zeros(K)
        e = np.zeros(K)
        for k in range(K):
            denom = sigma2
            for i in range(K):
                denom += mu[i] @ B[k, i] @ mu[i]
            numer = a[k] @ mu[k]
            v[k] = numer / (denom + 1e-30)
            e[k] = max(1.0 - numer ** 2 / (denom + 1e-30), 1e-12)

        # ── Step 4: update ω_k ────────────────────────────────────────
        if objective == "fairness":
            omega = np.array([-1.0 / (ek * np.log(ek) + 1e-30) for ek in e])
        else:  # max_sum_rate
            omega = 1.0 / (e + 1e-30)

        # ── Objective value (for convergence check) ───────────────────
        sinr = np.maximum(1.0 / e - 1.0, 1e-12)
        if objective == "fairness":
            obj = float(np.sum(np.log(np.log2(1 + sinr) + 1e-30)))
        else:
            obj = float(np.sum(np.log2(1 + sinr)))

        if abs(obj - prev_obj) / (abs(prev_obj) + 1e-12) < tol:
            break
        prev_obj = obj

        # ── Step 5: update μ via ADMM ─────────────────────────────────
        # Precompute C_j for each UE j
        wv2 = omega * v ** 2                               # (K,)
        C = np.zeros((K, L, L))
        for j in range(K):
            for k in range(K):
                C[j] += wv2[k] * B[k, j]

        rhs = np.zeros((K, L))
        for j in range(K):
            rhs[j] = omega[j] * v[j] * a[j]

        # Solve QCQP via ADMM
        mu = _admm_solve_qcqp(
            C, rhs, K, L, A_mask, P_max,
            mu_warm=mu, rho_admm=rho_admm,
            max_admm_iters=admm_iters, tol_admm=1e-6)

    # Convert μ → ρ = μ²
    rho = mu ** 2                                          # (K, L)
    return rho, obj, iteration + 1


# ── Public interface ─────────────────────────────────────────────────────────

def wmmse_admm_optimize(ue_csi, ap_orig, A, P, snr, ap_num, ue_num,
                        objective="max_sum_rate", max_iters=50, tol=1e-4):
    """WMMSE-ADMM power allocation using true CSI.

    Same interface as wmmse_optimize() but uses ADMM for the QCQP sub-problem.

    Parameters
    ----------
    ue_csi  : ndarray (L_full, K, Nc, Nt, Nr)
    ap_orig : ndarray (ap_num,)
    A       : ndarray (ap_num, ue_num)
    P       : ndarray (ap_num, ue_num)
    snr     : int
    ap_num, ue_num : int
    objective : str — "max_sum_rate" or "fairness"

    Returns
    -------
    dict with keys: A_best, w_best, score, n_evals, elapsed
    """
    t0 = time.time()

    L_full = ue_csi.shape[0]
    K      = ue_num

    # Map local A/P → global (L_full × K)
    A_global = np.zeros((L_full, K), dtype=np.float32)
    P_global = np.zeros((L_full, K), dtype=np.float32)
    for i, la in enumerate(ap_orig):
        A_global[la] = A[i]
        P_global[la] = P[i]

    # Noise
    sigma2 = 10 ** ((-87 - snr) / 10.0)
    P_max  = 1.0

    # Compute channel statistics
    csi_k = ue_csi[:, :K, :, :, :]
    a, B = _compute_channel_stats(csi_k, A_global)

    # Connection mask (K, L_full)
    A_mask = (A_global.T > 0.5)

    # Initialise μ from current P
    mu_init = np.sqrt(P_global.T * P_max + 1e-12) * A_mask

    # Run WMMSE-ADMM
    rho, obj, iters = _wmmse_admm_iterate(
        a, B, K, L_full, A_mask, sigma2, P_max,
        mu_init, objective, max_iters, tol)

    # Map back: global ρ (K, L_full) → local P (ap_num, ue_num)
    P_opt = np.zeros((ap_num, ue_num), dtype=np.float32)
    for i, la in enumerate(ap_orig):
        row = rho[:, la]
        s = row.sum()
        P_opt[i] = (row / s) if s > 1e-12 else P[i]

    elapsed = time.time() - t0
    return {
        "A_best":  A.copy(),
        "w_best":  P_opt,
        "score":   obj,
        "n_evals": iters,
        "elapsed": elapsed,
    }
