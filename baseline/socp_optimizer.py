"""socp_optimizer.py — max-min downlink power control via SOCP bisection.

Classical β-oracle max-min power control for cell-free massive MIMO:
bisection over the SINR target t, with an inner SOCP feasibility check.
Adapted for sparse association: the power variable is forced to zero
wherever A_mk = 0. Orthogonal pilots are assumed (no pilot contamination),
which drops the rho_{k'k} slack variables from the paper's formulation.

Reformulation for numerical conditioning
----------------------------------------
Let u_mk = sqrt(gamma_mk * eta_mk) = sqrt(P_mk) be the per-AP-UE amplitude.
After the substitution, the per-AP power constraint becomes a clean
unit-ball:  sum_k u^2_mk <= 1  =>  ||u_m|| <= 1.

With rescaled channel stats  tilde_gamma = gamma * rho_d  and
tilde_beta = beta * rho_d, the per-UE SINR target becomes

    sum_m sqrt(tilde_gamma_mk) u_mk
        >= sqrt(t) * || [ sqrt(tilde_beta_mk) u_mk'  for all (m, k'); 1 ] ||

All coefficients are O(1)-O(100) and ECOS converges cleanly.

Output convention: P_mk = u^2_mk matches the project's per-AP-sum-=-1
normalisation (what recompute_true expects).
"""

import numpy as np
import cvxpy as cp


def _compute_gamma(beta, tau_p, rho_p):
    num = tau_p * rho_p * beta ** 2
    den = tau_p * rho_p * beta + 1.0
    return num / den


def _build_parametric_socp(tilde_gamma, tilde_beta, A):
    L, K = tilde_beta.shape
    u = cp.Variable((L, K), nonneg=True)
    sqrt_t = cp.Parameter(nonneg=True)

    constraints = []

    zero_idx = np.where(A < 0.5)
    if len(zero_idx[0]) > 0:
        constraints.append(u[zero_idx[0], zero_idx[1]] == 0)

    for m in range(L):
        constraints.append(cp.norm(u[m, :]) <= 1.0)

    sqrt_tilde_gamma = np.sqrt(np.maximum(tilde_gamma, 0.0))
    sqrt_tilde_beta = np.sqrt(np.maximum(tilde_beta, 0.0))
    for k in range(K):
        signal_k = cp.sum(cp.multiply(sqrt_tilde_gamma[:, k], u[:, k]))
        row_coefs = sqrt_tilde_beta[:, k]                      # (L,)
        coef_mat = np.broadcast_to(row_coefs[:, None], (L, K))  # (L, K)
        interf = cp.multiply(coef_mat, u)                       # (L, K)
        interf_flat = cp.reshape(interf, (L * K,), order="C")
        stacked = cp.hstack([interf_flat, np.array([1.0])])
        constraints.append(cp.norm(stacked) <= signal_k / sqrt_t)

    return cp.Problem(cp.Minimize(0), constraints), u, sqrt_t


def socp_max_min_downlink(beta, A, rho_d, tau_p=None, rho_p=None,
                          t_min=1e-6, t_max=None, tol=1e-3, max_iter=20,
                          solver="ECOS"):
    L, K = beta.shape
    if tau_p is None:
        tau_p = K
    if rho_p is None:
        rho_p = rho_d

    beta = np.asarray(beta, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)

    gamma = _compute_gamma(beta, tau_p, rho_p)
    tilde_gamma = gamma * rho_d
    tilde_beta = beta * rho_d

    problem, u_var, sqrt_t = _build_parametric_socp(tilde_gamma, tilde_beta, A)

    if t_max is None:
        t_max = float(tilde_gamma.max() * L)

    best_u = None
    best_t = 0.0
    for _ in range(max_iter):
        if t_max - t_min < tol * max(1.0, t_min):
            break
        t_mid = 0.5 * (t_min + t_max)
        sqrt_t.value = float(np.sqrt(t_mid))
        try:
            problem.solve(solver=solver, verbose=False,
                          warm_start=True)
        except Exception:
            t_max = t_mid
            continue

        if problem.status in ("optimal", "optimal_inaccurate") and u_var.value is not None:
            t_min = t_mid
            best_u = np.maximum(u_var.value, 0.0)
            best_t = t_mid
        else:
            t_max = t_mid

    if best_u is None:
        n_per_ap = np.maximum(A.sum(axis=1, keepdims=True), 1.0)
        best_u = A * np.sqrt(1.0 / n_per_ap)

    P = best_u ** 2
    return P.astype(np.float32), best_t
