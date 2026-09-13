"""Pure-numpy helpers for moving UEs in the digital-twin app.

Ground truth exists only at the 2500 DeepMIMO ray-traced grid points, so a
moving UE is (a) predicted by the GNN at its exact continuous position and
(b) evaluated "truly" at the nearest grid point.  Motion is restricted to the
region within R_MAX metres of some grid point so that snap error stays small.

No Streamlit / torch imports here so the module is unit-testable in isolation
(tests/test_dyn_users.py) and safe for serve.py's pre-warm path.  The JS map
component (components/ue_map/ue_map.js) mirrors clamp_to_region and the
rejection rule of brownian_step exactly.
"""
import numpy as np

R_MAX = 3.0            # metres: max distance from the nearest measured grid point
DEFAULT_SIGMA = 1.0    # metres per motion tick (brownian_step default / tests)
SIGMA_MIN, SIGMA_MAX = 0.1, 5.0

# Pedestrian speed exposed in the UI (m/s); converted to a per-tick step below.
SPEED_MIN, SPEED_MAX = 0.0, 5.0
DEFAULT_SPEED = 2.5    # m/s, default demo speed


def speed_to_sigma(speed_mps, tick_ms):
    """Per-axis Gaussian step σ (m) for one animation tick so that the mean
    step length equals speed·Δt: for a 2-D isotropic Gaussian E|step| = σ·√(π/2)."""
    return float(max(speed_mps, 0.0)) * (tick_ms / 1000.0) * float(np.sqrt(2.0 / np.pi))


def nearest_grid(points, grid):
    """Index and distance of the nearest grid point for each row of `points`.

    points: (K,2), grid: (N,2)  ->  (idx (K,) int64, dist (K,) float32)
    Same formula as the argmin snap in app._load_sample.
    """
    p = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    g = np.asarray(grid, dtype=np.float32)
    d2 = ((p[:, None, :] - g[None, :, :]) ** 2).sum(-1)      # (K, N)
    idx = d2.argmin(axis=1)
    dist = np.sqrt(d2[np.arange(len(p)), idx]).astype(np.float32)
    return idx, dist


def snap_indices(points, grid):
    return nearest_grid(points, grid)[0]


def in_region(points, grid, r_max=R_MAX):
    return nearest_grid(points, grid)[1] <= r_max


def clamp_to_region(points, grid, r_max=R_MAX):
    """Pull every point farther than r_max from its nearest grid point back to
    exactly r_max along the same direction. Inside points are returned as-is."""
    p = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    g = np.asarray(grid, dtype=np.float32)
    idx, dist = nearest_grid(p, g)
    bad = dist > r_max
    if bad.any():
        anchor = g[idx[bad]]
        vec = p[bad] - anchor
        p[bad] = anchor + vec * (r_max / dist[bad])[:, None]
    return p


def brownian_step(points, sigma, grid, r_max=R_MAX, rng=None, max_tries=3):
    """One Gaussian random-walk step per UE. Steps that leave the region are
    redrawn up to max_tries times; a UE that still cannot move stays put."""
    rng = np.random.default_rng() if rng is None else rng
    p = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    if sigma <= 0 or len(p) == 0:
        return p
    g = np.asarray(grid, dtype=np.float32)
    pending = np.ones(len(p), dtype=bool)
    for _ in range(max_tries):
        if not pending.any():
            break
        cand = p[pending] + rng.normal(0.0, sigma, size=(int(pending.sum()), 2)).astype(np.float32)
        ok = in_region(cand, g, r_max)
        rows = np.where(pending)[0]
        p[rows[ok]] = cand[ok]
        pending[rows[ok]] = False
    return p


def rebuild_loc_norm(base_norm, ap_num, ue_num, ue_loc, loc_mean, loc_std):
    """Copy of the (42,2) normalised location tensor with only the UE rows
    [ap_num : ap_num+ue_num] replaced by (ue_loc - mean) / std.  AP rows and
    padding rows (which are (0-mean)/std in loc_test_norm, not 0) are kept."""
    out = np.array(base_norm, dtype=np.float32, copy=True)
    ue = np.asarray(ue_loc, dtype=np.float32).reshape(-1, 2)
    out[ap_num:ap_num + ue_num] = (ue - np.asarray(loc_mean, np.float32)) / np.asarray(loc_std, np.float32)
    return out


def grid_bounds(grid):
    g = np.asarray(grid, dtype=np.float32)
    return float(g[:, 0].min()), float(g[:, 0].max()), float(g[:, 1].min()), float(g[:, 1].max())


TOP_L = 4      # a UE may only be served by its TOP_L nearest APs (main.improved_random_selection)
MAX_K = 8      # an AP serves at most MAX_K UEs


def reassociate(A, ap_loc, ue_loc, top_l=TOP_L, max_k=MAX_K, rng=None):
    """Handover rule for moving UEs, mirroring main.CellFree.improved_random_selection.

    Keeps every existing link whose AP is still among the UE's top_l nearest;
    drops the others.  Then (step 1 of the generator) any UE left unserved is
    given its nearest AP with free capacity, and (step 2) random candidate
    links (nearest-top_l, AP has capacity) are added until the edge count is
    back to what it was, so the network keeps the density p it was drawn with.

    Returns (A_new float32 (L,K), added {(l,k)}, removed {(l,k)}).
    """
    rng = np.random.default_rng() if rng is None else rng
    A0 = np.asarray(A, dtype=np.float32)
    ap = np.asarray(ap_loc, dtype=np.float32); ue = np.asarray(ue_loc, dtype=np.float32)
    L, K = A0.shape
    if K == 0 or L == 0:
        return A0.copy(), set(), set()
    top_l = max(1, min(top_l, L))
    target = int(round(float(A0.sum())))

    diff = ap[:, None, :] - ue[None, :, :]
    dist2 = (diff * diff).sum(-1)                              # (L,K)
    order = np.argsort(dist2, axis=0)                          # nearest first per UE
    nearest = np.zeros((L, K), dtype=bool)
    nearest[order[:top_l, :], np.arange(K)] = True

    A1 = (A0 > 0.5) & nearest                                  # drop links to far APs
    # enforce capacity: shed the farthest UEs from over-full APs
    for l in range(L):
        served = np.where(A1[l])[0]
        if served.size > max_k:
            shed = served[np.argsort(dist2[l, served])[max_k:]]
            A1[l, shed] = False
    cap = max_k - A1.sum(axis=1)

    # step 1: every UE gets at least one serving AP (nearest with capacity)
    for k in np.where(A1.sum(axis=0) == 0)[0]:
        for l in order[:, k]:
            if cap[l] > 0:
                A1[l, k] = True; cap[l] -= 1
                break

    # step 2: restore the edge count with random nearest-top_l candidates.
    # UEs that just lost a link get first claim (a handover), then any UE.
    before = A0 > 0.5
    lost = np.where((before & ~A1).any(axis=0))[0]
    while A1.sum() < target:
        cand = nearest & ~A1 & (cap > 0)[:, None]
        if lost.size and cand[:, lost].any():
            mask = np.zeros(K, dtype=bool); mask[lost] = True
            cand &= mask[None, :]
        r_idx, c_idx = np.where(cand)
        if r_idx.size == 0:
            break
        i = int(rng.integers(r_idx.size))
        A1[r_idx[i], c_idx[i]] = True; cap[r_idx[i]] -= 1


    added = {(int(l), int(k)) for l, k in zip(*np.where(A1 & ~before))}
    removed = {(int(l), int(k)) for l, k in zip(*np.where(before & ~A1))}
    return A1.astype(np.float32), added, removed


def ratchet_range(prev, values, pad_abs, pad_rel, lo_floor=None):
    """Stable axis range for a live bar chart.

    prev=None  -> [min - m, max + m] with m = pad_rel*span + pad_abs (lo clipped
                  at lo_floor).  Otherwise prev is returned unchanged while all
                  finite values lie inside it, and only the exceeded side is
                  pushed out (again with margin).  None if no finite values.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None if prev is None else list(prev)
    vmin, vmax = float(v.min()), float(v.max())
    m = (vmax - vmin) * pad_rel + pad_abs
    if prev is None:
        lo, hi = vmin - m, vmax + m
    else:
        lo, hi = float(prev[0]), float(prev[1])
        if vmin < lo:
            lo = vmin - m
        if vmax > hi:
            hi = vmax + m
    if lo_floor is not None:
        lo = max(lo, lo_floor)
    return [lo, hi]
