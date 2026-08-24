"""Closed-loop validate → reweight → re-optimize logic.

Pure functions only — NO Streamlit / session_state / GNN dependency, so this
module is unit-testable in isolation. The orchestration that actually re-runs
``joint_optimize`` / ``power_optimize`` lives in ``app.py`` (``_closed_loop``),
which calls into the helpers here.

Design (see docs / CLAUDE.md §closed-loop):
  - Validation judges on the GNN digital-twin output ``pred_after`` against the
    user's hard requirements, with the optimiser's own baseline ``pred_before``.
  - When a requirement is unmet we ask the LLM to regenerate the *loss-shaping
    knobs* (``build_revision_prompt`` + ``parse_adjustments`` + ``apply_adjustments``);
    ``heuristic_reweight`` is the deterministic offline fallback.
"""
from __future__ import annotations

import copy
import json
import re

import numpy as np

# Loop budget / tolerance. Kept as module constants (no UI knob by design).
MAX_ITERS = 2         # 1 initial pass + at most 1 LLM-reweighted retry
TOL = 0.02            # a requirement counts as met at >= required * (1 - TOL)


# ─── Requirement detection ────────────────────────────────────────────────
def has_hard_requirement(goal: dict) -> bool:
    """True iff ``goal`` carries a per-UE / floor requirement we can validate.

    Plain ``max_sum_rate`` / ``fairness`` / floor-less ``max_min`` have no
    checkable "requirement" — the closed loop is a no-op for them and the
    caller should fall back to a single optimisation pass.
    """
    if goal.get("ue_multipliers"):
        return True
    if goal.get("ue_abs_floor"):
        return True
    if goal.get("min_rate_floor"):
        return True
    cf = goal.get("constraint_floor") or 0.0
    if cf and cf > 0:
        return True
    return False


# ─── Validation ───────────────────────────────────────────────────────────
def check_requirements(goal: dict, pred_before, pred_after, ue_num: int) -> dict:
    """Check ``pred_after`` (GNN digital twin) against ``goal``'s requirements.

    Returns ``{satisfied, score, total_gap, failures, n_req}`` where:
      - ``failures`` is a list of ``{ue, kind, required, achieved, gap_pct}``
      - ``score`` = met / n_req (fraction of requirements satisfied)
      - ``total_gap`` = Σ relative shortfall (tie-breaker for "best so far")
    """
    bl = np.asarray(pred_before, dtype=np.float64)[:ue_num]
    aft = np.asarray(pred_after, dtype=np.float64)[:ue_num]

    ue_mult = goal.get("ue_multipliers") or {}
    bare = set(goal.get("bare_targets") or [])
    uaf = goal.get("ue_abs_floor") or {}
    min_floor = goal.get("min_rate_floor")
    cf = goal.get("constraint_floor") or 0.0

    reqs = []   # (ue, kind, required, achieved)

    # Per-UE multipliers (covers primary targets + protected group).
    for k, m in ue_mult.items():
        k = int(k)
        if not (0 <= k < ue_num):
            continue
        if k in bare:
            # Bare target: user said "boost UEk" with no number → only require
            # it not to drop below baseline (the internal 1.5× is not a promise).
            reqs.append((k, "bare_target", float(bl[k]), float(aft[k])))
        else:
            reqs.append((k, "target", float(bl[k]) * float(m), float(aft[k])))

    # Per-UE absolute floors.
    for k, v in uaf.items():
        k = int(k)
        if 0 <= k < ue_num:
            reqs.append((k, "abs_floor", float(v), float(aft[k])))

    # Global min-rate floor (max_min): the weakest served UE must clear it.
    if min_floor and min_floor > 0:
        kmin = int(np.argmin(aft))
        reqs.append((kmin, "min_floor", float(min_floor), float(aft[kmin])))

    # constraint_floor: every *non-target* UE must keep >= floor × baseline.
    if cf and cf > 0:
        target_set = {int(k) for k in ue_mult.keys()}
        for k in range(ue_num):
            if k in target_set:
                continue
            reqs.append((k, "constraint_floor", float(bl[k]) * float(cf),
                         float(aft[k])))

    failures = []
    met = 0
    total_gap = 0.0
    for ue, kind, required, achieved in reqs:
        if achieved >= required * (1.0 - TOL):
            met += 1
        else:
            gap = required - achieved
            gap_pct = gap / (required + 1e-9) * 100.0
            total_gap += gap / (required + 1e-9)
            failures.append({
                "ue": ue, "kind": kind,
                "required": required, "achieved": achieved,
                "gap_pct": gap_pct,
            })

    n_req = len(reqs)
    score = 1.0 if n_req == 0 else met / n_req
    return {
        "satisfied": len(failures) == 0,
        "score": score,
        "total_gap": total_gap,
        "failures": failures,
        "n_req": n_req,
    }


# ─── Human-readable failure summary ────────────────────────────────────────
_KIND_ZH = {
    "target": "目标提升",
    "bare_target": "目标提升",
    "abs_floor": "绝对速率下限",
    "min_floor": "最差用户下限",
    "constraint_floor": "其他用户保护",
}
_KIND_EN = {
    "target": "target boost",
    "bare_target": "target boost",
    "abs_floor": "absolute rate floor",
    "min_floor": "min-rate floor",
    "constraint_floor": "others-protection",
}


def format_failures(failures: list, zh: bool) -> str:
    """One-line-per-failure description, reused in the prompt and the report."""
    if not failures:
        return "全部达标" if zh else "all requirements met"
    parts = []
    for f in failures:
        kind = (_KIND_ZH if zh else _KIND_EN).get(f["kind"], f["kind"])
        if zh:
            parts.append(
                f"UE{f['ue']} {kind}: 需 ≥{f['required']:.2f}, 实得 "
                f"{f['achieved']:.2f} (差 {f['gap_pct']:.0f}%)")
        else:
            parts.append(
                f"UE{f['ue']} {kind}: need ≥{f['required']:.2f}, got "
                f"{f['achieved']:.2f} (short {f['gap_pct']:.0f}%)")
    return "; ".join(parts)


# ─── LLM-driven loss regeneration ──────────────────────────────────────────
def build_revision_prompt(goal: dict, vr: dict, zh: bool) -> str:
    """Prompt the LLM to regenerate loss-shaping knobs for the failing items.

    The LLM must return ONLY a JSON object of *adjustments* (not a full goal),
    so its output can't be clobbered by the regex re-parsing in
    ``agent._validate_goal_fields``. Allowed keys mirror ``apply_adjustments``.
    """
    fail_str = format_failures(vr["failures"], zh)
    cur = {
        "ue_multipliers": goal.get("ue_multipliers") or {},
        "ue_abs_floor": goal.get("ue_abs_floor") or {},
        "min_rate_floor": goal.get("min_rate_floor"),
        "constraint_floor": goal.get("constraint_floor") or 0.0,
        "protected_ues": goal.get("protected_ues") or [],
        "secondary_type": goal.get("secondary_type"),
        "secondary_weight_scale": goal.get("secondary_weight_scale", 1.0),
    }
    cur_json = json.dumps(cur, ensure_ascii=False)
    if zh:
        return f"""你在调一个 cell-free MIMO 网络优化器的损失函数。上一轮优化后，数字孪生(GNN)验证显示以下要求**未达标**：

{fail_str}

当前损失旋钮(JSON)：
{cur_json}

请重新设计损失旋钮，让优化器**更用力地**满足这些未达标项。可用手段(按有效性排序)：
- **secondary_type(次要目标)**：当**主要目标已达成、但"最差用户下限/min_rate_floor"仍未达标**时，把 secondary_type 设成 "max_min"，把最差用户扶正成一个次要**目标**(而非仅仅罚项)。这是最有效的手段——单纯加大罚项权重通常顶不动最差用户，而把它变成目标能显著抬升(实测最差用户 0.36→0.82，吞吐几乎不掉)。
- **secondary_weight_scale(次要目标权重倍数，默认 1.0)**：在上面基础上再调大(如 2~4)可进一步放大次要/floor 目标的权重。单独用(没有 secondary_type 目标)基本无效。
- 提高未达标目标 UE 的 ue_multipliers(让优化器瞄得更高以补偿欠射，通常上调 10~30%)
- 提高未达标 UE 的 ue_abs_floor / min_rate_floor
- 把被违反保护的"其他用户"加入 protected_ues
- **不要**降低用户原本要求的目标值；只能往"更努力满足"的方向调

只输出一个 JSON 对象，键限定为：secondary_type("max_min"/"fairness"/"max_sum_rate"/null)、secondary_weight_scale(≥1 的数)、ue_multipliers(dict UE→倍数)、ue_abs_floor(dict UE→速率)、min_rate_floor(数)、constraint_floor(0~1 数)、protected_ues(UE 列表)。不要任何解释文字。"""
    return f"""You are tuning the loss function of a cell-free MIMO optimizer. After the last run, the digital-twin (GNN) validation shows these requirements were NOT met:

{fail_str}

Current loss knobs (JSON):
{cur_json}

Redesign the loss knobs so the optimizer pushes HARDER on the unmet items. Moves, in order of effectiveness:
- **secondary_type (secondary objective)**: when the PRIMARY objective is already met but a "worst-UE / min_rate_floor" requirement is still unmet, set secondary_type to "max_min" — this promotes the worst UE into a secondary *objective* rather than a bare penalty. This is the most effective move: merely increasing a penalty weight usually cannot lift the worst UE, whereas making it an objective does (measured: worst UE 0.36→0.82 with throughput almost unchanged).
- **secondary_weight_scale (secondary-objective weight multiplier, default 1.0)**: on top of the above, raise it (e.g. 2~4) to further amplify the secondary/floor objective. On its own (no secondary_type objective) it is largely ineffective.
- raise ue_multipliers for unmet target UEs (aim higher to compensate undershoot, typically +10~30%)
- raise ue_abs_floor / min_rate_floor for unmet floors
- add violated "other" UEs into protected_ues
- do NOT lower the user's original requested targets; only push toward "try harder to satisfy"

Output ONLY one JSON object with keys limited to: secondary_type ("max_min"/"fairness"/"max_sum_rate"/null), secondary_weight_scale (number ≥1), ue_multipliers (dict UE→factor), ue_abs_floor (dict UE→rate), min_rate_floor (number), constraint_floor (0..1 number), protected_ues (list of UE). No prose."""


def parse_adjustments(raw: str):
    """Extract the adjustments JSON object from an LLM reply (or ``None``)."""
    if not raw:
        return None
    # Prefer a fenced ```json block, else the first {...} span.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = m.group(1) if m else None
    if candidate is None:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        candidate = m.group(0) if m else None
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def apply_adjustments(goal: dict, adj: dict, ue_num: int) -> dict:
    """Merge an adjustments dict onto a deep copy of ``goal`` (clamped)."""
    g = copy.deepcopy(goal)
    if not isinstance(adj, dict):
        return g

    um = dict(g.get("ue_multipliers") or {})
    raw_um = adj.get("ue_multipliers")
    if isinstance(raw_um, dict):
        for k, v in raw_um.items():
            try:
                ki, vi = int(k), float(v)
            except (ValueError, TypeError):
                continue
            if 0 <= ki < ue_num and vi > 0:
                um[ki] = min(vi, 10.0)   # clamp runaway multipliers
    g["ue_multipliers"] = um

    uaf = dict(g.get("ue_abs_floor") or {})
    raw_uaf = adj.get("ue_abs_floor")
    if isinstance(raw_uaf, dict):
        for k, v in raw_uaf.items():
            try:
                ki, vi = int(k), float(v)
            except (ValueError, TypeError):
                continue
            if 0 <= ki < ue_num and vi > 0:
                uaf[ki] = vi
    g["ue_abs_floor"] = uaf or None

    if adj.get("min_rate_floor") is not None:
        try:
            mf = float(adj["min_rate_floor"])
            if mf > 0:
                g["min_rate_floor"] = mf
        except (ValueError, TypeError):
            pass

    if adj.get("constraint_floor") is not None:
        try:
            g["constraint_floor"] = max(0.0, min(1.0, float(adj["constraint_floor"])))
        except (ValueError, TypeError):
            pass

    # Secondary-objective escalation: the loop may promote a bare floor into a
    # real secondary objective (secondary_type="max_min" lifts the worst UE far
    # more than a heavier floor penalty). Validate; never duplicate the primary.
    st = adj.get("secondary_type")
    if isinstance(st, str) and st in ("fairness", "max_sum_rate", "max_min"):
        if st != g.get("type"):
            g["secondary_type"] = st

    # Secondary-objective weight multiplier (≥1; never below the current value
    # so the loop only ever pushes secondary harder, never softer). Clamped to
    # a sane ceiling to avoid a runaway that would crush the primary objective.
    if adj.get("secondary_weight_scale") is not None:
        try:
            sws = float(adj["secondary_weight_scale"])
            cur_sws = float(g.get("secondary_weight_scale", 1.0))
            g["secondary_weight_scale"] = max(cur_sws, min(sws, 8.0))
        except (ValueError, TypeError):
            pass

    raw_p = adj.get("protected_ues")
    if isinstance(raw_p, list):
        prot = set(g.get("protected_ues") or [])
        for k in raw_p:
            try:
                ki = int(k)
            except (ValueError, TypeError):
                continue
            if 0 <= ki < ue_num:
                prot.add(ki)
        g["protected_ues"] = sorted(prot)

    return g


# ─── Deterministic offline fallback ────────────────────────────────────────
def heuristic_reweight(goal: dict, vr: dict, ue_num: int) -> dict:
    """Offline reweighting when the LLM is unavailable.

    Strategy: for each failed item, push the corresponding loss knob in the
    "aim higher" direction by the undershoot ratio (capped), and promote
    violated ``constraint_floor`` UEs into the ue_multipliers/protected tier so
    they carry more gradient weight.
    """
    g = copy.deepcopy(goal)
    um = dict(g.get("ue_multipliers") or {})
    uaf = dict(g.get("ue_abs_floor") or {})
    prot = set(g.get("protected_ues") or [])

    for f in vr.get("failures", []):
        ue, kind = f["ue"], f["kind"]
        achieved = max(f["achieved"], 1e-6)
        ratio = min(f["required"] / achieved, 1.5)   # cap the over-correction

        if kind == "target":
            cur = um.get(ue, 1.0)
            um[ue] = min(cur * ratio, 10.0)
        elif kind == "bare_target":
            # Give the bare target a concrete, higher push so it stops drifting.
            um[ue] = min(max(um.get(ue, 1.5), 1.0) * ratio, 10.0)
        elif kind == "abs_floor":
            uaf[ue] = min(f["required"] * ratio, f["required"] * 1.5)
        elif kind == "min_floor":
            cur = g.get("min_rate_floor") or f["required"]
            g["min_rate_floor"] = cur * min(ratio, 1.3)
        elif kind == "constraint_floor":
            # Promote the violated "other" UE into the protected tier: give it a
            # concrete multiplier so _compute_loss weights it (5×→20× tier).
            floor = g.get("constraint_floor") or 0.9
            um[ue] = max(um.get(ue, 0.0), floor)
            prot.add(ue)

    # Objective escalation (offline mirror of the LLM's top move): if the
    # worst-UE / min-rate floor is unmet and nothing is driving the worst UE as
    # an objective yet, promote it to a max_min secondary objective — far more
    # effective than weight alone (weight only amplifies an existing objective).
    _has_min_floor_fail = any(f["kind"] == "min_floor" for f in vr.get("failures", []))
    if (_has_min_floor_fail and not g.get("secondary_type")
            and g.get("type") != "max_min"):
        g["secondary_type"] = "max_min"

    # Raise the secondary-objective weight when secondary / floor items are the
    # ones failing — the "primary met, secondary unmet → weight secondary
    # harder" rule. Bump by at least 2×, up to the worst undershoot, clamped.
    _sec_kinds = {"min_floor", "abs_floor", "constraint_floor"}
    _sec_ratios = [min(ff["required"] / max(ff["achieved"], 1e-6), 4.0)
                   for ff in vr.get("failures", []) if ff["kind"] in _sec_kinds]
    if _sec_ratios:
        cur_sws = float(g.get("secondary_weight_scale", 1.0))
        g["secondary_weight_scale"] = min(cur_sws * max(2.0, max(_sec_ratios)), 8.0)

    g["ue_multipliers"] = um
    g["ue_abs_floor"] = uaf or None
    g["protected_ues"] = sorted(prot)
    return g
