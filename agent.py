"""agent.py — LLM intent parser, goal validator, and report formatter.

No optimiser lives here anymore: the two proposed methods (Joint Optimize
and Power-Only) are in ``joint_optimize.py`` / ``power_only.py``. This
module only handles natural-language parsing and the bilingual HTML
report that the Streamlit UI renders.
"""

import re
import json
import numpy as np


# "Others" alternation — captures standard Chinese phrases plus common
# typos / alternate characters (e.g. 替他 for 其他, 其它 for 其他,
# 别的/另外 as synonyms). Used by every regex that detects an
# "其他用户" clause in the user's raw text.
_OTHERS_RE = r'(?:其[他它]|其余|其馀|剩[余下]|替他|别的|别人|另外|other|others|rest)'


# ─── Intent detection ─────────────────────────────────────────────────────────

_WHITE_PATTERNS = [
    r'(优化|最大化|最小化|提升|提高|改善|增强|boost|maximize|minimize|improve|optimize)',
    r'(保障|确保|保证).{0,6}(公平|fairness)',
    r'(公平|fairness|吞吐量|throughput|最差用户|worst.{0,4}ue|max.{0,4}min)',
    r'(让|使|make|帮).{0,15}(rate|速率|频谱效率|拉起|拉上|提上).{0,10}(更高|更好|higher|better|翻倍|倍|来|去)?',
    r'(整体|全局|overall|network).{0,10}(性能|performance|rate)',
    r'(翻倍|倍|double|triple|x\b)',
    r'(不能降|不降低|不下降|降低不超过|drop.{0,5}(at most|no more))',
    r'(牺牲|sacrifice|trade.?off)',
    r'(拉起来|拉上去|拉一拉|拉高)',
    r'(最大|最小).{0,4}(ue|UE|用户|user)',
    r'(不管|不care|无所谓|don.?t\s*care)',
]
_BLACK_PATTERNS = [
    r'(解释|说明|介绍|explain|describe|tell\s+me\s+about|what\s+is)',
    r'^(什么|如何|怎么|how\s+to|what\s+is|tell\s+me)',
    r'(rate|速率).{0,6}(是多少|是什么|how\s+much|what\s+is)',
]


def detect_optimization_intent(text: str) -> bool:
    """Return True if text is an optimization request (not a question/explanation)."""
    t = text.strip()
    for pat in _BLACK_PATTERNS:
        if re.search(pat, t, re.IGNORECASE):
            return False
    for pat in _WHITE_PATTERNS:
        if re.search(pat, t, re.IGNORECASE):
            return True
    return False


# ─── Goal parsing ─────────────────────────────────────────────────────────────


def build_unified_prompt(text: str, network_state: str, chat_history: str = "") -> str:
    """Single LLM prompt that handles ALL user inputs: routing + parsing.

    Returns a prompt whose expected output is always a JSON object with an "action" field:
      - "optimization": GoalDict fields for the optimizer
      - "query": {"response": "..."} direct answer using network data
      - "chat": {"response": "..."} greeting/casual reply
      - "command": {"actions": [...]} JSON action array
    """
    return (
        "LANGUAGE RULE (MUST follow): Default to ENGLISH. Only reply in Chinese if the user's input itself contains Chinese characters. Pure English / digits / punctuation → English reply, even if the topic sounds Chinese-related.\n\n"
        "You are an intelligent assistant for a cell-free massive MIMO network digital twin.\n\n"
        f"{network_state}\n\n"
        f"{chat_history}"
        f'User: "{text}"\n\n'
        "Classify the user input and respond with ONLY a compact JSON object (no markdown, no explanation).\n\n"
        "== ACTION TYPES ==\n\n"
        '1. OPTIMIZATION request (improve/change/optimize rates, boost UE, maximize throughput, fairness, etc.):\n'
        "{\n"
        '  "action": "optimization",\n'
        '  "power_only": false,  // true ONLY when user explicitly says "不改拓扑/只优化功率/keep topology/power only/fixed topology"\n'
        '  "type": "targeted" | "max_sum_rate" | "fairness" | "max_min",\n'
        '  "secondary_type": null | "fairness" | "max_sum_rate" | "max_min",\n'
        '  "target_ues": [int],\n'
        '  "target_multiplier": float|null,\n'
        '  "protected_ues": [int],\n'
        '  "protected_floor": float,\n'
        '  "constraint_floor": float,\n'
        '  "min_rate_floor": null | float,\n'
        '  "ue_abs_floor": null | {"ue_index": float},  // per-UE absolute rate floor (bits/s/Hz)\n'
        '  "post_query": null | "string",  // trailing question to answer AFTER optimizing — see POST-QUERY RULES\n'
        '  "summary": "string"  // concise goal description with priority labels — see SUMMARY RULES\n'
        "}\n\n"
        '2. QUERY about the network (rate queries, "which UE is best/worst", AP info, signal/interference):\n'
        '{"action": "query", "response": "concise answer in 1-2 sentences using the network data above"}\n\n'
        '3. GREETING or CASUAL CHAT (hello, 你好, good morning, chitchat):\n'
        '{"action": "chat", "response": "friendly short reply, briefly mention what you can do"}\n\n'
        '4. COMMAND (select UE, change connections, adjust power, reset, next/prev topology, set SNR):\n'
        '{"action": "command", "actions": [...]}\n'
        "Available commands:\n"
        '  {"action":"select_ue","ue_index":N}\n'
        '  {"action":"set_connection","ap_index":N,"ue_index":M,"connected":true|false}\n'
        '  {"action":"connect_nearest_aps","ue_index":N,"n":M}\n'
        '  {"action":"set_power_weight","ap_index":N,"ue_index":M,"weight":W}   (W: 0.05-1.0)\n'
        '  {"action":"reset"}\n'
        '  {"action":"next_topology"}\n'
        '  {"action":"prev_topology"}\n'
        '  {"action":"set_snr","value":N}   (N: 0/5/10/15/20/25/30)\n\n'
        "== OPTIMIZATION RULES ==\n"
        '- "AP9的用户" or "users served by AP9" → look up AP9\'s served UEs from the network state\n'
        '- Two-level protection: protected_ues use protected_floor, remaining non-target UEs use constraint_floor\n'
        '- Default protected_floor = 1.0\n'
        '- constraint_floor default: ALWAYS 0.0 unless the user EXPLICITLY mentions "其他用户/others drop at most X%". Never add implicit protection for non-target UEs.\n'
        '- "其他用户/others drop at most X%" → constraint_floor = 1 - X/100\n'
        '- "其他人无所谓/don\'t care about others" → constraint_floor = 0.0\n'
        '- CRITICAL: TWO boost groups with DIFFERENT multipliers → MUST split into target_ues + protected_ues. NEVER merge them into one target_ues!\n'
        '  First group → target_ues + target_multiplier\n'
        '  Second group → protected_ues + protected_floor (>1.0 means boost)\n'
        '  If user also mentions "其他用户降低不超过X%" → constraint_floor = 1-X/100. Otherwise constraint_floor = 0.0.\n'
        '  e.g. "用户1,3,5提升20%, 2,4,6提升30%, 其他用户降低不超过20%" → target_ues=[1,3,5], target_multiplier=1.2, protected_ues=[2,4,6], protected_floor=1.3, constraint_floor=0.8\n'
        '  e.g. "用户1,2,3提升20%, 4,5,6提升50%" → target_ues=[1,2,3], target_multiplier=1.2, protected_ues=[4,5,6], protected_floor=1.5, constraint_floor=0.0\n'
        '  e.g. "UE1,3,5提升50%, UE0,2,4提升两倍" → target_ues=[1,3,5], target_multiplier=1.5, protected_ues=[0,2,4], protected_floor=2.0, constraint_floor=0.0\n'
        '  Key: when user lists TWO separate UE groups with TWO different percentages/multipliers separated by comma/、, they are ALWAYS TWO separate objectives. Count the groups!\n'
        '- "两倍"/"double"/"2x" → target_multiplier=2.0. "提升X%" → target_multiplier=1+X/100\n'
        '- type="max_sum_rate": 总吞吐/sum rate/throughput/系统容量/频谱效率/spectral efficiency/最大化速率/最大速率\n'
        '- type="fairness": 公平/fairness/比例公平\n'
        '- type="max_min": 最差用户/worst UE (only when MAXIMIZING, not a numeric floor). When type="max_min", do NOT add target_ues or secondary_type — soft-min already handles the worst UE automatically.\n'
        '- type="targeted": ONLY when NO global objective keyword\n'
        '- CRITICAL: global objective keywords determine type FIRST. If user says a global keyword AND names UE groups, type=global keyword, UE groups go in target_ues/protected_ues.\n'
        '- CRITICAL: "在X的情况下/前提下/基础上" or "保障X" as precondition → X is PRIMARY (type). The following action → secondary_type.\n'
        '  e.g. "保障公平的情况下，最大化吞吐量" → type="fairness", secondary_type="max_sum_rate"\n'
        '- secondary_type: TWO global objectives with priority. null if only one.\n'
        '- "所有用户/all users" + multiplier → type="targeted", target_ues=[] (code fills all UEs)\n'
        '- min_rate_floor: ABSOLUTE minimum rate threshold for GLOBAL worst UE. "不低于X"/"至少X" + number → min_rate_floor=X. NOT max_min.\n'
        '  max_min = maximize worst UE (no number). min_rate_floor = worst UE must be ≥ X.\n'
        '- ue_abs_floor: per-UE ABSOLUTE rate floor. When user says SPECIFIC UEs must be ≥ N bits/s/Hz.\n'
        '  e.g. "用户2,4,6不得低于1" → ue_abs_floor={"2":1.0,"4":1.0,"6":1.0}. NOT protected_ues (which is ratio-based).\n'
        '  e.g. "UE3 rate >= 0.5, UE5 rate >= 1.0" → ue_abs_floor={"3":0.5,"5":1.0}\n'
        '\n== POST-QUERY RULES ==\n'
        '- If the input contains BOTH an optimization request AND a question to answer afterwards, '
        'classify as action="optimization" and copy the question part VERBATIM (original language, '
        'no rewriting/translation) into "post_query".\n'
        '- Pure optimization (no question) → omit post_query (or null). Pure question → action="query" as usual.\n'
        '- Examples:\n'
        '  "运行maxmin算法，使得最小用户速率最大，并且告诉我当前每个UE的速率" → optimization JSON + "post_query": "告诉我当前每个UE的速率"\n'
        '  "maximize throughput and then tell me which UE is worst" → optimization JSON + "post_query": "tell me which UE is worst"\n'
        '  "告诉我每个UE的速率" (question only, no optimization) → action="query"\n'
        '\n== SUMMARY RULES ==\n'
        '- "summary" is a concise description of ALL objectives, each prefixed with its priority label.\n'
        '- Use three priority levels in order: 主要(primary) → 次要(secondary) → 补充(supplementary)\n'
        '- Each objective gets the next available level. Separate with "，" (Chinese) or ", " (English).\n'
        '- Match user language (Chinese → Chinese labels, English → English labels).\n'
        '- Format: "主要: <desc>，次要: <desc>，补充: <desc>"\n'
        '- The order in summary MUST match: type first, then secondary_type, then target_ues, then protected_ues/others, then min_rate_floor — each taking the next priority level.\n'
        '- CRITICAL: target_ues and protected_ues are SEPARATE objectives at DIFFERENT priority levels. NEVER merge them into one level!\n'
        '  e.g. type=max_sum_rate + target_ues=[1,2,3] + protected_ues=[4,5,6] → THREE levels: 主要(type), 次要(target_ues), 补充(protected_ues)\n'
        '- Examples:\n'
        '  "最大化吞吐量" → "主要: 最大化总速率"\n'
        '  "最大化吞吐量，用户1,2,3提升20%" → "主要: 最大化总速率，次要: UE1,2,3提升20%"\n'
        '  "最大化吞吐量，用户1,2,3提升20%，最小用户rate不低于0.5" → "主要: 最大化总速率，次要: UE1,2,3提升20%，补充: 最差用户速率≥0.5"\n'
        '  "最大化频谱效率，用户1,2,3提升20%，用户4,5,6提升30%" → "主要: 最大化总速率，次要: UE1,2,3提升20%，补充: UE4,5,6提升30%"\n'
        '  "保障公平，最大化吞吐量" → "主要: 比例公平，次要: 最大化总速率"\n'
        '  "maximize throughput, UE1,2,3 boost 20%, UE4,5,6 boost 30%" → "primary: maximize sum rate, secondary: UE1,2,3 +20%, supplementary: UE4,5,6 +30%"\n'
        '  "maximize throughput, UE1 boost 2x, worst UE >= 0.5" → "primary: maximize sum rate, secondary: UE1 2x, supplementary: worst UE rate ≥0.5"\n'
        '  "用户1,3,5提升20%，用户2,4,6不低于1" → "主要: UE1,3,5提升20%，次要: UE2,4,6速率≥1.0 bps/Hz"\n'
        '  "提升用户1,2速率，系统最低用户不低于0.5" → "主要: UE1,2提升速率，次要: 最差用户速率≥0.5"\n'
        '- CRITICAL: When type="targeted", target_ues is ALWAYS 主要. min_rate_floor/ue_abs_floor/constraint are ALWAYS lower priority than target_ues.\n'
    )


def parse_unified_response(raw: str, ue_num: int, raw_text: str = "") -> dict:
    """Parse unified LLM JSON response. Returns dict with "action" key."""
    if not raw:
        return None
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

    action = parsed.get("action", "")
    if action in ("chat", "query"):
        return {"action": action, "response": parsed.get("response", "")}
    if action == "command":
        return {"action": "command", "actions": parsed.get("actions", [])}
    if action == "optimization":
        # Reuse existing GoalDict validation
        goal = _validate_goal_fields(parsed, ue_num, raw_text)
        if goal:
            goal["action"] = "optimization"
            # post_query: trailing question to answer after optimizing (compound
            # "optimize + query" instructions). Attached AFTER _validate_goal_fields
            # so its regex re-parse cannot clobber it (§14.1-style lesson).
            _pq = parsed.get("post_query")
            if isinstance(_pq, str) and _pq.strip():
                goal["post_query"] = _pq.strip()
            return goal
        return None
    return None


# ─── Post-query helpers (compound "optimize + query" instructions) ───────────

_ALL_UE_METRICS = {
    # metric → (keyword pattern, unit, zh label, en label, worst = argmax?)
    # 干扰越大越差(worst=max);速率/信号越小越差(worst=min)
    "rate":   (r'(?:速率|rate|频谱效率|吞吐|throughput)', "bits/s/Hz",
               "速率", "rates", False),
    "interf": (r'(?:干扰|interference|interf)', "dBW",
               "干扰", "interference", True),
    "signal": (r'(?:信号(?:功率|强度)?|signal(?:power|strength)?)', "dBW",
               "信号", "signal power", False),
}


def all_ue_metric_query(text: str):
    """Detect an "all/each UE <metric>" query; return "rate" / "interf" /
    "signal" or None (specific UE index or optimization phrasing → None).

    Pure function so it is unit-testable without importing app.py. The caller
    (app._regex_answer) additionally bails out on optimization keywords first.
    """
    if not text:
        return None
    t = re.sub(r"[\s\-'’]", '', text.lower())
    if re.search(r'(?:ue|user|用户)\d', t):
        return None           # specific UE → per-UE branch elsewhere
    if re.search(r'(提升|提高|优化|最大化|最小化|降低|减少|boost|improve|maximize|minimize|optimize|increase|reduce)', t):
        return None           # optimization phrasing, not a query
    _quant = r'(?:每个|每一个|每位|所有|全部|各个|各|all|each|every|per)'
    _ue    = r'(?:ue|user|用户)'
    gap    = r'[^,;。.?？]{0,12}?'
    for metric, (kw, *_rest) in _ALL_UE_METRICS.items():
        if (re.search(_quant + _ue + gap + kw, t) or
                re.search(kw + gap + _quant + _ue, t)):
            return metric
    return None


def format_all_ue_metric(true_vals, pred_vals, ue_num: int, zh: bool = True,
                         metric: str = "rate") -> str:
    """Styled HTML heat-table of a per-UE metric (true + GNN-predicted).

    ONE table, two data rows — UEs are columns, 真实 row + 预测 row — inside an
    ``overflow-x:auto`` container so a wide UE count scrolls horizontally instead
    of chunking into stacked blocks (user preference 2026-07-03). Cell fills use
    ONE sequential hue (light→dark blue #f3f8fd→#70a6d2, darker = higher value)
    with fixed ink text (contrast ≈6.7:1 on the dark end); the worst true-value
    UE (rate/signal: min, interference: max) is marked ▼ + bold + red outline
    (glyph + shape, not color-alone). Rendered via st.markdown(...,
    unsafe_allow_html=True).
    """
    _kw, unit, lbl_zh, lbl_en, worst_is_max = _ALL_UE_METRICS[metric]
    fmt = "{:.2f}" if metric == "rate" else "{:.1f}"
    lbl_true, lbl_pred = ("真实", "预测") if zh else ("True", "Pred")
    vals_t = [float(true_vals[k]) for k in range(ue_num)]
    vals_p = [float(pred_vals[k]) for k in range(ue_num)]
    lo = min(vals_t + vals_p)
    span = (max(vals_t + vals_p) - lo) or 1.0
    k_worst = vals_t.index(max(vals_t) if worst_is_max else min(vals_t))

    def bg(v):  # sequential ramp: #f3f8fd (low) → #70a6d2 (high)
        f = (v - lo) / span
        r, g, b = (round(243 + (112 - 243) * f), round(248 + (166 - 248) * f),
                   round(253 + (210 - 253) * f))
        return f"#{r:02x}{g:02x}{b:02x}"

    _cell = "padding:3px 9px;text-align:center;color:#1a1c22;white-space:nowrap"
    _head = ("padding:3px 9px;text-align:center;color:#555;font-weight:600;"
             "white-space:nowrap")
    ks = range(ue_num)
    hdr = "".join(
        f'<th style="{_head}">{"▼" if k == k_worst else ""}UE{k}</th>' for k in ks)
    row_t = "".join(
        f'<td style="{_cell};background:{bg(vals_t[k])}'
        f'{";font-weight:700;outline:2px solid #c0392b" if k == k_worst else ""}">'
        f'{fmt.format(vals_t[k])}</td>' for k in ks)
    row_p = "".join(
        f'<td style="{_cell};background:{bg(vals_p[k])}">{fmt.format(vals_p[k])}</td>'
        for k in ks)
    if zh:
        _worst_desc = {"rate": "最差", "interf": "干扰最大", "signal": "信号最弱"}[metric]
        worst_note = f"▼ = 当前{_worst_desc} UE（按真实值）"
        title = f"当前每个 UE 的{lbl_zh} ({unit})："
    else:
        _worst_desc = {"rate": "worst", "interf": "highest-interference",
                       "signal": "weakest-signal"}[metric]
        worst_note = f"▼ = current {_worst_desc} UE (by true value)"
        title = f"Current per-UE {lbl_en} ({unit}):"
    return (
        f"{title}\n"
        '<div style="overflow-x:auto;max-width:100%">'
        '<table style="border-collapse:collapse;font-size:0.82rem;'
        'margin:4px 0;border:1px solid #e0e4ea;border-radius:6px">'
        f'<tr><th style="{_head}"></th>{hdr}</tr>'
        f'<tr><th style="{_head}">{lbl_true}</th>{row_t}</tr>'
        f'<tr><th style="{_head}">{lbl_pred}</th>{row_p}</tr>'
        "</table></div>\n"
        f'<span style="color:#888;font-size:0.78rem">{worst_note}</span>')


def _regex_extract_ue_multipliers(raw_text: str, ue_num: int) -> dict:
    """Extract per-UE multipliers from raw text, supporting N boost groups.

    Handles two shapes:
      - number + unit:   "UE1,2,3提升20%", "UE4,5,6 boost 30%", "UE7 2x"
      - Chinese numeral: "2,4,6翻倍"/"double", "1,3,5两倍", "2,4三倍"
    Returns dict {ue_index: multiplier} or empty dict.
    """
    if not raw_text:
        return {}
    ue_multipliers = {}

    # Pattern A: digit value + unit
    _patt_digit = (
        r'(?:用户|ue|UE|user)?\s*'
        r'(\d+(?:[,，、\s]+\d+)*)(?!\d)'  # UE list; (?!\d) prevents splitting "10" into UE1+val0
        r'[^\d%％倍x,，;；。.]*?'      # skip filler (速率, 的rate, etc.); stop at clause boundaries
        r'(?:提升|提高|boost|increase|improve|\+)?\s*'
        r'(\d+(?:\.\d+)?)\s*([%％倍x])'  # value + unit
    )
    for m in re.finditer(_patt_digit, raw_text, re.IGNORECASE):
        ues = sorted(set(int(x) for x in re.findall(r'\d+', m.group(1))
                         if 0 <= int(x) < ue_num))
        val = float(m.group(2))
        unit = m.group(3)
        mult = val if unit in ('倍', 'x', 'X') else 1.0 + val / 100.0
        for k in ues:
            ue_multipliers[k] = mult

    # Pattern B: UE list + Chinese-numeral / word multiplier (no digit value)
    # e.g. "2,4,6翻倍", "1,3,5两倍", "UE2,4,6 double"
    _word_to_mult = {
        '翻倍': 2.0, '两倍': 2.0, '双倍': 2.0, 'double': 2.0,
        '三倍': 3.0, 'triple': 3.0,
        '四倍': 4.0, 'quadruple': 4.0,
    }
    _patt_word = (
        r'(?:用户|ue|UE|user)?\s*'
        r'(\d+(?:[,，、\s]+\d+)*)(?!\d)'
        r'[^,，;；。.]{0,15}?'
        r'(翻倍|两倍|双倍|三倍|四倍|double|triple|quadruple)'
    )
    for m in re.finditer(_patt_word, raw_text, re.IGNORECASE):
        ues = sorted(set(int(x) for x in re.findall(r'\d+', m.group(1))
                         if 0 <= int(x) < ue_num))
        word = m.group(2).lower()
        mult = _word_to_mult.get(word)
        if mult is None:
            continue
        for k in ues:
            if k not in ue_multipliers:      # digit pattern takes precedence
                ue_multipliers[k] = mult
    return ue_multipliers


def _validate_goal_fields(parsed: dict, ue_num: int, raw_text: str = "") -> dict:
    """Validate and sanitise GoalDict fields from LLM response."""
    goal_type = parsed.get("type", "max_sum_rate")
    if goal_type not in ("targeted", "max_sum_rate", "fairness", "max_min"):
        goal_type = "max_sum_rate"

    def _clean_ue_list(key):
        return sorted(set(
            int(k) for k in parsed.get(key, [])
            if 0 <= int(k) < ue_num
        ))

    target_ues    = _clean_ue_list("target_ues")
    protected_ues = _clean_ue_list("protected_ues")
    protected_ues = [k for k in protected_ues if k not in target_ues]

    mult = parsed.get("target_multiplier")
    _raw_text = raw_text or parsed.get("raw_text", "")
    if mult is not None:
        mult = float(mult)
        if mult <= 0:
            # Check if user wants to decrease — keep a small positive multiplier
            _wants_decrease = bool(re.search(
                r'减少|降低|减小|下降|decrease|reduce|lower|drop',
                _raw_text, re.IGNORECASE))
            mult = 0.01 if _wants_decrease else None

    # ── Build unified ue_multipliers from all sources ──
    ue_multipliers = {}

    # Source 1: LLM-provided ue_multipliers (preferred)
    raw_um = parsed.get("ue_multipliers")
    if raw_um and isinstance(raw_um, dict):
        for k_str, v in raw_um.items():
            k = int(k_str)
            if 0 <= k < ue_num:
                ue_multipliers[k] = float(v)

    # Source 2: regex N-group extraction from raw text
    if not ue_multipliers and _raw_text:
        ue_multipliers = _regex_extract_ue_multipliers(_raw_text, ue_num)

    # Source 2b: regex fallback for implicit boost / protect patterns the
    # LLM may have missed (no digit after the phrase):
    #   "提升用户1"      → target_ues += [1]
    #   "用户2,3,4不下降" → protected_ues += [2,3,4]
    # Regex findings OVERRIDE the LLM classification — they come straight
    # from the user's text and are more reliable than LLM guessing.
    _regex_t = set()
    _regex_p = set()
    if _raw_text:
        for _m in re.finditer(
            r'(?:提升|提高|boost|increase|improve)\s*(?:用户|UE|user)?\s*'
            r'(\d+(?:[,，、\s]+\d+)*)(?!\d)',
            _raw_text, re.IGNORECASE):
            # Skip if this boost has an explicit value (handled by Source 2)
            _after = _raw_text[_m.end():_m.end() + 10]
            if re.match(r'[^\d%％倍x]{0,5}[\d.]+\s*[%％倍x]', _after):
                continue
            for _k_s in re.findall(r'\d+', _m.group(1)):
                _k = int(_k_s)
                if 0 <= _k < ue_num:
                    _regex_t.add(_k)
        for _m in re.finditer(
            r'(?:用户|UE|user)?\s*(\d+(?:[,，、\s]+\d+)*)(?!\d)'
            r'\s*(?:不(?:下降|降低|能下降|能降低)|no\s*(?:degradation|drop))',
            _raw_text, re.IGNORECASE):
            for _k_s in re.findall(r'\d+', _m.group(1)):
                _k = int(_k_s)
                if 0 <= _k < ue_num:
                    _regex_p.add(_k)
    if _regex_t or _regex_p:
        # Merge: regex is authoritative for UEs it explicitly matched.
        target_ues    = sorted((set(target_ues) | _regex_t) - _regex_p)
        protected_ues = sorted((set(protected_ues) | _regex_p) - set(target_ues))

    # Source 2d: realign LLM target/protected groups against regex-extracted
    # ue_multipliers. Fires when the regex identified a multi-group boost
    # pattern (≥2 distinct multipliers) AND contains UEs the LLM missed.
    # Regex reads the user's text directly, so its grouping is
    # authoritative. Without this, inputs like "UE1,3,5 +30%, UE2,4,6 +50%"
    # where the LLM mis-extracts target_ues=[1,2] would cause UE1,2 to be
    # tagged as bare targets (unbounded push) and UE3-6 to fall through.
    _realigned = False
    if ue_multipliers and mult is None:
        _llm_covered = set(target_ues) | set(protected_ues)
        _regex_ues   = set(ue_multipliers.keys())
        _distinct    = len(set(ue_multipliers.values()))
        if _distinct >= 2 and (_regex_ues - _llm_covered):
            from collections import defaultdict
            _groups = defaultdict(list)
            for k, m in sorted(ue_multipliers.items()):
                _groups[m].append(k)
            _sorted_groups = sorted(_groups.items(),
                                    key=lambda x: len(x[1]), reverse=True)
            if _sorted_groups:
                mult        = _sorted_groups[0][0]
                target_ues  = sorted(_sorted_groups[0][1])
                if len(_sorted_groups) >= 2:
                    protected_ues             = sorted(_sorted_groups[1][1])
                    parsed["protected_floor"] = _sorted_groups[1][0]
                else:
                    protected_ues = []
                _realigned = True

    # Source 2c: regex fallback for absolute rate floors (LLM often
    # mis-classifies "UE2,4,6 不得低于 1" as protected_ues instead of
    # ue_abs_floor). Patterns:
    #   "用户2,4,6 不得低于 1[bit/s/Hz]"
    #   "UE1,3 rate ≥ 0.5"
    #   "UE5 至少 1.2"
    _regex_uaf = {}
    if _raw_text:
        for _m in re.finditer(
            r'(?:用户|UE|user)?\s*(\d+(?:[,，、\s]+\d+)*)(?!\d)'
            r'[^\d]{0,15}?'
            r'(?:不得低于|不低于|不能低于|至少|不少于|大于等于|大于|≥|>=|>|'
            r'at\s*least|no\s*less\s*than|greater\s*than\s*or\s*equal)'
            r'\s*([\d.]+)',
            _raw_text, re.IGNORECASE):
            try:
                _val = float(_m.group(2))
            except ValueError:
                continue
            if _val <= 0:
                continue
            for _k_s in re.findall(r'\d+', _m.group(1)):
                _k = int(_k_s)
                if 0 <= _k < ue_num:
                    _regex_uaf[_k] = _val
    if _regex_uaf:
        # These UEs are absolute-floor constraints, NOT relative boost/protect.
        # Strip them from target/protected and feed into parsed.ue_abs_floor.
        target_ues    = [k for k in target_ues if k not in _regex_uaf]
        protected_ues = [k for k in protected_ues if k not in _regex_uaf]
        _existing_uaf = parsed.get("ue_abs_floor") or {}
        if not isinstance(_existing_uaf, dict):
            _existing_uaf = {}
        for _k, _v in _regex_uaf.items():
            _existing_uaf[str(_k)] = _v
        parsed["ue_abs_floor"] = _existing_uaf

    # Source 3: backfill ue_multipliers from target_ues / protected_ues.
    # Runs unconditionally so any UE the regex missed still lands in _um.
    #
    # "Bare" targets = target_ues where the user did NOT give a specific
    # multiplier (e.g. "提升用户1" without a number). We still need a
    # concrete internal target for the optimiser, so we quietly default
    # to 1.5. The display layer flags these UEs via _bare_targets and
    # renders them as "提升UEx" (no explicit %) so the summary doesn't
    # invent numbers the user didn't say.
    _bare_targets = set()
    if target_ues and mult is None:
        # A UE is only "bare" (unbounded push) if it has no multiplier
        # from any source. If the regex already filled ue_multipliers for
        # some target UEs, those have a concrete target and must NOT be
        # treated as bare — otherwise they explode past the user's ask.
        _bare_targets = set(k for k in target_ues if k not in ue_multipliers)
        mult = 1.5  # internal default, hidden from display
    if target_ues and mult is not None:
        for k in target_ues:
            if k not in ue_multipliers:
                ue_multipliers[k] = mult
    _pf_raw = parsed.get("protected_floor")
    _pf_val = float(_pf_raw) if _pf_raw is not None else 1.0
    if protected_ues:
        for k in protected_ues:
            if k not in ue_multipliers:
                ue_multipliers[k] = _pf_val

    # Back-populate target_ues/protected_ues from ue_multipliers for the
    # regex-only case (LLM gave nothing). Skip when target_ues is already
    # set — otherwise we would erase LLM's intent and re-split the groups
    # purely by multiplier magnitude.
    if ue_multipliers and not target_ues:
        from collections import defaultdict
        _groups = defaultdict(list)
        for k, m in sorted(ue_multipliers.items()):
            _groups[m].append(k)
        _sorted_groups = sorted(_groups.items(), key=lambda x: len(x[1]), reverse=True)
        if len(_sorted_groups) >= 1:
            mult = _sorted_groups[0][0]
            target_ues = sorted(_sorted_groups[0][1])
        if len(_sorted_groups) >= 2:
            protected_ues = sorted(_sorted_groups[1][1])
            parsed["protected_floor"] = _sorted_groups[1][0]
        # Remaining groups (3+) only accessible via ue_multipliers

    # "其他用户不下降" 这类约束在中文里有 **两种语序**, 必须都覆盖,
    # 否则像 "并不降低其他用户" 会被悄悄丢掉:
    #   forward : 其他用户 ... 不下降 / 保持不变      (OTHERS 在前, 动词在后)
    #   reverse : 不降低 / 牺牲 / 保持 ... 其他用户   (动词在前, OTHERS 在后)
    # negation modals: 不 / 不得 / 不能 / 不要 / 不会 / 不应; lowering verbs
    # share one list so both word orders cover the same vocabulary.
    _NEG = r'不(?:得|能|要|会|应)?'
    _DOWN = r'(?:降低|下降|下滑|掉|变差|受影响|减少|减小|降)'
    _NODROP_FWD = (
        _OTHERS_RE + r'.{0,15}'
        r'(?:' + _NEG + _DOWN + r'|不变|保持|维持|'
        r'maintain|keep|hold|unchanged|no.{0,5}degradation|no.{0,5}drop)')
    _NODROP_REV = (
        r'(?:' + _NEG + r'\s*(?:降低|减少|减小|削减|牺牲|影响|损害|拉低|拖累|降)'
        r'|保持|维持|不影响'
        r'|don\'?t\s+(?:lower|reduce|hurt|drop)'
        r'|without\s+(?:lower|reduc|hurt|drop|degrad|affect)\w*)'
        r'\s*[^，,；;。]{0,6}?' + _OTHERS_RE)
    _others_nodrop = bool(re.search(_NODROP_FWD, _raw_text, re.IGNORECASE)
                          or re.search(_NODROP_REV, _raw_text, re.IGNORECASE))

    # Extract constraint_floor from text if present (percentage drop cap)
    _cf_match = re.search(
        _OTHERS_RE + r'[^%]*?'
        r'(?:降低|下降|减少|减小|降|drop|decrease|reduce)[^%]*?(\d+)\s*[%％]',
        _raw_text, re.IGNORECASE)
    if _cf_match:
        parsed["constraint_floor"] = 1.0 - float(_cf_match.group(1)) / 100.0
    elif _others_nodrop:
        parsed["constraint_floor"] = 1.0

    # If user wants to decrease specific UEs but no multiplier was set, default to 0.01
    # Exclude constraint phrases like "其他用户下降不超过/others drop at most" —
    # those describe a floor for non-target UEs, not an intent to decrease target UEs.
    if goal_type == "targeted" and target_ues and mult is None and not ue_multipliers:
        _decrease_text = re.sub(
            _OTHERS_RE + r'.{0,30}'
            r'(?:减少|降低|减小|下降|不超过|drop|decrease|at most)[^,，;；]*',
            '', _raw_text, flags=re.IGNORECASE)
        _wants_decrease = bool(re.search(
            r'减少|降低|减小|下降|decrease|reduce|lower|drop',
            _decrease_text, re.IGNORECASE))
        if _wants_decrease:
            mult = 0.01

    if goal_type == "targeted" and not target_ues:
        if mult is not None:
            target_ues = list(range(ue_num))
        else:
            goal_type = "max_sum_rate"

    # Gate: did the user actually mention an "others" constraint? Reuse the
    # both-word-order _others_nodrop result + the % path (_cf_match) so this
    # can never fall out of sync with the extraction above.
    _user_said_others = bool(
        _others_nodrop or _cf_match or re.search(
            _OTHERS_RE + r'.{0,15}(?:不超过|降低|下降|drop|decrease|at most)',
            _raw_text, re.IGNORECASE))
    # Only accept constraint_floor if the user's raw text actually mentioned
    # an "others" constraint — otherwise the LLM may hallucinate a spurious
    # value from unrelated numbers (e.g. "最低用户不得低于0.3" leaking into
    # constraint_floor=0.3).
    _raw_floor = parsed.get("constraint_floor")
    if _raw_floor is not None and _user_said_others:
        floor = float(_raw_floor)
    else:
        floor = 0.0
    floor = max(0.0, min(1.0, floor))

    _pf_raw = parsed.get("protected_floor")
    p_floor = float(_pf_raw) if _pf_raw is not None else 1.0
    if p_floor < 0.5 and p_floor < 1.0:
        p_floor = 0.5

    sec_type = parsed.get("secondary_type")
    if sec_type not in (None, "fairness", "max_sum_rate", "max_min"):
        sec_type = None
    if sec_type == goal_type:
        sec_type = None

    min_rate_floor = parsed.get("min_rate_floor")
    if min_rate_floor is not None:
        min_rate_floor = float(min_rate_floor)
        if min_rate_floor <= 0:
            min_rate_floor = None

    # A pure "raise the worst user to X" instruction has the min-rate floor as
    # its ONLY / primary objective. When nothing else was requested (no target
    # UEs, no per-UE multipliers, no secondary objective) AND the user's text
    # never asked for sum-rate / throughput, a max_sum_rate type — whether the
    # LLM's default or the empty-targeted downgrade above — is spurious: it
    # chases throughput and starves the worst UE, demoting the real goal to a
    # secondary penalty the optimiser fights against (the "最大化总速率 primary,
    # floor secondary, UE unmet" bug). Drive it with max_min so the floor is
    # the primary objective; the floor value is still checked by the closed loop.
    if (goal_type == "max_sum_rate" and min_rate_floor is not None
            and not target_ues and not ue_multipliers
            and parsed.get("secondary_type") in (None, "")
            and not re.search(r'总|sum|吞吐|throughput|total|系统容量|频谱效率|'
                              r'spectral\s*efficiency', _raw_text, re.IGNORECASE)):
        goal_type = "max_min"
        _zh_floor = bool(re.search(r'[一-鿿]', _raw_text))
        parsed["summary"] = (
            f"主要: 最差用户速率≥{min_rate_floor:g} bps/Hz" if _zh_floor
            else f"primary: worst-UE rate ≥ {min_rate_floor:g}")

    # Per-UE absolute rate floor
    _uaf_raw = parsed.get("ue_abs_floor")
    ue_abs_floor = {}
    if isinstance(_uaf_raw, dict):
        for k, v in _uaf_raw.items():
            try:
                ki = int(k)
                vi = float(v)
                if 0 <= ki < ue_num and vi > 0:
                    ue_abs_floor[ki] = vi
            except (ValueError, TypeError):
                pass

    # Prune _bare_targets to UEs that actually made it into ue_multipliers
    # (otherwise they were overridden by a concrete mult from another source).
    _bare_targets = {k for k in _bare_targets if k in ue_multipliers}

    # Regenerate summary when realignment overrode the LLM's groups — the
    # LLM's summary string reflects its mis-extracted target_ues and would
    # be confusing to display alongside the corrected optimisation.
    _summary = parsed.get("summary")
    if _realigned:
        _parts = []
        if target_ues and mult is not None:
            _pct = int(round((float(mult) - 1) * 100))
            _parts.append(
                f"主要: UE{','.join(str(k) for k in target_ues)}提升{_pct}%")
        if protected_ues:
            _pf = float(parsed.get("protected_floor") or 1.0)
            _pct2 = int(round((_pf - 1) * 100))
            _parts.append(
                f"次要: UE{','.join(str(k) for k in protected_ues)}提升{_pct2}%")
        if floor > 0 and floor < 1:
            _drop = int(round((1 - floor) * 100))
            _parts.append(f"补充: 其他用户下降不超过{_drop}%")
        if _parts:
            _summary = "，".join(_parts)

    result = {
        "type":              goal_type,
        "secondary_type":    sec_type,
        "target_ues":        target_ues,
        "alpha":             0.7,
        "constraint_floor":  floor,
        "protected_floor":   p_floor,
        "target_multiplier": mult,
        "protected_ues":     protected_ues,
        "ue_multipliers":    ue_multipliers,
        "min_rate_floor":    min_rate_floor,
        "ue_abs_floor":      ue_abs_floor if ue_abs_floor else None,
        "bare_targets":      sorted(_bare_targets) if _bare_targets else [],
        "summary":           _summary,
    }
    # Pass through power_only flag if present
    if parsed.get("power_only"):
        result["power_only"] = True
    return result


def _regex_parse_goal(text: str, ss_snap: dict, zh: bool) -> dict:
    """Regex-only goal parsing — fallback when LLM is unavailable."""
    t = text.lower()

    targets = [int(m) for m in re.findall(
        r'(?:ue|用户|user|UE)\s*(\d+)', t, re.IGNORECASE)]
    targets = sorted(set(k for k in targets if 0 <= k < ss_snap["ue_num"]))

    if re.search(r'不降低|不能降|不下降|must\s+not\s+(?:decrease|drop|degrade)|no\s+degradation|maintain', t):
        floor = 1.0
    elif re.search(r'不能|must\s+not|不可以', t):
        floor = 0.95
    elif re.search(r'稍微|slightly', t):
        floor = 0.80
    else:
        floor = 0.0

    _cn_num = {'两': 2, '二': 2, '三': 3, '四': 4, '五': 5,
               '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
    mult_match = (re.search(r'(\d+(?:\.\d+)?)\s*(?:倍|x|times)', t) or
                  re.search(r'([两二三四五六七八九十])\s*倍', t) or
                  re.search(r'(?:double|翻倍)', t))
    if mult_match:
        g0 = mult_match.group(0)
        if g0 in ('double', '翻倍'):
            target_multiplier = 2.0
        elif mult_match.group(1) in _cn_num:
            target_multiplier = float(_cn_num[mult_match.group(1)])
        else:
            target_multiplier = float(mult_match.group(1))
    else:
        target_multiplier = None

    if targets:
        goal_type = "targeted"
    elif re.search(r'最差|worst|max.?min|min.?max', t):
        goal_type = "max_min"
    elif re.search(r'公平|fairness|比例|proportional', t):
        goal_type = "fairness"
    elif re.search(r'总|sum|吞吐|throughput|total', t):
        goal_type = "max_sum_rate"
    else:
        goal_type = "max_sum_rate"

    return {
        "type":              goal_type,
        "target_ues":        targets,
        "alpha":             0.7,
        "constraint_floor":  floor,
        "protected_floor":   1.0,
        "target_multiplier": target_multiplier,
        "protected_ues":     [],
        "raw_text":          text,
        "zh":                zh,
    }


def parse_goal(text: str, ss_snap: dict, llm_fn=None) -> dict:
    """Regex-only goal parser (legacy fallback when the unified LLM path
    is unavailable). `llm_fn` kept for API compat but unused — the main
    UI path uses ``build_unified_prompt`` + ``parse_unified_response``
    directly. If you need LLM-level parsing, prefer the unified path.
    """
    zh = bool(re.search(r'[\u4e00-\u9fff]', text))
    return _regex_parse_goal(text, ss_snap, zh)


# ─── Report formatter ─────────────────────────────────────────────────────────

def format_report(goal: dict, before_rate: np.ndarray, after_rate: np.ndarray,
                  A_before: np.ndarray, A_after: np.ndarray,
                  P_before: np.ndarray, P_after: np.ndarray,
                  elapsed: float, n_evals: int, zh: bool,
                  pred_before: np.ndarray = None,
                  pred_after: np.ndarray = None,
                  orig_before: np.ndarray = None,
                  orig_pred_before: np.ndarray = None) -> str:
    """Build bilingual before/after optimization report.

    ``orig_before`` / ``orig_pred_before``: rates of the ORIGINAL (unmodified)
    allocation. Passed only when the optimization started from an
    already-optimized/modified state — the report then shows the gain vs the
    original allocation alongside the incremental one, and replaces the
    coin-flip ✔/❌ (noise around an already-converged point) with a neutral
    "converged" tag. See CLAUDE.md §15 / session 2026-07-03.
    """
    K       = len(before_rate)
    targets = set(goal.get("target_ues") or [])
    ap_num  = A_before.shape[0]

    lines = []
    g_type = goal.get("type", "max_sum_rate")

    if zh:
        lines.append(f"优化完成 ({elapsed:.1f}s)  ")
    else:
        lines.append(f"Optimization done ({elapsed:.1f}s)  ")

    # ── Parsed goal summary ──────────────────────────────────────────────────
    _summary = goal.get("_summary")
    if _summary:
        if zh:
            lines.append(f"语义解析: {_summary}  ")
        else:
            lines.append(f"Parsed: {_summary}  ")

    # Helper: status tag based on whether true metric improved
    def _global_status(improved: bool) -> str:
        if zh:
            return (' <span style="color:green">✔ 已达成</span>' if improved
                    else " ❌ 未达成")
        return (' <span style="color:green">✔ done</span>' if improved
                else " ❌ not met")

    _has_orig = orig_before is not None

    def _start_status(pct_true: float, improved: bool) -> str:
        """Status for the vs-start line. When the start was already an
        optimized state and the change vs it is within noise (<1%), show a
        neutral converged tag instead of a coin-flip ✔/❌."""
        if _has_orig and abs(pct_true) < 1.0:
            return (' <span style="color:green">✔ 已收敛（起点已是优化状态）</span>'
                    if zh else
                    ' <span style="color:green">✔ converged (start already optimized)</span>')
        return _global_status(improved)

    if _has_orig:
        lines.append("⚠ 本次起点已是先前优化/修改后的状态，以下同时给出相对原始分配的增益  "
                     if zh else
                     "⚠ Starting point was already optimized/modified; gains vs the original allocation are also shown  ")

    # ── Sum-rate summary (for max_sum_rate only) ────────────────────────────
    if g_type == "max_sum_rate":
        sum_b = float(np.sum(before_rate))
        sum_a = float(np.sum(after_rate))
        pct_s = (sum_a - sum_b) / (sum_b + 1e-9) * 100
        _st = _start_status(pct_s, sum_a >= sum_b - 1e-6)

        if pred_before is not None and pred_after is not None:
            ps_b = float(np.sum(pred_before))
            ps_a = float(np.sum(pred_after))
            pct_p = (ps_a - ps_b) / (ps_b + 1e-9) * 100
            if zh:
                _sr_content = f"预测速率: {ps_b:.2f} → {ps_a:.2f} bits/s/Hz ({pct_p:+.1f}%)&emsp;真实速率: {sum_b:.2f} → {sum_a:.2f} bits/s/Hz ({pct_s:+.1f}%){_st}"
            else:
                _sr_content = f"Predicted rate: {ps_b:.2f} → {ps_a:.2f} bits/s/Hz ({pct_p:+.1f}%)&emsp;True rate: {sum_b:.2f} → {sum_a:.2f} bits/s/Hz ({pct_s:+.1f}%){_st}"
        else:
            if zh:
                _sr_content = f"真实速率: {sum_b:.2f} → {sum_a:.2f} bits/s/Hz ({pct_s:+.1f}%){_st}"
            else:
                _sr_content = f"True rate: {sum_b:.2f} → {sum_a:.2f} bits/s/Hz ({pct_s:+.1f}%){_st}"
        if _has_orig:
            os_b = float(np.sum(orig_before))
            pct_os = (sum_a - os_b) / (os_b + 1e-9) * 100
            _ost = _global_status(sum_a >= os_b - 1e-6)
            _sr_content += (
                f"<br>相对原始分配: 真实速率 {os_b:.2f} → {sum_a:.2f} bits/s/Hz ({pct_os:+.1f}%){_ost}"
                if zh else
                f"<br>vs original allocation: true rate {os_b:.2f} → {sum_a:.2f} bits/s/Hz ({pct_os:+.1f}%){_ost}")
        _sr_label = "主要目标" if zh else "Primary Objective"
        lines.append("")
        lines.append(
            f'<details open><summary>{_sr_label}</summary>'
            f'<div style="background:#f0f2f6;padding:8px 12px;border-radius:6px;margin:6px 0">'
            f'{_sr_content}</div></details>'
        )

    # ── Proportional fairness summary (sum-log-rate objective) ──────────────
    if goal.get("type") == "fairness":
        def _sum_log(r):
            return float(np.sum(np.log2(np.maximum(r, 1e-9))))
        sl_b = _sum_log(before_rate)
        sl_a = _sum_log(after_rate)
        pct_sl = (sl_a - sl_b) / (abs(sl_b) + 1e-9) * 100
        _st = _start_status(pct_sl, sl_a >= sl_b - 1e-6)

        if pred_before is not None and pred_after is not None:
            psl_b = _sum_log(pred_before)
            psl_a = _sum_log(pred_after)
            pct_psl = (psl_a - psl_b) / (abs(psl_b) + 1e-9) * 100
            if zh:
                _fair_content = f"预测 Σlog2(rate): {psl_b:.2f} → {psl_a:.2f} ({pct_psl:+.1f}%)&emsp;真实 Σlog2(rate): {sl_b:.2f} → {sl_a:.2f} ({pct_sl:+.1f}%){_st}"
            else:
                _fair_content = f"Predicted Σlog2(rate): {psl_b:.2f} → {psl_a:.2f} ({pct_psl:+.1f}%)&emsp;True Σlog2(rate): {sl_b:.2f} → {sl_a:.2f} ({pct_sl:+.1f}%){_st}"
        else:
            if zh:
                _fair_content = f"真实 Σlog2(rate): {sl_b:.2f} → {sl_a:.2f} ({pct_sl:+.1f}%){_st}"
            else:
                _fair_content = f"True Σlog2(rate): {sl_b:.2f} → {sl_a:.2f} ({pct_sl:+.1f}%){_st}"
        if _has_orig:
            osl_b = _sum_log(orig_before)
            pct_osl = (sl_a - osl_b) / (abs(osl_b) + 1e-9) * 100
            _ost = _global_status(sl_a >= osl_b - 1e-6)
            _fair_content += (
                f"<br>相对原始分配: 真实 Σlog2(rate) {osl_b:.2f} → {sl_a:.2f} ({pct_osl:+.1f}%){_ost}"
                if zh else
                f"<br>vs original allocation: true Σlog2(rate) {osl_b:.2f} → {sl_a:.2f} ({pct_osl:+.1f}%){_ost}")
        _fair_label = "主要目标" if zh else "Primary Objective"
        lines.append("")
        lines.append(
            f'<details open><summary>{_fair_label}</summary>'
            f'<div style="background:#f0f2f6;padding:8px 12px;border-radius:6px;margin:6px 0">'
            f'{_fair_content}</div></details>'
        )

    # ── Max-min fairness summary ────────────────────────────────────────────
    if goal.get("type") == "max_min":
        k_before = int(np.argmin(before_rate))
        k_after  = int(np.argmin(after_rate))
        min_b    = before_rate[k_before]
        min_a    = after_rate[k_after]
        pct      = (min_a - min_b) / (min_b + 1e-9) * 100
        _st = _start_status(pct, min_a >= min_b - 1e-6)

        if pred_before is not None and pred_after is not None:
            pk_before = int(np.argmin(pred_before))
            pk_after  = int(np.argmin(pred_after))
            pmin_b = pred_before[pk_before]
            pmin_a = pred_after[pk_after]
            pct_p  = (pmin_a - pmin_b) / (pmin_b + 1e-9) * 100
            if zh:
                _mm_content = (f"预测最差用户: UE{pk_before} {pmin_b:.2f} → "
                               f"UE{pk_after} {pmin_a:.2f} bits/s/Hz ({pct_p:+.1f}%)"
                               f"&emsp;真实最差用户: UE{k_before} {min_b:.2f} → "
                               f"UE{k_after} {min_a:.2f} bits/s/Hz ({pct:+.1f}%){_st}")
            else:
                _mm_content = (f"Predicted worst UE: UE{pk_before} {pmin_b:.2f} → "
                               f"UE{pk_after} {pmin_a:.2f} bits/s/Hz ({pct_p:+.1f}%)"
                               f"&emsp;True worst UE: UE{k_before} {min_b:.2f} → "
                               f"UE{k_after} {min_a:.2f} bits/s/Hz ({pct:+.1f}%){_st}")
        else:
            if zh:
                _mm_content = (f"最差用户: UE{k_before} {min_b:.2f} → "
                               f"UE{k_after} {min_a:.2f} bits/s/Hz ({pct:+.1f}%){_st}")
            else:
                _mm_content = (f"Worst UE: UE{k_before} {min_b:.2f} → "
                               f"UE{k_after} {min_a:.2f} bits/s/Hz ({pct:+.1f}%){_st}")
        if _has_orig:
            ok_b = int(np.argmin(orig_before))
            omin_b = float(orig_before[ok_b])
            pct_o = (min_a - omin_b) / (omin_b + 1e-9) * 100
            _ost = _global_status(min_a >= omin_b - 1e-6)
            _mm_content += (
                f"<br>相对原始分配: 真实最差用户 UE{ok_b} {omin_b:.2f} → "
                f"UE{k_after} {min_a:.2f} bits/s/Hz ({pct_o:+.1f}%){_ost}"
                if zh else
                f"<br>vs original allocation: true worst UE{ok_b} {omin_b:.2f} → "
                f"UE{k_after} {min_a:.2f} bits/s/Hz ({pct_o:+.1f}%){_ost}")
        _mm_label = "主要目标" if zh else "Primary Objective"
        lines.append("")
        lines.append(
            f'<details open><summary>{_mm_label}</summary>'
            f'<div style="background:#f0f2f6;padding:8px 12px;border-radius:6px;margin:6px 0">'
            f'{_mm_content}</div></details>'
        )

    # ── Secondary objective metrics ──────────────────────────────────────────
    sec_type = goal.get("secondary_type")
    if sec_type:
        _sec_lbl = "次要目标" if zh else "Secondary Objective"

        if sec_type == "max_sum_rate":
            sum_b = float(np.sum(before_rate))
            sum_a = float(np.sum(after_rate))
            pct_s = (sum_a - sum_b) / (sum_b + 1e-9) * 100
            _st = _global_status(sum_a >= sum_b - 1e-6)
            if pred_before is not None and pred_after is not None:
                ps_b = float(np.sum(pred_before)); ps_a = float(np.sum(pred_after))
                pct_p = (ps_a - ps_b) / (ps_b + 1e-9) * 100
                _sec_content = (f"{'预测Σrate' if zh else 'Pred Σrate'}: {ps_b:.2f} → {ps_a:.2f} ({pct_p:+.1f}%)"
                                f"&emsp;{'真实Σrate' if zh else 'True Σrate'}: {sum_b:.2f} → {sum_a:.2f} ({pct_s:+.1f}%){_st}")
            else:
                _sec_content = f"{'真实Σrate' if zh else 'True Σrate'}: {sum_b:.2f} → {sum_a:.2f} ({pct_s:+.1f}%){_st}"
        elif sec_type == "fairness":
            def _sl(r): return float(np.sum(np.log2(np.maximum(r, 1e-9))))
            sl_b = _sl(before_rate); sl_a = _sl(after_rate)
            pct_sl = (sl_a - sl_b) / (abs(sl_b) + 1e-9) * 100
            _st = _global_status(sl_a >= sl_b - 1e-6)
            if pred_before is not None and pred_after is not None:
                psl_b = _sl(pred_before); psl_a = _sl(pred_after)
                pct_psl = (psl_a - psl_b) / (abs(psl_b) + 1e-9) * 100
                _sec_content = (f"{'预测Σlog2(rate)' if zh else 'Pred Σlog2(rate)'}: {psl_b:.2f} → {psl_a:.2f} ({pct_psl:+.1f}%)"
                                f"&emsp;{'真实Σlog2(rate)' if zh else 'True Σlog2(rate)'}: {sl_b:.2f} → {sl_a:.2f} ({pct_sl:+.1f}%){_st}")
            else:
                _sec_content = f"{'真实Σlog2(rate)' if zh else 'True Σlog2(rate)'}: {sl_b:.2f} → {sl_a:.2f} ({pct_sl:+.1f}%){_st}"
        elif sec_type == "max_min":
            k_b = int(np.argmin(before_rate)); k_a = int(np.argmin(after_rate))
            mn_b = before_rate[k_b]; mn_a = after_rate[k_a]
            pct_mn = (mn_a - mn_b) / (abs(mn_b) + 1e-9) * 100
            _st = _global_status(mn_a >= mn_b - 1e-6)
            if pred_before is not None and pred_after is not None:
                pk_b = int(np.argmin(pred_before)); pk_a = int(np.argmin(pred_after))
                pmn_b = pred_before[pk_b]; pmn_a = pred_after[pk_a]
                pct_pmn = (pmn_a - pmn_b) / (abs(pmn_b) + 1e-9) * 100
                _sec_content = (f"{'预测最差用户' if zh else 'Pred worst UE'}: UE{pk_b} {pmn_b:.2f} → UE{pk_a} {pmn_a:.2f} ({pct_pmn:+.1f}%)"
                                f"&emsp;{'真实最差用户' if zh else 'True worst UE'}: UE{k_b} {mn_b:.2f} → UE{k_a} {mn_a:.2f} ({pct_mn:+.1f}%){_st}")
            else:
                _sec_content = (f"{'最差用户' if zh else 'Worst UE'}: UE{k_b} {mn_b:.2f} → UE{k_a} {mn_a:.2f} ({pct_mn:+.1f}%){_st}")
        else:
            _sec_content = None

        if _sec_content:
            lines.append("")
            lines.append(
                f'<details open><summary>{_sec_lbl}</summary>'
                f'<div style="background:#f0f2f6;padding:8px 12px;border-radius:6px;margin:6px 0">'
                f'{_sec_content}</div></details>')

    # ── Per-UE sections (from ue_multipliers or legacy target/protected/others) ──
    _um = goal.get("ue_multipliers", {})
    _c_floor = goal.get("constraint_floor", 0.0)
    _uaf = goal.get("ue_abs_floor") or {}

    # Build per-UE floor for status check
    # Ensure ue_multipliers includes all target_ues (LLM may omit some)
    if _um and targets:
        _mult_default = goal.get("target_multiplier")
        for k in targets:
            if k < K and k not in _um:
                _um[k] = _mult_default if _mult_default else 1.0

    # "Bare" target UEs — user said "提升 UE1" without a specific %. We
    # still have an internal multiplier (for the optimiser) but the UI
    # should NOT show a concrete met/not-met status, because the user
    # didn't commit to a specific number.
    _bare_targets_set = set(goal.get("bare_targets") or [])

    _ue_floor = {}
    if _um:
        for k, m in _um.items():
            if k < K:
                if k in _bare_targets_set:
                    _ue_floor[k] = None  # suppress ✔/❌ status for bare targets
                else:
                    _ue_floor[k] = float(before_rate[k]) * m
        _all_boost_ues = sorted(k for k in _um if k < K)
    else:
        _mult = goal.get("target_multiplier")
        _p_floor = goal.get("protected_floor", 1.0)
        protected = set(goal.get("protected_ues") or [])
        _target_list = sorted(targets)
        _protect_list = sorted(k for k in protected if k < K and k not in targets)
        for k in _target_list:
            _ue_floor[k] = float(before_rate[k]) * _mult if _mult else float(before_rate[k])
        for k in _protect_list:
            _ue_floor[k] = float(before_rate[k]) * _p_floor
        _all_boost_ues = _target_list + _protect_list

    # Per-UE absolute floor UEs (not in boost groups)
    _uaf_ues = sorted(k for k in _uaf if k < K and k not in set(_all_boost_ues))
    for k in _uaf_ues:
        _ue_floor[k] = _uaf[k]  # absolute floor, not ratio

    # Others: non-boost, non-abs-floor UEs with constraint_floor > 0.
    # NOTE: do NOT require _covered to be non-empty. For a global max_min /
    # max_sum objective with no named UE group, _covered is empty but the
    # "其他用户不降低" constraint still applies to everyone. For max_min the
    # implicit primary is the current worst UE — exclude it so "others"
    # literally means "everyone except the one being maximised".
    _covered = set(_all_boost_ues) | set(_uaf_ues)
    if not _covered and g_type == "max_min":
        _covered = {int(np.argmin(before_rate))}
    _others_list = sorted(k for k in range(K) if k not in _covered) if _c_floor > 0 else []
    for k in _others_list:
        _ue_floor[k] = float(before_rate[k]) * _c_floor

    _all_ues = _all_boost_ues + _uaf_ues + _others_list

    # Build rows for all UEs
    _all_rows = {}
    for k in _all_ues:
        pred_part = ""
        if pred_before is not None and pred_after is not None:
            pb = float(pred_before[k]); pa = float(pred_after[k])
            pp = (pa - pb) / (abs(pb) + 1e-9) * 100
            pred_part = f"{'预测速率' if zh else 'predicted rate'}: {pb:.2f} → {pa:.2f} bits/s/Hz ({pp:+.1f}%)"
        b = float(before_rate[k]); a = float(after_rate[k])
        pct = (a - b) / (abs(b) + 1e-9) * 100
        true_part = f"{'真实速率' if zh else 'true rate'}: {b:.2f} → {a:.2f} bits/s/Hz ({pct:+.1f}%)"
        floor_k = _ue_floor.get(k)
        if k in _bare_targets_set:
            # Bare target: user said "提升 UEk" without a concrete %.
            # Success criterion = any meaningful improvement over baseline.
            met = a > b + 1e-6
            if zh:
                status = ' <span style="color:green">✔ 已达成</span>' if met else " ❌ 未达成"
            else:
                status = ' <span style="color:green">✔ done</span>' if met else " ❌ not met"
        elif floor_k is not None and floor_k > 0:
            # Determine if this UE's target is a ceiling (decrease) or floor (increase)
            _k_mult = _um.get(k) if _um else (goal.get("target_multiplier") if k in targets else None)
            _is_ceiling = (_k_mult is not None and _k_mult < 1.0)
            met = (a <= floor_k + 1e-6) if _is_ceiling else (a >= floor_k - 1e-6)
            if zh:
                status = ' <span style="color:green">✔ 已达成</span>' if met else " ❌ 未达成"
            else:
                status = ' <span style="color:green">✔ done</span>' if met else " ❌ not met"
        else:
            status = ""
        _all_rows[k] = (pred_part, true_part, status)

    _max_left = max(
        (len(f"UE{k} {pred}") for k, (pred, _, _) in _all_rows.items()),
        default=0)

    def _fmt_ue(k):
        pred, true, status = _all_rows[k]
        ll = len(f"UE{k} {pred}")
        pad = "&ensp;" * (_max_left - ll + 2)
        return f"UE{k} {pred}{pad}{true}{status}  "

    def _collapsible(label, ue_list, open=False):
        tag = "<details open>" if open else "<details>"
        inner = "<br>".join(f"•&ensp;&ensp;{_fmt_ue(k)}" for k in ue_list)
        return (
            f'{tag}<summary>{label}</summary>'
            f'<div style="background:#f0f2f6;padding:8px 12px;border-radius:6px;margin:6px 0">'
            f'{inner}</div></details>'
        )


    # ── Per-UE / constraint sections ───────────────────────────────────────
    # ALL sections below share a single running counter `_section_idx` so
    # their labels (主要 / 次要 / 补充) stay perfectly in sync with
    # ``_goal_summary``'s parts-based counting. The global primary block
    # (sum_rate / fairness / max_min) above already occupies slot 0, so
    # _section_idx starts at 1 in that case.
    _lbl_primary   = "主要目标" if zh else "Primary Objective"
    _lbl_secondary = "次要目标" if zh else "Secondary Objective"
    _lbl_suppl     = "补充目标" if zh else "Supplementary Objective"
    _labels = [_lbl_primary, _lbl_secondary, _lbl_suppl]

    _section_idx = 1 if g_type in ("max_sum_rate", "fairness", "max_min") else 0

    def _next_label():
        return _labels[min(_section_idx, 2)]

    # Compute _g_targets / _g_protected / _g_rest once so the _mrf section
    # below can still consult them for other purposes if needed.
    _g_targets = _g_protected = _g_rest = []
    if _um:
        _g_targets   = sorted(k for k in (goal.get("target_ues") or [])
                              if k < K and k in _um)
        _g_protected = sorted(k for k in (goal.get("protected_ues") or [])
                              if k < K and k in _um and k not in set(_g_targets))
        _g_rest      = sorted(k for k in _um if k < K
                              and k not in set(_g_targets)
                              and k not in set(_g_protected))
        for _klist in [_g_targets, _g_protected, _g_rest]:
            if not _klist:
                continue
            lines.append("")
            lines.append(_collapsible(_next_label(), _klist,
                                      open=(_section_idx < 2)))
            _section_idx += 1
    else:
        # Legacy path: target / protected directly
        _target_list = sorted(targets)
        protected = set(goal.get("protected_ues") or [])
        _protect_list = sorted(k for k in protected if k < K and k not in targets)
        if _target_list:
            lines.append("")
            lines.append(_collapsible(_next_label(), _target_list, open=True))
            _section_idx += 1
        if _protect_list:
            lines.append("")
            lines.append(_collapsible(_next_label(), _protect_list, open=True))
            _section_idx += 1

    # Per-UE absolute floor section
    if _uaf_ues:
        lines.append("")
        lines.append(_collapsible(_next_label(), _uaf_ues, open=True))
        _section_idx += 1

    # Global min-rate floor check (worst UE in the system)
    _mrf = goal.get("min_rate_floor")
    if _mrf is not None and _mrf > 0:
        _lbl_mrf = _next_label()
        _section_idx += 1
        k_worst_a = int(np.argmin(after_rate))
        worst_a   = float(after_rate[k_worst_a])
        _met = worst_a >= _mrf
        if zh:
            _status = '<span style="color:green">✔ 已达成</span>' if _met else "❌ 未达成"
            _mrf_content = f"最差用户速率≥{_mrf}: UE{k_worst_a} = {worst_a:.2f} bits/s/Hz {_status}"
        else:
            _status = '<span style="color:green">✔ done</span>' if _met else "❌ not met"
            _mrf_content = f"Worst UE rate ≥{_mrf}: UE{k_worst_a} = {worst_a:.2f} bits/s/Hz {_status}"
        lines.append("")
        lines.append(
            f'<details open><summary>{_lbl_mrf}</summary>'
            f'<div style="background:#f0f2f6;padding:8px 12px;border-radius:6px;margin:6px 0">'
            f'{_mrf_content}</div></details>')

    # Others section (implicit "其他用户不降低 / 降低≤X%" constraint_floor).
    # Shown as an OPEN aggregate-status block (like the primary objective) so
    # the secondary goal's completion is visible at a glance, plus a nested
    # collapsible with the per-UE breakdown.
    if _others_list:
        _viol = [k for k in _others_list
                 if after_rate[k] < _ue_floor.get(k, 0.0) - 1e-6]
        _met = not _viol
        # worst relative change among "others" (for the headline number)
        _wk = min(_others_list,
                  key=lambda k: (after_rate[k] - before_rate[k]) / (abs(before_rate[k]) + 1e-9))
        _wb, _wa = float(before_rate[_wk]), float(after_rate[_wk])
        _wpct = (_wa - _wb) / (abs(_wb) + 1e-9) * 100
        if zh:
            _desc = ("其他用户不降低" if _c_floor >= 1.0
                     else f"其他用户降低≤{round((1 - _c_floor) * 100)}%")
            _ost = ('<span style="color:green">✔ 已达成</span>' if _met
                    else f'❌ 未达成 ({len(_viol)} 个用户跌破下限)')
            _ocontent = (f"{_desc}（共 {len(_others_list)} 个用户）&emsp;"
                         f"最差变化: UE{_wk} {_wb:.2f} → {_wa:.2f} bits/s/Hz "
                         f"({_wpct:+.1f}%) {_ost}")
        else:
            _desc = ("others no degradation" if _c_floor >= 1.0
                     else f"others drop ≤{round((1 - _c_floor) * 100)}%")
            _ost = ('<span style="color:green">✔ done</span>' if _met
                    else f'❌ not met ({len(_viol)} UEs below floor)')
            _ocontent = (f"{_desc} ({len(_others_list)} UEs)&emsp;"
                         f"worst change: UE{_wk} {_wb:.2f} → {_wa:.2f} bits/s/Hz "
                         f"({_wpct:+.1f}%) {_ost}")
        _detail = "<br>".join(f"•&ensp;&ensp;{_fmt_ue(k)}" for k in _others_list)
        _detail_lbl = "逐用户明细" if zh else "per-UE detail"
        lines.append("")
        lines.append(
            f'<details open><summary>{_next_label()}</summary>'
            f'<div style="background:#f0f2f6;padding:8px 12px;border-radius:6px;margin:6px 0">'
            f'{_ocontent}'
            f'<details style="margin-top:6px"><summary>{_detail_lbl}</summary>'
            f'{_detail}</details>'
            f'</div></details>')
        _section_idx += 1

    # Topology changes (link additions/removals) and power weight changes
    g_type = goal.get("type", "max_sum_rate")
    k_worst = int(np.argmin(before_rate)) if g_type == "max_min" else -1
    effective_targets = targets | ({k_worst} if k_worst >= 0 else set())

    link_lines  = []
    power_lines = []
    for l in range(ap_num):
        for k in range(K):
            was = A_before[l, k] > 0.5
            now = A_after[l, k] > 0.5
            b   = float(before_rate[k])
            a   = float(after_rate[k])
            pct = (a - b) / (b + 1e-9) * 100
            if not was and now:
                p_new = float(P_after[l, k])
                if k in effective_targets:
                    reason = f"新增连接 功率 {p_new:.2f}" if zh else f"new link power {p_new:.2f}"
                else:
                    reason = f"附带调整 功率 {p_new:.2f}" if zh else f"collateral change power {p_new:.2f}"
                link_lines.append(f"•  AP{l} → UE{k}: {reason}")
            elif was and not now:
                reason = "断开连接" if zh else "removed"
                link_lines.append(f"•  AP{l} → UE{k}: {reason}")
            elif was and now:
                wb = float(P_before[l, k])
                wa = float(P_after[l, k])
                if abs(wa - wb) > 0.01:
                    if zh:
                        line = f"•  AP{l} → UE{k} 功率: {wb:.2f} → {wa:.2f}"
                    else:
                        line = f"•  AP{l} → UE{k} power: {wb:.2f} → {wa:.2f}"
                    power_lines.append(line)

    if link_lines:
        lines.append("")
        topo_label = "拓扑变更" if zh else "Topology Changes"
        topo_inner = "<br>".join(link_lines)
        lines.append(
            f'<details><summary>{topo_label}</summary>'
            f'<div style="background:#f0f2f6;padding:8px 12px;border-radius:6px;margin:6px 0">'
            f'{topo_inner}</div></details>'
        )

    # ── Power changes (collapsible; auto-open when no topology changes) ──
    if power_lines:
        lines.append("")
        pw_label = "功率变更" if zh else "Power Changes"
        pw_inner = "<br>".join(power_lines)
        _open = ""
        lines.append(
            f'<details{_open}><summary>{pw_label}</summary>'
            f'<div style="background:#f0f2f6;padding:8px 12px;border-radius:6px;margin:6px 0">'
            f'{pw_inner}</div></details>'
        )

    return "\n".join(lines)
