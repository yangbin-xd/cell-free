# ============================
# 29-UE radar plots (Signal / Interference / Rate)
# UE angular order is SORTED by the metric value (per figure),
# but axis tick labels show ORIGINAL UE IDs (no re-numbering).
# Copy-paste and run.
# ============================

import numpy as np
from demo import digital_twin
from baseline.query import baseline

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
# 'Times New Roman' is absent on Katana; Nimbus Roman is URW's
# metric-compatible clone of it. Without this chain matplotlib falls back
# to DejaVu Sans and the figures come out in the wrong typeface entirely.
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Nimbus Roman',
                                     'DejaVu Serif']
font1, font2 = 20, 17

# ----------------------------
# Load data (same as your script)
# ----------------------------
test_idx = 0
snr = 15

# true and proposed
ue_num, true_signal, true_interf, true_rate, pred_signal, pred_interf, pred_rate = \
    digital_twin(snr, test_idx, False, False)

# query baseline
query_signal, query_interf, query_rate = baseline(test_idx, False, False)

# map baseline
map_signal = np.load('result/map/map_signal.npy')
map_interf = np.load('result/map/map_interf.npy')
map_rate   = np.load(f'result/map/map_rate_{snr}dB.npy')

map_signal = map_signal[test_idx]
map_signal = map_signal[map_signal != 0]
map_interf = map_interf[test_idx]
map_interf = map_interf[map_interf != 0]
map_rate   = map_rate[test_idx]
map_rate   = map_rate[map_rate != 0]

# beam baseline
beam_signal = np.load('result/beam/beam_signal.npy')
beam_interf = np.load('result/beam/beam_interf.npy')
beam_rate   = np.load(f'result/beam/beam_rate_{snr}dB.npy')

beam_signal = beam_signal[test_idx]
beam_signal = beam_signal[beam_signal != 0]
beam_interf = beam_interf[test_idx]
beam_interf = beam_interf[beam_interf != 0]
beam_rate   = beam_rate[test_idx]
beam_rate   = beam_rate[beam_rate != 0]

# ----------------------------
# Safety: force consistent UE length (use ue_num)
# ----------------------------
n = int(ue_num)

def _cut(x, n):
    x = np.asarray(x).reshape(-1)
    if len(x) < n:
        raise ValueError(f"Array length {len(x)} is smaller than ue_num={n}.")
    return x[:n]

true_signal = _cut(true_signal, n)
true_interf = _cut(true_interf, n)
true_rate   = _cut(true_rate, n)

pred_signal = _cut(pred_signal, n)
pred_interf = _cut(pred_interf, n)
pred_rate   = _cut(pred_rate, n)

query_signal = _cut(query_signal, n)
query_interf = _cut(query_interf, n)
query_rate   = _cut(query_rate, n)

map_signal = _cut(map_signal, n)
map_interf = _cut(map_interf, n)
map_rate   = _cut(map_rate, n)

beam_signal = _cut(beam_signal, n)
beam_interf = _cut(beam_interf, n)
beam_rate   = _cut(beam_rate, n)

# ----------------------------
# Radar plot helper
# ----------------------------
def radar_metric_ue_sorted(metric_dict, title, unit, rmin, rmax, outpath,
                           sort_by="True", ue_tick_step=4, legend_ncol=3):
    """
    metric_dict: dict of {name: array(n,)}
    Sorting: only changes the angular DISPLAY ORDER.
    Tick labels show ORIGINAL UE IDs (sort_idx[k]).
    """
    if sort_by not in metric_dict:
        raise KeyError(f"sort_by='{sort_by}' not found. Available: {list(metric_dict.keys())}")

    # sorting reference (for display order only)
    ref = np.asarray(metric_dict[sort_by]).reshape(-1)
    if len(ref) != n:
        raise ValueError(f"Reference length {len(ref)} != n={n}")

    sort_idx = np.argsort(ref)  # ascending
    metric_sorted = {k: np.asarray(v).reshape(-1)[sort_idx] for k, v in metric_dict.items()}

    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw=dict(polar=True))

    # plot lines
    for name, vals in metric_sorted.items():
        vals_closed = np.concatenate([vals, vals[:1]])
        ax.plot(angles_closed, vals_closed, linewidth=2.0, label=name)

    ax.set_title(title, fontsize=font1, pad=18)
    ax.set_ylim(rmin, rmax)

    # x tick labels are ORIGINAL UE IDs (not re-numbered)
    xticklabels = []
    for k in range(n):
        xticklabels.append(str(sort_idx[k]) if (k % ue_tick_step == 0) else "")
    ax.set_xticks(angles)
    ax.set_xticklabels(xticklabels, fontsize=font2)

    ax.tick_params(axis='y', labelsize=font2)
    ax.grid(True, which='both', ls=':', color='gray', alpha=0.3)

    # legend
    ax.legend(fontsize=12, ncol=legend_ncol, loc='upper center',
              bbox_to_anchor=(0.5, 1.12), frameon=False,
              columnspacing=1.0, handlelength=1.2, handletextpad=0.4)

    # unit note
    ax.text(0.02, 0.02, f"Unit: {unit}", transform=ax.transAxes, fontsize=12)

    plt.tight_layout()
    # plt.savefig(outpath)
    # plt.close(fig)

# ----------------------------
# Build dicts and plot 3 radars
# ----------------------------
signal_dict = {
    "Baseline 1 [17]": map_signal,
    "Baseline 2 [24]": query_signal,
    "Baseline 3 [25]": beam_signal,
    "Proposed": pred_signal,
    "True": true_signal,
}
interf_dict = {
    "Baseline 1 [17]": map_interf,
    "Baseline 2 [24]": query_interf,
    "Baseline 3 [25]": beam_interf,
    "Proposed": pred_interf,
    "True": true_interf,
}
rate_dict = {
    "Baseline 1 [17]": map_rate,
    "Baseline 2 [24]": query_rate,
    "Baseline 3 [25]": beam_rate,
    "Proposed": pred_rate,
    "True": true_rate,
}

# Use the TRUE curve to determine the sorting order (recommended)
radar_metric_ue_sorted(signal_dict,
                       title="Signal Power over 29 UEs (Value-sorted, IDs preserved)",
                       unit="dBW", rmin=-100, rmax=-75,
                       outpath="result/radar_signal_sorted.pdf",
                       sort_by="True", ue_tick_step=4, legend_ncol=3)

radar_metric_ue_sorted(interf_dict,
                       title="Interference Power over 29 UEs (Value-sorted, IDs preserved)",
                       unit="dBW", rmin=-94, rmax=-78,
                       outpath="result/radar_interf_sorted.pdf",
                       sort_by="True", ue_tick_step=4, legend_ncol=3)

radar_metric_ue_sorted(rate_dict,
                       title="SE over 29 UEs (Value-sorted, IDs preserved)",
                       unit="bits/s/Hz", rmin=0.0, rmax=4.0,
                       outpath="result/radar_rate_sorted.pdf",
                       sort_by="True", ue_tick_step=4, legend_ncol=3)

plt.show()

# print("Saved:")
# print("  result/radar_signal_sorted.pdf")
# print("  result/radar_interf_sorted.pdf")
# print("  result/radar_rate_sorted.pdf")
