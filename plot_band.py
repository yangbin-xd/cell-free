
import numpy as np
from tqdm import tqdm
from generate import num_test
from demo import digital_twin

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
matplotlib.rcParams['mathtext.fontset'] = 'cm'
# 'Times New Roman' is absent on Katana; Nimbus Roman is URW's
# metric-compatible clone of it. Without this chain matplotlib falls back
# to DejaVu Sans and the figures come out in the wrong typeface entirely.
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Nimbus Roman',
                                     'DejaVu Serif']
font1, font2 = 24, 20 # 27, 23

# load data
snr = 15

query_signal = np.load('result/query/query_signal.npy')
query_interf = np.load('result/query/query_interf.npy')
query_rate = np.load(f'result/query/query_rate_{snr}dB.npy')

map_signal = np.load('result/map/map_signal.npy')
map_interf = np.load('result/map/map_interf.npy')
map_rate = np.load(f'result/map/map_rate_{snr}dB.npy')

beam_signal = np.load('result/beam/beam_signal.npy')
beam_interf = np.load('result/beam/beam_interf.npy')
beam_rate = np.load(f'result/beam/beam_rate_{snr}dB.npy')

# calculate mean rate
ue_num = np.empty(num_test)
true_signal = np.empty(num_test)
true_interf = np.empty(num_test)
true_rate = np.empty(num_test)

pred_signal_error, query_signal_error = np.empty(num_test), np.empty(num_test)
pred_interf_error, query_interf_error = np.empty(num_test), np.empty(num_test)
pred_rate_error, query_rate_error = np.empty(num_test), np.empty(num_test)

map_signal_error, beam_signal_error = np.empty(num_test), np.empty(num_test)
map_interf_error, beam_interf_error = np.empty(num_test), np.empty(num_test)
map_rate_error, beam_rate_error = np.empty(num_test), np.empty(num_test)

true_rate_sum, pred_rate_sum, query_rate_sum, map_rate_sum, beam_rate_sum = np.empty(num_test),\
    np.empty(num_test), np.empty(num_test), np.empty(num_test), np.empty(num_test)

# # iteration
# iterable = range(num_test)
# iterable = tqdm(iterable, desc="Testing samples", unit="sample")

# # generate data
# for i in iterable:
#     # true and proposed
#     ue_num[i], true_signal_i, true_interf_i, true_rate_i, pred_signal_i,\
#         pred_interf_i, pred_rate_i = digital_twin(snr, i, False, False)
    
#     true_signal[i], true_interf[i], true_rate[i] = np.mean(true_signal_i),\
#         np.mean(true_interf_i), np.mean(true_rate_i)
    
#     # query
#     query_signal_i, query_interf_i, query_rate_i = query_signal[i][query_signal[i] != 0],\
#         query_interf[i][query_interf[i] != 0], query_rate[i][query_rate[i] != 0],
    
#     # map
#     map_signal_i, map_interf_i, map_rate_i = map_signal[i][map_signal[i] != 0],\
#         map_interf[i][map_interf[i] != 0], map_rate[i][map_rate[i] != 0]
    
#     # beam
#     beam_signal_i, beam_interf_i, beam_rate_i = beam_signal[i][beam_signal[i] != 0],\
#         beam_interf[i][beam_interf[i] != 0], beam_rate[i][beam_rate[i] != 0]

#     pred_signal_error[i] = np.mean(np.abs(true_signal_i - pred_signal_i))
#     pred_interf_error[i] = np.mean(np.abs(true_interf_i - pred_interf_i))
#     pred_rate_error[i] = np.mean(np.abs(true_rate_i - pred_rate_i))
    
#     query_signal_error[i] = np.mean(np.abs(true_signal_i - query_signal_i))
#     query_interf_error[i] = np.mean(np.abs(true_interf_i - query_interf_i))
#     query_rate_error[i] = np.mean(np.abs(true_rate_i - query_rate_i))

#     map_signal_error[i] = np.mean(np.abs(true_signal_i - map_signal_i))
#     map_interf_error[i] = np.mean(np.abs(true_interf_i - map_interf_i))
#     map_rate_error[i] = np.mean(np.abs(true_rate_i - map_rate_i))

#     beam_signal_error[i] = np.mean(np.abs(true_signal_i - beam_signal_i))
#     beam_interf_error[i] = np.mean(np.abs(true_interf_i - beam_interf_i))
#     beam_rate_error[i] = np.mean(np.abs(true_rate_i - beam_rate_i))

#     true_rate_sum[i], pred_rate_sum[i], query_rate_sum[i], map_rate_sum[i], beam_rate_sum[i] = \
#         np.sum(true_rate_i), np.sum(pred_rate_i), np.sum(query_rate_i), np.sum(map_rate_i), np.sum(beam_rate_i)
    
# #save data
# np.save('result/ue_num.npy', ue_num)
# np.save('result/true_signal.npy', true_signal)
# np.save('result/true_interf.npy', true_interf)
# np.save('result/true_rate.npy', true_rate)

# np.save('result/pred/pred_signal_error.npy', pred_signal_error)
# np.save('result/pred/pred_interf_error.npy', pred_interf_error)
# np.save(f'result/pred/pred_rate_error_{snr}dB.npy', pred_rate_error)

# np.save('result/query/query_signal_error.npy', query_signal_error)
# np.save('result/query/query_interf_error.npy', query_interf_error)
# np.save(f'result/query/query_rate_error_{snr}dB.npy', query_rate_error)

# np.save('result/map/map_signal_error.npy', map_signal_error)
# np.save('result/map/map_interf_error.npy', map_interf_error)
# np.save(f'result/map/map_rate_error_{snr}dB.npy', map_rate_error)

# np.save('result/beam/beam_signal_error.npy', beam_signal_error)
# np.save('result/beam/beam_interf_error.npy', beam_interf_error)
# np.save(f'result/beam/beam_rate_error_{snr}dB.npy', beam_rate_error)

# np.save(f'result/true_rate_sum_{snr}dB.npy', true_rate_sum)
# np.save(f'result/pred/pred_rate_sum_{snr}dB.npy', pred_rate_sum)
# np.save(f'result/query/query_rate_sum_{snr}dB.npy', query_rate_sum)
# np.save(f'result/map/map_rate_sum_{snr}dB.npy', map_rate_sum)
# np.save(f'result/beam/beam_rate_sum_{snr}dB.npy', beam_rate_sum)


# load data
ue_num = np.load('result/ue_num.npy')
true_signal = np.load('result/true_signal.npy')
true_interf = np.load('result/true_interf.npy')
true_rate = np.load('result/true_rate.npy')

# Proposed = the ablation campaign's seed-0 full model (no epoch cap,
# early stopping only; matches plot_pdf.py and make_ablation_table.py).
# Signal/interf per-realization arrays come from ablation/make_band_arrays.py,
# rate error and sum-rate from ablation/snr_eval.py. The archived flat arrays
# one directory up are a different, unseeded run and are left untouched.
PROPOSED = 'result/pred/main_seed0'

pred_signal_error = np.load(f'{PROPOSED}/pred_signal_error.npy')
pred_interf_error = np.load(f'{PROPOSED}/pred_interf_error.npy')
pred_rate_error = np.load(f'{PROPOSED}/pred_rate_error_{snr}dB.npy')

query_signal_error = np.load('result/query/query_signal_error.npy')
query_interf_error = np.load('result/query/query_interf_error.npy')
query_rate_error = np.load(f'result/query/query_rate_error_{snr}dB.npy')

map_signal_error = np.load('result/map/map_signal_error.npy')
map_interf_error = np.load('result/map/map_interf_error.npy')
map_rate_error = np.load(f'result/map/map_rate_error_{snr}dB.npy')

beam_signal_error = np.load('result/beam/beam_signal_error.npy')
beam_interf_error = np.load('result/beam/beam_interf_error.npy')
beam_rate_error =   np.load(f'result/beam/beam_rate_error_{snr}dB.npy')

ngo_signal_error = np.load('result/closed/closed_pl_signal_error.npy')
ngo_interf_error = np.load('result/closed/closed_pl_interf_error.npy')
ngo_rate_error = np.load(f'result/closed/closed_pl_rate_error_{snr}dB.npy')

true_rate_sum = np.load(f'result/true_rate_sum_{snr}dB.npy')
pred_rate_sum = np.load(f'{PROPOSED}/pred_rate_sum_{snr}dB.npy')
query_rate_sum = np.load(f'result/query/query_rate_sum_{snr}dB.npy')
map_rate_sum = np.load(f'result/map/map_rate_sum_{snr}dB.npy')
beam_rate_sum = np.load(f'result/beam/beam_rate_sum_{snr}dB.npy')
ngo_rate_sum = np.load(f'result/closed/closed_pl_rate_sum_{snr}dB.npy')

pred_rate_sum_error = np.abs(pred_rate_sum - true_rate_sum)
query_rate_sum_error = np.abs(query_rate_sum - true_rate_sum)
map_rate_sum_error = np.abs(map_rate_sum - true_rate_sum)
beam_rate_sum_error = np.abs(beam_rate_sum - true_rate_sum)
ngo_rate_sum_error = np.abs(ngo_rate_sum - true_rate_sum)

def min_max_mean(signal_error, interf_error, rate_error):
    signal_min, signal_max, signal_mean = np.min(signal_error), np.max(signal_error),\
        np.mean(signal_error)
    interf_min, interf_max, interf_mean = np.min(interf_error), np.max(interf_error),\
        np.mean(interf_error)
    rate_min, rate_max, rate_mean = np.min(rate_error), np.max(rate_error),\
        np.mean(rate_error)
    return signal_min, signal_max, signal_mean, interf_min, interf_max, interf_mean,\
           rate_min, rate_max, rate_mean

pred_min_signal_error, pred_max_signal_error, pred_mean_signal_error,\
    pred_min_interf_error, pred_max_interf_error, pred_mean_interf_error,\
        pred_min_rate_error, pred_max_rate_error, pred_mean_rate_error = \
            min_max_mean(pred_signal_error, pred_interf_error, pred_rate_error)

query_min_signal_error, query_max_signal_error, query_mean_signal_error,\
    query_min_interf_error, query_max_interf_error, query_mean_interf_error,\
        query_min_rate_error, query_max_rate_error, query_mean_rate_error = \
            min_max_mean(query_signal_error, query_interf_error, query_rate_error)

map_min_signal_error, map_max_signal_error, map_mean_signal_error,\
    map_min_interf_error, map_max_interf_error, map_mean_interf_error,\
        map_min_rate_error, map_max_rate_error, map_mean_rate_error = \
            min_max_mean(map_signal_error, map_interf_error, map_rate_error)

beam_min_signal_error, beam_max_signal_error, beam_mean_signal_error,\
    beam_min_interf_error, beam_max_interf_error, beam_mean_interf_error,\
        beam_min_rate_error, beam_max_rate_error, beam_mean_rate_error = \
            min_max_mean(beam_signal_error, beam_interf_error, beam_rate_error)

# print("-" * 30)
# print("min error:")
# print("-" * 30)
# print(f"{pred_min_signal_error:.3f}, {query_min_signal_error:.3f}, {map_min_signal_error:.3f}, {beam_min_signal_error:.3f} dB")
# print(f"{pred_min_interf_error:.3f}, {query_min_interf_error:.3f}, {map_min_interf_error:.3f}, {beam_min_interf_error:.3f} dB")
# print(f"{pred_min_rate_error:.3f}, {query_min_rate_error:.3f}, {map_min_rate_error:.3f}, {beam_min_rate_error:.3f} bits/s/Hz")

# print("-" * 30)
# print("max error:")
# print("-" * 30)
# print(f"{pred_max_signal_error:.3f}, {query_max_signal_error:.3f}, {map_max_signal_error:.3f}, {beam_max_signal_error:.3f} dB")
# print(f"{pred_max_interf_error:.3f}, {query_max_interf_error:.3f}, {map_max_interf_error:.3f}, {beam_max_interf_error:.3f} dB")
# print(f"{pred_max_rate_error:.3f}, {query_max_rate_error:.3f}, {map_max_rate_error:.3f}, {beam_max_rate_error:.3f} bits/s/Hz")

# print("-" * 30)
# print("mean error:")
# print("-" * 30)
# print(f"{pred_mean_signal_error:.3f}, {query_mean_signal_error:.3f}, {map_mean_signal_error:.3f}, {beam_mean_signal_error:.3f} dB")
# print(f"{pred_mean_interf_error:.3f}, {query_mean_interf_error:.3f}, {map_mean_interf_error:.3f}, {beam_mean_interf_error:.3f} dB")
# print(f"{pred_mean_rate_error:.3f}, {query_mean_rate_error:.3f}, {map_mean_rate_error:.3f}, {beam_mean_rate_error:.3f} bits/s/Hz")


def plot_mean_band(x, y, term, color):

    x, y, xs = np.asarray(x), np.asarray(y), np.unique(x)
    means, los, his, counts = [], [], [], []

    for xv in xs:
        vals = y[x == xv]
        m = np.mean(vals)
        s = np.std(vals, ddof=1) if vals.size > 1 else 0.0
        lo, hi = m - s, m + s
        lo, hi = np.percentile(vals, [0, 100])
        means.append(m); los.append(lo); his.append(hi); counts.append(vals.size)

    xs, means, los, his = xs.astype(float), np.array(means), np.array(los), np.array(his)

    # plot
    plt.figure(figsize=(6.5, 5))
    plt.plot(xs, means, marker='o', color=color)
    plt.fill_between(xs, los, his, color=to_rgba(color, 0.18), edgecolor='none')
    plt.xlabel("UE number", fontsize=font1)
    plt.ylabel(f"{term}", fontsize=font1)
    plt.xticks(fontsize=font1)
    plt.yticks(fontsize=font1)
    plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)
    plt.tight_layout()

# # plot mean band
# plot_mean_band(ue_num, true_signal, "true signal (dB)", '#F65314')
# plot_mean_band(ue_num, true_interf, "true interf (dB)", '#FFBB00')
# plot_mean_band(ue_num, true_rate, "true rate (bits/s/Hz)", '#7CBB00')
# plot_mean_band(ue_num, true_rate_sum, "true rate sum (bits/s/Hz)", '#00A1F1')
# plot_mean_band(ue_num, beam_rate_sum, "map rate sum (bits/s/Hz)", '#68217A')
# plt.show()

def plot_mean_band3(x, y1, y2, y3, y4, y5, term, min, max, label1="Baseline 2 [19]", label2="Baseline 3 [26]", label3="Baseline 4 [27]", label4="Proposed",
                    label5="Baseline 1 [2]",
                    color1="#FFBB00", color2="#7CBB00", color3="#00A1F1", color4="#68217A",
                    color5="#F65314"):
    x, y1, y2, y3, y4, y5 = np.asarray(x), np.asarray(y1), np.asarray(y2), np.asarray(y3), np.asarray(y4), np.asarray(y5)

    m1 = np.isfinite(x) & np.isfinite(y1)
    m2 = np.isfinite(x) & np.isfinite(y2)
    m3 = np.isfinite(x) & np.isfinite(y3)
    m4 = np.isfinite(x) & np.isfinite(y4)
    m5 = np.isfinite(x) & np.isfinite(y5)

    xs = np.union1d(np.union1d(np.unique(x[m1]), np.unique(x[m2])), np.unique(x[m3]))

    def stats_by_x(xv_all, y_all, xs_keys):
        means, los, highs = [], [], []
        for xv in xs_keys:
            vals = y_all[xv_all == xv]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                means.append(np.nan); los.append(np.nan); highs.append(np.nan)
            else:
                m = np.mean(vals)
                lo, hi = np.percentile(vals, [0, 100])
                means.append(m); los.append(lo); highs.append(hi)
        return (np.asarray(means, float), np.asarray(los, float), np.asarray(highs, float))

    m1s, l1s, h1s = stats_by_x(x[m1], y1[m1], xs)
    m2s, l2s, h2s = stats_by_x(x[m2], y2[m2], xs)
    m3s, l3s, h3s = stats_by_x(x[m3], y3[m3], xs)
    m4s, l4s, h4s = stats_by_x(x[m4], y4[m4], xs)
    m5s, l5s, h5s = stats_by_x(x[m5], y5[m5], xs)

    plt.figure(figsize=(6.5, 5))
    plt.plot(xs, m5s, marker='o', ms=7, lw=2, color=color5, label=label5,
             markerfacecolor='none', markeredgecolor=color5, markeredgewidth=1.5)
    plt.fill_between(xs, l5s, h5s, color=to_rgba(color5, 0.1), edgecolor='none')
    plt.plot(xs, m1s, marker='s', ms=7, lw=2, color=color1, label=label1,
             markerfacecolor='none', markeredgecolor=color1, markeredgewidth=1.5)
    plt.fill_between(xs, l1s, h1s, color=to_rgba(color1, 0.1), edgecolor='none')
    plt.plot(xs, m2s, marker='^', ms=7, lw=2, color=color2, label=label2,
             markerfacecolor='none', markeredgecolor=color2, markeredgewidth=1.5)
    plt.fill_between(xs, l2s, h2s, color=to_rgba(color2, 0.1), edgecolor='none')
    plt.plot(xs, m3s, marker='d', ms=7, lw=2, color=color3, label=label3,
             markerfacecolor='none', markeredgecolor=color3, markeredgewidth=1.5)
    plt.fill_between(xs, l3s, h3s, color=to_rgba(color3, 0.1), edgecolor='none')
    plt.plot(xs, m4s, marker='p', ms=8, lw=2, color=color4, label=label4,
             markerfacecolor='none', markeredgecolor=color4, markeredgewidth=1.5)
    plt.fill_between(xs, l4s, h4s, color=to_rgba(color4, 0.1), edgecolor='none')

    plt.xlabel("UE number", fontsize=font1)
    plt.ylabel(f"{term}", fontsize=font1-3) # font1-7
    plt.ylim([min, max])
    plt.xticks(fontsize=font1)
    plt.yticks(fontsize=font1)
    plt.gca().set_xticks(np.arange(10, 31, 2))
    plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)
    plt.legend(fontsize=font2, ncol=1, handletextpad=0.3, handlelength=1.2, labelspacing=0.3)
    plt.tight_layout()

# plot mean band
plot_mean_band3(ue_num, map_signal_error, query_signal_error, beam_signal_error, pred_signal_error,
                ngo_signal_error, "Signal power prediction error (dB)", 0, 8)
plt.savefig('result/signal_band.pdf')
plot_mean_band3(ue_num, map_interf_error, query_interf_error, beam_interf_error, pred_interf_error,
                ngo_interf_error, "Interf. power prediction error (dB)", 0, 10)
plt.savefig('result/interf_band.pdf')
plot_mean_band3(ue_num, map_rate_error, query_rate_error, beam_rate_error, pred_rate_error,
                ngo_rate_error, "SE prediction error (bits/s/Hz)", 0, 2.5)
plt.savefig('result/rate_band.pdf')
plot_mean_band3(ue_num, map_rate_sum_error, query_rate_sum_error, beam_rate_sum_error, pred_rate_sum_error,
                ngo_rate_sum_error, "Sum SE prediction error (bits/s/Hz)", 0, 25)
plt.savefig('result/sumrate_band.pdf')
plt.show()

