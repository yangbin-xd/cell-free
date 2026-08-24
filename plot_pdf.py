
import numpy as np
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

# read data
snr = 15

# Proposed = the ablation campaign's seed-0 full model (no epoch cap,
# early stopping only; see ablation/make_ablation_table.py).
# The archived flat arrays (pred_signal_mae.npy etc.) are a different, unseeded
# run and are left in place untouched.
PROPOSED = 'main_seed0'

pred_signal_mae = np.load(f'result/pred/pred_signal_{PROPOSED}_mae.npy')
pred_interf_mae = np.load(f'result/pred/pred_interf_{PROPOSED}_mae.npy')
pred_rate_mae = np.load(f'result/pred/pred_rate_{PROPOSED}_mae.npy')
pred_rate_finetune = np.load(f'result/pred/pred_rate_{PROPOSED}_finetune.npy')

map_signal_mae = np.load('result/map/map_signal_mae.npy')
map_interf_mae = np.load('result/map/map_interf_mae.npy')
map_rate_mae = np.load(f'result/map/map_rate_mae_{snr}dB.npy')

query_signal_mae = np.load('result/query/query_signal_mae.npy')
query_interf_mae = np.load('result/query/query_interf_mae.npy')
query_rate_mae = np.load(f'result/query/query_rate_mae_{snr}dB.npy')

beam_signal_mae = np.load('result/beam/beam_signal_mae.npy')
beam_interf_mae = np.load('result/beam/beam_interf_mae.npy')
beam_rate_mae = np.load(f'result/beam/beam_rate_mae_{snr}dB.npy')

ngo_signal_mae = np.load('result/closed/closed_pl_signal_mae.npy')
ngo_interf_mae = np.load('result/closed/closed_pl_interf_mae.npy')
ngo_rate_mae = np.load(f'result/closed/closed_pl_rate_mae_{snr}dB.npy')


# print(np.mean(np.abs(pred_signal_mae)))
# print(np.mean(np.abs(map_signal_mae)))
# print(np.mean(np.abs(query_signal_mae)))
# print(np.mean(np.abs(beam_signal_mae)))

# print(np.mean(np.abs(pred_interf_mae)))
# print(np.mean(np.abs(map_interf_mae)))
# print(np.mean(np.abs(query_interf_mae)))
# print(np.mean(np.abs(beam_interf_mae)))


# print(np.mean(np.abs(map_rate_mae)))
# print(np.mean(np.abs(query_rate_mae)))
# print(np.mean(np.abs(beam_rate_mae)))
# print(np.nanmean(np.abs(pred_rate_finetune)))


# plot pdf
def plot_distrib(pred, query, map, beam, ngo, name, unit, range=10):

    density, edges = np.histogram(pred, bins='fd', density=True)
    density1, edges1 = np.histogram(query, bins='fd', density=True)
    density2, edges2 = np.histogram(map, bins='fd', density=True)
    density3, edges3 = np.histogram(beam, bins='fd', density=True)
    density4, edges4 = np.histogram(ngo, bins='fd', density=True)

    centers = 0.5 * (edges[1:] + edges[:-1])
    centers1 = 0.5 * (edges1[1:] + edges1[:-1])
    centers2 = 0.5 * (edges2[1:] + edges2[:-1])
    centers3 = 0.5 * (edges3[1:] + edges3[:-1])
    centers4 = 0.5 * (edges4[1:] + edges4[:-1])

    win = np.exp(-0.5*((np.arange(-3,4))/1.0)**2)
    win = win / win.sum()

    density = np.convolve(density, win, mode='same')
    density1 = np.convolve(density1, win, mode='same')
    density2 = np.convolve(density2, win, mode='same')
    density3 = np.convolve(density3, win, mode='same')
    density4 = np.convolve(density4, win, mode='same')

    plt.figure(figsize=(6.5, 5))
    plt.plot(centers4, density4, lw=2, c='#F65314', label='Baseline 1 [2]') # closed-form
    plt.plot(centers2, density2, lw=2, c='#FFBB00', label='Baseline 2 [19]') # [15]
    plt.plot(centers1, density1, lw=2, c='#7CBB00', label='Baseline 3 [26]') # [22]
    plt.plot(centers3, density3, lw=2, c='#00A1F1', label='Baseline 4 [27]') # [23]
    plt.plot(centers, density, lw=2, c='#68217A', label='Proposed')

    # keep the y-scale set by the data-driven methods (the closed-form signal
    # error is a near-delta at 0 whose density spike would flatten the rest)
    ymax = 1.15 * max(density.max(), density1.max(), density2.max(), density3.max())
    plt.ylim([0, ymax])

    plt.xlabel(f'{name} error ({unit})', fontsize=font1)
    plt.ylabel('Probability density', fontsize=font1)
    plt.xticks(fontsize=font1)
    plt.yticks(fontsize=font1)
    plt.xlim([-range, range])
    plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)
    plt.legend(fontsize=font2, handletextpad=0.3, handlelength=1, labelspacing=0.3, loc='upper right')
    plt.tight_layout()

plot_distrib(pred_signal_mae, query_signal_mae, map_signal_mae, beam_signal_mae,
             ngo_signal_mae, "Signal power prediction", "dB", 12)
plt.savefig('result/signal_pdf.pdf')
plot_distrib(pred_interf_mae, query_interf_mae, map_interf_mae, beam_interf_mae,
             ngo_interf_mae, "Interference power prediction", "dB", 12)
plt.savefig('result/interf_pdf.pdf')
plot_distrib(pred_rate_finetune, query_rate_mae, map_rate_mae, beam_rate_mae,
             ngo_rate_mae, "SE prediction", "bits/s/Hz", 3.5)
plt.savefig('result/rate_pdf.pdf')
plt.show()
