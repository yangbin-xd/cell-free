
import numpy as np
from generate import num_test

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'
font1, font2 = 27, 23

SNR = np.arange(0, 31, 5)
pred_rate_error = np.empty([SNR.size, num_test])
query_rate_error = np.empty([SNR.size, num_test])
map_rate_error = np.empty([SNR.size, num_test])
beam_rate_error = np.empty([SNR.size, num_test])

pred_rate_sum_error = np.empty([SNR.size, num_test])
query_rate_sum_error = np.empty([SNR.size, num_test])
map_rate_sum_error = np.empty([SNR.size, num_test])
beam_rate_sum_error = np.empty([SNR.size, num_test])

# load data
for i, snr in enumerate(SNR):

    pred_rate_error[i,:] = np.load(f'result/pred/pred_rate_error_{snr}dB.npy')
    query_rate_error[i,:] = np.load(f'result/query/query_rate_error_{snr}dB.npy')
    map_rate_error[i,:] = np.load(f'result/map/map_rate_error_{snr}dB.npy')
    beam_rate_error[i,:] = np.load(f'result/beam/beam_rate_error_{snr}dB.npy')

    true_rate_sum = np.load(f'result/true_rate_sum_{snr}dB.npy')
    pred_rate_sum = np.load(f'result/pred/pred_rate_sum_{snr}dB.npy')
    query_rate_sum = np.load(f'result/query/query_rate_sum_{snr}dB.npy')
    map_rate_sum = np.load(f'result/map/map_rate_sum_{snr}dB.npy')
    beam_rate_sum = np.load(f'result/beam/beam_rate_sum_{snr}dB.npy')

    pred_rate_sum_error[i,:] = np.abs(pred_rate_sum - true_rate_sum)
    query_rate_sum_error[i,:] = np.abs(query_rate_sum - true_rate_sum)
    map_rate_sum_error[i,:] = np.abs(map_rate_sum - true_rate_sum)
    beam_rate_sum_error[i,:] = np.abs(beam_rate_sum - true_rate_sum)

# pred_rate_error_mean = np.mean(pred_rate_error, -1)
# query_rate_error_mean = np.mean(query_rate_error, -1)
# map_rate_error_mean = np.mean(map_rate_error, -1)
# beam_rate_error_mean = np.mean(beam_rate_error, -1)

# pred_rate_sum_error_mean = np.mean(pred_rate_sum_error, -1)
# query_rate_sum_error_mean = np.mean(query_rate_sum_error, -1)
# map_rate_sum_error_mean = np.mean(map_rate_sum_error, -1)
# beam_rate_sum_error_mean = np.mean(beam_rate_sum_error, -1)

# # fig 1
# plt.figure(figsize=(6.5, 5.0))
# plt.xlabel("SNR (dB)", fontsize=font1)
# plt.ylabel("Rate error (bits/s/Hz)", fontsize=font1)

# plt.plot(SNR, map_rate_error_mean, marker='o', ms=7, lw=2, c='#F65314', label='Baseline 1 [24]')
# plt.plot(SNR, query_rate_error_mean, marker='s', ms=7, lw=2, c='#FFBB00', label='Baseline 2 [25]')
# plt.plot(SNR, beam_rate_error_mean, marker='^', ms=7, lw=2, c='#7CBB00', label='Baseline 3 [26]')
# plt.plot(SNR, pred_rate_error_mean, marker='d', ms=7, lw=2, c='#00A1F1', label='Prpoposed')

# plt.ylim([0,1.35])
# plt.xticks(fontsize=font1)
# plt.yticks(fontsize=font1)
# plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)
# plt.legend(fontsize=font2, ncol=2, columnspacing=0.3, handletextpad=0.3, handlelength=0.7, labelspacing=0.3, loc='upper left')
# plt.tight_layout()
# plt.savefig('result/rate_snr.pdf')


# # fig 2
# plt.figure(figsize=(6.5, 5.0))
# plt.xlabel("SNR (dB)", fontsize=font1)
# plt.ylabel("Rate error (bits/s/Hz)", fontsize=font1)

# plt.plot(SNR, map_rate_sum_error_mean, marker='o', ms=7, lw=2, c='#F65314', label='Baseline 1 [24]')
# plt.plot(SNR, query_rate_sum_error_mean, marker='s', ms=7, lw=2, c='#FFBB00', label='Baseline 2 [25]')
# plt.plot(SNR, beam_rate_sum_error_mean, marker='^', ms=7, lw=2, c='#7CBB00', label='Baseline 3 [26]')
# plt.plot(SNR, pred_rate_sum_error_mean, marker='d', ms=7, lw=2, c='#00A1F1', label='Prpoposed')

# plt.ylim([0,17])
# plt.xticks(fontsize=font1)
# plt.yticks(fontsize=font1)
# plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)
# plt.legend(fontsize=font2, ncol=2, columnspacing=0.3, handletextpad=0.3, handlelength=0.7, labelspacing=0.3, loc='upper left')
# plt.tight_layout()
# plt.savefig('result/sumrate_snr.pdf')
# plt.show()


def plot_mean_band(x, map_err, query_err, beam_err, pred_err, term, min, max,
        labels=("Baseline 1 [17]", "Baseline 2 [24]", "Baseline 3 [25]", "Proposed"),
        color1="#F65314", color2="#FFBB00", color3="#7CBB00", color4="#00A1F1",
        marker1='o', marker2='s', marker3='^', marker4='d'):

    x = np.asarray(x, dtype=float)

    series = [(map_err,  labels[0], color1, marker1),
              (query_err,labels[1], color2, marker2),
              (beam_err, labels[2], color3, marker3),
              (pred_err, labels[3], color4, marker4)]

    def _stats_from_matrix(Y, lo=0, hi=100):
        Y = np.asarray(Y)
        mean = np.nanmean(Y, axis=1)
        lo_v = np.nanpercentile(Y, lo, axis=1)
        hi_v = np.nanpercentile(Y, hi, axis=1)
        return mean, lo_v, hi_v

    plt.figure(figsize=(6.5, 5.0))
    for Y, lab, col, mk in series:
        mean, lo, hi = _stats_from_matrix(Y, lo=0, hi=100)
        line, = plt.plot(x, mean, ms=7, lw=2, marker=mk, label=lab, color=col)
        plt.fill_between(x, lo, hi, facecolor=to_rgba(col, 0.1), edgecolor='none')

    plt.xlabel("SNR (dB)", fontsize=font1)
    plt.ylabel(term, fontsize=font1-7)
    plt.ylim([min, max])
    plt.xticks(fontsize=font1)
    plt.yticks(fontsize=font1)
    plt.grid(True, which='both', ls=':', alpha=0.3)
    plt.legend(fontsize=font2, ncol=1, handletextpad=0.5, handlelength=1.2, labelspacing=0.3, loc='upper left')
    plt.tight_layout()


# plot results
plot_mean_band(SNR, map_rate_error, query_rate_error, beam_rate_error, pred_rate_error,
                "SE prediction error (bits/s/Hz)", 0, 2.5)
plt.savefig('result/rate_snr.pdf')
plot_mean_band(SNR, map_rate_sum_error, query_rate_sum_error, beam_rate_sum_error,
               pred_rate_sum_error, "Sum SE prediction error (bits/s/Hz)", 0, 30)
plt.savefig('result/sumrate_snr.pdf')
plt.show()
