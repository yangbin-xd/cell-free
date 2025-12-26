
# digital twin
import numpy as np
from demo import digital_twin
from query import baseline

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'
font1, font2 = 20, 17

test_idx = 0
snr = 15
# true and proposed
ue_num, true_signal, true_interf, true_rate, pred_signal, pred_interf, pred_rate\
    = digital_twin(snr, test_idx, False, False)
  
# query
query_signal, query_interf, query_rate = baseline(test_idx, False, False)
# query_interf = query_interf - 3.697

# map
map_signal = np.load('result/map/map_signal.npy')
map_interf = np.load('result/map/map_interf.npy')
map_rate = np.load(f'result/map/map_rate_{snr}dB.npy')

map_signal = map_signal[test_idx]
map_signal = map_signal[map_signal != 0]
map_interf = map_interf[test_idx]
map_interf = map_interf[map_interf != 0]
map_rate = map_rate[test_idx]
map_rate = map_rate[map_rate != 0]

# beam
beam_signal = np.load('result/beam/beam_signal.npy')
beam_interf = np.load('result/beam/beam_interf.npy')
beam_rate = np.load(f'result/beam/beam_rate_{snr}dB.npy')

beam_signal = beam_signal[test_idx]
beam_signal = beam_signal[beam_signal != 0]
beam_interf = beam_interf[test_idx]
beam_interf = beam_interf[beam_interf != 0]
beam_rate = beam_rate[test_idx]
beam_rate = beam_rate[beam_rate != 0]

# sort_idx_rate = np.argsort(true_rate)
# plot result
def plot_error(true_value, pred_value, query_value, map_value, beam_value, name, unit,
               min, max):

    # true_sorted  = true_value[sort_idx]
    # pred_sorted  = pred_value[sort_idx]
    # query_sorted = query_value[sort_idx]
    # map_sorted   = map_value[sort_idx]
    # beam_sorted  = beam_value[sort_idx]

    x = np.arange(true_value.shape[0])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    
    plt.plot(x, map_value, c='#F65314', linewidth=2.0, label="Baseline 1 [17]")
    plt.plot(x, query_value, c='#FFBB00', linewidth=2.0, label="Baseline 2 [24]")
    plt.plot(x, beam_value, c='#7CBB00', linewidth=2.0, label="Baseline 3 [25]")
    plt.plot(x, pred_value, c='#00A1F1', linewidth=2.0, label="Proposed")
    plt.plot(x, true_value, c='#68217A', linewidth=2.0, label="True")

    plt.xlabel("UE ID", fontsize=font1)
    plt.ylabel(f"{name} ({unit})", fontsize=font1)
    plt.legend(fontsize=font2, ncol=3, columnspacing=1, handletextpad=0.3, handlelength=1, labelspacing=0.3, loc='upper right')
    plt.xticks(fontsize=font1)
    plt.yticks(fontsize=font1)
    plt.ylim([min, max])
    plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)
    ax.set_xticks(np.arange(0, int(ue_num), 4))
    plt.tight_layout()

plot_error(true_signal, pred_signal, query_signal, map_signal, beam_signal, "Signal power", "dBW", -100, -75)
plt.savefig('result/signal_twin.pdf')
plot_error(true_interf, pred_interf, query_interf, map_interf, beam_interf, "Interference power", "dBW", -94, -78)
plt.savefig('result/interf_twin.pdf')
plot_error(true_rate, pred_rate, query_rate, map_rate, beam_rate, "SE", "bits/s/Hz", 0, 4.0)
plt.savefig('result/rate_twin.pdf')
plt.show()