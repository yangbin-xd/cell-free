# analyze signal, interference and rate distribution
from process import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
# 'Times New Roman' is absent on Katana; Nimbus Roman is URW's
# metric-compatible clone of it. Without this chain matplotlib falls back
# to DejaVu Sans and the figures come out in the wrong typeface entirely.
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Nimbus Roman',
                                     'DejaVu Serif']
font1, font2 = 24, 18

# read data
signal_reshape = signal_train.reshape(-1)
signal_valid = signal_reshape[signal_reshape != 0]
signal_norm = (signal_valid - signal_mean) / signal_std

interf_reshape = interf_train.reshape(-1)
interf_valid = interf_reshape[interf_reshape != 0]
interf_norm = (interf_valid - interf_mean) / interf_std

rate_reshape = rate_train.reshape(-1)
rate_valid = rate_reshape[rate_reshape != 0]
rate_norm = (rate_valid - rate_mean) / rate_std

def analyze_and_plot(data, name, bins='fd', range_lim=None):
    data_np = data.numpy().astype(float)
    mean = data_np.mean()
    std = data_np.std()
    var = data_np.var()
    median = np.median(data_np)
    minv = data_np.min()
    maxv = data_np.max()

    print("-" * 30)
    print(f"{name}:")
    print("-" * 30)
    print(f"Mean     : {mean:.3f}")
    print(f"Std      : {std:.3f}")
    print(f"Variance : {var:.3f}")
    print(f"Median   : {median:.3f}")
    print(f"Range    : [{minv:.3f}, {maxv:.3f}]")

    # plot pdf
    density, edges = np.histogram(data_np, bins=bins, density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    win = np.exp(-0.5 * ((np.arange(-3, 4)) / 1.0) ** 2)
    win = win / win.sum()
    density = np.convolve(density, win, mode='same')

    plt.figure(figsize=(6, 4))
    plt.plot(centers, density, label=f'{name}', linewidth=2)
    plt.xlabel(f'{name}', fontsize=font1)
    plt.ylabel('PDF', fontsize=font1)
    plt.xticks(fontsize=font2)
    plt.yticks(fontsize=font2)
    plt.grid(True, ls=':', color='gray', alpha=0.3)
    plt.xlim(range_lim)
    plt.tight_layout()

# plot results
analyze_and_plot(signal_valid, "Signal (dB)", range_lim=(-130, -70))
analyze_and_plot(interf_valid, "Interference (dB)", range_lim=(-130, -70))
analyze_and_plot(rate_valid, "Rate (bits/s/Hz)", range_lim=(0, 8))

# analyze_and_plot(rate_norm, "Rate (bits/s/Hz)", range_lim=(-2, 8))
# analyze_and_plot(signal_norm, "Signal (dB)", range_lim=(-5, 5))
# analyze_and_plot(interf_norm, "Interference (dB)", range_lim=(-5, 5))

plt.show()
