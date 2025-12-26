
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'
font1, font2 = 20, 17

# read data
snr = 15

pred_signal_mae = np.load('result/pred/pred_signal_mae.npy')
pred_interf_mae = np.load('result/pred/pred_interf_mae.npy')
pred_rate_mae = np.load(f'result/pred/pred_rate_mae_{snr}dB.npy')
pred_rate_finetune = np.load(f'result/pred/pred_rate_finetune_{snr}dB.npy')

map_signal_mae = np.load('result/map/map_signal_mae.npy')
map_interf_mae = np.load('result/map/map_interf_mae.npy')
map_rate_mae = np.load(f'result/map/map_rate_mae_{snr}dB.npy')

query_signal_mae = np.load('result/query/query_signal_mae.npy')
query_interf_mae = np.load('result/query/query_interf_mae.npy')
query_rate_mae = np.load(f'result/query/query_rate_mae_{snr}dB.npy')

beam_signal_mae = np.load('result/beam/beam_signal_mae.npy')
beam_interf_mae = np.load('result/beam/beam_interf_mae.npy')
beam_rate_mae = np.load(f'result/beam/beam_rate_mae_{snr}dB.npy')

# plot pdf
def plot_distrib(pred, query, map, beam, name, unit, range=10):

    density, edges = np.histogram(pred, bins='fd', density=True)
    density1, edges1 = np.histogram(query, bins='fd', density=True)
    density2, edges2 = np.histogram(map, bins='fd', density=True)
    density3, edges3 = np.histogram(beam, bins='fd', density=True)

    centers = 0.5 * (edges[1:] + edges[:-1])
    centers1 = 0.5 * (edges1[1:] + edges1[:-1])
    centers2 = 0.5 * (edges2[1:] + edges2[:-1])
    centers3 = 0.5 * (edges3[1:] + edges3[:-1])

    win = np.exp(-0.5*((np.arange(-3,4))/1.0)**2)
    win = win / win.sum()

    density = np.convolve(density, win, mode='same')
    density1 = np.convolve(density1, win, mode='same')
    density2 = np.convolve(density2, win, mode='same')
    density3 = np.convolve(density3, win, mode='same')

    plt.figure(figsize=(6.5, 5))
    plt.plot(centers2, density2, lw=2, c='#F65314', label='Baseline 1 [17]')
    plt.plot(centers1, density1, lw=2, c='#FFBB00', label='Baseline 2 [24]')
    plt.plot(centers3, density3, lw=2, c='#7CBB00', label='Baseline 3 [25]')
    plt.plot(centers, density, lw=2, c='#00A1F1', label='Proposed')
    
    plt.xlabel(f'{name} error ({unit})', fontsize=font1)
    plt.ylabel('Probability density', fontsize=font1)
    plt.xticks(fontsize=font1)
    plt.yticks(fontsize=font1)
    plt.xlim([-range, range])
    plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)
    plt.legend(fontsize=font2, handletextpad=0.3, handlelength=1, labelspacing=0.3, loc='upper right')
    plt.tight_layout()

plot_distrib(pred_signal_mae, query_signal_mae, map_signal_mae, beam_signal_mae,
             "Signal power prediction", "dB", 12)
plt.savefig('result/signal_pdf.pdf')
plot_distrib(pred_interf_mae, query_interf_mae, map_interf_mae, beam_interf_mae,
             "Interference power prediction", "dB", 12)
plt.savefig('result/interf_pdf.pdf')
plot_distrib(pred_rate_finetune, query_rate_mae, map_rate_mae, beam_rate_mae,
             "SE prediction", "bits/s/Hz", 3.5)
plt.savefig('result/rate_pdf.pdf')
plt.show()
