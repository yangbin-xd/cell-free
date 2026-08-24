# baseline 1
import numpy as np
from tqdm import tqdm
from main import BS_loc
from generate import num_train, num_test, UE_train, UE_test
from process import AP_num_train, UE_num_train, loc_train,\
                    signal_train, interf_train, rate_train
from process import AP_num_test, UE_num_test, loc_test,\
                    signal_test, interf_test, rate_test
from baseline.query import calculate_mae

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

snr = 30
noise_dB = -87 - snr
noise_power = 10 ** (noise_dB / 10)

UE_loc_train = np.zeros([num_train, 30, 2])
for i in np.arange(num_train):
    ap_num = AP_num_train[i]
    ue_num = UE_num_train[i]
    UE_loc_train[i, :ue_num, :] = loc_train[i, ap_num:ap_num+ue_num, :]  

ue_num_cate = 21

unique_Loc = np.zeros([ue_num_cate, UE_train.shape[0], 2])
mean_signal_W = np.zeros([ue_num_cate, UE_train.shape[0]])
mean_signal_dB = np.zeros([ue_num_cate, UE_train.shape[0]])
mean_interf_W = np.zeros([ue_num_cate, UE_train.shape[0]])
mean_interf_dB = np.zeros([ue_num_cate, UE_train.shape[0]])
mean_rate = np.zeros([ue_num_cate, UE_train.shape[0]])

for i in range(ue_num_cate):
    idx = np.where(UE_num_train == (10 + i))[0]
    loc_train_flatten = UE_loc_train[idx].reshape([-1, 2])
    loc_train_flatten = loc_train_flatten[~np.all(loc_train_flatten == 0, axis=1)]
    unique_loc, inv = np.unique(loc_train_flatten, axis=0, return_inverse=True)
    num_loc = int(unique_loc.shape[0])
    unique_Loc[i, :num_loc, :] = unique_loc

    signal_train_flatten = signal_train[idx].numpy().flatten()
    signal_train_flatten = signal_train_flatten[signal_train_flatten != 0]
    signal_train_W = 10 ** (signal_train_flatten / 10)
    mean_signal_W[i, :num_loc] = \
        np.bincount(inv, weights=signal_train_W) / np.bincount(inv)
    mean_signal_dB[i, :num_loc] = 10 * np.log10(mean_signal_W[i, :num_loc])

    interf_train_flatten = interf_train[idx].numpy().flatten()
    interf_train_flatten = interf_train_flatten[interf_train_flatten != 0]
    interf_train_W = 10 ** (interf_train_flatten / 10)
    mean_interf_W[i, :num_loc] = \
        np.bincount(inv, weights=interf_train_W) / np.bincount(inv)
    mean_interf_dB[i, :num_loc] = 10 * np.log10(mean_interf_W[i, :num_loc])

    rate_train_flatten = rate_train[int(snr/5), idx].numpy().flatten()
    rate_train_flatten = rate_train_flatten[rate_train_flatten != 0]
    mean_rate[i, :num_loc] = np.bincount(inv, weights=rate_train_flatten) / np.bincount(inv)
    
# show scenario
def show_mean_value(ue_num, mean_value, name):
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_loc = unique_Loc[ue_num-10, :]
    unique_loc = unique_loc[~np.all(unique_loc == 0, axis=1)]
    mean_value = mean_value[ue_num-10]
    mean_value = mean_value[mean_value != 0]

    # plot BS
    ax.scatter(BS_loc[:, 0], BS_loc[:, 1], marker='^', c='#F65314', s = 100,
               label='BS')
    # plot UE
    sc = ax.scatter(unique_loc[:,0], unique_loc[:,1], c=mean_value, 
                    marker='.', s=50, label='UE')

    cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label(f"{name}", fontsize=font1)
    cbar.ax.tick_params(labelsize=font2)

    ax.set_aspect(1)
    plt.legend(fontsize=font1)
    plt.xlabel('X (m)', fontsize=font1)
    plt.ylabel('Y (m)', fontsize=font1)
    plt.xlim(470, 770)
    plt.ylim(160, 360)
    plt.xticks(np.arange(500, 750+1, 50), fontsize=font1)
    plt.yticks(np.arange(160, 360+1, 40), fontsize=font1)
    ax.invert_xaxis()
    plt.tight_layout()

# show_mean_value(10, mean_signal_dB, "Signal (dB)")
# show_mean_value(10, mean_interf_dB, "Interference (dB)")
# show_mean_value(10, mean_rate, "Rate (bits/s/Hz)")
# plt.show()

UE_train = UE_train[:, :2]
UE_test = UE_test[:, :2]
N_train = UE_train.shape[0]
N_test = UE_test.shape[0]
# calculate distance to apply inverse distance weighting (IDW)

dist = np.empty([ue_num_cate, N_test, N_train])
for i in range(ue_num_cate):
    unique_loc = unique_Loc[i, :, :]
    unique_loc = unique_loc[~np.all(unique_loc == 0, axis=1)]
    unique_num = unique_loc.shape[0]

    for j in range(N_test):
        for k in range(unique_num):
            dist[i,j,k] = np.sqrt(np.sum((UE_test[j] - unique_loc[k]) ** 2))


def build_index(UE_loc):
    UE_loc_round = np.round(UE_loc, 3)
    return {tuple(row.tolist()): idx for idx, row in enumerate(UE_loc_round)}

idx_test = build_index(UE_test)

# K nearest neighbors (kNN) + IDW
def kNN(ue_num, loc_test, mean_value, k):
    """
    loc_test: (2,), UE_test: (500, 2)
    loc_train: (2000, 2), unique_loc: (x<=2000, 2)
    mean_value: (x<=2000)

    """
    key = tuple(loc_test.tolist())
    idx_test_query = idx_test[key]

    dist_query = dist[ue_num-10, idx_test_query, :]
    dist_query = dist_query[dist_query != 0]
    dist_sort = sorted(enumerate(dist_query), key=lambda x:x[1])
    index = [x[0] for x in dist_sort]
    K = index[0:k]
    dist_k = dist_query[K]
    value = np.squeeze(mean_value[K])
    
    weight = 1 / dist_k**2
    weight = weight/np.sum(weight)
    out_value = np.dot(weight, value)
    return out_value


UE_signal = np.zeros([num_test, 30])
UE_interf = np.zeros([num_test, 30])
UE_rate = np.zeros([num_test, 30])

iterable = range(num_test)
iterable = tqdm(iterable, desc="Testing samples", unit="sample")

for i in iterable:
    ap_num = AP_num_test[i]
    ue_num = UE_num_test[i]

    for j in range(ue_num):

        UE_signal[i, j] = kNN(ue_num, loc_test[i, ap_num + j], 
                              mean_signal_W[ue_num-10], 3)
        UE_interf[i, j] = kNN(ue_num, loc_test[i, ap_num + j],
                              mean_interf_W[ue_num-10], 3)
        
        # UE_rate[i, j] = kNN(loc_test[i,ap_num + j], unique_loc, mean_rate, 3)
        UE_rate[i, j] = np.log2(1 + UE_signal[i, j] / (UE_interf[i, j] + noise_power)) # + 0.046
        UE_signal[i, j] = 10 * np.log10(UE_signal[i, j]) # - 1.737
        UE_interf[i, j] = 10 * np.log10(UE_interf[i, j]) # - 0.739

np.save(f'result/map/map_signal.npy', UE_signal)
np.save(f'result/map/map_interf.npy', UE_interf)
np.save(f'result/map/map_rate_{snr}dB.npy', UE_rate)

UE_signal = np.load(f'result/map/map_signal.npy')
UE_interf = np.load(f'result/map/map_interf.npy')
UE_rate = np.load(f'result/map/map_rate_{snr}dB.npy')

signal_mae = calculate_mae(UE_signal, signal_test.numpy())
interf_mae = calculate_mae(UE_interf, interf_test.numpy())
rate_mae = calculate_mae(UE_rate, rate_test[int(snr/5)].numpy())

np.save(f'result/map/map_signal_mae.npy', signal_mae)
np.save(f'result/map/map_interf_mae.npy', interf_mae)
np.save(f'result/map/map_rate_mae_{snr}dB.npy', rate_mae)
