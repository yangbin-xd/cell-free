# baseline 3
import numpy as np
from tqdm import tqdm
from generate import num_train, num_test, UE_train
from process import AP_num_test, UE_num_test, loc_test,\
                    signal_test, interf_test, rate_test

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'
font1, font2 = 24, 18

snr = 30
noise_dB = -87 - snr
noise_power = 10 ** (noise_dB / 10)

A_full_train = np.load('data/train/A_full_train.npy') # (2000, 12, 30)
P_full_train = np.load('data/train/P_full_train.npy') # (2000, 12, 30)
A_full_test = np.load('data/test/A_full_test.npy') # (2000, 12, 30)
P_full_test = np.load('data/test/P_full_test.npy') # (2000, 12, 30)
signal_test = signal_test.numpy()
interf_test = interf_test.numpy()
rate_test = rate_test[int(snr/5)].numpy()

Kriging_map = np.load('result/Kriging_map.npz', allow_pickle=True)

def query_Kriging(bs_idx, xy):

    gridx = Kriging_map['gridx']
    gridy = Kriging_map['gridy']
    Z = Kriging_map['Z'][int(bs_idx)]

    x2i = {int(v): i for i, v in enumerate(gridx)}
    y2j = {int(v): j for j, v in enumerate(gridy)}

    xy = np.asarray(xy, float)
    def _one(pt):
        xi = int(np.rint(pt[0])); yi = int(np.rint(pt[1]))
        if (xi in x2i) and (yi in y2j):
            return Z[y2j[yi], x2i[xi]]
        return 1e-20

    if xy.ndim == 1:
        return _one(xy)
    else:
        out = np.empty(xy.shape[0], dtype=float)
        for k in range(xy.shape[0]):
            out[k] = _one(xy[k])
        return out


# main
signal_pred = np.zeros(signal_test.shape)
interf_pred = np.zeros(interf_test.shape)
rate_pred = np.zeros(rate_test.shape)

iterable = range(num_test)
iterable = tqdm(iterable, desc="Testing samples", unit="sample")

for i in iterable:

    ap_num = AP_num_test[i]
    ue_num = UE_num_test[i]
    ue_loc = loc_test[i, ap_num:ap_num+ue_num, :]
    a_test = A_full_test[i, :, :ue_num]
    p_test = P_full_test[i, :, :ue_num]

    amp = np.zeros([ue_num, 12])
    for j in np.arange(12):
        amp[:,j] = query_Kriging(j, ue_loc)

    for k in np.arange(ue_num):

        served = a_test[:, k]
        un_served = 1 - served
        power = p_test[:, k]

        signal = (np.sum(served * amp[k,:] * np.sqrt(power)))**2
        interf = np.sum(un_served * amp[k,:]**2) / 8 \
               + np.sum(served * amp[k,:] * np.sqrt(1 - power))**2 / 8
        rate = np.log2(1 + signal / (interf + noise_power))

        signal_pred[i,k] = 10 * np.log10(signal)
        interf_pred[i,k] = 10 * np.log10(interf)
        rate_pred[i,k] = rate

def calculate_mae(value_pred, value_test):
    value_pred_flatten = value_pred.flatten()
    value_pred_flatten = value_pred_flatten[value_pred_flatten != 0]
    value_test_flatten = value_test.flatten()
    value_test_flatten = value_test_flatten[value_test_flatten != 0]
    mae = value_pred_flatten - value_test_flatten
    return mae

np.save(f'result/beam/beam_signal.npy', signal_pred)
np.save(f'result/beam/beam_interf.npy', interf_pred)
np.save(f'result/beam/beam_rate_{snr}dB.npy', rate_pred)

signal_mae = calculate_mae(signal_pred, signal_test)
interf_mae = calculate_mae(interf_pred, interf_test)
rate_mae = calculate_mae(rate_pred, rate_test)

np.save(f'result/beam/beam_signal_mae.npy', signal_mae)
np.save(f'result/beam/beam_interf_mae.npy', interf_mae)
np.save(f'result/beam/beam_rate_mae_{snr}dB.npy', rate_mae)

# # plot results
# print("signal error mean:", np.mean(np.abs(signal_mae)))
# print("interf error mean:", np.mean(np.abs(interf_mae)))
# print("rate error mean:", np.mean(np.abs(rate_mae)))

# test case
def beam(test_idx=0, do_plot=True, do_print=True):

    ue_num = UE_num_test[test_idx]
    true_signal = signal_test[test_idx, :ue_num]
    true_interf = interf_test[test_idx, :ue_num]
    true_rate = rate_test[test_idx, :ue_num]

    pred_signal = signal_pred[test_idx, :ue_num]
    pred_interf = interf_pred[test_idx, :ue_num]
    pred_rate = rate_pred[test_idx, :ue_num]

    if do_print:
        print(f"Pred signal: {pred_signal} dB")
        print(f"Pred interf: {pred_interf} dB")
        print(f"Pred rate:   {pred_rate} bits/s/Hz")

    def plot_error(True_value, Pred_value, term):
        Abs_Error = np.abs(True_value - Pred_value)
        x = np.arange(True_value.shape[0])
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(x, True_value, label=f'True {term}', linewidth=2.0)
        plt.plot(x, Pred_value, label=f'Predicted {term}', linewidth=2.0)
        plt.plot(x, Abs_Error, label='Error', linewidth=2.0)
        plt.xlabel('UE ID', fontsize=font1)
        plt.ylabel(f'{term} (dB)', fontsize=font1)
        plt.legend(fontsize=font1)
        plt.xticks(fontsize=font1)
        plt.yticks(fontsize=font1)
        plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)
        ax.set_xticks(np.arange(0, ue_num, 2))
        plt.tight_layout()

    signal_error = np.abs(pred_signal - true_signal)
    interf_error = np.abs(pred_interf - true_interf)
    rate_error = np.abs(pred_rate - true_rate)

    if do_print:
        print(f"Signal: mean: {np.mean(signal_error)}, max: {np.max(signal_error)}")
        print(f"Interf: mean: {np.mean(interf_error)}, max: {np.max(interf_error)}")
        print(f"Rate: mean: {np.mean(rate_error)}, max: {np.max(rate_error)}")

    if do_plot:
        plot_error(true_signal, pred_signal, "Signal")
        plot_error(true_interf, pred_interf, "Interference")
        plot_error(true_rate,   pred_rate, "Rate")
        plt.show()

    return pred_signal, pred_interf, pred_rate

if __name__ == '__main__':

    test_idx = 0
    pred_signal, pred_interf, pred_rate = beam(test_idx, False, False)


