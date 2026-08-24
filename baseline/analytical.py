# baseline 4: statistical-CSI (large-scale fading) baseline
# Each AP reports its large-scale channel gains g_{l,k} to the CC, and the CC
# computes SE with a closed-form statistical expression (no training):
#   signal: S_k = N * (sum_{l in M_k} sqrt(p_{l,k} * g_{l,k}))^2
#           exact per subcarrier under MRT (h^H w = ||h||, ||h||^2 = N*g); the
#           only approximation is averaging g over the 12 subcarriers of one RB
#   interf: I_k = sum_l g_{l,k} * (sum_m p_{l,m} - p_{l,k})        (isotropic leakage)
#   rate:   R_k = log2(1 + S_k / (I_k + sigma^2))
# where g_{l,k} is the per-antenna gain averaged over antennas and subcarriers,
# exactly CellFree.gain in main.py.
import os
import numpy as np
from tqdm import tqdm
from generate import cf_test, num_test
from process import AP_num_test, UE_num_test, loc_test,\
                    signal_test, interf_test, rate_test

N_ant = 32
SNRs = np.arange(0, 31, 5)

A_full_test = np.load('data/test/A_full_test.npy') # (2000, 12, 30)
P_full_test = np.load('data/test/P_full_test.npy') # (2000, 12, 30)
signal_test = signal_test.numpy()
interf_test = interf_test.numpy()
rate_test = rate_test.numpy() # (7, 2000, 30)

# main
signal_pred = np.zeros(signal_test.shape)
interf_pred = np.zeros(interf_test.shape)
rate_pred = np.zeros(rate_test.shape)

# replay the per-sample RNG chain of generate.generate_data (base_seed=0) to
# recover each sample's UE subset and its large-scale gains; the stored UE
# locations serve as a bit-exact check that the replay matches the saved test set
ss_root = np.random.SeedSequence(0)
ss_child = ss_root.spawn(num_test)

for i in tqdm(range(num_test), desc="Analytical baseline", unit="sample"):

    rng = np.random.default_rng(ss_child[i])
    seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
    ue_num = int(rng.integers(10, 31))
    assert ue_num == int(UE_num_test[i]), f"sample {i}: replayed ue_num mismatch"

    cf_test.random_choose(ue_num, seed)
    ap_num = int(AP_num_test[i])
    stored_ue_loc = loc_test[i, ap_num:ap_num+ue_num, :].numpy()
    assert np.allclose(cf_test.loc_select, stored_ue_loc, atol=1e-3),\
        f"sample {i}: replayed UE locations mismatch (numpy RNG drift?)"

    g = cf_test.gain # (12, ue_num), per-antenna large-scale gain
    A = A_full_test[i, :, :ue_num]
    P = P_full_test[i, :, :ue_num]

    signal = N_ant * ((A * np.sqrt(P * g)).sum(axis=0)) ** 2
    leak = P.sum(axis=1, keepdims=True) - P # (12, ue_num), power leaked to others
    interf = (g * leak).sum(axis=0)

    signal_pred[i, :ue_num] = 10 * np.log10(signal)
    interf_pred[i, :ue_num] = 10 * np.log10(interf)
    for s, snr in enumerate(SNRs):
        noise_power = 10 ** ((-87 - snr) / 10)
        rate_pred[s, i, :ue_num] = np.log2(1 + signal / (interf + noise_power))

def calculate_mae(value_pred, value_test):
    value_pred_flatten = value_pred.flatten()
    value_pred_flatten = value_pred_flatten[value_pred_flatten != 0]
    value_test_flatten = value_test.flatten()
    value_test_flatten = value_test_flatten[value_test_flatten != 0]
    mae = value_pred_flatten - value_test_flatten
    return mae

os.makedirs('result/analytical', exist_ok=True)

np.save('result/analytical/analytical_signal.npy', signal_pred)
np.save('result/analytical/analytical_interf.npy', interf_pred)

signal_mae = calculate_mae(signal_pred, signal_test)
interf_mae = calculate_mae(interf_pred, interf_test)
np.save('result/analytical/analytical_signal_mae.npy', signal_mae)
np.save('result/analytical/analytical_interf_mae.npy', interf_mae)

# per-sample mean absolute errors and sum rates (plot_band.py conventions)
signal_error = np.empty(num_test)
interf_error = np.empty(num_test)
for i in range(num_test):
    ue_num = int(UE_num_test[i])
    signal_error[i] = np.mean(np.abs(signal_pred[i, :ue_num] - signal_test[i, :ue_num]))
    interf_error[i] = np.mean(np.abs(interf_pred[i, :ue_num] - interf_test[i, :ue_num]))
np.save('result/analytical/analytical_signal_error.npy', signal_error)
np.save('result/analytical/analytical_interf_error.npy', interf_error)

for s, snr in enumerate(SNRs):
    np.save(f'result/analytical/analytical_rate_{snr}dB.npy', rate_pred[s])
    rate_mae = calculate_mae(rate_pred[s], rate_test[s])
    np.save(f'result/analytical/analytical_rate_mae_{snr}dB.npy', rate_mae)

    rate_error = np.empty(num_test)
    rate_sum = np.empty(num_test)
    for i in range(num_test):
        ue_num = int(UE_num_test[i])
        rate_error[i] = np.mean(np.abs(rate_pred[s, i, :ue_num] - rate_test[s, i, :ue_num]))
        rate_sum[i] = np.sum(rate_pred[s, i, :ue_num])
    np.save(f'result/analytical/analytical_rate_error_{snr}dB.npy', rate_error)
    np.save(f'result/analytical/analytical_rate_sum_{snr}dB.npy', rate_sum)

if __name__ == '__main__':

    def pearson(pred, test):
        pred_f = pred.flatten()[pred.flatten() != 0]
        test_f = test.flatten()[test.flatten() != 0]
        return np.corrcoef(pred_f, test_f)[0, 1]

    print("\n=== Statistical-CSI (analytical) baseline ===")
    print(f"Signal: MAE {np.mean(np.abs(signal_mae)):.3f} dB, "
          f"bias {np.mean(signal_mae):+.3f} dB, r = {pearson(signal_pred, signal_test):.4f}")
    print(f"Interf: MAE {np.mean(np.abs(interf_mae)):.3f} dB, "
          f"bias {np.mean(interf_mae):+.3f} dB, r = {pearson(interf_pred, interf_test):.4f}")
    for s, snr in enumerate(SNRs):
        rate_mae = calculate_mae(rate_pred[s], rate_test[s])
        print(f"SE @ {snr:2d} dB: MAE {np.mean(np.abs(rate_mae)):.3f} bit/s/Hz")
