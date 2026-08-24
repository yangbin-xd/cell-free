# Literal application of Ngo et al. 2017 (TWC), Theorem 1, Eq. (24) to predict
# the rate, specialized to this system model:
#   - perfect local CSI  -> gamma_{l,k} = g_{l,k} (per-antenna large-scale gain)
#   - orthogonal pilots  -> pilot-contamination term = 0
#   - N-antenna MRT      -> array gain N on the coherent signal term
#   - per-AP unit power  -> eta_{l,k'} gamma_{l,k'} = p_{l,k'}
# giving
#   S_k      = N * (sum_l sqrt(p_{l,k} g_{l,k}))^2
#   I_k^(24) = sum_l g_{l,k} * sum_{k'} p_{l,k'}   (all k', incl. the k'=k
#              beamforming-uncertainty term of the UatF bound)
#   R_k      = log2(1 + S_k / (I_k^(24) + sigma^2))
# Differs from analytical.py only by the k'=k term: I^(24) = I^analytical + sum_l g p.
#
# Two interference variants are produced:
#   - literal Eq. (24) (saved as closed_iso_*): isotropy over all N=32 antennas,
#     E|h^H w|^2 = ||h||^2/N -> underestimates interference by ~6 dB here
#   - structure-aware (official, saved as closed_*): users share the elevation
#     angle of the 8x4 UPA, so the channel is h = a_el (x) b_az with common
#     a_el and only the N_az = 8 azimuth dimensions are effectively random:
#     E|h^H w|^2 = ||h||^2/N_az, i.e. interference scaled by N/N_az = 4
import os
import numpy as np
from tqdm import tqdm
from generate import cf_test, num_test
from process import AP_num_test, UE_num_test, loc_test,\
                    signal_test, interf_test, rate_test, DATA_ROOT

N_ant = 32
N_az = 8 # azimuth elements of the 8x4 UPA (users share the elevation angle)
SNRs = np.arange(0, 31, 5)

A_full_test = np.load(f'{DATA_ROOT}/test/A_full_test.npy') # (2000, 12, 30)
P_full_test = np.load(f'{DATA_ROOT}/test/P_full_test.npy') # (2000, 12, 30)
signal_test = signal_test.numpy()
interf_test = interf_test.numpy()
rate_test = rate_test.numpy() # (7, 2000, 30)

signal_pred = np.zeros(signal_test.shape)
interf_pred = np.zeros(interf_test.shape)
rate_pred = np.zeros(rate_test.shape)
interf_iso_pred = np.zeros(interf_test.shape)
rate_iso_pred = np.zeros(rate_test.shape)

ss_root = np.random.SeedSequence(0)
ss_child = ss_root.spawn(num_test)

for i in tqdm(range(num_test), desc="Ngo Eq.(24) baseline", unit="sample"):

    rng = np.random.default_rng(ss_child[i])
    seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
    ue_num = int(rng.integers(10, 31))
    assert ue_num == int(UE_num_test[i]), f"sample {i}: replayed ue_num mismatch"

    cf_test.random_choose(ue_num, seed)
    ap_num = int(AP_num_test[i])
    stored_ue_loc = loc_test[i, ap_num:ap_num+ue_num, :].numpy()
    assert np.allclose(cf_test.loc_select, stored_ue_loc, atol=1e-3),\
        f"sample {i}: replayed UE locations mismatch"

    g = cf_test.gain # (12, ue_num)
    A = A_full_test[i, :, :ue_num]
    P = P_full_test[i, :, :ue_num]

    signal = N_ant * ((A * np.sqrt(P * g)).sum(axis=0)) ** 2
    interf_iso = (g * P.sum(axis=1, keepdims=True)).sum(axis=0) # all k' incl. k'=k
    interf = (N_ant / N_az) * interf_iso # azimuth-only isotropy

    signal_pred[i, :ue_num] = 10 * np.log10(signal)
    interf_pred[i, :ue_num] = 10 * np.log10(interf)
    interf_iso_pred[i, :ue_num] = 10 * np.log10(interf_iso)
    for s, snr in enumerate(SNRs):
        noise_power = 10 ** ((-87 - snr) / 10)
        rate_pred[s, i, :ue_num] = np.log2(1 + signal / (interf + noise_power))
        rate_iso_pred[s, i, :ue_num] = np.log2(1 + signal / (interf_iso + noise_power))

def calculate_mae(value_pred, value_test):
    value_pred_flatten = value_pred.flatten()
    value_pred_flatten = value_pred_flatten[value_pred_flatten != 0]
    value_test_flatten = value_test.flatten()
    value_test_flatten = value_test_flatten[value_test_flatten != 0]
    mae = value_pred_flatten - value_test_flatten
    return mae

os.makedirs('result/closed', exist_ok=True)

np.save('result/closed/closed_signal.npy', signal_pred)
np.save('result/closed/closed_interf.npy', interf_pred)
np.save('result/closed/closed_signal_mae.npy', calculate_mae(signal_pred, signal_test))
np.save('result/closed/closed_interf_mae.npy', calculate_mae(interf_pred, interf_test))

# literal Eq. (24) variant (isotropy over all 32 antennas), kept for reference
np.save('result/closed/closed_iso_interf.npy', interf_iso_pred)
np.save('result/closed/closed_iso_interf_mae.npy', calculate_mae(interf_iso_pred, interf_test))
for s, snr in enumerate(SNRs):
    np.save(f'result/closed/closed_iso_rate_mae_{snr}dB.npy',
            calculate_mae(rate_iso_pred[s], rate_test[s]))

# per-sample mean absolute errors (plot_band.py conventions)
signal_error = np.empty(num_test)
interf_error = np.empty(num_test)
for i in range(num_test):
    ue_num = int(UE_num_test[i])
    signal_error[i] = np.mean(np.abs(signal_pred[i, :ue_num] - signal_test[i, :ue_num]))
    interf_error[i] = np.mean(np.abs(interf_pred[i, :ue_num] - interf_test[i, :ue_num]))
np.save('result/closed/closed_signal_error.npy', signal_error)
np.save('result/closed/closed_interf_error.npy', interf_error)

for s, snr in enumerate(SNRs):
    np.save(f'result/closed/closed_rate_{snr}dB.npy', rate_pred[s])
    np.save(f'result/closed/closed_rate_mae_{snr}dB.npy',
            calculate_mae(rate_pred[s], rate_test[s]))
    rate_error = np.empty(num_test)
    rate_sum = np.empty(num_test)
    for i in range(num_test):
        ue_num = int(UE_num_test[i])
        rate_error[i] = np.mean(np.abs(rate_pred[s, i, :ue_num] - rate_test[s, i, :ue_num]))
        rate_sum[i] = np.sum(rate_pred[s, i, :ue_num])
    np.save(f'result/closed/closed_rate_error_{snr}dB.npy', rate_error)
    np.save(f'result/closed/closed_rate_sum_{snr}dB.npy', rate_sum)

if __name__ == '__main__':

    for name, ip, rp in [("structure-aware (official, x%d)" % (N_ant // N_az), interf_pred, rate_pred),
                         ("literal Eq. (24) (32-dim isotropy)", interf_iso_pred, rate_iso_pred)]:
        interf_mae = calculate_mae(ip, interf_test)
        print(f"\n=== Closed-form, {name} ===")
        print(f"Interf: MAE {np.mean(np.abs(interf_mae)):.3f} dB, "
              f"bias {np.mean(interf_mae):+.3f} dB")
        for s, snr in enumerate(SNRs):
            rate_mae = calculate_mae(rp[s], rate_test[s])
            print(f"SE @ {snr:2d} dB: MAE {np.mean(np.abs(rate_mae)):.3f} bit/s/Hz")
