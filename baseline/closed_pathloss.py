# Closed-form baseline with MODEL-BASED beta: instead of APs feeding back the
# measured large-scale gains, the CC derives beta from AP-UE distance via a
# log-distance pathloss model calibrated on the training set:
#   10*log10(beta) = b0 - 10*alpha*log10(d)   (least-squares fit, 3D distance)
# and plugs beta_hat into Eq. (24):
#   S_k = N * (sum_l sqrt(p_{l,k} beta_{l,k}))^2
#   I_k = sum_l beta_{l,k} * sum_{k'} p_{l,k'}
# Inputs are then positions + (B, P) only -- the same information the proposed
# GNN uses, with zero feedback.
#
# Official files (closed_pl_*) use the structure-aware interference: users
# share the elevation angle of the 8x4 UPA, so only the N_az = 8 azimuth
# dimensions are effectively random and E|h^H w|^2 = ||h||^2/8, i.e. the
# Eq. (24) interference scaled by N/N_az = 4. The literal Eq. (24) variant
# (isotropy over all N = 32 antennas) is saved as closed_pl_iso_* for reference.
import os
import numpy as np
from tqdm import tqdm
from main import BS_loc, UE_loc
from generate import cf_train, cf_test, num_test, idx_train, idx_test
from process import AP_num_test, UE_num_test, loc_test,\
                    signal_test, interf_test, rate_test, DATA_ROOT

N_ant = 32
N_az = 8
SNRs = np.arange(0, 31, 5)

A_full_test = np.load(f'{DATA_ROOT}/test/A_full_test.npy')
P_full_test = np.load(f'{DATA_ROOT}/test/P_full_test.npy')
signal_test = signal_test.numpy()
interf_test = interf_test.numpy()
rate_test = rate_test.numpy()

# ---------- fit log-distance pathloss on the training UE pool ----------
# per-antenna mean gain for all (AP, train-pool UE) pairs
g_train = np.mean(np.abs(cf_train.CSI[..., 0]) ** 2, axis=(2, 3)) # (12, N_pool)

BS3, UE3 = BS_loc, UE_loc # (12, 3), (2500, 3), from main.py (raw, 3D)
d_train = np.linalg.norm(BS3[:, None, :] - UE3[None, idx_train, :], axis=-1) # (12, N_pool)

# exclude fully blocked pairs (zero gain, no ray-traced path)
valid = g_train.ravel() > 0
print(f"Pathloss fit pairs: {valid.sum()}/{valid.size} "
      f"({100*(1-valid.mean()):.1f}% blocked pairs excluded)")
x = 10 * np.log10(d_train.ravel()[valid])
y = 10 * np.log10(g_train.ravel()[valid])
Amat = np.stack([np.ones_like(x), -x], axis=1)
(b0, alpha), *_ = np.linalg.lstsq(Amat, y, rcond=None)
resid = y - (b0 - alpha * x)
sigma_sh = resid.std()
print(f"Pathloss fit: 10log10(beta) = {b0:.2f} - {10*alpha:.2f}*log10(d), "
      f"shadowing std {sigma_sh:.2f} dB")

def beta_from_dist(d):
    return 10 ** ((b0 - alpha * 10 * np.log10(d)) / 10)

# ---------- evaluate on the test set ----------
signal_pred = np.zeros(signal_test.shape)
interf_pred = np.zeros(interf_test.shape)
rate_pred = np.zeros(rate_test.shape)
interf_iso_pred = np.zeros(interf_test.shape)
rate_iso_pred = np.zeros(rate_test.shape)

ss_root = np.random.SeedSequence(0)
ss_child = ss_root.spawn(num_test)

for i in tqdm(range(num_test), desc="Pathloss closed-form", unit="sample"):

    rng = np.random.default_rng(ss_child[i])
    seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
    ue_num = int(rng.integers(10, 31))
    assert ue_num == int(UE_num_test[i])

    # replay the UE-subset selection of CellFree.random_choose
    np.random.seed(seed)
    select_index = np.random.choice(cf_test.N, size=ue_num, replace=False)
    ue3 = UE3[idx_test][select_index] # (ue_num, 3)

    ap_num = int(AP_num_test[i])
    stored_ue_loc = loc_test[i, ap_num:ap_num+ue_num, :].numpy()
    assert np.allclose(ue3[:, :2], stored_ue_loc, atol=1e-3),\
        f"sample {i}: replayed UE locations mismatch"

    d = np.linalg.norm(BS3[:, None, :] - ue3[None, :, :], axis=-1) # (12, ue_num)
    beta = beta_from_dist(d)
    A = A_full_test[i, :, :ue_num]
    P = P_full_test[i, :, :ue_num]

    signal = N_ant * ((A * np.sqrt(P * beta)).sum(axis=0)) ** 2
    interf_iso = (beta * P.sum(axis=1, keepdims=True)).sum(axis=0) # literal Eq. (24)
    interf = (N_ant / N_az) * interf_iso # structure-aware (E|h^H w|^2 = ||h||^2/8)

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
    return value_pred_flatten - value_test_flatten

os.makedirs('result/closed', exist_ok=True)
np.save('result/closed/closed_pl_signal.npy', signal_pred)
np.save('result/closed/closed_pl_interf.npy', interf_pred)
np.save('result/closed/closed_pl_signal_mae.npy', calculate_mae(signal_pred, signal_test))
np.save('result/closed/closed_pl_interf_mae.npy', calculate_mae(interf_pred, interf_test))

# per-sample mean absolute errors and sum rates (plot_band.py conventions)
signal_error = np.empty(num_test)
interf_error = np.empty(num_test)
for i in range(num_test):
    n = int(UE_num_test[i])
    signal_error[i] = np.mean(np.abs(signal_pred[i, :n] - signal_test[i, :n]))
    interf_error[i] = np.mean(np.abs(interf_pred[i, :n] - interf_test[i, :n]))
np.save('result/closed/closed_pl_signal_error.npy', signal_error)
np.save('result/closed/closed_pl_interf_error.npy', interf_error)

for s, snr in enumerate(SNRs):
    np.save(f'result/closed/closed_pl_rate_{snr}dB.npy', rate_pred[s])
    np.save(f'result/closed/closed_pl_rate_mae_{snr}dB.npy',
            calculate_mae(rate_pred[s], rate_test[s]))
    rate_error = np.empty(num_test)
    rate_sum = np.empty(num_test)
    for i in range(num_test):
        n = int(UE_num_test[i])
        rate_error[i] = np.mean(np.abs(rate_pred[s, i, :n] - rate_test[s, i, :n]))
        rate_sum[i] = np.sum(rate_pred[s, i, :n])
    np.save(f'result/closed/closed_pl_rate_error_{snr}dB.npy', rate_error)
    np.save(f'result/closed/closed_pl_rate_sum_{snr}dB.npy', rate_sum)

# literal Eq. (24) variant (kept out of the figures)
np.save('result/closed/closed_pl_iso_interf.npy', interf_iso_pred)
np.save('result/closed/closed_pl_iso_interf_mae.npy', calculate_mae(interf_iso_pred, interf_test))
for s, snr in enumerate(SNRs):
    np.save(f'result/closed/closed_pl_iso_rate_mae_{snr}dB.npy',
            calculate_mae(rate_iso_pred[s], rate_test[s]))

if __name__ == '__main__':
    sig = calculate_mae(signal_pred, signal_test)
    print(f"\nSignal: MAE {np.mean(np.abs(sig)):.3f} dB, bias {np.mean(sig):+.3f} dB")
    for name, ip, rp in [("structure-aware /8 (official)", interf_pred, rate_pred),
                         ("literal Eq. (24) /32 (reference)", interf_iso_pred, rate_iso_pred)]:
        intf = calculate_mae(ip, interf_test)
        print(f"\n=== Distance-based beta, {name} ===")
        print(f"Interf: MAE {np.mean(np.abs(intf)):.3f} dB, bias {np.mean(intf):+.3f} dB")
        for s, snr in enumerate(SNRs):
            rate_mae = calculate_mae(rp[s], rate_test[s])
            print(f"SE @ {snr:2d} dB: MAE {np.mean(np.abs(rate_mae)):.3f} bit/s/Hz")
