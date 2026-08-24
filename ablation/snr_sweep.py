# SE-vs-SNR sweep for an already-trained full-model variant.
#
# The signal and interference branches are SNR-independent and are never
# retrained here -- only the rate stage, which is fine-tuned separately at each
# SNR exactly as main_map.py does at 15 dB (same optimiser, LR 1e-4, 500-epoch
# budget, resumable checkpoint).
#
# Why this is not just `main_map.py --stage rate --snr 0`: main_map.py writes
# `pred_rate_{name}_finetune.npy` with no SNR in the name, so running it at a
# second SNR would silently overwrite the 15 dB arrays that the ablation table's
# reference row is built from. Every output here is SNR-keyed. The rate
# CHECKPOINT path is the same `model/rate_{name}_{snr}dB.pth` main_map.py uses,
# so the 15 dB point reuses the trained model behind the table instead of
# retraining it -- its MAE must come out identical.
#
# Outputs per (variant, snr):
#   result/pred/pred_rate_{name}_{snr}dB_mae.npy       per-UE err, pre-fine-tune
#   result/pred/pred_rate_{name}_{snr}dB_finetune.npy  per-UE err  <- the marker
#   result/pred/pred_rate_error_{name}_{snr}dB.npy     per-sample mean |err|
#   result/pred/pred_rate_sum_{name}_{snr}dB.npy       per-sample sum of pred
# The last two are the arrays plot_snr.py / plot_band.py consume; they are
# written under the variant name and do NOT touch the paper's existing
# pred_rate_error_{snr}dB.npy / pred_rate_sum_{snr}dB.npy.
import os
import torch
import numpy as np
import torch.nn as nn
from process import *
import rate_map as rm
from ablation_common import set_seed, add_common_args, variant_name, save_mae,\
    train_resumable, clear_ckpt, RATE_FORWARD


def predicted_sums(model, snr):
    """Per-test-sample sum of predicted rate, and of true rate.

    plot_band.py's (commented-out) generator defines the sum-rate point as
    np.sum over that realization's UEs, in bits/s/Hz -- i.e. de-normalised
    predictions, not the normalised targets the loss sees.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    ap_num, ue_num = AP_num_test.to(device), UE_num_test.to(device)
    loc, A, P = loc_test_norm.to(device), A_test.to(device), P_test.to(device)
    true_value = rate_test[int(snr / 5)].to(device)

    pred_sum = np.empty(loc.shape[0])
    true_sum = np.empty(loc.shape[0])
    with torch.no_grad():
        for i in range(loc.shape[0]):
            pred = model(loc[i], ap_num[i], ue_num[i], A[i], P[i])
            pred = pred * rate_std + rate_mean
            pred_sum[i] = pred.sum().item()
            true_sum[i] = true_value[i, :ue_num[i]].sum().item()
    return pred_sum, true_sum


if __name__ == "__main__":
    parser = add_common_args()
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=None)
    args = parser.parse_args()

    TR = args.train_ratio
    VR = round(1.0 - TR, 10) if args.val_ratio is None else args.val_ratio
    name = variant_name('main', args.tag, args.seed)
    snr = args.snr

    signal_path = f'model/signal_{name}.pth'
    interf_path = f'model/interf_{name}.pth'
    for p in (signal_path, interf_path):
        if not os.path.exists(p):
            raise SystemExit(
                f'branch checkpoint missing: {p}. This script only sweeps the '
                f'rate stage; train the branches with ablation/main_map.py '
                f'--stage signal / --stage interf first.')

    set_seed(args.seed)
    model = rm.RateModel(signal_model_path=signal_path,
                         interf_model_path=interf_path, snr=snr)
    model_path = f'model/rate_{name}_{snr}dB.pth'

    mse_loss, mae_loss, value, mae_mat = rm.evaluate_model(model, snr)
    save_mae(mae_mat, f'result/pred/pred_rate_{name}_{snr}dB_mae.npy')
    print(f"[rate_{name}@{snr}dB] pre-finetune Test MAE: {mae_loss:.3f}")

    if not os.path.exists(model_path):
        train_loader, val_loader = split_train_val(rate_norm[int(snr / 5)], TR, VR)
        model, train_losses, val_losses = train_resumable(
            model, train_loader, val_loader, args.max_epochs or 500,
            RATE_FORWARD, nn.L1Loss(reduction='mean'), 1e-4,
            model_path + '.ckpt')
        np.save(f'loss/rate_{name}_{snr}dB_train_losses.npy', train_losses)
        np.save(f'loss/rate_{name}_{snr}dB_val_losses.npy', val_losses)
        torch.save(model.state_dict(), model_path)
        clear_ckpt(model_path + '.ckpt')
    else:
        print(f'[rate_{name}@{snr}dB] reusing existing {model_path}')

    model.load_state_dict(torch.load(model_path, weights_only=True,
                                     map_location='cpu'))
    mse_loss, mae_loss, value, mae_mat = rm.evaluate_model(model, snr)

    # per-sample mean |err| (plot_snr.py's rate-error curve) comes straight off
    # the same matrix, so it cannot drift from the per-UE number in the table
    err = np.abs(mae_mat.numpy())
    np.save(f'result/pred/pred_rate_error_{name}_{snr}dB.npy', np.nanmean(err, axis=1))

    pred_sum, true_sum = predicted_sums(model, snr)
    np.save(f'result/pred/pred_rate_sum_{name}_{snr}dB.npy', pred_sum)
    ref = f'result/true_rate_sum_{snr}dB.npy'
    if os.path.exists(ref):
        d = np.max(np.abs(np.load(ref) - true_sum))
        print(f'[check] true sum-rate vs {ref}: max abs diff {d:.3e}')

    # written last: it is the marker the PBS chain tests for completion
    save_mae(mae_mat, f'result/pred/pred_rate_{name}_{snr}dB_finetune.npy')
    per_ue = float(np.nanmean(err))
    print(f"[rate_{name}@{snr}dB] Test MSE: {mse_loss:.3f}, "
          f"Test RMSE: {np.sqrt(mse_loss):.3f}, "
          f"Test MAE (per-sample): {mae_loss:.4f}, "
          f"Test MAE (per-UE): {per_ue:.4f}, "
          f"Test SUM: {value:.3f}")
