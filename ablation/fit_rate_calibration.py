# Monotone recalibration of the fine-tuned rate head's output.
#
# The surrogate compresses the rate distribution: on the campaign's test
# split it over-predicts low rates by ~+0.1 bits/s/Hz and under-predicts high
# rates by ~-0.23. For max-min this is provably harmless (a per-UE-identical
# monotone map commutes with min), but the proportional-fairness utility
# sum(log r) has curvature, so optimising log(pred) instead of log(true)
# systematically under-weights the UEs that are truly weakest.
#
# Fit: piecewise-linear monotone map g(pred) ~= E[true | pred], fitted on the
# VALIDATION slice of the training split (the same last-25% slice
# process.split_train_val holds out; the test split is never touched).
# Knots are quantile-bin means with an isotonic (cummax) pass. joint_optimize
# applies g inside the fairness loss when CF_RATE_CALIB points at the file.
#
#   CF_MODEL_STEM=main_seed0 python3 ablation/fit_rate_calibration.py
import argparse
import os
import numpy as np
import torch
from process import (loc_train_norm, AP_num_train, UE_num_train, A_train,
                     P_train, rate_train, rate_mean, rate_std)
import rate_map as rm

SNR = 15


def collect_pairs(model, idx):
    """(pred, true) per-UE rate pairs at 15 dB over the given realizations."""
    preds, trues = [], []
    with torch.no_grad():
        for i in idx:
            n = int(UE_num_train[i])
            p = model(loc_train_norm[i], AP_num_train[i], UE_num_train[i],
                      A_train[i], P_train[i])
            preds.append((p * rate_std + rate_mean).numpy()[:n])
            trues.append(rate_train[int(SNR / 5), i, :n].numpy())
    return np.concatenate(preds), np.concatenate(trues)


def fit_knots(pred, true, n_bins=25):
    """Quantile-binned conditional means, made strictly increasing."""
    edges = np.quantile(pred, np.linspace(0, 1, n_bins + 1))
    kx, ky = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pred >= lo) & (pred <= hi)
        if m.sum() < 10:
            continue
        kx.append(pred[m].mean())
        ky.append(true[m].mean())
    kx, ky = np.asarray(kx), np.asarray(ky)
    ky = np.maximum.accumulate(ky)          # isotonic
    ky = ky + 1e-4 * np.arange(len(ky))     # strictly increasing (nonzero slope)
    return np.stack([kx, ky]).astype(np.float32)


def apply_knots(knots, r):
    kx, ky = knots
    return np.interp(r, kx, ky)  # eval-only; torch version lives in joint_optimize


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', default=os.environ.get('CF_MODEL_STEM',
                                                     'main_seed0'))
    ap.add_argument('--limit', type=int, default=150,
                    help='validation realizations to use (from the val slice)')
    args = ap.parse_args()

    n_total = loc_train_norm.shape[0]
    n_train = int(0.75 * n_total)           # mirrors process.split_train_val
    val_idx = list(range(n_train, min(n_train + args.limit, n_total)))
    print(f'[calib] stem={args.stem}  val realizations '
          f'{val_idx[0]}..{val_idx[-1]} ({len(val_idx)})')

    model = rm.RateModel(signal_model_path=f'model/signal_{args.stem}.pth',
                         interf_model_path=f'model/interf_{args.stem}.pth',
                         snr=SNR)
    model.load_state_dict(torch.load(f'model/rate_{args.stem}_{SNR}dB.pth',
                                     weights_only=True, map_location='cpu'))
    model.eval()

    pred, true = collect_pairs(model, val_idx)
    print(f'[calib] {pred.size} per-UE pairs, raw MAE {np.mean(np.abs(pred - true)):.4f}')

    # fit on even realU pairs, sanity-check on odd
    half = pred.size // 2
    knots = fit_knots(pred[:half], true[:half])
    heldout = np.mean(np.abs(apply_knots(knots, pred[half:]) - true[half:]))
    print(f'[calib] held-out MAE after calibration {heldout:.4f} '
          f'(raw {np.mean(np.abs(pred[half:] - true[half:])):.4f})')

    knots = fit_knots(pred, true)           # final fit on all val pairs
    out = f'result/calib_rate_{args.stem}.npy'
    np.save(out, knots)
    with np.printoptions(precision=3, suppress=True):
        print(f'[calib] knots x: {knots[0]}')
        print(f'[calib] knots y: {knots[1]}')
    print(f'saved: {out}')
