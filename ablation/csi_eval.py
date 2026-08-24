# E2: perfect-CSI models evaluated against imperfect-CSI labels. Inference only.
#
# The GNN never consumes CSI -- its inputs are UE/AP coordinates, the
# association matrix and the power allocation, all of which are bit-identical
# across pilot SNRs (generate.py asserts this). Channel estimation error moves
# only the LABELS. So this script answers: what does it cost to train on
# perfect-CSI labels and deploy where CSI is estimated?
#
# Two de-normalisation modes, and the difference between them is the point:
#
#   asis   -- de-normalise with the constants the model was TRAINED with
#             (from data/train). This is the deployed artefact: a checkpoint
#             carries its normalisation, so a systematic label shift shows up
#             in full.
#   recal  -- de-normalise with the constants of the EVALUATION set. This hands
#             the model a free mean/std recalibration, so whatever gap remains
#             is genuine shape error rather than bias.
#
# Importing process.py with CF_DATA_ROOT set gives 'recal' by accident, which
# would silently hide exactly the systematic offset E2 exists to measure.
#
#   CF_DATA_ROOT=data_tau052 python3 ablation/csi_eval.py --seed 0 --label tau052
import argparse
import os
import numpy as np
import torch
from process import *          # test tensors + constants of the ACTIVE root
import signal_map as sm
import interf_map as im
import rate_map as rm
from ablation_common import main_signal_path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ref_constants(ref_root='data'):
    """process.py's normalisation, recomputed from the perfect-CSI training set.

    Mirrors process.py:68-88 exactly, including the `!= 0` mask that drops the
    padding slots of the ragged UE dimension.
    """
    out = {}
    for name in ('signal', 'interf', 'rate'):
        arr = torch.from_numpy(
            np.load(f'{ref_root}/train/{name}_train.npy')).float()
        flat = arr.reshape([7, -1]) if name == 'rate' else arr.reshape(-1)
        valid = flat[flat != 0]
        out[name] = (valid.mean(), valid.std())
    return out


def branch_mae(model, create_edges, true_value, consts):
    """Per-UE errors for one GNN branch, one entry per (mode, realization).

    Returns {mode: (K_padded,) stacked per-UE error matrix}. The forward pass
    is shared: only the de-normalisation differs between modes.
    """
    model = model.to(DEVICE).eval()
    rows = {m: [] for m in consts}
    with torch.no_grad():
        for i in range(loc_test_norm.shape[0]):
            n = int(UE_num_test[i])
            args = create_edges(model, i)
            pred_norm = model(loc_test_norm[i].to(DEVICE), *args)
            for mode, (mu, sd) in consts.items():
                err = (pred_norm * sd + mu) - true_value[i, :n].to(DEVICE)
                row = torch.full((50,), float('nan'))
                row[:n] = err.detach().cpu()
                rows[mode].append(row)
    return {m: torch.stack(v) for m, v in rows.items()}


def signal_edges(model, i):
    return model.create_edges(loc_test_norm[i].to(DEVICE), AP_num_test[i].to(DEVICE),
                              UE_num_test[i].to(DEVICE), A_test[i].to(DEVICE),
                              P_test[i].to(DEVICE))


def interf_edges(model, i):
    return model.create_edges(loc_test_norm[i].to(DEVICE), AP_num_test[i].to(DEVICE),
                              UE_num_test[i].to(DEVICE), A_test[i].to(DEVICE),
                              P_test[i].to(DEVICE))


def rate_mae(path, sig_path, itf_path, snr, consts):
    """Fine-tuned rate head. Its branch de-normalisation lives on the instance,
    so both are overridden per mode -- the closed-form SINR step in
    RateModel.forward consumes dB powers, and feeding it the wrong constants
    would corrupt the SINR itself, not just shift the output."""
    rows = {m: [] for m in consts}
    true_value = rate_test[int(snr / 5)]
    for mode, ref in consts.items():
        model = rm.RateModel(signal_model_path=sig_path, interf_model_path=itf_path,
                             snr=snr)
        model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
        model.signal_mean, model.signal_std = ref['signal']
        model.interf_mean, model.interf_std = ref['interf']
        model.rate_mean, model.rate_std = ref['rate']
        model = model.to(DEVICE).eval()
        with torch.no_grad():
            for i in range(loc_test_norm.shape[0]):
                n = int(UE_num_test[i])
                pred = model(loc_test_norm[i].to(DEVICE), AP_num_test[i].to(DEVICE),
                             UE_num_test[i].to(DEVICE), A_test[i].to(DEVICE),
                             P_test[i].to(DEVICE))
                err = (pred * ref['rate'][1] + ref['rate'][0]) - true_value[i, :n].to(DEVICE)
                row = torch.full((30,), float('nan'))
                row[:n] = err.detach().cpu()
                rows[mode].append(row)
    return {m: torch.stack(v) for m, v in rows.items()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--snr', type=int, default=15)
    ap.add_argument('--label', required=True, help='operating-point tag, e.g. tau052')
    ap.add_argument('--ref-root', default='data', help='root the models were trained on')
    ap.add_argument('--limit', type=int, default=None,
                    help='evaluate only the first N test realizations (smoke test)')
    args = ap.parse_args()

    if args.limit:
        N_EVAL = min(args.limit, loc_test_norm.shape[0])
        loc_test_norm = loc_test_norm[:N_EVAL]
        print(f'[csi_eval] SMOKE TEST: first {N_EVAL} realizations only')

    ref = ref_constants(args.ref_root)
    sig_path = main_signal_path(args.seed)
    stem = os.path.basename(sig_path)[len('signal_'):-len('.pth')]   # main_rerun_seed0
    itf_path = f'model/interf_{stem}.pth'
    rate_path = f'model/rate_{stem}_{args.snr}dB.pth'

    # 'asis' uses the training-set constants; 'recal' those of the active root.
    modes = {
        'signal': {'asis': ref['signal'], 'recal': (signal_mean, signal_std)},
        'interf': {'asis': ref['interf'], 'recal': (interf_mean, interf_std)},
    }
    rate_modes = {
        'asis': ref,
        'recal': {'signal': (signal_mean, signal_std),
                  'interf': (interf_mean, interf_std),
                  'rate': (rate_mean, rate_std)},
    }

    print(f'[csi_eval] label={args.label} seed={args.seed} '
          f'data_root={DATA_ROOT} models={stem}')

    smodel = sm.SignalModel()
    smodel.load_state_dict(torch.load(sig_path, weights_only=True, map_location='cpu'))
    res = {'signal': branch_mae(smodel, signal_edges, signal_test, modes['signal'])}

    imodel = im.InterfModel()
    imodel.load_state_dict(torch.load(itf_path, weights_only=True, map_location='cpu'))
    res['interf'] = branch_mae(imodel, interf_edges, interf_test, modes['interf'])

    res['rate'] = rate_mae(rate_path, sig_path, itf_path, args.snr, rate_modes)

    outdir = f'result/csi/{args.label}'
    os.makedirs(outdir, exist_ok=True)
    for branch, per_mode in res.items():
        for mode, mat in per_mode.items():
            np.save(f'{outdir}/{branch}_{mode}_seed{args.seed}.npy', mat.numpy())
            # per-UE MAE: the ablation table's convention, not the per-sample
            # mean the training logs print.
            mae = np.nanmean(np.abs(mat.numpy()))
            print(f'  {branch:6s} {mode:5s}  per-UE MAE = {mae:.4f}')
