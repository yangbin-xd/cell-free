# E2 across the SNR sweep: perfect-CSI models vs imperfect-CSI labels, 0-30 dB.
#
# csi_eval.py answers the same question at the single fine-tuning SNR (15 dB).
# This extends it to the seven points of result/se_snr_table.md, exploiting the
# same property snr_eval.py does: `snr` enters RateModel only through
# noise_dB = -87 - self.snr in the closed-form SINR step, no parameter is
# SNR-specific, and the rate_mean/rate_std round trip cancels. So one
# fine-tuned checkpoint is evaluated at every SNR without retraining, and the
# two GNN branches run once per test realization and are reused.
#
# asis / recal carry csi_eval.py's meaning exactly:
#   asis   -- de-normalise the branches with the constants the model was
#             TRAINED with (data/train). The deployed artefact.
#   recal  -- de-normalise with the ACTIVE root's constants, i.e. a free
#             mean/std recalibration; the residual is shape error, not bias.
# Only the branch constants matter here: rate_mean/rate_std cancel in
# RateModel.forward's normalise/denormalise round trip.
#
#   CF_DATA_ROOT=data_tau052 python3 ablation/csi_snr_eval.py --seed 0 --label tau052
import argparse
import os
import numpy as np
import torch
from process import *          # test tensors + constants of the ACTIVE root
import rate_map as rm
from ablation_common import MAIN_SIGNAL
from csi_eval import ref_constants

SNRS = (0, 5, 10, 15, 20, 25, 30)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def branch_outputs_norm(model):
    """Normalised signal / interference branch outputs, per test realization.

    De-normalisation is deliberately NOT applied here: the whole point is to
    push the same forward pass through two different constant sets.
    """
    model = model.to(DEVICE).eval()
    sig, itf = [], []
    with torch.no_grad():
        for i in range(loc_test_norm.shape[0]):
            x = loc_test_norm[i].to(DEVICE)
            ap_num, ue_num = AP_num_test[i].to(DEVICE), UE_num_test[i].to(DEVICE)
            A, P = A_test[i].to(DEVICE), P_test[i].to(DEVICE)
            s_idx, s_attr = model.signal_model.create_edges(x, ap_num, ue_num, A, P)
            i1, a1, i2, a2 = model.interf_model.create_edges(x, ap_num, ue_num, A, P)
            sig.append(model.signal_model(x, s_idx, s_attr).cpu())
            itf.append(model.interf_model(x, i1, a1, i2, a2).cpu())
    return sig, itf


def rate_errors(sig, itf, consts, snr):
    """Flat per-UE rate error (bits/s/Hz) at one SNR, one de-normalisation mode."""
    (s_mu, s_sd), (i_mu, i_sd) = consts['signal'], consts['interf']
    noise_power = 10 ** ((-87 - snr) / 10)
    true = rate_test[int(snr / 5)]
    out = []
    for i, (s_n, i_n) in enumerate(zip(sig, itf)):
        n = int(UE_num_test[i])
        signal_linear = torch.pow(10, (s_n * s_sd + s_mu) / 10)
        interf_linear = torch.pow(10, (i_n * i_sd + i_mu) / 10)
        pred = torch.log2(1 + signal_linear / (interf_linear + noise_power))
        out.append((pred - true[i, :n]).numpy())
    return np.concatenate(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fit-snr', type=int, default=15,
                    help='SNR the rate head was fine-tuned at (its checkpoint)')
    ap.add_argument('--label', required=True, help='operating-point tag, e.g. tau052')
    ap.add_argument('--ref-root', default='data', help='root the models were trained on')
    args = ap.parse_args()

    ref = ref_constants(args.ref_root)
    sig_path = MAIN_SIGNAL[args.seed]
    stem = os.path.basename(sig_path)[len('signal_'):-len('.pth')]
    itf_path = f'model/interf_{stem}.pth'
    rate_path = f'model/rate_{stem}_{args.fit_snr}dB.pth'
    for q in (sig_path, itf_path, rate_path):
        if not os.path.exists(q):
            raise SystemExit(f'checkpoint missing: {q}')

    modes = {
        'asis': {'signal': ref['signal'], 'interf': ref['interf']},
        'recal': {'signal': (signal_mean, signal_std),
                  'interf': (interf_mean, interf_std)},
    }

    print(f'[csi_snr_eval] label={args.label} seed={args.seed} '
          f'data_root={DATA_ROOT} models={stem} device={DEVICE.type}')

    model = rm.RateModel(signal_model_path=sig_path, interf_model_path=itf_path,
                         snr=args.fit_snr)
    model.load_state_dict(torch.load(rate_path, weights_only=True,
                                     map_location='cpu'))
    sig, itf = branch_outputs_norm(model)

    outdir = f'result/csi/{args.label}'
    os.makedirs(outdir, exist_ok=True)

    # Guard: at the fit SNR this must reproduce what csi_eval.py already wrote
    # for this label, otherwise the reimplemented arithmetic has drifted from
    # RateModel.forward. csi_eval stores a (N, 30) NaN-padded matrix; the valid
    # entries are in the same order as the flat array built here.
    for mode in modes:
        ref_path = f'{outdir}/rate_{mode}_seed{args.seed}.npy'
        if not os.path.exists(ref_path):
            print(f'  [skip guard] no {ref_path}')
            continue
        got = rate_errors(sig, itf, modes[mode], args.fit_snr)
        want = np.load(ref_path)
        want = want[~np.isnan(want)]
        if got.shape != want.shape:
            raise SystemExit(f'{mode}: shape {got.shape} vs {want.shape} in {ref_path}')
        gap = float(np.max(np.abs(got - want)))
        # Same tolerance rationale as snr_eval.py: not bit-exact across devices,
        # but 3 orders below the ~0.33 bits/s/Hz being measured.
        if gap > 1e-4:
            raise SystemExit(f'{mode}: does not reproduce {ref_path} '
                             f'(max abs diff {gap:.3e})')
        print(f'  [check] {args.fit_snr} dB {mode:5s} reproduces {os.path.basename(ref_path)} '
              f'(max abs diff {gap:.2e})')

    rows = {}
    for mode, consts in modes.items():
        maes = []
        for snr in SNRS:
            err = rate_errors(sig, itf, consts, snr)
            np.save(f'{outdir}/rate_{mode}_{snr}dB_seed{args.seed}.npy', err)
            maes.append(float(np.mean(np.abs(err))))
        rows[mode] = maes
        print(f'  {mode:5s}  ' + '  '.join(f'{s}dB {m:.4f}' for s, m in zip(SNRS, maes)))

    print(f'\nwrote {2 * len(SNRS)} arrays to {outdir}/')
    print('\n| Mode | ' + ' | '.join(f'{s} dB' for s in SNRS) + ' |')
    print('|---|' + '---|' * len(SNRS))
    for mode, maes in rows.items():
        print(f'| {args.label} {mode} | ' + ' | '.join(f'{m:.4f}' for m in maes) + ' |')
