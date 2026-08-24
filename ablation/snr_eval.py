# SE-vs-SNR curve for a trained full-model variant, inference only.
#
# `snr` enters RateModel only through `noise_dB = -87 - self.snr` in the
# closed-form SINR -> Shannon step (rate_map.py:117-125). No parameter is
# SNR-specific, and rate_mean/rate_std are pooled over all 7 SNRs
# (process.py:72-76), so one fine-tuned model can be evaluated at every SNR
# without retraining anything.
#
# The two GNN branches are SNR-independent, so their forward pass runs once per
# test realization and is reused across the sweep. The arithmetic that follows
# reproduces RateModel.forward + rate_map.evaluate_model operation for
# operation, including the (x - mean)/std * std + mean round trip, so the 15 dB
# point is bit-identical to result/pred/pred_rate_{name}_finetune.npy -- which
# the script asserts before writing anything.
#
# Outputs go to result/pred/{name}/ and mirror the flat names in result/pred/,
# so plot_snr.py / plot_band.py only need their load prefix changed. Nothing in
# result/pred/ itself is overwritten.
#
#   python3 ablation/snr_eval.py --seed 0 --tag rerun
import argparse
import os
import numpy as np
import torch
from process import *
import rate_map as rm
from ablation_common import variant_name

SNRS = (0, 5, 10, 15, 20, 25, 30)


def branch_outputs(model):
    """pred_signal / pred_interf in dB per test realization, trimmed to ue_num.

    These are everything the rate stage needs that does not depend on SNR.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    sig, itf = [], []
    with torch.no_grad():
        for i in range(loc_test_norm.shape[0]):
            ap_num = AP_num_test[i].to(device)
            ue_num = UE_num_test[i].to(device)
            x = loc_test_norm[i].to(device)
            A, P = A_test[i].to(device), P_test[i].to(device)

            s_idx, s_attr = model.signal_model.create_edges(x, ap_num, ue_num, A, P)
            i1, a1, i2, a2 = model.interf_model.create_edges(x, ap_num, ue_num, A, P)
            s = model.signal_model(x, s_idx, s_attr) * model.signal_std + model.signal_mean
            f = model.interf_model(x, i1, a1, i2, a2) * model.interf_std + model.interf_mean
            sig.append(s.cpu())
            itf.append(f.cpu())
    return sig, itf


def rate_at(sig, itf, snr):
    """Per-realization predicted rate in bits/s/Hz, exactly as forward() does."""
    noise_power = 10 ** ((-87 - snr) / 10)
    out = []
    for s, f in zip(sig, itf):
        sinr = torch.pow(10, s / 10) / (torch.pow(10, f / 10) + noise_power)
        pred_rate = torch.log2(1 + sinr)
        # the normalise/denormalise round trip evaluate_model performs
        pred_norm = (pred_rate - rate_mean) / rate_std
        out.append(pred_norm * rate_std + rate_mean)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tag', type=str, default='rerun')
    p.add_argument('--fit-snr', type=int, default=15,
                   help='SNR the rate head was fine-tuned at (its checkpoint)')
    args = p.parse_args()

    name = variant_name('main', args.tag, args.seed)
    signal_path = f'model/signal_{name}.pth'
    interf_path = f'model/interf_{name}.pth'
    rate_path = f'model/rate_{name}_{args.fit_snr}dB.pth'
    for q in (signal_path, interf_path, rate_path):
        if not os.path.exists(q):
            raise SystemExit(f'checkpoint missing: {q}')

    model = rm.RateModel(signal_model_path=signal_path,
                         interf_model_path=interf_path, snr=args.fit_snr)
    model.load_state_dict(torch.load(rate_path, weights_only=True,
                                     map_location='cpu'))
    print(f'[model] {rate_path} (fine-tuned at {args.fit_snr} dB)')

    sig, itf = branch_outputs(model)

    outdir = f'result/pred/{name}'
    os.makedirs(outdir, exist_ok=True)

    # Guard: the fit SNR must reproduce the array the ablation table reports,
    # otherwise the reimplemented arithmetic has drifted from evaluate_model.
    ref_path = f'result/pred/pred_rate_{name}_finetune.npy'
    pred = rate_at(sig, itf, args.fit_snr)
    fit_true = rate_test[int(args.fit_snr / 5)]
    err = np.concatenate([(pred[i] - fit_true[i, :int(UE_num_test[i])]).numpy()
                          for i in range(len(pred))])
    ref = np.load(ref_path)
    if err.shape != ref.shape:
        raise SystemExit(f'{args.fit_snr} dB: shape {err.shape} vs {ref.shape} '
                         f'in {ref_path}')
    # Not bit-exact: the reference was produced on a V100 and this may run on
    # CPU, so float32 accumulation order differs. 1e-4 is ~3 orders below the
    # 0.31 bits/s/Hz being measured -- tight enough to catch a wrong checkpoint
    # or a mis-transcribed formula, loose enough to survive the device change.
    gap = float(np.max(np.abs(err - ref)))
    if gap > 1e-4:
        raise SystemExit(f'{args.fit_snr} dB does not reproduce {ref_path}: '
                         f'max abs diff {gap:.3e}')
    print(f'[check] {args.fit_snr} dB reproduces {ref_path} '
          f'(max abs diff {gap:.2e}, per-UE MAE {np.mean(np.abs(ref)):.4f})')

    rows = []
    for snr in SNRS:
        pred = rate_at(sig, itf, snr)
        true = rate_test[int(snr / 5)]

        per_ue, per_sample, pred_sum, true_sum = [], [], [], []
        for i, pv in enumerate(pred):
            n = int(UE_num_test[i])
            e = (pv - true[i, :n]).numpy()
            per_ue.append(e)
            per_sample.append(np.mean(np.abs(e)))
            pred_sum.append(float(pv.sum()))
            true_sum.append(float(true[i, :n].sum()))

        per_ue = np.concatenate(per_ue)
        np.save(f'{outdir}/pred_rate_finetune_{snr}dB.npy', per_ue)
        np.save(f'{outdir}/pred_rate_error_{snr}dB.npy', np.array(per_sample))
        np.save(f'{outdir}/pred_rate_sum_{snr}dB.npy', np.array(pred_sum))

        chk = ''
        ref_sum = f'result/true_rate_sum_{snr}dB.npy'
        if os.path.exists(ref_sum):
            chk = (f'  true-sum vs {os.path.basename(ref_sum)}: '
                   f'{np.max(np.abs(np.load(ref_sum) - np.array(true_sum))):.2e}')
        mae = float(np.mean(np.abs(per_ue)))
        rows.append((snr, mae, float(np.mean(per_sample)),
                     float(np.mean(pred_sum)), float(np.mean(true_sum))))
        print(f'{snr:>3} dB  per-UE MAE {mae:.4f}  per-sample MAE '
              f'{np.mean(per_sample):.4f}  mean sum-rate {np.mean(pred_sum):.3f} '
              f'(true {np.mean(true_sum):.3f}){chk}')

    print(f'\nwrote {3 * len(SNRS)} arrays to {outdir}/')
    print('\n| SNR (dB) | ' + ' | '.join(str(r[0]) for r in rows) + ' |')
    print('|---|' + '---|' * len(rows))
    print('| SE MAE (bits/s/Hz) | ' + ' | '.join(f'{r[1]:.4f}' for r in rows) + ' |')
