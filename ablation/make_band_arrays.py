# Per-realization mean absolute error for the proposed model's signal and
# interference branches, which plot_band.py needs but the ablation campaign does
# not produce.
#
# No model inference required. save_mae() (ablation_common.py:77) flattens the
# (num_test, 30) error matrix in row-major order and then drops the NaN padding,
# so the stored per-UE array is already sample-ordered: splitting it at the
# cumulative UE counts recovers exactly the per-realization groups. The script
# checks that the total length matches sum(UE_num_test) before splitting, and
# cross-checks the rate branch against the array snr_eval.py wrote independently.
#
# Outputs (mirroring the flat names plot_band.py loads from result/pred/):
#   result/pred/{variant}/pred_signal_error.npy
#   result/pred/{variant}/pred_interf_error.npy
#
#   python3 ablation/make_band_arrays.py
import numpy as np
from process import UE_num_test

VARIANT = 'main_seed0'
SNR = 15


def per_sample(path, n):
    a = np.load(path)
    a = a[~np.isnan(a)]
    if a.size != n.sum():
        raise SystemExit(f'{path}: {a.size} per-UE errors but sum(UE_num_test) '
                         f'= {n.sum()} -- not sample-ordered, cannot split')
    return np.array([np.mean(np.abs(g)) for g in np.split(a, np.cumsum(n)[:-1])])


if __name__ == '__main__':
    n = UE_num_test.numpy().astype(int)
    outdir = f'result/pred/{VARIANT}'

    for branch in ('signal', 'interf'):
        e = per_sample(f'result/pred/pred_{branch}_{VARIANT}_mae.npy', n)
        np.save(f'{outdir}/pred_{branch}_error.npy', e)
        print(f'{branch:7s} per-realization MAE: mean {e.mean():.4f}, '
              f'min {e.min():.4f}, max {e.max():.4f}  -> {outdir}/pred_{branch}_error.npy')

    # the rate branch already has a per-sample array from snr_eval.py; rebuilding
    # it here from the per-UE array must agree, which validates the split
    rebuilt = per_sample(f'result/pred/pred_rate_{VARIANT}_finetune.npy', n)
    ref = np.load(f'{outdir}/pred_rate_error_{SNR}dB.npy')
    # 1e-4, not 1e-6: this array came off a V100 and snr_eval.py's came off CPU,
    # so float32 accumulation order differs. A wrong split would be O(0.1).
    gap = float(np.max(np.abs(rebuilt - ref)))
    if gap > 1e-4:
        raise SystemExit(f'rate branch disagrees with snr_eval.py output: {gap:.3e}')
    print(f'[check] rate branch split matches {outdir}/pred_rate_error_{SNR}dB.npy '
          f'(max abs diff {gap:.2e})')
