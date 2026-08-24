# Merge the chunked outputs of ablation/ste_baselines.py (PBS runs with
# --start/--n) into the single per-objective file plot_alloc_cdf.py reads.
#
#   python3 ablation/merge_ste_chunks.py --chunk 500 --total 2000
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--chunk', type=int, default=500)
ap.add_argument('--total', type=int, default=2000)
args = ap.parse_args()

for objective in ('max_sum_rate', 'max_min', 'fairness'):
    parts = []
    for start in range(0, args.total, args.chunk):
        suffix = f'{start}_{args.chunk}' if start else f'{args.chunk}'
        parts.append(np.load(
            f'result/ste_baselines_{objective}_{suffix}.npz'))
    keys = parts[0].files
    merged = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    n = {k: v.size for k, v in merged.items()}
    assert all(v == args.total for v in n.values()), n
    out = f'result/ste_baselines_{objective}_{args.total}.npz'
    np.savez(out, **merged)
    ms = [k for k in keys if not k.endswith('_time')]
    print(f'{objective}: {args.total} scenarios x {len(ms)} methods -> {out}')
