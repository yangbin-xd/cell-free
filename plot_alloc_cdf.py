# Empirical-CDF figures for the resource-allocation comparison, from the
# per-scenario arrays ablation/ste_baselines.py saves to
# result/ste_baselines_<objective>_<n>.npz.
#
# One figure per objective, ECDF of the metric that objective optimises.
# Colours follow the old plot_alloc.py (now in _trash_20260823/); curves are distinguished by line style
# (no markers).
#
#   python3 plot_alloc_cdf.py [--n 40]
#   python3 plot_alloc_cdf.py --n 500 --shards 0,500,1000,1500   # concat shards
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
# 'Times New Roman' is absent on Katana; Nimbus Roman is URW's
# metric-compatible clone of it. Without this chain matplotlib falls back
# to DejaVu Sans and the figures come out in the wrong typeface entirely.
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Nimbus Roman',
                                     'DejaVu Serif']
font1, font2 = 20, 17

# (npz key, legend label, colour, linestyle). Legend order = list order;
# colours run red / yellow / green / blue / purple down the legend. The
# oracle slot is WMMSE-ADMM for max_sum/fairness, SOCP for max_min.
METHODS = [
    ('original',   'Random',             '#F65314', '-'),                   # solid
    ('ORACLE',     None,                 '#FFBB00', (0, (12, 4))),          # long dash
    ('APG',        'APG [20]',           '#7CBB00', (0, (5, 2.5))),         # short dash
    ('Power-Only', 'Proposed (power)',  '#00A1F1', (0, (1, 2.5))),         # dotted
    ('Leaky STE',  'Proposed (joint)',  '#68217A', (0, (8, 2.5, 1.5, 2.5))),  # dash-dot
]

# Legend label (with citation) for each oracle npz key.
ORACLE_LABEL = {'WMMSE-ADMM': 'WMMSE-ADMM [19]', 'SOCP': 'SOCP [2]'}

# (xlabel, oracle method, output path, legend loc, xlim or None)
OBJECTIVES = {
    'max_sum_rate': ('Sum SE (bits/s/Hz)', 'WMMSE-ADMM',
                     'result/alloc_sum_cdf.pdf', 'lower right', (15, 78)),
    'max_min':      ('Min UE SE (bits/s/Hz)', 'SOCP',
                     'result/alloc_min_cdf.pdf', 'lower right', (None, 3.2)),
    'fairness':     ('Proportional fairness', 'WMMSE-ADMM',
                     'result/alloc_pf_cdf.pdf', 'upper left', (-35, 32)),
}


def ecdf(v):
    xs = np.sort(v)
    ys = np.arange(1, xs.size + 1) / xs.size
    return xs, ys


def load_shards(objective, n, starts):
    """Concatenate the per-shard npz files (start 0 has no start suffix,
    matching ablation/ste_baselines.py's naming)."""
    parts = []
    for st in starts:
        suffix = f'{st}_{n}' if st else f'{n}'
        parts.append(dict(np.load(
            f'result/ste_baselines_{objective}_{suffix}.npz')))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--shards', type=str, default='',
                    help='comma-separated --start offsets to concatenate '
                         '(each shard has --n scenarios)')
    args = ap.parse_args()
    starts = [int(x) for x in args.shards.split(',') if x] or [0]


    for objective, (xlabel, oracle, out, loc, xlim) in OBJECTIVES.items():
        data = load_shards(objective, args.n, starts)

        plt.figure(figsize=(6.5, 5))
        for key, label, color, ls in METHODS:
            if key == 'ORACLE':
                key, label = oracle, ORACLE_LABEL[oracle]
            xs, ys = ecdf(data[key])
            plt.step(xs, ys, where='post', lw=3, color=color, ls=ls,
                     label=label)

        plt.xlabel(xlabel, fontsize=font1)
        plt.ylabel('CDF', fontsize=font1)
        plt.xticks(fontsize=font1)
        plt.yticks(fontsize=font1)
        plt.ylim(0, 1)
        if xlim is not None:
            plt.xlim(*xlim)
        plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)
        plt.legend(fontsize=font2, loc=loc, framealpha=0.9,
                   handletextpad=0.3, handlelength=2.2, labelspacing=0.3)
        plt.tight_layout()
        plt.savefig(out)   # no bbox crop: keep the full 6.5x5 in page like plot_pdf.py / plot_band.py
        plt.close()
        print(f'{objective}: -> {out}  ({data[key].size} scenarios)')
