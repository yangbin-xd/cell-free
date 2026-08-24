# SE prediction MAE vs SNR, proposed model against baselines 1-4.
#
# All five sources use the SAME per-UE convention: a flat array of signed
# per-UE errors over the 2000 test realizations with the padding removed
# (39767 entries), so mean(abs(.)) is directly comparable. The baselines get
# there via calculate_mae()'s `!= 0` filter (query.py:65, beam.py:88,
# sinr_map.py, closed_pathloss.py:101); the proposed model via ablation_common's
# save_mae() NaN filter. The script asserts the lengths agree rather than
# trusting that.
#
# Baseline arrays are unchanged since 2026-07-26 and remain valid: they are
# built from the 2000-position measurement pool (radio_map.py:10-20), not from
# training realizations, so the realization count and the train/val split do not
# affect them.
#
#   python3 ablation/make_snr_table.py
import numpy as np

SNR = (0, 5, 10, 15, 20, 25, 30)
VARIANT = 'main_seed0'   # the ablation table's Proposed row (full model, no epoch cap, early stopping only), seed 0

ROWS = [
    ('Baseline 1 [2] (closed-form)', 'result/closed/closed_pl_rate_mae_{}dB.npy'),
    ('Baseline 2 [19] (radio map)', 'result/map/map_rate_mae_{}dB.npy'),
    ('Baseline 3 [26] (query)', 'result/query/query_rate_mae_{}dB.npy'),
    ('Baseline 4 [27] (beam)', 'result/beam/beam_rate_mae_{}dB.npy'),
    ('Proposed', f'result/pred/{VARIANT}/pred_rate_finetune_{{}}dB.npy'),
]


def mae(path):
    a = np.load(path)
    a = a[~np.isnan(a)]
    return a.size, float(np.mean(np.abs(a)))


def build():
    table, sizes = [], set()
    for label, pat in ROWS:
        vals = []
        for s in SNR:
            n, m = mae(pat.format(s))
            sizes.add(n)
            vals.append(m)
        table.append((label, vals))
    if len(sizes) != 1:
        raise SystemExit(f'per-UE counts disagree across sources: {sorted(sizes)} '
                         f'-- the rows are not the same metric, do not tabulate them')
    return table, sizes.pop()


def to_markdown(table):
    lines = ['| Method | ' + ' | '.join(f'{s} dB' for s in SNR) + ' |',
             '|---|' + '---|' * len(SNR)]
    for label, vals in table:
        lines.append(f'| {label} | ' + ' | '.join(f'{v:.4f}' for v in vals) + ' |')
    return '\n'.join(lines)


def to_latex(table):
    lines = [r'\begin{tabular}{l' + 'c' * len(SNR) + '}', r'\toprule',
             r'Method & ' + ' & '.join(rf'{s}\,dB' for s in SNR) + r' \\',
             r'\midrule']
    for label, vals in table:
        tex = label.replace('[', '\\cite{').replace(']', '}') \
            if label.startswith('Baseline') else label
        lines.append(tex + ' & ' + ' & '.join(f'{v:.4f}' for v in vals) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


if __name__ == '__main__':
    table, n = build()
    md = to_markdown(table)
    print(md)
    print(f'\n({n} per-UE errors per cell, 2000 test realizations)')
    with open('result/se_snr_table.md', 'w') as f:
        f.write(md + f'\n\nSE prediction MAE (bits/s/Hz), {n} per-UE errors per '
                     f'cell. Proposed = {VARIANT}.\n')
    with open('result/se_snr_table.tex', 'w') as f:
        f.write(to_latex(table) + '\n')
    print('\nsaved: result/se_snr_table.md, result/se_snr_table.tex')
