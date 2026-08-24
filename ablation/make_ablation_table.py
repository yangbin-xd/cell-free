# aggregate all ablation MAE arrays into one grouped table (markdown + LaTeX).
# Every variant, including the reference row, was trained under one protocol
# (no epoch cap, --max-epochs 1000000; runs end on early stopping only, 20
# epochs without val improvement) at seeds 0/1/2, so names are uniformly
# <stage>_<base>_seed<N>. Missing files render as '--'.
import glob
import numpy as np

PRED = 'result/pred'

MAIN = [f'main_seed{s}' for s in (0, 1, 2)]
MAIN_SIG = [f'{PRED}/pred_signal_{n}_mae.npy' for n in MAIN]
MAIN_ITF = [f'{PRED}/pred_interf_{n}_mae.npy' for n in MAIN]
MAIN_RATE = [f'{PRED}/pred_rate_{n}_finetune.npy' for n in MAIN]
MAIN_RATE_PREFT = [f'{PRED}/pred_rate_{n}_mae.npy' for n in MAIN]


def sig(base):
    return f'{PRED}/pred_signal_{base}_seed*_mae.npy'


def itf(base):
    return f'{PRED}/pred_interf_{base}_seed*_mae.npy'


def rate(base):
    return f'{PRED}/pred_rate_{base}_seed*_finetune.npy'


GROUPS = [
    ('', [
        ('Proposed (full model)', MAIN_SIG, MAIN_ITF, MAIN_RATE),
    ]),
    ('Structure ablations', [
        ('w/o decomposition (end-to-end)',
         None, None, f'{PRED}/pred_end2end_seed*_mae.npy'),
        ('w/o pre-training',
         None, None, f'{PRED}/pred_rate_scratch_seed*_mae.npy'),
        ('w/o fine-tuning', MAIN_SIG, MAIN_ITF, MAIN_RATE_PREFT),
        ('Single-layer interference MP',
         None, itf('single'), rate('single')),
    ]),
    ('Aggregation ablations', [
        ('GCN', sig('gcn'), itf('gcn'), rate('gcn')),
        ('GAT (w/o edge features)', sig('gat'), itf('gat'), rate('gat')),
        ('GraphSAGE (mean aggr.)', sig('sage'), itf('sage'), rate('sage')),
        ('GraphSAGE (sum aggr.)',
         sig('sage_sum'), itf('sage_sum'), rate('sage_sum')),
        ('Edge-aware GAT', sig('gat_edge'), itf('gat_edge'), rate('gat_edge')),
        ('H = 1 attention head', sig('h1'), itf('h1'), rate('h1')),
        ('H = 2 attention heads', sig('h2'), itf('h2'), rate('h2')),
        ('H = 4 (main model)', MAIN_SIG, MAIN_ITF, MAIN_RATE),
    ]),
    ('Edge-feature ablations', [
        ('w/o direction', sig('angle'), itf('angle'), rate('angle')),
        ('w/o distance', sig('dist'), itf('dist'), rate('dist')),
        ('w/o power', sig('power'), itf('power'), rate('power')),
        ('Interference edge power: 1-p -> p',
         None, itf('rep'), rate('rep')),
        ('Interference edge power: 1-p removed',
         None, itf('nopow2'), rate('nopow2')),
    ]),
]


def cell(pattern):
    if pattern is None:
        return '--'
    if isinstance(pattern, (list, tuple)):
        # explicit file list: every entry must exist, otherwise the row would
        # silently average over a subset of the seeds
        files = [p for p in pattern if glob.glob(p)]
        if len(files) != len(pattern):
            missing = [p for p in pattern if not glob.glob(p)]
            raise SystemExit(f'missing: {missing}')
    else:
        files = sorted(glob.glob(pattern)) if '*' in pattern else \
            ([pattern] if glob.glob(pattern) else [])
    if not files:
        return '--'
    maes = [float(np.mean(np.abs(np.load(f)))) for f in files]
    if len(maes) == 1:
        return f'{maes[0]:.3f}'
    return f'{np.mean(maes):.3f}±{np.std(maes):.3f}'


def to_markdown(rows):
    lines = ['| Variant | Signal MAE (dB) | Interf. MAE (dB) | SE MAE@15dB |',
             '|---|---|---|---|']
    for label, val in rows:
        if label == '__group__':
            lines.append(f'| ***— {val} —*** | | | |')
        else:
            s, i, r = val
            lines.append(f'| {label} | {s} | {i} | {r} |')
    return '\n'.join(lines)


def to_latex(rows):
    lines = [r'\begin{tabular}{lccc}', r'\toprule',
             r'Variant & Signal MAE (dB) & Interf. MAE (dB) & SE MAE (15 dB) \\',
             r'\midrule']
    for label, val in rows:
        if label == '__group__':
            lines.append(r'\midrule')
            lines.append(rf'\multicolumn{{4}}{{l}}{{\textit{{{val}}}}} \\')
        else:
            s, i, r = val
            tex = label.replace('1-p -> p', r'$1-p \rightarrow p$') \
                       .replace('1-p removed', r'$1-p$ removed') \
                       .replace('w/o', r'w/o')
            row = ' & '.join([tex] + [v.replace('±', r'$\pm$').replace('--', '--')
                                      for v in (s, i, r)])
            lines.append(row + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def build_rows():
    rows = []
    for group, entries in GROUPS:
        if group:
            rows.append(('__group__', group))
        for label, s, i, r in entries:
            rows.append((label, (cell(s), cell(i), cell(r))))
    return rows


if __name__ == '__main__':
    rows = build_rows()
    md = to_markdown(rows)
    print(md)
    with open('result/ablation_table.md', 'w') as f:
        f.write(md + '\n')
    with open('result/ablation_table.tex', 'w') as f:
        f.write(to_latex(rows) + '\n')
    print('\nsaved: result/ablation_table.md, result/ablation_table.tex')
