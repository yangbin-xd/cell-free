# Table II: per-UE MAE of the signal, interference and SE (15 dB) predictions
# versus channel estimation error (E2 protocol: perfect-CSI model, imperfect
# test labels, inference only). Model = seed-0 full model (model/*_main_seed0*);
# arrays live in result/csi/nmse*/ (tau^2 = median serving-link NMSE), written
# by jobs/csi_nmse*.pbs -> ablation/csi_eval.py on data/CE_error/nmse*/.
#
#   python3 ablation/make_csi_table.py
import numpy as np

POINTS = [('nmse00', '0 (perfect)'), ('nmse01', '0.01'), ('nmse05', '0.05'),
          ('nmse10', '0.10'), ('nmse20', '0.20'), ('nmse30', '0.30')]
BRANCHES = ('signal', 'interf', 'rate')


def mae(tag, branch, mode):
    a = np.load(f'result/csi/{tag}/{branch}_{mode}_seed0.npy')
    return float(np.nanmean(np.abs(a)))


def block(mode):
    lines = ['| median tau^2 | Signal MAE (dB) | Interf MAE (dB) | '
             'SE MAE @15 dB (bits/s/Hz) |', '|---|---|---|---|']
    vals = {}
    for tag, label in POINTS:
        row = [mae(tag, b, mode) for b in BRANCHES]
        vals[tag] = row
        lines.append(f'| {label} | ' + ' | '.join(f'{v:.4f}' for v in row) + ' |')
    return '\n'.join(lines), vals


def rel_block(vals):
    base = vals['nmse00']
    lines = ['| median tau^2 | Signal | Interf | SE @15 dB |', '|---|---|---|---|']
    for tag, label in POINTS[1:]:
        rel = [(v / b - 1) * 100 for v, b in zip(vals[tag], base)]
        lines.append(f'| {label} | ' + ' | '.join(f'{r:+.1f}%' for r in rel) + ' |')
    return '\n'.join(lines)


def to_latex(vals):
    rows = []
    for tag, label in POINTS:
        v = vals[tag]
        label_tex = '$0$ (perfect)' if tag == 'nmse00' else f'${label}$'
        rows.append(f'        {label_tex} & {v[0]:.3f} & {v[1]:.3f} & {v[2]:.3f} \\\\')
    body = '\n'.join(rows)
    return (
        '\\begin{table}[t]\n'
        '    \\centering\n'
        '    \\caption{MAE of the signal, interference, and per-UE SE\n'
        '    predictions versus channel estimation error.}\n'
        '    \\label{tab:csi_error}\n'
        '    \\begin{tabular}{cccc}\n'
        '        \\toprule\n'
        '        Median $\\tau^2$ & Signal (dB) & Interference (dB) &'
        ' SE (bits/s/Hz) \\\\\n'
        '        \\midrule\n'
        f'{body}\n'
        '        \\bottomrule\n'
        '    \\end{tabular}\n'
        '\\end{table}'
    )


if __name__ == '__main__':
    asis_md, asis = block('asis')
    recal_md, recal = block('recal')

    md = (
        'E2 grid: perfect-CSI training, imperfect-CSI test '
        'labels.\n\nModel: main_seed0 (no epoch cap, early stopping only), '
        'trained on data/ (perfect CSI).\nOnly the TEST labels carry '
        'estimation error. Inference only -- no retraining.\nPer-UE MAE, '
        'seed 0. SE at 15 dB receive SNR. Datasets: data/CE_error/nmse01 ... '
        'data/CE_error/nmse30\n(tau^2 = median serving-link NMSE; the '
        'tau^2 = 0 row is data/ itself).\n\n'
        '## asis -- de-normalised with the training-set constants '
        '(the deployed checkpoint)\n\n' + asis_md +
        '\n\nRelative to perfect CSI:\n\n' + rel_block(asis) +
        '\n\n## recal -- de-normalised with the evaluation set\'s own '
        'constants\n\n' + recal_md +
        '\n\nRelative to perfect CSI:\n\n' + rel_block(recal) + '\n'
    )

    print(md)
    with open('result/csi_table.md', 'w') as f:
        f.write(md)
    with open('result/csi_table.tex', 'w') as f:
        f.write(to_latex(asis) + '\n')
    print('saved: result/csi_table.md, result/csi_table.tex')
