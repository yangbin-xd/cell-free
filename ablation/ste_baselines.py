# Paired comparison of Leaky STE (current config, ES seed-0 model) against
# the baselines on the first N test scenarios, per objective:
#   max_sum / fairness : original, WMMSE-ADMM, APG
#   max_min            : original, SOCP,       APG
#
#   CF_MODEL_STEM=main_seed0 PYTHONPATH=. python3 ablation/ste_baselines.py --n 40
import argparse
import os
import time

os.environ.setdefault('CF_MODEL_STEM', 'main_seed0')

import numpy as np

from compare import (load_scenario, recompute_true, rate_models, _renorm,
                     _device, _log_fairness, run_apg, run_wmmse_beta,
                     run_socp, run_power_only)
from joint_optimize import joint_optimize

SNR = 15


def run_ste(sc, objective):
    A_opt, P_opt, _, _ = joint_optimize(
        rate_models[SNR], sc['loc_norm'], sc['ap_num'], sc['ue_num'],
        sc['A'], SNR, _device, n_joint_iters=20, n_refine_iters=100,
        lr_a=0.1, lr_p=0.05, objective=objective)
    return recompute_true(sc['ue_csi'], sc['ap_orig'], A_opt,
                          _renorm(A_opt, P_opt), SNR)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--start', type=int, default=0,
                    help='first test scenario index (for chunked PBS runs)')
    ap.add_argument('--objective', default=None,
                    choices=['max_sum_rate', 'max_min', 'fairness'],
                    help='run a single objective (default: all three)')
    args = ap.parse_args()

    OBJ = {
        'max_sum_rate': ('sum', lambda r: float(r.sum())),
        'max_min':      ('min', lambda r: float(r.min())),
        'fairness':     ('pf',  _log_fairness),
    }
    if args.objective:
        OBJ = {args.objective: OBJ[args.objective]}

    for objective, (key, metric) in OBJ.items():
        oracle = 'SOCP' if objective == 'max_min' else 'WMMSE-ADMM'
        methods = ['original', oracle, 'APG', 'Power-Only', 'Leaky STE']
        res = {m: {'v': [], 't': []} for m in methods}

        for idx in range(args.start, args.start + args.n):
            sc = load_scenario(idx)

            r = recompute_true(sc['ue_csi'], sc['ap_orig'], sc['A'],
                               sc['P'], SNR)
            res['original']['v'].append(metric(r)); res['original']['t'].append(0.0)

            t0 = time.time()
            r, _ = (run_socp(sc, SNR, objective) if objective == 'max_min'
                    else run_wmmse_beta(sc, SNR, objective))
            res[oracle]['v'].append(metric(r)); res[oracle]['t'].append(time.time() - t0)

            t0 = time.time()
            r, _ = run_apg(sc, SNR, objective)
            res['APG']['v'].append(metric(r)); res['APG']['t'].append(time.time() - t0)

            t0 = time.time()
            r, _ = run_power_only(sc, SNR, objective)
            res['Power-Only']['v'].append(metric(r)); res['Power-Only']['t'].append(time.time() - t0)

            t0 = time.time()
            r = run_ste(sc, objective)
            res['Leaky STE']['v'].append(metric(r)); res['Leaky STE']['t'].append(time.time() - t0)

            if (idx + 1 - args.start) % 10 == 0:
                print(f'[{objective}] {idx + 1 - args.start}/{args.n} done',
                      flush=True)

        ste = np.array(res['Leaky STE']['v'])
        suffix = (f'{args.start}_{args.n}' if args.start else f'{args.n}')
        np.savez(f'result/ste_baselines_{objective}_{suffix}.npz',
                 **{m: np.array(res[m]['v']) for m in methods},
                 **{f'{m}_time': np.array(res[m]['t']) for m in methods})
        print(f'\n=== {objective} ({args.n} scenarios, metric: {key}) ===')
        print(f'{"method":12s} {"mean":>8s} {"median":>8s} {"STE-wins":>9s} {"time(s)":>8s}')
        for m in methods:
            v = np.array(res[m]['v']); t = np.array(res[m]['t'])
            win = f'{int((ste > v).sum()):2d}/{args.n}' if m != 'Leaky STE' else '--'
            print(f'{m:12s} {v.mean():8.3f} {np.median(v):8.3f} {win:>9s} {t.mean():8.3f}')
