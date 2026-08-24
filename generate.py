# generate train and test realization of cell-free networks
import argparse
import os
from main import *
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# split training and test set
idx = np.arange(UE_loc.shape[0])
idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=0, shuffle=True)
UE_train, UE_test = UE_loc[idx_train], UE_loc[idx_test]
CSI_train, CSI_test = CSI[:, idx_train, ...], CSI[:, idx_test, ...]

# input normalize
loc_conc = np.concatenate([BS_loc, UE_train], axis=0)
loc_mean = loc_conc[:,:2].mean(axis=0)
loc_std = loc_conc[:,:2].std(axis=0)

# training and test samples
num_train = 10000
num_test = 2000

# training and test class
cf_train = CellFree(BS_loc, UE_train, CSI_train)
cf_test = CellFree(BS_loc, UE_test, CSI_test)

def generate_data(cf, num_samples, max_ap_num=12, min_ue_num=10,
                  max_ue_num=30, base_seed=0,
                  pilot_snr=np.inf, mc=1, err_stream=0):
    """Realizations of the network plus their signal / interference / rate labels.

    pilot_snr / mc are forwarded to CellFree.calculate_sinr; the defaults
    (inf, 1) are the perfect-CSI setting that produced the committed datasets.

    The estimation-error draws come from a SeedSequence keyed on err_stream and
    kept SEPARATE from ss_child, which seeds the network structure. That
    separation is what makes loc/A/P/AP_num/UE_num bit-identical across pilot
    SNRs -- only the labels move, which is the whole claim being tested. Train
    and test must pass different err_stream values so the test realization is
    independent of anything the model saw.
    """
    ss_root = np.random.SeedSequence(base_seed)
    ss_child = ss_root.spawn(num_samples)
    ss_err = np.random.SeedSequence([base_seed, err_stream]).spawn(num_samples)

    A_full = np.zeros([num_samples, max_ap_num, max_ue_num], dtype=np.int8)
    P_full = np.zeros([num_samples, max_ap_num, max_ue_num], dtype=np.float32)
    AP_num = np.zeros(num_samples, dtype=np.int32)
    UE_num = np.zeros(num_samples, dtype=np.int32)
    loc_set = np.zeros([num_samples, max_ue_num + max_ap_num, 2], dtype=np.float32)
    A_set = np.zeros([num_samples, max_ap_num, max_ue_num], dtype=np.int8)
    P_set = np.zeros([num_samples, max_ap_num, max_ue_num], dtype=np.float32)
    signal_set = np.zeros([num_samples, max_ue_num], dtype=np.float32)
    interf_set = np.zeros([num_samples, max_ue_num], dtype=np.float32)
    rate_set = np.zeros([7, num_samples, max_ue_num], dtype=np.float32)
    
    iterable = range(num_samples)
    iterable = tqdm(iterable, desc="Generating samples", unit="sample")

    for i in iterable:
        rng = np.random.default_rng(ss_child[i])
        seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))

        ue_num = rng.integers(min_ue_num, max_ue_num + 1)
        cf.random_choose(ue_num, seed)

        # access point selection
        # p = 0.5
        p = rng.uniform(0.5, 1.0)

        # A = cf.random_selection(p, max_k=max_k)
        A = cf.improved_random_selection(p, top_l=4, max_k=8)
        ap_num = int((A.sum(axis=1) > 0).sum())

        # power allocation
        P = cf.random_power(A, seed)

        # calculate rate
        signal, interf, rate = cf.calculate_sinr(
            A, P, pilot_snr=pilot_snr, mc=mc,
            rng=np.random.default_rng(ss_err[i]))

        UE_num[i] = ue_num
        AP_num[i] = ap_num

        loc_set[i,:ap_num+ue_num,:] = \
            np.concatenate([cf.BS_loc[np.where(A.sum(axis=1) > 0)[0]], cf.loc_select], 0)

        A_full[i, :, :ue_num] = A
        P_full[i, :, :ue_num] = P
        A_set[i,:ap_num,:ue_num] = A[A.sum(axis=1) > 0]
        P_set[i,:ap_num,:ue_num] = P[P.sum(axis=1) > 0]
        signal_set[i,:ue_num] = signal
        interf_set[i,:ue_num] = interf
        rate_set[:,i,:ue_num] = rate
        
    return A_full, P_full, AP_num, UE_num, loc_set, A_set, P_set, signal_set,\
           interf_set, rate_set


FIELDS = ('A_full', 'P_full', 'AP_num', 'UE_num', 'loc', 'A', 'P',
          'signal', 'interf', 'rate')
# Everything except the last three is network structure, which must not depend
# on the pilot SNR -- see check_inputs_unchanged.
INPUT_FIELDS = FIELDS[:-3]


def check_inputs_unchanged(arrays, split, ref_root='data'):
    """Assert the structural arrays match the committed perfect-CSI dataset.

    Only signal/interf/rate may differ between operating points. If an input
    array moved, the pilot-SNR comparison is no longer single-factor and every
    conclusion drawn from it is void -- so fail loudly rather than train on it.
    """
    for name in INPUT_FIELDS:
        path = f'{ref_root}/{split}/{name}_{split}.npy'
        if not os.path.exists(path):
            print(f'  [skip] no reference at {path}')
            continue
        ref = np.load(path)
        got = arrays[name]
        if not np.array_equal(ref, got):
            raise SystemExit(
                f'FATAL: {name}_{split} differs from {path}. Network structure '
                f'must be identical across pilot SNRs; only the labels may move.')
    print(f'  [ok] {split}: all {len(INPUT_FIELDS)} structural arrays match {ref_root}/')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot-snr', type=float, default=float('inf'),
                    help='uplink pilot SNR rho_bar_p = tau_p*rho_p/sigma^2. '
                         'inf (default) = perfect CSI, reproduces the committed '
                         'datasets exactly. Nominal operating point is 3.17e11 '
                         '(tau_p=K orthogonal pilots, rho_p=1, 15 dB design '
                         'point), which gives median serving-link tau^2 = 0.052.')
    ap.add_argument('--mc', type=int, default=1,
                    help='estimation-error draws to average per subcarrier. 1 is '
                         'correct for the deployment-realistic label (a real '
                         'system measures one realization); raise only if the '
                         'measured label spread turns out to matter.')
    ap.add_argument('--out-root', default='data',
                    help='dataset root to write, e.g. data_pilot_tau0.052')
    args = ap.parse_args()

    if num_train != 10000:
        raise SystemExit(f'num_train={num_train}, expected 10000. A truncated '
                         f'dataset silently invalidates every retrained model.')

    print(f'pilot_snr={args.pilot_snr:.4g}  mc={args.mc}  -> {args.out_root}/')

    # err_stream differs between splits so the test labels' estimation-error
    # realization is independent of the training one.
    train = dict(zip(FIELDS, generate_data(
        cf_train, num_train, pilot_snr=args.pilot_snr, mc=args.mc, err_stream=1001)))
    test = dict(zip(FIELDS, generate_data(
        cf_test, num_test, pilot_snr=args.pilot_snr, mc=args.mc, err_stream=1002)))

    check_inputs_unchanged(train, 'train')
    check_inputs_unchanged(test, 'test')

    for split, arrays in (('train', train), ('test', test)):
        outdir = f'{args.out_root}/{split}'
        os.makedirs(outdir, exist_ok=True)
        for name in FIELDS:
            np.save(f'{outdir}/{name}_{split}.npy', arrays[name])
        print(f'  wrote {len(FIELDS)} arrays to {outdir}/')
    