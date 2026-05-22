# generate train and test realization of cell-free networks
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
num_train = 8000
num_test = 2000

# training and test class
cf_train = CellFree(BS_loc, UE_train, CSI_train)
cf_test = CellFree(BS_loc, UE_test, CSI_test)

def generate_data(cf, num_samples, max_ap_num=12, min_ue_num=10,
                  max_ue_num=30, base_seed=0):

    ss_root = np.random.SeedSequence(base_seed)
    ss_child = ss_root.spawn(num_samples)

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
        signal, interf, rate = cf.calculate_sinr(A, P)

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


if __name__ == "__main__":
    
    # generate train data
    A_full_train, P_full_train, AP_num_train, UE_num_train, loc_train, A_train, P_train,\
        signal_train, interf_train, rate_train = generate_data(cf_train, num_train)

    # generate test data
    A_full_test, P_full_test, AP_num_test, UE_num_test, loc_test, A_test, P_test,\
        signal_test, interf_test, rate_test = generate_data(cf_test, num_test)
    
    # A_full_train = generate_data(cf_train, num_train)
    # A_full_test = generate_data(cf_test, num_test)

    # save train data
    np.save('data/train/A_full_train.npy', A_full_train)
    np.save('data/train/P_full_train.npy', P_full_train)
    np.save('data/train/AP_num_train.npy', AP_num_train)
    np.save('data/train/UE_num_train.npy', UE_num_train)
    np.save('data/train/loc_train.npy', loc_train)
    np.save('data/train/A_train.npy', A_train)
    np.save('data/train/P_train.npy', P_train)
    np.save('data/train/signal_train.npy', signal_train)
    np.save('data/train/interf_train.npy', interf_train)
    np.save('data/train/rate_train.npy', rate_train)

    # save test data
    np.save('data/test/A_full_test.npy', A_full_test)
    np.save('data/test/P_full_test.npy', P_full_test)
    np.save('data/test/AP_num_test.npy', AP_num_test)
    np.save('data/test/UE_num_test.npy', UE_num_test)
    np.save('data/test/loc_test.npy', loc_test)
    np.save('data/test/A_test.npy', A_test)
    np.save('data/test/P_test.npy', P_test)
    np.save('data/test/signal_test.npy', signal_test)
    np.save('data/test/interf_test.npy', interf_test)
    np.save('data/test/rate_test.npy', rate_test)
    
