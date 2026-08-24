# data preprocess
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from generate import loc_mean, loc_std

# Single choke point for the dataset root, so an imperfect-CSI operating point
# can be swapped in without touching any of the ~30 training scripts:
#   CF_DATA_ROOT=data_pilot_tau0.052 python3 ablation/main_map.py --seed 0
# Unset (the default) loads exactly what it always did.
DATA_ROOT = os.environ.get('CF_DATA_ROOT', 'data')
if DATA_ROOT != 'data':
    print(f'[process] CF_DATA_ROOT={DATA_ROOT}')

def _load(split, name):
    return np.load(f'{DATA_ROOT}/{split}/{name}_{split}.npy')

# read data
AP_num_train = _load('train', 'AP_num') # (8000, )
UE_num_train = _load('train', 'UE_num') # (8000, )
loc_train = _load('train', 'loc') # (8000, max_ap_num+max_ue_num, 2)
A_train = _load('train', 'A') # (8000, 3, max_ue_num)
P_train = _load('train', 'P') # (8000, 3, max_ue_num)
signal_train = _load('train', 'signal') # (8000, max_ue_num)
interf_train = _load('train', 'interf') # (8000, max_ue_num)
rate_train = _load('train', 'rate') # (7, 8000, max_ue_num)

AP_num_test = _load('test', 'AP_num') # (2000, )
UE_num_test = _load('test', 'UE_num') # (2000, )
loc_test = _load('test', 'loc') # (2000, max_ap_num+max_ue_num, 2)
A_test = _load('test', 'A') # (2000, 3, max_ue_num)
P_test = _load('test', 'P') # (2000, 3, max_ue_num)
signal_test = _load('test', 'signal') # (2000, max_ue_num)
interf_test = _load('test', 'interf') # (2000, max_ue_num)
rate_test = _load('test', 'rate') # (7, 2000, max_ue_num)

# c7 = int(np.count_nonzero(AP_num_train == 7))
# c8 = int(np.count_nonzero(AP_num_train == 8))
# c9 = int(np.count_nonzero(AP_num_train == 9))
# c10 = int(np.count_nonzero(AP_num_train == 10))
# c11 = int(np.count_nonzero(AP_num_train == 11))
# c12 = int(np.count_nonzero(AP_num_train == 12))
# print("7:", c7, "8:", c8, "9:", c9, "10:", c10, "11:", c11, "12:", c12)

# numpy to tensor
AP_num_train = torch.from_numpy(AP_num_train).int()
UE_num_train = torch.from_numpy(UE_num_train).int()
loc_train = torch.from_numpy(loc_train).float()
A_train = torch.from_numpy(A_train).int()
P_train = torch.from_numpy(P_train).float()
signal_train = torch.from_numpy(signal_train).float()
interf_train = torch.from_numpy(interf_train).float()
rate_train = torch.from_numpy(rate_train).float()

AP_num_test = torch.from_numpy(AP_num_test).int()
UE_num_test = torch.from_numpy(UE_num_test).int()
loc_test = torch.from_numpy(loc_test).float()
A_test = torch.from_numpy(A_test).int()
P_test = torch.from_numpy(P_test).float()
signal_test = torch.from_numpy(signal_test).float()
interf_test = torch.from_numpy(interf_test).float()
rate_test = torch.from_numpy(rate_test).float()

# input normalize
loc_train_norm = (loc_train - loc_mean) / loc_std
loc_test_norm = (loc_test - loc_mean) / loc_std

# signal normalize
signal_reshape = signal_train.reshape(-1)
signal_valid = signal_reshape[signal_reshape != 0]
signal_mean = signal_valid.mean()
signal_std  = signal_valid.std()
signal_norm = (signal_train - signal_mean) / signal_std

# interf normalize
interf_reshape = interf_train.reshape(-1)
interf_valid = interf_reshape[interf_reshape != 0]
interf_mean = interf_valid.mean()
interf_std  = interf_valid.std()
interf_norm = (interf_train - interf_mean) / interf_std

# rate normalize
rate_reshape = rate_train.reshape([7,-1])
rate_valid = rate_reshape[rate_reshape != 0]
rate_mean = rate_valid.mean()
rate_std  = rate_valid.std()
rate_norm = (rate_train - rate_mean) / rate_std

# dataset
class RadioMapDataset(Dataset):
    def __init__(self, ap_num, ue_num, data_loc, data_A, data_P, data_value):
        self.ap_num = ap_num
        self.ue_num = ue_num
        self.loc = data_loc
        self.A = data_A
        self.P = data_P
        self.value = data_value
        
    def __len__(self):
        return len(self.loc)
    
    def __getitem__(self, idx):
        return self.ap_num[idx], self.ue_num[idx], self.loc[idx], self.A[idx],\
               self.P[idx], self.value[idx]

def split_train_val(data_norm, train_ratio=0.75, val_ratio=0.25):
    
    n_samples = len(loc_train)
    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)
    
    train_idx = np.arange(0, n_train)
    val_idx   = np.arange(n_train, n_train + n_val)
    
    train_dataset = RadioMapDataset(AP_num_train[train_idx], UE_num_train[train_idx], 
                                    loc_train_norm[train_idx], A_train[train_idx],
                                    P_train[train_idx], data_norm[train_idx])
    val_dataset = RadioMapDataset(AP_num_train[val_idx], UE_num_train[val_idx],
                                  loc_train_norm[val_idx], A_train[val_idx],
                                  P_train[val_idx], data_norm[val_idx])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    return train_loader, val_loader

# main
if __name__ == "__main__":
    
    print("loc_mean:", loc_mean)
    print("loc_std:", loc_std)
    print("signal mean:", signal_mean)
    print("signal std:", signal_std)
    print("interf mean:", interf_mean)
    print("interf std:", interf_std)