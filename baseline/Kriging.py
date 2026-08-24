# construct radio map for beam.py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
matplotlib.rcParams['mathtext.fontset'] = 'cm'
# 'Times New Roman' is absent on Katana; Nimbus Roman is URW's
# metric-compatible clone of it. Without this chain matplotlib falls back
# to DejaVu Sans and the figures come out in the wrong typeface entirely.
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Nimbus Roman',
                                     'DejaVu Serif']
font1, font2 = 24, 18

from main import BS_loc
from generate import UE_train, CSI_train

BS_loc = BS_loc[:, 0:2] # (12, 2)
UE_loc = UE_train[:, 0:2] # (2000, 2)

CSI_train = CSI_train.transpose(0,1,4,2,3) # (12, 2000, 12, 1, 32)
CSI_train_H = np.transpose(np.conj(CSI_train), (0,1,2,4,3)) # (12, 2000, 12, 32, 1)
signal_power = np.abs(CSI_train @ CSI_train_H) # (12, 2000, 12, 1, 1)
MRT_BF = CSI_train_H / (np.sqrt(signal_power) + 1e-20)
signal_amp = np.abs(CSI_train @ MRT_BF)
signal_amp_mean = np.mean(np.squeeze(signal_amp), -1) # (12, 2000)
signal_dB = 10 * np.log10(signal_amp_mean + 1e-20) # (12, 2000)

# discrete data
x = UE_loc[:,0]
y = UE_loc[:,1]
z = signal_dB[0,:]

OK = OrdinaryKriging(x, y, z, variogram_model='spherical',
        verbose=False, enable_plotting=False)

# grid
grid = 1
gridx = np.linspace(470, 760, int((760-470)/grid))
gridy = np.linspace(170, 350, int((350-170)/grid))
z_pred, ss = OK.execute('grid', gridx, gridy)

plt.figure(figsize=(8,6))
plt.imshow(z_pred, origin='lower', extent=(470,760,170,350))
plt.colorbar(label='Predicted Power (dB)')
plt.title('Kriging Interpolated Radio Map')
# plt.show()


Z_all = np.empty((BS_loc.shape[0], gridy.size, gridx.size), dtype=float)
V_all = np.empty_like(Z_all) 

# Kriging interpolation
for BS_idx in range(BS_loc.shape[0]):
    z = signal_amp_mean[BS_idx,:]
    OK = OrdinaryKriging(x, y, z, variogram_model='spherical',
        verbose=False, enable_plotting=False)
    
    z_pred, ss = OK.execute('grid', gridx, gridy)  # (Ny, Nx)
    Z_all[BS_idx] = np.asarray(z_pred)
    V_all[BS_idx] = np.asarray(ss)

# save
out_path = 'result/Kriging_map.npz'
np.savez_compressed(out_path, gridx=gridx, gridy=gridy, Z=Z_all, V=V_all)
