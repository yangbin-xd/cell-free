# show scenario
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
# 'Times New Roman' is absent on Katana; Nimbus Roman is URW's
# metric-compatible clone of it. Without this chain matplotlib falls back
# to DejaVu Sans and the figures come out in the wrong typeface entirely.
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Nimbus Roman',
                                     'DejaVu Serif']
font1 = 20

from main import BS_loc, UE_loc
BS_loc = BS_loc[:,0:2]
UE_loc = UE_loc[:,0:2]
from sklearn.model_selection import train_test_split

idx = np.arange(UE_loc.shape[0])
idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=0, shuffle=True)
UE_train, UE_test = UE_loc[idx_train], UE_loc[idx_test]

# show scenario
fig,ax = plt.subplots(figsize=(10, 8))
# bs = ax.scatter(BS_loc[:, 0], BS_loc[:, 1], marker='^', c='#F65314', s = 100,
#                 label=f'BS')
# ue = ax.scatter(UE_loc[:, 0], UE_loc[:, 1], marker='.', c='#7CBB00', s = 50,
#                 label='UE')

# # plot BS
for i, (x, y) in enumerate(BS_loc):
        plt.scatter(x, y, c='#F65314', marker='^', s=100, label='AP'
                    if i == 0 else "")
        # plt.text(x+7, y-10, f'AP {i}', fontsize=font1, color='black')
# plot UE
# for i, (x, y) in enumerate(UE_loc):
#     plt.scatter(x, y, c='#7CBB00', marker='.', s=50, label='UE'
#                 if i == 0 else "")

ue = ax.scatter(UE_loc[:, 0], UE_loc[:, 1], marker='.', c='#7CBB00', s = 50,
                label='UE')
# ue = ax.scatter(UE_test[:, 0], UE_test[:, 1], marker='.', c='#FFBB00', s = 50,
#                 label='UE test')
        
ax.set_aspect(1)
plt.legend(fontsize=font1-2, loc=(0.015,0.425), frameon=True, borderpad=0.3,
           labelspacing=0.3, handletextpad=0.3, handlelength=1.0)
plt.xlabel('X (m)', fontsize=font1)
plt.ylabel('Y (m)', fontsize=font1)
plt.xlim(450, 800)
plt.ylim(160, 360)
plt.xticks(np.arange(450, 800+1, 50), fontsize=font1)
plt.yticks(np.arange(160, 360+1, 40), fontsize=font1)
ax.invert_xaxis()
plt.tight_layout()
plt.savefig('result/scenario.svg')
plt.show()
