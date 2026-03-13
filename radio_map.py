# construct radio map for query.py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'
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

# query
def query_mean_amp(xy_test, UE_train_xy, signal_amp_mean, cell_small=5.0, cell_big=10.0):
    
    xy_test = np.asarray(xy_test, float)
    xs_t, ys_t = xy_test[:, 0], xy_test[:, 1]

    xs_tr, ys_tr = UE_train_xy[:, 0], UE_train_xy[:, 1]
    vals = np.asarray(signal_amp_mean, float)

    ox, oy = (0.0, 0.0)

    def cell_bounds_from_origin(coord, cell, o):
        k = np.floor((coord - o) / cell)
        c0 = o + k * cell
        return c0, c0 + cell   # [c0, c0+cell)

    out_amp = np.full(xs_t.shape, np.nan, float)

    for i in range(xs_t.shape[0]):
        x0, x1 = cell_bounds_from_origin(xs_t[i], cell_small, ox)
        y0, y1 = cell_bounds_from_origin(ys_t[i], cell_small, oy)
        m = (xs_tr >= x0) & (xs_tr < x1) & (ys_tr >= y0) & (ys_tr < y1)
        if np.any(m):
            out_amp[i] = np.mean(vals[m])
            continue

        X0, X1 = cell_bounds_from_origin(xs_t[i], cell_big, ox)
        Y0, Y1 = cell_bounds_from_origin(ys_t[i], cell_big, oy)
        M = (xs_tr >= X0) & (xs_tr < X1) & (ys_tr >= Y0) & (ys_tr < Y1)
        if np.any(M):
            out_amp[i] = np.mean(vals[M])
            continue

    out_dB = 10 * np.log10(out_amp + 1e-20)
    return out_amp, out_dB

if __name__ == '__main__':

    BS_idx = 0
    # plot single AP
    fig,ax = plt.subplots(figsize=(10,6))
    ax.scatter(BS_loc[BS_idx, 0], BS_loc[BS_idx, 1], marker='^', c='#F65314', s = 50,
               label='BS')
    sc=ax.scatter(UE_loc[:, 0], UE_loc[:, 1], marker='.', c=signal_dB[BS_idx],
                  s = 50, label='UE')
    ax.set_aspect(1)
    ax.invert_xaxis()
    # plt.legend(loc='upper right', fontsize=font1)
    plt.xlabel('X (m)', fontsize=font1)
    plt.ylabel('Y (m)', fontsize=font1)
    plt.xticks(fontsize=font2)
    plt.yticks(fontsize=font2)

    cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label('Signal amplitude (dB)', fontsize=font1)
    cbar.ax.tick_params(labelsize=font2)
    plt.tight_layout()


    # plot all APs
    fig, axes = plt.subplots(6, 2, figsize=(10, 18), constrained_layout=True)
    for i in range(12):
        ax = axes.flat[i]
        ax.scatter(BS_loc[i, 0], BS_loc[i, 1], marker='^', c='#F65314', s=50, label='BS')
        sc = ax.scatter(UE_loc[:, 0], UE_loc[:, 1], marker='.', c=signal_dB[i],
                        s=20, vmin=np.nanmin(signal_dB), vmax=np.nanmax(signal_dB))
        ax.set_aspect(1)
        ax.invert_xaxis()
        ax.set_title(f'BS {i}', fontsize=font2)
        ax.set_xlabel('X (m)', fontsize=font2)
        ax.set_ylabel('Y (m)', fontsize=font2)
        ax.tick_params(labelsize=font2)

    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), pad=0.02, fraction=0.03)
    cbar.set_label('Signal amplitude (dB)', fontsize=font1)
    cbar.ax.tick_params(labelsize=font2)


    # grid-based radio map
    grid1 = 5.0
    grid2 = 10.0

    x_min, x_max = 600, 635
    y_min, y_max = 240, 280

    x = UE_loc[:, 0]
    y = UE_loc[:, 1]
    val = signal_amp_mean[BS_idx]
    mask_range = ((x >= x_min) & (x <= x_max)) | ((y >= y_min) & (y <= y_max))
    x, y, val = x[mask_range], y[mask_range], val[mask_range]

    # generate cell
    x_edges = np.arange(x.min(), x.max() + grid1, grid1)
    y_edges = np.arange(y.min(), y.max() + grid1, grid1)
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    ix = np.searchsorted(x_edges, x, side='right') - 1
    iy = np.searchsorted(y_edges, y, side='right') - 1

    sum_grid   = np.zeros((ny, nx), dtype=float)
    count_grid = np.zeros((ny, nx), dtype=int)
    np.add.at(sum_grid,   (iy, ix), val)
    np.add.at(count_grid, (iy, ix), 1)

    # generate big cell
    x_edges_big = np.arange(x.min(), x.max() + grid2, grid2)
    y_edges_big = np.arange(y.min(), y.max() + grid2, grid2)
    nx_big = len(x_edges_big) - 1
    ny_big = len(y_edges_big) - 1

    ix_big = np.searchsorted(x_edges_big, x, side='right') - 1
    iy_big = np.searchsorted(y_edges_big, y, side='right') - 1

    sum_grid_big   = np.zeros((ny_big, nx_big), dtype=float)
    count_grid_big = np.zeros((ny_big, nx_big), dtype=int)
    np.add.at(sum_grid_big,   (iy_big, ix_big), val)
    np.add.at(count_grid_big, (iy_big, ix_big), 1)

    mean_grid_W = np.divide(sum_grid, count_grid, out=np.zeros_like(sum_grid),
                            where=count_grid>0)
    for i in range(ny):
        for j in range(nx):
            if count_grid[i, j] == 0:
                x_center = x_edges[j] + grid1/2
                y_center = y_edges[i] + grid1/2
                ix2 = np.searchsorted(x_edges_big, x_center, side='right') - 1
                iy2 = np.searchsorted(y_edges_big, y_center, side='right') - 1
                if (0 <= ix2 < nx_big) and (0 <= iy2 < ny_big) and \
                    (count_grid_big[iy2, ix2] > 0):
                    mean_grid_W[i, j] = sum_grid_big[iy2, ix2] / count_grid_big[iy2, ix2]

    mean_grid_dB = 10 * np.log10(mean_grid_W + 1e-20)
    # mean_grid_dB[mean_grid_dB <= -199.0] = np.nan

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    pcm = ax.pcolormesh(x_edges, y_edges, mean_grid_dB, shading='flat')
    ax.scatter(BS_loc[BS_idx, 0], BS_loc[BS_idx, 1], marker='^', c='#F65314', s=60, 
               label=f'BS {BS_idx}')
    ax.set_aspect(1)
    ax.invert_xaxis()
    ax.set_xlabel('X (m)', fontsize=font1)
    ax.set_ylabel('Y (m)', fontsize=font1)
    ax.tick_params(labelsize=font2)

    cbar = plt.colorbar(pcm, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label('Signal amplitude (dB)', fontsize=font1)
    cbar.ax.tick_params(labelsize=font2)

    plt.tight_layout()
    plt.show()