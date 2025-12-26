# main
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'
font1, font2 = 22, 18
np.set_printoptions(precision=3, suppress=True, linewidth=100)

# read data
BS_loc = np.load('data/BS_loc.npy') # (12, 3)
UE_loc = np.load('data/UE_loc.npy') # (2500, 3)
CSI = np.load('data/CSI.npy') # (12, 2500, 1, 32, 12)

# change x-y axis in DeepMIMO
BS_loc = BS_loc[:, [1,0,2]].astype(np.float32)
UE_loc = UE_loc[:, [1,0,2]].astype(np.float32)

# cell free system
class CellFree:
    def __init__(self, BS_loc, UE_loc, CSI):
        self.BS_loc = BS_loc[:, :2] # (3, 2)
        self.UE_loc = UE_loc[:, :2] # (N, 2)
        self.CSI = CSI.transpose(0,1,4,3,2) # (3, N, 12, 32, 1)
        self.L, self.N, self.Nr, self.Nt, self.Nc = CSI.shape
        
    def random_choose(self, K=10, seed=3):
        np.random.seed(seed)
        self.K = K
        select_index = np.random.choice(self.N, size=self.K, replace=False)
        self.loc_select = self.UE_loc[select_index,:]
        self.CSI_select = self.CSI[:,select_index,:]
        gain = np.abs(self.CSI_select[..., 0]) ** 2
        self.gain = np.mean(gain, axis=(2,3)) # (L, K)

    ''' AP selection ''' 
    # UE select top l APs to serve itself
    def select_AP(self, top_l=2):
        A = np.zeros((self.L, self.K))
        for k in range(self.K):
            best_APs = np.argsort(self.gain[:, k])[-top_l:]
            A[best_APs, k] = 1
        return A
    
    # AP select top k UEs to serve
    def select_UE(self, max_k=4):
        # choose max k UEs for service
        A = np.zeros((self.L, self.K))
        for l in range(self.L):
            num_serve = min(self.K, max_k)
            serve_ue = np.argsort(self.gain[l, :])[-num_serve:]
            A[l, serve_ue] = 1
        return A
    
    # bilateral selection
    # 1) UE select the best AP to serve itself
    # 2) the best AP invite other APs to join
    # 3) other APs accept/decline
    def bilateral_select(self, max_k=4, compete=True, guarantee=True):
        # max_k: maximum serving UEs per AP
        # UE compete for service
        # guarantee every UE been served
        A = np.zeros((self.L, self.K))
        ap_load = np.zeros(self.L)

        # Step 1: each UE selects master AP with capacity constraints
        for k in range(self.K):
            # sort APs by channel gain for UE k
            ap_order = np.argsort(self.gain[:, k])[::-1]

            # find first AP with available capacity
            for ap in ap_order:
                if ap_load[ap] < max_k:
                    A[ap, k] = 1
                    ap_load[ap] += 1
                    break

        # Steps 2-3: neighbors make local decisions with competition
        for k in range(self.K):
            for l in range(self.L):
                if A[l, k] == 1: # skip if already serving (master AP)
                    continue
                
                # check if channel quality is good enough
                threshold = np.median(self.gain[l, :]) * 0.8
                if self.gain[l, k] <= threshold:
                    continue # Channel too weak, skip
                
                if ap_load[l] < max_k:
                    # has capacity, directly accept
                    A[l, k] = 1
                    ap_load[l] += 1
                elif compete:
                    # at capacity, compete with existing UEs
                    served_ues = np.where(A[l, :] == 1)[0]
                    weakest_ue = served_ues[np.argmin(self.gain[l, served_ues])]

                    # if new UE has better channel, replace the weakest
                    if self.gain[l, k] > self.gain[l, weakest_ue]:
                        A[l, weakest_ue] = 0
                        A[l, k] = 1

        if compete and guarantee:
        # find UEs that are not served by any AP
            unserved_ues = []
            for k in range(self.K):
                if np.sum(A[:, k]) == 0:  # UE k is not served by any AP
                    unserved_ues.append(k)

            # for each unserved UE, force the best AP to serve it
            for k in unserved_ues:
                # find the best AP for this UE
                best_ap = np.argmax(self.gain[:, k])

                if ap_load[best_ap] < max_k:
                    # best AP has capacity, directly serve
                    A[best_ap, k] = 1
                    ap_load[best_ap] += 1
                else:
                    # best AP is at capacity, need to disconnect a UE
                    served_ues = np.where(A[best_ap, :] == 1)[0]

                    # find UEs that are served by multiple APs (not uniquely served)
                    non_unique_ues = []
                    for ue in served_ues:
                        if np.sum(A[:, ue]) > 1: # served by more than one AP
                            non_unique_ues.append(ue)

                    if len(non_unique_ues) > 0:
                        # disconnect the weakest UE among non-uniquely served UEs
                        weakest_non_unique = non_unique_ues[np.argmin(self.gain[best_ap, non_unique_ues])]
                        A[best_ap, weakest_non_unique] = 0
                        ap_load[best_ap] -= 1

                    A[best_ap, k] = 1
                    ap_load[best_ap] += 1
        return A
    
    # random selection satify three requirements
    # 1) each UE is served by at least one AP
    # 2) the number of UE served by each 1 <= AP <= max_k
    # 3) connection probability <= p
    def random_selection(self, p, max_k=4):
        A = np.zeros((self.L, self.K), dtype=int)

        cap = np.full(self.L, max_k)
        for c in range(self.K):
            rows_can = np.where(cap > 0)[0]
            r = np.random.choice(rows_can)
            A[r, c] = 1
            cap[r] -= 1

        mask = (A == 0)
        proposed = (np.random.rand(self.L, self.K) < p) & mask

        for r in range(self.L):
            idx = np.where(proposed[r])[0]
            if idx.size > cap[r]:
                keep = np.random.choice(idx, size=cap[r], replace=False)
                row_add = np.zeros_like(proposed[r], dtype=bool)
                row_add[keep] = True
                proposed[r] = row_add

        A |= proposed.astype(int)
        return A
    
    # improved random selection satify one more requirements
    # UE can onlye select AP from top_l nearest APs
    def improved_random_selection(self, p, top_l=4, max_k=8):

        A = np.zeros((self.L, self.K), dtype=int)

        # calculate distance
        diff = self.BS_loc[:, None, :] - self.loc_select[None, :, :]  # (L,K,2)
        dist2 = np.sum(diff * diff, axis=-1) # (L,K)
        idx = np.argpartition(dist2, kth=top_l-1, axis=0)[:top_l, :] # (top_l,K)
        nearest_mask = np.zeros((self.L, self.K), dtype=bool)
        cols = np.arange(self.K)
        nearest_mask[idx, cols] = True

        # AP capacity
        cap = np.full(self.L, max_k, dtype=int)

        # Step 1: each UE is first served by one nearest AP
        for c in range(self.K):
            # choose from top_l APs
            ap_cands = idx[:, c] # (top_l,)
            order = ap_cands[np.argsort(dist2[ap_cands, c])]
            chosen = None
            for r in order:
                if cap[r] > 0:
                    chosen = r
                    break
            if chosen is None:
                non_top = np.where((~nearest_mask[:, c]) & (cap > 0))[0]
                if non_top.size > 0:
                    r = non_top[np.argmin(dist2[non_top, c])]
                    chosen = r
            if chosen is not None:
                A[chosen, c] = 1
                cap[chosen] -= 1

        # Step 2: add served APs according to p
        total_slots = 4 * self.K
        target_edges = int(np.ceil(p * total_slots))

        if A.sum() >= target_edges:
            return A

        while A.sum() < target_edges:
            # add edge
            cand_mask = nearest_mask & (A == 0)
            if np.any(cap <= 0):
                cand_mask[cap <= 0, :] = False

            r_idx, c_idx = np.where(cand_mask)
            if r_idx.size == 0:
                break

            i = np.random.randint(r_idx.size)
            r, c = r_idx[i], c_idx[i]
            if cap[r] > 0 and A[r, c] == 0:
                A[r, c] = 1
                cap[r] -= 1

        return A


    ''' power allocation '''
    # average power allocation
    def average_power(self, A):
        P = np.zeros_like(A, dtype=float)
        for l in range(self.L):
            num_served = np.sum(A[l, :])
            if num_served > 0:
                P[l, :] = A[l, :] / num_served
        return P
    
    # random power allocation
    def random_power(self, A, seed):
        rng = np.random.default_rng(seed)
        A = A.astype(bool) 
        P = np.zeros_like(A, dtype=float)
        for l in range(A.shape[0]):
            idx = np.where(A[l])[0]
            k = idx.size
            if k > 0:
                w = rng.dirichlet(np.full(k, 1))
                P[l, idx] = w
        return P
    
    # calculate SINR
    def calculate_sinr(self, A, P):
        SNR = np.arange(0,31,5)
        signal_power = np.zeros((self.K, self.Nc))
        interf_power = np.zeros((self.K, self.Nc))
        # interf_temp = np.zeros((self.L, self.K, self.Nc)) # add
        SINR = np.zeros((SNR.size, self.K, self.Nc))
        rate = np.zeros((SNR.size, self.K, self.Nc))
        for k in range(self.K):
            for nc in range(self.Nc):
                
                # calculate signal
                signal = 0
                for l in range(self.L):
                    h_lk = self.CSI_select[l, k, nc, :] # (32, 1)
                    if A[l, k] == 1:    
                        w_lk = h_lk / (np.linalg.norm(h_lk))
                        signal += np.sqrt(P[l, k]) * \
                                  np.transpose(h_lk.conjugate()) @ w_lk
                
                # calculate interference
                interf_pow = 0
                for m in range(self.K):
                    if m == k:
                        continue

                    interf = 0
                    for l in range(self.L):
                        if A[l, m] == 1:
                            h_lk = self.CSI_select[l, k, nc, :]
                            h_lm = self.CSI_select[l, m, nc, :]
                            w_lm = h_lm / (np.linalg.norm(h_lm))
                            interf += np.sqrt(P[l, m]) * np.transpose(h_lk.conjugate()) @ w_lm

                    interf_pow += np.abs(interf) ** 2

                signal_power[k, nc] = np.abs(signal) ** 2
                interf_power[k, nc] = interf_pow

                for i, snr in enumerate(SNR):
                    noise_dB = -87 - snr
                    noise_power = 10 ** (noise_dB / 10)
                    SINR[i, k, nc] = signal_power[k, nc] / (interf_power[k, nc] + noise_power)
                    rate[i, k, nc] = np.log2(1 + SINR[i, k, nc])

        signal_power_mean = 10 * np.log10(np.mean(signal_power, -1))
        interf_power_mean = 10 * np.log10(np.mean(interf_power, -1))
        rate_mean = np.mean(rate, -1)

        return signal_power_mean, interf_power_mean, rate_mean
    
def plot_result(BS_loc, UE_loc, A, P):
    # plot resutls
    fig, ax = plt.subplots(figsize=(10, 8))
    # plot BS
    for i, (x, y) in enumerate(BS_loc):
        plt.scatter(x, y, c='#F65314', marker='^', s=200, label='AP'
                    if i == 0 else "")
        plt.text(x+10, y-13, f'AP {i}', fontsize=font2, color='black')
    # plot UE
    for i, (x, y) in enumerate(UE_loc):
        plt.scatter(x, y, c='#7CBB00', marker='o', s=100, label='UE'
                    if i == 0 else "")
        plt.text(x+10, y-12, f'UE {i}', fontsize=font2, color='black')

    # plot edge
    for l in range(BS_loc.shape[0]):
        for k in range(UE_loc.shape[0]):
            if A[l, k] == 1:
                x1, y1 = BS_loc[l]
                x2, y2 = UE_loc[k]
                plt.plot([x1, x2], [y1, y2], 'gray', linestyle='--', linewidth=2)
                # edge attr
                xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                dx, dy = x2 - x1, y2 - y1
                norm = np.hypot(dx, dy)
                offset = 5
                ox, oy = (-dy / norm) * offset, (dx / norm) * offset
                val = P[l, k]
                ax.text(xm + ox, ym + oy, f'{val:.2f}', fontsize=font1/2, 
                        ha='center', va='center', color='gray')
                        # path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    
    # plt.legend(loc='upper right', fontsize=font2)
    plt.xlabel('X (m)', fontsize=font1)
    plt.ylabel('Y (m)', fontsize=font1)
    plt.xlim(460, 770)
    plt.ylim(150, 350)
    plt.xticks(np.arange(500, 750+1, 50), fontsize=font1)
    plt.yticks(np.arange(150, 350+1, 50), fontsize=font1)
    ax.set_aspect(1)
    ax.invert_xaxis()
    plt.tight_layout()

# main
if __name__ == '__main__':
    
    cf = CellFree(BS_loc, UE_loc, CSI)
    cf.random_choose(K=15, seed=0)

    # access point selection
    # A = cf.select_AP(top_l=2)
    # A = cf.select_UE(max_k=4)
    # A = cf.bilateral_select(max_k=4, compete=False, guarantee=False)
    # A = cf.bilateral_select(max_k=4, compete=True, guarantee=False)
    # A = cf.bilateral_select(max_k=4, compete=True, guarantee=True)
    # A = cf.random_selection(p=0.2, max_k=4)
    A = cf.improved_random_selection(p=0.8, top_l=4, max_k=8)
    print("Association matrix:\n", A)

    # power allocation
    P = cf.average_power(A)
    # P = cf.random_power(A, seed=0)
    print("Power matrix:\n", np.round(P, 3))

    # calculate rate
    signal, interf, rate = cf.calculate_sinr(A, P)
    snr = 30
    rate = rate[int(snr/5)]
    
    np.set_printoptions(
    formatter={'float_kind': '{:.3f}'.format}, linewidth=100)

    print('signal', signal, 'dB')
    print('interf', interf, 'dB')
    print('rate:', rate[-1], 'bits/s/Hz')
    print('average rate:', np.round(np.mean(rate), 3), 'bits/s/Hz')
    print('minimum rate:', np.round(np.min(rate), 3), 'bits/s/Hz')

    plot_result(cf.BS_loc, cf.loc_select, A, P)
    plt.show()