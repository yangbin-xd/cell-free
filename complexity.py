# computational complexity comparison for R3-7:
# closed-form analytical model (large-scale gains) vs the proposed GNN radio map
# - analytical: time the closed-form S/I/R evaluation itself (in deployment the
#   CC receives g_{l,k} directly; the RNG replay below only reconstructs g for
#   the stored test set and is excluded from timing)
# - GNN: full inference path = create_edges + signal model + interference model
#   + rate combination, per network snapshot, batch size 1 (same as demo.py)
import time
import numpy as np
import torch
from generate import cf_test, num_test
from process import AP_num_test, UE_num_test, loc_test_norm
from process import A_test, P_test
from rate_map import RateModel

N_ant = 32
snr = 15
N_TIME = 200 # samples used for timing
N_WARM = 10

# ---------- reconstruct g for the timed samples (not part of timing) ----------
A_full_test = np.load('data/test/A_full_test.npy')
P_full_test = np.load('data/test/P_full_test.npy')

ss_child = np.random.SeedSequence(0).spawn(num_test)
g_list, A_list, P_list = [], [], []
for i in range(N_TIME):
    rng = np.random.default_rng(ss_child[i])
    seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
    ue_num = int(rng.integers(10, 31))
    cf_test.random_choose(ue_num, seed)
    g_list.append(cf_test.gain.copy())
    A_list.append(A_full_test[i, :, :ue_num].astype(float))
    P_list.append(P_full_test[i, :, :ue_num])

noise_power = 10 ** ((-87 - snr) / 10)

# ---------- closed-form analytical ----------
def analytical_rate(g, A, P):
    signal = N_ant * ((A * np.sqrt(P * g)).sum(axis=0)) ** 2
    interf = (g * (P.sum(axis=1, keepdims=True) - P)).sum(axis=0)
    return np.log2(1 + signal / (interf + noise_power))

for i in range(N_WARM):
    analytical_rate(g_list[i], A_list[i], P_list[i])
t0 = time.perf_counter()
for i in range(N_TIME):
    analytical_rate(g_list[i], A_list[i], P_list[i])
t_analytical = (time.perf_counter() - t0) / N_TIME

# ---------- proposed GNN ----------
# Proposed model = seed-0 full model (Table I "Proposed"; see demo.py). The
# 15 dB rate head is reused at every SNR (snr enters RateModel only through
# noise_dB), so no per-SNR checkpoint is needed.
_STEM = 'main_seed0'
model = RateModel(signal_model_path=f'model/signal_{_STEM}.pth',
                  interf_model_path=f'model/interf_{_STEM}.pth', snr=snr)
model.load_state_dict(torch.load(f'model/rate_{_STEM}_15dB.pth',
                                 weights_only=True, map_location='cpu'))
model.eval()

def gnn_timing(device):
    # create_edges in signal_map/interf_map hardcodes cuda-if-available, so
    # force it to see CPU when timing the CPU path (no file modification)
    if device.type == 'cpu':
        torch.cuda.is_available, _orig_avail = (lambda: False), torch.cuda.is_available
    model.to(device)
    x = loc_test_norm.to(device)
    A = A_test.to(device)
    P = P_test.to(device)
    with torch.no_grad():
        for i in range(N_WARM):
            model(x[i], AP_num_test[i], UE_num_test[i], A[i], P[i])
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(N_TIME):
            model(x[i], AP_num_test[i], UE_num_test[i], A[i], P[i])
        if device.type == 'cuda':
            torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / N_TIME
    if device.type == 'cpu':
        torch.cuda.is_available = _orig_avail
    return t

t_gnn_cpu = gnn_timing(torch.device('cpu'))
t_gnn_gpu = gnn_timing(torch.device('cuda')) if torch.cuda.is_available() else None

# ---------- parameter counts ----------
n_signal = sum(p.numel() for p in model.signal_model.parameters())
n_interf = sum(p.numel() for p in model.interf_model.parameters())

# ---------- report ----------
avg_ue = float(np.mean([g.shape[1] for g in g_list]))
avg_edges_sig = float(np.mean([a.sum() for a in A_list]))
print("\n=== Computational complexity (per network snapshot, L=12, "
      f"avg K={avg_ue:.1f}) ===")
print(f"Closed-form analytical : {t_analytical*1e6:8.1f} us "
      f"(O(L*K) flops ~ {int(3*12*avg_ue)} mult-adds, 0 parameters, no training)")
print(f"Proposed GNN (CPU)     : {t_gnn_cpu*1e6:8.1f} us")
if t_gnn_gpu is not None:
    print(f"Proposed GNN (GPU)     : {t_gnn_gpu*1e6:8.1f} us "
          f"({torch.cuda.get_device_name(0)})")
print(f"GNN parameters         : signal {n_signal:,} + interf {n_interf:,} "
      f"= {n_signal + n_interf:,}")
print(f"Signal graph edges     : avg {avg_edges_sig:.1f} (served links)")
print(f"Interf graph edges     : served links + 8*K AP->UE leakage edges")
print(f"GNN per-layer cost     : O((L+K)*d^2 + |E|*d^2), d=128")
