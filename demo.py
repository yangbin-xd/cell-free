# proposed method digital twin
import os
from process import *
from signal_map import SignalModel
from interf_map import InterfModel
from rate_map import RateModel
from generate import loc_mean, loc_std
from main import plot_result

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
# 'Times New Roman' is absent on Katana; Nimbus Roman is URW's
# metric-compatible clone of it. Without this chain matplotlib falls back
# to DejaVu Sans and the figures come out in the wrong typeface entirely.
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Nimbus Roman',
                                     'DejaVu Serif']
font1, font2 = 24, 18

# The proposed model IS the ablation campaign's seed-0 full model -- the
# composition reported as the reference row of result/ablation_table.md (see
# MAIN in ablation/make_ablation_table.py). These replace the archived
# signal_map.pth / interf_map.pth / rate_map_{snr}dB.pth, which were a
# different, unseeded training run.
#
# Only a 15 dB rate head exists for this variant, and it is the right one at
# every SNR: snr enters RateModel solely through `noise_dB = -87 - self.snr` in
# the closed-form SINR step (rate_map.py:117-125), so no parameter is
# SNR-specific. See ablation/snr_eval.py, which sweeps 0-30 dB this way.
PROPOSED = 'main_seed0'
SIGNAL_PATH = f'model/signal_{PROPOSED}.pth'
INTERF_PATH = f'model/interf_{PROPOSED}.pth'
RATE_PATH = f'model/rate_{PROPOSED}_15dB.pth'


def _load(module, path, what):
    """Load a checkpoint, failing loudly if it is absent.

    The `if os.path.exists(path): load(...)` guard this replaces is what let the
    broken RATE_PATH f-string go unnoticed: a missing checkpoint left the module
    at its constructor weights and everything downstream ran normally. Same
    reasoning as ablation_common.load_branch.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{what} checkpoint not found: {path!r}. digital_twin evaluates the '
            f'proposed model and must not fall back to untrained weights.')
    module.load_state_dict(torch.load(path, weights_only=True,
                                      map_location='cpu'))


def digital_twin(snr=15, test_idx=0, do_plot=True, do_print=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ap_num = int(AP_num_test[test_idx].item())
    ue_num = int(UE_num_test[test_idx].item())

    loc = loc_test_norm[test_idx].to(device)
    A = A_test[test_idx].to(device)
    P = P_test[test_idx].to(device)
    true_signal = signal_test[test_idx][:ue_num].numpy()
    true_interf = interf_test[test_idx][:ue_num].numpy()
    true_rate = rate_test[int(snr/5)][test_idx][:ue_num].numpy()

    # plot scenario
    BS_loc = loc_test_norm[test_idx, :ap_num, :] * loc_std + loc_mean
    UE_loc = loc_test_norm[test_idx, ap_num:ap_num+ue_num, :] * loc_std + loc_mean

    if do_plot:
        plot_result(BS_loc, UE_loc, A, P)
        # plt.savefig('result/twin.pdf')

    # digital twin
    SignalMap = SignalModel()
    _load(SignalMap, SIGNAL_PATH, 'signal')

    SignalMap = SignalMap.to(device)
    SignalMap.eval()
    with torch.no_grad():
        edge_index, edge_attr = SignalMap.create_edges(loc, ap_num, ue_num, A, P)
        pred_value_norm = SignalMap(loc, edge_index, edge_attr)
        pred_value = pred_value_norm * signal_std + signal_mean
    pred_signal = pred_value.detach().cpu().numpy()

    InterfMap = InterfModel()
    _load(InterfMap, INTERF_PATH, 'interference')

    InterfMap = InterfMap.to(device)
    SignalMap.eval()
    with torch.no_grad():
        edge1_index, edge1_attr, edge2_index, edge2_attr = \
            InterfMap.create_edges(loc, ap_num, ue_num, A, P)
        pred_value_norm = InterfMap(loc, edge1_index, edge1_attr, edge2_index, edge2_attr)
        pred_value = pred_value_norm * interf_std + interf_mean
    pred_interf = pred_value.detach().cpu().numpy()

    # RATE_PATH holds a full RateModel state dict (both branches plus the
    # fine-tuned heads), so it overrides the constructor's branches wholesale.
    # The path was previously a plain string, not an f-string, so
    # os.path.exists() was always False and the rate head was never loaded --
    # every array produced through this function before 2026-07-30
    # (plot_band.py's generator block -> pred_rate_error_*dB.npy,
    # pred_rate_sum_*dB.npy, and the *_twin.pdf figures) is the PRE-fine-tuning
    # composition of the archived branches, not the model it claimed to be.
    RateMap = RateModel(signal_model_path=SIGNAL_PATH,
                        interf_model_path=INTERF_PATH, snr=snr)
    _load(RateMap, RATE_PATH, 'rate')

    RateMap = RateMap.to(device)
    RateMap.eval()
    with torch.no_grad():
        pred_value_norm = RateMap(loc, ap_num, ue_num, A, P)
        pred_value = pred_value_norm * rate_std + rate_mean
    pred_rate = pred_value.detach().cpu().numpy()

    if do_print:
        print(f"Pred signal: {pred_signal} dB")
        print(f"Pred interf: {pred_interf} dB")
        print(f"Pred rate:   {pred_rate} bits/s/Hz")

    def plot_error(True_value, Pred_value, term):
        Abs_Error = np.abs(True_value - Pred_value)
        x = np.arange(True_value.shape[0])
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(x, True_value, label=f'True {term}', linewidth=2.0)
        plt.plot(x, Pred_value, label=f'Predicted {term}', linewidth=2.0)
        plt.plot(x, Abs_Error, label='Error', linewidth=2.0)
        plt.xlabel('UE ID', fontsize=font1)
        plt.ylabel(f'{term} (dB)', fontsize=font1)
        plt.legend(fontsize=font1)
        plt.xticks(fontsize=font1)
        plt.yticks(fontsize=font1)
        plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)
        ax.set_xticks(np.arange(0, 30, 2))
        plt.tight_layout()

    if do_plot:
        plot_error(true_signal, pred_signal, "Signal")
        plot_error(true_interf, pred_interf, "Interference")
        plot_error(true_rate,   pred_rate, "Rate")
        plt.show()

    signal_error = np.abs(pred_signal - true_signal)
    interf_error = np.abs(pred_interf - true_interf)
    rate_error = np.abs(pred_rate - true_rate)

    if do_print:
        print(f"Signal: mean: {np.mean(signal_error)}, max: {np.max(signal_error)}")
        print(f"Interf: mean: {np.mean(interf_error)}, max: {np.max(interf_error)}")
        print(f"Rate: mean: {np.mean(rate_error)}, max: {np.max(rate_error)}")

    return ue_num, true_signal, true_interf, true_rate,\
           pred_signal, pred_interf, pred_rate

if __name__ == '__main__':

    ue_num, true_signal, true_interf, true_rate, pred_signal, pred_interf,\
        pred_rate = digital_twin(snr=15, test_idx=0)