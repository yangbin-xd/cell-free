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
matplotlib.rcParams['font.family'] = 'times new roman'
font1, font2 = 24, 18

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
    SignalMap_path = 'model/signal_map.pth'
    if os.path.exists(SignalMap_path):
        SignalMap.load_state_dict(torch.load(SignalMap_path, weights_only=True)) 

    SignalMap = SignalMap.to(device)
    SignalMap.eval()
    with torch.no_grad():
        edge_index, edge_attr = SignalMap.create_edges(loc, ap_num, ue_num, A, P)
        pred_value_norm = SignalMap(loc, edge_index, edge_attr)
        pred_value = pred_value_norm * signal_std + signal_mean
    pred_signal = pred_value.detach().cpu().numpy()

    InterfMap = InterfModel()
    InterfMap_path = 'model/interf_map.pth'
    if os.path.exists(InterfMap_path):
        InterfMap.load_state_dict(torch.load(InterfMap_path, weights_only=True))

    InterfMap = InterfMap.to(device)
    SignalMap.eval()
    with torch.no_grad():
        edge1_index, edge1_attr, edge2_index, edge2_attr = \
            InterfMap.create_edges(loc, ap_num, ue_num, A, P)
        pred_value_norm = InterfMap(loc, edge1_index, edge1_attr, edge2_index, edge2_attr)
        pred_value = pred_value_norm * interf_std + interf_mean
    pred_interf = pred_value.detach().cpu().numpy()

    RateMap = RateModel(snr=snr)
    RateMap_path = f'model/rate_map_{snr}dB.pth'
    if os.path.exists(RateMap_path):
        RateMap.load_state_dict(torch.load(RateMap_path, weights_only=True))

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