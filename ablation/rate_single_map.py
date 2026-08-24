# rate prediction with the single-layer interference model (R1-3e);
# signal model reused from the main pipeline
import os
import torch
import numpy as np
import torch.nn as nn
from process import *
from signal_map import SignalModel
from interf_single_map import InterfModel
import rate_map as rm
from ablation_common import set_seed, add_common_args, variant_name, save_mae,\
    train_resumable, clear_ckpt, RATE_FORWARD, main_signal_path, load_branch


class RateModel(nn.Module):
    # Both branch paths are REQUIRED: this stage fine-tunes pre-trained branches.
    def __init__(self, signal_model_path, interf_model_path, snr=15):
        super().__init__()
        self.snr = snr

        self.signal_model = SignalModel()
        self.interf_model = InterfModel()

        load_branch(self.signal_model, signal_model_path, 'signal')
        load_branch(self.interf_model, interf_model_path, 'interference')

        for param in self.signal_model.parameters():
            param.requires_grad = False
        for param in self.interf_model.parameters():
            param.requires_grad = False

        for param in self.signal_model.mlp.parameters():
            param.requires_grad = True
        for param in self.signal_model.ffn_ue.parameters():
            param.requires_grad = True
        for param in self.signal_model.norm_ffn.parameters():
            param.requires_grad = True

        # single-layer interf model has no AP-side FFN/norm
        for param in self.interf_model.mlp.parameters():
            param.requires_grad = True
        for param in self.interf_model.ffn_ue.parameters():
            param.requires_grad = True
        for param in self.interf_model.norm_ffn_ue.parameters():
            param.requires_grad = True

        self.signal_mean = signal_mean
        self.signal_std = signal_std
        self.interf_mean = interf_mean
        self.interf_std = interf_std
        self.rate_mean = rate_mean
        self.rate_std = rate_std

    def forward(self, x, ap_num, ue_num, A, P):
        signal_edge_index, signal_edge_attr = self.signal_model.create_edges(
            x, ap_num, ue_num, A, P)
        interf_edge2_index, interf_edge2_attr = self.interf_model.create_edges(
            x, ap_num, ue_num, A, P)

        pred_signal_norm = self.signal_model(x, signal_edge_index, signal_edge_attr)
        pred_interf_norm = self.interf_model(x, interf_edge2_index, interf_edge2_attr)

        pred_signal = pred_signal_norm * self.signal_std + self.signal_mean
        pred_interf = pred_interf_norm * self.interf_std + self.interf_mean

        signal_linear = torch.pow(10, pred_signal / 10)
        interf_linear = torch.pow(10, pred_interf / 10)

        noise_dB = -87 - self.snr
        noise_power = 10 ** (noise_dB / 10)

        SINR = signal_linear / (interf_linear + noise_power)
        pred_rate = torch.log2(1 + SINR)
        pred_rate_norm = (pred_rate - self.rate_mean) / self.rate_std

        return pred_rate_norm


# main
if __name__ == "__main__":
    args = add_common_args().parse_args()
    snr = args.snr
    name = variant_name('single', args.tag, args.seed)

    set_seed(args.seed)
    # signal branch comes from the proposed model at the SAME seed -- this
    # variant ablates the interference branch only
    model = RateModel(signal_model_path=main_signal_path(args.seed),
                      interf_model_path=f'model/interf_{name}.pth', snr=snr)
    model_path = f'model/rate_{name}_{snr}dB.pth'

    mse_loss, mae_loss, value, mae_mat = rm.evaluate_model(model, snr)
    save_mae(mae_mat, f'result/pred/pred_rate_{name}_mae.npy')
    print(f"[rate_{name}] pre-finetune Test MAE: {mae_loss:.3f}")

    if not os.path.exists(model_path):
        train_loader, val_loader = split_train_val(rate_norm[int(snr/5)], 0.8, 0.2)
        model, train_losses, val_losses = train_resumable(
            model, train_loader, val_loader, args.max_epochs or 500,
            RATE_FORWARD, nn.L1Loss(reduction='mean'), 1e-4,
            model_path + '.ckpt')
        np.save(f'loss/rate_{name}_train_losses.npy', train_losses)
        np.save(f'loss/rate_{name}_val_losses.npy', val_losses)
        torch.save(model.state_dict(), model_path)
        clear_ckpt(model_path + '.ckpt')

    model.load_state_dict(torch.load(model_path, weights_only=True,
                                     map_location='cpu'))
    mse_loss, mae_loss, value, mae_mat = rm.evaluate_model(model, snr)
    save_mae(mae_mat, f'result/pred/pred_rate_{name}_finetune.npy')
    print(f"[rate_{name}] Test MSE: {mse_loss:.3f}, "
          f"Test RMSE: {np.sqrt(mse_loss):.3f}, Test MAE: {mae_loss:.3f}, "
          f"Test SUM: {value:.3f}")
