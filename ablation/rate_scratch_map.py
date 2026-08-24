# no-pretraining ablation (R1-3c): the two-stage signal+interf architecture
# trained end-to-end from random init with the rate loss only (nothing frozen,
# no signal/interference supervision) — distinguish from end2end.py, which is
# a single GNN without the decomposition
import os
import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from process import *
from signal_map import SignalModel
from interf_map import InterfModel
import rate_map as rm
from ablation_common import set_seed, add_common_args, variant_name, save_mae,\
    train_resumable, clear_ckpt, RATE_FORWARD


class RateModel(nn.Module):
    # clone of rate_map.RateModel: no checkpoint loading, all params trainable
    def __init__(self, snr=15):
        super().__init__()
        self.snr = snr

        self.signal_model = SignalModel()
        self.interf_model = InterfModel()

        self.signal_mean = signal_mean
        self.signal_std = signal_std
        self.interf_mean = interf_mean
        self.interf_std = interf_std
        self.rate_mean = rate_mean
        self.rate_std = rate_std

    forward = rm.RateModel.forward


# main
if __name__ == "__main__":
    args = add_common_args().parse_args()
    snr = args.snr
    name = variant_name('scratch', args.tag, args.seed)

    set_seed(args.seed)
    model = RateModel(snr=snr)
    print(f"[rate_{name}] params: "
          f"{sum(p.numel() for p in model.parameters()):,}")
    model_path = f'model/rate_{name}_{snr}dB.pth'

    if not os.path.exists(model_path):
        train_loader, val_loader = split_train_val(rate_norm[int(snr/5)], 0.8, 0.2)
        model, train_losses, val_losses = train_resumable(
            model, train_loader, val_loader, args.max_epochs or 1500,
            RATE_FORWARD, nn.L1Loss(reduction='mean'), 1e-3,
            model_path + '.ckpt')
        np.save(f'loss/rate_{name}_train_losses.npy', train_losses)
        np.save(f'loss/rate_{name}_val_losses.npy', val_losses)
        torch.save(model.state_dict(), model_path)
        clear_ckpt(model_path + '.ckpt')

    model.load_state_dict(torch.load(model_path, weights_only=True,
                                     map_location='cpu'))
    mse_loss, mae_loss, value, mae_mat = rm.evaluate_model(model, snr)
    save_mae(mae_mat, f'result/pred/pred_rate_{name}_mae.npy')
    print(f"[rate_{name}] Test MSE: {mse_loss:.3f}, "
          f"Test RMSE: {np.sqrt(mse_loss):.3f}, Test MAE: {mae_loss:.3f}, "
          f"Test SUM: {value:.3f}")
