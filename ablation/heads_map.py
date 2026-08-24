# attention-head sweep (R1-3h): H in {1, 2}; H = 4 is the main model.
# The main model classes already take a `head` argument — reuse them directly.
# Note: TransformerConv params scale with H even with concat=False
# (conv ~82k @H=1 / ~148k @H=2 / ~280k @H=4), reported per row in the table.
import os
import torch
import numpy as np
import torch.nn as nn
from process import *
import signal_map as sm
import interf_map as im
import rate_map as rm
from ablation_common import set_seed, add_common_args, variant_name, save_mae,\
    train_resumable, clear_ckpt, GRAPH_FORWARD, RATE_FORWARD, load_branch


class RateModel(nn.Module):
    # clone of rate_map.RateModel with head-parameterized sub-models
    def __init__(self, head, signal_model_path, interf_model_path, snr=15):
        super().__init__()
        self.snr = snr

        self.signal_model = sm.SignalModel(head=head)
        self.interf_model = im.InterfModel(head=head)

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

        for param in self.interf_model.mlp.parameters():
            param.requires_grad = True
        for param in self.interf_model.ffn_ap.parameters():
            param.requires_grad = True
        for param in self.interf_model.ffn_ue.parameters():
            param.requires_grad = True
        for param in self.interf_model.norm_ffn_ap.parameters():
            param.requires_grad = True
        for param in self.interf_model.norm_ffn_ue.parameters():
            param.requires_grad = True

        self.signal_mean = signal_mean
        self.signal_std = signal_std
        self.interf_mean = interf_mean
        self.interf_std = interf_std
        self.rate_mean = rate_mean
        self.rate_std = rate_std

    forward = rm.RateModel.forward


# main
if __name__ == "__main__":
    parser = add_common_args()
    parser.add_argument('--head', type=int, required=True, choices=[1, 2, 8])
    parser.add_argument('--stage', default='all',
                        choices=['signal', 'interf', 'rate', 'all'])
    args = parser.parse_args()

    name = variant_name(f'h{args.head}', args.tag, args.seed)
    stages = ['signal', 'interf', 'rate'] if args.stage == 'all' else [args.stage]

    signal_path = f'model/signal_{name}.pth'
    interf_path = f'model/interf_{name}.pth'

    if 'signal' in stages:
        set_seed(args.seed)
        model = sm.SignalModel(head=args.head)
        print(f"[signal_{name}] params: "
              f"{sum(p.numel() for p in model.parameters()):,}")
        if not os.path.exists(signal_path):
            train_loader, val_loader = split_train_val(signal_norm, 0.8, 0.2)
            model, train_losses, val_losses = train_resumable(
                model, train_loader, val_loader, args.max_epochs or 1500,
                GRAPH_FORWARD, nn.L1Loss(reduction='mean'), 1e-3,
                signal_path + '.ckpt')
            np.save(f'loss/signal_{name}_train_losses.npy', train_losses)
            np.save(f'loss/signal_{name}_val_losses.npy', val_losses)
            torch.save(model.state_dict(), signal_path)
            clear_ckpt(signal_path + '.ckpt')

        model.load_state_dict(torch.load(signal_path, weights_only=True,
                                         map_location='cpu'))
        mse_loss, mae_loss, mae_mat = sm.evaluate_model(model)
        save_mae(mae_mat, f'result/pred/pred_signal_{name}_mae.npy')
        print(f"[signal_{name}] Test MSE: {mse_loss:.3f}, "
              f"Test RMSE: {np.sqrt(mse_loss):.3f}, Test MAE: {mae_loss:.3f}")

    if 'interf' in stages:
        set_seed(args.seed)
        model = im.InterfModel(head=args.head)
        print(f"[interf_{name}] params: "
              f"{sum(p.numel() for p in model.parameters()):,}")
        if not os.path.exists(interf_path):
            train_loader, val_loader = split_train_val(interf_norm, 0.8, 0.2)
            model, train_losses, val_losses = train_resumable(
                model, train_loader, val_loader, args.max_epochs or 500,
                GRAPH_FORWARD, nn.MSELoss(), 1e-3, interf_path + '.ckpt')
            np.save(f'loss/interf_{name}_train_losses.npy', train_losses)
            np.save(f'loss/interf_{name}_val_losses.npy', val_losses)
            torch.save(model.state_dict(), interf_path)
            clear_ckpt(interf_path + '.ckpt')

        model.load_state_dict(torch.load(interf_path, weights_only=True,
                                         map_location='cpu'))
        mse_loss, mae_loss, mae_mat = im.evaluate_model(model)
        save_mae(mae_mat, f'result/pred/pred_interf_{name}_mae.npy')
        print(f"[interf_{name}] Test MSE: {mse_loss:.3f}, "
              f"Test RMSE: {np.sqrt(mse_loss):.3f}, Test MAE: {mae_loss:.3f}")

    if 'rate' in stages:
        snr = args.snr
        set_seed(args.seed)
        model = RateModel(args.head, signal_path, interf_path, snr=snr)
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
