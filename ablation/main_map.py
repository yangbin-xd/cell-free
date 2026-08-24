# Seeded reruns of the PROPOSED model itself, so the ablation table's reference
# row carries the same mean +/- std as every variant it is compared against.
# (Without this the reference is a single run and the "N sigma" effect sizes
# have no measured baseline.)
#
# The main scripts signal_map.py / interf_map.py / rate_map.py are imported
# unchanged and their model classes reused verbatim -- nothing here touches the
# artefacts the paper's existing figures depend on (model/signal_map.pth etc.),
# because every output is written under the variant name `main_seed{N}`.
import os
import torch
import numpy as np
import torch.nn as nn
from process import *
import signal_map as sm
import interf_map as im
import rate_map as rm
from ablation_common import set_seed, add_common_args, variant_name, save_mae,\
    train_resumable, clear_ckpt, GRAPH_FORWARD, RATE_FORWARD


if __name__ == "__main__":
    parser = add_common_args()
    parser.add_argument('--stage', default='all',
                        choices=['signal', 'interf', 'rate', 'all'])
    # The paper states 6000 train / 2000 val (= process.py's default 0.75), but
    # every committed *_map.py overrides it to 0.8 (6400/1600). Exposed here so
    # the two can be compared directly against the archived checkpoints.
    parser.add_argument('--train-ratio', type=float, default=0.8)
    # Val ratio defaults to the remainder, but is separately settable so a run
    # can use a strict subset of the realizations: with 10000 available,
    # --train-ratio 0.6 --val-ratio 0.2 reproduces the paper's literal
    # 6000 train / 2000 val and leaves indices 8000..9999 unused.
    parser.add_argument('--val-ratio', type=float, default=None)
    args = parser.parse_args()

    TR = args.train_ratio
    VR = round(1.0 - TR, 10) if args.val_ratio is None else args.val_ratio
    name = variant_name('main', args.tag, args.seed)
    N = len(loc_train)
    print(f'[split] {N} realizations, train_ratio={TR} val_ratio={VR} '
          f'-> {int(TR*N)} train / {int(VR*N)} val')
    stages = ['signal', 'interf', 'rate'] if args.stage == 'all' else [args.stage]

    signal_path = f'model/signal_{name}.pth'
    interf_path = f'model/interf_{name}.pth'

    if 'signal' in stages:
        set_seed(args.seed)
        model = sm.SignalModel()
        if not os.path.exists(signal_path):
            train_loader, val_loader = split_train_val(signal_norm, TR, VR)
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
        model = im.InterfModel()
        if not os.path.exists(interf_path):
            train_loader, val_loader = split_train_val(interf_norm, TR, VR)
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
        # The pred arrays below are not SNR-keyed, so running this stage at a
        # second SNR would overwrite the 15 dB arrays the ablation table's
        # reference row is built from -- with no error and no visible sign.
        if snr != 15:
            raise SystemExit(
                f'--snr {snr}: the rate stage writes pred_rate_{name}_mae.npy '
                f'/ _finetune.npy without an SNR in the name and would clobber '
                f'the 15 dB table arrays. For an SE-vs-SNR curve use '
                f'ablation/snr_eval.py (evaluates one fine-tuned model at every '
                f'SNR -- no training, since snr only enters the closed-form '
                f'noise term); use ablation/snr_sweep.py only if you actually '
                f'want a separately fine-tuned rate head per SNR.')
        set_seed(args.seed)
        model = rm.RateModel(signal_model_path=signal_path,
                             interf_model_path=interf_path, snr=snr)
        model_path = f'model/rate_{name}_{snr}dB.pth'

        # pre-finetune pass doubles as the "w/o fine-tuning" row
        mse_loss, mae_loss, value, mae_mat = rm.evaluate_model(model, snr)
        save_mae(mae_mat, f'result/pred/pred_rate_{name}_mae.npy')
        print(f"[rate_{name}] pre-finetune Test MAE: {mae_loss:.3f}")

        if not os.path.exists(model_path):
            train_loader, val_loader = split_train_val(rate_norm[int(snr/5)], TR, VR)
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
