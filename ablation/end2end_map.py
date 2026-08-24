# w/o decomposition ablation (R1-3b, R3-2): single GNN regressing the rate
# directly. Identical to end2end.py except the interfering-edge neighborhood
# is top_l = 8 — the SAME as the proposed model (end2end.py used 6, which
# confounded the ablation with a sparser graph). The graph construction is
# reused verbatim from interf_map so the only difference vs. the proposed
# pipeline is the absence of the signal/interference decomposition.
import os
import torch
import numpy as np
import torch.nn as nn
from process import *
import end2end as e2e
import interf_map as im
from ablation_common import set_seed, add_common_args, variant_name, save_mae,\
    train_resumable, clear_ckpt, GRAPH_FORWARD


class End2EndModel(e2e.InterfModel):
    # same edge set as the proposed interference model (top_l = 8)
    create_edges = im.InterfModel.create_edges


# main
if __name__ == "__main__":
    args = add_common_args().parse_args()
    snr = args.snr
    name = variant_name('end2end', args.tag, args.seed)

    set_seed(args.seed)
    model = End2EndModel()
    print(f"[{name}] params: "
          f"{sum(p.numel() for p in model.parameters()):,}")
    model_path = f'model/{name}.pth'

    if not os.path.exists(model_path):
        train_loader, val_loader = split_train_val(rate_norm[int(snr/5)], 0.8, 0.2)
        model, train_losses, val_losses = train_resumable(
            model, train_loader, val_loader, args.max_epochs or 500,
            GRAPH_FORWARD, nn.MSELoss(), 1e-3, model_path + '.ckpt')
        np.save(f'loss/{name}_train_losses.npy', train_losses)
        np.save(f'loss/{name}_val_losses.npy', val_losses)
        torch.save(model.state_dict(), model_path)
        clear_ckpt(model_path + '.ckpt')

    model.load_state_dict(torch.load(model_path, weights_only=True,
                                     map_location='cpu'))
    mse_loss, mae_loss, mae_mat = e2e.evaluate_model(model, snr)
    save_mae(mae_mat, f'result/pred/pred_{name}_mae.npy')
    print(f"[{name}] Test MSE: {mse_loss:.3f}, "
          f"Test RMSE: {np.sqrt(mse_loss):.3f}, Test MAE: {mae_loss:.3f}")
