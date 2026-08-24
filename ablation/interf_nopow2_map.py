# interference edge-feature ablation: drop the power channel from the AP->UE
# (stage-2) edges only, leaving [dist2, unit_x, unit_y] there. The UE->AP
# (stage-1) served edges keep their raw allocation p_{l,k}.
#
# Distinct from two neighbouring rows, and the distinction is the whole point:
#   * 'power' (w/o power) strips the power channel from BOTH edge sets, so it
#     also removes p from the serving link and is not a statement about the
#     interference edges specifically.
#   * 'rep' (1-p -> p) keeps a power channel on stage 2 and only changes its
#     parameterisation.
# This variant is the one that asks whether the interference edges carry any
# usable power information at all, with the serving link left intact.
import os
import torch
import numpy as np
import torch.nn as nn
from process import *
import interf_map as im
from ablation_common import set_seed, add_common_args, variant_name, save_mae,\
    train_resumable, clear_ckpt, GRAPH_FORWARD


class InterfModel(im.InterfModel):
    # identical scaffold; only edge2 is reduced to [dist2, unit_x, unit_y]
    def __init__(self, hidden=128, head=4, layers=1):
        super().__init__(hidden, head, layers)
        self.edge_mlp2 = nn.Sequential(nn.Linear(3, int(hidden/2)), nn.GELU(),
                                       nn.Linear(int(hidden/2), hidden))

    def create_edges(self, x, num_aps, num_ues, A, P):
        edge1_index, edge1_attr, edge2_index, edge2_attr = super().create_edges(
            x, num_aps, num_ues, A, P)
        # column 0 of edge2_attr is 1 - p_{l,k}; edge1_attr is left untouched
        return edge1_index, edge1_attr, edge2_index, edge2_attr[:, 1:]


# main
if __name__ == "__main__":
    args = add_common_args().parse_args()
    name = variant_name('nopow2', args.tag, args.seed)

    set_seed(args.seed)
    model = InterfModel()
    model_path = f'model/interf_{name}.pth'

    if not os.path.exists(model_path):
        train_loader, val_loader = split_train_val(interf_norm, 0.8, 0.2)
        model, train_losses, val_losses = train_resumable(
            model, train_loader, val_loader, args.max_epochs or 500,
            GRAPH_FORWARD, nn.MSELoss(), 1e-3, model_path + '.ckpt')
        np.save(f'loss/interf_{name}_train_losses.npy', train_losses)
        np.save(f'loss/interf_{name}_val_losses.npy', val_losses)
        torch.save(model.state_dict(), model_path)
        clear_ckpt(model_path + '.ckpt')

    model.load_state_dict(torch.load(model_path, weights_only=True,
                                     map_location='cpu'))
    mse_loss, mae_loss, mae_mat = im.evaluate_model(model)
    save_mae(mae_mat, f'result/pred/pred_interf_{name}_mae.npy')
    print(f"[interf_{name}] Test MSE: {mse_loss:.3f}, "
          f"Test RMSE: {np.sqrt(mse_loss):.3f}, Test MAE: {mae_loss:.3f}")
