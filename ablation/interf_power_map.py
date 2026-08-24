# predict interference (remove power from edge attr.) — R1-3h edge-feature ablation
import os
import torch
import numpy as np
import torch.nn as nn
from process import *
import interf_map as im
from ablation_common import set_seed, add_common_args, variant_name, save_mae,\
    train_resumable, clear_ckpt, GRAPH_FORWARD


class InterfModel(im.InterfModel):
    # identical scaffold; both edge sets reduced to [dist2, unit_x, unit_y]
    def __init__(self, hidden=128, head=4, layers=1):
        super().__init__(hidden, head, layers)
        self.edge_mlp1 = nn.Sequential(nn.Linear(3, int(hidden/2)), nn.GELU(),
                                       nn.Linear(int(hidden/2), hidden))
        self.edge_mlp2 = nn.Sequential(nn.Linear(3, int(hidden/2)), nn.GELU(),
                                       nn.Linear(int(hidden/2), hidden))

    def create_edges(self, x, num_aps, num_ues, A, P):
        edge1_index, edge1_attr, edge2_index, edge2_attr = super().create_edges(
            x, num_aps, num_ues, A, P)
        return edge1_index, edge1_attr[:, 1:], edge2_index, edge2_attr[:, 1:]


# main
if __name__ == "__main__":
    args = add_common_args().parse_args()
    name = variant_name('power', args.tag, args.seed)

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
