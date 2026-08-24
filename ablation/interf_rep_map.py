# interference edge power representation ablation (R5-2): replace the AP->UE
# edge power component 1 - p_{l,k} with the raw allocation p_{l,k}.
# ("total non-target power" and "explicit per-interferer sum" are both
# mathematically identical to 1 - p_{l,k} under per-AP power normalization,
# so they are argued analytically in the reply letter instead of trained.)
import os
import torch
import numpy as np
import torch.nn as nn
from process import *
import interf_map as im
from ablation_common import set_seed, add_common_args, variant_name, save_mae,\
    train_resumable, clear_ckpt, GRAPH_FORWARD


class InterfModel(im.InterfModel):
    # identical architecture; only the interfering-edge power feature changes
    def create_edges(self, x, num_aps, num_ues, A, P):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_aps = num_aps
        self.num_ues = num_ues
        A = A[:self.num_aps, :self.num_ues]
        P = P[:self.num_aps, :self.num_ues]
        x_ap = x[:self.num_aps, :]
        x_ue = x[self.num_aps:self.num_aps+self.num_ues, :]

        # calculate distance
        delta_all = x_ap[:, None, :] - x_ue[None, :, :] # [L,K,2]
        dist2_all = (delta_all * delta_all).sum(dim=-1) # [L,K]
        dist_all  = dist2_all.sqrt()                    # [L,K]
        unit_all  = delta_all / dist_all[..., None]     # [L,K,2]

        # UE -> AP: served edges connected
        serve_mask = (A > 0.5)
        ap_idx_s, ue_idx_s = serve_mask.nonzero(as_tuple=True)

        sort_key = ap_idx_s * self.num_ues + ue_idx_s
        order = torch.argsort(sort_key)
        ap_idx_s = ap_idx_s[order]
        ue_idx_s = ue_idx_s[order]

        power_s = P[ap_idx_s, ue_idx_s]
        dist2_s = dist2_all[ap_idx_s, ue_idx_s]
        unit_s  = unit_all[ap_idx_s, ue_idx_s, :]

        edge1_index = torch.stack([ue_idx_s, ap_idx_s], dim=0).to(device)
        edge1_attr = torch.stack([power_s, dist2_s, unit_s[:, 0], unit_s[:, 1]], dim=1).to(device)

        # AP -> UE: partially fully connected
        top_l = min(8, num_aps)
        top_d, top_ap_idx = torch.topk(dist_all, k=top_l, dim=0, largest=False)

        ap_idx_i = top_ap_idx.t().reshape(-1)
        ue_idx_i = torch.arange(self.num_ues, device=device).unsqueeze(1).repeat(1, top_l).reshape(-1)

        keep_mask = (P[ap_idx_i, ue_idx_i] < (1.0 - 1e-20))
        ap_idx_i = ap_idx_i[keep_mask]
        ue_idx_i = ue_idx_i[keep_mask]

        # ablated representation: raw p_{l,k} instead of 1 - p_{l,k}
        power_i = P[ap_idx_i, ue_idx_i]
        dist2_i = dist2_all[ap_idx_i, ue_idx_i]
        unit_i  = unit_all[ap_idx_i, ue_idx_i, :]

        edge2_index = torch.stack([ap_idx_i, ue_idx_i], dim=0).to(device)
        edge2_attr  = torch.stack([power_i, dist2_i, unit_i[:, 0], unit_i[:, 1]], dim=1).to(device)

        return edge1_index, edge1_attr, edge2_index, edge2_attr


# main
if __name__ == "__main__":
    args = add_common_args().parse_args()
    name = variant_name('rep', args.tag, args.seed)

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
