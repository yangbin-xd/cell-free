# single-layer interference message passing ablation (R1-3e):
# drop the UE->AP stage entirely; only the AP->UE step remains, with the
# same interfering edge set (topk-8 nearest APs, edge power = 1 - p)
import os
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from process import *
import interf_map as im
from ablation_common import set_seed, add_common_args, variant_name, save_mae,\
    train_resumable, clear_ckpt, GRAPH_FORWARD


class InterfModel(nn.Module):
    def __init__(self, hidden=128, head=4, layers=1):
        super().__init__()
        self.layers = layers

        self.embed_ap = nn.Linear(2, hidden)
        self.embed_ue = nn.Linear(2, hidden)

        self.edge_mlp2 = nn.Sequential(nn.Linear(4, int(hidden/2)), nn.GELU(),
                                       nn.Linear(int(hidden/2), hidden))

        self.ap2ue = nn.ModuleList([TransformerConv((hidden, hidden),
                     out_channels=hidden, heads=head, concat=False,
                     edge_dim=hidden, beta=False) for _ in range(layers)])

        self.norm_ue = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

        self.ffn_ue = nn.ModuleList([nn.Sequential(
                      nn.Linear(hidden, 2*hidden), nn.GELU(), nn.Dropout(0.2),
                      nn.Linear(2*hidden, hidden)) for _ in range(layers)])

        self.norm_ffn_ue = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

        self.mlp = nn.Sequential(nn.Linear(hidden, int(hidden/2)), nn.GELU(),
                                 nn.Dropout(0.2), nn.Linear(int(hidden/2), 1))

    def forward(self, x, edge2_index, edge2_attr):
        x_ap = x[:self.num_aps, :]
        x_ue = x[self.num_aps:self.num_aps+self.num_ues, :]

        h_ap = self.embed_ap(x_ap)
        h_ue = self.embed_ue(x_ue)

        edge2_attr = self.edge_mlp2(edge2_attr)

        for l in range(self.layers):
            # AP -> UE only (no UE -> AP interference-generation stage)
            h_ue = self.norm_ue[l](self.ap2ue[l]((h_ap, h_ue), edge2_index, edge2_attr)
                                                            + h_ue)
            h_ue = self.norm_ffn_ue[l](self.ffn_ue[l](h_ue) + h_ue)

        return self.mlp(h_ue).squeeze(-1)

    def create_edges(self, x, num_aps, num_ues, A, P):
        # reuse the main model's edge construction, keep only the AP->UE set
        _, _, edge2_index, edge2_attr = im.InterfModel.create_edges(
            self, x, num_aps, num_ues, A, P)
        return edge2_index, edge2_attr


def evaluate_model(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    criterion = nn.MSELoss()
    mse_loss, mae_loss = 0, 0
    test_samples = loc_test.shape[0]
    ap_num, ue_num, loc, A, P, true_value = AP_num_test.to(device),\
        UE_num_test.to(device), loc_test_norm.to(device), A_test.to(device),\
            P_test.to(device), interf_test.to(device)

    with torch.no_grad():
        mae_list = []
        for i in range(test_samples):
            edge2_index, edge2_attr = model.create_edges(loc[i], ap_num[i],
                                                         ue_num[i], A[i], P[i])
            pred_value_norm = model(loc[i], edge2_index, edge2_attr)
            pred_value = pred_value_norm * interf_std + interf_mean

            mse_loss += criterion(pred_value, true_value[i, :ue_num[i]])
            mae = pred_value - true_value[i, :ue_num[i]]
            mae_loss += torch.mean(torch.abs(mae))

            row = torch.full((50,), float('nan'), device='cpu')
            row[:int(ue_num[i])] = mae.detach().cpu()
            mae_list.append(row)

    return mse_loss.item() / test_samples, mae_loss.item() / test_samples,\
           torch.stack(mae_list, dim=0)


# main
if __name__ == "__main__":
    args = add_common_args().parse_args()
    name = variant_name('single', args.tag, args.seed)

    set_seed(args.seed)
    model = InterfModel()
    print(f"[interf_{name}] params: "
          f"{sum(p.numel() for p in model.parameters()):,} "
          f"(main InterfModel: 719,617)")
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
    mse_loss, mae_loss, mae_mat = evaluate_model(model)
    save_mae(mae_mat, f'result/pred/pred_interf_{name}_mae.npy')
    print(f"[interf_{name}] Test MSE: {mse_loss:.3f}, "
          f"Test RMSE: {np.sqrt(mse_loss):.3f}, Test MAE: {mae_loss:.3f}")
