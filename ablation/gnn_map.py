# aggregation ablation / learning baselines: GCN, GAT (no edge feat.),
# edge-aware GAT, GraphSAGE — same scaffold as the main model, only the
# aggregation layer is swapped (R1-3f/g, R5-3)
import os
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from process import *
import signal_map as sm
import interf_map as im
import rate_map as rm
from ablation_common import set_seed, add_common_args, variant_name, save_mae,\
    train_resumable, clear_ckpt, GRAPH_FORWARD, RATE_FORWARD, load_branch

# hidden widths chosen so total params match the main model
# (signal 364,353 / interf 719,617; all variants within ~1-4%)
DEFAULT_HIDDEN = {'gcn': 256, 'gat': 170, 'gat_edge': 146, 'sage': 240}
HEAD = 4


def make_conv(conv_type, hidden, aggr='mean'):
    if conv_type == 'gcn':
        return GCNConv(hidden, hidden, add_self_loops=False, normalize=True)
    if conv_type == 'gat':
        return GATConv((hidden, hidden), hidden, heads=HEAD, concat=False,
                       add_self_loops=False)
    if conv_type == 'gat_edge':
        return GATConv((hidden, hidden), hidden, heads=HEAD, concat=False,
                       add_self_loops=False, edge_dim=hidden)
    if conv_type == 'sage':
        # aggr is a pure aggregation choice: parameter count is identical to
        # the mean variant, so the sum/mean comparison isolates the effect of
        # normalising away neighbour count on an additive physical quantity
        return SAGEConv((hidden, hidden), hidden, aggr=aggr)
    raise ValueError(conv_type)


def propagate(conv_type, conv, h_src, h_dst, edge_index, edge_attr, edge_mlp=None):
    # GCN has no bipartite mode: run on the combined node set and slice back
    if conv_type == 'gcn':
        h_all = torch.cat([h_src, h_dst], dim=0)
        ei = torch.stack([edge_index[0], edge_index[1] + h_src.size(0)], dim=0)
        out = conv(h_all, ei, edge_weight=edge_attr[:, 0])
        return out[h_src.size(0):]
    if conv_type == 'gat':
        return conv((h_src, h_dst), edge_index)
    if conv_type == 'gat_edge':
        return conv((h_src, h_dst), edge_index, edge_attr=edge_mlp(edge_attr))
    return conv((h_src, h_dst), edge_index)  # sage


class SignalModel(nn.Module):
    # same scaffold/attribute names as signal_map.SignalModel so the rate
    # stage's selective fine-tuning applies verbatim
    create_edges = sm.SignalModel.create_edges

    def __init__(self, conv_type, hidden, layers=1, aggr='mean'):
        super().__init__()
        self.conv_type = conv_type
        self.layers = layers

        self.embed_ap = nn.Linear(2, hidden)
        self.embed_ue = nn.Linear(2, hidden)

        if conv_type == 'gat_edge':
            self.edge_mlp = nn.Sequential(nn.Linear(4, int(hidden/2)), nn.GELU(),
                                          nn.Linear(int(hidden/2), hidden))

        self.ap2ue = nn.ModuleList([make_conv(conv_type, hidden, aggr)
                                    for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.ffn_ue = nn.ModuleList([nn.Sequential(
                      nn.Linear(hidden, 2*hidden), nn.GELU(),
                      nn.Linear(2*hidden, hidden)) for _ in range(layers)])
        self.norm_ffn = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.mlp = nn.Sequential(nn.Linear(hidden, int(hidden/2)), nn.GELU(),
                                 nn.Linear(int(hidden/2), 1))

    def forward(self, x, edge_index, edge_attr):
        x_ap = x[:self.num_aps, :]
        x_ue = x[self.num_aps:self.num_aps+self.num_ues, :]

        h_ap = self.embed_ap(x_ap)
        h_ue = self.embed_ue(x_ue)

        edge_mlp = getattr(self, 'edge_mlp', None)
        for l in range(self.layers):
            h_ue = self.norm[l](propagate(self.conv_type, self.ap2ue[l], h_ap,
                                h_ue, edge_index, edge_attr, edge_mlp) + h_ue)
            h_ue = self.norm_ffn[l](self.ffn_ue[l](h_ue) + h_ue)

        return self.mlp(h_ue).squeeze(-1)


class InterfModel(nn.Module):
    create_edges = im.InterfModel.create_edges

    def __init__(self, conv_type, hidden, layers=1, aggr='mean'):
        super().__init__()
        self.conv_type = conv_type
        self.layers = layers

        self.embed_ap = nn.Linear(2, hidden)
        self.embed_ue = nn.Linear(2, hidden)

        if conv_type == 'gat_edge':
            self.edge_mlp1 = nn.Sequential(nn.Linear(4, int(hidden/2)), nn.GELU(),
                                           nn.Linear(int(hidden/2), hidden))
            self.edge_mlp2 = nn.Sequential(nn.Linear(4, int(hidden/2)), nn.GELU(),
                                           nn.Linear(int(hidden/2), hidden))

        self.ue2ap = nn.ModuleList([make_conv(conv_type, hidden, aggr)
                                    for _ in range(layers)])
        self.ap2ue = nn.ModuleList([make_conv(conv_type, hidden, aggr)
                                    for _ in range(layers)])

        self.norm_ap = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.norm_ue = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

        self.ffn_ap = nn.ModuleList([nn.Sequential(
                      nn.Linear(hidden, 2*hidden), nn.GELU(), nn.Dropout(0.2),
                      nn.Linear(2*hidden, hidden)) for _ in range(layers)])
        self.ffn_ue = nn.ModuleList([nn.Sequential(
                      nn.Linear(hidden, 2*hidden), nn.GELU(), nn.Dropout(0.2),
                      nn.Linear(2*hidden, hidden)) for _ in range(layers)])

        self.norm_ffn_ap = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.norm_ffn_ue = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

        self.mlp = nn.Sequential(nn.Linear(hidden, int(hidden/2)), nn.GELU(),
                                 nn.Dropout(0.2), nn.Linear(int(hidden/2), 1))

    def forward(self, x, edge1_index, edge1_attr, edge2_index, edge2_attr):
        x_ap = x[:self.num_aps, :]
        x_ue = x[self.num_aps:self.num_aps+self.num_ues, :]

        h_ap = self.embed_ap(x_ap)
        h_ue = self.embed_ue(x_ue)

        edge_mlp1 = getattr(self, 'edge_mlp1', None)
        edge_mlp2 = getattr(self, 'edge_mlp2', None)
        for l in range(self.layers):
            # UE -> AP
            h_ap = self.norm_ap[l](propagate(self.conv_type, self.ue2ap[l], h_ue,
                                   h_ap, edge1_index, edge1_attr, edge_mlp1) + h_ap)
            h_ap = self.norm_ffn_ap[l](self.ffn_ap[l](h_ap) + h_ap)

            # AP -> UE
            h_ue = self.norm_ue[l](propagate(self.conv_type, self.ap2ue[l], h_ap,
                                   h_ue, edge2_index, edge2_attr, edge_mlp2) + h_ue)
            h_ue = self.norm_ffn_ue[l](self.ffn_ue[l](h_ue) + h_ue)

        return self.mlp(h_ue).squeeze(-1)


class RateModel(nn.Module):
    # clone of rate_map.RateModel with the variant sub-models
    def __init__(self, conv_type, hidden, signal_model_path, interf_model_path,
                 snr=15, aggr='mean'):
        super().__init__()
        self.snr = snr

        self.signal_model = SignalModel(conv_type, hidden, aggr=aggr)
        self.interf_model = InterfModel(conv_type, hidden, aggr=aggr)

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

    def forward(self, x, ap_num, ue_num, A, P):
        signal_edge_index, signal_edge_attr = self.signal_model.create_edges(
            x, ap_num, ue_num, A, P)
        interf_edge1_index, interf_edge1_attr, interf_edge2_index, interf_edge2_attr = \
            self.interf_model.create_edges(x, ap_num, ue_num, A, P)

        pred_signal_norm = self.signal_model(x, signal_edge_index, signal_edge_attr)
        pred_interf_norm = self.interf_model(x, interf_edge1_index, interf_edge1_attr,
                                             interf_edge2_index, interf_edge2_attr)

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


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# main
if __name__ == "__main__":
    parser = add_common_args()
    parser.add_argument('--conv', required=True,
                        choices=['gcn', 'gat', 'gat_edge', 'sage'])
    parser.add_argument('--stage', default='all',
                        choices=['signal', 'interf', 'rate', 'all'])
    parser.add_argument('--hidden', type=int, default=None)
    parser.add_argument('--aggr', default='mean', choices=['mean', 'sum', 'max'],
                        help='SAGEConv aggregation; ignored by the other convs')
    args = parser.parse_args()

    hidden = args.hidden if args.hidden else DEFAULT_HIDDEN[args.conv]
    # fold a non-default aggregator into the variant name, otherwise a sum run
    # would silently overwrite the mean results under the same filenames
    base = args.conv
    if args.conv == 'sage' and args.aggr != 'mean':
        base = f'sage_{args.aggr}'
    name = variant_name(base, args.tag, args.seed)
    stages = ['signal', 'interf', 'rate'] if args.stage == 'all' else [args.stage]

    signal_path = f'model/signal_{name}.pth'
    interf_path = f'model/interf_{name}.pth'

    if 'signal' in stages:
        set_seed(args.seed)
        model = SignalModel(args.conv, hidden, aggr=args.aggr)
        print(f"[signal_{name}] params: {n_params(model):,} "
              f"(main TransformerConv: 364,353)")
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
        model = InterfModel(args.conv, hidden, aggr=args.aggr)
        print(f"[interf_{name}] params: {n_params(model):,} "
              f"(main TransformerConv: 719,617)")
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
        model = RateModel(args.conv, hidden, signal_path, interf_path,
                          snr=snr, aggr=args.aggr)
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
