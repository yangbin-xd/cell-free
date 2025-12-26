# predict signal
import os
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from process import *

class SignalModel(nn.Module):
    def __init__(self, hidden=128, head=4, layers=1):
        super().__init__()
        self.layers = layers

        self.embed_ap = nn.Linear(2, hidden)
        self.embed_ue = nn.Linear(2, hidden)

        self.edge_mlp = nn.Sequential(nn.Linear(4, int(hidden/2)), nn.GELU(),
                                      nn.Linear(int(hidden/2), hidden))
        
        self.ap2ue = nn.ModuleList([TransformerConv((hidden, hidden),
                     out_channels=hidden, heads=head, concat=False,
                     edge_dim=hidden, beta=False) for _ in range(layers)])

        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

        self.ffn_ue = nn.ModuleList([nn.Sequential(
                      nn.Linear(hidden, 2*hidden), nn.GELU(), 
                      nn.Linear(2*hidden, hidden)) for _ in range(layers)])
        
        self.norm_ffn = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

        self.mlp = nn.Sequential(nn.Linear(hidden, int(hidden/2)), nn.GELU(), 
                                 nn.Linear(int(hidden/2), 1))

    def forward(self, x, edge_index, edge_attr):
        # x: [13,2], edge_index: [2,E]
        x_ap = x[:self.num_aps, :]
        x_ue = x[self.num_aps:self.num_aps+self.num_ues, :]

        h_ap = self.embed_ap(x_ap)
        h_ue = self.embed_ue(x_ue)

        edge_attr = self.edge_mlp(edge_attr)

        for l in range(self.layers):
            # AP -> UE
            h_ue = self.norm[l](self.ap2ue[l]((h_ap, h_ue), edge_index, edge_attr) 
                                                         + h_ue)
            h_ue = self.norm_ffn[l](self.ffn_ue[l](h_ue) + h_ue)

        return self.mlp(h_ue).squeeze(-1)

    def create_edges(self, x, num_aps, num_ues, A, P):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_aps = num_aps # L
        self.num_ues = num_ues # K
        A = A[:self.num_aps, :self.num_ues]
        P = P[:self.num_aps, :self.num_ues]
        x_ap = x[:self.num_aps, :]
        x_ue = x[self.num_aps:self.num_aps+self.num_ues, :]

        # calculate distance
        delta_all = x_ap[:, None, :] - x_ue[None, :, :] # [L,K,2]
        dist2_all = (delta_all * delta_all).sum(dim=-1) # [L,K]
        dist_all  = dist2_all.sqrt()                    # [L,K]
        unit_all  = delta_all / dist_all[..., None]     # [L,K,2]

        # served edges connected
        serve_mask = (A > 0.5)                             # [L,K] bool
        ap_idx, ue_idx = serve_mask.nonzero(as_tuple=True) # [E], [E]

        # for ap in ...: for ue in ...
        sort_key = ap_idx * self.num_ues + ue_idx
        order = torch.argsort(sort_key)
        ap_idx = ap_idx[order]
        ue_idx = ue_idx[order]

        # combine edge_attr: [power, dist2, delta_x/dist, delta_y/dist]
        power = P[ap_idx, ue_idx]           # [E]
        dist2 = dist2_all[ap_idx, ue_idx]   # [E]
        unit  = unit_all[ap_idx, ue_idx, :] # [E,2]

        edge_index = torch.stack([ap_idx, ue_idx], dim=0).to(device) # [2,E]
        edge_attr = torch.stack([power, dist2, unit[:, 0], unit[:, 1]], dim=1).to(device) # [E,4]

        return edge_index, edge_attr


def train_model(model, train_loader, val_loader, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print('device:', torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('device: CPU')
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10, min_lr=1e-6)

    # criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss(beta=0.5)
    criterion = nn.L1Loss(reduction='mean')

    train_losses, val_losses = [], []
    best_val, best_state, no_improve, early_stop = float('inf'), None, 0, 20

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for ap_num, ue_num, loc, A, P, true_value in train_loader:
            ap_num, ue_num, loc, A, P, true_value = ap_num.to(device), ue_num.to(device),\
                loc.to(device), A.to(device), P.to(device), true_value.to(device)
            batch_size = loc.shape[0]
            batch_loss = 0
            
            for i in range(batch_size):
                edge_index, edge_attr = model.create_edges(loc[i], ap_num[i], 
                                                           ue_num[i], A[i], P[i])
                pred_value = model(loc[i], edge_index, edge_attr)
                loss = criterion(pred_value, true_value[i, :ue_num[i]])
                batch_loss += loss

            batch_loss = batch_loss / batch_size
            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for ap_num, ue_num, loc, A, P, true_value in val_loader:
                ap_num, ue_num, loc, A, P, true_value = ap_num.to(device),\
                    ue_num.to(device), loc.to(device), A.to(device), P.to(device),\
                        true_value.to(device)
                batch_size = loc.size(0)
                batch_loss = 0
                
                for i in range(batch_size):
                    edge_index, edge_attr = model.create_edges(loc[i], ap_num[i],
                                                               ue_num[i], A[i], P[i])
                    pred_value = model(loc[i], edge_index, edge_attr) 
                    loss = criterion(pred_value, true_value[i, :ue_num[i]])
                    batch_loss += loss
                
                val_loss += batch_loss.item() / batch_size
        
        # Record losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if epoch % 1 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], "
                  f"Train loss: {avg_train_loss:.4f}, "
                  f"Val loss: {avg_val_loss:.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.4e}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop:
                print(f"[EarlyStop] no improvement in {early_stop} epochs. "
                      f"Best Val: {best_val:.3f}")
                break
    
    # Load best model    
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses

def evaluate_model(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()
    mse_loss, mae_loss = 0, 0
    test_samples = loc_test.shape[0]
    ap_num, ue_num, loc, A, P, true_value = AP_num_test.to(device),\
        UE_num_test.to(device), loc_test_norm.to(device), A_test.to(device),\
            P_test.to(device), signal_test.to(device)
    
    with torch.no_grad():
        mae_list = []
        for i in range(test_samples):
            edge_index, edge_attr = model.create_edges(loc[i], ap_num[i], ue_num[i],
                                                       A[i], P[i])
            pred_value_norm = model(loc[i], edge_index, edge_attr)
            pred_value = pred_value_norm * signal_std + signal_mean

            mse_loss += criterion(pred_value, true_value[i, :ue_num[i]])
            mae = pred_value - true_value[i, :ue_num[i]]
            mae_loss += torch.mean(torch.abs(mae))

            row = torch.full((50,), float('nan'), device='cpu')
            row[:int(ue_num[i])] = mae.detach().cpu()
            mae_list.append(row)

    return mse_loss.item() / test_samples, mae_loss.item() / test_samples,\
           torch.stack(mae_list, dim=0)

def test_model(model, test_idx=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    loc = loc_test_norm[test_idx].to(device)
    ap_num = AP_num_test[test_idx].to(device)
    ue_num = UE_num_test[test_idx].to(device)
    A = A_test[test_idx].to(device)
    P = P_test[test_idx].to(device)
    true_value = signal_test[test_idx][:ue_num]
    
    with torch.no_grad():
        edge_index, edge_attr = model.create_edges(loc, ap_num, ue_num, A, P)
        pred_value_norm = model(loc, edge_index, edge_attr)
        pred_value = pred_value_norm * signal_std + signal_mean

    pred_value = pred_value.detach().cpu()
    error = pred_value - true_value

    print("True signal:", true_value.numpy(), "dB")
    print("Pred signal:", pred_value.numpy(), "dB")
    print("Abs Error  :", np.abs(error.numpy()), "dB")

# main
if __name__ == "__main__":

    model = SignalModel()
    model_path = 'model/signal_map.pth'
    
    if os.path.exists(model_path):
        pass

    else:
        train_loader, val_loader = split_train_val(signal_norm, 0.8, 0.2)
        # Train model
        model, train_losses, val_losses = train_model(model, train_loader, val_loader,
                                                      num_epochs=1500)
        # save loss
        np.save('loss/signal_train_losses.npy', train_losses)
        np.save('loss/signal_val_losses.npy', val_losses)

        # Save model
        torch.save(model.state_dict(), model_path)

    # Test model
    model.load_state_dict(torch.load(model_path, weights_only=True))
    mse_loss, mae_loss, mae_mat = evaluate_model(model)
    mae_flatten = mae_mat.numpy().flatten()
    mae_flatten = mae_flatten[~np.isnan(mae_flatten)]
    np.save('result/pred/pred_signal_mae.npy', mae_flatten)
    print(f"Test MSE: {mse_loss:.3f}, "
          f"Test RMSE: {np.sqrt(mse_loss):.3f}, "
          f"Test MAE: {mae_loss:.3f}")
    test_model(model, test_idx=0)
