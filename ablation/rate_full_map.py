
# predict rate
import os
import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from process import *
from signal_map import SignalModel
from interf_map import InterfModel

# Combined models for rate prediction
class RateModel(nn.Module):
    def __init__(self, signal_model_path='model/signal_map.pth', 
                       interf_model_path='model/interf_full_map.pth',
                       signal_finetune=True, interf_finetune=True, snr=15):
        super().__init__()
        self.snr = snr
       
        # Load models
        self.signal_model = SignalModel()
        self.interf_model = InterfModel()
        
        # A missing checkpoint is a hard error: this stage fine-tunes pre-trained
        # branches, and silently falling back to random weights yields a run that
        # trains and logs normally without being the experiment it claims to be.
        for module, path, what in ((self.signal_model, signal_model_path, 'signal'),
                                   (self.interf_model, interf_model_path, 'interference')):
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f'{what} checkpoint not found: {path!r}')
            module.load_state_dict(torch.load(path, weights_only=True,
                                              map_location='cpu'))


        for param in self.signal_model.parameters():
            param.requires_grad = False
        for param in self.interf_model.parameters():
            param.requires_grad = False

        if signal_finetune:

            for param in self.signal_model.mlp.parameters():
                param.requires_grad = True
            # for param in self.signal_model.ap2ue[-1].parameters():
            #     param.requires_grad = True
            # for param in self.signal_model.norm[-1].parameters():
            #     param.requires_grad = True
            for param in self.signal_model.ffn_ue.parameters():
                param.requires_grad = True
            for param in self.signal_model.norm_ffn.parameters():
                param.requires_grad = True

        #     print("Signal model parameters tunable") 
        # else:
        #     print("Signal model parameters frozen")
            
        if interf_finetune:

            for param in self.interf_model.mlp.parameters():
                param.requires_grad = True
            # for param in self.interf_model.ap2ue[-1].parameters():
            #     param.requires_grad = True
            # for param in self.interf_model.ue2ap[-1].parameters():
            #     param.requires_grad = True
            # for param in self.interf_model.norm_ap[-1].parameters():
            #     param.requires_grad = True
            # for param in self.interf_model.norm_ue[-1].parameters():
            #     param.requires_grad = True
            for param in self.interf_model.ffn_ap.parameters():
                param.requires_grad = True
            for param in self.interf_model.ffn_ue.parameters():
                param.requires_grad = True
            for param in self.interf_model.norm_ffn_ap.parameters():
                param.requires_grad = True
            for param in self.interf_model.norm_ffn_ue.parameters():
                param.requires_grad = True
            
        #     print("Interf model parameters tunable")
        # else:
        #     print("Interf model parameters frozen")
            
        # Store normalization parameters
        self.signal_mean = signal_mean
        self.signal_std = signal_std
        self.interf_mean = interf_mean
        self.interf_std = interf_std
        self.rate_mean = rate_mean
        self.rate_std = rate_std
        
    def forward(self, x, ap_num, ue_num, A, P):
        # Create edges for signal model
        signal_edge_index, signal_edge_attr = self.signal_model.create_edges(x, ap_num,
                                                                          ue_num, A, P)

        # Create edges for interference model
        interf_edge1_index, interf_edge1_attr, interf_edge2_index, interf_edge2_attr = \
            self.interf_model.create_edges(x, ap_num, ue_num, A, P)
        
        # Get signal predictions (normalized)
        pred_signal_norm = self.signal_model(x, signal_edge_index, signal_edge_attr)
        
        # Get interference predictions (normalized)
        pred_interf_norm = self.interf_model(x, interf_edge1_index, interf_edge1_attr, 
                                             interf_edge2_index, interf_edge2_attr)
        
        # Denormalize predictions to get actual dB values
        pred_signal = pred_signal_norm * self.signal_std + self.signal_mean
        pred_interf = pred_interf_norm * self.interf_std + self.interf_mean
        
        # Convert from dB to linear scale
        signal_linear = torch.pow(10, pred_signal / 10)
        interf_linear = torch.pow(10, pred_interf / 10)
        
        # Noise power
        noise_dB = -87 - self.snr
        noise_power = 10 ** (noise_dB / 10)
        
        # Calculate SINR (Signal to Interference + Noise Ratio)
        SINR = signal_linear / (interf_linear + noise_power)
        
        # Calculate rate using Shannon capacity formula: log2(1 + SINR)
        pred_rate = torch.log2(1 + SINR)
        pred_rate_norm = (pred_rate - self.rate_mean) / self.rate_std
        
        return pred_rate_norm

def train_model(model, train_loader, val_loader, num_epochs=100):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print('device:', torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('device: CPU')
    model = model.to(device)
    
    # Use smaller learning rate for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
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
                pred_value = model(loc[i], ap_num[i], ue_num[i], A[i], P[i])
                loss = criterion(pred_value, true_value[i, :ue_num[i]])
                batch_loss += loss
            
            batch_loss = batch_loss / batch_size
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for ap_num, ue_num, loc, A, P, true_value in val_loader:
                ap_num, ue_num, loc, A, P, true_value = ap_num.to(device), ue_num.to(device),\
                    loc.to(device), A.to(device), P.to(device), true_value.to(device)
                batch_size = loc.shape[0]
                batch_loss = 0
                
                for i in range(batch_size):
                    pred_value = model(loc[i], ap_num[i], ue_num[i], A[i], P[i])
                    loss = criterion(pred_value, true_value[i, :ue_num[i]])
                    batch_loss += loss
                
                val_loss += batch_loss.item() / batch_size
        
        # Record losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if epoch % 1 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Train loss: {avg_train_loss:.4f}, "
                  f"Val loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.4e}")
        
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

def evaluate_model(model, snr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss(beta=0.5)
    # criterion = nn.L1Loss(reduction='mean')

    mse_loss, mae_loss, value_sum = 0, 0, 0
    test_samples = loc_test.shape[0]
    ap_num, ue_num, loc, A, P, true_value = AP_num_test.to(device),\
        UE_num_test.to(device), loc_test_norm.to(device), A_test.to(device),\
            P_test.to(device), rate_test[int(snr/5)].to(device)

    with torch.no_grad():
        mae_list = []
        for i in range(test_samples):
            pred_value = model(loc[i], ap_num[i], ue_num[i], A[i], P[i])
            pred_value = pred_value * rate_std + rate_mean
            mse_loss += criterion(pred_value, true_value[i, :ue_num[i]])
            mae = pred_value - true_value[i, :ue_num[i]]
            mae_loss += torch.mean(torch.abs(mae))
            value_sum += torch.mean(true_value[i, :ue_num[i]])

            row = torch.full((30,), float('nan'), device='cpu')
            row[:int(ue_num[i])] = mae.detach().cpu()
            mae_list.append(row)
            
    return mse_loss.item() / test_samples, mae_loss.item() / test_samples, \
           value_sum.item() / test_samples, torch.stack(mae_list, dim=0)

def test_model(model, snr, test_idx=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    ap_num = AP_num_test[test_idx].to(device)
    ue_num = UE_num_test[test_idx].to(device)
    loc = loc_test_norm[test_idx].to(device)
    A = A_test[test_idx].to(device)
    P = P_test[test_idx].to(device)
    true_value = rate_test[int(snr/5), test_idx, :ue_num]
    
    with torch.no_grad():
        pred_value = model(loc, ap_num, ue_num, A, P)
        pred_value = pred_value * rate_std + rate_mean
    
    pred_value = pred_value.detach().cpu()
    
    print("True rate:", true_value.numpy(), "bits/s/Hz")
    print("Pred rate:", pred_value.numpy(), "bits/s/Hz")
    print("Abs Error:", np.abs(pred_value.numpy() - true_value.numpy()), "bits/s/Hz")

# Main execution
if __name__ == "__main__":

    snr = 15
    model = RateModel(snr=snr)
    model_path = 'model/rate_full_map.pth'

    mse_loss, mae_loss, value, mae_mat = evaluate_model(model, snr)
    mae_flatten = mae_mat.numpy().flatten()
    mae_flatten = mae_flatten[~np.isnan(mae_flatten)]
    np.save('result/pred/pred_rate_full_mae.npy', mae_flatten)
    print(f"Test MSE: {mse_loss:.3f}, "
          f"Test RMSE: {np.sqrt(mse_loss):.3f}, "
          f"Test MAE: {mae_loss:.3f}, "
          f"Test SUM: {value:.3f}")
    test_model(model, snr, test_idx=0)

    if os.path.exists(model_path):
        pass
    
    else:
        train_loader, val_loader = split_train_val(rate_norm[int(snr/5)], 0.8, 0.2)

        # Fine-tune the model
        model, train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                                      num_epochs=500)
        # Save loss
        np.save('loss/rate_full_train_losses.npy', train_losses)
        np.save('loss/rate_full_val_losses.npy', val_losses)

        # Save models
        torch.save(model.state_dict(), model_path)

    # Test model
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
    mse_loss, mae_loss, value, mae_mat = evaluate_model(model, snr)
    mae_flatten = mae_mat.numpy().flatten()
    mae_flatten = mae_flatten[~np.isnan(mae_flatten)]
    np.save('result/pred/pred_rate_full_finetune.npy', mae_flatten)
    print(f"Test MSE: {mse_loss:.3f}, "
          f"Test RMSE: {np.sqrt(mse_loss):.3f}, "
          f"Test MAE: {mae_loss:.3f}, "
          f"Test SUM: {value:.3f}")
    test_model(model, snr, test_idx=0)