# shared helpers for the ablation suite (seeding + CLI + resumable training)
import argparse
import os
import random
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_common_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='override epoch budget (smoke test)')
    parser.add_argument('--snr', type=int, default=15)
    parser.add_argument('--tag', type=str, default='',
                        help='suffix appended to variant name (smoke test isolation)')
    return parser


def variant_name(base, tag, seed):
    if tag:
        base = f'{base}_{tag}'
    return f'{base}_seed{seed}'


# The proposed model's signal branch, per seed, exactly as the reference row of
# result/ablation_table.md is composed (MAIN in ablation/make_ablation_table.py).
#
# Variants that ablate ONLY the interference branch (single, rep, nopow2) must
# fine-tune against these checkpoints. Using anything else -- notably the
# default 'model/signal_map.pth' -- makes their row differ from the reference
# row in two places at once, so it is no longer a single-factor ablation.
# Keep in sync with MAIN in ablation/make_ablation_table.py.
MAIN_SIGNAL = {
    0: 'model/signal_main_seed0.pth',
    1: 'model/signal_main_seed1.pth',
    2: 'model/signal_main_seed2.pth',
}


def main_signal_path(seed):
    # A retrained campaign must pair interference-only variants with ITS OWN
    # main signal branch; the override keeps MAIN_SIGNAL as the default so
    # existing results stay reproducible.
    tmpl = os.environ.get('CF_MAIN_SIGNAL_TMPL')
    if tmpl:
        return tmpl.format(seed=seed)
    if seed not in MAIN_SIGNAL:
        raise SystemExit(f'no reference signal branch registered for seed {seed} '
                         f'(known: {sorted(MAIN_SIGNAL)}); add it to MAIN_SIGNAL '
                         f'in ablation/ablation_common.py')
    return MAIN_SIGNAL[seed]


def load_branch(module, path, what):
    """Load a pre-trained branch into `module`, failing loudly if it is absent.

    The rate stage FINE-TUNES from pre-trained branches. The guard this replaces
    -- `if os.path.exists(path): load(...)` -- silently left the branch randomly
    initialised when the path was wrong, producing a run that trains, logs and
    writes output completely normally while not being the experiment it claims
    to be. That is how rate_single/rate_rep came to be fine-tuned against the
    archived model/signal_map.pth instead of their seed-matched branch.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f'{what} checkpoint not found: {path!r}. The rate stage fine-tunes '
            f'from pre-trained branches and must never start from random weights.')
    module.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))


def save_mae(mae_mat, path):
    flat = mae_mat.numpy().flatten()
    flat = flat[~np.isnan(flat)]
    np.save(path, flat)
    return flat


# forward conventions shared by all model families
def GRAPH_FORWARD(model, loc, ap_num, ue_num, A, P):
    # signal / interf / single / end2end: build edges, then forward
    return model(loc, *model.create_edges(loc, ap_num, ue_num, A, P))


def RATE_FORWARD(model, loc, ap_num, ue_num, A, P):
    return model(loc, ap_num, ue_num, A, P)


def train_resumable(model, train_loader, val_loader, num_epochs, forward_fn,
                    criterion, lr, ckpt_path, save_every=10):
    # same protocol as the main scripts' train_model (AdamW wd=1e-3,
    # ReduceLROnPlateau 0.1/10, early stop 20, best-state restore), plus an
    # epoch-level checkpoint so a hard session cutoff loses at most
    # `save_every` epochs. Rerunning the script resumes automatically; the
    # caller deletes ckpt_path after saving the final model.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print('device:', torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('device: CPU')
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10, min_lr=1e-6)

    start_epoch = 0
    train_losses, val_losses = [], []
    best_val, best_state, no_improve, early_stop = float('inf'), None, 0, 20

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        start_epoch = ck['epoch'] + 1
        train_losses, val_losses = ck['train_losses'], ck['val_losses']
        best_val, best_state = ck['best_val'], ck['best_state']
        no_improve = ck['no_improve']
        print(f"[resume] {ckpt_path}: continuing from epoch {start_epoch} "
              f"(best val {best_val:.4f})")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        for ap_num, ue_num, loc, A, P, true_value in train_loader:
            ap_num, ue_num, loc, A, P, true_value = ap_num.to(device), ue_num.to(device),\
                loc.to(device), A.to(device), P.to(device), true_value.to(device)
            batch_size = loc.shape[0]
            batch_loss = 0

            for i in range(batch_size):
                pred_value = forward_fn(model, loc[i], ap_num[i], ue_num[i],
                                        A[i], P[i])
                loss = criterion(pred_value, true_value[i, :ue_num[i]])
                batch_loss += loss

            batch_loss = batch_loss / batch_size
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for ap_num, ue_num, loc, A, P, true_value in val_loader:
                ap_num, ue_num, loc, A, P, true_value = ap_num.to(device),\
                    ue_num.to(device), loc.to(device), A.to(device), P.to(device),\
                        true_value.to(device)
                batch_size = loc.shape[0]
                batch_loss = 0

                for i in range(batch_size):
                    pred_value = forward_fn(model, loc[i], ap_num[i], ue_num[i],
                                            A[i], P[i])
                    loss = criterion(pred_value, true_value[i, :ue_num[i]])
                    batch_loss += loss

                val_loss += batch_loss.item() / batch_size

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch}/{num_epochs}], Train loss: {avg_train_loss:.4f}, "
              f"Val loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.4e}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val:
            best_val = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        stop = no_improve >= early_stop
        if stop or (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            tmp = ckpt_path + '.tmp'
            torch.save({'epoch': epoch,
                        'model': {k: v.cpu() for k, v in model.state_dict().items()},
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'train_losses': train_losses, 'val_losses': val_losses,
                        'best_val': best_val, 'best_state': best_state,
                        'no_improve': no_improve}, tmp)
            os.replace(tmp, ckpt_path)

        if stop:
            print(f"[EarlyStop] no improvement in {early_stop} epochs. "
                  f"Best Val: {best_val:.3f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


def clear_ckpt(ckpt_path):
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
