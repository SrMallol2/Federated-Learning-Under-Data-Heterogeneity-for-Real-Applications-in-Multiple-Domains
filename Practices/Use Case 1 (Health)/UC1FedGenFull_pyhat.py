"""
UC1FedGenFull_pyhat.py
─────────────────────────────────────────────────────────────────────────────
Full-sharing analogue of UC1FedGenPartial_pyhat.py.

Same paper-faithful p̂(y) treatment as the partial file (see that file's
header for the full nuance list); the only structural change is that the
entire model θ = [θ^f; θ^p] is averaged each round (FedAvg on the full
model), and evaluation uses the global model end-to-end (no per-client
encoders).

This file only contains the zhu (KL-ensemble KD) variants — those are the
ones where p̂(y) materially changes the math. The hard-CE and GMM variants
in UC1FedGenFull.py are not reproduced here because (a) the GMM sampler has
its own balanced-per-class sampling logic and (b) the hard-CE variants would
just inherit p̂(y) without illustrating any new dynamic.

Generator-side caveat: ``update_generator_full`` in UC1FLUtils.py also
samples ŷ for its own CE loss. Strictly faithful would mean using p̂(y)
there too; that fix is out of scope for this test (would require editing
UC1FLUtils.py).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from UC1FLUtils import (
    MLP, Generator,
    fed_avg, make_criterion, evaluate_global,
    _compute_local_prototypes, _compute_local_medoid_proxy,
    _aggregate_prototypes,
    update_generator_full, update_generator_full_pyhat,
    N_CLIENTS, FL_ROUNDS, LOCAL_EPOCHS, PATIENCE, NOISE_DIM, LAMBDA_PROTO,
    device,
)

NUM_CLASSES = 2


# ═════════════════════════════════════════════════════════════════════════════
# Helpers (identical to partial counterpart — kept local to avoid cross-file
# coupling on the test side)
# ═════════════════════════════════════════════════════════════════════════════

def _label_counter(y_train, num_classes=NUM_CLASSES):
    y_np = y_train.numpy() if isinstance(y_train, torch.Tensor) else np.asarray(y_train)
    counts = np.bincount(y_np.astype(np.int64), minlength=num_classes).astype(np.float64)
    return torch.tensor(counts, dtype=torch.float32)


def _aggregate_phat(c_list, num_classes=NUM_CLASSES, eps=1e-8):
    total = torch.stack(c_list).sum(dim=0)
    total = total + eps
    return total / total.sum()


def _eval_full(global_model, clients, n_clients):
    global_model.eval()
    vp, vy, tp, ty = [], [], [], []
    with torch.no_grad():
        for i in range(n_clients):
            vp.append(torch.softmax(global_model(clients[i]['X_val'].to(device)),
                                    dim=1)[:, 1].cpu().numpy())
            vy.append(clients[i]['y_val'].numpy())
            tp.append(torch.softmax(global_model(clients[i]['X_test'].to(device)),
                                    dim=1)[:, 1].cpu().numpy())
            ty.append(clients[i]['y_test'].numpy())
    val_auc  = roc_auc_score(np.concatenate(vy), np.concatenate(vp))
    test_auc = roc_auc_score(np.concatenate(ty), np.concatenate(tp))
    return val_auc, test_auc


# ═════════════════════════════════════════════════════════════════════════════
# Local training
# ═════════════════════════════════════════════════════════════════════════════

def local_train_fedgen_zhu_pyhat_full(model, generator, all_predictor_states, p_hat_y,
                                       X_train, y_train, X_val, y_val,
                                       device, batch_size=128, lr=1e-3,
                                       local_epochs=LOCAL_EPOCHS, noise_dim=NOISE_DIM):
    """
    Full-model update with KL-ensemble KD and ŷ ~ p̂(y).

    Difference vs. local_train_fedgen_zhu_full (baseline):
        y_hat = torch.randint(0, 2, ...)                              # baseline
        y_hat = torch.multinomial(p_hat_y, n_syn, replacement=True)   # this file
    """
    loader      = DataLoader(TensorDataset(X_train, y_train),
                             batch_size=batch_size, shuffle=True, drop_last=True)
    criterion_r = make_criterion(y_train.numpy(), device)
    opt_full    = torch.optim.Adam(model.parameters(),           lr=lr)
    opt_gr      = torch.optim.Adam(model.predictor.parameters(), lr=lr)

    all_W = [sd['weight'].to(device) for sd in all_predictor_states]
    all_b = [sd['bias'].to(device)   for sd in all_predictor_states]

    p_hat_dev = p_hat_y.to(device)

    best_val_auc = 0.0
    best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    for _ in range(local_epochs):
        model.train()
        n_syn = min(batch_size, len(X_train))
        with torch.no_grad():
            y_hat = torch.multinomial(p_hat_dev, n_syn, replacement=True)
            eps   = torch.randn(n_syn, noise_dim, device=device)
            Z_hat = generator(y_hat, eps)
            ensemble_probs = torch.stack([
                torch.softmax(Z_hat @ W.T + b, dim=1)
                for W, b in zip(all_W, all_b)
            ]).mean(dim=0)

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_full.zero_grad()
            criterion_r(model(xb), yb).backward()
            opt_full.step()

        opt_gr.zero_grad()
        log_local = F.log_softmax(model.predictor(Z_hat.detach()), dim=1)
        loss_kd   = -(ensemble_probs.detach() * log_local).sum(dim=1).mean()
        loss_kd.backward()
        opt_gr.step()

        model.eval()
        with torch.no_grad():
            vp = torch.softmax(model(X_val.to(device)), dim=1)[:, 1].cpu().numpy()
        if len(np.unique(y_val.numpy())) > 1:
            auc = roc_auc_score(y_val.numpy(), vp)
            if auc > best_val_auc:
                best_val_auc = auc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    per_class_stats = _compute_local_prototypes(model, X_train, y_train, device)
    c_k = _label_counter(y_train)
    return best_state, len(X_train), per_class_stats, c_k


def local_train_fedgen_zhu_pyhat_full_medoid(model, generator, all_predictor_states, p_hat_y,
                                              X_train, y_train, X_val, y_val,
                                              device, batch_size=128, lr=1e-3,
                                              local_epochs=LOCAL_EPOCHS, noise_dim=NOISE_DIM):
    """KL-ensemble KD + medoid anchor + p̂(y), full-model variant."""
    loader      = DataLoader(TensorDataset(X_train, y_train),
                             batch_size=batch_size, shuffle=True, drop_last=True)
    criterion_r = make_criterion(y_train.numpy(), device)
    opt_full    = torch.optim.Adam(model.parameters(),           lr=lr)
    opt_gr      = torch.optim.Adam(model.predictor.parameters(), lr=lr)

    all_W = [sd['weight'].to(device) for sd in all_predictor_states]
    all_b = [sd['bias'].to(device)   for sd in all_predictor_states]

    p_hat_dev = p_hat_y.to(device)

    best_val_auc = 0.0
    best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    for _ in range(local_epochs):
        model.train()
        n_syn = min(batch_size, len(X_train))
        with torch.no_grad():
            y_hat = torch.multinomial(p_hat_dev, n_syn, replacement=True)
            eps   = torch.randn(n_syn, noise_dim, device=device)
            Z_hat = generator(y_hat, eps)
            ensemble_probs = torch.stack([
                torch.softmax(Z_hat @ W.T + b, dim=1)
                for W, b in zip(all_W, all_b)
            ]).mean(dim=0)

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_full.zero_grad()
            criterion_r(model(xb), yb).backward()
            opt_full.step()

        opt_gr.zero_grad()
        log_local = F.log_softmax(model.predictor(Z_hat.detach()), dim=1)
        loss_kd   = -(ensemble_probs.detach() * log_local).sum(dim=1).mean()
        loss_kd.backward()
        opt_gr.step()

        model.eval()
        with torch.no_grad():
            vp = torch.softmax(model(X_val.to(device)), dim=1)[:, 1].cpu().numpy()
        if len(np.unique(y_val.numpy())) > 1:
            auc = roc_auc_score(y_val.numpy(), vp)
            if auc > best_val_auc:
                best_val_auc = auc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    per_class_stats = _compute_local_medoid_proxy(model, X_train, y_train, device)
    c_k = _label_counter(y_train)
    return best_state, len(X_train), per_class_stats, c_k


# ═════════════════════════════════════════════════════════════════════════════
# Run loop
# ═════════════════════════════════════════════════════════════════════════════

def _run_fedgen_full_pyhat_generic(clients, input_dim, seed,
                                    hidden_dim, dropout, lr, batch_size,
                                    local_train_fn,
                                    n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                                    local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                                    lambda_proto=LAMBDA_PROTO,
                                    noise_dim=NOISE_DIM,
                                    variant_name='fedgen_zhu_pyhat_full'):
    """
    Run loop for full-sharing zhu variants with paper-faithful p̂(y).
    Maintains and broadcasts p_hat_y across rounds; logs its trajectory.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    latent_dim   = hidden_dim // 4
    global_model = MLP(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    generator_   = Generator(latent_dim, noise_dim).to(device)

    n_full      = sum(p.numel() for p in global_model.parameters())
    n_gen       = sum(p.numel() for p in generator_.parameters())
    bytes_round = (2 * n_full + n_gen) * 4 * n_clients + NUM_CLASSES * 4 * n_clients

    best_val_auc, best_state, best_gen_state, no_improve = 0.0, None, None, 0
    history, cumul_mb, total_bytes = [], [], 0

    p_hat_y = torch.full((NUM_CLASSES,), 1.0 / NUM_CLASSES)

    all_predictor_states = [
        {k: v.clone() for k, v in global_model.predictor.state_dict().items()}
        for _ in range(n_clients)
    ]

    for fl_round in range(fl_rounds):
        state_dicts, gr_states, counts, client_stats, client_c = [], [], [], [], []

        for i in range(n_clients):
            local = MLP(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
            local.load_state_dict(global_model.state_dict())

            sd, n, stats, c_k = local_train_fn(
                local, generator_, all_predictor_states, p_hat_y,
                clients[i]['X_train'], clients[i]['y_train'],
                clients[i]['X_val'],   clients[i]['y_val'],
                device, batch_size=batch_size, lr=lr,
                local_epochs=local_epochs, noise_dim=noise_dim,
            )
            state_dicts.append(sd)
            gr_states.append({
                'weight': sd['predictor.weight'].clone(),
                'bias':   sd['predictor.bias'].clone(),
            })
            counts.append(n)
            client_stats.append(stats)
            client_c.append(c_k)

        # ── Server aggregation (line 13) ───────────────────────────────────
        global_model.load_state_dict(fed_avg(state_dicts, counts))
        all_predictor_states = gr_states
        p_hat_y = _aggregate_phat(client_c)

        global_prototypes = _aggregate_prototypes(client_stats)
        # NOTE: update_generator_full() also samples ŷ uniformly internally.
        # Strictly faithful FedGen would inject p̂(y) here too. Out of scope.
        update_generator_full_pyhat(generator_, state_dicts, global_prototypes, device,
                                        p_hat_y=p_hat_y,
                              noise_dim=noise_dim, lambda_proto=lambda_proto)
        total_bytes += bytes_round

        val_auc, test_auc = _eval_full(global_model, clients, n_clients)
        history.append({
            'val': val_auc,
            'test': test_auc,
            'p_hat_y': p_hat_y.tolist(),
        })
        cumul_mb.append(total_bytes / (1024 ** 2))
        print(f'  [{variant_name}] R{fl_round+1:02d}: val={val_auc:.4f} '
              f'test={test_auc:.4f} p̂={[f"{x:.3f}" for x in p_hat_y.tolist()]} '
              f'cumul={cumul_mb[-1]:.2f}MB')

        if val_auc > best_val_auc:
            best_val_auc   = val_auc
            best_state     = {k: v.clone() for k, v in global_model.state_dict().items()}
            best_gen_state = {k: v.clone() for k, v in generator_.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'  Early stopping at round {fl_round+1}.')
                break

    global_model.load_state_dict(best_state)
    global_auc, per_client = evaluate_global(global_model, clients)
    return global_auc, per_client, history, cumul_mb


def run_fedgen_zhu_pyhat_full(clients, input_dim, seed,
                               hidden_dim, dropout, lr, batch_size,
                               n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                               local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                               noise_dim=NOISE_DIM):
    """zhu KL-ensemble KD + centroid + p̂(y), full model sharing."""
    return _run_fedgen_full_pyhat_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_zhu_pyhat_full,
        n_clients=n_clients, fl_rounds=fl_rounds,
        local_epochs=local_epochs, patience=patience,
        noise_dim=noise_dim, variant_name='fedgen_zhu_pyhat_full',
    )


def run_fedgen_zhu_pyhat_full_medoid(clients, input_dim, seed,
                                      hidden_dim, dropout, lr, batch_size,
                                      n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                                      local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                                      noise_dim=NOISE_DIM):
    """zhu KL-ensemble KD + medoid + p̂(y), full model sharing."""
    return _run_fedgen_full_pyhat_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_zhu_pyhat_full_medoid,
        n_clients=n_clients, fl_rounds=fl_rounds,
        local_epochs=local_epochs, patience=patience,
        noise_dim=noise_dim, variant_name='fedgen_zhu_pyhat_full_medoid',
    )


def run_fedgen_zhu_pyhat_full_no_proto(clients, input_dim, seed,
                                        hidden_dim, dropout, lr, batch_size,
                                        n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                                        local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                                        noise_dim=NOISE_DIM):
    """Cleanest paper-faithful full variant: KL-ensemble KD + p̂(y), no prototype."""
    return _run_fedgen_full_pyhat_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_zhu_pyhat_full,
        n_clients=n_clients, fl_rounds=fl_rounds,
        local_epochs=local_epochs, patience=patience,
        noise_dim=noise_dim, lambda_proto=0.0,
        variant_name='fedgen_zhu_pyhat_full_no_proto',
    )
