"""
UC1FedGenFull.py
─────────────────────────────────────────────────────────────────────────────
Full-sharing FedGen variants.

Both the feature extractor and the predictor head are averaged across
clients every round (standard FedAvg on the full model), along with the
lightweight generator (or GMM statistics). Evaluation uses the global
model end-to-end, no local encoders.

Variants
--------
    fedgen_full              — hard CE KD    + centroid anchor
    fedgen_full_medoid       — hard CE KD    + medoid anchor
    fedgen_zhu_full          — KL ensemble KD + centroid anchor   (Zhu Algorithm 1)
    fedgen_zhu_full_medoid   — KL ensemble KD + medoid anchor
    fedgen_gmm_full          — hard CE KD    + GMM sampler (no neural generator)

Every variant exposes `local_train_<variant>` and `run_<variant>`. Signatures
mirror UC1FedGenPartial.py so the notebook loop can treat partial/full pairs
interchangeably.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

from UC1FLUtils import (
    MLP, Generator, GMMSampler,
    fed_avg, make_criterion, evaluate_global,
    _compute_local_prototypes, _compute_local_medoid_proxy,
    _aggregate_prototypes, compute_client_distribution,
    update_generator_full,
    N_CLIENTS, FL_ROUNDS, LOCAL_EPOCHS, PATIENCE, NOISE_DIM, LAMBDA_PROTO,
    device,
)


# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════════════

def _eval_full(global_model, clients, n_clients):
    """
    Full-sharing evaluation: single global model over all clients' test data.
    Returns (val_auc, test_auc) on concatenated patient-level scores.
    """
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
# Local training — one function per variant
# ═════════════════════════════════════════════════════════════════════════════

def local_train_fedgen_full(model, generator, X_train, y_train, X_val, y_val,
                             device, batch_size=128, lr=1e-3,
                             local_epochs=LOCAL_EPOCHS, noise_dim=NOISE_DIM):
    loader       = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    criterion_r  = make_criterion(y_train.numpy(), device)
    criterion_kd = nn.CrossEntropyLoss()
    opt_full     = torch.optim.Adam(model.parameters(),           lr=lr)
    opt_gr       = torch.optim.Adam(model.predictor.parameters(), lr=lr)

    best_val_auc = 0.0
    best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    for _ in range(local_epochs):
        model.train()
        n_syn = min(batch_size, len(X_train))
        with torch.no_grad():
            y_hat = torch.randint(0, 2, (n_syn,), device=device)
            eps   = torch.randn(n_syn, noise_dim, device=device)
            Z_hat = generator(y_hat, eps)

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_full.zero_grad()
            criterion_r(model(xb), yb).backward()
            opt_full.step()
            opt_gr.zero_grad()
            criterion_kd(model.predictor(Z_hat), y_hat).backward()
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
    return best_state, len(X_train), per_class_stats


def local_train_fedgen_full_medoid(model, generator, X_train, y_train, X_val, y_val,
                                    device, batch_size=128, lr=1e-3,
                                    local_epochs=LOCAL_EPOCHS, noise_dim=NOISE_DIM):
    """Identical to local_train_fedgen_full except the anchor is a medoid."""
    loader       = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    criterion_r  = make_criterion(y_train.numpy(), device)
    criterion_kd = nn.CrossEntropyLoss()
    opt_full     = torch.optim.Adam(model.parameters(),           lr=lr)
    opt_gr       = torch.optim.Adam(model.predictor.parameters(), lr=lr)

    best_val_auc = 0.0
    best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    for _ in range(local_epochs):
        model.train()
        n_syn = min(batch_size, len(X_train))
        with torch.no_grad():
            y_hat = torch.randint(0, 2, (n_syn,), device=device)
            eps   = torch.randn(n_syn, noise_dim, device=device)
            Z_hat = generator(y_hat, eps)

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_full.zero_grad()
            criterion_r(model(xb), yb).backward()
            opt_full.step()
            opt_gr.zero_grad()
            criterion_kd(model.predictor(Z_hat), y_hat).backward()
            opt_gr.step()

        model.eval()
        with torch.no_grad():
            vp = torch.softmax(model(X_val.to(device)), dim=1)[:, 1].cpu().numpy()
        if len(np.unique(y_val.numpy())) > 1:
            auc = roc_auc_score(y_val.numpy(), vp)
            if auc > best_val_auc:
                best_val_auc = auc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    # ← medoid anchor
    per_class_stats = _compute_local_medoid_proxy(model, X_train, y_train, device)
    return best_state, len(X_train), per_class_stats


def local_train_fedgen_zhu_full(model, generator, all_predictor_states,
                                 X_train, y_train, X_val, y_val,
                                 device, batch_size=128, lr=1e-3,
                                 local_epochs=LOCAL_EPOCHS, noise_dim=NOISE_DIM):
    """
    Full-model update with paper-faithful KL-ensemble KD loss (Zhu Algorithm 1).
    Both feature extractor and predictor are shared and averaged each round.
    """
    loader      = DataLoader(TensorDataset(X_train, y_train),
                             batch_size=batch_size, shuffle=True, drop_last=True)
    criterion_r = make_criterion(y_train.numpy(), device)
    opt_full    = torch.optim.Adam(model.parameters(),           lr=lr)
    opt_gr      = torch.optim.Adam(model.predictor.parameters(), lr=lr)

    all_W = [sd['weight'].to(device) for sd in all_predictor_states]
    all_b = [sd['bias'].to(device)   for sd in all_predictor_states]

    best_val_auc = 0.0
    best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    for _ in range(local_epochs):
        model.train()
        n_syn = min(batch_size, len(X_train))
        with torch.no_grad():
            y_hat = torch.randint(0, 2, (n_syn,), device=device)
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
    return best_state, len(X_train), per_class_stats


def local_train_fedgen_zhu_full_medoid(model, generator, all_predictor_states,
                                        X_train, y_train, X_val, y_val,
                                        device, batch_size=128, lr=1e-3,
                                        local_epochs=LOCAL_EPOCHS, noise_dim=NOISE_DIM):
    """KL-ensemble KD + medoid anchor. Differs from _zhu_full only in the proto call."""
    loader      = DataLoader(TensorDataset(X_train, y_train),
                             batch_size=batch_size, shuffle=True, drop_last=True)
    criterion_r = make_criterion(y_train.numpy(), device)
    opt_full    = torch.optim.Adam(model.parameters(),           lr=lr)
    opt_gr      = torch.optim.Adam(model.predictor.parameters(), lr=lr)

    all_W = [sd['weight'].to(device) for sd in all_predictor_states]
    all_b = [sd['bias'].to(device)   for sd in all_predictor_states]

    best_val_auc = 0.0
    best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    for _ in range(local_epochs):
        model.train()
        n_syn = min(batch_size, len(X_train))
        with torch.no_grad():
            y_hat = torch.randint(0, 2, (n_syn,), device=device)
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

    # ← medoid anchor
    per_class_stats = _compute_local_medoid_proxy(model, X_train, y_train, device)
    return best_state, len(X_train), per_class_stats


def local_train_fedgen_gmm_full(model, gmm_sampler,
                                 X_train, y_train, X_val, y_val,
                                 device, batch_size=128, lr=1e-3,
                                 local_epochs=LOCAL_EPOCHS):
    """
    Full-model training with GMMSampler synthetic samples. No neural generator,
    no prototype constraint — the GMM is updated server-side from client
    latent stats each round.
    """
    loader       = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    criterion_r  = make_criterion(y_train.numpy(), device)
    criterion_kd = nn.CrossEntropyLoss()
    opt_full     = torch.optim.Adam(model.parameters(),           lr=lr)
    opt_gr       = torch.optim.Adam(model.predictor.parameters(), lr=lr)

    best_val_auc = 0.0
    best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    for _ in range(local_epochs):
        model.train()
        n_per_class = max(1, min(batch_size // 2, len(X_train) // 2))
        with torch.no_grad():
            Z_hat, y_hat = gmm_sampler.sample(n_per_class)

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_full.zero_grad()
            criterion_r(model(xb), yb).backward()
            opt_full.step()
            opt_gr.zero_grad()
            criterion_kd(model.predictor(Z_hat), y_hat).backward()
            opt_gr.step()

        model.eval()
        with torch.no_grad():
            vp = torch.softmax(model(X_val.to(device)), dim=1)[:, 1].cpu().numpy()
        if len(np.unique(y_val.numpy())) > 1:
            auc = roc_auc_score(y_val.numpy(), vp)
            if auc > best_val_auc:
                best_val_auc = auc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    return best_state, len(X_train)  # GMM run loop fetches stats separately


# ═════════════════════════════════════════════════════════════════════════════
# Run loops
# ═════════════════════════════════════════════════════════════════════════════

def _run_fedgen_full_generic(clients, input_dim, seed,
                              hidden_dim, dropout, lr, batch_size,
                              local_train_fn, use_ensemble_teacher,
                              n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                              local_epochs=LOCAL_EPOCHS, patience=PATIENCE,lambda_proto=LAMBDA_PROTO,
                              noise_dim=NOISE_DIM, variant_name='fedgen_full'):
    """
    Unified run loop for the 4 neural-generator full variants.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    latent_dim   = hidden_dim // 4
    global_model = MLP(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    generator_   = Generator(latent_dim, noise_dim).to(device)

    n_full      = sum(p.numel() for p in global_model.parameters())
    n_gen       = sum(p.numel() for p in generator_.parameters())
    bytes_round = (2 * n_full + n_gen) * 4 * n_clients

    best_val_auc, best_state, best_gen_state, no_improve = 0.0, None, None, 0
    history, cumul_mb, total_bytes = [], [], 0

    all_predictor_states = [
        {k: v.clone() for k, v in global_model.predictor.state_dict().items()}
        for _ in range(n_clients)
    ] if use_ensemble_teacher else None

    for fl_round in range(fl_rounds):
        state_dicts, gr_states, counts, client_stats = [], [], [], []

        for i in range(n_clients):
            local = MLP(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
            local.load_state_dict(global_model.state_dict())

            if use_ensemble_teacher:
                sd, n, stats = local_train_fn(
                    local, generator_, all_predictor_states,
                    clients[i]['X_train'], clients[i]['y_train'],
                    clients[i]['X_val'],   clients[i]['y_val'],
                    device, batch_size=batch_size, lr=lr,
                    local_epochs=local_epochs, noise_dim=noise_dim,
                )
            else:
                sd, n, stats = local_train_fn(
                    local, generator_,
                    clients[i]['X_train'], clients[i]['y_train'],
                    clients[i]['X_val'],   clients[i]['y_val'],
                    device, batch_size=batch_size, lr=lr,
                    local_epochs=local_epochs, noise_dim=noise_dim,
                )
            state_dicts.append(sd)
            # Extract predictor slice for next-round ensemble teacher
            gr_states.append({
                'weight': sd['predictor.weight'].clone(),
                'bias':   sd['predictor.bias'].clone(),
            })
            counts.append(n)
            client_stats.append(stats)

        global_model.load_state_dict(fed_avg(state_dicts, counts))
        if use_ensemble_teacher:
            all_predictor_states = gr_states

        global_prototypes = _aggregate_prototypes(client_stats)
        update_generator_full(generator_, state_dicts, global_prototypes, device,
                              noise_dim=noise_dim, lambda_proto=LAMBDA_PROTO)
        total_bytes += bytes_round

        val_auc, test_auc = _eval_full(global_model, clients, n_clients)
        history.append({'val': val_auc, 'test': test_auc})
        cumul_mb.append(total_bytes / (1024 ** 2))
        print(f'  [{variant_name}] R{fl_round+1:02d}: val={val_auc:.4f} '
              f'test={test_auc:.4f} cumul={cumul_mb[-1]:.2f}MB')

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


def run_fedgen_full(clients, input_dim, seed,
                    hidden_dim, dropout, lr, batch_size,
                    n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                    local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                    noise_dim=NOISE_DIM):
    return _run_fedgen_full_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_full,
        use_ensemble_teacher=False,
        n_clients=n_clients, fl_rounds=fl_rounds,
        local_epochs=local_epochs, patience=patience,
        noise_dim=noise_dim, variant_name='fedgen_full',
    )


def run_fedgen_full_medoid(clients, input_dim, seed,
                            hidden_dim, dropout, lr, batch_size,
                            n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                            local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                            noise_dim=NOISE_DIM):
    return _run_fedgen_full_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_full_medoid,
        use_ensemble_teacher=False,
        n_clients=n_clients, fl_rounds=fl_rounds,
        local_epochs=local_epochs, patience=patience,
        noise_dim=noise_dim, variant_name='fedgen_full_medoid',
    )


def run_fedgen_zhu_full(clients, input_dim, seed,
                         hidden_dim, dropout, lr, batch_size,
                         n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                         local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                         noise_dim=NOISE_DIM):
    return _run_fedgen_full_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_zhu_full,
        use_ensemble_teacher=True,
        n_clients=n_clients, fl_rounds=fl_rounds,
        local_epochs=local_epochs, patience=patience,
        noise_dim=noise_dim, variant_name='fedgen_zhu_full',
    )


def run_fedgen_zhu_full_medoid(clients, input_dim, seed,
                                hidden_dim, dropout, lr, batch_size,
                                n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                                local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                                noise_dim=NOISE_DIM):
    return _run_fedgen_full_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_zhu_full_medoid,
        use_ensemble_teacher=True,
        n_clients=n_clients, fl_rounds=fl_rounds,
        local_epochs=local_epochs, patience=patience,
        noise_dim=noise_dim, variant_name='fedgen_zhu_full_medoid',
    )

def run_fedgen_zhu_full_no_proto(clients, input_dim, seed,
                                  hidden_dim, dropout, lr, batch_size,
                                  n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                                  local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                                  noise_dim=NOISE_DIM):
    """Full-sharing counterpart of fedgen_zhu_no_proto."""
    return _run_fedgen_full_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_zhu_full,
        use_ensemble_teacher=True,
        n_clients=n_clients, fl_rounds=fl_rounds,
        local_epochs=local_epochs, patience=patience,
        noise_dim=noise_dim, lambda_proto=0.0,
        variant_name='fedgen_zhu_full_no_proto',
    )


def run_fedgen_gmm_full(clients, input_dim, seed,
                         hidden_dim, dropout, lr, batch_size,
                         n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                         local_epochs=LOCAL_EPOCHS, patience=PATIENCE):
    """
    Full model sharing + GMMSampler (no neural generator).
    Mirrors fedgen_gmm but shares the full model each round.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    latent_dim   = hidden_dim // 4
    global_model = MLP(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    gmm          = GMMSampler(latent_dim, device)

    n_full      = sum(p.numel() for p in global_model.parameters())
    bytes_round = 2 * n_full * 4 * n_clients + gmm.param_bytes() * n_clients

    best_val_auc, best_state, no_improve = 0.0, None, 0
    history, cumul_mb, total_bytes = [], [], 0

    for fl_round in range(fl_rounds):
        state_dicts, counts, client_dists = [], [], []

        for i in range(n_clients):
            local = MLP(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
            local.load_state_dict(global_model.state_dict())

            sd, n = local_train_fedgen_gmm_full(
                local, gmm,
                clients[i]['X_train'], clients[i]['y_train'],
                clients[i]['X_val'],   clients[i]['y_val'],
                device, batch_size=batch_size, lr=lr,
                local_epochs=local_epochs,
            )
            dist = compute_client_distribution(
                local, clients[i]['X_train'],
                clients[i]['y_train'].numpy(), device,
            )
            state_dicts.append(sd)
            counts.append(n)
            client_dists.append(dist)

        global_model.load_state_dict(fed_avg(state_dicts, counts))
        gmm.update(client_dists, counts)
        total_bytes += bytes_round

        val_auc, test_auc = _eval_full(global_model, clients, n_clients)
        history.append({'val': val_auc, 'test': test_auc})
        cumul_mb.append(total_bytes / (1024 ** 2))
        print(f'  [fedgen_gmm_full] R{fl_round+1:02d}: val={val_auc:.4f} '
              f'test={test_auc:.4f} cumul={cumul_mb[-1]:.2f}MB')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.clone() for k, v in global_model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'  Early stopping at round {fl_round+1}.')
                break

    global_model.load_state_dict(best_state)
    global_auc, per_client = evaluate_global(global_model, clients)
    return global_auc, per_client, history, cumul_mb
