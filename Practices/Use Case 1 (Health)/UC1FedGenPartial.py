"""
UC1FedGenPartial.py
─────────────────────────────────────────────────────────────────────────────
Partial-sharing FedGen variants.

Each client keeps its own feature extractor. Only the predictor head and
the lightweight generator (or GMM statistics) are shared each round.
Evaluation uses the per-client local encoder + the global predictor.

Variants
--------
    fedgen_partial          — hard CE KD    + centroid anchor
    fedgen_partial_medoid   — hard CE KD    + medoid anchor
    fedgen_zhu              — KL ensemble KD + centroid anchor   (paper-faithful)
    fedgen_zhu_medoid       — KL ensemble KD + medoid anchor
    fedgen_gmm              — hard CE KD    + GMM sampler (no neural generator)

Every variant exposes `local_train_<variant>` and `run_<variant>`. The run
functions all return `(global_auc, per_client, history, cumul_mb)` so the
notebook loop can treat them interchangeably.

Full-sharing counterparts live in UC1FedGenFull.py.
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
    update_generator,
    N_CLIENTS, FL_ROUNDS, LOCAL_EPOCHS, PATIENCE, NOISE_DIM, LAMBDA_PROTO,
    device,
)


# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════════════

def _eval_partial(global_model, local_models, clients, n_clients):
    """
    Partial-sharing evaluation: per-client local encoder + global predictor.
    Returns (val_auc, test_auc) on concatenated patient-level scores.
    """
    global_model.eval()
    vp, vy, tp, ty = [], [], [], []
    with torch.no_grad():
        for i in range(n_clients):
            z_v = local_models[i].encode(clients[i]['X_val'].to(device))
            z_t = local_models[i].encode(clients[i]['X_test'].to(device))
            vp.append(torch.softmax(global_model.predictor(z_v), dim=1)[:, 1].cpu().numpy())
            vy.append(clients[i]['y_val'].numpy())
            tp.append(torch.softmax(global_model.predictor(z_t), dim=1)[:, 1].cpu().numpy())
            ty.append(clients[i]['y_test'].numpy())
    val_auc  = roc_auc_score(np.concatenate(vy), np.concatenate(vp))
    test_auc = roc_auc_score(np.concatenate(ty), np.concatenate(tp))
    return val_auc, test_auc


# ═════════════════════════════════════════════════════════════════════════════
# 1. fedgen_partial — hard CE KD, centroid anchor
# ═════════════════════════════════════════════════════════════════════════════

def local_train_fedgen_partial(model, generator, X_train, y_train, X_val, y_val,
                                device, batch_size=128, lr=1e-3,
                                local_epochs=LOCAL_EPOCHS, noise_dim=NOISE_DIM):
    loader       = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    criterion_r  = make_criterion(y_train.numpy(), device)
    criterion_kd = nn.CrossEntropyLoss()
    opt_full     = torch.optim.Adam(model.parameters(),           lr=lr)
    opt_gr       = torch.optim.Adam(model.predictor.parameters(), lr=lr)

    best_val_auc  = 0.0
    best_gr_state = {k: v.clone() for k, v in model.predictor.state_dict().items()}

    for _ in range(local_epochs):
        model.train()
        # Fresh synthetic batch each epoch — preserves generator stochasticity
        n_syn = min(batch_size, len(X_train))
        with torch.no_grad():
            y_hat = torch.randint(0, 2, (n_syn,), device=device)
            eps   = torch.randn(n_syn, noise_dim, device=device)
            Z_hat = generator(y_hat, eps)

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            model.zero_grad()
            criterion_r(model(xb), yb).backward()
            opt_full.step()
            model.predictor.zero_grad()
            criterion_kd(model.predictor(Z_hat), y_hat).backward()
            opt_gr.step()

        model.eval()
        with torch.no_grad():
            vp = torch.softmax(model(X_val.to(device)), dim=1)[:, 1].cpu().numpy()
        if len(np.unique(y_val.numpy())) > 1:
            auc = roc_auc_score(y_val.numpy(), vp)
            if auc > best_val_auc:
                best_val_auc  = auc
                best_gr_state = {k: v.clone()
                                 for k, v in model.predictor.state_dict().items()}

    per_class_stats = _compute_local_prototypes(model, X_train, y_train, device)
    return best_gr_state, len(X_train), per_class_stats


def local_train_fedgen_partial_medoid(model, generator, X_train, y_train, X_val, y_val,
                                       device, batch_size=128, lr=1e-3,
                                       local_epochs=LOCAL_EPOCHS, noise_dim=NOISE_DIM):
    """
    Identical to local_train_fedgen_partial except the per-class anchor is a
    medoid (nearest real point to the centroid) rather than the centroid.
    """
    loader       = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    criterion_r  = make_criterion(y_train.numpy(), device)
    criterion_kd = nn.CrossEntropyLoss()
    opt_full     = torch.optim.Adam(model.parameters(),           lr=lr)
    opt_gr       = torch.optim.Adam(model.predictor.parameters(), lr=lr)

    best_val_auc  = 0.0
    best_gr_state = {k: v.clone() for k, v in model.predictor.state_dict().items()}

    for _ in range(local_epochs):
        model.train()
        n_syn = min(batch_size, len(X_train))
        with torch.no_grad():
            y_hat = torch.randint(0, 2, (n_syn,), device=device)
            eps   = torch.randn(n_syn, noise_dim, device=device)
            Z_hat = generator(y_hat, eps)

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            model.zero_grad()
            criterion_r(model(xb), yb).backward()
            opt_full.step()
            model.predictor.zero_grad()
            criterion_kd(model.predictor(Z_hat), y_hat).backward()
            opt_gr.step()

        model.eval()
        with torch.no_grad():
            vp = torch.softmax(model(X_val.to(device)), dim=1)[:, 1].cpu().numpy()
        if len(np.unique(y_val.numpy())) > 1:
            auc = roc_auc_score(y_val.numpy(), vp)
            if auc > best_val_auc:
                best_val_auc  = auc
                best_gr_state = {k: v.clone()
                                 for k, v in model.predictor.state_dict().items()}

    # ← only change vs centroid variant: medoid instead of centroid
    per_class_stats = _compute_local_medoid_proxy(model, X_train, y_train, device)
    return best_gr_state, len(X_train), per_class_stats


def local_train_fedgen_zhu(model, generator, all_predictor_states,
                            X_train, y_train, X_val, y_val,
                            device, batch_size=128, lr=1e-3,
                            local_epochs=LOCAL_EPOCHS, noise_dim=NOISE_DIM):
    """
    Paper-faithful KD loss (Zhu et al. 2021 Eq. 5):

        L_kd = KL( p_ensemble(y|Ẑ) ‖ p_local(y|Ẑ) )
             = -Σ p_ens · log p_local   + const

    p_ensemble = mean of softmax(W_k Ẑ + b_k) across all client heads.
    The feature extractor is updated only via real-data loss.
    """
    loader      = DataLoader(TensorDataset(X_train, y_train),
                             batch_size=batch_size, shuffle=True, drop_last=True)
    criterion_r = make_criterion(y_train.numpy(), device)
    opt_full    = torch.optim.Adam(model.parameters(),           lr=lr)
    opt_gr      = torch.optim.Adam(model.predictor.parameters(), lr=lr)

    all_W = [sd['weight'].to(device) for sd in all_predictor_states]
    all_b = [sd['bias'].to(device)   for sd in all_predictor_states]

    best_val_auc  = 0.0
    best_gr_state = {k: v.clone() for k, v in model.predictor.state_dict().items()}

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
            ]).mean(dim=0)  # (n_syn, 2) — soft teacher

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
                best_val_auc  = auc
                best_gr_state = {k: v.clone()
                                 for k, v in model.predictor.state_dict().items()}

    per_class_stats = _compute_local_prototypes(model, X_train, y_train, device)
    return best_gr_state, len(X_train), per_class_stats


def local_train_fedgen_zhu_medoid(model, generator, all_predictor_states,
                                   X_train, y_train, X_val, y_val,
                                   device, batch_size=128, lr=1e-3,
                                   local_epochs=LOCAL_EPOCHS, noise_dim=NOISE_DIM):
    """
    KL-ensemble KD + medoid anchor. Only differs from local_train_fedgen_zhu
    in the final prototype computation.
    """
    loader      = DataLoader(TensorDataset(X_train, y_train),
                             batch_size=batch_size, shuffle=True, drop_last=True)
    criterion_r = make_criterion(y_train.numpy(), device)
    opt_full    = torch.optim.Adam(model.parameters(),           lr=lr)
    opt_gr      = torch.optim.Adam(model.predictor.parameters(), lr=lr)

    all_W = [sd['weight'].to(device) for sd in all_predictor_states]
    all_b = [sd['bias'].to(device)   for sd in all_predictor_states]

    best_val_auc  = 0.0
    best_gr_state = {k: v.clone() for k, v in model.predictor.state_dict().items()}

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
                best_val_auc  = auc
                best_gr_state = {k: v.clone()
                                 for k, v in model.predictor.state_dict().items()}

    # ← medoid anchor
    per_class_stats = _compute_local_medoid_proxy(model, X_train, y_train, device)
    return best_gr_state, len(X_train), per_class_stats


def local_train_fedgen_gmm(model, gmm_sampler,
                            X_train, y_train, X_val, y_val,
                            device, batch_size=128, lr=1e-3,
                            local_epochs=LOCAL_EPOCHS):
    """
    Same structure as local_train_fedgen_partial but draws synthetic samples
    from the GMMSampler instead of the neural Generator.
    """
    loader       = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    criterion_r  = make_criterion(y_train.numpy(), device)
    criterion_kd = nn.CrossEntropyLoss()
    opt_full     = torch.optim.Adam(model.parameters(),           lr=lr)
    opt_gr       = torch.optim.Adam(model.predictor.parameters(), lr=lr)

    best_val_auc  = 0.0
    best_gr_state = {k: v.clone() for k, v in model.predictor.state_dict().items()}

    for _ in range(local_epochs):
        model.train()
        n_per_class = max(1, min(batch_size // 2, len(X_train) // 2))
        with torch.no_grad():
            Z_hat, y_hat = gmm_sampler.sample(n_per_class)  # balanced batch

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
                best_val_auc  = auc
                best_gr_state = {k: v.clone()
                                 for k, v in model.predictor.state_dict().items()}

    per_class_stats = _compute_local_prototypes(model, X_train, y_train, device)
    return best_gr_state, len(X_train), per_class_stats


# ═════════════════════════════════════════════════════════════════════════════
# Run loops
# ═════════════════════════════════════════════════════════════════════════════

def _run_fedgen_partial_generic(clients, input_dim, seed,
                                 hidden_dim, dropout, lr, batch_size,
                                 local_train_fn, use_medoid, use_ensemble_teacher,
                                 n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                                 local_epochs=LOCAL_EPOCHS, patience=PATIENCE,lambda_proto=LAMBDA_PROTO,
                                 noise_dim=NOISE_DIM, variant_name='fedgen_partial'):
    """
    Unified run loop for the 4 neural-generator partial variants.

    Parameters
    ----------
    local_train_fn : callable
        One of local_train_fedgen_partial / _partial_medoid / _zhu / _zhu_medoid.
    use_medoid : bool
        Only affects logging; the anchor choice is baked into local_train_fn.
    use_ensemble_teacher : bool
        If True, maintains `all_predictor_states` from the previous round and
        passes it to local_train_fn (Zhu variants).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    latent_dim   = hidden_dim // 4
    global_model = MLP(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    generator_   = Generator(latent_dim, noise_dim).to(device)
    local_models = {
        i: MLP(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
        for i in range(n_clients)
    }
    for i in range(n_clients):
        local_models[i].load_state_dict(global_model.state_dict())

    n_pred      = sum(p.numel() for p in global_model.predictor.parameters())
    n_gen       = sum(p.numel() for p in generator_.parameters())
    bytes_round = (2 * n_pred + n_gen) * 4 * n_clients

    best_val_auc, best_gr_state, best_gen_state, no_improve = 0.0, None, None, 0
    history, cumul_mb, total_bytes = [], [], 0

    # Ensemble teacher (Zhu variants only) — initialised from global model
    all_predictor_states = [
        {k: v.clone() for k, v in global_model.predictor.state_dict().items()}
        for _ in range(n_clients)
    ] if use_ensemble_teacher else None

    for fl_round in range(fl_rounds):
        gr_states, counts, client_stats = [], [], []

        for i in range(n_clients):
            local_models[i].predictor.load_state_dict(
                global_model.predictor.state_dict()
            )
            if use_ensemble_teacher:
                sd, n, stats = local_train_fn(
                    local_models[i], generator_, all_predictor_states,
                    clients[i]['X_train'], clients[i]['y_train'],
                    clients[i]['X_val'],   clients[i]['y_val'],
                    device, batch_size=batch_size, lr=lr,
                    local_epochs=local_epochs, noise_dim=noise_dim,
                )
            else:
                sd, n, stats = local_train_fn(
                    local_models[i], generator_,
                    clients[i]['X_train'], clients[i]['y_train'],
                    clients[i]['X_val'],   clients[i]['y_val'],
                    device, batch_size=batch_size, lr=lr,
                    local_epochs=local_epochs, noise_dim=noise_dim,
                )
            gr_states.append(sd)
            counts.append(n)
            client_stats.append(stats)

        # Server aggregation
        global_model.predictor.load_state_dict(fed_avg(gr_states, counts))
        if use_ensemble_teacher:
            all_predictor_states = gr_states  # teacher for next round

        global_prototypes = _aggregate_prototypes(client_stats)
        update_generator(generator_, gr_states, global_prototypes, device,
                         noise_dim=noise_dim, lambda_proto=lambda_proto)
        total_bytes += bytes_round

        val_auc, test_auc = _eval_partial(global_model, local_models, clients, n_clients)
        history.append({'val': val_auc, 'test': test_auc})
        cumul_mb.append(total_bytes / (1024 ** 2))
        print(f'  [{variant_name}] R{fl_round+1:02d}: val={val_auc:.4f} '
              f'test={test_auc:.4f} cumul={cumul_mb[-1]:.2f}MB')

        if val_auc > best_val_auc:
            best_val_auc   = val_auc
            best_gr_state  = {k: v.clone()
                              for k, v in global_model.predictor.state_dict().items()}
            best_gen_state = {k: v.clone() for k, v in generator_.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'  Early stopping at round {fl_round+1}.')
                break

    global_model.predictor.load_state_dict(best_gr_state)
    global_auc, per_client = evaluate_global(global_model, clients,
                                             use_local_encoders=True,
                                             local_models=local_models)
    return global_auc, per_client, history, cumul_mb


def run_fedgen_partial(clients, input_dim, seed,
                       hidden_dim, dropout, lr, batch_size,
                       n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                       local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                       noise_dim=NOISE_DIM):
    return _run_fedgen_partial_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_partial,
        use_medoid=False, use_ensemble_teacher=False,
        n_clients=n_clients, fl_rounds=fl_rounds,
        local_epochs=local_epochs, patience=patience,
        noise_dim=noise_dim, variant_name='fedgen_partial',
    )


def run_fedgen_partial_medoid(clients, input_dim, seed,
                               hidden_dim, dropout, lr, batch_size,
                               n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                               local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                               noise_dim=NOISE_DIM):
    return _run_fedgen_partial_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_partial_medoid,
        use_medoid=True, use_ensemble_teacher=False,
        n_clients=n_clients, fl_rounds=fl_rounds,
        local_epochs=local_epochs, patience=patience,
        noise_dim=noise_dim, variant_name='fedgen_partial_medoid',
    )


def run_fedgen_zhu(clients, input_dim, seed,
                    hidden_dim, dropout, lr, batch_size,
                    n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                    local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                    noise_dim=NOISE_DIM):
    return _run_fedgen_partial_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_zhu,
        use_medoid=False, use_ensemble_teacher=True,
        n_clients=n_clients, fl_rounds=fl_rounds,
        local_epochs=local_epochs, patience=patience,
        noise_dim=noise_dim, variant_name='fedgen_zhu',
    )


def run_fedgen_zhu_medoid(clients, input_dim, seed,
                           hidden_dim, dropout, lr, batch_size,
                           n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                           local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                           noise_dim=NOISE_DIM):
    return _run_fedgen_partial_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_zhu_medoid,
        use_medoid=True, use_ensemble_teacher=True,
        n_clients=n_clients, fl_rounds=fl_rounds,
        local_epochs=local_epochs, patience=patience,
        noise_dim=noise_dim, variant_name='fedgen_zhu_medoid',
    )

def run_fedgen_zhu_no_proto(clients, input_dim, seed,
                             hidden_dim, dropout, lr, batch_size,
                             n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                             local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                             noise_dim=NOISE_DIM):
    """
    Paper-faithful Eq. 4 + Eq. 5 without prototype constraint.
    KL-ensemble KD on the local side, generator trained with CE on
    ensemble predictions only (lambda_proto=0).
    """
    return _run_fedgen_partial_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_zhu,
        use_medoid=False, use_ensemble_teacher=True,
        n_clients=n_clients, fl_rounds=fl_rounds,
        local_epochs=local_epochs, patience=patience,
        noise_dim=noise_dim, lambda_proto=0.0,
        variant_name='fedgen_zhu_no_proto',
    )


def run_fedgen_gmm(clients, input_dim, seed,
                    hidden_dim, dropout, lr, batch_size,
                    n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                    local_epochs=LOCAL_EPOCHS, patience=PATIENCE):
    """
    GMM ablation: no neural generator, no prototype constraint.
    The server fits per-class Gaussians from client latent stats each round.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    latent_dim   = hidden_dim // 4
    global_model = MLP(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    gmm          = GMMSampler(latent_dim, device)
    local_models = {
        i: MLP(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
        for i in range(n_clients)
    }
    for i in range(n_clients):
        local_models[i].load_state_dict(global_model.state_dict())

    n_pred      = sum(p.numel() for p in global_model.predictor.parameters())
    bytes_round = 2 * n_pred * 4 * n_clients + gmm.param_bytes() * n_clients

    best_val_auc, best_gr_state, no_improve = 0.0, None, 0
    history, cumul_mb, total_bytes = [], [], 0

    for fl_round in range(fl_rounds):
        gr_states, counts, client_dists = [], [], []

        for i in range(n_clients):
            local_models[i].predictor.load_state_dict(
                global_model.predictor.state_dict()
            )
            sd, n, _stats = local_train_fedgen_gmm(
                local_models[i], gmm,
                clients[i]['X_train'], clients[i]['y_train'],
                clients[i]['X_val'],   clients[i]['y_val'],
                device, batch_size=batch_size, lr=lr,
                local_epochs=local_epochs,
            )
            dist = compute_client_distribution(
                local_models[i],
                clients[i]['X_train'], clients[i]['y_train'].numpy(), device,
            )
            gr_states.append(sd)
            counts.append(n)
            client_dists.append(dist)

        global_model.predictor.load_state_dict(fed_avg(gr_states, counts))
        gmm.update(client_dists, counts)
        total_bytes += bytes_round

        val_auc, test_auc = _eval_partial(global_model, local_models, clients, n_clients)
        history.append({'val': val_auc, 'test': test_auc})
        cumul_mb.append(total_bytes / (1024 ** 2))
        print(f'  [fedgen_gmm] R{fl_round+1:02d}: val={val_auc:.4f} '
              f'test={test_auc:.4f} cumul={cumul_mb[-1]:.2f}MB')

        if val_auc > best_val_auc:
            best_val_auc  = val_auc
            best_gr_state = {k: v.clone()
                             for k, v in global_model.predictor.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'  Early stopping at round {fl_round+1}.')
                break

    global_model.predictor.load_state_dict(best_gr_state)
    global_auc, per_client = evaluate_global(global_model, clients,
                                             use_local_encoders=True,
                                             local_models=local_models)
    return global_auc, per_client, history, cumul_mb
