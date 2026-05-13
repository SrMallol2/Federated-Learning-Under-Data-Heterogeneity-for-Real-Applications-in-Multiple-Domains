"""
UC1FedGenPartial.py
─────────────────────────────────────────────────────────────────────────────
Partial-sharing FedGen variants for UC1 (Diabetes 130-US Hospitals).

Each client keeps its own feature extractor. Only the predictor head and
the lightweight generator (or GMM statistics) are shared each round.
Evaluation uses per-client local encoder + global predictor.

Variants                                    KD loss     Label     Proto     p̂(y)
──────────────────────────────────────────────────────────────────────────────────
fedgen_partial                              hard CE     uniform   centroid    ✗
fedgen_partial_medoid                       hard CE     uniform   medoid      ✗
fedgen_zhu                                  KL ensemb.  uniform   centroid    ✗
fedgen_zhu_medoid                           KL ensemb.  uniform   medoid      ✗
fedgen_zhu_no_proto                         KL ensemb.  uniform   (λ=0)       ✗
fedgen_gmm                                 hard CE     balanced  centroid    ✗
fedgen_zhu_pyhat                            KL ensemb.  p̂(y)      centroid    ✓
fedgen_zhu_pyhat_medoid                     KL ensemb.  p̂(y)      medoid      ✓
fedgen_paper_partial                        hard CE     p̂(y)      centroid    ✓
fedgen_paper_partial_no_proto               hard CE     p̂(y)      (λ=0)       ✓

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
    _label_counter, _aggregate_phat,
    update_generator,
    N_CLIENTS, FL_ROUNDS, LOCAL_EPOCHS, PATIENCE, NOISE_DIM, LAMBDA_PROTO,
    NUM_CLASSES, device,
)


# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════════════

def _eval_partial(global_model, local_models, clients, n_clients):
    """Per-client local encoder + global predictor → concatenated AUC."""
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
# LOCAL TRAINING FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

# ── hard CE KD (centroid / medoid) — KD applied inside batch loop ─────────

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
    """Identical to local_train_fedgen_partial but uses medoid anchor."""
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

    per_class_stats = _compute_local_medoid_proxy(model, X_train, y_train, device)
    return best_gr_state, len(X_train), per_class_stats


# ── KL ensemble KD (centroid / medoid) — KD applied once per epoch ────────

def local_train_fedgen_zhu(model, generator, all_predictor_states,
                            X_train, y_train, X_val, y_val,
                            device, batch_size=128, lr=1e-3,
                            local_epochs=LOCAL_EPOCHS, noise_dim=NOISE_DIM):
    """
    KL-ensemble KD (Zhu et al. 2021 Eq. 5):
        L_kd = KL( p_ensemble(y|Ẑ) ‖ p_local(y|Ẑ) )
    Labels sampled uniformly.
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

    per_class_stats = _compute_local_prototypes(model, X_train, y_train, device)
    return best_gr_state, len(X_train), per_class_stats


def local_train_fedgen_zhu_medoid(model, generator, all_predictor_states,
                                   X_train, y_train, X_val, y_val,
                                   device, batch_size=128, lr=1e-3,
                                   local_epochs=LOCAL_EPOCHS, noise_dim=NOISE_DIM):
    """KL-ensemble KD + medoid anchor. Differs only in the proto call."""
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

    per_class_stats = _compute_local_medoid_proxy(model, X_train, y_train, device)
    return best_gr_state, len(X_train), per_class_stats


# ── KL ensemble KD + p̂(y) (centroid / medoid) ────────────────────────────

def local_train_fedgen_zhu_pyhat(model, generator, all_predictor_states, p_hat_y,
                                  X_train, y_train, X_val, y_val,
                                  device, batch_size=128, lr=1e-3,
                                  local_epochs=LOCAL_EPOCHS, noise_dim=NOISE_DIM):
    """KL-ensemble KD with ŷ ~ p̂(y) + centroid anchor.  Returns c_k."""
    loader      = DataLoader(TensorDataset(X_train, y_train),
                             batch_size=batch_size, shuffle=True, drop_last=True)
    criterion_r = make_criterion(y_train.numpy(), device)
    opt_full    = torch.optim.Adam(model.parameters(),           lr=lr)
    opt_gr      = torch.optim.Adam(model.predictor.parameters(), lr=lr)

    all_W = [sd['weight'].to(device) for sd in all_predictor_states]
    all_b = [sd['bias'].to(device)   for sd in all_predictor_states]
    p_hat_dev = p_hat_y.to(device)

    best_val_auc  = 0.0
    best_gr_state = {k: v.clone() for k, v in model.predictor.state_dict().items()}

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
                best_val_auc  = auc
                best_gr_state = {k: v.clone()
                                 for k, v in model.predictor.state_dict().items()}

    per_class_stats = _compute_local_prototypes(model, X_train, y_train, device)
    c_k = _label_counter(y_train)
    return best_gr_state, len(X_train), per_class_stats, c_k


def local_train_fedgen_zhu_pyhat_medoid(model, generator, all_predictor_states, p_hat_y,
                                         X_train, y_train, X_val, y_val,
                                         device, batch_size=128, lr=1e-3,
                                         local_epochs=LOCAL_EPOCHS, noise_dim=NOISE_DIM):
    """KL-ensemble KD with ŷ ~ p̂(y) + medoid anchor.  Returns c_k."""
    loader      = DataLoader(TensorDataset(X_train, y_train),
                             batch_size=batch_size, shuffle=True, drop_last=True)
    criterion_r = make_criterion(y_train.numpy(), device)
    opt_full    = torch.optim.Adam(model.parameters(),           lr=lr)
    opt_gr      = torch.optim.Adam(model.predictor.parameters(), lr=lr)

    all_W = [sd['weight'].to(device) for sd in all_predictor_states]
    all_b = [sd['bias'].to(device)   for sd in all_predictor_states]
    p_hat_dev = p_hat_y.to(device)

    best_val_auc  = 0.0
    best_gr_state = {k: v.clone() for k, v in model.predictor.state_dict().items()}

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
                best_val_auc  = auc
                best_gr_state = {k: v.clone()
                                 for k, v in model.predictor.state_dict().items()}

    per_class_stats = _compute_local_medoid_proxy(model, X_train, y_train, device)
    c_k = _label_counter(y_train)
    return best_gr_state, len(X_train), per_class_stats, c_k


# ── Paper-faithful hard CE + p̂(y) ────────────────────────────────────────

def local_train_paper_partial(model, generator, p_hat_y,
                               X_train, y_train, X_val, y_val,
                               device, batch_size=128, lr=1e-3,
                               local_epochs=LOCAL_EPOCHS, noise_dim=NOISE_DIM):
    """
    Algorithm 1, lines 6-9 — paper-faithful local training.

    Each local step t:
        1. Sample B real samples {x_i, y_i} ~ T_k
        2. Sample B synthetic pairs {ẑ_i ~ G(·|ŷ_i), ŷ_i ~ p̂(y)}  (fresh each step)
        3. Compute combined gradient ∇_{θ_k} J(θ_k) where
           J(θ_k) = L̂_k(θ_k) + E[CE(h(z; θ^p_k), ŷ)]   (Equation 5)
        4. Single optimizer step on full model

    Partial sharing: only predictor state is returned for FedAvg.

    Differences from earlier "paper" variant:
        - Fresh synthetic batch every step (not reused across batches)
        - Combined gradient (not two separate optimizer.step() calls)
        - Single optimizer on model.parameters() (not separate opt_full / opt_gr)
    """
    loader       = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    criterion_r  = make_criterion(y_train.numpy(), device)
    criterion_kd = nn.CrossEntropyLoss()
    optimizer    = torch.optim.Adam(model.parameters(), lr=lr)
    p_hat_dev    = p_hat_y.to(device)

    best_val_auc  = 0.0
    best_gr_state = {k: v.clone() for k, v in model.predictor.state_dict().items()}

    for _ in range(local_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            # Alg 1 line 7: fresh synthetic batch every step
            n_syn = len(xb)
            with torch.no_grad():
                y_hat = torch.multinomial(p_hat_dev, n_syn, replacement=True)
                eps   = torch.randn(n_syn, noise_dim, device=device)
                Z_hat = generator(y_hat, eps)

            # Alg 1 line 9: single ∇_{θ_k} J(θ_k) — Equation 5
            optimizer.zero_grad()
            loss_real = criterion_r(model(xb), yb)
            loss_kd   = criterion_kd(model.predictor(Z_hat.detach()), y_hat)
            (loss_real + loss_kd).backward()
            optimizer.step()

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
    c_k = _label_counter(y_train)
    return best_gr_state, len(X_train), per_class_stats, c_k


# ── GMM sampler — KD applied inside batch loop ───────────────────────────

def local_train_fedgen_gmm(model, gmm_sampler,
                            X_train, y_train, X_val, y_val,
                            device, batch_size=128, lr=1e-3,
                            local_epochs=LOCAL_EPOCHS):
    """Hard CE KD with GMM synthetic samples (balanced, no neural generator)."""
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
                best_val_auc  = auc
                best_gr_state = {k: v.clone()
                                 for k, v in model.predictor.state_dict().items()}

    per_class_stats = _compute_local_prototypes(model, X_train, y_train, device)
    return best_gr_state, len(X_train), per_class_stats


# ═════════════════════════════════════════════════════════════════════════════
# RUN LOOPS
# ═════════════════════════════════════════════════════════════════════════════

def _run_fedgen_partial_generic(clients, input_dim, seed,
                                 hidden_dim, dropout, lr, batch_size,
                                 local_train_fn, use_medoid, use_ensemble_teacher,
                                 n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                                 local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                                 lambda_proto=LAMBDA_PROTO,
                                 noise_dim=NOISE_DIM, variant_name='fedgen_partial'):
    """Unified run loop for neural-generator partial variants (uniform sampling)."""
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

        global_model.predictor.load_state_dict(fed_avg(gr_states, counts))
        if use_ensemble_teacher:
            all_predictor_states = gr_states

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


def _run_fedgen_partial_pyhat_generic(clients, input_dim, seed,
                                       hidden_dim, dropout, lr, batch_size,
                                       local_train_fn, use_ensemble_teacher=True,
                                       n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                                       local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                                       lambda_proto=LAMBDA_PROTO,
                                       noise_dim=NOISE_DIM,
                                       logit_avg=False,
                                       variant_name='fedgen_zhu_pyhat'):
    """Unified run loop for partial variants with paper-faithful p̂(y) maintenance."""
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
    bytes_round = (2 * n_pred + n_gen) * 4 * n_clients + NUM_CLASSES * 4 * n_clients

    best_val_auc, best_gr_state, best_gen_state, no_improve = 0.0, None, None, 0
    history, cumul_mb, total_bytes = [], [], 0

    p_hat_y = torch.full((NUM_CLASSES,), 1.0 / NUM_CLASSES)

    all_predictor_states = [
        {k: v.clone() for k, v in global_model.predictor.state_dict().items()}
        for _ in range(n_clients)
    ] if use_ensemble_teacher else None

    for fl_round in range(fl_rounds):
        gr_states, counts, client_stats, client_c = [], [], [], []

        for i in range(n_clients):
            local_models[i].predictor.load_state_dict(
                global_model.predictor.state_dict()
            )
            if use_ensemble_teacher:
                sd, n, stats, c_k = local_train_fn(
                    local_models[i], generator_, all_predictor_states, p_hat_y,
                    clients[i]['X_train'], clients[i]['y_train'],
                    clients[i]['X_val'],   clients[i]['y_val'],
                    device, batch_size=batch_size, lr=lr,
                    local_epochs=local_epochs, noise_dim=noise_dim,
                )
            else:
                sd, n, stats, c_k = local_train_fn(
                    local_models[i], generator_, p_hat_y,
                    clients[i]['X_train'], clients[i]['y_train'],
                    clients[i]['X_val'],   clients[i]['y_val'],
                    device, batch_size=batch_size, lr=lr,
                    local_epochs=local_epochs, noise_dim=noise_dim,
                )
            gr_states.append(sd)
            counts.append(n)
            client_stats.append(stats)
            client_c.append(c_k)

        global_model.predictor.load_state_dict(fed_avg(gr_states, counts))
        if use_ensemble_teacher:
            all_predictor_states = gr_states
        p_hat_y = _aggregate_phat(client_c)

        global_prototypes = _aggregate_prototypes(client_stats)
        update_generator(generator_, gr_states, global_prototypes, device,
                         p_hat_y=p_hat_y,
                         noise_dim=noise_dim, lambda_proto=lambda_proto,
                         logit_avg=logit_avg)
        total_bytes += bytes_round

        val_auc, test_auc = _eval_partial(global_model, local_models, clients, n_clients)
        history.append({
            'val': val_auc, 'test': test_auc,
            'p_hat_y': p_hat_y.tolist(),
        })
        cumul_mb.append(total_bytes / (1024 ** 2))
        print(f'  [{variant_name}] R{fl_round+1:02d}: val={val_auc:.4f} '
              f'test={test_auc:.4f} p̂={[f"{x:.3f}" for x in p_hat_y.tolist()]} '
              f'cumul={cumul_mb[-1]:.2f}MB')

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


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API — one function per variant
# ═════════════════════════════════════════════════════════════════════════════

def run_fedgen_partial(clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
                       n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                       local_epochs=LOCAL_EPOCHS, patience=PATIENCE, noise_dim=NOISE_DIM):
    return _run_fedgen_partial_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_partial,
        use_medoid=False, use_ensemble_teacher=False,
        n_clients=n_clients, fl_rounds=fl_rounds, local_epochs=local_epochs,
        patience=patience, noise_dim=noise_dim, variant_name='fedgen_partial')

def run_fedgen_partial_medoid(clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
                               n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                               local_epochs=LOCAL_EPOCHS, patience=PATIENCE, noise_dim=NOISE_DIM):
    return _run_fedgen_partial_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_partial_medoid,
        use_medoid=True, use_ensemble_teacher=False,
        n_clients=n_clients, fl_rounds=fl_rounds, local_epochs=local_epochs,
        patience=patience, noise_dim=noise_dim, variant_name='fedgen_partial_medoid')

def run_fedgen_zhu(clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
                    n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                    local_epochs=LOCAL_EPOCHS, patience=PATIENCE, noise_dim=NOISE_DIM):
    return _run_fedgen_partial_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_zhu,
        use_medoid=False, use_ensemble_teacher=True,
        n_clients=n_clients, fl_rounds=fl_rounds, local_epochs=local_epochs,
        patience=patience, noise_dim=noise_dim, variant_name='fedgen_zhu')

def run_fedgen_zhu_medoid(clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
                           n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                           local_epochs=LOCAL_EPOCHS, patience=PATIENCE, noise_dim=NOISE_DIM):
    return _run_fedgen_partial_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_zhu_medoid,
        use_medoid=True, use_ensemble_teacher=True,
        n_clients=n_clients, fl_rounds=fl_rounds, local_epochs=local_epochs,
        patience=patience, noise_dim=noise_dim, variant_name='fedgen_zhu_medoid')

def run_fedgen_zhu_no_proto(clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
                             n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                             local_epochs=LOCAL_EPOCHS, patience=PATIENCE, noise_dim=NOISE_DIM):
    return _run_fedgen_partial_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_zhu,
        use_medoid=False, use_ensemble_teacher=True,
        n_clients=n_clients, fl_rounds=fl_rounds, local_epochs=local_epochs,
        patience=patience, noise_dim=noise_dim, lambda_proto=0.0,
        variant_name='fedgen_zhu_no_proto')

def run_fedgen_zhu_pyhat(clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
                          n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                          local_epochs=LOCAL_EPOCHS, patience=PATIENCE, noise_dim=NOISE_DIM):
    return _run_fedgen_partial_pyhat_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_zhu_pyhat, use_ensemble_teacher=True,
        n_clients=n_clients, fl_rounds=fl_rounds, local_epochs=local_epochs,
        patience=patience, noise_dim=noise_dim, variant_name='fedgen_zhu_pyhat')

def run_fedgen_zhu_pyhat_medoid(clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
                                 n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                                 local_epochs=LOCAL_EPOCHS, patience=PATIENCE, noise_dim=NOISE_DIM):
    return _run_fedgen_partial_pyhat_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_zhu_pyhat_medoid, use_ensemble_teacher=True,
        n_clients=n_clients, fl_rounds=fl_rounds, local_epochs=local_epochs,
        patience=patience, noise_dim=noise_dim, variant_name='fedgen_zhu_pyhat_medoid')

def run_fedgen_paper_partial(clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
                              n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                              local_epochs=LOCAL_EPOCHS, patience=PATIENCE, noise_dim=NOISE_DIM):
    """Algorithm 1 paper-faithful + centroid anchor. Partial sharing."""
    return _run_fedgen_partial_pyhat_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_paper_partial, use_ensemble_teacher=False,
        n_clients=n_clients, fl_rounds=fl_rounds, local_epochs=local_epochs,
        patience=patience, noise_dim=noise_dim, logit_avg=True,
        variant_name='fedgen_paper_partial')

def run_fedgen_paper_partial_no_proto(clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
                                       n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                                       local_epochs=LOCAL_EPOCHS, patience=PATIENCE, noise_dim=NOISE_DIM):
    """Algorithm 1 paper-faithful, no anchor (λ=0). Partial sharing. Purest reproduction."""
    return _run_fedgen_partial_pyhat_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_paper_partial, use_ensemble_teacher=False,
        n_clients=n_clients, fl_rounds=fl_rounds, local_epochs=local_epochs,
        patience=patience, noise_dim=noise_dim, lambda_proto=0.0, logit_avg=True,
        variant_name='fedgen_paper_partial_no_proto')


# ═════════════════════════════════════════════════════════════════════════════
# GMM variant (standalone run loop — no neural generator)
# ═════════════════════════════════════════════════════════════════════════════

def run_fedgen_gmm(clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
                    n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                    local_epochs=LOCAL_EPOCHS, patience=PATIENCE):
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


# ═════════════════════════════════════════════════════════════════════════════
# ZHU CODE-FAITHFUL VARIANT
# ═════════════════════════════════════════════════════════════════════════════

def run_fedgen_zhu_code_partial(clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
                                 n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                                 local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                                 noise_dim=NOISE_DIM):
    """
    Zhu et al. GitHub code-faithful variant (partial sharing).

    Client side: Eq 5 — CE(h(z; θ^p_k), ŷ), ŷ ~ p̂(y), combined gradient.
                 Round 0 warm-up: no KD (regularization = glob_iter > 0).

    Server side: 3-term adversarial generator loss from GitHub code:
        L = α·teacher − β·student + η·diversity
        (NOT Eq 4 from the paper)

    No prototype constraint (paper has none, code has none).
    """
    from UC1FLUtils import (
        MLP, Generator, fed_avg, evaluate_global,
        _compute_local_prototypes, _aggregate_prototypes,
        _label_counter, _aggregate_phat,
        update_generator_zhu_code,
        N_CLIENTS as _NC, NOISE_DIM as _ND, NUM_CLASSES, device,
    )

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
    bytes_round = (2 * n_pred + n_gen) * 4 * n_clients + NUM_CLASSES * 4 * n_clients

    best_val_auc, best_gr_state, best_gen_state, no_improve = 0.0, None, None, 0
    history, cumul_mb, total_bytes = [], [], 0

    p_hat_y = torch.full((NUM_CLASSES,), 1.0 / NUM_CLASSES)

    for fl_round in range(fl_rounds):
        gr_states, counts, client_stats, client_c = [], [], [], []
        use_kd = fl_round > 0  # warm-up: no KD in round 0

        for i in range(n_clients):
            local_models[i].predictor.load_state_dict(
                global_model.predictor.state_dict()
            )
            if use_kd:
                sd, n, stats, c_k = local_train_paper_partial(
                    local_models[i], generator_, p_hat_y,
                    clients[i]['X_train'], clients[i]['y_train'],
                    clients[i]['X_val'],   clients[i]['y_val'],
                    device, batch_size=batch_size, lr=lr,
                    local_epochs=local_epochs, noise_dim=noise_dim,
                )
            else:
                # Round 0: plain local training, no generator
                loader = DataLoader(
                    TensorDataset(clients[i]['X_train'], clients[i]['y_train']),
                    batch_size=batch_size, shuffle=True, drop_last=True)
                criterion_r = make_criterion(clients[i]['y_train'].numpy(), device)
                optimizer   = torch.optim.Adam(local_models[i].parameters(), lr=lr)
                best_auc_i  = 0.0
                best_sd_i   = {k: v.clone()
                               for k, v in local_models[i].predictor.state_dict().items()}
                for _ in range(local_epochs):
                    local_models[i].train()
                    for xb, yb in loader:
                        xb, yb = xb.to(device), yb.to(device)
                        optimizer.zero_grad()
                        criterion_r(local_models[i](xb), yb).backward()
                        optimizer.step()
                    local_models[i].eval()
                    with torch.no_grad():
                        vp = torch.softmax(
                            local_models[i](clients[i]['X_val'].to(device)), dim=1
                        )[:, 1].cpu().numpy()
                    if len(np.unique(clients[i]['y_val'].numpy())) > 1:
                        auc = roc_auc_score(clients[i]['y_val'].numpy(), vp)
                        if auc > best_auc_i:
                            best_auc_i = auc
                            best_sd_i  = {k: v.clone()
                                          for k, v in local_models[i].predictor.state_dict().items()}
                sd    = best_sd_i
                n     = len(clients[i]['X_train'])
                stats = _compute_local_prototypes(
                    local_models[i], clients[i]['X_train'], clients[i]['y_train'], device)
                c_k   = _label_counter(clients[i]['y_train'])

            gr_states.append(sd)
            counts.append(n)
            client_stats.append(stats)
            client_c.append(c_k)

        global_model.predictor.load_state_dict(fed_avg(gr_states, counts))
        p_hat_y = _aggregate_phat(client_c)

        # 3-term adversarial generator update (Zhu's GitHub code)
        update_generator_zhu_code(
            generator_, gr_states,
            global_predictor_state={k: v.clone()
                                    for k, v in global_model.predictor.state_dict().items()},
            label_counts=client_c,
            device=device,
            p_hat_y=p_hat_y,
            noise_dim=noise_dim,
        )
        total_bytes += bytes_round

        val_auc, test_auc = _eval_partial(global_model, local_models, clients, n_clients)
        history.append({
            'val': val_auc, 'test': test_auc,
            'p_hat_y': p_hat_y.tolist(),
        })
        cumul_mb.append(total_bytes / (1024 ** 2))
        kd_flag = '(KD)' if use_kd else '(warm-up)'
        print(f'  [fedgen_zhu_code_partial] R{fl_round+1:02d} {kd_flag}: '
              f'val={val_auc:.4f} test={test_auc:.4f} cumul={cumul_mb[-1]:.2f}MB')

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
