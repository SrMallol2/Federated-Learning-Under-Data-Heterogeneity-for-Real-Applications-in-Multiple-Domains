"""
UC1FedGenFull_paper.py
─────────────────────────────────────────────────────────────────────────────
True paper-faithful FedGen (Zhu et al., ICML 2021) for full-sharing.

What "paper-faithful" means here, line by line against Algorithm 1:

    Client-side KD loss (Equation 5):
        L_KD = CE(h(z; θ_k^p), ŷ)
    This is a HARD label loss — the client trains its predictor to classify
    the synthetic feature z for the label ŷ that conditioned the generator.
    NO ensemble of per-client heads appears on the client side. The ensemble
    knowledge is implicit in the generator, which was trained on the server
    via Equation 4.

    Synthetic label sampling (lines 1, 7):
        ŷ ~ p̂(y), NOT uniform.
    p̂(y) is the empirically estimated global label prior, initialised
    uniform and updated each round from client label counters c_k.

    Generator training (Equation 4, line 14):
        y ~ p̂(y), NOT the hardcoded 50/50 split.

    Label counters (lines 8, 11, 13):
        Each client computes c_k, uploads it, server aggregates into p̂(y).

Variants in this file:
    fedgen_paper_full              — Eq 5 hard CE + centroid anchor + p̂(y)
    fedgen_paper_full_no_proto     — Eq 5 hard CE + no anchor     + p̂(y)
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
    _compute_local_prototypes,
    _aggregate_prototypes,
    N_CLIENTS, FL_ROUNDS, LOCAL_EPOCHS, PATIENCE, NOISE_DIM, LAMBDA_PROTO,
    GEN_LR, GEN_STEPS,
    device,
)

NUM_CLASSES = 2


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _label_counter(y_train, num_classes=NUM_CLASSES):
    y_np = y_train.numpy() if isinstance(y_train, torch.Tensor) else np.asarray(y_train)
    counts = np.bincount(y_np.astype(np.int64), minlength=num_classes).astype(np.float64)
    return torch.tensor(counts, dtype=torch.float32)


def _aggregate_phat(c_list, eps=1e-8):
    total = torch.stack(c_list).sum(dim=0) + eps
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


def update_generator_paper(generator, full_model_states, global_prototypes, device,
                            p_hat_y, n_gen_samples=128,
                            noise_dim=NOISE_DIM, gen_steps=GEN_STEPS,
                            lambda_proto=LAMBDA_PROTO, gen_lr=GEN_LR):
    """
    Server-side generator update — Equation 4, paper-faithful.

    y ~ p̂(y) instead of hardcoded 50/50.
    Loss = CE(σ(1/K Σ_k g(z; θ_k^p)), y) + λ · ‖z − z̄_y‖²

    Note: the paper averages logits then softmaxes. We average softmax
    probabilities (same as existing codebase). See TFG footnote on this.
    """
    all_w = [sd['predictor.weight'].to(device) for sd in full_model_states]
    all_b = [sd['predictor.bias'].to(device)   for sd in full_model_states]

    proto_t = {
        cls: torch.tensor(v['mean'], dtype=torch.float32, device=device)
        for cls, v in global_prototypes.items()
    }

    p_hat_dev = p_hat_y.to(device)
    opt = torch.optim.Adam(generator.parameters(), lr=gen_lr)
    generator.train()

    for _ in range(gen_steps):
        opt.zero_grad()
        y_gen = torch.multinomial(p_hat_dev, n_gen_samples, replacement=True)
        eps   = torch.randn(n_gen_samples, noise_dim, device=device)
        Z_gen = generator(y_gen, eps)

        probs   = torch.stack([
            torch.softmax(Z_gen @ w.T + b, dim=1)
            for w, b in zip(all_w, all_b)
        ]).mean(0)
        loss_ce = -(torch.log(probs[range(n_gen_samples), y_gen] + 1e-8)).mean()

        if lambda_proto > 0:
            proto_tgt = torch.stack([proto_t[int(yy.item())] for yy in y_gen])
            loss_p    = ((Z_gen - proto_tgt) ** 2).mean()
        else:
            loss_p = 0.0

        (loss_ce + lambda_proto * loss_p).backward()
        opt.step()

    generator.eval()


# ═════════════════════════════════════════════════════════════════════════════
# Local training — Equation 5: hard CE against ŷ, no ensemble on client
# ═════════════════════════════════════════════════════════════════════════════

def local_train_paper_full(model, generator, p_hat_y,
                            X_train, y_train, X_val, y_val,
                            device, batch_size=128, lr=1e-3,
                            local_epochs=LOCAL_EPOCHS, noise_dim=NOISE_DIM):
    """
    Equation 5 of Zhu et al.:
        J(θ_k) = L̂_k(θ_k) + E_{ŷ~p̂(y), z~G(z|ŷ)} [ l(h(z; θ_k^p), ŷ) ]

    The KD loss is CE(predictor(z), ŷ) — hard label, no ensemble.
    ŷ is sampled from p̂(y), not uniform.
    No per-client heads are needed on the client side.
    """
    loader       = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    criterion_r  = make_criterion(y_train.numpy(), device)
    criterion_kd = nn.CrossEntropyLoss()
    opt_full     = torch.optim.Adam(model.parameters(),           lr=lr)
    opt_gr       = torch.optim.Adam(model.predictor.parameters(), lr=lr)

    p_hat_dev = p_hat_y.to(device)

    best_val_auc = 0.0
    best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    for _ in range(local_epochs):
        model.train()
        n_syn = min(batch_size, len(X_train))
        with torch.no_grad():
            # Line 7: ŷ ~ p̂(y)
            y_hat = torch.multinomial(p_hat_dev, n_syn, replacement=True)
            eps   = torch.randn(n_syn, noise_dim, device=device)
            Z_hat = generator(y_hat, eps)

        # Real data loss — updates full model
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_full.zero_grad()
            criterion_r(model(xb), yb).backward()
            opt_full.step()

        # KD loss — Eq 5 second term: CE(h(z; θ^p), ŷ), updates predictor only
        opt_gr.zero_grad()
        criterion_kd(model.predictor(Z_hat.detach()), y_hat).backward()
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


# ═════════════════════════════════════════════════════════════════════════════
# Run loop
# ═════════════════════════════════════════════════════════════════════════════

def _run_paper_full_generic(clients, input_dim, seed,
                             hidden_dim, dropout, lr, batch_size,
                             n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                             local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                             lambda_proto=LAMBDA_PROTO,
                             noise_dim=NOISE_DIM,
                             variant_name='fedgen_paper_full'):
    """
    Full-sharing run loop with paper-faithful Eq 4 + Eq 5 + p̂(y).

    No ensemble teacher is maintained — clients receive only θ, G, p̂(y).
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

    # Line 1: p̂(y) uniformly initialised
    p_hat_y = torch.full((NUM_CLASSES,), 1.0 / NUM_CLASSES)

    for fl_round in range(fl_rounds):
        state_dicts, counts, client_stats, client_c = [], [], [], []

        for i in range(n_clients):
            local = MLP(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
            local.load_state_dict(global_model.state_dict())

            # Client receives: θ, G, p̂(y). NO per-client heads.
            sd, n, stats, c_k = local_train_paper_full(
                local, generator_, p_hat_y,
                clients[i]['X_train'], clients[i]['y_train'],
                clients[i]['X_val'],   clients[i]['y_val'],
                device, batch_size=batch_size, lr=lr,
                local_epochs=local_epochs, noise_dim=noise_dim,
            )
            state_dicts.append(sd)
            counts.append(n)
            client_stats.append(stats)
            client_c.append(c_k)

        # Line 13: server aggregation
        global_model.load_state_dict(fed_avg(state_dicts, counts))
        p_hat_y = _aggregate_phat(client_c)

        # Line 14: generator training with p̂(y)
        global_prototypes = _aggregate_prototypes(client_stats)
        update_generator_paper(generator_, state_dicts, global_prototypes, device,
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


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def run_fedgen_paper_full(clients, input_dim, seed,
                           hidden_dim, dropout, lr, batch_size,
                           n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                           local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                           noise_dim=NOISE_DIM):
    """Eq 5 hard CE + centroid anchor + p̂(y). Full model sharing."""
    return _run_paper_full_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        n_clients=n_clients, fl_rounds=fl_rounds,
        local_epochs=local_epochs, patience=patience,
        lambda_proto=LAMBDA_PROTO, noise_dim=noise_dim,
        variant_name='fedgen_paper_full',
    )


def run_fedgen_paper_full_no_proto(clients, input_dim, seed,
                                    hidden_dim, dropout, lr, batch_size,
                                    n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                                    local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                                    noise_dim=NOISE_DIM):
    """Eq 5 hard CE + no anchor (λ=0) + p̂(y). Full model sharing."""
    return _run_paper_full_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        n_clients=n_clients, fl_rounds=fl_rounds,
        local_epochs=local_epochs, patience=patience,
        lambda_proto=0.0, noise_dim=noise_dim,
        variant_name='fedgen_paper_full_no_proto',
    )
