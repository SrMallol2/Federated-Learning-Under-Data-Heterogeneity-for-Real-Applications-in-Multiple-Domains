"""
UC1FedGenPartial_paper.py
─────────────────────────────────────────────────────────────────────────────
True paper-faithful FedGen (Zhu et al., ICML 2021) for partial-sharing.

Same Equation 5 logic as UC1FedGenFull_paper.py but only the predictor
head is shared and averaged each round; feature extractors stay local.

See UC1FedGenFull_paper.py header for the full line-by-line justification
against Algorithm 1.

Variants in this file:
    fedgen_paper_partial           — Eq 5 hard CE + centroid anchor + p̂(y)
    fedgen_paper_partial_no_proto  — Eq 5 hard CE + no anchor       + p̂(y)
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


def _eval_partial(global_model, local_models, clients, n_clients):
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


def update_generator_paper(generator, gr_states, global_prototypes, device,
                            p_hat_y, n_gen_samples=128,
                            noise_dim=NOISE_DIM, gen_steps=GEN_STEPS,
                            lambda_proto=LAMBDA_PROTO, gen_lr=GEN_LR):
    """
    Server-side generator update — Equation 4, paper-faithful.
    Partial variant: predictor states are {weight, bias} dicts.
    """
    all_w = [sd['weight'].to(device) for sd in gr_states]
    all_b = [sd['bias'].to(device)   for sd in gr_states]

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
# Local training — Equation 5: hard CE against ŷ, no ensemble
# ═════════════════════════════════════════════════════════════════════════════

def local_train_paper_partial(model, generator, p_hat_y,
                               X_train, y_train, X_val, y_val,
                               device, batch_size=128, lr=1e-3,
                               local_epochs=LOCAL_EPOCHS, noise_dim=NOISE_DIM):
    """
    Equation 5: J(θ_k) = L̂_k(θ_k) + E_{ŷ~p̂(y), z~G(z|ŷ)} [ CE(h(z; θ^p_k), ŷ) ]

    Partial sharing: only predictor state is returned for FedAvg.
    """
    loader       = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    criterion_r  = make_criterion(y_train.numpy(), device)
    criterion_kd = nn.CrossEntropyLoss()
    opt_full     = torch.optim.Adam(model.parameters(),           lr=lr)
    opt_gr       = torch.optim.Adam(model.predictor.parameters(), lr=lr)

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

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_full.zero_grad()
            criterion_r(model(xb), yb).backward()
            opt_full.step()

        opt_gr.zero_grad()
        criterion_kd(model.predictor(Z_hat.detach()), y_hat).backward()
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


# ═════════════════════════════════════════════════════════════════════════════
# Run loop
# ═════════════════════════════════════════════════════════════════════════════

def _run_paper_partial_generic(clients, input_dim, seed,
                                hidden_dim, dropout, lr, batch_size,
                                n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                                local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                                lambda_proto=LAMBDA_PROTO,
                                noise_dim=NOISE_DIM,
                                variant_name='fedgen_paper_partial'):
    """
    Partial-sharing run loop: Eq 4 + Eq 5 + p̂(y).
    No ensemble teacher. Clients receive only G, θ^p (global predictor), p̂(y).
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
    bytes_round = (2 * n_pred + n_gen) * 4 * n_clients + NUM_CLASSES * 4 * n_clients

    best_val_auc, best_gr_state, best_gen_state, no_improve = 0.0, None, None, 0
    history, cumul_mb, total_bytes = [], [], 0

    p_hat_y = torch.full((NUM_CLASSES,), 1.0 / NUM_CLASSES)

    for fl_round in range(fl_rounds):
        gr_states, counts, client_stats, client_c = [], [], [], []

        for i in range(n_clients):
            local_models[i].predictor.load_state_dict(
                global_model.predictor.state_dict()
            )
            sd, n, stats, c_k = local_train_paper_partial(
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
        p_hat_y = _aggregate_phat(client_c)

        global_prototypes = _aggregate_prototypes(client_stats)
        update_generator_paper(generator_, gr_states, global_prototypes, device,
                                p_hat_y=p_hat_y,
                                noise_dim=noise_dim, lambda_proto=lambda_proto)
        total_bytes += bytes_round

        val_auc, test_auc = _eval_partial(global_model, local_models, clients, n_clients)
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
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def run_fedgen_paper_partial(clients, input_dim, seed,
                              hidden_dim, dropout, lr, batch_size,
                              n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                              local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                              noise_dim=NOISE_DIM):
    """Eq 5 hard CE + centroid anchor + p̂(y). Partial sharing."""
    return _run_paper_partial_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        n_clients=n_clients, fl_rounds=fl_rounds,
        local_epochs=local_epochs, patience=patience,
        lambda_proto=LAMBDA_PROTO, noise_dim=noise_dim,
        variant_name='fedgen_paper_partial',
    )


def run_fedgen_paper_partial_no_proto(clients, input_dim, seed,
                                       hidden_dim, dropout, lr, batch_size,
                                       n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                                       local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                                       noise_dim=NOISE_DIM):
    """Eq 5 hard CE + no anchor (λ=0) + p̂(y). Partial sharing."""
    return _run_paper_partial_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        n_clients=n_clients, fl_rounds=fl_rounds,
        local_epochs=local_epochs, patience=patience,
        lambda_proto=0.0, noise_dim=noise_dim,
        variant_name='fedgen_paper_partial_no_proto',
    )
