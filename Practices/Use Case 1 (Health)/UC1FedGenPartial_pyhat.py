"""
UC1FedGenPartial_pyhat.py
─────────────────────────────────────────────────────────────────────────────
Paper-faithful p̂(y) variant of the partial-sharing FedGen family.

Algorithm 1 of Zhu et al. (2021) prescribes that synthetic labels ŷ used to
condition the generator are drawn from p̂(y), the empirically estimated global
label prior — *not* from a uniform Bernoulli as the baseline implementation
does. This file reproduces that detail end-to-end:

    Line 1  — p̂(y) initialised uniform.
    Line 3  — server broadcasts p̂(y) to active users (all users here).
    Line 7  — ŷ_i ~ p̂(y) when sampling synthetic features.
    Line 8  — each client computes its label counter c_k.
    Line 11 — c_k is uploaded alongside θ_k.
    Line 13 — server updates p̂(y) ∝ Σ_k c_k after FedAvg.

Important nuances kept in mind:

1. **Round-0 staleness**: p̂(y) is uniform [0.5, 0.5] at the start of round 1,
   so the very first round of local training samples uniformly — same as the
   baseline. From round 2 onward, p̂(y) converges to the true global prior.

2. **Counter scope**: c_k counts labels in the *training* set actually used
   by the client during local training, not the validation set.

3. **Privacy**: uploading c_k leaks the per-client label distribution. For
   binary readmission prediction with population-level prevalence (~11.5% in
   the Diabetes 130-US Hospitals data) this is a minor leak, but it is a
   genuine deviation from FedAvg's information disclosure profile and worth
   noting in the TFG write-up.

4. **Active subset**: line 3 prescribes a random subset A ⊆ {1..K}. The TFG
   uses full participation (|A| = K = 5), so {c_k}_{k∈A} = {c_k}_{k=1..K}.

5. **Generator-side sampling — NOT covered here**. ``update_generator`` in
   ``UC1FLUtils.py`` also samples ŷ when computing the generator's CE loss
   (Equation 4). To be strictly faithful to the paper, that sampling should
   also use p̂(y). Modifying that helper is out of scope for this test file
   (it would require editing UC1FLUtils.py). The client-side fix here is
   sufficient to demonstrate the effect of p̂(y) on the KD signal and
   per-class equity, which is what the test is meant to probe.

This file mirrors the structure of UC1FedGenPartial.py and only contains the
zhu (KL-ensemble KD) variants, since those are the closest to Algorithm 1
and the only ones where p̂(y) materially changes the math.
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
    update_generator, update_generator_pyhat,
    N_CLIENTS, FL_ROUNDS, LOCAL_EPOCHS, PATIENCE, NOISE_DIM, LAMBDA_PROTO,
    device,
)

NUM_CLASSES = 2  # binary readmission task




# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _label_counter(y_train, num_classes=NUM_CLASSES):
    """
    Compute c_k: per-class counts on the local training set.
    Returns a torch.float tensor of shape (num_classes,).
    """
    y_np = y_train.numpy() if isinstance(y_train, torch.Tensor) else np.asarray(y_train)
    counts = np.bincount(y_np.astype(np.int64), minlength=num_classes).astype(np.float64)
    return torch.tensor(counts, dtype=torch.float32)


def _aggregate_phat(c_list, num_classes=NUM_CLASSES, eps=1e-8):
    """
    Aggregate {c_k} into a normalised p̂(y) over classes (line 13).

    p̂(y=c) = (Σ_k count_k(c)) / (Σ_k Σ_c' count_k(c'))

    A small eps floor is added to prevent zeroing out a class entirely if
    it happens to be absent from every active client (would crash the
    multinomial sampler). For Marc's setup with α=0.5 this rarely matters,
    but it makes the aggregation robust against unlucky Dirichlet draws.
    """
    total = torch.stack(c_list).sum(dim=0)            # (num_classes,)
    total = total + eps
    p_hat = total / total.sum()
    return p_hat


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
# Local training — KL-ensemble KD with p̂(y) sampling
# ═════════════════════════════════════════════════════════════════════════════

def local_train_fedgen_zhu_pyhat(model, generator, all_predictor_states, p_hat_y,
                                  X_train, y_train, X_val, y_val,
                                  device, batch_size=128, lr=1e-3,
                                  local_epochs=LOCAL_EPOCHS, noise_dim=NOISE_DIM):
    """
    Paper-faithful zhu-partial variant with ŷ ~ p̂(y).

    Difference vs. local_train_fedgen_zhu (baseline):
        y_hat = torch.randint(0, 2, ...)                       # baseline
        y_hat = torch.multinomial(p_hat_y, n_syn, replacement=True)  # this file

    Returns
    -------
    best_gr_state : dict     — best predictor state by val AUC
    n_train       : int      — len(X_train), used as FedAvg weight
    per_class_stats : dict   — centroid prototypes (for the prototype anchor)
    c_k           : Tensor   — label counter for this client (for p̂(y) update)
    """
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
            # ── line 7: ŷ_i ~ p̂(y) ─────────────────────────────────────────
            y_hat = torch.multinomial(p_hat_dev, n_syn, replacement=True)
            eps   = torch.randn(n_syn, noise_dim, device=device)
            Z_hat = generator(y_hat, eps)
            # KL-ensemble teacher: average of softmax(W_k Ẑ + b_k)
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
    """KL-ensemble KD + medoid anchor + p̂(y) sampling."""
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


# ═════════════════════════════════════════════════════════════════════════════
# Run loop with p̂(y) maintenance
# ═════════════════════════════════════════════════════════════════════════════

def _run_fedgen_partial_pyhat_generic(clients, input_dim, seed,
                                       hidden_dim, dropout, lr, batch_size,
                                       local_train_fn,
                                       n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                                       local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                                       lambda_proto=LAMBDA_PROTO,
                                       noise_dim=NOISE_DIM,
                                       variant_name='fedgen_zhu_pyhat'):
    """
    Run loop for partial-sharing zhu variants with paper-faithful p̂(y).

    Differences from _run_fedgen_partial_generic:
      • Maintains p_hat_y across rounds (initialised uniform).
      • Passes p_hat_y to local_train_fn.
      • Collects c_k from each client and updates p_hat_y after FedAvg.
      • Logs the p_hat_y trajectory in the history dict for inspection.
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
    # +NUM_CLASSES floats for c_k upload — negligible vs. predictor weights
    bytes_round = (2 * n_pred + n_gen) * 4 * n_clients + NUM_CLASSES * 4 * n_clients

    best_val_auc, best_gr_state, best_gen_state, no_improve = 0.0, None, None, 0
    history, cumul_mb, total_bytes = [], [], 0

    # ── line 1: p̂(y) uniformly initialised ────────────────────────────────────
    p_hat_y = torch.full((NUM_CLASSES,), 1.0 / NUM_CLASSES)

    # Ensemble teacher (per-client previous heads) — initialised from global
    all_predictor_states = [
        {k: v.clone() for k, v in global_model.predictor.state_dict().items()}
        for _ in range(n_clients)
    ]

    for fl_round in range(fl_rounds):
        gr_states, counts, client_stats, client_c = [], [], [], []

        for i in range(n_clients):
            local_models[i].predictor.load_state_dict(
                global_model.predictor.state_dict()
            )
            sd, n, stats, c_k = local_train_fn(
                local_models[i], generator_, all_predictor_states, p_hat_y,
                clients[i]['X_train'], clients[i]['y_train'],
                clients[i]['X_val'],   clients[i]['y_val'],
                device, batch_size=batch_size, lr=lr,
                local_epochs=local_epochs, noise_dim=noise_dim,
            )
            gr_states.append(sd)
            counts.append(n)
            client_stats.append(stats)
            client_c.append(c_k)

        # ── Server aggregation (line 13) ───────────────────────────────────
        global_model.predictor.load_state_dict(fed_avg(gr_states, counts))
        all_predictor_states = gr_states                          # next-round teacher
        p_hat_y = _aggregate_phat(client_c)                       # ← new each round

        global_prototypes = _aggregate_prototypes(client_stats)
        # NOTE: update_generator() in UC1FLUtils.py still samples ŷ uniformly
        # for its own CE loss. A fully paper-faithful version would also use
        # p_hat_y here; that fix lives outside the scope of this test file.
        update_generator_pyhat(generator_, gr_states, global_prototypes, device,
                                p_hat_y=p_hat_y,
                         noise_dim=noise_dim, lambda_proto=lambda_proto)
        total_bytes += bytes_round

        val_auc, test_auc = _eval_partial(global_model, local_models, clients, n_clients)
        history.append({
            'val': val_auc,
            'test': test_auc,
            'p_hat_y': p_hat_y.tolist(),       # logged for inspection
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


def run_fedgen_zhu_pyhat(clients, input_dim, seed,
                          hidden_dim, dropout, lr, batch_size,
                          n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                          local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                          noise_dim=NOISE_DIM):
    """zhu KL-ensemble KD + centroid anchor + p̂(y) sampling."""
    return _run_fedgen_partial_pyhat_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_zhu_pyhat,
        n_clients=n_clients, fl_rounds=fl_rounds,
        local_epochs=local_epochs, patience=patience,
        noise_dim=noise_dim, variant_name='fedgen_zhu_pyhat',
    )


def run_fedgen_zhu_pyhat_medoid(clients, input_dim, seed,
                                 hidden_dim, dropout, lr, batch_size,
                                 n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                                 local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                                 noise_dim=NOISE_DIM):
    """zhu KL-ensemble KD + medoid anchor + p̂(y) sampling."""
    return _run_fedgen_partial_pyhat_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_zhu_pyhat_medoid,
        n_clients=n_clients, fl_rounds=fl_rounds,
        local_epochs=local_epochs, patience=patience,
        noise_dim=noise_dim, variant_name='fedgen_zhu_pyhat_medoid',
    )


def run_fedgen_zhu_pyhat_no_proto(clients, input_dim, seed,
                                   hidden_dim, dropout, lr, batch_size,
                                   n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                                   local_epochs=LOCAL_EPOCHS, patience=PATIENCE,
                                   noise_dim=NOISE_DIM):
    """
    Cleanest paper-faithful variant: Eq. 4 + Eq. 5 with p̂(y) sampling and
    no prototype constraint (lambda_proto = 0).
    """
    return _run_fedgen_partial_pyhat_generic(
        clients, input_dim, seed, hidden_dim, dropout, lr, batch_size,
        local_train_fn=local_train_fedgen_zhu_pyhat,
        n_clients=n_clients, fl_rounds=fl_rounds,
        local_epochs=local_epochs, patience=patience,
        noise_dim=noise_dim, lambda_proto=0.0,
        variant_name='fedgen_zhu_pyhat_no_proto',
    )
