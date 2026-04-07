"""
UC1FLUtils.py
All FL model classes and training functions for UC1.
Imported by 04_UC1_Experiments.ipynb and 05_UC1_Results.ipynb.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
import os
import sys


sys.path.insert(0, '..')
from UC1Utils import prepare_data

SEED         = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH    = '../diabetes_data/diabetic_data.csv'
OUTPUT_DIR   = '../Federated/federated_data'

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_CLIENTS    = 5       # must match the partition in 02_UC1_Federated.ipynb
FL_ROUNDS    = 30
LOCAL_EPOCHS = 10
BATCH_SIZE   = 128
LR           = 1e-3
GEN_LR       = 1e-3
NOISE_DIM    = 16
HIDDEN_DIM   = 512     # must match centralized checkpoint
DROPOUT      = 0.3
LAMBDA_PROTO = 0.1
GEN_STEPS    = 10      # generator update steps per round
PATIENCE     = 5


# ── Load clients ───────────────────────────────────────────────────────────────
def load_clients(output_dir, n_clients):
    clients = {}
    for i in range(n_clients):
        d = os.path.join(output_dir, f'client_{i}')
        clients[i] = {
            'X_train': torch.tensor(np.load(f'{d}/X_train.npy'), dtype=torch.float32),
            'y_train': torch.tensor(np.load(f'{d}/y_train.npy'), dtype=torch.long),
            'X_val':   torch.tensor(np.load(f'{d}/X_val.npy'),   dtype=torch.float32),
            'y_val':   torch.tensor(np.load(f'{d}/y_val.npy'),   dtype=torch.long),
            'X_test':  torch.tensor(np.load(f'{d}/X_test.npy'),  dtype=torch.float32),
            'y_test':  torch.tensor(np.load(f'{d}/y_test.npy'),  dtype=torch.long),
        }

    # ── Validate: reject clients with no positive training examples ───────────
    problems = []
    for i, c in clients.items():
        pos_rate = c['y_train'].float().mean().item()
        n        = len(c['y_train'])
        n_pos    = c['y_train'].sum().item()
        print(f'  client_{i}: {n:,} train  pos_rate={pos_rate*100:.1f}%  n_pos={int(n_pos)}')
        if n_pos == 0:
            problems.append(i)

    if problems:
        raise ValueError(
            f"Clients {problems} have zero positive training examples. "
            f"Re-run 02_UC1_Federated.ipynb with N_CLIENTS={n_clients} "
            f"and MIN_POS_RATE >= 0.01 to regenerate the partition."
        )
    return clients


clients   = load_clients(OUTPUT_DIR, N_CLIENTS)
input_dim = clients[0]['X_train'].shape[1]
latent_dim = HIDDEN_DIM // 4   # 128


# ── Communication cost helpers ─────────────────────────────────────────────────
def model_bytes(model):
    return sum(p.numel() for p in model.parameters()) * 4  # float32

def mb(n_bytes):
    return n_bytes / (1024 ** 2)

class MLP(nn.Module):
    """Shared architecture across all four variants."""
    def __init__(self, input_dim, hidden_dim=256, output_dim=2, dropout=0.3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4), nn.ReLU(), nn.Dropout(dropout),
        )
        self.predictor = nn.Linear(hidden_dim // 4, output_dim)

    def forward(self, x):
        return self.predictor(self.feature_extractor(x))

    def encode(self, x):
        return self.feature_extractor(x)


class Generator(nn.Module):
    """
    Lightweight generator G_w: (y, ε) → z ∈ ℝ^latent_dim.
    Learns to produce latent vectors consistent with the ensemble
    of client prediction rules. ~5k parameters.
    """
    def __init__(self, latent_dim=128, noise_dim=16, n_classes=2, embed_dim=4):
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_dim)
        self.net   = nn.Sequential(
            nn.Linear(embed_dim + noise_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

    def forward(self, y, eps):
        return self.net(torch.cat([self.embed(y), eps], dim=1))


# ── GMM Sampler: alternative to neural generator ──────────────────────────────
class GMMSampler:
    """
    Replaces the neural Generator with direct Gaussian sampling from
    per-class latent distributions estimated from client data.

    More interpretable than the neural generator:
    - No learned network — samples from N(μ_y, σ_y) estimated from real patients
    - Statistics are directly auditable
    - Updated each round from client latent distributions
    - Eliminates any concern about learned generator producing implausible vectors
    """
    def __init__(self, latent_dim, device):
        self.latent_dim = latent_dim
        self.device     = device
        # Initialise with unit Gaussians; will be updated after round 1
        self.params = {
            cls: {
                'mean': torch.zeros(latent_dim, device=device),
                'std':  torch.ones(latent_dim,  device=device),
            }
            for cls in [0, 1]
        }

    def update(self, client_distributions, sample_counts):
        """
        Weighted average of per-client (mean, std) for each class.
        client_distributions: list of {cls: {'mean', 'std', 'n'}} dicts
        """
        total = sum(sample_counts)
        for cls in [0, 1]:
            pairs = [
                (d[cls], n)
                for d, n in zip(client_distributions, sample_counts)
                if cls in d and d[cls]['n'] > 0
            ]
            if not pairs:
                continue
            wmean = sum(
                torch.tensor(d['mean'], dtype=torch.float32, device=self.device) * (n / total)
                for d, n in pairs
            )
            wstd = sum(
                torch.tensor(d['std'],  dtype=torch.float32, device=self.device) * (n / total)
                for d, n in pairs
            )
            self.params[cls] = {'mean': wmean, 'std': wstd}

    def sample(self, n_per_class):
        """Return (Z, y) balanced batch sampled from class-conditional Gaussians."""
        Z_list, y_list = [], []
        for cls in [0, 1]:
            mu  = self.params[cls]['mean']
            std = self.params[cls]['std']
            Z_c = torch.randn(n_per_class, self.latent_dim, device=self.device) * std + mu
            y_c = torch.full((n_per_class,), cls, dtype=torch.long, device=self.device)
            Z_list.append(Z_c)
            y_list.append(y_c)
        return torch.cat(Z_list), torch.cat(y_list)

    def model_bytes(self):
        """Bytes to transmit the GMM parameters (mean + std per class)."""
        return 2 * 2 * self.latent_dim * 4  # 2 classes × 2 arrays × latent_dim × 4 bytes
    

def make_criterion(y_train_numpy, device):
    """Class-weighted CE loss. Falls back to unweighted for single-class clients."""
    classes = np.unique(y_train_numpy)
    if len(classes) < 2:
        return nn.CrossEntropyLoss()
    cw = compute_class_weight('balanced', classes=classes, y=y_train_numpy)
    return nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))


def evaluate_global(global_model, clients, device, use_local_encoder=False,
                    local_models=None):
    """
    Evaluate global predictor on all clients' test data.
    use_local_encoder=True → partial sharing: each client uses its own g_f.
    use_local_encoder=False → full sharing: global g_f used for all.
    """
    all_proba, all_y = [], []
    per_client = {}

    for i, c in clients.items():
        if use_local_encoder:
            enc_model = local_models[i]
        else:
            enc_model = global_model

        enc_model.eval()
        global_model.eval()

        with torch.no_grad():
            z     = enc_model.encode(c['X_test'].to(device))
            logits = global_model.predictor(z)
            proba  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        y_test = c['y_test'].numpy()
        all_proba.append(proba)
        all_y.append(y_test)

        if len(np.unique(y_test)) > 1:
            per_client[i] = roc_auc_score(y_test, proba)

    global_auc = roc_auc_score(np.concatenate(all_y), np.concatenate(all_proba))
    return global_auc, per_client


def compute_client_distribution(model, X_tensor, y_numpy, device):
    """Per-class (mean, std, n) of latent representations."""
    model.eval()
    with torch.no_grad():
        Z = model.encode(X_tensor.to(device)).cpu().numpy()
    dist = {}
    for cls in [0, 1]:
        mask = y_numpy == cls
        if mask.sum() > 0:
            dist[cls] = {
                'mean': Z[mask].mean(0),
                'std':  Z[mask].std(0) + 1e-6,
                'n':    int(mask.sum()),
            }
    return dist


def fed_avg(state_dicts, sample_counts):
    """Weighted average of state dicts."""
    total = sum(sample_counts)
    return {
        k: sum(sd[k] * (sample_counts[i] / total) for i, sd in enumerate(state_dicts))
        for k in state_dicts[0].keys()
    }


def local_train_fedavg_full(model, X_train, y_train, X_val, y_val,
                             epochs, batch_size, device):
    """
    Standard FedAvg local training.
    Full model updated, full model returned to server.
    """
    loader    = DataLoader(TensorDataset(X_train, y_train),
                           batch_size=batch_size, shuffle=True)
    criterion = make_criterion(y_train.numpy(), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_auc, best_state = 0.0, None

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_p = torch.softmax(model(X_val.to(device)), dim=1)[:, 1].cpu().numpy()
        if len(np.unique(y_val.numpy())) > 1:
            auc = roc_auc_score(y_val.numpy(), val_p)
            if auc > best_val_auc:
                best_val_auc = auc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    return best_state or {k: v.clone() for k, v in model.state_dict().items()}, len(X_train)


def run_fedavg_full(clients, input_dim, n_rounds, patience, device):
    """
    Full model FedAvg federation loop.
    Communication per round per client: 2 × full_model_size (upload + download).
    """
    global_model = MLP(input_dim, HIDDEN_DIM, dropout=DROPOUT).to(device)
    _probe       = MLP(input_dim, HIDDEN_DIM, dropout=DROPOUT)
    bytes_per_round_per_client = 2 * model_bytes(_probe)

    history          = []   # list of {'val': float, 'test': float}
    cumul_mb         = []   # cumulative MB after each round
    total_bytes      = 0
    best_val, no_imp = 0.0, 0
    best_state_dict  = None

    for r in range(n_rounds):
        states, counts = [], []

        for i, c in clients.items():
            m = MLP(input_dim, HIDDEN_DIM, dropout=DROPOUT).to(device)
            m.load_state_dict(global_model.state_dict())           # full model broadcast
            sd, n = local_train_fedavg_full(
                m, c['X_train'], c['y_train'], c['X_val'], c['y_val'],
                LOCAL_EPOCHS, BATCH_SIZE, device
            )
            states.append(sd)
            counts.append(n)

        global_model.load_state_dict(fed_avg(states, counts))
        total_bytes += bytes_per_round_per_client * N_CLIENTS

        val_auc, _  = evaluate_global(global_model, clients, device)
        test_auc, _ = evaluate_global(global_model, clients, device)
        history.append({'val': val_auc, 'test': test_auc})
        cumul_mb.append(mb(total_bytes))
        print(f'Round {r+1:>2} | Val AUC {val_auc:.4f} | Test AUC {test_auc:.4f} '
              f'| {mb(total_bytes):.1f} MB total')

        if val_auc > best_val:
            best_val        = val_auc
            no_imp          = 0
            best_state_dict = {k: v.clone() for k, v in global_model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f'Early stopping at round {r+1}')
                break

    global_model.load_state_dict(best_state_dict)
    return global_model, history, cumul_mb


def update_generator_full(generator, full_model_states, prototypes,
                           n_steps, device):
    """
    Server-side generator update using ensemble of all clients' full models.
    Gradient flows through generator only — client models are frozen.
    Loss = CE(ensemble_logits(G(y,ε)), y) + λ·‖G(y,ε) − z̄_y‖²
    """
    # Extract predictor weights from full model state dicts
    # and average them for an efficient ensemble approximation
    mean_w = torch.stack([
        sd['predictor.weight'] for sd in full_model_states
    ]).mean(0).to(device)
    mean_b = torch.stack([
        sd['predictor.bias'] for sd in full_model_states
    ]).mean(0).to(device)

    proto_t = {
        cls: torch.tensor(v, dtype=torch.float32, device=device)
        for cls, v in prototypes.items()
    }

    opt = torch.optim.Adam(generator.parameters(), lr=GEN_LR)
    criterion = nn.CrossEntropyLoss()
    generator.train()

    for _ in range(n_steps):
        opt.zero_grad()
        half  = 64
        y_gen = torch.cat([
            torch.zeros(half, dtype=torch.long, device=device),
            torch.ones( half, dtype=torch.long, device=device),
        ])
        eps    = torch.randn(len(y_gen), NOISE_DIM, device=device)
        Z_gen  = generator(y_gen, eps)

        loss_ce    = criterion(Z_gen @ mean_w.T + mean_b, y_gen)
        proto_tgt  = torch.stack([proto_t[int(y.item())] for y in y_gen])
        loss_proto = ((Z_gen - proto_tgt) ** 2).mean()

        (loss_ce + LAMBDA_PROTO * loss_proto).backward()
        opt.step()

    generator.eval()


def local_train_fedgen_full(model, generator, X_train, y_train, X_val, y_val,
                             epochs, batch_size, device):
    """
    Full model FedGen local training.
    Full model updated on real data + KD loss from frozen generator.
    Full model returned to server for aggregation.
    """
    loader        = DataLoader(TensorDataset(X_train, y_train),
                               batch_size=batch_size, shuffle=True)
    criterion_real = make_criterion(y_train.numpy(), device)
    criterion_kd   = nn.CrossEntropyLoss()
    optimizer      = torch.optim.Adam(model.parameters(), lr=LR)

    # Generate synthetic batch once per round (generator frozen during local training)
    n_syn = min(batch_size, len(X_train))
    with torch.no_grad():
        y_hat  = torch.randint(0, 2, (n_syn,), device=device)
        eps    = torch.randn(n_syn, NOISE_DIM, device=device)
        Z_hat  = generator(y_hat, eps)

    best_val_auc, best_state = 0.0, None

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            # Real data: update full model
            loss_real = criterion_real(model(xb), yb)
            # KD: synthetic z through predictor only (g_f not involved)
            loss_kd   = criterion_kd(model.predictor(Z_hat), y_hat)
            (loss_real + loss_kd).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_p = torch.softmax(model(X_val.to(device)), dim=1)[:, 1].cpu().numpy()
        if len(np.unique(y_val.numpy())) > 1:
            auc = roc_auc_score(y_val.numpy(), val_p)
            if auc > best_val_auc:
                best_val_auc = auc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    return best_state or {k: v.clone() for k, v in model.state_dict().items()}, len(X_train)


def run_fedgen_full(clients, input_dim, prototypes, n_rounds, patience, device):
    """
    Full model FedGen federation loop.
    Communication per round per client:
      upload:   full model
      download: full model + generator
    """
    global_model = MLP(input_dim, HIDDEN_DIM, dropout=DROPOUT).to(device)
    generator    = Generator(latent_dim, NOISE_DIM).to(device)

    _probe_m = MLP(input_dim, HIDDEN_DIM, dropout=DROPOUT)
    _probe_g = Generator(latent_dim, NOISE_DIM)
    bytes_up   = model_bytes(_probe_m)                      # upload: full model
    bytes_down = model_bytes(_probe_m) + model_bytes(_probe_g)  # download: model + generator

    history, cumul_mb = [], []
    total_bytes = 0
    best_val, no_imp = 0.0, 0
    best_state_dict, best_gen_state = None, None

    for r in range(n_rounds):
        states, counts = [], []

        for i, c in clients.items():
            m = MLP(input_dim, HIDDEN_DIM, dropout=DROPOUT).to(device)
            m.load_state_dict(global_model.state_dict())    # full model broadcast
            sd, n = local_train_fedgen_full(
                m, generator, c['X_train'], c['y_train'],
                c['X_val'], c['y_val'], LOCAL_EPOCHS, BATCH_SIZE, device
            )
            states.append(sd)
            counts.append(n)

        global_model.load_state_dict(fed_avg(states, counts))
        update_generator_full(generator, states, prototypes, GEN_STEPS, device)

        total_bytes += (bytes_up + bytes_down) * N_CLIENTS

        val_auc, _  = evaluate_global(global_model, clients, device)
        test_auc, _ = evaluate_global(global_model, clients, device)
        history.append({'val': val_auc, 'test': test_auc})
        cumul_mb.append(mb(total_bytes))
        print(f'Round {r+1:>2} | Val AUC {val_auc:.4f} | Test AUC {test_auc:.4f} '
              f'| {mb(total_bytes):.1f} MB total')

        if val_auc > best_val:
            best_val        = val_auc
            no_imp          = 0
            best_state_dict = {k: v.clone() for k, v in global_model.state_dict().items()}
            best_gen_state  = {k: v.clone() for k, v in generator.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f'Early stopping at round {r+1}')
                break

    global_model.load_state_dict(best_state_dict)
    generator.load_state_dict(best_gen_state)
    return global_model, generator, history, cumul_mb


def local_train_fedavg_partial(model, X_train, y_train, X_val, y_val,
                                epochs, batch_size, device):
    """
    Partial sharing FedAvg: feature extractor stays local, only predictor head shared.
    This is the correct baseline for comparing against partial-sharing FedGen.
    """
    loader        = DataLoader(TensorDataset(X_train, y_train),
                               batch_size=batch_size, shuffle=True)
    criterion     = make_criterion(y_train.numpy(), device)
    optimizer     = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_auc, best_gr_state = 0.0, None

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_p = torch.softmax(model(X_val.to(device)), dim=1)[:, 1].cpu().numpy()
        if len(np.unique(y_val.numpy())) > 1:
            auc = roc_auc_score(y_val.numpy(), val_p)
            if auc > best_val_auc:
                best_val_auc  = auc
                best_gr_state = {k: v.clone() for k, v in model.predictor.state_dict().items()}
        else:
            best_gr_state = {k: v.clone() for k, v in model.predictor.state_dict().items()}

    return (best_gr_state or
            {k: v.clone() for k, v in model.predictor.state_dict().items()}), len(X_train)


def run_fedavg_partial(clients, input_dim, n_rounds, patience, device):
    """
    Partial-sharing FedAvg.
    Communication per round per client:
      upload:   predictor head only
      download: predictor head only
    """
    global_model  = MLP(input_dim, HIDDEN_DIM, dropout=DROPOUT).to(device)
    local_models  = {
        i: MLP(input_dim, HIDDEN_DIM, dropout=DROPOUT).to(device)
        for i in range(N_CLIENTS)
    }
    for i in range(N_CLIENTS):
        local_models[i].load_state_dict(global_model.state_dict())

    _probe = MLP(input_dim, HIDDEN_DIM, dropout=DROPOUT)
    bytes_per_round_per_client = 2 * model_bytes(_probe.predictor)

    history, cumul_mb = [], []
    total_bytes = 0
    best_val, no_imp = 0.0, 0
    best_gr_state = None

    for r in range(n_rounds):
        gr_states, counts = [], []

        for i, c in clients.items():
            local_models[i].predictor.load_state_dict(
                global_model.predictor.state_dict()   # only predictor broadcast
            )
            sd, n = local_train_fedavg_partial(
                local_models[i], c['X_train'], c['y_train'],
                c['X_val'], c['y_val'], LOCAL_EPOCHS, BATCH_SIZE, device
            )
            gr_states.append(sd)
            counts.append(n)

        global_model.predictor.load_state_dict(fed_avg(gr_states, counts))
        total_bytes += bytes_per_round_per_client * N_CLIENTS

        val_auc, _  = evaluate_global(global_model, clients, device,
                                      use_local_encoder=True, local_models=local_models)
        test_auc, _ = evaluate_global(global_model, clients, device,
                                      use_local_encoder=True, local_models=local_models)
        history.append({'val': val_auc, 'test': test_auc})
        cumul_mb.append(mb(total_bytes))
        print(f'Round {r+1:>2} | Val AUC {val_auc:.4f} | Test AUC {test_auc:.4f} '
              f'| {mb(total_bytes):.1f} MB total')

        if val_auc > best_val:
            best_val      = val_auc
            no_imp        = 0
            best_gr_state = {k: v.clone() for k, v in global_model.predictor.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f'Early stopping at round {r+1}')
                break

    global_model.predictor.load_state_dict(best_gr_state)
    return global_model, local_models, history, cumul_mb

def local_train_fedgen_partial(model, generator, X_train, y_train, X_val, y_val,
                                epochs, batch_size, device):
    """
    Partial-sharing FedGen (your current approach, fixed).
    Feature extractor stays local. Only predictor head shared.
    Generator provides KD signal for the predictor.
    Two-step update per batch:
      Step 1 — real data → update full model (both g_f and g_r)
      Step 2 — synthetic z → update predictor only (g_r)
    """
    loader         = DataLoader(TensorDataset(X_train, y_train),
                                batch_size=batch_size, shuffle=True)
    criterion_real = make_criterion(y_train.numpy(), device)
    criterion_kd   = nn.CrossEntropyLoss()
    opt_full       = torch.optim.Adam(model.parameters(),           lr=LR)
    opt_gr         = torch.optim.Adam(model.predictor.parameters(), lr=LR)

    n_syn = min(batch_size, len(X_train))
    with torch.no_grad():
        y_hat = torch.randint(0, 2, (n_syn,), device=device)
        eps   = torch.randn(n_syn, NOISE_DIM, device=device)
        Z_hat = generator(y_hat, eps)

    best_val_auc, best_gr_state = 0.0, None

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            # Step 1: real data — full model
            model.zero_grad()
            criterion_real(model(xb), yb).backward()
            opt_full.step()

            # Step 2: synthetic KD — predictor only
            model.predictor.zero_grad()
            criterion_kd(model.predictor(Z_hat), y_hat).backward()
            opt_gr.step()

        model.eval()
        with torch.no_grad():
            val_p = torch.softmax(model(X_val.to(device)), dim=1)[:, 1].cpu().numpy()
        if len(np.unique(y_val.numpy())) > 1:
            auc = roc_auc_score(y_val.numpy(), val_p)
            if auc > best_val_auc:
                best_val_auc  = auc
                best_gr_state = {k: v.clone() for k, v in model.predictor.state_dict().items()}
        else:
            best_gr_state = {k: v.clone() for k, v in model.predictor.state_dict().items()}

    if best_gr_state is None:
        best_gr_state = {k: v.clone() for k, v in model.predictor.state_dict().items()}

    dist = compute_client_distribution(model, X_train, y_train.numpy(), device)
    return best_gr_state, len(X_train), dist


def update_generator_partial(generator, gr_states, prototypes, n_steps, device):
    """
    Server-side generator update using only the predictor heads.
    Same objective as full-model version but only 258-param heads are needed.
    """
    mean_w = torch.stack([
        sd['weight'] for sd in gr_states
    ]).mean(0).to(device)
    mean_b = torch.stack([
        sd['bias'] for sd in gr_states
    ]).mean(0).to(device)

    proto_t = {
        cls: torch.tensor(v, dtype=torch.float32, device=device)
        for cls, v in prototypes.items()
    }

    opt = torch.optim.Adam(generator.parameters(), lr=GEN_LR)
    criterion = nn.CrossEntropyLoss()
    generator.train()

    for _ in range(n_steps):
        opt.zero_grad()
        half  = 64
        y_gen = torch.cat([
            torch.zeros(half, dtype=torch.long, device=device),
            torch.ones( half, dtype=torch.long, device=device),
        ])
        eps    = torch.randn(len(y_gen), NOISE_DIM, device=device)
        Z_gen  = generator(y_gen, eps)

        loss_ce    = criterion(Z_gen @ mean_w.T + mean_b, y_gen)
        proto_tgt  = torch.stack([proto_t[int(y.item())] for y in y_gen])
        loss_proto = ((Z_gen - proto_tgt) ** 2).mean()

        (loss_ce + LAMBDA_PROTO * loss_proto).backward()
        opt.step()

    generator.eval()


def run_fedgen_partial(clients, input_dim, prototypes, n_rounds, patience, device):
    """
    Partial-sharing FedGen — your current approach, corrected.
    Communication per round per client:
      upload:   predictor head
      download: predictor head + generator
    """
    global_model = MLP(input_dim, HIDDEN_DIM, dropout=DROPOUT).to(device)
    generator    = Generator(latent_dim, NOISE_DIM).to(device)
    local_models = {
        i: MLP(input_dim, HIDDEN_DIM, dropout=DROPOUT).to(device)
        for i in range(N_CLIENTS)
    }
    for i in range(N_CLIENTS):
        local_models[i].load_state_dict(global_model.state_dict())

    _probe_m = MLP(input_dim, HIDDEN_DIM, dropout=DROPOUT)
    _probe_g = Generator(latent_dim, NOISE_DIM)
    bytes_up   = model_bytes(_probe_m.predictor)
    bytes_down = model_bytes(_probe_m.predictor) + model_bytes(_probe_g)

    history, cumul_mb = [], []
    total_bytes = 0
    best_val, no_imp = 0.0, 0
    best_gr_state, best_gen_state = None, None

    for r in range(n_rounds):
        gr_states, counts, dists = [], [], []

        for i, c in clients.items():
            local_models[i].predictor.load_state_dict(
                global_model.predictor.state_dict()
            )
            sd, n, dist = local_train_fedgen_partial(
                local_models[i], generator,
                c['X_train'], c['y_train'], c['X_val'], c['y_val'],
                LOCAL_EPOCHS, BATCH_SIZE, device
            )
            gr_states.append(sd)
            counts.append(n)
            dists.append(dist)

        global_model.predictor.load_state_dict(fed_avg(gr_states, counts))
        update_generator_partial(generator, gr_states, prototypes, GEN_STEPS, device)
        total_bytes += (bytes_up + bytes_down) * N_CLIENTS

        val_auc, _  = evaluate_global(global_model, clients, device,
                                      use_local_encoder=True, local_models=local_models)
        test_auc, _ = evaluate_global(global_model, clients, device,
                                      use_local_encoder=True, local_models=local_models)
        history.append({'val': val_auc, 'test': test_auc})
        cumul_mb.append(mb(total_bytes))
        print(f'Round {r+1:>2} | Val AUC {val_auc:.4f} | Test AUC {test_auc:.4f} '
              f'| {mb(total_bytes):.1f} MB total')

        if val_auc > best_val:
            best_val       = val_auc
            no_imp         = 0
            best_gr_state  = {k: v.clone() for k, v in global_model.predictor.state_dict().items()}
            best_gen_state = {k: v.clone() for k, v in generator.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f'Early stopping at round {r+1}')
                break

    global_model.predictor.load_state_dict(best_gr_state)
    generator.load_state_dict(best_gen_state)
    return global_model, generator, local_models, history, cumul_mb