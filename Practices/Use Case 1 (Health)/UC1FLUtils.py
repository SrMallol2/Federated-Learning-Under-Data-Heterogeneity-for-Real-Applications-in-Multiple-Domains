"""
UC1FLUtils.py
─────────────────────────────────────────────────────────────────────────────
Shared FL utility functions for UC1 (Diabetes 130-US Hospitals).

Design principles
-----------------
- No module-level side effects: no data loading, no random seeds, no file I/O.
- No hardcoded paths: every function that touches the filesystem receives
  output_dir / results_dir as an explicit argument.
- Constants here are architectural defaults only (N_CLIENTS, FL_ROUNDS …).
  Hyperparameters tuned by Optuna (HIDDEN_DIM, LR, DROPOUT, BATCH_SIZE) are
  expected to come from the notebook via fl_hyperparams.json and are passed
  as arguments to functions that need them.

Scope
-----
This module holds the shared core used by both sharing families.
FedGen training loops live elsewhere:
    - Partial-sharing variants  → UC1FedGenPartial.py
    - Full-sharing    variants  → UC1FedGenFull.py

Exported symbols
----------------
Data utilities:
    dirichlet_partition, find_feasible_params, create_clients_raw_csv,
    preprocess_and_save_client, preprocess_clients,
    load_clients, load_partitions_from_disk, verify_leakage, save_result

Model classes:
    MLP, Generator, GMMSampler

Training helpers:
    model_bytes, mb, make_criterion, fed_avg, evaluate_global

Prototype / latent-stats helpers (shared by all FedGen variants):
    compute_client_distribution
    _compute_local_prototypes      — centroid (arithmetic mean) anchor
    _compute_local_medoid_proxy    — medoid (nearest real point) anchor
    _aggregate_prototypes
    _label_counter                 — per-client label counts for p̂(y)
    _aggregate_phat                — aggregate {c_k} into normalised p̂(y)

Generator updates (server-side):
    update_generator       — partial-sharing variant (predictor weights only)
    update_generator_full  — full-sharing variant   (reads predictor.weight/bias)
    Both accept optional p_hat_y for paper-faithful label sampling.

FedAvg baselines:
    local_train_fedavg_full,    local_train_fedavg_partial
    run_fedavg_full,            run_fedavg_partial

Hyperparameter search:
    fl_objective
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
import optuna

sys.path.insert(0, '..')
from UC1Utils import prepare_data_aligned, split_data, scale_data
import joblib

# ── Architectural / FL defaults ───────────────────────────────────────────────
# These are stable across experiments. Hyperparameters that Optuna tunes
# (HIDDEN_DIM, LR, DROPOUT, BATCH_SIZE) are NOT listed here — they are
# passed as arguments to functions.

N_CLIENTS    = 5
FL_ROUNDS    = 30
LOCAL_EPOCHS = 10
PATIENCE     = 5

GEN_LR       = 1e-3
NOISE_DIM    = 16
LAMBDA_PROTO = 0.1
GEN_STEPS    = 10

MIN_PATIENTS = 500
MIN_POS_RATE = 0.01
MAX_RETRIES  = 500

ALPHA_SWEEP  = [0.5, 1.0, 5.0, 10.0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ═════════════════════════════════════════════════════════════════════════════
# DATA UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def dirichlet_partition(df_raw, n_clients=N_CLIENTS, alpha=0.5, seed=42,
                        min_patients=MIN_PATIENTS, min_pos_rate=MIN_POS_RATE,
                        min_pos_abs=200, max_retries=MAX_RETRIES,
                        filtered=True):
    """
    Partition patients into n_clients groups via a Dirichlet draw.

    Parameters
    ----------
    filtered : bool
        True  → applies min_patients, min_pos_rate, min_pos_abs constraints.
                 Use for the training partition.
        False → only requires each client to be non-empty.
                 Use for the unfiltered analysis comparison.

    Returns
    -------
    list of lists: client_patient_ids[k] = list of patient_nbr values.
    """
    rng = np.random.default_rng(seed)
    patient_labels = (
        df_raw.groupby('patient_nbr')['readmitted']
        .apply(lambda x: 1 if (x == '<30').any() else 0)
        .reset_index()
    )

    for attempt in range(max_retries):
        client_patient_ids = [[] for _ in range(n_clients)]
        for cls in [0, 1]:
            patients_in_class = patient_labels[
                patient_labels['readmitted'] == cls
            ]['patient_nbr'].values.copy()
            rng.shuffle(patients_in_class)
            proportions = rng.dirichlet([alpha] * n_clients)
            counts = (proportions * len(patients_in_class)).astype(int)
            counts[-1] = len(patients_in_class) - counts[:-1].sum()
            start = 0
            for k, n in enumerate(counts):
                client_patient_ids[k].extend(
                    patients_in_class[start:start + n].tolist()
                )
                start += n

        sizes     = [len(c) for c in client_patient_ids]
        pos_rates = []
        for ids in client_patient_ids:
            labels = patient_labels[
                patient_labels['patient_nbr'].isin(ids)
            ]['readmitted']
            pos_rates.append(labels.mean())

        if not filtered:
            if min(sizes) > 0:
                return client_patient_ids
        else:
            if min(sizes) >= min_patients and min(pos_rates) >= min_pos_rate:
                pos_counts = [round(pos_rates[k] * sizes[k]) for k in range(n_clients)]
                if min(pos_counts) >= min_pos_abs:
                    print(f'  Valid split on attempt {attempt + 1}.')
                    print(f'  Sizes:      min={min(sizes):,}  max={max(sizes):,}')
                    print(f'  Pos rates:  min={min(pos_rates):.3f}  max={max(pos_rates):.3f}')
                    print(f'  Pos counts: min={min(pos_counts):,}  max={max(pos_counts):,}')
                    return client_patient_ids

    kind = 'filtered' if filtered else 'unfiltered'
    raise ValueError(
        f'No valid {kind} split found after {max_retries} attempts for α={alpha}. '
        f'Try higher alpha, fewer clients, or lower min_pos_abs.'
    )

def find_feasible_params(df_raw, n_clients_list, alpha_list,
                         min_patients=500, min_pos_abs=200,
                         n_trials=500, seed=42):
    rng = np.random.default_rng(seed)

    patient_labels = (
        df_raw.groupby('patient_nbr')['readmitted']
        .apply(lambda x: 1 if (x == '<30').any() else 0)
        .reset_index()
    )
    pos_patients = patient_labels[patient_labels['readmitted'] == 1]['patient_nbr'].values
    neg_patients = patient_labels[patient_labels['readmitted'] == 0]['patient_nbr'].values

    print(f"Total patients: {len(patient_labels):,} "
          f"| Positive: {len(pos_patients):,} ({len(pos_patients)/len(patient_labels)*100:.1f}%)")
    print(f"\n{'N_clients':>10} | {'Alpha':>8} | {'Feasible trials':>16} | "
          f"{'Mean min pos_abs':>18} | {'p5 min pos_abs':>15}")
    print("-" * 75)

    for n in n_clients_list:
        for alpha in alpha_list:
            feasible_count = 0
            min_pos_abs_vals = []

            for _ in range(n_trials):
                prop_pos = rng.dirichlet([alpha] * n)
                prop_neg = rng.dirichlet([alpha] * n)

                counts_pos = (prop_pos * len(pos_patients)).astype(int)
                counts_neg = (prop_neg * len(neg_patients)).astype(int)
                client_total = counts_pos + counts_neg

                # Both constraints must pass — matching dirichlet_partition(filtered=True)
                size_ok = min(client_total) >= min_patients
                pos_ok  = min(counts_pos)   >= min_pos_abs

                if size_ok and pos_ok:
                    feasible_count += 1
                    min_pos_abs_vals.append(min(counts_pos))

            feasible_pct     = feasible_count / n_trials * 100
            mean_min_pos_abs = np.mean(min_pos_abs_vals) if min_pos_abs_vals else 0
            p5_min_pos_abs   = np.percentile(min_pos_abs_vals, 5) if min_pos_abs_vals else 0

            print(f"{n:>10} | {alpha:>8.3f} | "
                  f"{feasible_pct:>15.0f}% | "
                  f"{mean_min_pos_abs:>18.1f} | "
                  f"{p5_min_pos_abs:>15.1f}")


def create_clients_raw_csv(df_raw, output_dir, alpha_sweep,
                            filtered=True, seed=42, n_clients=N_CLIENTS):
    """
    Generate Dirichlet partitions and save raw_data.csv for each client.
    Returns partitions_by_alpha dict for downstream analysis.
    """
    partitions_by_alpha = {}

    for alpha in alpha_sweep:
        print(f'\n  α={alpha}: generating ({"filtered" if filtered else "unfiltered"})...')
        try:
            client_patient_ids = dirichlet_partition(
                df_raw, n_clients=n_clients, alpha=alpha,
                seed=seed, filtered=filtered
            )
        except ValueError as e:
            print(f'  α={alpha} → FAILED: {e}')
            continue

        manifest = {}
        for i, patient_ids in enumerate(client_patient_ids):
            client_dir = os.path.join(output_dir, f'alpha_{alpha}', f'client_{i}')
            os.makedirs(client_dir, exist_ok=True)
            client_df  = df_raw[df_raw['patient_nbr'].isin(patient_ids)].copy()
            client_df.to_csv(os.path.join(client_dir, 'raw_data.csv'), index=False)
            n_pos    = int((client_df['readmitted'] == '<30').sum())
            pos_rate = n_pos / len(client_df) * 100 if len(client_df) > 0 else 0
            manifest[f'client_{i}'] = {
                'n_patients'   : int(len(patient_ids)),
                'n_encounters' : int(len(client_df)),
                'pos_rate_pct' : round(pos_rate, 2),
                'n_pos'        : n_pos,
                'passes_filter': n_pos >= 200 and len(patient_ids) >= 500,
            }
            flag = '✓' if manifest[f'client_{i}']['passes_filter'] else '✗'
            print(f'    client_{i}: {len(patient_ids):,} patients | '
                  f'pos={pos_rate:.1f}% | n_pos={n_pos} {flag}')

        manifest['_meta'] = {
            'alpha': alpha, 'n_clients': n_clients,
            'seed': seed, 'filtered': filtered,
        }
        with open(os.path.join(output_dir, f'alpha_{alpha}', 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)

        partitions_by_alpha[alpha] = {i: ids for i, ids in enumerate(client_patient_ids)}

    return partitions_by_alpha
                 

def preprocess_and_save_client(client_dir, global_columns, seed=42,
                                test_size=0.2, val_size=0.15,
                                regenerate=False, verbose=False):
    """
    Preprocess one client's raw_data.csv and save train/val/test arrays.

    Skips silently if X_train.npy already exists and regenerate=False.
    """
    raw_path = os.path.join(client_dir, 'raw_data.csv')
    npy_path = os.path.join(client_dir, 'X_train.npy')

    if os.path.exists(npy_path) and not regenerate:
        return

    X, y, groups, _ = prepare_data_aligned(
        path=raw_path, global_columns=global_columns, verbose=verbose
    )
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, groups, test_size=test_size, val_size=val_size, seed=seed
    )
    X_train_s, X_val_s, X_test_s, scaler = scale_data(X_train, X_val, X_test)

    np.save(os.path.join(client_dir, 'X_train.npy'), X_train_s)
    np.save(os.path.join(client_dir, 'X_val.npy'),   X_val_s)
    np.save(os.path.join(client_dir, 'X_test.npy'),  X_test_s)
    np.save(os.path.join(client_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(client_dir, 'y_val.npy'),   y_val)
    np.save(os.path.join(client_dir, 'y_test.npy'),  y_test)
    joblib.dump(scaler, os.path.join(client_dir, 'scaler.joblib'))

    info = {
        'n_train'      : int(len(X_train)),
        'n_val'        : int(len(X_val)),
        'n_test'       : int(len(X_test)),
        'n_features'   : int(X_train_s.shape[1]),
        'positive_rate': float(round(float(y_train.mean()), 4)),
    }
    with open(os.path.join(client_dir, 'client_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f'  Pos rate={info["positive_rate"]:.3f} | '
          f'train={info["n_train"]:,} val={info["n_val"]:,} test={info["n_test"]:,}')
    
def preprocess_clients(output_dir, alpha_sweep, global_columns,
                       seed=42, n_clients=N_CLIENTS, regenerate=False):
    """
    Loop over all alphas and clients, calling preprocess_and_save_client for each.
    Only call this on the filtered partition — unfiltered is analysis-only.
    """
    for alpha in alpha_sweep:
        print(f'\n  α={alpha}:')
        for i in range(n_clients):
            client_dir = os.path.join(output_dir, f'alpha_{alpha}', f'client_{i}')
            print(f'    client_{i}: ', end='', flush=True)
            preprocess_and_save_client(
                client_dir=client_dir, global_columns=global_columns,
                seed=seed, regenerate=regenerate,
            )    


def load_clients(alpha, output_dir, n_clients=N_CLIENTS):
    """
    Load pre-generated client tensors for one alpha from disk.

    Parameters
    ----------
    alpha      : float — the Dirichlet alpha value
    output_dir : str   — path to federated_data/filtered (or /unfiltered)
    n_clients  : int

    Returns
    -------
    dict {int: {'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'}}
    """
    alpha_dir = os.path.join(output_dir, f'alpha_{alpha}')
    clients   = {}
    for i in range(n_clients):
        d = os.path.join(alpha_dir, f'client_{i}')
        clients[i] = {
            'X_train': torch.tensor(np.load(f'{d}/X_train.npy'), dtype=torch.float32),
            'y_train': torch.tensor(np.load(f'{d}/y_train.npy'), dtype=torch.long),
            'X_val':   torch.tensor(np.load(f'{d}/X_val.npy'),   dtype=torch.float32),
            'y_val':   torch.tensor(np.load(f'{d}/y_val.npy'),   dtype=torch.long),
            'X_test':  torch.tensor(np.load(f'{d}/X_test.npy'),  dtype=torch.float32),
            'y_test':  torch.tensor(np.load(f'{d}/y_test.npy'),  dtype=torch.long),
        }
        n_pos    = clients[i]['y_train'].sum().item()
        pos_rate = clients[i]['y_train'].float().mean().item() * 100
        print(f'  client_{i}: {len(clients[i]["y_train"]):,} train  '
              f'pos_rate={pos_rate:.1f}%  n_pos={int(n_pos)}')
        if n_pos == 0:
            raise ValueError(
                f'client_{i} at α={alpha} has zero positive training examples. '
                f'Regenerate the partition with a higher min_pos_abs.'
            )
    return clients

def load_partitions_from_disk(output_dir, alpha_sweep,
                               patient_labels_map, n_clients=N_CLIENTS):
    """
    Load patient ID lists from raw_data.csv for analysis purposes.
    Returns partitions_by_alpha dict {alpha: {client_id: [patient_ids]}}.
    This is different from load_clients which loads tensors for training.
    """
    partitions_by_alpha = {}

    for alpha in alpha_sweep:
        alpha_dir = os.path.join(output_dir, f'alpha_{alpha}')
        if not os.path.exists(alpha_dir):
            print(f'  α={alpha} → not found at {alpha_dir}, skipping.')
            continue

        partition = {}
        all_ok    = True
        for i in range(n_clients):
            raw_path = os.path.join(alpha_dir, f'client_{i}', 'raw_data.csv')
            if not os.path.exists(raw_path):
                print(f'  α={alpha} client_{i} → raw_data.csv missing.')
                all_ok = False
                break
            patient_ids = (pd.read_csv(raw_path, usecols=['patient_nbr'])
                           ['patient_nbr'].unique().tolist())
            partition[i] = patient_ids

        if not all_ok:
            continue

        sizes     = [len(v) for v in partition.values()]
        pos_rates = [
            sum(patient_labels_map.get(p, 0) for p in v) / len(v)
            for v in partition.values()
        ]
        print(f'  α={alpha} → {n_clients} clients | '
              f'sizes: {[f"{s:,}" for s in sizes]} | '
              f'pos rates: {[f"{r:.3f}" for r in pos_rates]}')

        partitions_by_alpha[alpha] = partition

    print(f'\nLoaded partitions for α ∈ {list(partitions_by_alpha.keys())}')
    return partitions_by_alpha


def verify_leakage(alpha_sweep, output_dir, n_clients=N_CLIENTS):
    """
    Assert zero patient overlap between any two clients for every alpha.
    Loads from disk so the check reflects what was actually saved.
    """
    for alpha in alpha_sweep:
        all_sets = []
        for i in range(n_clients):
            raw_path = os.path.join(output_dir, f'alpha_{alpha}',
                                    f'client_{i}', 'raw_data.csv')
            pids = set(
                pd.read_csv(raw_path, usecols=['patient_nbr'])['patient_nbr'].tolist()
            )
            all_sets.append(pids)
        for a in range(n_clients):
            for b in range(a + 1, n_clients):
                overlap = all_sets[a] & all_sets[b]
                assert len(overlap) == 0, (
                    f'LEAK α={alpha}: client_{a} and client_{b} '
                    f'share {len(overlap)} patients'
                )
        print(f'  α={alpha}: ✓ zero leakage confirmed')


def save_result(results_dir, alpha, seed, variant,
                test_auc, per_client, history, cumul_mb):
    """Save one experiment result to results_dir/alpha_{alpha}/seed_{seed}/{variant}.json."""
    result_dir = os.path.join(results_dir, f'alpha_{alpha}', f'seed_{seed}')
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, f'{variant}.json')
    with open(path, 'w') as f:
        json.dump({
            'test_auc'  : test_auc,
            'per_client': per_client,
            'history'   : history,
            'cumul_mb'  : cumul_mb,
        }, f, indent=2)
    print(f'  Saved → {path}  (test_auc={test_auc:.4f})')


# ═════════════════════════════════════════════════════════════════════════════
# MODEL CLASSES
# ═════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """
    Federated MLP with LayerNorm.
    Split into feature_extractor + predictor for partial sharing.
    """
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
    Lightweight generator G_ω: (y, ε) → ẑ ∈ ℝ^latent_dim.
    Noise ε ~ N(0, I) — Gaussian, zero-centered so class embedding cleanly
    determines class direction without positive-bias artefacts.
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


class GMMSampler:
    """
    Interpretable alternative to the neural Generator.
    Samples directly from per-class Gaussians N(μ_y, σ_y) estimated from
    client latent representations. No learned parameters — fully auditable.
    Updated each round from client latent distributions.
    """
    def __init__(self, latent_dim, device):
        self.latent_dim = latent_dim
        self.device     = device
        self.params = {
            cls: {
                'mean': torch.zeros(latent_dim, device=device),
                'std':  torch.ones(latent_dim,  device=device),
            }
            for cls in [0, 1]
        }

    def update(self, client_distributions, sample_counts):
        """Weighted average of per-client (mean, std) for each class."""
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
                torch.tensor(d['std'], dtype=torch.float32, device=self.device) * (n / total)
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

    def param_bytes(self):
        """Bytes to transmit mean + std per class (negligible overhead)."""
        return 2 * 2 * self.latent_dim * 4


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def model_bytes(model):
    """Parameter size of a model in bytes (float32)."""
    return sum(p.numel() for p in model.parameters()) * 4


def mb(n_bytes):
    return n_bytes / (1024 ** 2)


def make_criterion(y_numpy, device):
    """Class-weighted CE loss. Falls back to unweighted for single-class slices."""
    classes = np.unique(y_numpy)
    if len(classes) < 2:
        return nn.CrossEntropyLoss()
    cw = compute_class_weight('balanced', classes=classes, y=y_numpy)
    return nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))


def fed_avg(state_dicts, sample_counts):
    """Sample-size weighted average of state dicts."""
    total = sum(sample_counts)
    return {
        key: sum(sd[key] * (sample_counts[i] / total)
                 for i, sd in enumerate(state_dicts))
        for key in state_dicts[0].keys()
    }


def evaluate_global(model, clients, use_local_encoders=False, local_models=None):
    """
    Returns (global_auc, per_client_dict).
    Uses local encoders when feature extractor stays local (partial sharing).
    """
    model.eval()
    all_proba, all_y = [], []
    per_client = {}
    with torch.no_grad():
        for i, c in clients.items():
            enc   = local_models[i] if use_local_encoders else model
            z     = enc.encode(c['X_test'].to(device))
            proba = torch.softmax(model.predictor(z), dim=1)[:, 1].cpu().numpy()
            y_test = c['y_test'].numpy()
            all_proba.append(proba)
            all_y.append(y_test)
            if len(np.unique(y_test)) > 1:
                per_client[str(i)] = float(roc_auc_score(y_test, proba))
    return float(roc_auc_score(np.concatenate(all_y), np.concatenate(all_proba))), per_client


def compute_client_distribution(model, X_tensor, y_numpy, device):
    """Per-class (mean, std, n) of latent representations — sent to server."""
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


def _compute_local_prototypes(model, X_train, y_train, device):
    """
    Per-class centroids in the client's own latent space.
    Called at end of local training — reflects actual encoder state after adaptation.
    """
    model.eval()
    with torch.no_grad():
        Z_all = model.encode(X_train.to(device)).cpu().numpy()
    y_np = y_train.numpy()
    per_class_stats = {}
    for cls in [0, 1]:
        mask = y_np == cls
        if mask.sum() > 0:
            Z_cls = Z_all[mask]
            per_class_stats[cls] = {
                'mean': Z_cls.mean(axis=0),
                'std':  Z_cls.std(axis=0) + 1e-8,
                'n':    int(mask.sum()),
            }
    return per_class_stats


def _compute_local_medoid_proxy(model, X_train, y_train, device):
    """
    Per-class medoid proxy in the client's current latent space.

    For each class y, finds the training point z* = argmin_{z in Z_y} ||z - mean(Z_y)||.
    z* is a real patient encoding — guaranteed to lie in a populated region
    even on non-convex or horseshoe-shaped latent manifolds.

    Complexity: O(n × latent_dim) — one pass, no pairwise distances.

    Drop-in replacement for _compute_local_prototypes: same dict structure,
    same 'mean', 'std', 'n' keys. 'std' is still the class covariance (used
    by GMMSampler but not by update_generator's prototype constraint).
    """
    model.eval()
    with torch.no_grad():
        Z_all = model.encode(X_train.to(device)).cpu().numpy()
    y_np = y_train.numpy()
    stats = {}
    for cls in [0, 1]:
        mask = y_np == cls
        if mask.sum() > 0:
            Z_cls    = Z_all[mask]
            centroid = Z_cls.mean(axis=0)
            dists    = np.linalg.norm(Z_cls - centroid, axis=1)  # (n_cls,)
            medoid   = Z_cls[np.argmin(dists)]                   # real patient encoding
            stats[cls] = {
                'mean': medoid,                      # ← medoid replaces centroid
                'std':  Z_cls.std(axis=0) + 1e-8,   # class spread (unchanged)
                'n':    int(mask.sum()),
            }
    return stats


NUM_CLASSES = 2  # binary readmission task


def _label_counter(y_train, num_classes=NUM_CLASSES):
    """Per-class label counts for one client's training set (Algorithm 1, line 8)."""
    y_np = y_train.numpy() if isinstance(y_train, torch.Tensor) else np.asarray(y_train)
    counts = np.bincount(y_np.astype(np.int64), minlength=num_classes).astype(np.float64)
    return torch.tensor(counts, dtype=torch.float32)


def _aggregate_phat(c_list, eps=1e-8):
    """Aggregate client label counters into normalised p̂(y) (Algorithm 1, line 13)."""
    total = torch.stack(c_list).sum(dim=0) + eps
    return total / total.sum()


def _aggregate_prototypes(client_stats):
    """
    Weighted aggregation of per-client per-class centroids.
    Returns {cls: {'mean': array, 'std': array}} — used as generator constraint.
    """
    global_prototypes = {}
    for cls in [0, 1]:
        cls_means, cls_vars, cls_ns = [], [], []
        for stats in client_stats:
            if cls in stats:
                cls_means.append(stats[cls]['mean'])
                cls_vars.append(stats[cls]['std'] ** 2)
                cls_ns.append(stats[cls]['n'])
        if not cls_means:
            continue
        total_cls = sum(cls_ns)
        weights   = [n / total_cls for n in cls_ns]
        global_prototypes[cls] = {
            'mean': sum(w * m for w, m in zip(weights, cls_means)),
            'std':  np.sqrt(sum(w * v for w, v in zip(weights, cls_vars))) + 1e-8,
        }
    return global_prototypes


# ═════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER SEARCH
# ═════════════════════════════════════════════════════════════════════════════

def fl_objective(trial, min_client_size, input_dim,
                 X_tr, y_tr, X_va, y_va,
                 tune_epochs=50, tune_seed=42):
    """
    Optuna objective for single-client federated hyperparameter search.
    Tuned on one client to preserve the federated assumption.
    """
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    dropout    = trial.suggest_float('dropout', 0.1, 0.5)
    lr         = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    max_bs     = min(256, min_client_size)
    batch_size = trial.suggest_categorical(
        'batch_size', [b for b in [64, 128, 256] if b <= max_bs]
    )

    torch.manual_seed(tune_seed)
    model_t  = MLP(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    loader_t = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    cw = compute_class_weight('balanced',
                              classes=np.unique(y_tr.numpy()),
                              y=y_tr.numpy())
    criterion_t = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32).to(device)
    )
    optimizer_t = torch.optim.Adam(model_t.parameters(), lr=lr)

    best_auc = 0.0
    for epoch in range(tune_epochs):
        model_t.train()
        for xb, yb in loader_t:
            xb, yb = xb.to(device), yb.to(device)
            optimizer_t.zero_grad()
            criterion_t(model_t(xb), yb).backward()
            optimizer_t.step()

        model_t.eval()
        with torch.no_grad():
            vp = torch.softmax(model_t(X_va.to(device)), dim=1)[:, 1].cpu().numpy()
        if len(np.unique(y_va.numpy())) > 1:
            auc = roc_auc_score(y_va.numpy(), vp)
            best_auc = max(best_auc, auc)
            trial.report(auc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return best_auc


# ═════════════════════════════════════════════════════════════════════════════
# FEDAVG TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def local_train_fedavg_full(model, X_train, y_train, X_val, y_val,
                             epochs, batch_size, lr, device):
    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=batch_size, shuffle=True, drop_last=True)
    criterion = make_criterion(y_train.numpy(), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_auc = 0.0
    best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vp = torch.softmax(model(X_val.to(device)), dim=1)[:, 1].cpu().numpy()
        if len(np.unique(y_val.numpy())) > 1:
            auc = roc_auc_score(y_val.numpy(), vp)
            if auc > best_val_auc:
                best_val_auc = auc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    return best_state, len(X_train)


def local_train_fedavg_partial(model, X_train, y_train, X_val, y_val,
                                epochs, batch_size, lr, device):
    """Feature extractor stays local; returns predictor state dict only."""
    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=batch_size, shuffle=True, drop_last=True)
    criterion  = make_criterion(y_train.numpy(), device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_auc    = 0.0
    best_pred_state = {k: v.clone() for k, v in model.predictor.state_dict().items()}

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vp = torch.softmax(model(X_val.to(device)), dim=1)[:, 1].cpu().numpy()
        if len(np.unique(y_val.numpy())) > 1:
            auc = roc_auc_score(y_val.numpy(), vp)
            if auc > best_val_auc:
                best_val_auc    = auc
                best_pred_state = {k: v.clone()
                                   for k, v in model.predictor.state_dict().items()}

    return best_pred_state, len(X_train)


def run_fedavg_full(clients, input_dim, seed,
                    hidden_dim, dropout, lr, batch_size,
                    n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                    local_epochs=LOCAL_EPOCHS, patience=PATIENCE):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model        = MLP(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    n_params     = sum(p.numel() for p in model.parameters())
    bytes_round  = 2 * n_params * 4 * n_clients
    latent_dim   = hidden_dim // 4

    best_val_auc, best_state, no_improve = 0.0, None, 0
    history, cumul_mb, total_bytes = [], [], 0

    for fl_round in range(fl_rounds):
        state_dicts, counts = [], []
        for i in range(n_clients):
            local = MLP(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
            local.load_state_dict(model.state_dict())
            sd, n = local_train_fedavg_full(
                local, clients[i]['X_train'], clients[i]['y_train'],
                clients[i]['X_val'], clients[i]['y_val'],
                local_epochs, batch_size, lr, device
            )
            state_dicts.append(sd)
            counts.append(n)
        model.load_state_dict(fed_avg(state_dicts, counts))
        total_bytes += bytes_round

        model.eval()
        vp, vy, tp, ty = [], [], [], []
        with torch.no_grad():
            for i in range(n_clients):
                vp.append(torch.softmax(model(clients[i]['X_val'].to(device)),
                                        dim=1)[:,1].cpu().numpy())
                vy.append(clients[i]['y_val'].numpy())
                tp.append(torch.softmax(model(clients[i]['X_test'].to(device)),
                                        dim=1)[:,1].cpu().numpy())
                ty.append(clients[i]['y_test'].numpy())
        val_auc  = roc_auc_score(np.concatenate(vy), np.concatenate(vp))
        test_auc = roc_auc_score(np.concatenate(ty), np.concatenate(tp))
        history.append({'val': val_auc, 'test': test_auc})
        cumul_mb.append(total_bytes / (1024 ** 2))
        print(f'  [fedavg_full] R{fl_round+1:02d}: val={val_auc:.4f} '
              f'test={test_auc:.4f} cumul={cumul_mb[-1]:.2f}MB')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'  Early stopping at round {fl_round+1}.')
                break

    model.load_state_dict(best_state)
    global_auc, per_client = evaluate_global(model, clients)
    return global_auc, per_client, history, cumul_mb


def run_fedavg_partial(clients, input_dim, seed,
                       hidden_dim, dropout, lr, batch_size,
                       n_clients=N_CLIENTS, fl_rounds=FL_ROUNDS,
                       local_epochs=LOCAL_EPOCHS, patience=PATIENCE):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model        = MLP(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    local_models = {
        i: MLP(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
        for i in range(n_clients)
    }
    for i in range(n_clients):
        local_models[i].load_state_dict(model.state_dict())

    n_params_pred = sum(p.numel() for p in model.predictor.parameters())
    bytes_round   = 2 * n_params_pred * 4 * n_clients

    best_val_auc, best_pred_state, no_improve = 0.0, None, 0
    history, cumul_mb, total_bytes = [], [], 0

    for fl_round in range(fl_rounds):
        pred_states, counts = [], []
        for i in range(n_clients):
            local_models[i].predictor.load_state_dict(model.predictor.state_dict())
            sd, n = local_train_fedavg_partial(
                local_models[i], clients[i]['X_train'], clients[i]['y_train'],
                clients[i]['X_val'], clients[i]['y_val'],
                local_epochs, batch_size, lr, device
            )
            pred_states.append(sd)
            counts.append(n)
        model.predictor.load_state_dict(fed_avg(pred_states, counts))
        total_bytes += bytes_round

        model.eval()
        vp, vy, tp, ty = [], [], [], []
        with torch.no_grad():
            for i in range(n_clients):
                z_v = local_models[i].encode(clients[i]['X_val'].to(device))
                z_t = local_models[i].encode(clients[i]['X_test'].to(device))
                vp.append(torch.softmax(model.predictor(z_v), dim=1)[:,1].cpu().numpy())
                vy.append(clients[i]['y_val'].numpy())
                tp.append(torch.softmax(model.predictor(z_t), dim=1)[:,1].cpu().numpy())
                ty.append(clients[i]['y_test'].numpy())
        val_auc  = roc_auc_score(np.concatenate(vy), np.concatenate(vp))
        test_auc = roc_auc_score(np.concatenate(ty), np.concatenate(tp))
        history.append({'val': val_auc, 'test': test_auc})
        cumul_mb.append(total_bytes / (1024 ** 2))
        print(f'  [fedavg_partial] R{fl_round+1:02d}: val={val_auc:.4f} '
              f'test={test_auc:.4f} cumul={cumul_mb[-1]:.2f}MB')

        if val_auc > best_val_auc:
            best_val_auc    = val_auc
            best_pred_state = {k: v.clone()
                               for k, v in model.predictor.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'  Early stopping at round {fl_round+1}.')
                break

    model.predictor.load_state_dict(best_pred_state)
    global_auc, per_client = evaluate_global(model, clients,
                                             use_local_encoders=True,
                                             local_models=local_models)
    return global_auc, per_client, history, cumul_mb


# ═════════════════════════════════════════════════════════════════════════════
# FEDGEN — GENERATOR UPDATES
# ═════════════════════════════════════════════════════════════════════════════


def update_generator(generator, gr_states, global_prototypes, device,
                     p_hat_y=None, n_gen_samples=128,
                     noise_dim=NOISE_DIM, gen_steps=GEN_STEPS,
                     lambda_proto=LAMBDA_PROTO, gen_lr=GEN_LR,
                     logit_avg=False):
    """
    Server-side generator update (partial sharing — predictor state dicts).

    Loss = CE(ensemble(z), y) + λ · ‖z − z̄_y‖²

    Parameters
    ----------
    p_hat_y : Tensor or None
        If provided, labels are sampled from p̂(y) (paper-faithful).
        If None, falls back to balanced 50/50 sampling.
    logit_avg : bool
        If True, ensemble = softmax(mean(logits))    — Eq 4 paper-faithful.
        If False, ensemble = mean(softmax(logits))    — original codebase.
    """
    all_w = [sd['weight'].to(device) for sd in gr_states]
    all_b = [sd['bias'].to(device)   for sd in gr_states]

    proto_t = {
        cls: torch.tensor(v['mean'], dtype=torch.float32, device=device)
        for cls, v in global_prototypes.items()
    }

    opt = torch.optim.Adam(generator.parameters(), lr=gen_lr)
    generator.train()

    for _ in range(gen_steps):
        opt.zero_grad()
        if p_hat_y is not None:
            y_gen = torch.multinomial(p_hat_y.to(device), n_gen_samples,
                                      replacement=True)
        else:
            half = n_gen_samples // 2
            y_gen = torch.cat([
                torch.zeros(half, dtype=torch.long, device=device),
                torch.ones(n_gen_samples - half, dtype=torch.long, device=device),
            ])
        eps   = torch.randn(len(y_gen), noise_dim, device=device)
        Z_gen = generator(y_gen, eps)

        if logit_avg:
            # Eq 4 paper-faithful: σ(1/K Σ_k g(z; θ^p_k))
            mean_logits = torch.stack([
                Z_gen @ w.T + b for w, b in zip(all_w, all_b)
            ]).mean(0)
            probs = torch.softmax(mean_logits, dim=1)
        else:
            # Original codebase: 1/K Σ_k σ(g(z; θ^p_k))
            probs = torch.stack([
                torch.softmax(Z_gen @ w.T + b, dim=1)
                for w, b in zip(all_w, all_b)
            ]).mean(0)
        loss_ce = -(torch.log(probs[range(len(y_gen)), y_gen] + 1e-8)).mean()

        if lambda_proto > 0:
            proto_tgt = torch.stack([proto_t[int(yy.item())] for yy in y_gen])
            loss_p    = ((Z_gen - proto_tgt) ** 2).mean()
        else:
            loss_p = 0.0

        (loss_ce + lambda_proto * loss_p).backward()
        opt.step()

    generator.eval()


def update_generator_full(generator, full_model_states, global_prototypes, device,
                           p_hat_y=None, n_gen_samples=128,
                           noise_dim=NOISE_DIM, gen_steps=GEN_STEPS,
                           lambda_proto=LAMBDA_PROTO, gen_lr=GEN_LR,
                           logit_avg=False):
    """
    Server-side generator update (full sharing — extracts predictor from full state dicts).
    Same loss as update_generator. See that docstring for p_hat_y / logit_avg semantics.
    """
    all_w = [sd['predictor.weight'].to(device) for sd in full_model_states]
    all_b = [sd['predictor.bias'].to(device)   for sd in full_model_states]

    proto_t = {
        cls: torch.tensor(v['mean'], dtype=torch.float32, device=device)
        for cls, v in global_prototypes.items()
    }

    opt = torch.optim.Adam(generator.parameters(), lr=gen_lr)
    generator.train()

    for _ in range(gen_steps):
        opt.zero_grad()
        if p_hat_y is not None:
            y_gen = torch.multinomial(p_hat_y.to(device), n_gen_samples,
                                      replacement=True)
        else:
            half = n_gen_samples // 2
            y_gen = torch.cat([
                torch.zeros(half, dtype=torch.long, device=device),
                torch.ones(n_gen_samples - half, dtype=torch.long, device=device),
            ])
        eps   = torch.randn(len(y_gen), noise_dim, device=device)
        Z_gen = generator(y_gen, eps)

        if logit_avg:
            mean_logits = torch.stack([
                Z_gen @ w.T + b for w, b in zip(all_w, all_b)
            ]).mean(0)
            probs = torch.softmax(mean_logits, dim=1)
        else:
            probs = torch.stack([
                torch.softmax(Z_gen @ w.T + b, dim=1)
                for w, b in zip(all_w, all_b)
            ]).mean(0)
        loss_ce = -(torch.log(probs[range(len(y_gen)), y_gen] + 1e-8)).mean()

        if lambda_proto > 0:
            proto_tgt = torch.stack([proto_t[int(yy.item())] for yy in y_gen])
            loss_p    = ((Z_gen - proto_tgt) ** 2).mean()
        else:
            loss_p = 0.0

        (loss_ce + lambda_proto * loss_p).backward()
        opt.step()

    generator.eval()



# ═════════════════════════════════════════════════════════════════════════════
# ZHU CODE-FAITHFUL GENERATOR UPDATES
# ═════════════════════════════════════════════════════════════════════════════


def _diversity_loss(eps, gen_output):
    """
    Mode-seeking diversity loss (Mao et al., CVPR 2019).
    Encourages generator to produce diverse outputs for different noise vectors.
    L_div = -mean( ||G(z1) - G(z2)||_1 / (||z1 - z2||_1 + eps) )
    """
    n = len(eps)
    if n < 2:
        return torch.tensor(0.0, device=eps.device)
    idx = torch.randperm(n, device=eps.device)
    eps_perm    = eps[idx]
    out_perm    = gen_output[idx]
    dist_z      = (eps - eps_perm).abs().sum(dim=1) + 1e-8
    dist_output = (gen_output - out_perm).abs().sum(dim=1)
    return -(dist_output / dist_z).mean()


def update_generator_zhu_code(generator, gr_states, global_predictor_state,
                               label_counts, device,
                               p_hat_y=None, n_gen_samples=128,
                               noise_dim=NOISE_DIM, gen_steps=GEN_STEPS,
                               gen_lr=GEN_LR,
                               ensemble_alpha=1.0, ensemble_beta=1.0,
                               ensemble_eta=0.01):
    """
    Zhu's GitHub code-faithful generator update (partial sharing).

    Three-term loss:
        L = α·teacher_loss − β·student_loss + η·diversity_loss

    teacher_loss: weighted sum of per-user CEs
        Σ_k  w_k(y) · CE(softmax(z·W_k^T + b_k), y)
        where w_k(y) = n_k(y) / Σ_j n_j(y)

    student_loss (ADVERSARIAL — note the minus sign):
        KL( log_softmax(student(z)) || softmax(teacher_ensemble(z)) )
        Generator tries to MAXIMIZE this — finds knowledge gaps.

    diversity_loss: mode-seeking loss encouraging diverse outputs.

    Parameters
    ----------
    gr_states : list of state dicts — per-client predictor {weight, bias}
    global_predictor_state : state dict — aggregated global predictor (student)
    label_counts : list of tensors — per-client label counts [n_0, n_1]
    """
    K = len(gr_states)
    all_w = [sd['weight'].to(device) for sd in gr_states]
    all_b = [sd['bias'].to(device)   for sd in gr_states]

    # Student = aggregated global predictor
    student_w = global_predictor_state['weight'].to(device)
    student_b = global_predictor_state['bias'].to(device)

    # Per-label, per-user weights: w_k(y) = n_k(y) / Σ_j n_j(y)
    counts = torch.stack(label_counts).to(device)  # (K, num_classes)
    label_weight_matrix = counts / (counts.sum(dim=0, keepdim=True) + 1e-8)  # (K, num_classes)

    opt = torch.optim.Adam(generator.parameters(), lr=gen_lr)
    generator.train()

    for _ in range(gen_steps):
        opt.zero_grad()

        if p_hat_y is not None:
            y_gen = torch.multinomial(p_hat_y.to(device), n_gen_samples,
                                      replacement=True)
        else:
            half = n_gen_samples // 2
            y_gen = torch.cat([
                torch.zeros(half, dtype=torch.long, device=device),
                torch.ones(n_gen_samples - half, dtype=torch.long, device=device),
            ])
        eps   = torch.randn(len(y_gen), noise_dim, device=device)
        Z_gen = generator(y_gen, eps)

        # ── teacher_loss: weighted per-user CE ────────────────────────
        teacher_loss  = torch.tensor(0.0, device=device)
        teacher_logit = torch.zeros(len(y_gen), all_w[0].shape[0], device=device)

        for k in range(K):
            logits_k = Z_gen @ all_w[k].T + all_b[k]
            log_probs_k = F.log_softmax(logits_k, dim=1)
            w_k = label_weight_matrix[k][y_gen].unsqueeze(1)  # (B, 1)

            ce_k = F.nll_loss(log_probs_k, y_gen, reduction='none')  # (B,)
            teacher_loss += (ce_k * w_k.squeeze(1)).mean()
            teacher_logit += logits_k * w_k.expand_as(logits_k)

        # ── student_loss: KL (adversarial — will be subtracted) ──────
        student_logit = Z_gen.detach() @ student_w.T + student_b
        student_loss  = F.kl_div(
            F.log_softmax(student_logit, dim=1),
            F.softmax(teacher_logit.detach(), dim=1),
            reduction='batchmean',
        )

        # ── diversity_loss ────────────────────────────────────────────
        div_loss = _diversity_loss(eps, Z_gen)

        # ── combined loss (note MINUS on student) ─────────────────────
        loss = (ensemble_alpha * teacher_loss
                - ensemble_beta * student_loss
                + ensemble_eta * div_loss)

        loss.backward()
        opt.step()

    generator.eval()


def update_generator_zhu_code_full(generator, full_model_states, global_model_state,
                                    label_counts, device,
                                    p_hat_y=None, n_gen_samples=128,
                                    noise_dim=NOISE_DIM, gen_steps=GEN_STEPS,
                                    gen_lr=GEN_LR,
                                    ensemble_alpha=1.0, ensemble_beta=1.0,
                                    ensemble_eta=0.01):
    """
    Zhu's GitHub code-faithful generator update (full sharing).
    Same 3-term loss but extracts predictor weights from full model state dicts.
    """
    K = len(full_model_states)
    all_w = [sd['predictor.weight'].to(device) for sd in full_model_states]
    all_b = [sd['predictor.bias'].to(device)   for sd in full_model_states]

    student_w = global_model_state['predictor.weight'].to(device)
    student_b = global_model_state['predictor.bias'].to(device)

    counts = torch.stack(label_counts).to(device)
    label_weight_matrix = counts / (counts.sum(dim=0, keepdim=True) + 1e-8)

    opt = torch.optim.Adam(generator.parameters(), lr=gen_lr)
    generator.train()

    for _ in range(gen_steps):
        opt.zero_grad()

        if p_hat_y is not None:
            y_gen = torch.multinomial(p_hat_y.to(device), n_gen_samples,
                                      replacement=True)
        else:
            half = n_gen_samples // 2
            y_gen = torch.cat([
                torch.zeros(half, dtype=torch.long, device=device),
                torch.ones(n_gen_samples - half, dtype=torch.long, device=device),
            ])
        eps   = torch.randn(len(y_gen), noise_dim, device=device)
        Z_gen = generator(y_gen, eps)

        teacher_loss  = torch.tensor(0.0, device=device)
        teacher_logit = torch.zeros(len(y_gen), all_w[0].shape[0], device=device)

        for k in range(K):
            logits_k = Z_gen @ all_w[k].T + all_b[k]
            log_probs_k = F.log_softmax(logits_k, dim=1)
            w_k = label_weight_matrix[k][y_gen].unsqueeze(1)
            ce_k = F.nll_loss(log_probs_k, y_gen, reduction='none')
            teacher_loss += (ce_k * w_k.squeeze(1)).mean()
            teacher_logit += logits_k * w_k.expand_as(logits_k)

        student_logit = Z_gen.detach() @ student_w.T + student_b
        student_loss  = F.kl_div(
            F.log_softmax(student_logit, dim=1),
            F.softmax(teacher_logit.detach(), dim=1),
            reduction='batchmean',
        )

        div_loss = _diversity_loss(eps, Z_gen)

        loss = (ensemble_alpha * teacher_loss
                - ensemble_beta * student_loss
                + ensemble_eta * div_loss)

        loss.backward()
        opt.step()

    generator.eval()