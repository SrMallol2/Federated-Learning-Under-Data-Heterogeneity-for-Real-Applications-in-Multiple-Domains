"""
UC2Utils.py — Utility module for UC2: Wi-Fi AP Load Prediction
Wraps the ndac-distributed-ml-architecture codebase for notebook usage.
"""
import sys, os, pickle, copy, time, json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════
# PATH SETUP — adjust LIB_DIR if your codebase is elsewhere
# ═══════════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
LIB_DIR   = os.path.join(BASE_DIR, "lib")
DATA_RAW  = os.path.join(BASE_DIR, "data", "raw")
DATA_PART = os.path.join(BASE_DIR, "data", "partitions")
RESULTS   = os.path.join(BASE_DIR, "results")

for d in [DATA_RAW, DATA_PART, RESULTS]:
    os.makedirs(d, exist_ok=True)

if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
# Paper defaults (Table I)
DEFAULT_CONFIG = dict(
    # Dataset
    dataset_pickle="pickle_2019-05-13-on7_10min.pkl",  # raw dataset filename
    n_users=20,
    train_ratio=100,     # 100 APs for training
    test_ratio=16,       # 16 APs for testing
    lookback=60,
    steps=1,
    random_seed=1111,
    # Model
    model="lstm",
    problem_type="regression",
    output_range=[1, 2],
    # Training
    batch_size=32,
    gen_batch_size=32,
    learning_rate=0.01,
    personal_learning_rate=0.01,
    ensemble_lr=1e-4,
    beta=1.0,
    lamda=1,
    mix_lambda=0.1,
    embedding=1,
    num_glob_iters=300,
    local_epochs=20,
    num_users=20,
    K=1,
    times=1,
    # Early stopping
    early_stopping="True",
    early_stopping_criteria="mae",
    early_stopping_patience=50,
    # Device
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    # Misc
    train=1,
    specified_mode=False,
    track_energy_consumption=False,
)

# Alpha sweep for experiments
ALPHA_SWEEP = [0.5, 1.0, 5.0, 10.0]

# Model sizes in MB (from the paper, Section IV-C)
MODEL_SIZE_MB = 3.7           # |θ| full LSTM model
MODEL_SHARED_SIZE_MB = 2e-3   # |gr(θ)| last layer only
GENERATOR_SIZE_MB = 0.113     # |ω| generative model


# ═══════════════════════════════════════════════════════════════
# ARGS NAMESPACE — feed into existing server classes
# ═══════════════════════════════════════════════════════════════
def make_args(algorithm, alpha, result_path=None, seed =0 ,**overrides,):
    """
    Create an args namespace compatible with the codebase's server classes.

    Parameters
    ----------
    algorithm : str
        One of 'Centralized', 'FedAvg', 'FedGen', 'Isolated'
    alpha : float
        Dirichlet alpha used during data generation
    result_path : str, optional
        Where to save model checkpoints and results
    **overrides : dict
        Override any default config key
    """
    cfg = {**DEFAULT_CONFIG, **overrides}

    # Dataset string expected by model_utils.get_data_dir
    dataset_str = f"FlagsRegression-user{cfg['n_users']}-alpha{alpha}-ratio1"

    # Dataset path: where the partitioned data lives
    dataset_path = os.path.join(
        DATA_PART,
        f"lookback_{cfg['lookback']}",
        f"steps_{cfg['steps']}"
    )

    # Result path
    if result_path is None:
        result_path = os.path.join(
            RESULTS, algorithm.lower(), f"alpha_{alpha}",
            cfg["model"], f"rep_{seed}"
        )
    # Use relative path so the lib never sees spaces in parent dirs
    result_path = os.path.relpath(result_path)
    os.makedirs(result_path, exist_ok=True)

    args = SimpleNamespace(
        dataset=dataset_str,
        dataset_path=dataset_path,
        model=cfg["model"],
        train=cfg["train"],
        algorithm=algorithm,
        problem_type=cfg["problem_type"],
        output_range=cfg["output_range"],
        steps=cfg["steps"],
        batch_size=cfg["batch_size"],
        gen_batch_size=cfg["gen_batch_size"],
        learning_rate=cfg["learning_rate"],
        personal_learning_rate=cfg["personal_learning_rate"],
        ensemble_lr=cfg["ensemble_lr"],
        beta=cfg["beta"],
        lamda=cfg["lamda"],
        mix_lambda=cfg["mix_lambda"],
        embedding=cfg["embedding"],
        num_glob_iters=cfg["num_glob_iters"],
        local_epochs=cfg["local_epochs"],
        num_users=cfg["num_users"],
        K=cfg["K"],
        times=cfg["times"],
        device=cfg["device"],
        result_path=result_path,
        specified_mode=cfg["specified_mode"],
        lookback=cfg["lookback"],
        early_stopping=cfg["early_stopping"],
        early_stopping_criteria=cfg["early_stopping_criteria"],
        early_stopping_patience=cfg["early_stopping_patience"],
        track_energy_consumption=cfg["track_energy_consumption"],
    )

    # Energy tracker (optional)
    if args.track_energy_consumption:
        import eco2ai
        args.tracker = eco2ai.Tracker(
            project_name=f"{algorithm}/{cfg['model']}/{cfg['lookback']}/{cfg['steps']}",
            experiment_description=f"Training {algorithm}",
            file_name=f"{result_path}/emission.csv",
            alpha_2_code="ES"  # Spain
        )

    return args


# ═══════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════
def get_raw_dataset_path():
    """Returns path to the raw pickle dataset. Raises if not found."""
    path = os.path.join(DATA_RAW, DEFAULT_CONFIG["dataset_pickle"])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw dataset not found at: {path}\n"
            f"Please place '{DEFAULT_CONFIG['dataset_pickle']}' in {DATA_RAW}/"
        )
    return path


def generate_partitions(alpha, lookback=60, steps=1, n_users=20, force=False):
    """
    Generate Dirichlet-partitioned data for a given alpha.
    Wraps data/FlagsRegression/generate_niid_dirichlet.py logic.

    Returns the destination path.
    """
    from data.FlagsRegression.dataset import DatasetGenerator
    from data.FlagsRegression.generate_niid_dirichlet import (
        rearrange_data_by_ap, divide_train_data, divide_test_data
    )

    dest_path = os.path.join(DATA_PART, f"lookback_{lookback}", f"steps_{steps}")
    partition_path = os.path.join(dest_path, f"u{n_users}-alpha{alpha}-ratio1")

    if os.path.exists(os.path.join(partition_path, "train", "train.pt")) and not force:
        print(f"[✓] Partition already exists for α={alpha} at {partition_path}")
        return dest_path

    print(f"[…] Generating partitions for α={alpha}, {n_users} users, "
          f"lookback={lookback}, steps={steps}")

    raw_path = get_raw_dataset_path()

    # Build dataset (same params as the paper)
    ds = DatasetGenerator(
        path=raw_path, split_by="AP ID",
        train_ratio=DEFAULT_CONFIG["train_ratio"],
        test_ratio=DEFAULT_CONFIG["test_ratio"],
        lookback=lookback, steps=steps,
        scaler_range=(1, 2),
        random_seed=DEFAULT_CONFIG["random_seed"]
    )

    train_data, train_target, train_ap = ds.getSplit("train")
    val_data, val_target, val_ap = ds.getSplit("validation")

    train_data_by_ap, train_target_by_ap = rearrange_data_by_ap(
        train_data, train_target, train_ap)
    val_data_by_ap, val_target_by_ap = rearrange_data_by_ap(
        val_data, val_target, val_ap)

    src_aps_train = np.unique(train_ap)
    src_aps_val = np.unique(val_ap)

    # Dirichlet partitioning of training data
    X, y, Labels, idx_batch, samples_per_user = divide_train_data(
        train_data_by_ap, train_target_by_ap,
        src_aps_train, n_users,
        min_sample=10, alpha=alpha, sampling_ratio=1.0
    )

    # Split test/validation data
    test_X, test_y = divide_test_data(
        n_users, src_aps_val, val_data_by_ap, val_target_by_ap,
        Labels, unknown_test=True
    )

    # Save train data
    _save_partition(partition_path, "train", X, y, n_users)
    _save_partition(partition_path, "test", test_X, test_y, n_users)

    # Save scalers
    with open(os.path.join(partition_path, "train_scaler.pkl"), "wb") as f:
        pickle.dump(ds.train_scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(partition_path, "test_scaler.pkl"), "wb") as f:
        pickle.dump(ds.test_scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[✓] Saved partitions for α={alpha}")
    print(f"    Samples per user: {samples_per_user}")
    return dest_path

def partitions_exist(alpha, lookback=60, steps=1, n_users=20):
    """Check if partition files already exist for the given parameters."""
    dest_path = os.path.join(DATA_PART, f"lookback_{lookback}", f"steps_{steps}")
    partition_path = os.path.join(dest_path, f"u{n_users}-alpha{alpha}-ratio1")
    train_file = os.path.join(partition_path, "train", "train.pt")
    test_file = os.path.join(partition_path, "test", "test.pt")
    return os.path.exists(train_file) and os.path.exists(test_file)


def _save_partition(partition_path, mode, X, y, n_users):
    """Save partitioned data in the .pt format expected by the codebase."""
    data_path = os.path.join(partition_path, mode)
    os.makedirs(data_path, exist_ok=True)

    dataset = {"users": [], "user_data": {}, "num_samples": []}
    for i in range(n_users):
        uname = f"f_{i:05d}"
        dataset["users"].append(uname)
        dataset["user_data"][uname] = {
            "x": torch.tensor(X[i], dtype=torch.float32),
            "y": torch.tensor(y[i], dtype=torch.float32),
        }
        dataset["num_samples"].append(len(X[i]))

    file_path = os.path.join(data_path, f"{mode}.pt")
    with open(file_path, "wb") as f:
        torch.save(dataset, f)
    print(f"    {mode}: {dataset['num_samples']}")


def generate_all_partitions(alphas=None, lookback=60, steps=1, force=False):
    """Generate partitions for all alpha values in the sweep."""
    if alphas is None:
        alphas = ALPHA_SWEEP
    for alpha in alphas:
        generate_partitions(alpha, lookback, steps, force=force)


# ═══════════════════════════════════════════════════════════════
# SERVER CREATION & TRAINING
# ═══════════════════════════════════════════════════════════════
def create_server(args):
    """
    Create the appropriate server object based on the algorithm.
    Returns: server instance
    """
    from utils.model_utils import create_model
    from FLAlgorithms.servers.serveravg import FedAvg
    from FLAlgorithms.servers.serverpFedGen import FedGen
    from FLAlgorithms.servers.serverCentralized import Centralized
    from FLAlgorithms.servers.serverIsolated import Isolated

    model = create_model(
        args.model, args.dataset, args.algorithm,
        args.problem_type, args.output_range, args.steps
    )
    model[0].to(args.device)

    if "Centralized" in args.algorithm:
        return Centralized(args, model, 0)
    elif "FedAvg" in args.algorithm:
        return FedAvg(args, model, 0)
    elif "FedGen" in args.algorithm:
        return FedGen(args, model, 0)
    elif "Isolated" in args.algorithm:
        return Isolated(args, model, 0)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")


def train_server(server, args):
    """Train the server and return training metrics."""
    print(f"\n{'='*60}")
    print(f"Training {args.algorithm} | α={_get_alpha(args)} | model={args.model}")
    print(f"{'='*60}\n")

    t0 = time.time()
    if args.track_energy_consumption:
        args.tracker.start()

    server.train(args)

    if args.track_energy_consumption:
        args.tracker.stop()

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s")
    return server.metrics


def evaluate_server(server):
    """
    Run final evaluation. Returns per-user metrics.
    """
    ids, num_samples, test_metrics, losses = server.test()
    return {
        "ids": ids,
        "num_samples": num_samples,
        "metrics": test_metrics,
        "losses": [l.cpu().detach().item() for l in losses],
    }


def result_exists(algorithm, alpha, model="lstm",seed =0):
    """Check whether full_results.pkl exists for a given (algorithm, alpha) combo."""
    path = os.path.join(
        RESULTS, algorithm.lower(), f"alpha_{alpha}", model, f"rep_{seed}",
        "full_results.pkl"
    )
    return os.path.exists(path)


def run_experiment(algorithm, alpha, seed=0, **overrides):
    """
    Full pipeline: create args → create server → train → evaluate → save.
    Returns (server, result).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    args = make_args(algorithm, alpha, seed=seed, **overrides)
    server = create_server(args)
    metrics = train_server(server, args)

    # Per-user evaluation
    per_user = evaluate_server(server)

    # Save comprehensive results
    result = {
        "algorithm": algorithm,
        "alpha": alpha,
        "model": args.model,
        "config": {k: v for k, v in vars(args).items()
                   if not k.startswith("_") and k != "tracker"},
        "metrics": metrics,
        "per_user": per_user,
        "n_rounds": len(metrics.get("glob_test_metric", [])),
    }

    result_file = os.path.join(args.result_path, "full_results.pkl")
    with open(result_file, "wb") as f:
        pickle.dump(result, f)
    print(f"[✓] Results saved to {result_file}")

    return server, result


# ═══════════════════════════════════════════════════════════════
# RESULTS LOADING & PROCESSING
# ═══════════════════════════════════════════════════════════════
def _get_alpha(args):
    """Extract alpha from the dataset string."""
    parts = args.dataset.lower().split("-")
    for p in parts:
        if p.startswith("alpha"):
            return p.replace("alpha", "")
    return "?"


def load_result(algorithm, alpha, model="lstm",seed =0):
    """Load a saved result for a specific (algorithm, alpha) combo."""
    path = os.path.join(
        RESULTS, algorithm.lower(), f"alpha_{alpha}", model, f"rep_{seed}",
        "full_results.pkl"
    )
    if not os.path.exists(path):
        print(f"[!] No results found at {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_all_results(algorithms=None, alphas=None, model="lstm"):
    """Load all available results into a dict keyed by (algorithm, alpha)."""
    if algorithms is None:
        algorithms = ["Centralized", "FedAvg", "FedGen"]
    if alphas is None:
        alphas = ALPHA_SWEEP

    results = {}
    for alg in algorithms:
        for alpha in alphas:
            r = load_result(alg, alpha, model)
            if r is not None:
                results[(alg, alpha)] = r
    return results


def extract_final_metrics(results):
    """
    From loaded results dict, extract final metrics per (algorithm, alpha).
    Returns a list of dicts for easy DataFrame construction.
    """
    rows = []
    for (alg, alpha), r in results.items():
        glob_metrics = r["metrics"].get("glob_test_metric", [])
        if not glob_metrics:
            continue
        # Get the best metric (lowest MAE)
        best_idx = np.argmin([m.get("unscaled_mae", float("inf"))
                              for m in glob_metrics])
        best = glob_metrics[best_idx]

        # Per-user equity
        per_user_metrics = r.get("per_user", {}).get("metrics", {})
        per_user_mae = per_user_metrics.get("unscaled_mae", [])

        rows.append({
            "algorithm": alg,
            "alpha": alpha,
            "best_round": best_idx,
            "n_rounds": r["n_rounds"],
            "mae": best.get("mae", None),
            "unscaled_mae": best.get("unscaled_mae", None),
            "mape": best.get("mape", None),
            "unscaled_mape": best.get("unscaled_mape", None),
            "per_user_mae_list": per_user_mae,
            "per_user_mae_std": np.std(per_user_mae) if per_user_mae else None,
            "per_user_mae_cv": (np.std(per_user_mae) / np.mean(per_user_mae)
                                if per_user_mae and np.mean(per_user_mae) > 0
                                else None),
        })
    return rows


# ═══════════════════════════════════════════════════════════════
# COMMUNICATION COST (Eq. 2 from the paper)
# ═══════════════════════════════════════════════════════════════
def comm_cost_fedavg(n_rounds, n_users_per_round, model_size_mb=MODEL_SIZE_MB):
    """C(FL) = 2|θ| * Σ|St|"""
    return 2 * model_size_mb * n_rounds * n_users_per_round


def comm_cost_fedgen(n_rounds, n_users_per_round,
                     shared_size_mb=MODEL_SHARED_SIZE_MB,
                     gen_size_mb=GENERATOR_SIZE_MB):
    """C(KD-gen) = (2|gr(θ)| + |ω|) * Σ|St|"""
    return (2 * shared_size_mb + gen_size_mb) * n_rounds * n_users_per_round


def comm_cost_fedgen_full(n_rounds, n_users_per_round,
                          model_size_mb=MODEL_SIZE_MB,
                          gen_size_mb=GENERATOR_SIZE_MB):
    """FedGen with full model exchange (non-partial mode)."""
    return (2 * model_size_mb + gen_size_mb) * n_rounds * n_users_per_round


def compute_comm_costs(results, n_users_per_round=20):
    """Compute communication cost for each result entry."""
    costs = {}
    for (alg, alpha), r in results.items():
        n_rounds = r["n_rounds"]
        if "Centralized" in alg:
            costs[(alg, alpha)] = 0  # data already local (or: dataset size)
        elif "FedAvg" in alg:
            costs[(alg, alpha)] = comm_cost_fedavg(n_rounds, n_users_per_round)
        elif "FedGen" in alg:
            # Check if partial mode
            if "partial" in alg.lower() or "pFed" in alg:
                costs[(alg, alpha)] = comm_cost_fedgen(
                    n_rounds, n_users_per_round)
            else:
                costs[(alg, alpha)] = comm_cost_fedgen_full(
                    n_rounds, n_users_per_round)
        elif "Isolated" in alg:
            costs[(alg, alpha)] = 0
    return costs


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════
COLORS = {
    "Centralized": "#2196F3",
    "FedAvg":      "#FF9800",
    "FedGen":      "#4CAF50",
    "FedGen_partial": "#8BC34A",
    "Isolated":    "#9E9E9E",
}
MARKERS = {
    "Centralized": "s",
    "FedAvg":      "o",
    "FedGen":      "D",
    "FedGen_partial": "d",
    "Isolated":    "x",
}


def setup_plot_style():
    """Set consistent matplotlib style."""
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_alpha_sensitivity(rows, metric="unscaled_mae", ylabel="Unscaled MAE (MB)"):
    """
    Plot metric vs alpha, one curve per algorithm.
    `rows` is the output of extract_final_metrics().
    """
    setup_plot_style()
    fig, ax = plt.subplots()

    # Group by algorithm
    by_alg = defaultdict(list)
    for r in rows:
        by_alg[r["algorithm"]].append(r)

    for alg, data in by_alg.items():
        data = sorted(data, key=lambda x: x["alpha"])
        alphas = [d["alpha"] for d in data]
        vals = [d[metric] for d in data]
        ax.plot(alphas, vals,
                marker=MARKERS.get(alg, "o"),
                color=COLORS.get(alg, "#333"),
                label=alg, linewidth=2, markersize=8)

    ax.set_xlabel("Dirichlet α (higher = more IID)")
    ax.set_ylabel(ylabel)
    ax.set_title("Algorithm Performance vs Data Heterogeneity")
    ax.legend()
    ax.set_xscale("log")
    plt.tight_layout()
    return fig, ax


def plot_pareto_frontier(rows, costs, metric="unscaled_mae",
                         ylabel="Unscaled MAE (MB)"):
    """
    Communication cost (x) vs accuracy (y) Pareto plot.
    Each point is one (algorithm, alpha) combination.
    """
    setup_plot_style()
    fig, ax = plt.subplots()

    for r in rows:
        key = (r["algorithm"], r["alpha"])
        if key not in costs:
            continue
        c = costs[key]
        v = r[metric]
        alg = r["algorithm"]
        ax.scatter(c, v,
                   marker=MARKERS.get(alg, "o"),
                   color=COLORS.get(alg, "#333"),
                   s=100, zorder=5)
        ax.annotate(f"α={r['alpha']}", (c, v),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, alpha=0.7)

    # Legend (one entry per algorithm)
    seen = set()
    for r in rows:
        alg = r["algorithm"]
        if alg not in seen:
            ax.scatter([], [],
                       marker=MARKERS.get(alg, "o"),
                       color=COLORS.get(alg, "#333"),
                       label=alg, s=100)
            seen.add(alg)

    ax.set_xlabel("Communication Cost (MB)")
    ax.set_ylabel(ylabel)
    ax.set_title("Communication–Accuracy Pareto Frontier")
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_per_client_equity(rows, metric_key="per_user_mae_list",
                           ylabel="Unscaled MAE per deployment"):
    """
    Box plot of per-client MAE for each (algorithm, alpha).
    """
    setup_plot_style()
    n_alphas = len(set(r["alpha"] for r in rows))
    algorithms = sorted(set(r["algorithm"] for r in rows))
    alphas = sorted(set(r["alpha"] for r in rows))

    fig, axes = plt.subplots(1, n_alphas, figsize=(5 * n_alphas, 6),
                             sharey=True)
    if n_alphas == 1:
        axes = [axes]

    for i, alpha in enumerate(alphas):
        ax = axes[i]
        data_to_plot = []
        labels = []
        colors_list = []
        for alg in algorithms:
            matching = [r for r in rows
                        if r["algorithm"] == alg and r["alpha"] == alpha]
            if matching and matching[0][metric_key]:
                data_to_plot.append(matching[0][metric_key])
                labels.append(alg)
                colors_list.append(COLORS.get(alg, "#333"))

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(f"α = {alpha}")
        ax.set_ylabel(ylabel if i == 0 else "")

    fig.suptitle("Per-Deployment Performance Equity", fontsize=14)
    plt.tight_layout()
    return fig, axes


def plot_training_curves(results, metric="unscaled_mae",
                         ylabel="Unscaled MAE (MB)"):
    """Plot training metric over rounds for multiple results."""
    setup_plot_style()
    fig, ax = plt.subplots()

    for (alg, alpha), r in sorted(results.items()):
        glob_metrics = r["metrics"].get("glob_test_metric", [])
        if not glob_metrics:
            continue
        vals = [m.get(metric, None) for m in glob_metrics]
        vals = [v for v in vals if v is not None]
        ax.plot(vals,
                color=COLORS.get(alg, "#333"),
                label=f"{alg} (α={alpha})",
                alpha=0.8)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel(ylabel)
    ax.set_title("Training Convergence")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig, ax