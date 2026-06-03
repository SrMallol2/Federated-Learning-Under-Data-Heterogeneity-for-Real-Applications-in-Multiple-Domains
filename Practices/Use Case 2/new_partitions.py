"""
new_partitions.py — Label-skew partition generator for UC2 (TFG).

WHY THIS EXISTS
---------------
The original partitioner (generate_niid_dirichlet.py) applies the Dirichlet
distribution over `AP ID`. On this dataset that produces almost no
heterogeneity in the *target* distribution (per-user y-mean Gini ~= 0.03 even
at alpha=0.01), so FedAvg is insensitive to alpha. See notebook 02 cell 19.

This module instead bins the regression target `y` into quantiles and applies
the Dirichlet over those bins. At low alpha, clients receive skewed slices of
the load distribution (some mostly-low-load, some mostly-high-load) -> genuine
LABEL skew that FedAvg cannot average away.

It reuses the existing DatasetGenerator (global scaler, unchanged) and writes
the IDENTICAL .pt save format, so every downstream loader (FedAvg, FedGen,
Centralized) works without modification. Only the partition folder changes.

OUTPUT LAYOUT
-------------
<DATA_PART>/new_partitions/lookback_<L>/steps_<S>/u<N>-alpha<A>-ratio1/
    train/train.pt
    test/test.pt
    train_scaler.pkl
    test_scaler.pkl
    partition_meta.json     <- bin edges, seed, per-user bin histogram
"""
import os
import json
import pickle

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Core: quantile-bin the target, Dirichlet-partition over bins
# ---------------------------------------------------------------------------
def _quantile_bins(all_y, n_bins):
    """Return a per-sample bin id (0..n_bins-1) from global quantiles of y."""
    edges = np.quantile(all_y, np.linspace(0, 1, n_bins + 1))
    # widen the outer edges so min/max land inside
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    bin_id = np.digitize(all_y, edges) - 1
    bin_id = np.clip(bin_id, 0, n_bins - 1)
    return bin_id, edges


def divide_train_data_by_ybin(data_by_ap, target_by_ap, src_aps, n_users,
                              alpha, n_bins=10, sampling_ratio=1.0,
                              min_sample=10, seed=42, max_tries=50,
                              floor_frac=0.0):
    """
    Dirichlet partition over QUANTILE BINS of the target instead of over AP ID.

    Parameters mirror the original divide_train_data so the rest of the
    pipeline is unchanged. Returns:
        X, y          : lists (len n_users) of per-user windows / targets
        Labels        : list of sets — here the set of y-bins each user holds
        samples_per_user, bin_hist (per-user bin counts), edges
    """
    # 1) Flatten every window into a global pool. Keep AP ids (which may be
    #    strings like '7-1002') SEPARATE from integer local indices, so the
    #    index array stays purely numeric.
    flat_ap = []            # AP id per global sample (object dtype, any type)
    flat_local = []         # local index within that AP (int)
    all_y = []
    for ap in src_aps:
        ys = target_by_ap[ap]
        for i in range(len(ys)):
            flat_ap.append(ap)
            flat_local.append(i)
            all_y.append(float(np.asarray(ys[i]).reshape(-1)[0]))
    all_y = np.asarray(all_y, dtype=np.float64)
    flat_ap = np.asarray(flat_ap, dtype=object)
    flat_local = np.asarray(flat_local, dtype=np.int64)
    n_sample = len(all_y)

    bin_id, edges = _quantile_bins(all_y, n_bins)
    by_bin = {b: np.where(bin_id == b)[0] for b in range(n_bins)}

    rng = np.random.RandomState(seed)
    max_per_user = sampling_ratio * n_sample / n_users

    # 2) Redraw until the smallest user has >= min_sample windows
    for attempt in range(max_tries):
        samples_per_user = [0] * n_users
        user_global_idx = [[] for _ in range(n_users)]
        for b in range(n_bins):
            idx = by_bin[b].copy()
            rng.shuffle(idx)
            if sampling_ratio < 1:
                idx = idx[:int(sampling_ratio * len(idx))]

            # --- uniform floor: hand each user a small equal share first ---
            # guarantees no client starves at extreme alpha while leaving the
            # bulk of the bin to be Dirichlet-skewed. floor_frac=0 -> pure Dirichlet.
            floor_assigned = [[] for _ in range(n_users)]
            if floor_frac > 0 and len(idx) >= n_users:
                n_floor = int(floor_frac * len(idx))
                per = n_floor // n_users
                if per > 0:
                    floor_pool = idx[:per * n_users]
                    idx = idx[per * n_users:]
                    for u in range(n_users):
                        floor_assigned[u] = floor_pool[u * per:(u + 1) * per]

            prop = rng.dirichlet(np.repeat(alpha, n_users))
            # cap users that already hit the size ceiling (quantity balance off
            # by default at ratio=1 because max_per_user == mean size; the cap
            # only prevents pathological single-user hoarding)
            prop = np.array([p * (s < max_per_user)
                             for p, s in zip(prop, samples_per_user)])
            if prop.sum() == 0:
                prop = np.ones(n_users)
            prop = prop / prop.sum()
            cut = (np.cumsum(prop) * len(idx)).astype(int)[:-1]
            for u, part in enumerate(np.split(idx, cut)):
                part = part.astype(np.int64)          # guard: np.split float edge-case
                part = np.concatenate([np.asarray(floor_assigned[u], dtype=np.int64),
                                       part]) if len(floor_assigned[u]) else part
                if len(part) == 0:
                    continue
                user_global_idx[u].extend(part.tolist())
                samples_per_user[u] += len(part)
        if min(samples_per_user) >= min_sample:
            break
    else:
        raise RuntimeError(
            f"Could not give every user >= {min_sample} samples after "
            f"{max_tries} tries at alpha={alpha}. Lower n_bins or raise alpha.")

    # 3) Materialise per-user X / y from the global references (numpy-efficient)
    # Pre-stack the per-AP arrays once so we can index without python-list blowup.
    ap_offsets = {}
    stack_X_parts, stack_y_parts = [], []
    running = 0
    for ap in src_aps:
        arr = np.asarray(data_by_ap[ap], dtype=np.float32)
        ty = np.asarray(target_by_ap[ap], dtype=np.float32)
        ap_offsets[ap] = running
        stack_X_parts.append(arr)
        stack_y_parts.append(ty)
        running += len(arr)
    big_X = np.concatenate(stack_X_parts, axis=0)
    big_y = np.concatenate(stack_y_parts, axis=0)
    del stack_X_parts, stack_y_parts

    # map each global pool index -> row in big_X
    # global index gi -> (flat_ap[gi], flat_local[gi]) -> ap_offset + local
    def _big_row(gi):
        return ap_offsets[flat_ap[gi]] + int(flat_local[gi])

    X = [None] * n_users
    y = [None] * n_users
    Labels = [set() for _ in range(n_users)]
    bin_hist = [np.zeros(n_bins, dtype=int) for _ in range(n_users)]
    for u in range(n_users):
        rows = np.fromiter((_big_row(gi) for gi in user_global_idx[u]),
                           dtype=np.int64, count=len(user_global_idx[u]))
        X[u] = big_X[rows]
        y[u] = big_y[rows]
        ubins = bin_id[np.asarray(user_global_idx[u], dtype=np.int64)]
        for b in range(n_bins):
            c = int(np.sum(ubins == b))
            bin_hist[u][b] = c
            if c > 0:
                Labels[u].add(int(b))
    return X, y, Labels, samples_per_user, [h.tolist() for h in bin_hist], edges.tolist()


def divide_test_data_global(n_users, val_data_by_ap, val_target_by_ap,
                            src_aps_val, max_test_per_user=3000, seed=42):
    """
    Build a per-user test set. For label-skew experiments the fair, comparable
    choice is to give EVERY user the SAME global validation pool (capped),
    so test performance reflects the global objective and per-user differences
    come only from the trained model, not from skewed test sets.
    """
    rng = np.random.RandomState(seed)
    all_X = np.concatenate([np.asarray(val_data_by_ap[ap], dtype=np.float32)
                            for ap in src_aps_val], axis=0)
    all_y = np.concatenate([np.asarray(val_target_by_ap[ap], dtype=np.float32)
                            for ap in src_aps_val], axis=0)
    if len(all_X) > max_test_per_user:
        sel = rng.choice(len(all_X), max_test_per_user, replace=False)
        all_X, all_y = all_X[sel], all_y[sel]
    test_X = [all_X for _ in range(n_users)]
    test_y = [all_y for _ in range(n_users)]
    return test_X, test_y


def _assign_test_indices_to_bins(all_y, edges, n_bins):
    """Bin the TEST targets using the TRAIN quantile edges, so bin b means the
    same load regime in train and test. Returns {bin: array_of_test_indices}."""
    bin_id = np.digitize(all_y, edges) - 1
    bin_id = np.clip(bin_id, 0, n_bins - 1)
    return {b: np.where(bin_id == b)[0] for b in range(n_bins)}, bin_id


def divide_test_data_by_ybin(n_users, val_data_by_ap, val_target_by_ap,
                             src_aps_val, train_bin_hist, edges, n_bins=10,
                             min_test_per_user=30, seed=43):
    """
    OPTION A — per-client test sets that MIRROR each client's TRAIN skew.

    `train_bin_hist` is the per-user bin histogram returned by
    divide_train_data_by_ybin (list[n_users] of list[n_bins] counts).
    `edges` are the TRAIN quantile edges, so a test sample is assigned to the
    same regime label it would have in training.

    Each client draws test samples from each bin in proportion to its TRAIN
    bin mass. Sampling is WITH replacement per bin (test pools per bin are far
    smaller than train), so a narrow-regime client gets a narrow-regime test
    set. A small uniform floor (min_test_per_user) prevents empty test sets for
    clients whose dominant train-bin happens to be sparse in validation.
    """
    rng = np.random.RandomState(seed)
    all_X = np.concatenate([np.asarray(val_data_by_ap[ap], dtype=np.float32)
                            for ap in src_aps_val], axis=0)
    all_y = np.concatenate([np.asarray(val_target_by_ap[ap], dtype=np.float32)
                            for ap in src_aps_val], axis=0)
    all_y_flat = all_y.reshape(len(all_y), -1)[:, 0].astype(np.float64)

    by_bin, _ = _assign_test_indices_to_bins(all_y_flat, np.asarray(edges), n_bins)
    nonempty_bins = [b for b in range(n_bins) if len(by_bin[b]) > 0]

    train_bin_hist = np.asarray(train_bin_hist, dtype=np.float64)  # (n_users, n_bins)
    test_X, test_y = [], []
    for u in range(n_users):
        # target test size for this client, proportional to its train size,
        # but capped so the file stays small and clients stay comparable in N.
        n_u_train = train_bin_hist[u].sum()
        n_target = int(np.clip(n_u_train * 0.2, min_test_per_user, 3000))

        w = train_bin_hist[u].copy()
        # zero out bins that have no validation samples, renormalise
        for b in range(n_bins):
            if b not in by_bin or len(by_bin[b]) == 0:
                w[b] = 0.0
        if w.sum() == 0:                      # client's regime absent in val:
            w = np.ones(n_bins)               # fall back to uniform over
            for b in range(n_bins):           # whatever bins exist
                if len(by_bin.get(b, [])) == 0:
                    w[b] = 0.0
        w = w / w.sum()

        # how many test samples to draw from each bin
        per_bin = np.random.multinomial(n_target, w) if False else \
            (w * n_target).astype(int)
        # fix rounding so the totals land near n_target
        deficit = n_target - per_bin.sum()
        if deficit > 0:
            order = np.argsort(-w)
            for b in order[:deficit]:
                per_bin[b] += 1

        rows = []
        for b in range(n_bins):
            k = int(per_bin[b])
            if k <= 0 or len(by_bin.get(b, [])) == 0:
                continue
            pick = rng.choice(by_bin[b], size=k,
                              replace=len(by_bin[b]) < k)
            rows.extend(pick.tolist())
        if len(rows) == 0:                    # last-resort guard
            rows = rng.choice(np.concatenate([by_bin[b] for b in nonempty_bins]),
                              size=min_test_per_user, replace=True).tolist()
        rows = np.asarray(rows, dtype=np.int64)
        test_X.append(all_X[rows])
        test_y.append(all_y[rows])
    return test_X, test_y


def _save_partition(partition_path, mode, X, y, n_users):
    """Identical format to UC2Utils._save_partition / the lib loader."""
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
    with open(os.path.join(data_path, f"{mode}.pt"), "wb") as f:
        torch.save(dataset, f)
    return dataset["num_samples"]


def generate_labelskew_partitions(uc2, alpha, lookback=60, steps=1, n_users=20,
                                  n_bins=10, seed=42, force=False,
                                  test_mode="client_local", floor_frac=0.02,
                                  subdir=None):
    """
    Top-level entry. `uc2` is the imported UC2Utils module (for paths + the
    existing DatasetGenerator). Writes into DATA_PART/<subdir>/...

    test_mode:
        "client_local" (Option A) — each client tested on its OWN train regime.
                                     Heterogeneity shows in per-client spread.
        "global"       (Option B / 3rd partition) — every client tested on the
                                     SAME global pool. Measures the global
                                     objective; pair with centralized gap.

    subdir: output root folder name under DATA_PART. Defaults to
        "new_partitions"        for test_mode="client_local"
        "new_partitions_global" for test_mode="global"
    so the two test designs never overwrite each other and make_args can point
    at whichever you want to train on.

    Returns the dest_path root (what make_args' dataset_path should point to).
    """
    from data.FlagsRegression.dataset import DatasetGenerator
    from data.FlagsRegression.generate_niid_dirichlet import rearrange_data_by_ap

    if subdir is None:
        subdir = "new_partitions" if test_mode == "client_local" \
            else "new_partitions_global"

    new_root = os.path.join(uc2.DATA_PART, subdir,
                            f"lookback_{lookback}", f"steps_{steps}")
    part_path = os.path.join(new_root, f"u{n_users}-alpha{alpha}-ratio1")
    train_pt = os.path.join(part_path, "train", "train.pt")
    if os.path.exists(train_pt) and not force:
        print(f"[OK] label-skew partition exists for alpha={alpha} "
              f"[{test_mode}]: {part_path}")
        return new_root

    print(f"[..] Generating LABEL-SKEW partition: alpha={alpha}, bins={n_bins}, "
          f"users={n_users}, test_mode={test_mode}")
    raw_path = uc2.get_raw_dataset_path()
    ds = DatasetGenerator(
        path=raw_path, split_by="AP ID",
        train_ratio=uc2.DEFAULT_CONFIG["train_ratio"],
        test_ratio=uc2.DEFAULT_CONFIG["test_ratio"],
        lookback=lookback, steps=steps, scaler_range=(1, 2),
        random_seed=uc2.DEFAULT_CONFIG["random_seed"],
    )
    tr_x, tr_y, tr_ap = ds.getSplit("train")
    va_x, va_y, va_ap = ds.getSplit("validation")
    tr_x_by, tr_y_by = rearrange_data_by_ap(tr_x, tr_y, tr_ap)
    va_x_by, va_y_by = rearrange_data_by_ap(va_x, va_y, va_ap)
    src_tr = np.unique(tr_ap)
    src_va = np.unique(va_ap)

    X, y, Labels, spu, bin_hist, edges = divide_train_data_by_ybin(
        tr_x_by, tr_y_by, src_tr, n_users,
        alpha=alpha, n_bins=n_bins, sampling_ratio=1.0, seed=seed,
        floor_frac=floor_frac)

    if test_mode == "client_local":
        test_X, test_y = divide_test_data_by_ybin(
            n_users, va_x_by, va_y_by, src_va,
            train_bin_hist=bin_hist, edges=edges, n_bins=n_bins, seed=seed + 1)
    elif test_mode == "global":
        test_X, test_y = divide_test_data_global(
            n_users, va_x_by, va_y_by, src_va, seed=seed)
    else:
        raise ValueError(f"unknown test_mode={test_mode!r}")

    os.makedirs(part_path, exist_ok=True)
    n_tr = _save_partition(part_path, "train", X, y, n_users)
    n_te = _save_partition(part_path, "test", test_X, test_y, n_users)

    with open(os.path.join(part_path, "train_scaler.pkl"), "wb") as f:
        pickle.dump(ds.train_scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(part_path, "test_scaler.pkl"), "wb") as f:
        pickle.dump(ds.test_scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(part_path, "partition_meta.json"), "w") as f:
        json.dump({"alpha": alpha, "n_bins": n_bins, "seed": seed,
                   "test_mode": test_mode, "floor_frac": floor_frac,
                   "bin_edges": edges, "samples_per_user": spu,
                   "bin_histogram_per_user": bin_hist}, f, indent=2)

    print(f"[OK] train per-user: {n_tr}")
    if test_mode == "client_local":
        print(f"[OK] test  per-user (client-local): {n_te}")
    else:
        print(f"[OK] test  per-user: {n_te[0]} (shared global pool x{n_users})")
    return new_root