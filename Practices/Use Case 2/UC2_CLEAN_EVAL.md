# UC2 Clean Evaluation — audit follow-up (June 2026)

This documents the post-audit fixes applied to UC2 **without retraining** the
existing models, plus one small set of new sanity runs. Use this as the source
for the Results chapter caveats.

## What was wrong

1. **Test-set contamination.** The per-user test sets (both `client_local` and
   `global` protocols) were drawn from `DatasetGenerator`'s *validation* split:
   a random 80/20 split of stride-1 overlapping windows over the **same 100 APs
   and the same time range as training**. Measured: **96.8%** of test windows
   had a training window shifted by ±1 timestep (59/60 identical input rows).
   The evaluation therefore measured interpolation, not generalization.
2. **Test-set model selection.** Early stopping and checkpoint saving in
   `serveravg.py` / `serverpFedGen.py` read `glob_test_metric` — there was no
   validation split in the FL loop.
3. A temporal-tail holdout cannot fix (1) for the existing checkpoints,
   because training windows were sampled over each AP's **entire** timeline.

## The fix (no retraining)

- **Clean test set = the 16 held-out test APs** that `DatasetGenerator`
  reserved (`test_ratio=16`) but the FL pipeline never used. Zero windows from
  these APs appear in any training set. Scaled with the saved **train** scaler
  (also fixing the original codebase's separate-test-scaler leak).
- Degenerate **dead windows** (constant zero-load; bitwise-identical across
  unrelated APs) with an exact or ±1-shifted twin in the train pool were
  dropped: 709 of 14,232 windows (5%). Verified contamination after filtering:
  **0.00%** on all seven test files.
- Both protocols rebuilt with the original machinery (`divide_test_data_by_ybin`
  with the saved `bin_edges` / per-user train histograms, seed 43; global pool
  capped at 3000, seed 42). Files: `data/partitions/new_partitions/lookback_60/steps_1/clean_test/`.
- **Saved checkpoints re-evaluated** on the clean sets (`uc2_reeval.py`,
  114 runs). The old test set is hereby demoted to a **validation set**: it
  selected the checkpoint (early stopping on MAE), and the clean set is seen
  exactly once. State this explicitly in the thesis.

Scripts: `uc2_clean_test.py`, `uc2_reeval.py`.
Outputs: `uc2_clean_results.csv` (per rep), `uc2_clean_results_agg.csv`
(mean/std over reps), `uc2_clean_results_detail.pkl` (per-user metrics).
Load from notebooks via `uc2_honest_table.clean_dataframe()`.

## Results on the clean test set (client_local, scaled MSE, mean over reps)

| α    | centralized | fedavg | fedavg-partial | fedgen | fedgen-partial |
|------|-------------|--------|----------------|--------|----------------|
| 0.01 | 0.158*      | 0.164  | 0.059          | 0.060  | 0.043          |
| 0.1  | 0.032       | 0.217  | 0.100          | 0.044  | 0.055          |
| 0.5  | 0.033       | 0.221  | 0.157          | 0.041  | 0.048          |
| 1.0  | 0.033       | 0.220  | 0.144          | 0.039  | 0.057          |
| 5.0  | 0.033       | 0.222  | 0.190          | 0.042  | 0.042          |
| 10.0 | 0.033       | 0.222  | 0.199          | 0.041  | 0.041          |

- **The ranking survives.** Old-vs-clean inflation ratio is within ±6% of 1.0
  for every FL cell: the window contamination, while methodologically real,
  was not driving the comparative results. FedGen ≈ near-centralized;
  SGD-FedAvg ~5× worse — now on a provably uncontaminated test set.
- (*) Exception: **centralized at α=0.01** degrades 0.032 → 0.158 on clean
  data. The α=0.01 client-local test sets concentrate on extreme load bins;
  on seen-AP data the model interpolated those windows, on unseen APs it does
  not generalize there. Note this in the discussion; do not headline the
  α=0.01 centralized cell.
- The le-sweep ordering also reproduces on clean data (fedgen-le10 ≪ fedavg-le10).
- FedAvg numbers correspond to the MAE-selected checkpoint (≈ final round);
  the old `best_mse` column for FedAvg was the untrained init and must not be
  used for comparison.

## FedAvg optimizer sanity check (the one set of NEW runs)

Question: is FedAvg's stall (MSE never beating init, MAE ≈ constant-median)
a property of federated averaging here, or of the inherited optimizer config
(plain SGD, lr 0.01 — validated in prior work under a different partitioning
and on the insensitive unscaled metric)?

Runs (`uc2_fedavg_sanity.py`, same partitions, same local budget of 20
batch-steps/round): `fedavg-adam` (Adam lr 0.01 wd 1e-2 — the Centralized
baseline's exact optimizer; enabled by a one-line fix in `serveravg.py`
passing `use_adam=self.use_adam`) and `fedavg-lr0.05` (SGD, 5× lr), at
α ∈ {1.0, 0.1}, seed 0, 300 rounds.

**RESULT (clean test set, client_local):**

| run                | clean MSE | clean MAE | vs fedgen (clean)      |
|--------------------|-----------|-----------|------------------------|
| fedavg-adam α=1.0  | **0.0344**| 0.0806    | better (0.0394/0.0941) |
| fedavg-adam α=0.1  | 0.0479    | 0.1185    | slightly worse (0.0442/0.1068) |

FedAvg-Adam converged in 300 rounds (α=1.0; early-stopped at 189 rounds for
α=0.1) under the SAME local budget, reaching near-centralized error
(centralized: 0.0330). **The SGD stall was an optimizer artifact, not an FL
property.** A one-line optimizer change makes plain FedAvg match — at α=1.0,
slightly beat — FedGen.

### Implication for the thesis claim (the second outcome occurred)

The claim "FedGen ≫ FedAvg" must be reframed. Supported claims:

1. Under the configuration inherited from prior work (plain SGD, lr 0.01),
   FedAvg fails to learn input dependence at every α — including near-IID
   α=10, which rules out heterogeneity as the cause — while FedGen converges
   to near-centralized error. FedGen's generator losses act as an effective
   optimization aid in this constrained regime.
2. With the optimizer the centralized baseline always used (Adam, one-line
   change), plain FedAvg matches FedGen (α=1.0: 0.034 vs 0.039; α=0.1:
   0.048 vs 0.044). There is **no evidence of absolute FedGen superiority
   over a tuned FedAvg** on this task.
3. Corollary: the prior work's FedAvg-vs-FedGen comparison — performed under
   this SGD config and reported on the insensitive unscaled metric — likely
   overstated FedGen's advantage. This is a legitimate critical-reproduction
   contribution of the thesis.

Caveats for these sanity runs: single seed, α ∈ {1.0, 0.1} only; FedGen was
not re-run with Adam (a symmetric FedGen-adam run would complete the picture
and works without code changes — serverpFedGen already forwards use_adam).
fedavg-lr0.05 (SGD, 5× lr) results pending; secondary to the Adam result.

## Remaining caveats to state in the thesis

1. Old test set = validation (model selection); clean held-out-AP set = test
   (reported numbers). Single evaluation, no test-set selection.
2. One Dirichlet partition per α; reps re-seed model init only.
3. Clean test measures generalization to **new APs** in the same time period
   (the realistic "new deployment" FL scenario), not forecasting future
   periods of known APs — temporal extrapolation would require retraining
   with a temporally split train set.
4. Centralized uses Adam + full-epoch rounds (~100× the FL gradient steps):
   frame it as an upper bound, not a budget-matched baseline.
5. n=1 rep for centralized and fedavg-partial; n=3 for the headline methods.
