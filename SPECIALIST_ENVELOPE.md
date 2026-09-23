# K4 allocation-specific specialist envelope

Run these commands on the GPU server, from `~/project/03_PRECODER`.
No model, baseline training logic, or existing checkpoint is changed by this workflow.

## 1. Inspect paths, then train the missing four specialists

```bash
cd ~/project/03_PRECODER
python run_as_screen.py train-specialists-k4 --dry-run
python -u run_as_screen.py train-specialists-k4
```

The dry run checks paths/status without importing torch or loading weights.
The batch requires the existing pure-AS `best.pth`; it NEVER trains `(0,256)`.
It checks all five runs, skips existing `best.pth` or `latest.pth`, and trains
missing `(64,0)`, `(48,64)`, `(32,128)`, `(16,192)` runs in that order.
Existing run directories without a checkpoint cause an error, not an overwrite.
Interrupted runs are preserved and not automatically resumed by this command;
the evaluator rejects a known incomplete training status.

Each new run calls the existing `main_train.train_proposal` with the same
`k4_n256_standard` recipe as the original pure-AS specialist: seed 42,
50,000 steps, unchanged optimizer/loss/SNR curriculum/CDL/validation settings,
and `mean_all` validation checkpoint selection (not selection at 25 dB only).
The training allocation grid has one point, but the architecture retains
`D_F_MAX_PER_USER=64`, `D_A_MAX_SHARED=256`, and four subbands. Unused branches
are not removed. Each run starts from scratch with the existing initializer.

All checkpoint paths are relative to `03_PRECODER`:

```text
runs_as_screen/proposal_specialists/checkpoints/hybrid_gnn_specialist_K4_Nsc256_Dfb256_f100_a000_df64_da0_sb4_k4_n256_standard_seed42/best.pth
runs_as_screen/proposal_specialists/checkpoints/hybrid_gnn_specialist_K4_Nsc256_Dfb256_f075_a025_df48_da64_sb4_k4_n256_standard_seed42/best.pth
runs_as_screen/proposal_specialists/checkpoints/hybrid_gnn_specialist_K4_Nsc256_Dfb256_f050_a050_df32_da128_sb4_k4_n256_standard_seed42/best.pth
runs_as_screen/proposal_specialists/checkpoints/hybrid_gnn_specialist_K4_Nsc256_Dfb256_f025_a075_df16_da192_sb4_k4_n256_standard_seed42/best.pth
```

Protected, already trained pure-AS checkpoint:

```text
runs_as_screen/proposal_specialists/checkpoints/hybrid_gnn_specialist_K4_Nsc256_Dfb256_f000_a100_df0_da256_sb4_k4_n256_standard_seed42/best.pth
```

## 2. After training, run the short GPU preflight

```bash
python -u run_as_screen.py preflight-specialists-k4
```

This loads all eight checkpoints with the existing strict loaders, then runs
one shared batch of eight samples and checks all 12 result rows. It does not
train anything. Its timestamped `_preflight_...` output is separate from formal
results. These eight-sample results are only a smoke check, not paper results.

Default universal and baseline checkpoints are resolved from current configs:

- Universal: `runs_deployzf_v1/checkpoints/.../best.pth` for K4/Nsc256/Dtot256.
- Swin: `runs_as_screen/baselines/checkpoints/swin_task_K4_Nsc256_Dfb256_Df64_sharedUE_seed42/best.pth`.
- CsiNet+: `runs_as_screen/baselines/checkpoints/csinet_task_K4_Nsc256_Dfb256_Df64_sharedUE_seed42/best.pth`.

If your existing checkpoints live elsewhere, pass `--proposal-root`,
`--baseline-root`, `--specialist-root`, or the explicit
`--universal-checkpoint`, `--swin-checkpoint`, `--csinet-checkpoint` options
to both preflight and formal evaluation. No checkpoint is automatically
retrained. Missing completion metadata is reported and recorded as `unknown`;
it does not prove that training completed. Keep `status.json` with copied runs.

## 3. Run formal evaluation; no further training

```bash
python -u run_as_screen.py eval-specialists-k4
```

The protocol is fixed: K=4, Nsc=256, Dtot=256, feedback SNR=25 dB,
downlink SNR=25 dB, seed=20260921, batch size=8, 200 batches, 1600 samples.
The helper calls `baseline_evaluate.evaluate_scenario` ONCE with all methods.
Each batch's exact same `H_dl` and `H_ul` tensors are used by every method.
The existing feedback-noise convention and reseeding are unchanged; identical
noise realizations across different architectures are not guaranteed when
random-call shapes/order differ.

There are 12 summary rows: five universal allocations, five separate
single-allocation specialists, Swin, and CsiNet+. Distinct `method_key` and
`method_type` values prevent result collisions. A specialist must have exactly
its singleton supported allocation grid; the universal must have the complete
five-point grid. Both retain full architecture dimensions and load through
`build_proposed_method(...): model.load_state_dict(..., strict=True)`.

Output directory:

```text
runs_as_screen/evaluation/k4_n256_d256_specialist_envelope/
```

- `summary_specialist_envelope.csv`: 12 rows, including `sum_rate_mean`,
  `sum_rate_se`, `nmse_db_mean`, allocation, `method_type`, `method_key`,
  `as_fraction`, checkpoint path/step, seed and sample count.
- `plot_specialist_envelope.csv`: five rows ordered by AS fraction
  0, 0.25, 0.5, 0.75, 1; universal/specialist rate, SE and raw NMSE columns,
  plus fixed Swin/CsiNet+ reference columns and specialist-minus-universal rate.
  Repeated baseline columns mean horizontal FDMA references, not new allocations.
- `samples_specialist_envelope.csv.gz`: per-sample results for paired analysis.
- `envelope_manifest.json`, `metadata_specialist_envelope.json` and
  `results_specialist_envelope.json`: provenance and protocol.

Before saving the summary, the helper checks all 12 method/allocation pairs,
the scenario/SNR/seed, finite metrics, and each pair's complete sample indices
0 through 1599. Existing result directories are refused. For an intentional
repeat, supply a NEW `--output-dir`; nothing is deleted automatically.

The existing NMSE calculation and CSV field names are unchanged. The printed
heading is `Raw representation NMSE (dB)`. This is a secondary reconstruction
diagnostic, not a cross-objective ranking criterion.

The specialist curve reports the one preselected fixed-allocation checkpoint
at each point, not a maximum selected on test data. Negative gaps are retained;
specialists are not assumed to beat the universal or Swin at every allocation.
The universal covers allocations only within this fixed system setting.
After these four runs, the remaining commands only evaluate existing weights.
