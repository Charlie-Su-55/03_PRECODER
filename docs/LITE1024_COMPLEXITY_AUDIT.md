# Lite1024 UE complexity and separate rate-loss audit

## Status (2026-10-02)

Base/data commit: `f0dd125` (`Add formal Lite1024 evaluation data`).

- Original Table V **source-formula check passed**. This is not a completed
  model-instantiation/profile run.
- Actual K8 Original/Lite CPU profiling is **pending**: local PyTorch is
  `2.6.0+cpu`, but importing the existing model fails with
  `ModuleNotFoundError: No module named 'sionna.phy'` via
  `utils/__init__.py -> utils/channel_generator.py`. No dependencies were
  installed, imports stubbed, or model code altered.
- No `lite1024_complexity_tablev.csv` is published until actual Original
  profiling reproduces the paper. No Lite K8 numbers are inferred from K4.
- CSV-only K4 formal audit completed; six signed rate-loss rows are saved in
  `paper/data/lite1024_tradeoff_summary.csv`.
- No training, performance evaluation, checkpoint loading, plotting, latency
  profiling, or paper modification was performed.

## Table V provenance and scope

Reference: manuscript `tab:ue_complexity` and complexity subsection. Its
corresponding code is `eval_ue_complexity.py`, introduced in Git `5bbc0e5`.
The tracked profiler/model have no later content changes through the base
commit. The historical output
`runs_evaluation/ue_complexity/k8_n128_d128/ue_complexity.csv` is absent locally;
its exact historical execution cannot be verified here.

`count_ue_parameters.py` defaults to K16/Nsc256/Dtot256, so it is **not** the
Table V configuration. The old profiler's final console sanity message
`~10.871 M` is stale; its actual module scoping/shapes reproduce `10.756 M`.
Neither legacy script was edited.

The new `paper/scripts/profile_lite1024_complexity.py --profile` uses:

- K8, Nt32, Nsc128, Dtot128, Nsb4, D_MODEL256; maximum dimensions (16,128).
  Config construction uses the existing `k8_n128_soft_anchor` configuration
  only to obtain architecture attributes; it never starts training.
- One UE and one feedback instance, not a batch of eight UEs.
- Stored parameters: all unique parameters of `encoder` plus `cond_embed`,
  including inactive-branch storage. No BS decoder or RZF parameters.
- MACs: reuse `eval_ue_complexity.proposed_result`, its `MACCounter` and
  `unique_parameter_count` without changing those files. Linear projections,
  attention Q/K/V/output projections and QK/AV products are counted. The old
  convolution/VQ rules are unchanged but do not add operations to this UE.
- Bias additions, fixed transforms, normalization, activation, softmax, and
  elementwise attention-pooling weighting/reduction are excluded under the
  old rule. Pooling's learned MLP **is** counted. This is not total FLOPs.
- An inactive branch is bypassed. A nonzero branch computes its maximum-width
  head before prefix slicing. Condition embedding is evaluated once.
- Fresh CPU model instances only, no checkpoint reads or full-model forward.
  Synthetic encoder input is only for counting operations, not measuring rate.
- As in the old hook profiler, MHA fastpath is explicitly disabled **only for
  profiling**, with its previous setting restored in `finally`. This prevents
  fused Transformer execution bypassing hooks. No precision setting, physical
  noise convention, training or formal-evaluation backend is changed.

The Original must first match the paper's rounding (3 decimals in M parameters,
1 decimal in M MACs). On disagreement the script stops **before constructing
Lite**, and writes no complexity CSV. After success it constructs the actual
K8 `AdaptiveHybridPrecoderLite(..., ue_ffn_dim=1024)`, verifies the same state keys,
permits shape changes only in the UE FFNs, and checks all other initial state
values under the same seed. BS FFNs must remain 2048 and 512. No fallback,
parameter slicing, or K4-to-K8 scaling is used.

### Original source-formula cross-check (not measured locally)

| Quantity | Exact source-derived count | Paper rounding |
|---|---:|---:|
| Encoder parameters | 10,689,354 | 10.689 M |
| Condition parameters | 66,816 | 0.067 M |
| Total stored UE parameters | 10,756,170 | 10.756 M |
| Hybrid (4,96) MACs | 345,131,776 | 345.1 M |
| Pure FDMA (16,0) MACs | 174,146,304 | 174.1 M |
| Pure AS (0,128) MACs | 171,051,776 | 171.1 M |

For S=32 tokens, D=256 and F=2048, each Transformer layer contributes
`4*S*D^2 + 2*S^2*D + 2*S*D*F = 42,467,328` MACs; four layers contribute
169,869,312 per branch. FDMA input/pooling/head contribute
2,097,152 / 2,105,344 / 8,192; AS contributes 524,288 / 526,336 / 65,536.
The condition MLP contributes 66,304 once. These formulas reproduce the
published values but do not replace the required actual-instantiation check.

## Separate K4 formal performance evidence

Read only these four artifacts in
`paper/data/k4_pure_as_lite1024/formal_20261002T124602_541062Z/`:

1. `summary_k4_pure_as_lite1024.csv`
2. `paired_deltas_k4_pure_as_lite1024.csv`
3. `parameters_k4_pure_as_lite1024.csv`
4. `manifest.json`

Protocol checked against manifest and CSVs: K4/Nt32/Nsc256/Dtot256/Nsb4,
pure AS (0,256), DL25 dB, six FB SNRs, seed20261002, batch8 x 200 = 1600
complete multiuser channel samples per condition; formal held-out test, not
preflight. Full summary coverage is 24 conditions, full paired coverage is 12;
the internal output selects Original versus Lite only, preserving all six SNRs.
Saved selected steps are 50000/49500, while both completion steps are 50000.
These facts come from the synchronized artifacts, not a local checkpoint load.

| FB dB | Original bps/Hz | Lite bps/Hz | Paired Lite minus Original | Saved paired 95% CI |
|---:|---:|---:|---:|---:|
| 0 | 11.302495 | 11.306296 | +0.003801 | [-0.047593, +0.055195] |
| 5 | 16.098120 | 16.113148 | +0.015027 | [-0.044565, +0.074620] |
| 10 | 21.657120 | 21.704151 | +0.047031 | [-0.011917, +0.105979] |
| 15 | 26.042984 | 25.947960 | -0.095025 | [-0.141761, -0.048290] |
| 20 | 28.253941 | 28.102678 | -0.151260 | [-0.186075, -0.116445] |
| 25 | 29.172312 | 29.009785 | -0.162523 | [-0.192479, -0.132568] |

The table is a rounded display of generated CSV values. At 25 dB the relative
mean-rate change `(Lite_mean / Original_mean - 1)*100` is **-0.557128%**.
The 0/5/10 dB intervals include zero; this does not establish equivalence.
The 15/20/25 dB intervals are negative for these fixed trained checkpoints;
they do not characterize variability across training seeds.

The evaluator's summary uses float32 reductions; its paired CSV reduces
per-sample differences in Python double. Their mean differences differ by at
most 0.000003673435 bps/Hz for Original versus Lite. Both saved statistics are
retained, with separate `lite_minus_original_mean_difference` and `delta_mean`
columns. The script checks agreement within an explicit float32 reduction
bound and does not recreate paired SE from marginal SE. Paired signs and saved
SE/CIs are copied unchanged. The underlying per-sample pairing is not rerun in
this four-artifact audit.

CSV provenance columns explicitly use LF-normalized SHA-256 (`sha256_lf`) so
Windows/Linux Git line endings do not change the generated summary. Read-only
integrity checks still compare the actual input bytes before and after.

K4 saved encoder-plus-condition counts (10,877,090 / 6,674,594) and full-model
counts (20,209,115 / 16,006,619) are tagged K4 in the output and never used to
fill Table V. K8 structural complexity and K4 held-out performance are
**different experiments**. Parameter reduction is not latency reduction;
no MAC or speedup claim follows just from the K4 parameter CSV.

## Checks and remaining action

Executed locally:

- Safe fetch: main equals origin/main at `f0dd125`, with no staged/unpushed
  changes. Pre-existing plotting edit and four untracked figures preserved.
- Read-only source/history/formula audit; formal CSV/manifest checks and
  generation of the six-row internal summary; AST syntax and CLI help checks.
- `python -X utf8 -B -m unittest discover -s tests -p test_lite_complexity.py -v`:
  25 tests, 24 passed, 1 skipped (actual CPU model profile needs `sionna.phy`).
  Tests cover formal coverage, signed deltas, K8/K4 separation, rejection of
  preflight/duplicate/missing/non-finite/mismatched records, Original rounding
  gate fixtures, and non-overwriting output. Gate-fixture tests are not actual
  Original profile measurements.
- Real `--profile` attempted: stopped at missing `sionna.phy` with nonzero exit,
  before any model instantiation/profile or complexity CSV creation.
- No original/model/data/checkpoint or Overleaf edits.

The original source SHA-256 before and after is
`EFAAD80D29D238326BCDD8A7E8E4212705A00D7A2EDBA1BFF8BFB565A836AD25`;
Lite source is
`AF065F237572350D0C24282E74A18B31F4CD78EEA31B88B573E93DC7E36944C7`.
These are local byte hashes (Git checkout line endings may differ on Linux).

Remaining action: in the existing server environment, from the repository
root after safe synchronization, run the **CPU-only** profiler:

```bash
conda activate precoder
CUDA_VISIBLE_DEVICES="" python -B -m unittest discover -s tests -p 'test_lite_complexity.py' -v
CUDA_VISIBLE_DEVICES="" python -B paper/scripts/profile_lite1024_complexity.py --profile --tradeoff
```

The successful profiler creates `paper/data/lite1024_complexity_tablev.csv`
with exact counts and absolute/percentage reductions for all three allocations.
It only verifies (does not rewrite) an identical existing tradeoff CSV; a
differing output is refused. Return the profiler output and generated complexity
CSV for the remaining audit. Until that run, actual K8 Original/Lite counts,
shape/state equality and measured MAC reductions remain **unexecuted**. This
does not request any training, performance evaluation, or GPU profiling.
