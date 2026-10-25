# ASReceiver v1: fixed two-arm, two-seed screening batch

This is a new, isolated experiment, not a replacement of Full, Lite1024 or
ASDirect. No performance improvement is claimed before the server produces
the results. No existing manuscript, weights, results or plotting work is edited.

## Scope and architecture

`models/adaptive_hybrid_as_receiver.py` implements
`AdaptiveHybridPrecoderASReceiver`, architecture `as_receiver_v1`. The Full
forward is preserved except for one insertion after Stage-1 `w_init`. Full has
no intermediate injection API, so the local forward retains the original
operations; a focused AST check guards drift. No hook, monkeypatch or cached
transmit tensor is used to introduce the new branch.

The receiver uses only the same actual noisy `Y_ac`, receiver-available
`H_ul_est` (the original fallback to `H_ul` applies only when omitted), the
existing Stage-1 queries, allocation and feedback SNR. It does not access true
codewords, the downlink channel, noiseless received signals or hidden transmit
power. The physical AS slice is exactly `K*Df:K*Df+Da`.

For each active frequency it solves `(G^H G + lambda I) z = G^H y`, in
**complex64**, with `lambda=max(mean(abs(Y_ac)^2)/(1+gamma), 1e-6)`.
The received-power mean includes the full batch, antennas and active AS slots,
consistent with the scope of the existing received-SNR convention. This is an
approximate regularized recovery under a unit-codeword-variance assumption,
not a claim of optimal LMMSE. The clipped diagonal feature from
`lambda*solve(G^H G+lambda I,I)` is a model-based reliability proxy, not a
calibrated posterior error.

Per-user/per-subband real, imaginary and reliability tokens use the existing
subband-prefix order and learned codeword positions. Attention width is 128,
four heads, dropout zero. Existing Stage-1 angular positions provide queries.
The final 128-to-256 output weight and bias start at zero; no scalar gate is
added. `Da=0` bypasses the new branch entirely. Original UE, power handling,
AS context, five refinement layers and normalized RZF are unchanged.

The static parameter accounting is 165,248 additional BS parameters at this
fixed configuration. The controller records the **actual instantiated** count
and UE equality on the server; parameter count is not latency/MAC reduction.

`migrate_original_state_dict(model, original_state)` verifies all old keys,
shapes and dtypes, fills only the explicit eleven-key `as_receiver.*` whitelist,
requires a fresh zero output projection, and loads with `strict=True`. Normal
new-checkpoint restoration also uses `strict=True`, never migration or reset.
`receiver_enabled=False` is a nonpersistent inference intervention; it is not
a separately retrained ablation.

## Fixed training and evaluation

| Item | Fixed protocol |
| --- | --- |
| System | K4, Nt32, Nsc256, Dtot256, Nsb4; pure-AS `(0,256)`; maxima `(64,256)` |
| Arms | Full control / ASReceiver; all start from the same archived robust specialist |
| Training seeds | 20261030 and 20261031, reported separately |
| New updates | 20,000 per arm/seed, four runs total |
| Recipe | batch48, AdamW peak2e-5, weight decay1e-5, OneCycle warmup10%, clip0.5 |
| Objective | H_hat → normalized RZF → sum rate; all auxiliary weights zero |
| SNR | FB uniform0–25 dB, DL fixed25 dB |
| Validation | every500 updates, fixed64 channels, FB0/5/10/15/20/25, best by mean_all |
| Independent test | seed20261101; 200 batches×8 channels, six FB points, DL25 |
| Methods per training seed | trained receiver, same-budget Full, frozen robust, frozen Swin, receiver-off |

Both training arms explicitly declare the accepted `training-channel-data-v1`
protocol in `to_dict()`/fingerprint; its data identity excludes arm/model name.
Model initialization is seeded, and default RNG is reset after fresh migration
before optimizer construction to prevent new-parameter initialization from
shifting subsequent feedback/dropout streams. The new branch has zero dropout.
On resume, checkpoint RNG is restored, not reset to the fresh-run seed.

`baseline_evaluate.evaluate_scenario(..., channel_data=None)` gains an optional
source with a legacy-default branch. The new evaluator explicitly supplies
replayable data: each batch is shared across all five configurations and six
SNR points, and both fine-tuning seeds use the same independent test channels.
No historical evaluation is silently relabeled as reproducible.

All weights load strictly; original maximum widths, singleton running-power
index0, original noise convention and RZF1e-3 remain. Before/after checkpoint
hashes, buffers, modes and receiver switch are checked. ASReceiver-off strictly
loads the same complete trained state. No per-SNR/oracle checkpoint selection.

Per-sample paired differences and sample-index//8 batch-mean intervals are both
retained. Low-SNR and six-point averages are formed within the same complete
multi-user sample before calculating SE/CI. Each fine-tuning seed has its own
1600 samples and200 batch units; the two runs are **not** pooled into3200
independent observations. Swin has a different training budget. Common feedback
RNG does not imply equal absolute noise power or equal total UE energy.

Engineering continuation requires ASReceiver−Full low(0/5/10) mean≥0.5 bps/Hz,
batch-paired95%CI lower bound>0, and25dB batch-CI lower bound≥−0.1 bps/Hz.
Each seed's decision is reported separately, with no post-hoc selection or
additional training. This is not a journal acceptance criterion. Both positive
and negative outcomes, all SNR points and raw individual samples are retained.

## Controller and recovery

`run_as_receiver_batch.py --run` executes:

1. Exact authorized archive/cleanup targets only.
2. CUDA/Sionna2.0.1/plotting dependency and pinned source-weight checks.
3. New data-interface acceptance if no compatible successful evidence exists.
4. Focused new-model CUDA tests and real-source zero-projection equality checks.
5. Four trainings, each first500 updates then automatic strict resume to20000.
6. Two separately reported independent evaluations, including receiver-off.
7. CSV tables, vector PDF/PNG previews, Chinese/English internal draft paragraphs,
   engineering criteria summary and a small export bundle.

The500 updates are part of the20k schedule, not an extra run. The controller
never resets the scheduler after smoke. A nonfinite value, identity/config
mismatch, failed check or unknown existing output stops the batch and preserves
failure evidence. No automatic seed changes, favorable-result filtering or
extra training is implemented. Completed stages validate their evidence before
reuse. Only explicit `--run --resume` resumes a recorded batch. Unknown locks
remain; a dead-owner lock can be recovered only with verified identity on the
same POSIX host. Completed arms are not retrained.

Training outputs live under `runs_as_screen/proposal_as_receiver_v1/`;
evaluations under `runs_as_screen/evaluation/k4_as_receiver_v1/`. The controller
pointer is `runs_as_screen/proposal_as_receiver_v1/last_batch.json`.
`batch_summary.json` contains stage states, cleanup inventory/archive locations,
training completion/best steps and hashes, numerical checks, parameter counts,
per-seed paired results and six-point gaps to Swin. A small tarball contains the
summary and export files, **not** checkpoints, raw samples or channel caches.
Failure bundles have separate names and are never mistaken for final results.

The channel interface proves channel replay only. Full optimizer/scheduler/model
pause-resume bitwise equivalence is not claimed.

## Restricted cleanup and test consolidation

Only these four exact directories are eligible on the server:

- `runs_as_screen/diagnostics/as_direct_v1/preflight_20261025T112943_010217Z_9ea0286d/`
- `runs_as_screen/evaluation/k4_as_direct_v1/preflight_20261025T173028_269988Z_e8b82cda/`
- `terminal_logs/as_direct_cpu_OZ1waP0M/`
- `terminal_logs/as_direct_cuda_2yCQhEoi/`

Missing directories are skipped. Unknown file types, links, unexpected large
content, weights/caches, unconfirmed preflight status or formal sample counts
preserve the whole directory. An original-path/size/SHA256 inventory is written,
the tar.gz is verified member-by-member, and the unchanged exact original files
are unlinked only afterward. There is no wildcard deletion, rmtree, directory
move or checkpoint deletion. Archives retain the recoverable copies.

Local test consolidation:

- Three repeated fresh-process CLI import/write/weight guards in the ASDirect
  preflight, training and evaluation tests now share `tests/cli_contracts.py`.
  Each entry's specific metadata assertions remain.
- Two AST-only ordering assertions in `test_replayable_channel_data.py` were
  removed because actual extracted guard/loader invocation tests already verify
  early rejection. One comment-text assertion was removed; it tested wording,
  not runtime behavior.
- Numerical, physical-link, strict tensor/state, RNG, paired statistics and
  output-protection checks remain. No runtime helper file was deleted;
  `preflight_proposal_as_direct.py` and `diagnose_sionna_rng.py` remain dependencies.

## Execution and validation boundary

Local work runs only stdlib/synthetic checks and dry-runs. No weights are fetched,
no Sionna is installed/faked, no real channels/model evaluations/trainings or GPU
profiling are executed. CUDA checks, actual parameter counts, cleanup and the
experiment results are pending the explicit server batch command.

Actual local validation of this preparation: controller/cleanup/report contracts,
both-arm recipe/resume guards and evaluator pairing/integrity checks passed.
The corresponding suites contained13,31 and27 checks, respectively. Two static
model boundary checks passed; the numerical model test class was explicitly
skipped because the installed Sionna is0.15.1, not2.0.1. Consolidated ASDirect
CLI/metadata/statistics regressions passed; three old tensor-only guards were
skipped under the intentional standard-library `-S` run. The modified data
interface suite passed its metadata tests and skipped its runtime-availability
check under `-S`. Syntax and standard-library batch dry-run passed. None of
these statements is a claim that CUDA/source-checkpoint/model-gradient or real
channel-interface acceptance ran locally.

After safe server synchronization, in the existing experiment environment:

```bash
cd ~/project/03_PRECODER
python -B -S run_as_receiver_batch.py --dry-run
mkdir -p terminal_logs
asr_log="$(mktemp terminal_logs/as_receiver_batch_XXXXXXXX.log)"
nohup python -B -u run_as_receiver_batch.py --run > "$asr_log" 2>&1 < /dev/null &
printf 'PID=%s LOG=%s\n' "$!" "$asr_log"
```

The unique log name avoids replacing an earlier terminal transcript.
Read the authoritative latest summary without loading weights:

```bash
python -B -S run_as_receiver_batch.py --summary
```

After an interruption, use the same environment and unchanged code:
`python -B -u run_as_receiver_batch.py --run --resume`. Do not start a second
fresh batch. Return only the two paths listed in `return_files` in the summary;
failure summaries also identify the precise server log/report needed.
