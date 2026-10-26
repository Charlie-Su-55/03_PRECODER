# ASSpatial v1: isolated spatial-decoder screening

This batch prepares a new BS architecture hypothesis. It does not import any
08 project decoder, alter 08, replace the historical models, or claim a measured
performance improvement. Existing controls are evaluated on new test channels,
not retrained. No manuscript is changed.

## Architecture and information boundary

`AdaptiveHybridPrecoderASSpatial` in `models/adaptive_hybrid_as_spatial.py` is an
independent `nn.Module`. It directly reuses the original `AdaptiveHybridEncoder`,
`ConditionEmbed`, physical `transmit`/noise and `zf_solver` implementations. Full's
Stage1, shared context, refinement, channel-estimation head and residual head are
not constructed: there is no hidden old prediction shortcut.

The first version accepts only K4/Nt32/Nsc256/Dtot256/Nsb4, allocation `(0,256)`,
cfg_idx0, singleton running power and original architectural maxima `(64,256)`.
It preserves the full UE storage, including the inactive FDMA branch.

The receiver calls the existing `regularized_recovery` with the actual noisy
AS observation and receiver-available `H_ul_est` (the original perfect-CSI
fallback uses `H_ul` when the estimate is omitted). Recovery retains complex64,
lambda floor1e-6 and the existing approximate receiver power estimate. It is
not a claim of optimal LMMSE or a calibrated posterior covariance.

Per user, the 256 ordered complex soft symbols and reliability proxies become
256 ordered `(Re, Im, r)` triples, followed by SNR/25: 769 real inputs. A learned
`Linear(769,8192)` explicitly maps these codeword positions to an 8x8 grid of
128-wide spatial patch tokens. It does not identify arbitrary codeword indices
with physical angle/delay coordinates. Four **existing** `SwinTransformerBlock`
instances use width128, four heads, window4 and MLP ratio4. These particular
existing blocks do not shift windows; no additional shift or decoder from
another project is claimed. Patch4 unembedding yields two real/imaginary
32-delay-by-32-angle planes. Zero-padding to256 delay bins and the existing
inverse angular-delay transform produce `H_hat`, followed by original
normalized RZF. All new BS weights start randomly, with no imported Full or
Swin BS weights.

Only recovered symbols, reliability and SNR enter the spatial decoder. The
explicit `forward_with_diagnostics` API returns detached true transmitted
symbols **after** prediction, for offline soft-codeword error statistics.
The normal representation API does not expose truth. No persistent truth cache,
forward hook, extra physical transmission or second diagnostic forward is used.

Static parameter accounting (to be checked against real instantiation on CUDA):

| Component | Parameters |
| --- | ---: |
| UE encoder plus condition, original K4 configuration | 10,877,090 |
| Explicit spatial expansion | 6,307,840 |
| Four blocks, final norm and patch output | 801,568 |
| New BS total | 7,109,408 |
| Complete ASSpatial model | 17,986,498 |

The frozen arm trains only the new BS; the joint arm also enables UE parameters.
Inactive FDMA parameters remain stored but need not receive gradients. Counts
are not MACs or latency, and this is not an equal-parameter comparison. Actual
counts for all evaluated methods are exported by the server.

## Two arms and fixed data protocol

| Setting | Protocol |
| --- | --- |
| Arms | `frozen_ue`, `joint` |
| Fine-tuning seeds | 20261030 and20261031, matching the existing controls |
| Source | Same SHA256-pinned archived robust specialist; import UE/condition/running power only |
| Initialization | Same source UE and same randomly initialized BS between same-seed arms |
| Budget | 20,000 new optimizer updates per arm/seed; four new runs |
| Batch/optimizer | batch48, AdamW, weight decay1e-5, clip0.5 |
| Peak learning rates | BS1e-4; joint UE2e-5; real separate optimizer groups |
| Schedule | OneCycle20k, warmup10%; first500 updates included in20k |
| Objective | Original H_hat -> normalized RZF -> sum rate, all auxiliary weights zero |
| Training SNR | FB uniform0–25dB, DL25dB |
| Validation | Fixed64 channels, six FB points0/5/10/15/20/25, every500 updates, best by mean_all |
| Independent test | New seed20261103, 200 batches of8, 1600 multi-user samples per condition |
| Optional evaluator preflight | Separate seed20261104, one batch, not a formal result |

Frozen UE means both parameters and normalization statistics are frozen:
`encoder`/`cond_embed` stay in eval mode even when the BS trains, and source
`running_pwr[0]` supplies the normalization scale without EMA updates. Joint
training retains the original batch-power/EMA behavior. Consequently UE training
modes differ by design. Both arms share addressed training channels, fixed
validation channels and the SNR protocol; do not claim that different training
modes consume identical dropout/noise streams at every operation.

The accepted replayable-channel protocol is explicitly included in the new
config fingerprint. No old config fingerprint changes. New `main_train`
`optimizer_scheduler_factory=None` leaves the original optimizer/scheduler
branch intact; ASSpatial opts into separate LR groups after its model factory
initializes the UE. There is no copied trainer, global monkeypatch or parameters
created during forward. Resume strictly loads the complete new model, optimizer
groups, scheduler and RNG, without re-importing UE weights or resetting the BS.
Bitwise equivalence of complete interrupted and uninterrupted training is not
claimed.

## Independent evaluation and interpretation

Each fine-tuning seed evaluates `frozen_ue`, `joint`, the existing same-seed
Full+20k and ASReceiver+20k, and the pinned Swin checkpoint. The controller checks
that those old controls completed before starting any new training; a missing
control stops the batch, never triggers automatic retraining. All five methods
share each actual channel batch across all six feedback SNRs. Common feedback
RNG does not mean equal absolute noise power or equal UE total energy.

The evaluator preserves all30 conditions, sample records, negative gains, raw
representation NMSE and power diagnostics. Power is the full-band sum, nominal
256, not per-subcarrier1. Seven prespecified pairwise comparisons include each
new arm minus Full, ASReceiver and Swin, plus joint minus frozen. Each has all
six individual points, within-sample low(0/5/10) mean, and all-six mean. Both
sample-level and sample_index//8 batch-level SE/95%CI are saved. Cross-SNR
averages occur before cross-sample statistics; the two fine-tuning seeds are
reported separately, never pooled into3200 independent channels.

For **each arm and each seed**, the engineering continuation criterion is a
low-SNR gain over the same-seed Full of at least0.5bps/Hz, batch CI lower>0, and
25dB batch CI lower>=-0.1bps/Hz. This is not a publication standard, an
equal-parameter comparison, or a promise of superiority. A failed screen cannot
establish a UE performance ceiling because the new BS started randomly and has
only the prescribed20k budget. No automatic extra training, favorable seed
selection or removal of low-SNR points is implemented.

Soft-codeword MSE, linear/dB NMSE and mean reliability are recorded during the
same evaluation forward for the new arms. Their sample- and batch-level
statistics remain secondary diagnostics. True transmitted symbols never become
decoder inputs, and reliability is not relabeled as measured post-detection SNR.

## One server execution and recovery

`run_as_spatial_batch.py` reuses existing locks, stage evidence, subprocess logs,
output-path protection, metadata/loading, deterministic data and paired
statistics helpers. It does not rerun completed old model/RNG test suites, old
training experiments, or the previous cleanup. It performs necessary environment
and old-control checks, focused new-model CUDA checks, four500+19500 training
runs, two independent evaluations, CSV tables, vector PDF/PNG previews and
Chinese/English internal result drafts.

Training/controller root: `runs_as_screen/proposal_as_spatial_v1/`.
Evaluation root: `runs_as_screen/evaluation/k4_as_spatial_v1/`.
Existing directories cannot be overwritten. Resumption validates source hashes
and evidence before skipping completed stages; unknown outputs and active locks
stop safely. Successful results, weights and individual samples remain on the
server. Return only `batch_summary.json` and the small bundle listed in its
`return_files`; the bundle includes relevant reports/terminal logs and exported
tables/plots/drafts, not checkpoint or per-sample archives. Failures also produce
a summary and evidence bundle. Every package has a unique name, is written to
a temporary archive, verified member-by-member, then atomically published;
interrupted archives are preserved but never mistaken for completed packages.

After safe Git synchronization, from the existing experiment environment:

```bash
cd ~/project/03_PRECODER
python -B -S run_as_spatial_batch.py --dry-run
mkdir -p terminal_logs
ass_log="$(mktemp terminal_logs/as_spatial_batch_XXXXXXXX.log)"
nohup python -B -u run_as_spatial_batch.py --run >"$ass_log" 2>&1 </dev/null &
printf 'PID=%s LOG=%s\n' "$!" "$ass_log"
```

Read summary: `python -B -S run_as_spatial_batch.py --summary`.
After interruption: `python -B -u run_as_spatial_batch.py --run --resume`.
Do not start another fresh batch to replace a failed or negative one.

## Change boundary and validation record

New: model, isolated config/train/evaluate/batch/result scripts, four focused
test files and this document. Minimal opt-in extensions only in `main_train.py`,
`baseline_evaluate.py` and generic functions of `evaluate_proposal_as_receiver.py`;
their historical defaults are retained. No original model, old config, checkpoint,
historical CSV, manuscript, plotting work or 08 file is changed.

Local validation is lightweight only. Four static model-boundary checks passed;
the numerical model test class was explicitly skipped because local Sionna is
0.15.1, not the required genuine2.0.1. No weights were downloaded, dependencies
installed/faked, or local model training/evaluation performed. CUDA physical-path,
gradient, strict-state, freezing and instantiated-count checks remain server
requirements; the explicit CUDA test mode treats missing dependencies as failure,
not skip. CLI, configuration/resume, paired statistics and controller checks are
passed locally, including affected old-helper regressions. Standalone test-module
imports were corrected to work both through discovery and direct module names.
The real CPU-only optimizer check used three synthetic leaf tensors, without
forward, backward or updates; a separate tensor digest check also passed. Under
standard-library-only `-S`, those two optional tensor checks are explicit skips.
The final bundle recovery test verifies that old/partial archives are not reused
and that the published archive contains the current summary. No local performance
results are generated by these checks.
