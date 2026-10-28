# Legacy + FDMA supplementary batch

## Scope and status

This independent extension adds only **F = Legacy + FDMA** to the completed
A/B/C/D/E grouped-codec screen. It does not modify the original five-arm enum,
configuration fingerprints, training code, models, tests, documents or results.
No model runs, server connection, training, performance evaluation or plotting
have been performed on the laptop for this change. Runtime checks below are
executable acceptance requirements, not claims that they already passed.

The user reports that the ten original 20k runs completed: grouping improved
low-gamma performance, replacing Legacy with Ordered did not improve the
current rates, and the D-E engineering rule failed for both seeds. This batch
must not change those results or retrain those ten runs. Their evidence is read
only from this exact path, never a latest/glob discovery:

`runs_as_screen/grouped_codec_v1/batches/batch_b314d7520c844f259ffa5db18ab0b149/batch_summary.json`

The path is absent from the laptop. Before GPU checks or training, the server
controller verifies completion, all stage evidence hashes, source hashes,
the ten training config/status records, selected checkpoint hashes, exact
evaluation pointers, manifests, original compressed samples and old statistics.
Unexpected evidence fails closed; the tool does not repair or recreate it.

## Model and physical protocol

| Item | F implementation |
| --- | --- |
| Identity | `legacy_fdma`, `GroupedCodecFDMAPrecoder`, `grouped_codec_fdma_v1` |
| UE | Existing `LegacyUE`; original **AS** branch, not its FDMA branch |
| Stored UE width | 256 canonical complex slots, four subbands of 64 |
| Activity | First 16 slots in each subband; 64 active symbols per UE |
| Conditioning | Active length 64; `[0,64/256,gamma/25]` |
| Physical layout | `LayoutSpec("fdma")`; physical resource `q=4*j+k` |
| BS | Unchanged `GroupedJointSwinDecoder`, full joint self/cross attention, codeword residual, spatial cross-attention, four Swin blocks, 256 taps |
| Deployment | Existing inverse angular-delay transform and normalized RZF, alpha=1e-3 |

The model overrides construction and identity only, inheriting the original
numeric forward and diagnostic methods. Inactive original FDMA parameters
remain stored and are not executed. No hidden original-BS prediction path is
added. The source model is randomly initialized, not loaded from C or another
trained checkpoint.

The original `RandomAddress(seed,'init',0,'ue_legacy')` and `bs` streams are
reused. F's UE initialization must match A/C tensor by tensor, the BS must match
A-E, and the layout buffers must match E. Names and output paths do not enter
channel, gamma or noise addresses. The F CUDA preflight compares actual
initialization with the recorded old hashes and independently reconstructed
common UE/BS modules; it does not rerun old model forwards.

Static expected storage counts (not yet measured for F on the server):
UE **10,877,090**, BS **34,933,678**, total **45,810,768**. The preflight requires
actual instantiation to reproduce them. These are not MACs or latency claims.

All methods occupy 256 RE and spend energy 64 per frame per UE. F's 64 active
symbols have mean energy 1. The common noise is
`N0=0.25*10**(-gamma/10)` with perfect UL CSI. Gamma is a transmit-reference
quantity, **not** historical received feedback SNR. Nothing here may be spliced
into the old feedback-SNR plots or attributed solely to power control or reduced
interference: changing layout also changes codeword length and separation.

## Fixed new training budget

Only F is trained, with seeds 20261030 and 20261031: 20,000 new updates each,
40,000 total. Batch48, common UE/BS peak LR1e-4, AdamW wd1e-5, OneCycle warmup10%,
gradient clip0.5, DL25 and continuous uniform gamma0..25 are unchanged from the
old screen. All auxiliary losses are zero. A fixed 64-channel validation set
and six gamma points select one best checkpoint by mean_all every500 updates.

The shared trainer and original `GroupedExecutionProtocol` numeric methods are
reused through their existing optional interfaces. The new config explicitly
declares its own identity and fingerprint while retaining the same data and
physical protocol. No change is made to `main_train.py` or the old entries.
FP32/complex64, highest matmul, TF32 disabled and the original activation
checkpointing policy are retained. There is no automatic precision, batch,
microbatch or architecture fallback.

The initial 500 updates are part of the 20k plan. On success, the controller
resumes at501 with the original 20k scheduler and optimizer/RNG/best state.
Strict resume rejects cross-arm/config/state mismatches. A final-step status
write interruption can be finalized without any extra optimizer update. This
is checked state restoration, not a claim of bitwise equivalence to an
uninterrupted training trajectory.

New checkpoints/logs live under `runs_as_screen/grouped_codec_fdma_v1/`.
The controller's `last_batch.json` is inside that new root only. Existing run
directories are not overwritten. Explicit `--resume` and run locks protect
interrupted work; completed stages are skipped only after evidence hashes pass.

## Two separate evaluations

Both use DL25, gamma0/5/10/15/20/25, batch8 x200 =1600 complete multi-user channels
per condition. Training seeds are reported separately.

1. **old_completion**, testseed20261210: only F runs. Before each F batch,
   H_dl, H_ul and standard_noise SHA256 must exactly equal the corresponding
   archived old batch hashes. A-E original samples are read, not re-evaluated
   or rewritten. The old seven contrasts and A-E rates are checked unchanged.
2. **new_replication**, testseed20261220: all six already-selected best models
   run on shared actual channels and standard noise. No old model is trained.
   This preset new test is the primary comparison; old completion is a
   post-screen supplementary comparison, not an untouched initial test.

Outputs are isolated under
`runs_as_screen/evaluation/k4_grouped_codec_fdma_v1/<dataset>/`.
Each evaluation writes 36 summary rows, raw samples, 96 paired-statistic rows,
raw paired differences, parameters, metadata/results JSON and a manifest.
Weights, compressed raw samples and full logs stay on the server.
Evaluation checks exact identities, strict loading, finite outputs, input hashes,
eval mode, immutable buffers, power and physical diagnostics. It never selects
an allocation or checkpoint using an individual test sample or a test SNR.

## Statistics and predeclared decisions

Retain B-A, D-C, C-A, D-B, D-E, B-E and D-C-B+A. Add C-F, A-F, F-E,
C-E and `(C-F)-(D-E)`. All are signed, direct within-channel linear contrasts.
First average the same sample over gamma0/5/10 (or all six), then compute mean,
ddof=1 SE and mean +/-1.96SE across channels. Also group by sample_index//8 and
report batch-level SE/CI across200 batches. No CI-endpoint subtraction or
independent-means SE substitution is allowed. Both interval types remain visible;
neither datasets nor the two training seeds are pooled.

The primary C-F engineering rule on new_replication is fixed before F results:
low-three-point mean >=0.5 bps/Hz, low-point batch CI lower bound >0, and gamma25
batch CI lower bound >=-0.1. The same corresponding old-completion outcome and
the original D-E rule are also reported. This is an engineering screening rule,
not a journal acceptance criterion. CI describes fixed-checkpoint sampling,
not stability over training seeds. Negative values and every gamma are retained.

All twelve exact metrics paths are read for best step, the final five validation
points and the last5k descriptive trend (15000..20000 inclusive, eleven points).
A best checkpoint at20k does not establish convergence or trigger more training.
Curves show both seeds in separate panels, all six methods and all six gamma.
Internal bilingual drafts distinguish same-family layout effects from whole
UE-family comparisons. They do not modify or make new claims in the paper.

## Interfaces and server operation

All entrypoints default to help. `--dry-run` is standard-library-only, reads no
weights and creates no output. The controller fails if actual old evidence,
dependencies, source hashes, initialization or true batch48 CUDA checks fail.
The scratch preflight update is a disposable technical check, not saved model
training; its synthetic rate is not an experimental result. No old RNG/model
suite is rerun and no cleanup is performed.

```bash
python -B run_grouped_codec_fdma_batch.py --dry-run
python -B -u run_grouped_codec_fdma_batch.py --run
python -B run_grouped_codec_fdma_batch.py --summary
# Only after an interrupted/failed run, never concurrently:
python -B -u run_grouped_codec_fdma_batch.py --run --resume
```

The summary reads small status/metrics files for last recorded step and time,
not checkpoint tensors. Missing results are explicitly unknown. A completed
summary includes F's two completions/best records, old-input hash pairing,
both complete six-method datasets, paired intervals, engineering decisions and
exact small-package paths. Send the complete `--summary` output. On failure it
also identifies the failed stage and evidence paths. Small packages contain
CSV/JSON/docs/figures only, excluding weights, raw samples and terminal logs.

## Local validation boundary

The local checks cover independent identity/protocol/config fingerprints,
output and resume rejection, optimizer/scheduler metadata, immutable old source,
small synthetic pairing/statistics, failed evidence, reporting and CLI import
boundaries. They do not instantiate the full F network. Server-only items are
the real old batch verification, tensor-level model initialization, actual
parameter totals, batch48 CUDA memory/gradients, strict real checkpoint loading,
actual channel/noise replay and all training/evaluation/results generation.

Local checks actually executed (2026-10-28):

- `python -X utf8 -B -m unittest tests.test_grouped_codec_fdma tests.test_grouped_codec_fdma_batch tests.test_grouped_codec_fdma_results -v`: **48 passed**, no failures/errors/skips.
- `python -X utf8 -B -S -m unittest tests.test_grouped_codec_fdma -q`:
  14 passed and two expected skips because site packages, including Torch,
  are deliberately unavailable in this standard-library-only invocation.
  Those two small CPU tensor/optimizer-construction checks passed in the normal
  environment; neither creates a large network nor performs a training update.
- All ten new Python files parsed successfully with `ast.parse`.
- Guarded fresh-process dry-runs cover both F training seeds, both evaluation
  datasets/seeds, controller and preflight. They reject heavyweight imports,
  checkpoint reads and filesystem writes. Default summary returned not_started.
- Result-package end-to-end checks used synthetic temporary files and
  `make_plots=False`; no plot or performance result was generated locally.
- Local distributions are Torch2.6.0 and Sionna0.15.1; the accepted server
  Sionna2.0.1/CUDA execution is not available here. No environment was installed,
  upgraded or faked. Real-model/CUDA acceptance and all server stages remain
  **not executed**, not passed.

Only the eleven new batch files are included in the code change. Existing
manual plotting edits and four untracked figures are protected. There is no
paper, historical result, old model, original trainer or 08-project change.
