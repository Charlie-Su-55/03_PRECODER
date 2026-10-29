# Grouped hybrid exclusive-anchor screening

This is an independent continuation experiment, not a revision of the previous grouped-codec results. The complete two-layer Joint / spatial-cross-attention / four-Swin decoder already existed. This batch tests genuine user-exclusive observations alongside group-shared observations, an explicit own-user anchor, and a historical diagonal-power receiver model. No performance improvement is assumed.

Only 03 files are used. There is no cleanup, no 08 access, no paper edit, and no historical retraining. Old Full/Lite/ASDirect/ASReceiver/ASSpatial/grouped-codec weights, results and manual plots remain untouched.

## Fixed geometry and energy

K4 / Nt32 / Nsc256 / Nsb4; DL25; groups `{0,1}` and `{2,3}`. `Da` below is **per group**, and the budget is **4 Df + 2 Da = 256**, not the historical hybrid budget.

| Arm | Exclusive symbols / UE | Shared symbols / group | Active symbols / UE |
|---|---:|---:|---:|
| fdma_control | 64 | 0 | 64 |
| hybrid_f16_a96 | 16 | 96 | 112 |
| hybrid_f32_a64 | 32 | 64 | 96 |

Each canonical UE vector has 256 complex slots, arranged as four consecutive 64-slot subbands. Within each subband: the first `Df/4` slots are exclusive, the next `Da/4` shared, and the remainder inactive. For group `g`, its ordered 32 physical positions in each subband satisfy `q % 4 // 2 == g`. The first `2*(Df/4)` positions alternate between its two users; the rest are shared. The FDMA endpoint therefore exactly recovers the previous F layout `q=4*j+k`. Canonical-to-physical and inverse tables, occupancy and group orthogonality are explicit.

One UE vector is jointly normalized over **all** active exclusive/shared slots to energy64 per frame. It is not energy64 per branch. All methods occupy 256 physical RE. One actual four-user MAC and one full complex noise tensor produce the observation. `N0=.25*10**(-gamma/10)`; gamma is the absolute-energy reference protocol, **not** historical received feedback SNR. Perfect UL CSI, no pilots, channel-estimation experiment, dynamic grouping or power controller is added.

## Receiver and complete network

The unchanged Legacy AS encoder generates maximum-width canonical symbols; the original stored FDMA encoder is not executed. The activity length supplied to conditioning is 64/112/96 respectively. Same-seed arms strictly inherit the same completed F UE and common BS tensors. The three arms have identical parameter capacity and identical newly initialized anchor parameters; geometry/masks differ.

For each physically observed exclusive or two-user shared RE, the receiver uses noisy `Y`, available UL channels, public N0 and committed historical powers pi:

```
L = solve(G^H G + N0 diag(1/pi) + 1e-6 I, G^H)
soft = L Y
error = diag((LG-I) diag(pi) (LG-I)^H + N0 L L^H)
feature = log1p(error / max(pi, 1e-8))
```

This diagonal-power linear error proxy is a model feature, not a calibrated posterior error. Current true codewords, true DL CSI, noiseless observations and simulator noise realizations never enter the BS neural decoder. Actual normalized symbols are detached only for post-update historical statistics and post-output offline diagnostics.

The explicit anchor uses only that user's exclusive soft symbols, transformed error features and canonical/physical coordinates. Linear9→128, GELU, exclusive-mask mean pooling and Linear128→128 create a user context added to its own Joint queries. Output weights use small nonzero initialization (std .001), without a dead zero gate. Shared/inactive symbols are excluded. This is not claimed to be an independently decodable base layer.

The complete existing learned path remains: raw observation memory → 1024 codeword queries → two Joint self/cross/FFN layers → complex codeword residual → per-UE FC512→65536 → all-user spatial cross-attention → four Swin blocks → `[B,4,2,256,32]` angular-delay representation → full `[B,4,32,256]` channel representation → original normalized RZF (regularization .001). Inactive corrected slots are masked; received symbols are not re-normalized as transmissions. There is no old prediction shortcut.

Static added anchor parameter arithmetic is 17,792; this is not a local full-model measurement. Server acceptance records actual UE / BS / full counts and confirms equality across the three new arms. Historical F has no anchor/prior receiver and is only a frozen secondary reference.

## Historical prior lifecycle

Six nodes at gamma0/5/10/15/20/25 store `[6,4,256]` powers. Initial active powers are uniform with sum64 per UE/node; inactive powers are zero. Reads linearly interpolate neighboring **committed** nodes. Updates use detached batch-mean actual jointly normalized coordinate power, shrunk 5% toward uniform. Each adjacent node takes EMA rate `.01 * interpolation_weight`; each UE/node is normalized to sum64. Integer update count and float64 node update masses are registered buffers.

Forward at update `t` reads committed state through `t-1` and returns detached pending statistics. The opt-in `main_train.train_proposal(..., lifecycle_protocol=None)` interface validates a matching serialized declaration before output creation. Only after `optimizer.step()` succeeds does it commit the pending statistics, then the scheduler and validation/checkpoint stages proceed. An unsuccessful optimizer step, extra forward, step0 validation, ordinary validation or evaluation never commits. Stale, repeated, cross-layout, unnormalized or nonfinite pending statistics are rejected.

The default None lifecycle preserves previous trainer behavior and previous configuration fingerprints. The new lifecycle additionally selects step0 and stores complete optimizer/scheduler/RNG state in best as well as latest. The new interface is not silently activated for any historical run.

## Source identity, initialization and restoration

Source paths come only from `fdma_config(seed)`; no glob or latest-result discovery. Pinned **best** hashes:

| Training seed | F best SHA256 |
|---|---|
| 20261030 | 3aab138084319dc9d875a4a8b53217eaf99eb6d6a0e42b3f13de0ff2994a010f |
| 20261031 | 7f3731a54fbc568378f5dbeb31800ffeb6ba25051be47a24d01f721529cd47f5 |

Best / latest / full config / status are checked; source latest must have completed20k, while source best may precede latest. Every old state key, shape and dtype must match. Explicit migration accounts for each source tensor, initializes only anchor parameters and the three historical-prior buffers, then performs a complete strict load. No old optimizer is inherited. Because geometry, conditioning length and receiver features change, migration does **not** imply equal forward outputs.

New restores require complete arm/config/fingerprint/source/code identity, complete model and prior state, matching prior update count, optimizer, scheduler and Python/NumPy/Torch CPU/all-CUDA RNG. No remigration or reinitialization on resume. Missing prior state is fatal. Best step0 and latest step0 are valid new-protocol states (empty optimizer moments, scheduler0, prior count0); the original data helper's historical step>=1 behavior remains unchanged. Best has the prior corresponding to its own selected step, not the latest prior.

New namespaces address channel data, gamma, physical noise and initialization. Same training seed across all three arms gives identical actual channel/random addresses; the namespace is distinct from the earlier20k experiment. Data reuse is only promised for the same physical configuration and batch definition. This does not prove bitwise uninterrupted-versus-resumed complete training equivalence.

## Fixed training and independent evaluation

Six runs: three arms × seeds20261030/20261031, each20k **new** optimizer updates. Batch48; AdamW shared UE/BS peakLR1e-4, wd1e-5, clip .5; OneCycle warmup10%; FP32/complex64, highest matmul, TF32off, no automatic precision/batch fallback. Pure sum-rate through normalizedRZF; no auxiliary reconstruction loss. DL25, gamma continuous uniform0–25.

Validation uses one fixed256-channel set, chunks8, six equal-weight gamma nodes. Step0 and every500 compete for one best by `mean_all`. The first500 updates are part of the same20k schedule; the controller resumes501–20000 automatically, without another smoke budget. Six validation curves show steps0–20000 only, never old F's different validation history. End-of-budget best does not prove convergence.

Independent test seed20261228, new namespace, batch8×200 =1600 complete multi-user channels per condition. Four methods per training seed: the three new selected checkpoints and frozen same-seed source F. All four and all gamma nodes share actual batches and standard noise; hashes are recorded. The two training seeds are reported separately, with actual test-input hashes checked equal. No test-based selection, rerun seed search or oracle rows.

Fixed contrasts: H16−Fnew, H32−Fnew, H16−H32, Fnew−Ffrozen. Per-gamma, low0/5/10 and all-six deltas are computed per same sample first, then summarized with ddof1 SE and mean±1.96SE. Batch intervals use200 means of8 complete channels. Both sample and batch intervals are retained; seeds are not pooled. Each hybrid separately passes the engineering screen only if low-three mean≥.5, low-three **batch** CI lower>0, and gamma25 batch CI lower≥−.1. This is not a journal acceptance criterion or multi-training-seed confidence interval.

Outputs include all24 conditions and32 contrast/scope rows per seed, raw samples and paired samples (server only), signed differences, energy/error/anchor/interference diagnostics, actual parameters, frozen checks, manifests, full six-point two-panel curves, paired-CI plots, all six validation curves, and bilingual internal drafts. Frozen F's unavailable prior/anchor diagnostics are marked unavailable, not fabricated as zero. No Swin or historical received-SNR curve is spliced into this protocol.

## Execution and evidence

Default CLI shows help. All dry-runs use standard-library paths/configuration only: no models, weights, output directory or heavy dependencies. The controller uses independent stages and locks, checks source identity, runs only this batch's required real-batch48 synthetic CUDA acceptance, trains six runs, evaluates twice and organizes results. CUDA OOM/numerical/identity failures preserve reports and stop; no automatic structure/batch/precision change. Completed stages are reused only after their recorded evidence hashes match. Existing outputs require explicit resume. No cleanup is performed.

The preflight checks real source migration and same-seed initial parameters, strict prior restoration, complete module execution, own-user anchor information boundary, active task gradients, fixed power/energy, scratch optimizer-before-prior ordering and frozen evaluation buffers. Scratch preflight weights are never saved or used for training. It is not an experiment or a rerun of old model/RNG suites.

On the CUDA server, after safe main/origin synchronization:

```sh
python -B run_grouped_hybrid_anchor_batch.py --dry-run
python -B -u run_grouped_hybrid_anchor_batch.py --run
python -B run_grouped_hybrid_anchor_batch.py --summary
# Only following an interruption/failure, after reading the recorded error:
python -B -u run_grouped_hybrid_anchor_batch.py --run --resume
```

The sole batch pointer is `runs_as_screen/grouped_hybrid_anchor_v1/last_batch.json`; it resolves an exact `batches/batch_<id>/batch_summary.json`. `--summary` reads small sidecars, not weights: current stage/error, latest metrics time, six run states/verified best records, four methods×six points, all paired intervals, engineering decisions, exact package path/hash. Unavailable running information is explicitly unknown. Small packages include only CSV/JSON/Markdown/PDF/PNG; checkpoints, compressed per-sample data and terminal logs stay on the server.

## Verification boundary for this delivery

Local checks are limited to syntax, standard-library dry-runs, synthetic statistics/config/lifecycle tests and small CPU physical/anchor/prior tensor checks. Real source checkpoint loading, complete model forward/backward, batch48 peak memory, Sionna2.0.1 channels, six training runs and independent performance evaluations are **not executed locally**. No dependencies are installed or faked. The accepted historical server results are not re-labelled as results of this new batch.

Executed locally for this delivery: `python -B -m unittest tests.test_grouped_hybrid_anchor tests.test_grouped_hybrid_anchor_batch -q` reported `Ran 39 tests`, no failures/errors, and one explicitly skipped server-only full-model test class. The executed lightweight cases cover the actual small CPU physics/anchor/prior code, optional lifecycle order, complete resume schemas, paired statistics, result export, output protection, completed-stage hash rejection and guarded planning CLIs. All11 Python files passed AST syntax checks. Standard-library-only dry-runs passed for all three training arms, preflight, evaluation and controller, with planning import/write/weight-read guards also exercised. No model, checkpoint or CUDA acceptance is claimed by these results.

Server-only acceptance and all performance claims remain pending until this controller finishes and its reports are returned.

## Registered recovery of the 266d9e5 JSON-provenance defect

This section supersedes only the recovery instructions for
`batch_cf66957ea2914720bbdd9ed15dbf3884`. The user reports that environment,
source verification and CUDA preflight completed at
`266d9e52428d136f9de5d44a95da4afeea812fb6`; only `fdma_control/20261030`
reached update500, with best also500. The other five runs have not started.
These are user-reported server facts, not locally inspected checkpoint evidence.

The actual `source_identity` contains tuple-valued SNR settings under the saved
source recipe. JSON turns these into lists. The original direct comparison
therefore rejected unchanged metadata. The connected trainer resume boundary now
compares source metadata through the existing `_json` representation, without
rewriting historical JSON. Actual field changes remain errors, with field paths
and old/new values. Configuration/fingerprint errors and exact source-code SHA256
errors are reported separately.

### Narrow authorization, not a hash bypass

`grouped_hybrid_resume_compat.py` accepts only the exact batch above. The explicit
`--repair-commit` must be the delivered 40-character repair commit, current HEAD,
and a direct child of the old commit. The complete commit diff must contain only
the five files in this repair. The two executable replacements are additionally
pinned by their exact old/new SHA256 inside the tool. The supplied release commit
pins the compatibility tool itself, tests and documentation; all runtime sources
must match their Git blobs. Models, configuration, physical protocol, trainer,
optimizer/scheduler implementation and data/RNG sources retain their original
hashes. This avoids a self-referential embedded commit hash while still checking
the tool against the explicitly delivered release. An arbitrary filename match,
unregistered executable edit, other batch, other commit or edited record is not
authorization. Unrelated protected plotting changes are not included in the commit.

Registration requires the evidence-owning CUDA server. It acquires the existing
controller and all six run locks, rejects live owners, verifies old completed
stage reports and their files, and checks the original result-package hash. It
then reads actual best/latest and strictly checks the paused500 state, equal
best/latest model state, full config/fingerprint, same-seed pinned F source,
prior count/masses, optimizer moments, fixed20k scheduler and complete RNG states.
It verifies the other five run directories are absent. No optimizer is created,
no forward/backward is executed, and no checkpoint or saved prior is written or
reset. A temporary model is constructed solely for strict tensor/state checks.
Missing or different evidence stops registration; nothing is repaired in place.

The exclusive directory
`runs_as_screen/grouped_hybrid_anchor_v1/batches/batch_cf66957ea2914720bbdd9ed15dbf3884/resume_compat_266d9e5/`
contains `record.json`, `record.sha256` and the byte-for-byte
`batch_summary.before.json`. Original initialization, completed-stage evidence,
old result package and old logs remain untouched. The record includes exact
release hashes, evidence hashes and checkpoint verification results. Registration
compares all active evidence before and after preparation. Existing records are
verified, not overwritten; an incomplete registration is preserved and rejected.

Both controller and trainer require this same record for repaired-batch resumes.
The live summary adds its digest and version chain, retaining the original
`source` and completed-stage records. Every compatible controller resume checks
the installed runtime/backend against the old environment record without
rerunning its stage or the CUDA model preflight. The first compatible resume also
checks that active files still match their registered500-step hashes. Later
resumes allow normal best/latest/status/metrics updates, while retaining the
original strict checkpoint restoration checks and immutable evidence checks.
Source hashes in historical initialization are never replaced with new ones.

The existing500-step run continues at501; the original20k schedule, prior state
and optimizer/RNG restoration remain unchanged. The remaining five runs each
perform their original500+19500 plan. Completed stages are hash-verified and
reused, not rerun. Normal runs without a compatibility record retain strict
source checking. This repair does not claim bitwise equality with uninterrupted
training or any new performance result.

### Server use and verification boundary

After safe fast-forward synchronization to the exact delivered repair commit and
confirmation that the old controller/training processes have exited:

```sh
python -B grouped_hybrid_resume_compat.py --prepare --repair-commit <delivered-repair-commit>
# Only if preparation succeeds:
python -B -u run_grouped_hybrid_anchor_batch.py --run --resume --resume-compat runs_as_screen/grouped_hybrid_anchor_v1/batches/batch_cf66957ea2914720bbdd9ed15dbf3884/resume_compat_266d9e5/record.json
python -B run_grouped_hybrid_anchor_batch.py --summary
```

Keep the same explicit `--resume-compat` argument after any later interruption.
Do not edit JSON, hashes or weights, delete old directories, or restart the500
updates. The user-facing delivery includes a fully pinned safe-sync/tmux command.
Local verification uses real source metadata and temporary schema-only evidence,
not fabricated server acceptance. Actual server checkpoint loading, compatibility
registration and resumed training remain **not executed locally**.

Executed for this repair: the following standard-library-only regression command
passed **34 tests, zero failures/errors/skips** (23 new targeted cases plus 11
existing config/checkpoint-schema/controller cases):

```sh
python -B -S -m unittest tests.test_grouped_hybrid_resume_compat tests.test_grouped_hybrid_anchor_batch.HybridConfigTests tests.test_grouped_hybrid_anchor_batch.HybridCheckpointTests tests.test_grouped_hybrid_anchor_batch.ControllerContractTests -v
```

Tests exercise the connected provenance boundary with actual source metadata,
exact release policy, shared controller/trainer receipt, repeat-resume behavior,
evidence corruption rejection, completed-stage reuse and continuation commands.
The actual preparation control flow is exercised on temporary schema fixtures
with the server-only checkpoint-verification boundary explicitly mocked; original
fixture bytes and existing registration bytes remain unchanged. This is not a
real checkpoint/CUDA test. No Torch/Sionna dependencies were faked or installed.
Guarded help/dry-runs, Python AST syntax and staged whitespace/scope checks are
also part of local delivery verification. Server-only strict tensor/optimizer/RNG
validation remains pending and is mandatory in `--prepare` before continuation.
