# Grouped-codec representation × layout screen

This is a separate, randomly initialized five-arm experiment. It does not replace
the historical relative-received-SNR protocol, any historical model/result, or
the manuscript. No historical checkpoint is loaded. E is **Ordered-UE Joint-FDMA**,
not the historical Swin/Swin-adapted baseline. There is no predicted winner.

## Fixed configurations and physical protocol

All arms use K=4, Nt=32, Nsc=256, Nsb=4, CDL-C, perfect UL CSI, 256 occupied
feedback REs, and DL SNR=25 dB.

| Arm | UE | Physical layout | Complex symbols/UE | Active users/RE |
|---|---|---|---:|---:|
| A `legacy_all4` | Original AS encoder + conditioning | all four share all REs | 256 | 4 |
| B `ordered_all4` | Full angular-delay ordered-window encoder | all four share all REs | 256 | 4 |
| C `legacy_group2` | Same stored architecture as A | fixed {0,1}/{2,3} | 128 | 2 |
| D `ordered_group2` | Same stored architecture as B | fixed {0,1}/{2,3} | 128 | 2 |
| E `ordered_fdma` | Same stored architecture as B/D | interleaved exclusive REs | 64 | 1 |

FDMA UE k uses q=4j+k. Each group's support is the sorted union of its two FDMA
supports; each occupies 128 REs, and the two supports partition the full 256.
Every UE has 256 canonical slots, `slot=subband*64+j`. Per subband, the first
64/32/16 slots are active for all4/group2/fdma. Active slots map in order to the
sorted physical support, and recovery scatters back to those same canonical
slots. Group2 is not a fictitious `(Df,Da)` or a 128-RE feedback budget.

Each frame and UE transmits `z_active=8*u_active/||u_active||`, with total energy
64. Inactive physical REs are exactly zero. No batch-power normalization or
running-power buffer is used. Nonfinite or squared norm <=1e-12 fails rather
than repairing/replacing a sample. FP32 energy check tolerance is 1e-4.

The new axis is `gamma_db=10*log10(64/(256*N0))`, so
`N0=0.25*10**(-gamma_db/10)`. This is a transmit-block reference, **not** the old
received feedback SNR. A full `[B,32,256]` standard complex tensor has independent
unit-normal real/imaginary parts; exactly one `sqrt(N0/2)*eps` is added after
the full four-UE MAC. All arms use the same absolute N0, standard noise, and
channel tensors. There is no per-method/group/received-power noise rescaling.

## Receiver and complete tensor path

At each physical RE the public receiver uses only noisy Y, perfect public
`H_ul_est`, LayoutSpec, and N0. For the actual 1/2/4 active users it solves in
complex64 with `p=64/D_active`, `lambda=N0/p+1e-6`,
`A=G^H G+lambda I`, `z_hat=solve(A,G^H y)`. The feature
`clip(1-real(diag(lambda*solve(A,I))),0,1)` is a nominal-power reliability proxy,
not a calibrated posterior precision. True codewords and realized noise power
are detached diagnostic outputs only after decoding and RZF finish.

| Component | Shape / implementation |
|---|---|
| Legacy UE | Original direction normalization, stride4, width256/four Transformer layers, subband pooling; max AS head256 |
| Legacy condition | `[0,D_active/256,gamma_db/25]`; active length is not a physical allocation budget |
| Ordered UE input | Antenna FFT/frequency IFFT, full `[B*4,2,256,32]`, no delay cropping |
| Ordered UE tokens | patch4, width128, four existing nonshifted Swin blocks, `[B*4,512,128]` on64×8 grid |
| Ordered pooling | Mean only within each4×4 token window; ordered16×2×128 features→4096→FC512 |
| Complex serialization | Ordered UE: first256 real, next256 imaginary; legacy: subband-major, each original head first64 real/next64 imaginary |
| Recovery | Canonical soft symbols/reliability `[B,4,256]`, inactive exactly zero |
| Observation memory | `[B,256,128]`, noisy Y, H_ul_est, physical q, active-user mask, reliability proxy, gamma/25 |
| Joint queries | `[B,1024,128]`, real/imag, canonical/physical coordinates, roles, UE identity, proxy |
| Two joint layers | pre-LN self-attention→observation cross-attention→GELU FFN128→256→128; four heads |
| Codeword correction | Small-nonzero initialized complex residual, no zero scalar gate or receiver power renormalization |
| Spatial expansion | Same FC512→65536 per UE,64×8×128 patches |
| Spatial cross-attention | All2048 spatial queries attend to all1024 canonical context keys, with inactive keys masked |
| Full AD output | Four common nonshifted Swin blocks, patch4/window4/heads4/width128/MLP ratio4→`[B,4,2,256,32]` |
| Deployment/task loss | Inverse transform→`H_hat[B,4,32,256]`→original RZF1e-3→power1 per subcarrier→original sum-rate loss |

Joint self/resource biases use **physical q**, not canonical slot indices.
Inactive queries are zero at initialization and after every residual update;
self/spatial attention masks inactive keys; corrected symbols are masked before
the FC. Group2 still has one jointly interacting four-user neural decoder.
The old stored FDMA UE branch is retained but not executed, and is excluded from
active-gradient requirements. It is included in reported storage parameters.

## Read-only reference provenance

Reference: `08_PilotAware_CSI`, branch `research/step4h-r1`, commit
`bfa04279dd55873a61ecdd476c0144a8025cd496`. Files were actually inspected read-only;
there is no import, runtime dependency, weight migration, or write to08.
Paths below are relative to `src/icc08/resource_research/`.

| Reference | Correspondence / explicit difference in03 |
|---|---|
| `round6_resource/codec.py` | Ordered window pooling and configurable codeword dimension→`OrderedWindowUE`;256 slots rather than its216-slot setting |
| `round3_spatial/codec.py` | Full AD patch/Swin spatial organization; retained all256 taps, no32-tap crop |
| `round8/decoder.py` | Observation-memory and residual corrections informed the public receiver feature interface; no fabricated pilot inputs/calibrated precision |
| `round6_resource/decoder.py` | Joint self/observation attention and spatial context→mapped/masked `GroupedJointSwinDecoder`; explicit physical resource versus canonical coordinates |
| `round8/protocol.py` | Explicit resource ownership inspired LayoutSpec; no reserved pilot/key resources, all256 REs are data |
| `round8/physics.py` | Explicit energy/noise accounting; this batch uses E=64 and the new declared gamma, not08's pilot/energy/EMA conventions |

Pinned Git-blob SHA256 values (not platform newline conversions):

```text
round6_resource/codec.py   843b400676699d194f456461fe468a535c47ea8f929ac33241cb6e41606ab8d1
round3_spatial/codec.py    a933df5b2e2e0fb868f7fc26036dc691f6bdd0bfd4b7f527c2fecb934b8e0031
round8/decoder.py         b4ac804d7aa0d6d0f2da2566801cd6a70f0b5daa84c8bfd68518547160968124
round6_resource/decoder.py fed46bad6071dc7bc6cc9c11031a2c4406f292f4eb5785038756b53e9aa28590
round8/protocol.py        b550fe92e1a5c296729a9af4409d1d68d12eae20e99211d2759422fee60917f1
round8/physics.py         2ef5f3244ef79d1335dbab3a26aeb3ac7ddd008421d9e6322cb90d919af34b94
```

The new implementation reuses the previously source-checked03 nonshifted Swin
and all-key SDPA components. This is not a reproduction of08's whole system.

## Initialization, training, and resume

All ten runs are fresh, two seeds20261030/20261031 × five arms ×20,000 updates.
The shared BS is initialized in its own addressed CPU RNG stream, identical
across five arms within seed. A/C share initial UE weights; B/D/E share initial
UE weights and storage counts. Layout/position buffers differ legitimately.
Two UE families are not parameter-matched.

Recipe: physical batch48; one AdamW parameter group, peak LR1e-4 for both UE/BS,
weight decay1e-5, OneCycle10% warmup, gradient clip0.5. DL25, continuous uniform
gamma[0,25], task sum-rate only (auxiliary weights zero). Explicit FP32/complex64,
highest matmul precision, TF32 off, no AMP or gradient accumulation. Fixed64
validation channels, six gamma points0/5/10/15/20/25 every500 steps; single best
selected by their mean. No test-set selection and no source step0 weights.

ReplayableChannelData provides shared channels. Gamma and complete standard
noise use separate deterministic addresses `(version,seed,split,index,stream)`;
SHA256-derived63-bit seeds, never Python hash. Training index is global_step;
validation index0 and test batch_index use distinct namespaces. Arm, UE family,
gamma, dropout, wall-clock and path do not enter the physical noise address.
Each gamma/method uses the same noise base with the declared N0 scale. Changing
physical batch division is not claimed equivalent.

The sole existing-source change is the default-None `execution_protocol` hook
in `main_train.train_proposal`, for layout-aware forward/validation/metadata and
the new numerical policy. Old None calls keep the original model/allocations,
data, noise and numerical path. New configs explicitly declare/fingerprint this
protocol and reject missing/mismatched objects before outputs. The independent
config contains no fake ALLOCATION_GRID. Resume requires exact layout/config,
fingerprint, full state keys/shapes/dtypes, active AdamW moments, scheduler20k
and last_epoch, selected-best evidence, default RNG state and protocol identity.
Private Sionna RNG objects are not checkpoint payloads. Full-training bitwise
equivalence to an uninterrupted run is **not yet validated**.

Output roots:

```text
runs_as_screen/grouped_codec_v1/checkpoints/<arm/seed experiment name>/
runs_as_screen/grouped_codec_v1/logs/<arm/seed experiment name>/
runs_as_screen/grouped_codec_v1/batches/<unique batch>/
runs_as_screen/evaluation/k4_grouped_codec_v1/<batch>/seed_<seed>/<unique attempt>/
```

The controller uses existing locks, child-process logs, hash-verified completed
stages and small-result packaging. It does no cleanup or historical training.
Fresh execution refuses existing destinations. First500 updates are included
in20k, then strict resume continues501–20000 with the same scheduler.

## Preflight and memory

Server-only checks verify exact initialization, strict loading, physics,
canonical mapping, inactive isolation/gradient, complete Joint execution,
active gradients and per-subcarrier power. Each arm runs at real batch48 on
synthetic channels (no CDL in model preflight). One **disposable synthetic**
AdamW update allocates optimizer state for memory checks; no weights are saved
and it is not one of the200,000 official updates. Normal training starts fresh.
Fixed-recovery intervention must show legal observation-memory influence.

SDPA, query chunks64 with all keys, and nonreentrant activation checkpointing
are preserved. OOM/nonfinite/error stops and retains evidence. There is no
automatic microbatch, structure reduction, precision change, retry seed, or
extra training. Previously accepted old model/RNG test suites are not rerun.

Static storage expectations, **not local measured counts or latency**:

| UE family | UE including condition/stored inactive branch | BS | Total |
|---|---:|---:|---:|
| Legacy A/C | 10,877,090 | 34,933,678 | 45,810,768 |
| Ordered B/D/E | 2,899,584 | 34,933,678 | 37,833,262 |

The common FC alone is33,619,968 parameters. Server preflight must verify actual
counts and save measured batch48 peak allocated/reserved memory. No lightweight
or equal-parameter comparison is claimed.

## Independent evaluation and interpretation

Test seed20261210, batch8×200=1600 complete multi-user channels per condition,
DL25, all six gamma points. All five arms share actual channels and standard
noise. Each training seed is reported separately; not pooled as3200 samples.

Predeclared contrasts: B−A, D−C, C−A, D−B, D−E, B−E, and D−C−B+A. Each is formed
directly per common channel before mean/SE. Low0/5/10 and all-six averages are
formed within that same sample before statistics. Both sample-based and
batch8-cluster ddof=1 SE/mean±1.96SE intervals are retained; one-batch optional
evaluation preflight cannot estimate batch SE. All negative values remain.

Primary engineering rule: D−E low mean>=0.5 bps/Hz, low batch CI lower>0,
gamma25 batch CI lower>=−0.1. It is neither a publication guarantee nor a claim
of outperforming all SOTA. Layout changes length, per-coordinate energy and
separation jointly; do not interpret it as isolated demixing gain.20k does not
establish convergence. CI describes sampling for fixed checkpoints, not
multi-training-seed uncertainty. Latent diagnostics are within-representation,
not a common CSI reconstruction ranking.

Each seed exports30 summary rows, raw sample CSV.gz,56 paired rows
(7contrasts×8estimands), raw paired CSV.gz, parameters and manifest. Input hashes,
eval modes/buffers, shared input immutability, energy, noise, power and finite
metrics are checked. Results include full curves, paired batch-CI plots, tables,
and English/Chinese drafts. The small verified bundle excludes weights and raw
compressed samples; these remain on the server. No manuscript write occurs.

## Commands (not executed during laptop preparation)

After safe `main` fast-forward synchronization, activate the existing environment:

```bash
cd ~/project/03_PRECODER
conda activate precoder
tmux new-session -d -s grouped_codec_v1 'python -B -u run_grouped_codec_batch.py --run'
```

Exact terminal report, no glob or downloads:

```bash
cd ~/project/03_PRECODER && python -B run_grouped_codec_batch.py --summary
```

After interruption, only when the previous process is no longer active:

```bash
cd ~/project/03_PRECODER
conda activate precoder
tmux new-session -d -s grouped_codec_v1_resume 'python -B -u run_grouped_codec_batch.py --run --resume'
```

Completed stages are hash-checked and skipped. Failed stages preserve evidence;
interrupted training resumes the last complete checkpoint without adding to
the20k schedule. If the last20k checkpoint was saved before the completion
status write was interrupted, strict model/optimizer/scheduler/RNG/best checks
precede status finalization, with zero further updates and no trainer call.
`--summary` prints stage status, all ten completed training
records, measured parameters/memory, both seeds' six-point rates, all contrast
intervals including interaction, and the exact summary/result-bundle paths.
If failing, return this terminal report and its named small failure bundle;
do not send checkpoints or the raw sample archive.

## Verification status

Local checks are limited to syntax, standard-library CLI/path/metadata tests,
small synthetic statistics/physical tensors, and static source contracts.
No local large-model construction, CDL generation, CUDA profiling, training,
performance evaluation, or plotting is permitted. Real model/backend/initialization,
batch48 memory, training/resume and screening results remain server validation
items until this batch is actually executed.

Preparation validation (2026-10-27): the combined six new test suites reported
`Ran 60 tests; OK (skipped=1)`. The skip is the real-Sionna small-module numerical
class (four cases), because the laptop has Sionna0.15.1 rather than2.0.1. No
replacement dependencies were installed or faked. The protocol tests used only
small genuine CPU Torch tensors; no full model was instantiated. Ten training
dry-runs, both evaluation dry-runs and controller/preflight dry-runs were guarded
against heavy imports, weight reads and output writes. A separate `python -S`
regression combining the new trainer tests and existing replayable-data tests
reported56 tests, OK with3 skips (two unavailable-Torch cases and one real
Sionna class). Nine original numeric/default helper ASTs match commit1dac61b.
Syntax and whitespace checks passed. These are code-preparation results, not
server CUDA/checkpoint/training/evaluation acceptance.
