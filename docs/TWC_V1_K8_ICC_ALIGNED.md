# K8 ICC mechanism-aligned controlled batch

This is an independent K8 adaptation, `twc_v1_k8_icc_aligned_v1`. It asks whether
the complete ordered encoder, joint normalization, historical coordinate power
and corrected-pilot receiver chain improve Hybrid relative to a matched FDMA
control. It does not assert an improvement before the server experiment.

The implementation reads the fixed ICC training source at
`b8ea3a51b051b7aae22916d99aa52d811edb0850`. The later documentation/reference
commit is `bfa04279dd55873a61ecdd476c0144a8025cd496`, not the ICC repository's
main branch. No ICC weights are used. Reference files were read through Git
without checking out or modifying the reference worktree.

## Reference-to-adaptation map

| Archived ICC mechanism | New K8 implementation | Deliberate difference |
| --- | --- | --- |
| `round6_resource/codec.py`: full spatial Swin UE and ordered window features | `models/twc_v1_k8_icc_aligned.py`: complete 32x256 AD transform, patch4, width128, four nonshifted Swin blocks, 32 ordered 4x4 windows flattened | K8 common216 complex head; no global-mean replacement, no32-tap truncation |
| `round8/physics.py`: select active216-prefix then joint energy normalization | `utils/twc_v1_k8_icc_aligned_physics.py`: select24 or44 coordinates and normalize the complete selected vector | K8 data energy48, pilot16, frame64; no fixed key/shared energy ratio |
| `estimation.py`, `acquisition/receiver.py`: unit-column global Fourier basis and actual pilot design | same new physics file: rank8 basis, actual64-pilot observation,64x64 joint design |8UE,8pilots/UE of amplitude sqrt2; Gram diagonal16/256, ridge.001, ridge/Gram=.016 |
| `ul_unet/estimator.py`: shared pilot-only residual frequency U-Net | `K8ICCAlignedPilotUNet`: widths32/64/96, GroupNorm8/GELU, full-frequency residual |8UE and public gamma0..25; no data support or true UL/DL/codeword accepted |
| `round8/statistics.py`: causal coordinate EMA and uniform shrinkage | `ReceiverStatistics`: decay.99,5%uniform, sum48/UE, explicit successful-update commit |one commit per32-frame effective batch; never forward; CE node interpolation replaces the ICC nonnegative affine fit |
| `ul_unet/receiver.py`: one corrected UL estimate in scalar/shared recovery | same new physics file: coordinate prior, per-RE CE uncertainty andN0 in ridge and recovery variance |scalar FDMA24 or pairwise shared40; K8 noise coefficient.25 rather thanICC.125 |
| `round8/decoder.py`: observation memory, two Joint layers, latent residual, FC expansion, spatial cross-attention, four Swin blocks | `K8ICCAlignedJointDecoder` |public K8 layout/coordinates, all8UE jointly attended; full256-tap physical CSI output |
| `ul_unet/train.py`, `round2/contracts.py`: separate CE and codec NMSE-only AdamW | `train_twc_v1_k8_icc_aligned.py` |microbatch4 x accumulation8; single arm resident at a time; fixed terminal checkpoints |

The unchanged common SDPA implementation is reused from
`models/twc_v1_k8_joint_decoder.py`; the two new arms use the same receiver
capacity. The normalized recovered/Joint residual intermediate is a
reconstruction latent. It is never reinterpreted as an actual transmitted
symbol for a new Y-Hx iteration.

### Pinned raw Git-blob SHA256

All following hashes refer to the training commit above, not mutable files in
the reference worktree. Additional model reference hashes are recorded with the
batch's reference evidence.

| Reference path under `src/icc08/` | SHA256 |
| --- | --- |
| `resource_research/round8/physics.py` | `2ef5f3244ef79d1335dbab3a26aeb3ac7ddd008421d9e6322cb90d919af34b94` |
| `resource_research/round8/statistics.py` | `23c62f5a38663b50e0855051477238fb51b1ba71552d445f7044b331133ae7d2` |
| `resource_research/round8/protocol.py` | `b550fe92e1a5c296729a9af4409d1d68d12eae20e99211d2759422fee60917f1` |
| `resource_research/ul_unet/receiver.py` | `4353bffe9450550a93ea87826881b428448cacc9133f42b413252553b7897110` |
| `resource_research/ul_unet/estimator.py` | `9bdf18e8af0db97c5c46e560f83760170a3762cf85aa39832e8681f1fea577ce` |
| `acquisition/receiver.py` | `2b07d8e77dfa9e785e7ecd53f419ac1e2a883eebdafaa780badcfebe974bcdfa` |
| `estimation.py` | `49e81a25cb3fc5b1b3205f4aae96918aeef6507c42bc892c9e98fbbc1421d021` |

## Fixed protocol and accounting

- K8/Nt32/Nsc256, independent normalized CDL-C DL/UL snapshots and unchanged
  4x8 BS array/channel parameters.
-256 occupied feedbackRE including64 orthogonal pilotRE,8 perUE. Frame energy
 64/UE =16pilot+48data. Both arms share actual channels and one standard complex
  noise tensor per physical frame; the single MAC adds noise only once.
- FDMA:24 exclusive complex data coordinates/UE. HybridG4: groups(0,1),(2,3),
 (4,5),(6,7), key4/UE and shared40/group. Budget64+8x4+4x40=256.
- N0=.25*10^(-gamma/10), E|noise|²=N0. Gamma is a public physical protocol
  parameter, not a measured received or post-detection SNR. Training gamma is
  independently Uniform[0,25] per MU frame. ICC used K4 and[5,25].
- All codec parameters start fresh from common seed849301. Common encoder and
  decoder weights match by value; the two models/optimizers/buffers must have
  independent storage. Independent CE initialization seed859301. The batch
  sampling seed is20261121. No historical trained codec is inherited.
- FP32/complex64; noAMP/TF32. AdamW lr1e-4, weight_decay1e-4,
  betas(.9,.999), eps1e-8, clip1, constant scheduler. CE loss is mean physical
  per-link ULNMSE; codec loss is mean physical per-link DLNMSE. No rate,
  cosine, physical-symbol or auxiliary loss.
- Microbatch4, accumulation8, effective batch32. Loss is divided by8 before
  each backward; clip and optimizer/scheduler update once per effective batch.
  The prior used for all8microbatches is the same historical prior; its explicit
  commit occurs only after finite parameters and a successful optimizer update.

| Stage | Optimizer updates | Microbatches | Complete MU frames | UE channel inputs |
| --- | ---: | ---: | ---: | ---: |
| New CE |5000|40000|160000|1280000|
| Each codec |20000|160000|640000|5120000|
| ICC K4 codec reference |20000|not equated|1280000|5120000|

The two codecs see the same640000 paired MUframes; adding method exposures
does not create1280000 independent channels. Matching approximate UE input
count with ICC does not match complete-MU-frame budgets.

## Public information and calibration boundary

Simulator-only true UL enters the physical MAC. True UL labels enter only new
CE supervision and independent CE calibration. True DL is the UE input and
the physical reconstruction/optional offline rate score target. Neither true
channel, the true/current-frame data codeword, nor its per-frame key/shared
energy is accepted by the deployable receiver/decoder.

Paid-pilot initialCE uses only the64 pilotRE. The new U-Net reads masked pilot
observations, actual known pilot amplitudes, initial estimate, masks and public
gamma. It predicts a full256-tone residual and is not projected back to rank8.
After5000CE updates, an independent256-MU calibration at each of the six
gamma nodes records per-user/per-RE complex error power `[6,8,256]`. Between
nodes the receiver uses nonnegative piecewise-linear interpolation in gamma;
there is no extrapolation or reuse of legacy interpolation-CE calibration.

Statistical recovery uses a shrunk normalized historical power prior per
active coordinate. With prior pi, CE diagonal error v and publicN0, the scalar
ridge is `.001+(N0+pi*v)/pi`. For a two-user sharedRE, effective noise is
`N0+sum(pi_i*v_i)` and each diagonal ridge is `.001+effective_noise/pi_i`.
Reliability is the inverse of the corresponding modelled diagonal recovery
MSE, including regularization bias. It is not a true posterior precision and
omits spatial/inter-user error correlation and conditional estimate bias.
The same corrected UL estimate enters recovery and observation memory.

## Data, identity, checkpoint and resume boundaries

All output is under `runs/twc_v1_k8_icc_aligned_v1/`; no old source/results,
STATUS, last_batch, weight or failed evidence is modified. New source identity
locks model, physics, data, optimizer and critical evaluation implementations.
Launcher/report identity and Git commit provenance are recorded separately;
report-only changes do not silently alter the training core. Any changed core,
recipe/fingerprint, data protocol, frozen CE/calibration or model tensor shape,
dtype/key set is rejected.

The existing isolated replayable Sionna channel generator is reused with a
new namespace. CE train, codec train, CE calibration, monitor, alpha reserve
and final validation use distinct roles/addresses. Same configuration and
microbatch/address replay is promised, not invariance to a changed batch split.
Old test is never sampled.

Rolling checkpoints are saved only at complete100-update boundaries and retain
strict model/EMA state, AdamW moments, scheduler, Python/NumPy/Torch CPU/allCUDA
RNG, full counters and the next microbatch address. Resume starts after that
boundary; missing/mismatched state is an error. The first100 updates are part
of the5000/20000 budgets, not additional benchmark training. No claim of
bitwise-equivalent uninterrupted full CUDA training is made by local tests.
Weights remain on the server and are never included in the small result bundle.

## Monitoring, final validation and interpretation

The controller runs necessary new CUDA checks, CE100→5000, independent CE
calibration, FDMA100→20000 and Hybrid100→20000, final paired validation and a
small CSV/JSON/vector-figure bundle. It does not rerun historical Full/Lite/RNG
test suites or ask for an intermediate100-step manual continuation.

The fixed monitor has128 complete MUframes at all six gamma points. It reports
trends without selecting weights. The primary final checkpoints are the fixed
20000-update terminal states; final validation has1024 new paired MUframes.
RZF is used only for monitor/validation: fixedalpha.03, DL25dB and per-tone
total transmit power1. This batch does not tune alpha or use test samples.

Reports include linear-mean-then-dB DLNMSE, physical downlink sum spectral
efficiency and user rates, desired signal/interference, offline soft-codeword
error (Hybrid key/shared separately), actual energy distributions, coordinate
prior, actual parameters and server-measured memory/time. Paired differences
are formed within complete MUframes, with cross-gamma averages first computed
inside the same frame. Sample-level and batch-of4 intervals are both retained.
This is a single-training-seed validation experiment; negative results remain.

Default summary is bounded near40lines/8KB and prints own failed-stage error
first, stage/counters/ETA, six-point NMSE/rate, batch paired intervals,
key/shared error and exact artifact paths. Details belong in files rather than
an unbounded terminal JSON dump.

## Server execution

Use the existing CUDA/Sionna environment. Safe Git sync must verify main,
origin, empty staging and no foreign unpushed commits; keep all local changes.
Stop if a03 process is still using this code before pulling. Do not terminate
another user's process. An example after safe sync:

```bash
conda activate precoder
cd ~/project/03_PRECODER
task_python="$CONDA_PREFIX/bin/python"
tmux new-session -d -s k8_icc_aligned \
  "cd '$PWD' && '$task_python' -B -u run_twc_v1_k8_icc_aligned.py --run --allow-shared-gpu --min-free-mib 12000 --max-gpu-util 80 > '$PWD/terminal_logs/k8_icc_aligned.log' 2>&1"
```

Explicit shared-GPU admission settings are recorded separately from training
identity. They are an initial observation, not a continuous memory/compute
limit or an automatic pause mechanism. The conservative default remains
available by omitting shared options. OOM/nonfinite/identity failures preserve
evidence and stop; they never silently shrink batch, model or budget.

Summary to return in full:

```bash
"$CONDA_PREFIX/bin/python" -B run_twc_v1_k8_icc_aligned.py --summary
```

Only after the original process has exited, continue owned stages explicitly:

```bash
"$CONDA_PREFIX/bin/python" -B -u run_twc_v1_k8_icc_aligned.py --run --resume --allow-shared-gpu --min-free-mib 12000 --max-gpu-util 80
```

Completed stages are reused only after evidence/hash validation. No old lock is
deleted and no completed training is automatically repeated. Partial failure
outputs stay for inspection.

## Local verification boundary

Only small CPU components, standard-library metadata/CLI checks and syntax
were run locally. Physics checks cover resource/energy/single-noise equations,
actual rank64 pilot design and Gram scale, Fourier ridge shrink, data-support
isolation, causal EMA/freeze/strict roundtrip, nonnegative node interpolation,
scalar/shared recovery/conjugate-axis math and finite small-tensor gradients.
When PyTorch is absent the numerical checks explicitly skip rather than install
or mock it. Final executed pass/skip counts are recorded in the batch delivery.

Final integrated local run:43 passed,0 failures/errors/skips; independent
standard-library-only run:22 passed and21 explicit real-dependency skips.
Four entry-point dry-runs and the no-start summary passed. All14 new Python
files passed Python3.10 AST/compile checks without writing bytecode. Controller
tests exercise completed-stage hash reuse, exact child-failure logs and the
bounded full six-point summary. The existing atomic union-CSV writer is reused;
optional missing diagnostics remain blank rather than zero.

No full codec was constructed locally; no real checkpoint, CDL sample, CUDA
preflight, optimizer training batch or model performance evaluation was run.
Actual full-size common initialization/storage checks, pilot-only newCE CUDA
invariance, full gradient/optimizer/RNG resume, parameter count, allocation
checks, memory/throughput and all final experiment conclusions await the
server controller. Server reports, not the architectural plan, establish those
facts.
