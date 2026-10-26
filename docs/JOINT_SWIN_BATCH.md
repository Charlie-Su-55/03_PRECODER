# Joint + full-delay Swin256 screening

## Scope and status

Independent 03-only implementation. Both new arms are **K4 / Nt32 / Nsc256 /
Dtot256 / Nsb4, pure-AS (0,256)**, with the original max-width UE (64,256).
Full, Lite, ASDirect, ASReceiver, the 32-delay ASSpatial decoder, original and
adapted Swin, old results and the manuscript are unchanged. No runtime imports,
weights, writes or synchronization in 08. No new performance conclusion exists
until this batch actually completes on the CUDA server.

Local acceptance covers syntax, standard-library planning, configuration and
strict-resume guards, tensor-flow/source contracts and synthetic CSV statistics.
Large model execution, real weights, numerical attention tests, B48 CUDA memory,
training, independent evaluation and generated result figures remain **not run
locally**. The six small genuine-runtime numerical tests are explicitly opt-in;
missing dependencies are not simulated or installed. Common-weight equality and
actual parameter counts are checked on real instances by the server preflight.

Local execution record: 12 new Python files parsed successfully; the five scoped
test modules ran 56 lightweight tests with no failures/errors. The explicit
numerical class (six small real-dependency tests) was skipped, not marked passed.
All four training plans, both evaluation plans and controller/preflight dry-runs
were exercised with heavy imports, weight reads and file writes prohibited.

## Pinned reference actually inspected

08 branch `research/step4h-r1`, commit
`bfa04279dd55873a61ecdd476c0144a8025cd496`. These blobs were read, not executed or
changed. Hashes are SHA256 of the pinned Git blob bytes, not the checkout's CRLF.

| Reference path | SHA256 |
| --- | --- |
| `src/icc08/resource_research/round8/decoder.py` | `b4ac804d7aa0d6d0f2da2566801cd6a70f0b5daa84c8bfd68518547160968124` |
| `src/icc08/resource_research/round6_resource/decoder.py` | `fed46bad6071dc7bc6cc9c11031a2c4406f292f4eb5785038756b53e9aa28590` |
| `src/icc08/resource_research/round4_joint_decoder/decoder.py` | `1997cc57c1003c4239861f84c30dfb92ff38f4ef24f2566de404baac8ea62d25` |
| `src/icc08/shared/swin_codec.py` | `08249826ea8a01832a6ebcfda8aa1e0d6358b2863500f64be3258aa1f3875d78` |

Necessary codec/transform dependencies were also read. AST comparison found all
seven shared window/patch/encoder/decoder definitions identical to 03's existing
`models/swin_precoder.py`; 03 directly reuses its non-shifted decoder blocks.
This is **not** reproduction of the entire 08 communication system.

| Audited 08 component | Actual 03 module / operation | Adaptation or boundary |
| --- | --- | --- |
| Round8 checkpointed full joint route | `JointSwin256Decoder.forward`, `_run` | Full route, nonreentrant checkpointing; no Direct-DETR shortcut |
| Round6 observation memory | `decoder.observation_memory` | Raw noisy Y, estimated UL channel, resource coordinates, reliability proxy, public FB SNR; no fake pilots/absolute N0 |
| Round6 resource/user joint tokens | `query_projection`, `query_geometry` | 4 x 256 = 1024 queries, not four independent decoders |
| Round4 pre-LN `_JointLayer` | `joint_layers.0/1` | Self attention, memory cross attention, GELU FFN128→256→128; four heads, no dropout |
| Round6 overlap/channel/resource biases | `overlap_bias`, `coupling_bias`, `resource_cross_bias` | Same-resource matches and normalized estimated-channel overlap, computed per query chunk |
| Round6 residual correction | `residual_norm`, `residual_head` | Complex additive residual; weight std .001, zero bias; no gate or receive-power renormalization |
| Shared codec expansion | `decoder.codec.fc` | 512 real values → 65536 values = 512 spatial tokens x 128 |
| Round6 spatial positions and all-user cross attention | `spatial_position`, `spatial_attention` | 2048 queries attend all 1024 joint context tokens |
| Shared Swin decoder | `decoder.codec.blocks.0..3`, `norm`, `final_proj` | Four non-shifted blocks, patch4/window4/heads4/MLP ratio4 |
| Shared inverse angular-delay transform | 03 `inverse_angular_delay_transform` | All 256 delay taps, no zero-padding from 32 taps |

08's exclusive-resource summary is not copied as a dead zero-input module: this
screen has only pure-AS resources. The query role features are explicitly [0,1].
No 08 BS/UE weights or pilot-grid model are migrated.

## Physical and neural data flow

The unchanged 03 UE performs per-subband prefix output, joint batch-power
normalization during training and the original running-power EMA (momentum
.02). There is one `transmit` call and one AS noise draw. The receive slice is
`K*Df : K*Df+Da`; here it is all 256 resource positions.

Existing `regularized_recovery` uses receiver-visible `H_ul_est` (only falls back
to `H_ul` when omitted), noisy Y and public FB SNR:

```
G = H_ul_est.permute(0,3,2,1)          [B,256,32,4]
y = Y_ac.transpose(1,2)[...,None]      [B,256,32,1]
noise_est = mean(abs(Y_ac)^2)/(1+10^(FB_SNR/10))
lambda = max(noise_est, 1e-6)
z_hat = solve(G^H G + lambda I, G^H y) [B,256,4]
r = clip(1-real(diag(lambda*solve(G^H G+lambda I,I))),0,1)
```

Solves are complex64. The mean retains the original batch/antenna/active-resource
scope. This is approximate unit-codeword-variance regularized recovery; r and
`log1p(r)` are modeled features, **not calibrated posterior precision**.

Memory uses interleaved real/imag received vectors (64 values/resource) and
estimated channel vectors (256/resource), three coordinate features, four
`log1p(r)` features and FB-SNR/25, projected and summed then LayerNorm128.
Coordinates are position/(extent−1), sine and cosine of 2π times that value.
Joint query features are real/imag soft symbol, resource coordinates, codeword
slot coordinates, two role features, four-user one-hot and `log1p(r)` (15 total).
Channel coupling is resource-averaged squared normalized Hermitian overlap;
each channel norm is floored at 1e-12 before their product, as in the reference.
Pure-AS codeword slot j maps to actual resource j for every user. Prefix order
is subband-major; no resource permutation is introduced.

| Tensor | Shape / ordering |
| --- | --- |
| DL/UL inputs and receiver UL estimate | `[B,4,32,256]`, complex64 |
| Noisy received AS | `[B,32,256]` |
| Recovery and reliability | `[B,256,4]` |
| Memory | `[B,256,128]` |
| Joint context | `[B,1024,128]`, user-major then symbol position |
| Corrected codeword | `[B,4,256]`, complex |
| FC input | `[B*4,512]`, **all 256 real then all 256 imaginary** |
| FC/spatial tokens | `[B,4,512,128]`, row-major 64-delay-patch x 8-angle-patch |
| Spatial cross queries/keys | `[B,2048,128]` / `[B,1024,128]` |
| Final patch projection | `[B*4,64,8,2,4,4]` |
| Full angular-delay output | `[B,4,2,256,32]` after explicit patch-axis permutation |
| Frequency-domain H_hat | `[B,4,32,256]` |
| Deployed normalized RZF | `[B,32,4,256]`, per-subcarrier power 1 |

True H_dl is used by the original UE and the task loss, never as a BS neural
input. True emitted symbols are detached and returned only **after** H_hat and
W exist, for same-forward offline error diagnostics. No noise-free Y, true
noise variance or live transmitter power is passed to the decoder.

## Matched control and initialization

`plain_swin256` is recovery → the same FC/spatial position/four Swin/output →
full256 inverse transform → RZF. It does not construct or invoke joint memory,
joint self/cross attention, codeword residual or spatial cross attention.

Both arms construct the original encoder/condition modules, then common BS
modules in the same order. Joint-only initialization occurs afterward in a
local CPU RNG fork; it cannot perturb common weights or subsequent RNG state.
Both arms are joint UE+BS training. The pinned robust manifest identity supplies
only encoder, condition embedding and singleton `running_pwr`; explicit key,
shape and dtype checks precede strict state loading. No original BS is retained.
The new FC/Swin are random, not initialized from an existing Swin checkpoint.
Additional Linear weights use std .02 except codeword/spatial-cross output
weights std .001; biases zero. No scalar residual gates.

Static parameter derivation (not a locally executed model count):

| Component | Plain | Joint |
| --- | ---: | ---: |
| Original UE + condition, including inactive FDMA storage | 10,877,090 | 10,877,090 |
| Common FC alone: 512×65536 + 65536 | 33,619,968 | 33,619,968 |
| Complete new BS | 34,422,944 | 34,933,038 |
| Total stored parameters | 45,300,034 | 45,810,128 |

Joint-only modules add 510,094 parameters. Four unchanged Swin blocks total
797,184 parameters (their relative bias is a full four-head 16x16 table per
block). Actual instance counts must match these expectations in CUDA preflight;
the evaluator also exports measured full/encoder+condition/component counts.
This is not a lightweight or equal-parameter comparison, nor a latency claim.

## Training, memory and strict recovery

Two arms x seeds20261030/20261031, 20,000 updates each. B48, DL25, FB uniformly
0–25, task-only H_hat→RZF(1e-3)→per-subcarrier normalization→sum rate. AdamW:
BS peak1e-4 / UE peak2e-5, weight decay1e-5, clip.5, OneCycle warmup10%. Fixed
64-channel validation at FB0/5/10/15/20/25 every500, one mean_all-selected best.
Random new BS does not compete as step0. Data uses the already accepted explicit
ReplayableChannelData protocol, identical split/global-step seeds across arms.

The existing trainer is reused through model/data/optimizer factories, with
`init_checkpoint=None`. New configuration serialization includes architecture,
source identity, full recipe, seed and channel protocol; no legacy fingerprint
or default entry is changed. Resume strictly checks arm, source, full config,
fingerprint, model, optimizer groups/state, scheduler step and RNG. A private
read-only group-policy adapter reuses the existing joint UE/BS helper; its
serialization remains the real new config and it is never given to the model
or trainer. Initial source migration is not repeated on resume.

First500 pauses belong to the same20k schedule and automatically resume through
20000. Completed phases are reused only with unchanged evidence hashes. This
does not claim that complete training resumes are bitwise identical to an
uninterrupted run; actual real-checkpoint resume is still a server verification.

Global attention uses SDPA, query chunks64 with **every key/value retained**,
on-demand chunk biases, nonreentrant checkpoints at layer and per-query
projection/attention boundaries. Spatial attention and each Swin block are also
checkpointed. No local-user restriction, context truncation or microbatching.
Window attention remains the audited existing non-shifted implementation.

For context, dense B48/four-head/1024x1024 float32 attention alone is768MiB;
one query64 all-key mask is48MiB. These are tensor-size illustrations, not a
prediction of total runtime peak. The required preflight measures actual B48
forward/backward plus a **disposable** AdamW update for each arm, including
optimizer state and acceptance-check overhead. It saves no weights and adds no
updates to any20k experiment. Memory OOM/nonfinite/check failures save evidence
and stop; no precision downgrade, smaller batch or structural fallback occurs.
The preflight explicitly uses the same `matmul_precision=high` as main_train and
records before/after backend settings; it is not changed in response to failure.

## Evaluation and interpretation

Independent testseed20261120, 200 batches of8, six FB points, DL25. Per training
seed: plain, joint, old ASSpatial joint, old Full control, original Swin and
completed corresponding-seed adapted Swin. All share actual replayable channel
batches. Old controls are loaded, never trained. Adapted Swin can legitimately
have beststep0, but its latest must show20k completed and step0 state must equal
the pinned original source. New random-BS best must be positive.

Primary difference is Joint−Plain. Also report Joint−ASSpatial, Joint−adapted
Swin, Joint−original Swin and Joint−Full. Per-SNR paired differences are formed
by sample_index; low0/5/10 and all-six averages are first formed **within each
channel**, then sample and batch(8) SE/95%CI are computed. Two training seeds are
never pooled into3200 samples. Both interval types and all negative results
remain visible. Fixed-checkpoint sampling CIs do not establish training-seed
stability. Common feedback RNG is not common absolute noise or UE energy.

Engineering continuation criterion, **reference=Plain only**: low mean>=.5,
low batch-CI lower>0, 25dB batch-CI lower>=−.1. Passing does not imply superiority
to adapted Swin or journal acceptance; those gaps are separately reported.
Only the matched Plain contrast isolates the added joint chain. The comparison
with historical ASSpatial changes full-delay output and capacity too. Soft
codeword errors use each model's own UE representation, not a common CSI target.
Raw representation NMSE remains a secondary diagnostic. The old precoder_power
column sums all subcarriers (nominal256), not per-subcarrier power1.

## Execution and output

From the 03 repository in the existing CUDA environment:

```bash
python -B run_joint_swin_batch.py --dry-run
python -B -u run_joint_swin_batch.py --run
```

For an interrupted batch only, use `--run --resume`. No old model/RNG test suite
is repeated: only this decoder's new small numeric tests and B48 preflight run.
The controller checks required old controls before starting any long training.
It uses exact configured checkpoint paths, not discovery of alternative weights.

Training root: `runs_as_screen/proposal_joint_swin256_v1/`.
Evaluation root: `runs_as_screen/evaluation/k4_joint_swin256_v1/`.
`last_batch.json` names one exact batch directory; `batch_summary.json` records
stages, source hashes, actual checks/parameters/peak memory, best/latest hashes,
per-seed paired conclusions and return_files. New attempts use exclusive paths.

Each evaluation exports36-condition summary, compressed per-channel records,
40 paired estimands plus compressed paired rows, parameters/component counts,
same-forward recovered/corrected errors, runtime shapes and manifests. Verified
CSV-only reporting exports both-seed full-six-point curves and paired CI figures
(vectorPDF and PNG), bilingual internal result drafts and comparison_summary.json.
The small verified return bundle excludes weights and compressed raw samples;
these remain on the server. No automatic manuscript changes or result Git push.

Exact compact terminal return command, without wildcard selection or downloads:

```bash
python -B run_joint_swin_batch.py --summary
```

It prints status, four training selections/hashes, actual parameters/peak memory,
all six rates and paired intervals by training seed, plus exact return-file paths.
On failure it prints the failed stage/error and failure-bundle location instead
of inventing outcomes. On an unstarted checkout it says `not_started`.
