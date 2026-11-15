# UE + BS joint rate training with archived BS-only controls

The previous rate experiments trained only the BS. This experiment enables
both the unchanged UE encoder and BS decoder to receive the complete-MU
sum-rate gradient through the existing physical MAC, estimated-channel
receiver and fixed-alpha RZF. It tests joint task learning before adding UE
SNR inputs, projections, extra experiment arms or a new precoding head.

## Fixed design and comparison limits

| Item | Prespecified value |
|---|---|
| Namespace/output | `twc_v1_k8_ue_bs_rate_v1`, `runs/twc_v1_k8_ue_bs_rate_v1/` |
| New training | FDMA and Hybrid G4, 10000 UE+BS updates each |
| Source | Same pinned rate5000 source used by the completed BS-only batch |
| Archived controls | Exact original-receiver BS-only10000 terminals from `twc_v1_k8_ul_adapt_v1` |
| Input/model changes | None; UE still takes its own full CSI only |
| Optimizer | Fresh AdamW, LR1e-5, weight decay1e-4, clip1, constant schedule |
| Effective batch | Microbatch4 × accumulation8 =32 complete8UE MU/update |
| Frozen | CE parameters/buffers and independently calibrated CE-error table |
| Updated statistics | Historical power prior; one causal commit per successful update |
| Target | Negative complete-MU downlink sum rate through RZF alpha=.03, DL25dB |
| Physical protocol | Original K8/Nt32/Nsc256;256RE including64pilot; Ep16+Ed48 perUE |
| Gamma | All six0/5/10/15/20/25dB nodes, balanced within every three updates |
| Selection | Fixed10000 terminal; no monitor checkpoint selection; test sealed |
| Delivery | Verified scalar CSV/JSON directory; explicit Git upload; no ZIP |

Each arm starts from the exact source encoder/decoder bytes and a fresh
optimizer; it does not continue the archived control's optimizer or add
updates to old experiment outputs. The controls are not retrained. Their
actual checkpoint, manifest, initialization, progress and terminal receipt
hashes are pinned to the audited result commit
`391b68d08db8b2639f12686bdbbad6147fe7f34b` (running code
`2986b8502a876b7926dcb6d121194ca80cc4b115`).

The archived BS-only controls used a frozen power prior. A changing UE
requires a causally updated prior in this new experiment. The final
comparison therefore tests a joint-training pipeline with necessary prior
adaptation against the archived frozen-UE/frozen-prior pipeline. A gain must
not be attributed to UE gradients alone. Source, update count and actual
training channel/noise/pilot/gamma inputs match.

## Differentiation and causal statistics

The training graph is UE(H_DL) → active-coordinate joint energy normalization
→ simulated physical MAC → unchanged estimated-only receiver → BS decoder
→ original differentiable RZF → complete-MU rate objective. True H_UL appears
only in the physical simulator; true H_DL is the UE source and loss/scoring
label. The receiver receives estimated channel, paid pilot observations,
public gamma and historical statistics, never true UL or current-frame power.
Only the pilot-only CE calculation is under `no_grad`; the data waveform and
recovery path preserve UE gradients. CE remains in eval mode with no gradients.

The source EMA counter is retained. All eight microbatches of one effective
update read the same prior. Their detached, actually transmitted normalized
powers are accumulated; only after a finite successful optimizer/scheduler
step does the receiver commit once. Monitoring, validation, failed gradients,
checkpoint retry and file verification do not commit power. Checkpoints save
the complete UE+BS state, statistics/EMA state, optimizer, scheduler, RNG,
source counter, saved cursor and complete actual input trace.

## Replayed training and fresh validation

Training intentionally replays the completed control's namespace and seed
(`twc_v1_k8_ul_adapt_v1`,20261124), `codec_train` indices1–80000, its actual
noise/pilot streams and balanced gamma schedule. The trace uses the old
explicit input-trace format, because it hashes the same physical inputs; the
new experiment output, core and resume identity remain separate. The full
common trace endpoint must equal the pinned control endpoint. Waveforms,
receivedY, estimated soft symbols and updated priors may differ under the
new UE and are not falsely required to match.

Monitoring and final validation use new namespace
`twc_v1_k8_ue_bs_rate_v1`, seed20261125. Monitor addresses are validation0–31;
final validation addresses are4096–4351. Final samples are generated only
after both exact new joint terminals and both archived controls pass strict
checks. Source, BS-only and joint models then replay the same new final
channels/noise/pilots/gamma:36864 complete-MU scalar rows in total.

Every model/statistics instance stays frozen within final evaluation. UE and
power-prior states may differ between source/control and joint models; CE
and its error table must remain common. Reports retain all gamma points,
negative results, individual user rates, CSI/soft errors, interference,
desired signal, energy and calibration measures. Cross-gamma scope averages
are formed within each MU before sample and batch4 intervals. Intervals
condition on one source training seed and fixed trained models.

## Old-server launch after merging

Run on the existing old server `r50037942@w00611266-ELITE`, where the pinned
source, calibration and completed controls remain. Keep all prior run trees.

```bash
conda activate precoder
bash <<'START_UE_BS_RATE' || printf '%s\n' '启动停止，请回传上面的错误。'
set -Ee -o pipefail
test "$(id -un)" = r50037942
cd -P /home/r50037942/project/03_PRECODER
test "$(git branch --show-current)" = main
git pull --ff-only origin main
bash tools/start_twc_v1_k8_ue_bs_rate_old.sh
START_UE_BS_RATE
```

The launcher preserves unrelated dirty work, runs the new CPU suite and
stdlib-only dry-run, refuses an existing output/session or active03 Python
process, and starts tmux `k8_ue_bs_rate` with
`--allow-shared-gpu --min-free-mib 12000 --max-gpu-util 80`.
It never resets, cleans, kills processes, rewrites a source or silently resumes.

```bash
conda activate precoder
cd -P /home/r50037942/project/03_PRECODER
"$CONDA_PREFIX/bin/python" -B run_twc_v1_k8_ue_bs_rate.py --summary
```

Step0 includes initial monitoring before training. Until the first saved100
boundary, ETA is not yet measured. The100 updates are part of the10000
budget. The immutable100 receipt survives overwriting latest at the terminal.
If a failure occurs, retain its attempts and exact child log. After the
cause is resolved and the process exits, explicit resume is:

```bash
"$CONDA_PREFIX/bin/python" -B -u run_twc_v1_k8_ue_bs_rate.py \
  --run --resume --allow-shared-gpu --min-free-mib 12000 --max-gpu-util 80
```

When summary shows `completed | stage=directory_export`, upload explicitly:

```bash
"$CONDA_PREFIX/bin/python" -B publish_twc_v1_k8_ue_bs_rate.py --publish
```

The summary prints the exact `--export-dir` command as well. Return the result
branch and commit; do not merge the result branch into main. Publishing uses
Git plumbing, preserving source HEAD, index and dirty files. Only checked
scalars and redacted provenance enter a namespace-specific result branch.
Exact expected-commit lease deletion is supported; deletion does not guarantee
history/cache erasure. Local weights and scientific evidence stay on the server.

## Cleanup and the deferred SNR questions

Only the separately audited five legacy entrypoints with no running callers
or source-SHA dependencies are removed from Git. Historical manifests retain
their earlier tracked-change records. All61 previous scientific core files, their useful tests, papers,
manual figures and historical results remain. The SSH cleanup helper defaults
to a read-only plan and archives only exact matching stale copies on explicit
apply; changed/untracked files are preserved. See
[`CLEANUP_03_OBSOLETE.md`](CLEANUP_03_OBSOLETE.md) for its exact manifest and
commands. The training launcher applies this user-authorized exact cleanup
after CPU checks and before starting tmux; it never touches scientific core
files or results.

UE SNR input can enable a transmission policy to react to public feedback
noise conditions, but it changes the model and should be a subsequent
controlled experiment. This batch leaves the UE input unchanged. The public
feedback gamma is the same protocol parameter for all UEs in a MU
(`N0=.25*10**(-gamma/10)`); UE and BS can share that declared parameter. It is not
the downlink25dB parameter, each user's realized received SNR, or the BS's
post-detection SINR. Local UE/BS knowledge and any control overhead must be
specified before adding an adaptive transmission rule. The BS current
post-detection SINR or true UL must not be provided to the UE without an
explicitly accounted control mechanism. These questions remain for the next
architecture discussion.

## Verification boundary

Cloud verification uses actual CPU tensors, real RZF/rate backpropagation,
UE+BS optimizer steps, causal EMA, real Torch atomic serialization/resume and
local Git publication fixtures. Fixtures cannot pass production checks.
No SSH connection, CUDA/Sionna run, real model loading or scientific training
is performed from the cloud. The new code does not assert a performance gain.
