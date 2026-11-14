# K8 matched-budget multi-SNR receiver adaptation

The completed independent calibration batch changed receiver inputs while
leaving the BS decoder frozen. Its moment model improved several soft/CSI
error measures, but the frozen decoder's sum rate declined. That is a
zero-shot receiver-interface result, not a test of decoder adaptation.
This new experiment trains the decoder for that interface while retaining
an equally trained original-receiver control.

The preceding public report was read at exact commit
`266ef14e014bc66f23c4d745bab8622bb08ee796` (export `51b6e9f44c57`).
Its ten artifact identities and nine payload SHA256 values matched. An
independent recomputation of 1536 calibration records reproduced every
locked coefficient, and 9216 validation records reproduced 48 summary rows
and 4168 comparison rows (maximum absolute differences below 9e-15).
Against the original receiver, the moment receiver's six-node mean sum-rate
change was -0.045314 for FDMA (batch95% CI [-0.053979,-0.036648]) and
-0.135565 for Hybrid ([-0.151664,-0.119466]). Hybrid soft NMSE improved
about14.1% and complete-CSI NMSE about11.1%, while rate declined. These
fixed-model observations motivate the matched adaptation test; they do not
establish input distribution shift as the unique cause or guarantee a gain
after training. The public audit did not load real checkpoints or regenerate
CUDA channel tensors.

## Fixed study design

| Item | Prespecified value |
|---|---|
| Namespace/output | `twc_v1_k8_ul_adapt_v1`, `runs/twc_v1_k8_ul_adapt_v1/` |
| Physical arms | FDMA24 and Hybrid G4 key4/shared40 |
| Train methods | Original receiver continuation; locked-moment receiver adaptation |
| Trajectories | Two physical arms × two methods = four |
| New BS updates | 10000 per trajectory;40000 total |
| Source weights | Same pinned rate5000 source decoder within each physical arm |
| Fresh optimizer | AdamW, LR1e-5, weight decay1e-4, constant schedule, grad clip1 |
| Effective batch | Microbatch4 × accumulation8 =32 complete8UE MU/update |
| Frozen | UE encoder, CE, historical power prior/CE table, calibration model |
| Training target | Negative complete-MU downlink sum rate through original RZF |
| Precoder/downlink | Fixed RZF alpha=.03; downlink25dB; per-tone total power1 |
| Training/evaluation SNR | 0 / 5 / 10 / 15 / 20 / 25 dB, all retained |
| Monitor |128MU/six points at steps0/100/1000/5000/10000; no selection |
| Final validation |1024 fresh paired MU per arm/method/phase/gamma |
| Test | Sealed |
| Delivery | Checked scalar CSV/JSON directory; explicit Git publication; no ZIP |

The budget of10000 new updates per trajectory was explicitly chosen by the
user. Previous CE5000, codec20000, rate5000, W and diagnostic outputs remain
read-only. No old failed run is resumed. The source optimizer is not continued;
each trajectory starts a fresh optimizer from the same source decoder bytes
within its physical arm. Extra optimizer updates belong only to this new
namespace, and only BS decoder parameters receive gradients.

The source network already accepts gamma as a conditioning input and had
multi-SNR training. This study does not claim to add SNR awareness for the
first time. It adapts one conditional decoder per trajectory to its prescribed
receiver; it does not train six separate models or choose a model per SNR.

For1-based training microbatch index `i` and within-microbatch position `j`,
the gamma is `GAMMAS[(4*(i-1)+j)%6]`. Every three updates contain exactly16MU
at each node. Over10000 updates each node receives53333 or53334MU per
trajectory, recorded by checkpoint counters. Methods and arms use the same
channel addresses, noise/pilot streams and gamma schedule. The schedule is
fixed, not adjusted in response to validation. This verifies six-node
adaptation, not generalization at unseen continuous SNR values.

## Locked correction and data roles

The moment receiver uses the already fitted common calibration model at
`runs/twc_v1_k8_ul_calibration_v1/calibration/attempt_a0dbcecec7b3/model.json`.
Its SHA256 is
`d0477d049f5808d9f9a34c0792767edc5fcdcbe8aa89951b5ff5a91b6d243be1`;
the calibration manifest SHA256 is
`0e86e2a78fcaca74438d194c1c4895a273c3316eb40f65ea832fbc74b742fe48`.
Both pins and the original production calibration verifier are required.
The model is not refitted, interpolated to undeclared nodes or selected using
the new validation set. Receiver deployment inputs still contain no true UL,
true DL, clean symbols or current-frame transmitted power. Truth appears in
the UE's normal source input, simulator and offline/loss targets only.

The new data namespace uses seed20261124; source training seeds do not change.
Training uses `codec_train` indices1–80000. Monitoring uses validation
indices0–31; final validation uses indices2048–2303, in the new namespace.
These roles/addresses are independent of the completed calibration and its
validation. The original K8/Nt32/Nsc256,256RE/64pilot, per-UE Ep16+Ed48,
rank8 Fourier-LS, source CE network and physical grouping stay unchanged.

The original receiver is the unchanged analytical control. The moment
receiver uses the locked corrected channel mean and PSD uncertainty in both
recovery and the decoder's corresponding channel input. Receiver and UE/CE
calculation occur without gradients; decoder output passes through the
original differentiable RZF/rate loss. There is no direct W head or learned
alpha in this experiment.

## Sequence, checkpoints and comparison

The controller preregisters recipe/core/source/calibration identities before
work. A source preflight verifies all four initializations, including equal
decoder hashes within each arm. It then runs original-FDMA100→10000,
moment-FDMA100→10000, original-Hybrid100→10000 and
moment-Hybrid100→10000. Shared GPU admission is checked before work;
no other processes are terminated.

Saving uses the corrected atomic Torch serializer. Checkpoints include fresh
optimizer/scheduler state, complete RNG state, microbatch cursor and gamma
counts. The immutable100-update receipt remains after latest reaches10000;
the controller never mistakes the later checkpoint for a historical100-step
payload. Explicit resume requires unchanged identities, complete saved update
boundaries and verified evidence. Failure preserves earlier attempts and
does not delete locks, reset source files or silently restart a trajectory.

All four exact10000-update terminals must be accepted before final data is
generated. Source/zero-shot and adapted terminals are then replayed on the
same new final inputs for each arm/method. This separates extra-training
gain from the receiver/interface change. Fixed terminal weights are used;
monitor scores never select checkpoints or enlarge the budget.

The report retains soft reconstruction/error calibration, complete CSI NMSE,
desired signal/interference, user rates and sum rate for every gamma. Paired
contrasts include each method's final-minus-source gain, the final method
gap, source method gap, phase-specific Hybrid-minus-FDMA gap and differences
in adaptation gains. Negative results remain. Gamma points average within
each complete MU before sample and batch4 uncertainty is calculated.
Intervals condition on one source training seed and a fixed calibrator;
calibration-refit and multiple-training-seed uncertainty are not measured.

## Old-server launch after merging the PR

The intended host is `r50037942@w00611266-ELITE`; project path remains
`/home/r50037942/project/03_PRECODER`. It already has the pinned source runs
and completed calibration directory. Do not copy proxy settings or old
credential files to launch this batch.

```bash
conda activate precoder
bash <<'START_UL_ADAPT' || printf '%s\n' '启动停止，请回传错误。'
set -Ee -o pipefail
test "$(id -un)" = r50037942
cd -P /home/r50037942/project/03_PRECODER
test "$(git branch --show-current)" = main
git pull --ff-only origin main
bash tools/start_twc_v1_k8_ul_adapt_old.sh
START_UL_ADAPT
```

The operational launcher preserves unrelated dirty work, checks for active03
Python processes/existing output or tmux sessions, runs the CPU checks and
starts `k8_ul_adapt` with
`--allow-shared-gpu --min-free-mib12000 --max-gpu-util80`.
The actual CLI spells those separate arguments with spaces. It never resets,
cleans, kills processes, overwrites old results or starts an implicit resume.
Training and evaluation use local packages, weights and generated channels;
the proxy is needed only for code synchronization and explicit result upload.

Read the progress summary in another old-server terminal:

```bash
conda activate precoder
cd -P /home/r50037942/project/03_PRECODER
"$CONDA_PREFIX/bin/python" -B run_twc_v1_k8_ul_adapt.py --summary
```

Each trajectory reports its saved complete update boundary, measured ETA and
gamma counts. ETA is a running estimate under the current shared-GPU load,
not an elapsed-time guarantee. Local Git fixture tests deliberately print
rejected pushes/missing remotes under `/tmp`; their final unittest `OK`
determines success, and those fixtures never upload to GitHub.

When `completed | stage=directory_export` appears, explicitly publish:

```bash
"$CONDA_PREFIX/bin/python" -B publish_twc_v1_k8_ul_adapt.py --publish
```

The summary also prints an exact `--export-dir` path. Return the result
branch and commit for review, without merging that result branch into main.
Only controlled scalars, content hashes, redacted provenance and the small
locked calibration coefficients are published. No neural weights, channel
arrays, credentials, proxy files or full environment dump enter the handoff.
The original HEAD, index and dirty files are preserved by Git plumbing.

Removing the exact temporary result ref later does not guarantee history or
cache erasure. Local experiment evidence remains. Deletion requires the
namespace-specific branch and exact expected commit lease.

If the batch fails, preserve its summary, exact child log and all attempts.
After the old controller exits and the cause is addressed, only this batch
can use the following explicit resume with the unchanged recipe:

```bash
"$CONDA_PREFIX/bin/python" -B -u run_twc_v1_k8_ul_adapt.py \
  --run --resume --allow-shared-gpu --min-free-mib 12000 --max-gpu-util 80
```

## Verification boundary

Cloud checks use real CPU tensors, actual small gradient/optimizer updates,
Torch save/load, interrupted-versus-uninterrupted continuation and local Git
fixtures. Those fixtures are not scientific training and cannot pass the
production terminal/evaluation verifier. Source weights, CUDA/Sionna and
full model training/performance are checked only by the user's SSH batch.
No real server training has been started from the cloud. The54 historical
scientific files remain byte-identical and their pins are not relaxed.
