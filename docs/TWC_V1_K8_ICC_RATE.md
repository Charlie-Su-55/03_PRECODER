# ICC-aligned K8: decoder-only rate objective

This is a new paired experiment, `twc_v1_k8_icc_rate_v1`. It tests whether training
the existing BS network for downlink sum-rate improves interference management
relative to the completed CSI-NMSE experiment. It does not presume that Hybrid
must beat FDMA. The original fixed20k results remain unchanged.

This is the objective-alignment control discussed before a direct-precoder
Transformer: the BS still outputs CSI/effective CSI and uses the original RZF.
The task-oriented feedback-to-precoding motivation is related to
[Deep Learning for Distributed Channel Feedback and Multiuser Precoding in FDD Massive MIMO](https://arxiv.org/abs/2007.06512);
its bit-feedback setting is different from this paid analog feedback system.

## Controlled change

The complete chain remains UE CSI encoder → paid pilot/key/shared physical
feedback → original CE/statistical recovery → original BS CSI decoder → fixed
RZF. Only the **BS decoder parameters** receive new optimizer updates. The
loss changes to negative downlink sum-rate through the existing differentiable
RZF and SINR computation. There is no CSI auxiliary loss, direct-W head,
learned alpha, new experimental arm, per-user rate weight or changed energy
rule in this experiment.

For every complete MU sample and subcarrier, use the same physical true-DL
score as the previous experiment, sum the eight users' rates, average256 tones,
then average the microbatch. Divide each microbatch loss by8 before backward.
The32-MU effective batch receives exactly one optimizer/scheduler update.
The model continues to see only its paid observations, estimated UL,
recovered symbols, public layout/gamma and historical calibration. True DL is
used by the training loss and offline evaluation, not as a BS network input.

The trained CSI output can become a task-oriented effective channel rather
than the minimum-MSE physical estimate. DL NMSE remains a diagnostic; rate and
interference are the research objective. Individual-user rates are retained so
a sum-rate improvement cannot hide its distribution over users.
Predicted-CSI energy and its true-channel energy ratio are also reported.
Although the numeric RZF alpha stays.03, changing the network output's scale
changes regularization relative to its Gram matrix. Such gains cannot be
attributed solely to better physical CSI estimation or interference direction;
no new output-CSI normalization is silently introduced.

## Fixed source, immutable outputs

Source training commit: `9d345b2385d99eec1d047bd9c602638b568c84f8`.
Source files remain under `runs/twc_v1_k8_icc_aligned_savefix_v1/`.
The configuration pins the14 original core file hashes, all three checkpoint
hashes, terminal receipt hashes and CE calibration hash from the actual
completed batch. The server must verify these files and strict model states;
a configuration declaration is not a substitute for checking the weights.

| Source | Fixed update | Checkpoint SHA256 |
|---|---:|---|
| CE |5000|`95c1e786b2036ecab81a865408555069ae3ce699620d1afacd14fc11982db494`|
| FDMA |20000|`68c8672dc0088da2d8cb24cda5491e1858e60ea9ae49a11911bc86fa9eaaa0f8`|
| Hybrid G4 |20000|`e98df0e507626e39817c3245e7d0b6cefa76a3460563aed88d564c4a8081f155`|

Each arm starts from its own fixed20k trained state, not a selected monitor
checkpoint or a new common reinitialization. Its original optimizer is not
continued. The source UE encoder, CE, calibration and historical receiver prior
remain frozen in inference mode, with parameters and buffers checked for
mutation. Their historical EMA count stays20000; **new EMA commits are zero**.
The frozen UE keeps the transmitted representation/energy allocation fixed,
so this isolates how the BS objective exploits the same feedback information.

All new outputs belong to `runs/twc_v1_k8_icc_rate_v1/`. No old checkpoint,
monitor, failure evidence, result CSV, plot or source file is rewritten. No08,
paper or manual plotting change is part of this experiment. New checkpoint
and resume identities include the new core/recipe, original source identity,
frozen state identity and fresh data protocol; mismatches are fatal.

## Declared budget and physical protocol

| Item | Value |
|---|---|
| Arms |FDMA24 and Hybrid G4 key4/shared40|
| New trainable state |BS decoder only|
| New updates per arm |5000|
| Microbatch / accumulation |4 / 8; effective32 MU|
| Additional frames per arm |160000 MU /1280000 UE inputs|
| Actual unique paired train frames |160000 MU, shared across methods|
| Optimizer |fresh AdamW, lr1e-5, weight decay1e-4, betas(.9,.999), eps1e-8|
| Scheduler / clip |constant / gradient norm1|
| CE / UE / EMA new updates |0 /0 /0|
| Arithmetic |FP32/complex64; no AMP/TF32|
| Monitor |128MU at steps0/100/1000/2500/5000; all six gamma|
| Final |1024 paired MU per condition; fixed5000 terminal states|

These are **additional** updates of a separately reported experiment. They do
not retroactively extend or replace the original20k benchmark. FDMA and Hybrid
receive equal additional update/data budgets.

The original physical protocol is unchanged: K8/Nt32/Nsc256,256 occupied RE
including64 pilots, per-UE energy64=Ep16+Ed48, original CDL-C4x8 array, noise
`N0=.25*10**(-gamma/10)`, and per-frame training gamma Uniform[0,25]. RZF remains
alpha.03 with total power1 per tone, evaluated at DL25dB. There is no per-user
column renormalization, alternative alpha selection or new rate constraint.

## New samples, common comparisons

The seed remains20261121 but the data/noise namespace is independently
`twc_v1_k8_icc_rate_v1`. Both arms use the same actual fresh channels, standard
noise and pilot polarity. Source data addresses are never resumed or changed.
Training uses codec_train indices1–40000. Monitor uses validation0–31; final
uses validation2048–2303. The namespaces separate these from the original
experiment. Test is never sampled; there is no alpha-calibration stage.

Evaluate the original fixed source decoder and the new fixed terminal decoder
on the **same new final inputs**, for both arms. The before/after scores may
differ from the uploaded source table because the evaluation namespace is new;
the paired comparison is within this experiment. Report each arm's rate gain,
Hybrid−FDMA before/after, the change in that gap, interference and all eight
user rates. Across-gamma uncertainty must average inside the same complete MU
before calculating sample or batch4 intervals. These intervals describe fixed
models and one training seed, not independent training-seed variation.

The primary readout is the **all-six mean rate interaction**: Hybrid's rate
gain minus FDMA's rate gain. Low-three and individual gamma results are
secondary diagnostics. A positive interaction means Hybrid benefits more;
it does not by itself establish that Hybrid exceeds FDMA after adaptation.

New checkpoints store only decoder parameters plus their fresh AdamW,
scheduler and complete RNG state. A zero-update checkpoint is saved before
monitor0, so a failed initial monitor can be resumed without training an
extra update. Complete checkpoint/hash validation precedes any missing
same-boundary receipt export repair; changed existing evidence is rejected.

## Server startup after review and merge

Cloud preparation only changes code and runs CPU checks. Full-size source
checkpoint loading, source-file protection checks against the actual server,
CUDA gradients/memory/throughput and the research result require SSH execution.
Do not treat a CPU tiny-tensor check as evidence of model performance.

Use the existing `precoder` environment and checkout. Every cloud task already
has an isolated checkout; do not create a Git worktree unless explicitly asked.
The block below uses an external Bash, so failed checks return to the existing
terminal. It preserves unrelated unstaged edits or stops on Git conflicts. No
reset, clean, lock removal or process termination is part of synchronization.

```bash
bash <<'ICC_RATE' || printf '%s\n' 'Rate launch stopped. Read the error above; the terminal stays open.'
set -Ee -o pipefail
trap 'task_rc=$?; printf "Stopped at script line %s (exit %s).\n" "$LINENO" "$task_rc" >&2; exit "$task_rc"' ERR
fail() { printf 'STOP: %s\n' "$*" >&2; exit 1; }
cd -P ~/project/03_PRECODER
task_conda_base="$("${CONDA_EXE:-conda}" info --base)"
test -r "$task_conda_base/etc/profile.d/conda.sh" || fail 'Cannot locate conda.sh.'
source "$task_conda_base/etc/profile.d/conda.sh"
conda activate precoder
task_python="$(python -c 'import sys; print(sys.executable)')"
case "$task_python" in /*) ;; *) fail 'Absolute Python path required.' ;; esac
for task_pid in $(pgrep -u "$(id -u)" -x 'python([0-9.]*)?' || true); do
  task_cwd="$(readlink -f "/proc/$task_pid/cwd" 2>/dev/null || true)"
  case "$task_cwd/" in "$PWD/"*) fail "03 Python process still active: $task_pid" ;; esac
done
test "$(git branch --show-current)" = main || fail 'Expected main.'
test "$(git remote get-url origin)" = https://github.com/Charlie-Su-55/03_PRECODER.git || fail 'Unexpected origin.'
git status --short --branch
git diff --cached --quiet || fail 'Staged changes found; preserve them before synchronizing.'
git fetch origin main
test "$(git rev-list --count origin/main..HEAD)" -eq 0 || fail 'Local-only commits found.'
git pull --ff-only origin main
test -f configs/twc_v1_k8_icc_rate_config.py || fail 'The rate experiment is not yet on main.'
CUDA_VISIBLE_DEVICES='' "$task_python" -B -m unittest discover -s tests -p 'test_twc_v1_k8_icc_rate_*.py' -v
"$task_python" -B -S run_twc_v1_k8_icc_rate.py --dry-run
test ! -e runs/twc_v1_k8_icc_rate_v1 || fail 'Rate output already exists; inspect its summary first.'
command -v tmux >/dev/null || fail 'tmux is not installed.'
if tmux has-session -t '=k8_icc_rate' 2>/dev/null; then
  fail 'The k8_icc_rate tmux session already exists.'
fi
mkdir -p terminal_logs
task_log="$PWD/terminal_logs/k8_icc_rate_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
tmux new-session -d -s k8_icc_rate \
  "cd '$PWD' && '$task_python' -B -u run_twc_v1_k8_icc_rate.py --run --allow-shared-gpu --min-free-mib 12000 --max-gpu-util 80 > '$task_log' 2>&1"
printf 'Log: %s\n' "$task_log"
ICC_RATE
```

Shared GPU admission uses the same requested settings and is a startup
observation rather than a continuous allocation/compute cap. Failures preserve
evidence; they do not silently reduce the model, batch or budget.

Return the complete compact summary after completion or failure:

```bash
conda activate precoder
cd -P ~/project/03_PRECODER
"$CONDA_PREFIX/bin/python" -B run_twc_v1_k8_icc_rate.py --summary
```

Only after the previous rate process has exited, an explicit resume of this
**new rate experiment** uses the same source/recipe identities:

```bash
"$CONDA_PREFIX/bin/python" -B -u run_twc_v1_k8_icc_rate.py --run --resume --allow-shared-gpu --min-free-mib 12000 --max-gpu-util 80
```

No resume command in this document targets the old completed NMSE batch.

## Cloud validation

The integrated ICC suites passed80 tests with0 failures/errors/skips on CPU
(46 original regression checks plus34 new checks). Standard-library-only
discovery passed38 and explicitly skipped42 tensor-dependent checks, with no
errors. The real CPU integration exercised100 small-decoder AdamW updates,
800 microbatches through the original RZF and rate gradient, monitor0/100,
checkpoint serialization and resume; it did not instantiate a full codec or
generate CDL channels. Source, frozen state and optimizer boundaries were
checked, including failed serialization and missing-receipt recovery.

All three CLI dry-runs and the no-start summary passed without creating the
experiment output root. Python3.10 and documented Bash syntax checks passed.
Actual CUDA memory/gradients, full original checkpoint loading, source data
replay and model performance remain **unrun in the cloud**. No SSH connection
or server experiment was started.
