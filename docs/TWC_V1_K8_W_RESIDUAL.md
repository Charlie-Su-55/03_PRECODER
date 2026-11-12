# K8: fair rate continuation versus a residual-W Transformer

This study tests whether a different BS precoding parameterization improves
multiuser interference management after the completed BS-rate experiment.
For each physical arm, both new methods start from the **same pinned rate5000
terminal decoder** and receive the same additional data and update budget.
The unchanged rate5000 source is replayed as a third evaluation policy.

The new namespace is `twc_v1_k8_w_residual_v1`; all outputs belong to
`runs/twc_v1_k8_w_residual_v1/`. Earlier source, rate, alpha, failed batches,
paper files, manual plots and the 08 reference project remain unchanged.

## The controlled comparison

| Policy | Trainable state | New updates per physical arm |
|---|---|---:|
| `rate_source` | None; unchanged rate5000 terminal | 0 |
| `rzf_continue` | Original BS CSI decoder | 5000 |
| `w_residual` | The same BS decoder plus a residual-W Transformer | 5000 |

The physical arms remain FDMA24 and Hybrid G4 key4/shared40. There are four
training runs and six final methods. The continuation control is necessary:
comparing a newly trained residual head only with the frozen source would
confound the head with additional training. Equal updates and paired inputs
control this training budget; the residual method still has extra parameters
and compute, so this is not a capacity-matched comparison.

Both trained methods retain the original differentiable RZF with alpha .03.
Let `W0` be its total-power-normalized output from the estimated CSI. The
residual method uses

```
W = normalize_per_tone(W0 + DeltaW / 16)
```

where each tone has total matrix Frobenius power one. Columns are not
individually normalized. Here `DeltaW` means the raw projection; exported
`out['delta_W']` already includes the factor 1/16 and must not be divided again.
The final residual projection starts at exactly
zero, so its initial precoder agrees with the source RZF up to floating-point
normalization tolerance. The head uses a shared fixed initialization seed
20261112 in both physical arms.

This is a residual-W parameterization, **not a pure direct-W system**. RZF
continues to participate in the forward path and gradient. A gain would show
that this residual parameterization and its extra capacity help under this
budget. It would not establish a larger mathematical output space than an
unconstrained effective-CSI decoder or prove that a pure direct-W Transformer
is necessary.

## Head and information boundary

At each of the 256 tones, a Transformer attends jointly to all eight user
tokens. Each token has 76 real inputs: 32 complex estimated channel
coefficients, public feedback gamma, eight fixed user-identity coordinates
and three fixed frequency coordinates. The head has width64, two attention
layers, four heads, feedforward width128, dropout0 and 76160 parameters. Its
last linear map produces 32 complex residual coefficients per user. The head
does not consume true DL/UL channels, current-frame transmit power, clean
feedback symbols, evaluation labels or an oracle precoder.

The original BS decoder still consumes the paid pilot/key/shared observations
and original recovery statistics. Its weights can change in **both** methods.
UE encoder parameters and buffers, CE parameters and buffers, calibration
and historical receiver prior remain frozen in inference mode; new CE, UE and
EMA updates are all zero. Historical EMA commits remain20000. The model input
boundary is checked separately from the loss: true DL CSI is allowed only in
the training objective and offline metrics.

The objective is negative complete-MU sum-rate, summed over eight users and
averaged over tones and the microbatch. Eight microbatch losses accumulate
into one effective32-MU update. Both methods use fresh AdamW, lr1e-5,
weight decay1e-4, betas(.9,.999), eps1e-8, gradient norm clip1 and a constant
LambdaLR scheduler. No optimizer state is continued from the source. There
is no CSI auxiliary loss, alpha fitting or monitor checkpoint selection.

## Sources, budget and independent samples

The original aligned source remains at
`runs/twc_v1_k8_icc_aligned_savefix_v1/`, trained with commit
`9d345b2385d99eec1d047bd9c602638b568c84f8`. The new starting BS state is the
existing rate5000 terminal at `runs/twc_v1_k8_icc_rate_v1/`, trained with commit
`957b730d56f07fae9c93cb2611f5e460f7566e67`. Its two checkpoint SHA256 pins are:

| Arm | Pinned rate5000 checkpoint SHA256 |
|---|---|
| FDMA | `bbe98a72295c5b2985a1c41b59e010fc8d9c01cec830ba85cddc52c116fef96d` |
| Hybrid G4 | `26494a2a76100b432f301b6073958dc20f11fbe78fefa2f43676f6e037b5ad30` |

The configuration inherits the existing source/terminal/CE-calibration pins.
Strict loading checks actual bytes, complete receipts, source core, recipe,
data protocol, counters and frozen state. Pins in a config do not replace
actual server checkpoint verification. The 26 earlier core files remain
unchanged; this experiment adds seven production files.

| Item | Value |
|---|---|
| New updates per arm and method | 5000; four runs total |
| Microbatch / accumulation | 4 / 8 |
| Additional samples per run | 160000 MU / 1280000 UE inputs |
| Unique paired training samples | 160000 MU across all four runs |
| Train channel addresses | `codec_train` 1–40000 |
| Monitor channel addresses | `validation` 0–31; 128MU per gamma |
| Monitor boundaries | 0 / 100 / 1000 / 2500 / 5000 |
| Final channel addresses | `validation` 2048–2303; 1024MU per gamma |
| Final policies | Source, continuation and residual, for both arms |
| Feedback gamma | 0 / 5 / 10 / 15 / 20 / 25 dB; retain all six |
| Selection | Fixed final5000 weights; test stays sealed |

Seed20261121 is retained, with the new independent data/noise namespace.
Metadata tests compare actual Sionna seed sets, for CPU and `cuda:0`, against
the used training and validation addresses of all three prior namespaces.
Training channels, feedback standard noise, pilot polarity and sampled gamma
are paired across both methods and both arms. Final inputs are also paired
across all six methods. Replayed source rates can differ from an earlier
table because this final namespace is new.

Training pairing evidence checks the last eight actual microbatches at each
100-update checkpoint boundary: channel, noise, pilot, gamma and addresses.
The four jobs' `progress.jsonl` files enter the small result package for
review; this does not export fingerprints for all 40000 training microbatches.

K8/Nt32/Nsc256, 256 occupied RE including64 pilots, per-UE energy64=Ep16+Ed48,
CDL-C, the noise rule `N0=.25*10**(-gamma/10)`, original statistical recovery,
FP32/complex64 without AMP/TF32 and DL25dB are unchanged.

## Readout and limits

The primary comparison is `w_residual - rzf_continue`, within each physical
arm, averaged over all six gamma points **inside each paired MU first**.
The interaction compares Hybrid's residual gain with FDMA's residual gain.
Also retain each policy's Hybrid−FDMA comparison, all six individual points,
low-three mean, interference, all eight user rates and minimum-user rate.
Preserve negative differences and failures without pruning gamma points.

Positive residual gain would support this next parameterization under equal
updates. A positive interaction alone does not mean Hybrid beats FDMA. If
the continuation control improves similarly, additional optimization rather
than the head explains the source-relative gain. If neither improves, this
experiment does not establish that interference management is exhausted.

Intervals are conditional validation-sampling intervals for fixed terminal
models and one training seed; batch4 intervals do not measure training-seed
variation. Monitor curves diagnose optimization and never select weights.
The alpha study is historical context, not a selection stage for this study.

## SSH startup on the existing server after review and merge

Cloud work implements and tests code only. Actual source checkpoint loading,
CUDA training gradients/memory/throughput and scientific performance require
the SSH machine. Tiny CPU tests are not model-performance evidence.

Run this study on the existing SSH server using its existing `precoder`
environment and `~/project/03_PRECODER` checkout. The original aligned, rate
and alpha directories and their complete receipts remain in place. Old
absolute paths participate in strict identity checks; do not move or rewrite
checkpoints, receipts or pins. No migration or new environment is required.

The following external Bash preserves unrelated unstaged files or stops on
conflicts; it never resets, cleans, deletes locks or terminates processes.
In particular, keep the server's `.gitignore` edits, deleted `old/` files,
untracked handoff documents, paper material and historical results. Git
synchronization stops if an incoming change conflicts with local work.

```bash
bash <<'W_RESIDUAL' || printf '%s\n' 'Residual-W launch stopped. Read the error above; the terminal stays open.'
set -Ee -o pipefail
trap 'task_rc=$?; printf "Stopped at script line %s (exit %s).\n" "$LINENO" "$task_rc" >&2; exit "$task_rc"' ERR
fail() { printf 'STOP: %s\n' "$*" >&2; exit 1; }
cd -P ~/project/03_PRECODER
test "$PWD" = /home/r50037942/project/03_PRECODER || fail 'The real project path changed.'
task_conda_base="$("${CONDA_EXE:-conda}" info --base)"
test -r "$task_conda_base/etc/profile.d/conda.sh" || fail 'Cannot locate conda.sh.'
source "$task_conda_base/etc/profile.d/conda.sh"
conda activate precoder
task_python="$(python -B -c 'import sys; print(sys.executable)')"
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
test -f configs/twc_v1_k8_w_residual_config.py || fail 'The residual-W control is not yet on main.'
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$task_python" -B -m unittest discover -s tests -p 'test_twc_v1_k8_w_residual_*.py' -v
"$task_python" -B -S run_twc_v1_k8_w_residual.py --dry-run
test ! -e runs/twc_v1_k8_w_residual_v1 && test ! -L runs/twc_v1_k8_w_residual_v1 || fail 'Residual output already exists; inspect its summary first.'
command -v tmux >/dev/null || fail 'tmux is not installed.'
if tmux has-session -t '=k8_w_residual' 2>/dev/null; then
  fail 'The k8_w_residual tmux session already exists.'
fi
mkdir -p terminal_logs
task_log="$PWD/terminal_logs/k8_w_residual_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
tmux new-session -d -s k8_w_residual \
  "cd '$PWD' && '$task_python' -B -u run_twc_v1_k8_w_residual.py --run --allow-shared-gpu --min-free-mib 12000 --max-gpu-util 80 > '$task_log' 2>&1"
printf 'Log: %s\n' "$task_log"
W_RESIDUAL
```

The first stage checks actual source files and frozen state before training.
Each run saves a zero-update checkpoint and an initial monitor, then trains
to100 and continues its own new checkpoint to5000. The 100-update stage is
part of the declared5000 budget. It is not an extra100 updates or a resume of
the earlier rate/aligned experiment. GPU policy is a startup admission check,
not a continuous quota; failures do not reduce budget or model silently.

Read and return this complete compact summary:

```bash
conda activate precoder
cd -P ~/project/03_PRECODER
"$CONDA_PREFIX/bin/python" -B run_twc_v1_k8_w_residual.py --summary
```

After its previous process has exited, an interrupted **new residual-W
queue only** can resume its own validated saved boundaries:

```bash
"$CONDA_PREFIX/bin/python" -B -u run_twc_v1_k8_w_residual.py --run --resume --allow-shared-gpu --min-free-mib 12000 --max-gpu-util 80
```

No network operation is required for summary or resume. Use `--summary
--details` if detailed evidence is needed. Full checkpoint/CUDA training and
scientific results remain unrun in cloud; delivered CPU check counts and the
actual commit are reported separately with the PR.
