# K8 independent CE-moment calibration and frozen validation

This batch follows the completed fixed-filter propagation diagnostic. That
diagnostic independently matched the actual soft-symbol error decomposition,
but the original reported variance understated error. Its true-UL conditional
score explained most of the gap without changing the receiver. This batch
tests a deployable statistical correction rather than feeding true UL into
the validation receiver.

## Prespecified design

| Item | Value |
|---|---|
| Namespace/output | `twc_v1_k8_ul_calibration_v1`, `runs/twc_v1_k8_ul_calibration_v1/` |
| Physical arms | FDMA24 and Hybrid G4 key4/shared40 |
| Receiver controls per arm | `original`, `reliability_only`, `moment_receiver` |
| Gamma | 0 / 5 / 10 / 15 / 20 / 25 dB; all retained |
| Calibration | 64 batches × micro4 =256 complete8UE MU per gamma |
| Validation | A separate64 batches × micro4 =256 MU per arm/gamma |
| Source | Existing pinned rate5000 codecs, CE and historical power prior |
| Neural/optimizer/UE/CE/BS/EMA updates | All zero |
| Statistical fit | Common CE moments, fit on calibration only and locked before validation |
| RZF/downlink | Original fixed alpha=.03, downlink25dB |
| Test | Sealed |
| Delivery | Local scalar CSV/JSON directory; explicit Git publication; no ZIP |

The statistical fit is learning from calibration data. Zero optimizer or
neural-network updates does not mean no fitting. The validation set never
selects strata, model form, shrinkage, a receiver variant, a checkpoint, an
alpha or an additional training budget.

All old46 scientific core files and their source checkpoints remain pinned.
The eight new production files extend the core to54. The new namespace uses
seed20261123 and separate replay-data roles: `ce_calibration` for fitting and
`validation` for evaluation, both with indices0–63. Their role tags and
role-dependent noise/pilot-polarity seeds are disjoint. The original physical
protocol,256RE/64pilot, per-UE Ep16+Ed48, K8/Nt32/Nsc256 and paired inputs are
unchanged. The new roles do not reuse the completed diagnostic's data address.
Seed20261123 addresses this new calibration/validation data only; it does not
change the training seed of any frozen source checkpoint.

## What is calibrated

The common CE calibration observes only a paid FDMA waveform, because both
arms use the identical pilot support, source CE state and pilot-only CE
mechanism. Validation requires each arm's frozen CE state to match this
calibration identity. It does not fit a different CE model to each physical
arm or use validation labels to tailor the receiver.

The observable feature is the pilot-only U-Net estimate's pilot residual
power divided by the known N0. For each gamma, calibration fixes two feature
quantiles, creating three residual strata. Inside each stratum the fit pools
UEs/frequencies and estimates a complex mean correction along the estimated
channel plus two nonnegative residual powers parallel and perpendicular to
that channel. Parameters use a fixed16-MU shrinkage toward the original mean
and isotropic variance assumptions; PSD and numerical floors are fixed in
the recipe. There is no model-size or hyperparameter search on validation.

For original estimate `h`, direction `u=h/||h||`, and original CE table value
`v`, the statistical model has the form

```
mean = (1 + alpha) h
CE covariance = v [c_perp I + (c_parallel - c_perp) u u^H].
```

The two covariance eigenvalues are positive by construction. This is a small
conditional moment model, not an exact posterior or an information-theoretic
bound. True UL supplies calibration labels and diagnostic scores only.
Deployment functions accept estimated channels, observed pilots/Y, the known
gamma and frozen historical statistics; they have no true-channel argument.

Pilot residuals require care. The old rank8 Fourier design has64 coefficients
per BS antenna (8UE ×rank8) and64 pilot observations, so a training-internal
LS residual is not an independent unbiased noise/CE-error measurement. This
batch treats the pilot-only U-Net residual as a candidate observable feature
whose usefulness must be evaluated on the separate validation role. It does
not hold out or add paid pilots, change the LS basis, or infer true CE error
from an in-sample residual by assumption.

## Receiver controls and measured outputs

- `original` executes the unchanged receiver.
- `reliability_only` retains its exact soft symbols, estimated channel and
  received observation. It recomputes variance/reliability using the locked
  CE moments, isolating the reliability interface's effect.
- `moment_receiver` uses the locked conditional mean and PSD error covariance
  to design the analytical filter and its reliability. The corrected channel
  is also supplied consistently to the frozen decoder. It reduces to the
  original regularized receiver in the zero-bias/isotropic unit-moment limit.

All three controls see the same paid waveform, Y, noise, pilots and historical
prior. No current-frame transmitted power replaces the prior. Every source
network stays in evaluation mode with gradients disabled; hashes before and
after bind frozen codec, CE and prior state. Calibration artifacts are hashed
before validation and checked again afterward. A validation attempt cannot
rewrite, refit or silently replace the calibration model.

The scalar diagnostics cover CE parallel/perpendicular error power, relative
error projections and covariance anisotropy, as well as the shared estimated
Gram's correlation, singular values and condition number. These describe
conditional structure; they do not by themselves prove a unique causal
mechanism. Validation also executes the original full frozen CSI decoder and
fixed-alpha RZF, reporting soft error/calibration, CSI NMSE, desired signal,
interference, per-user rates and sum rate. All six gamma points, low-three
and all-six within-MU summaries, and negative results remain in the export.
Columns prefixed `source_` describe the common original CE input and are
deliberately repeated across receiver controls. The separate
`effective_ce_nmse_linear` scores each control's actual channel input to the
receiver/decoder, including the moment receiver's corrected mean. It is an
offline score, never a deployment input.

Coordinate counts and calibration labels are not independent CI samples.
Gamma views first average within each complete MU; uncertainty is then
computed across MU and across64 physical batches of4. Intervals condition on
the fixed source models and this fitted calibration. They exclude training
seed and calibration-refit uncertainty. Cross-arm differences and matched
receiver-control changes use the same input pairing.

The calibration may improve MSE prediction and still worsen rate through a
frozen decoder's changed input distribution. Such an outcome is retained.
No training or performance improvement is claimed from cloud CPU checks.

## Run on the old SSH server after merging the PR

The target login is `r50037942@w00611266-ELITE`, project
`/home/r50037942/project/03_PRECODER`, Conda environment `precoder`. The old
machine already has the pinned aligned/rate/alpha/W evidence and source
weights. New-machine proxy migration is independent of this experiment;
do not change a working old proxy configuration to launch it.

Copy the whole block into the old SSH terminal. It preserves unrelated dirty
files, stops on source conflicts or existing output, and never resets,
cleans, deletes a lock or terminates another process.

```bash
conda activate precoder

bash <<'UL_CALIBRATION' || printf '%s\n' '同步或启动停止，请回传错误。'
set -Ee -o pipefail
fail() { printf 'STOP: %s\n' "$*" >&2; exit 1; }
test "$(id -un)" = r50037942 || fail '请在旧机 r50037942 账号运行。'
test "${CONDA_DEFAULT_ENV:-}" = precoder || fail 'precoder 环境未激活。'
cd -P /home/r50037942/project/03_PRECODER
test "$(git branch --show-current)" = main || fail '当前不在 main。'
test "$(git remote get-url origin)" = https://github.com/Charlie-Su-55/03_PRECODER.git || fail 'origin 不符。'
git diff --cached --quiet || fail '存在已暂存修改，已保留。'
for task_pid in $(pgrep -u "$(id -u)" -x 'python([0-9.]*)?' || true); do
  task_cwd="$(readlink -f "/proc/$task_pid/cwd" 2>/dev/null || true)"
  case "$task_cwd/" in "$PWD/"*) fail "03 Python 进程仍在运行：$task_pid" ;; esac
done
git fetch origin main
test "$(git rev-list --count origin/main..HEAD)" -eq 0 || fail '存在本地独立提交，已保留。'
git pull --ff-only origin main
test -f run_twc_v1_k8_ul_calibration.py || fail '独立校准 PR 尚未合并。'
task_python="$CONDA_PREFIX/bin/python"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$task_python" -B -m unittest discover -s tests -p 'test_twc_v1_k8_ul_calibration_*.py'
"$task_python" -B -S run_twc_v1_k8_ul_calibration.py --dry-run
test ! -e runs/twc_v1_k8_ul_calibration_v1 \
  && test ! -L runs/twc_v1_k8_ul_calibration_v1 \
  || fail '本批目录已存在，请先查看摘要。'
command -v tmux >/dev/null || fail '未找到 tmux。'
if tmux has-session -t '=k8_ul_calibration' 2>/dev/null; then
  fail 'k8_ul_calibration 会话已存在，请先查看。'
fi
mkdir -p terminal_logs
task_log="$PWD/terminal_logs/k8_ul_calibration_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
tmux new-session -d -s k8_ul_calibration \
  "cd '$PWD' && '$task_python' -B -u run_twc_v1_k8_ul_calibration.py --run --allow-shared-gpu --min-free-mib 12000 --max-gpu-util 80 > '$task_log' 2>&1"
printf 'Log: %s\n' "$task_log"
UL_CALIBRATION
```

Computation uses local installed packages, weights and generated channels. It
does not need a foreground proxy; pull and explicit publication do need the
machine's normal authenticated network route. Full decoder validation adds
work compared with the earlier soft-only diagnostic; no fixed elapsed-time
guarantee is given before measuring that server's inference speed.

Some CPU Git-fixture checks deliberately produce rejected pushes or a missing
remote error under `/tmp`. Those are local negative tests, not GitHub upload
attempts. The final unittest `OK` determines whether the checks passed.

## Summary, explicit result upload and retry

Read the complete summary without starting a batch:

```bash
conda activate precoder
cd -P /home/r50037942/project/03_PRECODER
"$CONDA_PREFIX/bin/python" -B run_twc_v1_k8_ul_calibration.py --summary
```

When it displays `completed | stage=directory_export`, publish the completed
local report using the old machine's existing GitHub authentication:

```bash
"$CONDA_PREFIX/bin/python" -B publish_twc_v1_k8_ul_calibration.py --publish
```

The summary also prints an exact `--export-dir` command. Upload is explicit;
the runner never contacts GitHub. Only the scalar/model-statistic/hash JSON
and CSV whitelist is published to a separate `results/twc_v1_k8_ul_calibration_v1/…`
branch. There is no ZIP, neural weight, channel tensor, private proxy file,
token or full machine environment dump. Return the result branch, commit and
summary to the conversation; do not merge the result branch into main.

This repository is public. Removing a temporary result ref later does not
guarantee erasure from Git history, clones or caches. The local server report
is preserved. The publisher's `--delete-remote --branch ... --expected-commit ...`
accepts only this namespace's exact result ref and deletes with a matching
commit lease. Its Git plumbing does not checkout a branch or alter the source
HEAD/index/dirty worktree. Authentication belongs in the private Git prompt,
never in chat or an inline command.

If the controller fails, keep its summary, exact child log and partial attempt
directories. After the previous process exits and the issue is addressed,
only this same batch may use explicit `--run --resume`, with the same GPU
arguments. Completed stages are reused only after their evidence hashes and
production manifests pass again. A completed calibration model cannot be
changed to fit validation; a source/core/recipe mismatch causes a stop.
Never remove locks or directories to bypass an identity failure.

## Validation boundary

Cloud checks use real CPU tensors, local fixture codecs and temporary Git
repositories. They exercise PSD/receiver algebra, original-receiver limits,
truth-free prediction interfaces, role separation, model locks, complete-MU
statistics, failure/retry protection and isolated publication. Fixture
manifests are marked `test_fixture` and cannot pass production verification.
Actual pinned weights, CUDA/Sionna channels, physical server runtime and new
CSI/rate results must be checked by the SSH batch. No such real training or
full model evaluation has been run in cloud.

The completed combined cloud regression passed **229 tests, zero skips,
failures or errors, in18.195 seconds**:143 preceding UL checks plus86 new
calibration checks. Torch was2.9.1+cpu. Actual small CPU tensors exercised the
paid waveform, Fourier-LS, pilot CE fixture, statistical fit, three receivers,
tiny frozen decoder and original RZF/rate scorer. The resulting687-column
summary and all7 scalar payloads passed the real publisher schema validator.
Real temporary Git repositories tested publication, retries, conflicts and
expected-commit deletion. Separate read-only review checked truth isolation
and the receiver formulas; all46 historical scientific files remain byte
identical to main. These checks establish implementation behavior only.
