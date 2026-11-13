# K8 frozen-codeword UL mechanism diagnostics

This diagnostic measures sensitivity to replacing the estimated UL channel
and its CE-uncertainty term together, alongside the remaining error of the
current shared-feedback recovery. It does not isolate those two substitutions
as separate causes, train a new method or choose a new checkpoint. The completed aligned, BS-rate,
alpha-control and residual-W batches remain historical evidence.

The completed residual-W experiment improved the six-gamma mean by only
0.02824 bit/s/Hz for FDMA and 0.02356 for Hybrid compared with equally trained
RZF continuation. It did not remove Hybrid's low-gamma gap. Because that
experiment froze the UE, CE and historical prior, it did not test a new
transmit mapping or a different UL recovery mechanism. This diagnostic
examines that boundary before committing to another training design.

## Fixed scope and source

| Item | Prespecified value |
|---|---|
| Namespace and output | `twc_v1_k8_ul_diagnostics_v1`, `runs/twc_v1_k8_ul_diagnostics_v1/` |
| Physical arms | FDMA24 and Hybrid G4 key4/shared40 |
| Feedback gamma | 0 / 5 / 10 / 15 / 20 / 25 dB; retain every point |
| Replay | New namespace, seed20261121, `validation` indices0–63 |
| Samples | Microbatch4 ×64 batches =256 complete8UE MU per arm/gamma |
| Physical observations | 2 arms ×6 gamma ×64 batches =768 microbatch conditions |
| Recovery conditions | Two per observation, using the exact same received tensor |
| New optimizer / UE / CE / BS / EMA updates | All zero |
| Weights | Existing pinned BS-rate5000 terminals; no monitor selection |
| DL decoder, RZF and rate evaluation | Not executed |
| Test split | Remains sealed |

K8/Nt32/Nsc256, 256 occupied RE including64 pilots, each UE's energy
64=Ep16+Ed48, CDL-C and `N0=.25*10**(-gamma/10)` are unchanged. The encoder,
CE, BS decoder and historical receiver statistics are in frozen inference
mode. The decoder is loaded as part of the strict source model but is not
called. No alpha fitting, backward pass or optimizer is constructed.

The source is the pinned rate5000 batch under
`runs/twc_v1_k8_icc_rate_v1/`, trained with commit
`957b730d56f07fae9c93cb2611f5e460f7566e67`. Its UE encoder, CE and historical
prior are the same frozen states used by the completed W experiment. Actual
checkpoint bytes, terminal receipts, initialization, CE calibration and
source identities must pass the unchanged strict loaders. Hardcoded config
hashes never substitute for loading the actual files.

All 33 existing W core files are checked against the fixed hashes recorded
in the completed W small package from commit
`8d7f0c2e2052ffbdfaa3377c1290cfe3aad317c9`. This diagnostic adds six production
files to the core; it does not edit or relax the earlier core or weight pins.

The new replay namespace keeps this diagnostic independent of used aligned,
rate, alpha and W validation addresses. The same actual DL/UL channel,
standard noise and pilot polarity are paired across physical arms at each
batch/gamma. Within an arm, both recovery conditions consume the same
codeword, transmitted waveform, received observation and nominal prior.

## The paired recovery conditions

| Name | UL channel supplied to recovery | CE-error term | Status |
|---|---|---|---|
| `estimated_ul` | Original pilot-based CE estimate | Independently calibrated CE variance | Normal receiver |
| `true_ul_diagnostic` | Simulator's actual UL channel | Zero | Privileged offline sensitivity diagnostic |

Both conditions retain the actual feedback noise and the existing frozen
historical power prior. The true channel is never used to change the
encoder, transmitted energy, waveform or received observation. Clean symbols
and actual current-frame power are permitted for offline error scoring only;
they are not passed as receiver side information or substituted for the
nominal prior.

This is not a fair competing deployable method and is not an information-
theoretic or end-to-end performance upper bound. Recovery with exact UL still
uses a statistical model, finite dimensions, the same historical prior and
actual noise. A particular frozen decoder need not improve monotonically
when its input distribution changes; this diagnostic therefore stops before
that decoder and makes no new sum-rate claim.

## What is recorded

The report retains each paired MU, both arms, both recovery conditions and
all six gamma points. It checks feedback energy, frozen state and actual
input fingerprints, and exports source/core/recipe identities with the
diagnostic evidence.

- Soft-symbol error is scored over the actual selected transmitted
  coordinates, with FDMA, Hybrid key and Hybrid shared components retained.
  Each UE's error energy is divided by its own transmitted component energy,
  then the eight UE ratios are averaged within the MU, as in the completed W
  evaluator. Key/shared energies are not pooled across UEs before division.
- Actual CE error is compared with its calibrated per-antenna complex
  coefficient error model.
- Effective shared-user geometry uses the spatial/frequency-selective
  receive channel. It includes group Gram coherence and conditioning; a
  transmit-codeword inner product alone is not the UL separation geometry.
- Reliability calibration compares actual squared soft-symbol error and
  the receiver's modelled error variance on the same historical nominal-
  power scale. Reliability is not assumed to be an exact posterior
  covariance.
- Calibration bins have finite lower edges
  `[0,.01,.03,.1,.3,1,3,10]`; the final upper edge is JSON `null`. Empty bins
  and zero-prediction cases retain counts and sums without inventing a
  zero-over-zero ratio.

The pairwise statistics use one complete8UE MU as the paired unit. Gamma
averages are taken within each MU before estimating a six-point contrast.
Intervals are conditional validation-sampling intervals for fixed models
and one training seed. They do not estimate training-seed variability or
resolve architecture selection uncertainty. Negative differences are
preserved.

An improvement with true UL suggests that CE and its interaction with
statistical recovery merit attention. A remaining shared-error gap, especially
alongside poorly conditioned receive geometry, motivates checking shared
separation, noise amplification and modelled reliability. Neither pattern
alone proves a unique cause or an information-theoretic bottleneck. Current
frame power fluctuating around the historical prior is not automatically
evidence that the prior is incorrect.

## SSH handoff and startup after merge

The new server uses the real project path
`/home/r50037942/project/03_PRECODER`, even though its login user is `rong`
and its Conda installation is `/home/rong/miniforge3`. Keep all migrated
checkpoint and receipt bytes and original absolute paths unchanged.

After reviewing and merging the PR, first run the following on the old
`r50037942` SSH machine. It fast-forwards main and exports committed code;
it preserves unrelated dirty files, historical results and old failed runs.
It stops if a local process, staged work or a conflicting change requires
attention. It never resets, cleans, deletes locks or terminates a process.

```bash
bash <<'SYNC_UL_DIAG' || printf '%s\n' '代码同步或导出停止，请回传错误。'
set -Ee -o pipefail
fail() { printf 'STOP: %s\n' "$*" >&2; exit 1; }
test "$(id -un)" = r50037942 || fail '请在旧 SSH 执行。'
cd -P /home/r50037942/project/03_PRECODER
test "$(git branch --show-current)" = main || fail '当前不在 main。'
test "$(git remote get-url origin)" = https://github.com/Charlie-Su-55/03_PRECODER.git || fail 'origin 不符。'
git diff --cached --quiet || fail '存在已暂存修改，请保留并回传状态。'
for task_pid in $(pgrep -u "$(id -u)" -x 'python([0-9.]*)?' || true); do
  task_cwd="$(readlink -f "/proc/$task_pid/cwd" 2>/dev/null || true)"
  case "$task_cwd/" in "$PWD/"*) fail "03 Python 进程仍在运行：$task_pid" ;; esac
done
git fetch origin main
test "$(git rev-list --count origin/main..HEAD)" -eq 0 || fail '存在本地独立提交，已保留。'
git pull --ff-only origin main
git cat-file -e HEAD:run_twc_v1_k8_ul_diagnostics.py || fail '诊断 PR 尚未合并 main。'
test "$(git rev-parse --is-shallow-repository)" = false || fail '浅克隆请回传状态。'
task_export="$(mktemp -d "$HOME/precoder_ul_code.XXXXXXXX")"
git bundle create "$task_export/03_PRECODER-ul-diagnostics.bundle" refs/heads/main
git bundle verify "$task_export/03_PRECODER-ul-diagnostics.bundle"
cd "$task_export"
sha256sum 03_PRECODER-ul-diagnostics.bundle > 03_PRECODER-ul-diagnostics.bundle.sha256
printf '\n请下载此目录中的 bundle 和 sha256，上传新机 /home/rong/：\n%s\n' "$task_export"
SYNC_UL_DIAG
```

Download the generated `03_PRECODER-ul-diagnostics.bundle` and its `.sha256`
sidecar, then upload both to the new machine's `/home/rong/`. This is a small
code handoff. The already migrated approximately2.9GB of results and weights
need not be copied again. It also avoids the new machine's unresolved GitHub
HTTPS certificate problem without disabling TLS verification.

Run the following complete block in the new `rong` SSH session. It locally
fetches the verified bundle, fast-forwards main, runs CPU checks and a dry
run, then launches the diagnostic in tmux. No network request is needed on
the new machine for this sequence. It does not launch training.

```bash
bash <<'START_UL_DIAG' || printf '%s\n' '同步或启动停止，请回传上面的错误。'
set -Ee -o pipefail
fail() { printf 'STOP: %s\n' "$*" >&2; exit 1; }
test "$(id -un)" = rong || fail '请在新 SSH 的 rong 账号执行。'
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate precoder
cd "$HOME"
sha256sum -c 03_PRECODER-ul-diagnostics.bundle.sha256
task_bundle="$HOME/03_PRECODER-ul-diagnostics.bundle"
task_commit="$(git bundle list-heads "$task_bundle" refs/heads/main)"
task_commit="${task_commit%% *}"
[[ "$task_commit" =~ ^[0-9a-f]{40}$ ]] || fail 'bundle main 提交无效。'
cd -P /home/r50037942/project/03_PRECODER
test "$PWD" = /home/r50037942/project/03_PRECODER
test "$(git branch --show-current)" = main || fail '当前不在 main。'
git diff --cached --quiet || fail '存在已暂存修改，请保留并回传状态。'
for task_pid in $(pgrep -u "$(id -u)" -x 'python([0-9.]*)?' || true); do
  task_cwd="$(readlink -f "/proc/$task_pid/cwd" 2>/dev/null || true)"
  case "$task_cwd/" in "$PWD/"*) fail "03 Python 进程仍在运行：$task_pid" ;; esac
done
git fetch "$task_bundle" refs/heads/main:refs/remotes/origin/main
git merge --ff-only origin/main
test "$(git rev-parse HEAD)" = "$task_commit" || fail '同步后 commit 不一致。'
task_python="$CONDA_PREFIX/bin/python"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$task_python" -B -m unittest discover -s tests -p 'test_twc_v1_k8_ul_diagnostics_*.py' -v
"$task_python" -B -S run_twc_v1_k8_ul_diagnostics.py --dry-run
test ! -e runs/twc_v1_k8_ul_diagnostics_v1 \
  && test ! -L runs/twc_v1_k8_ul_diagnostics_v1 \
  || fail '诊断目录已存在，请先看 --summary。'
command -v tmux >/dev/null || fail '未找到 tmux。'
if tmux has-session -t '=k8_ul_diagnostics' 2>/dev/null; then
  fail 'k8_ul_diagnostics 会话已存在，请先查看。'
fi
mkdir -p terminal_logs
task_log="$PWD/terminal_logs/k8_ul_diagnostics_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
tmux new-session -d -s k8_ul_diagnostics \
  "cd '$PWD' && '$task_python' -B -u run_twc_v1_k8_ul_diagnostics.py --run --allow-shared-gpu --min-free-mib 12000 --max-gpu-util 80 > '$task_log' 2>&1"
printf 'Log: %s\n' "$task_log"
START_UL_DIAG
```

Read and return the complete summary:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate precoder
cd -P /home/r50037942/project/03_PRECODER
"$CONDA_PREFIX/bin/python" -B run_twc_v1_k8_ul_diagnostics.py --summary
```

The summary's `Small package` path identifies the diagnostic archive to
return for analysis. Historical W and rate result archives are not the new
diagnostic package. An interrupted diagnostic can resume its own verified
batch using `--run --resume` after the prior diagnostic process has exited;
this never resumes an earlier training experiment.

## Proxy and remote connection

The proxy is needed for network operations such as downloading dependencies
or synchronizing GitHub. This diagnostic uses installed packages, local
weights and locally generated Sionna channels; no proxy is required while it
runs. It contains no model download or remote data fetch.

The tmux process runs on the server and continues after disconnecting VS Code
Remote or closing the laptop. Reconnect later to read `--summary`; use
`tmux attach -t k8_ul_diagnostics` to view the session. Proxy uptime is separate
from diagnostic runtime. Network connectivity used to establish the SSH
connection itself still needs to work when reconnecting.

Cloud CPU checks verify formulas, pairing, provenance controls and failure
handling. Actual pinned-weight loading, CUDA/Sionna execution, runtime and
scientific diagnostics must be checked on the SSH server. Passing small
CPU tests does not validate model performance or establish the cause of the
Hybrid–FDMA gap.

## Cloud verification for this change

The combined real-CPU regression completed **241 tests with zero skips**:
118 aligned/rate/alpha checks, 67 residual-W checks and 56 new diagnostic
checks. The new integration check executes actual waveform construction,
Fourier LS, a tiny frozen CE module, both existing recovery paths and scalar
statistics; it rejects changed frozen state, wrong roles and unpaired inputs.
Its injected CPU evidence is explicitly marked `test_fixture` and cannot be
accepted as a production CUDA result. This cloud environment uses
Torch2.9.1+cpu; actual pinned server checkpoints and the new CUDA/Sionna
mechanism diagnostic have not been executed in cloud.

The two pasted shell blocks pass `bash -n`. A real temporary-repository
bundle fetch/fast-forward check also preserved original tracked source and
untracked history bytes. All 33 historical core files remain unchanged.
