# K8 fixed-normal-filter UL error propagation

This diagnostic holds the normal uplink receiver fixed and checks how its
channel-estimation error propagates into soft-symbol error. It creates no
optimizer, takes no training updates and does not execute the downlink
decoder, RZF or a new rate evaluation. All previous code, checkpoints,
receipts and result directories remain unchanged.

## Why this diagnostic follows the previous result

The completed first UL diagnostic found that the calibrated marginal CE
error variance was close to the observed mean error, while normal recovery's
modelled MSE understated actual soft error by roughly 2–8 times, depending
on gamma and component. Recovery using actual UL truth and zero CE-error
variance had actual/modelled MSE ratios close to one. Both physical arms
received byte-identical full-grid CE estimates and calibrated CE variances.

These observations identify an uncertainty-propagation mismatch worth
checking. They do not establish one cause. Spatial correlation, conditional
bias or heteroscedasticity, the estimated receive geometry and the prior can
interact. A marginal CE MSE close to its calibration table does not imply
that the CE channel estimate itself is accurate.

The affected quantities are the recovered feedback symbols and their
reported uncertainty/reliability features. These can affect subsequent CSI
reconstruction and interference management, but the previous diagnostic did
not measure the resulting downlink-rate change. A reliable uncertainty
model is therefore a mechanism target, not an already validated rate gain.

The first diagnostic changed both the channel supplied to recovery and its
CE-error term, consequently changing the recovery filter. This follow-up
uses the unchanged normal filter for every calculation. Actual UL truth
is consumed only by the simulator and offline scorer.

## Prespecified scope

| Item | Value |
|---|---|
| Namespace/output | `twc_v1_k8_ul_propagation_v1`, `runs/twc_v1_k8_ul_propagation_v1/` |
| Arms | FDMA24; Hybrid G4 key4/shared40 |
| Feedback gamma | 0 / 5 / 10 / 15 / 20 / 25 dB, all retained |
| Source | Existing pinned rate5000 encoder, CE and historical receiver prior |
| New optimizer / UE / CE / BS / EMA updates | All zero |
| Samples | Microbatch4 ×64 batches =256 complete8UE MU per arm/gamma |
| Replay | New namespace, seed20261121, validation indices0–63 |
| True channel | Simulator and offline scoring only |
| Receiver matrix and recovered soft symbols | Normal estimated-UL receiver, unchanged |
| BS decoder / RZF / rate scoring | Not executed |
| Test | Remains sealed |
| Delivery | Local CSV/JSON report; explicit Git upload; no ZIP |

K8/Nt32/Nsc256, 256 occupied RE including64 pilots, per-UE energy
64=Ep16+Ed48, CDL-C, the noise convention and FP32/complex64 receiver are
unchanged. The seven new production files extend the prior39-file core to46.
The prior39 files are checked against their exact completed-diagnostic hashes
from commit `40c931f6be702951b14a9e84e8e8eccb80d49b67`; the old weight and
receipt pins are not relaxed.

Every condition uses one actual paid waveform and noisy received signal.
The normal estimated channel, calibrated CE variance and frozen nominal
prior define the receiver. True current-frame transmit powers never replace
the prior, and actual noise is used only to score the decomposition. Actual
channels, standard noise and pilot polarity are paired across arms.

## Conditional model MSE and exact error decomposition

Let `A` be the normal receiver reconstructed from the estimated UL channel
and the normal receiver's actual effective regularization. Its reconstructed
`A @ Y` must agree with the unchanged original soft recovery. The reported
variance is also independently reconstructed and checked against the
original receiver output. No filter is designed from true UL.

For true effective channel `G` and the historical diagonal power model `P`,
the offline conditional model is

```
v_conditional = diag[(A G - I) P (A G - I)^H + N0 A A^H].
```

This retains the diagonal nominal-source assumptions. It is not an exact
posterior or an information-theoretic bound. Truth changes the scoring
calculation; it never changes `A`, the received observation or soft symbols.

The empirical error is decomposed using clean symbols `z` and actual
simulated data noise `n`, exclusively for offline scoring:

```
b = (A G_est - I) z
c = A (G_true - G_est) z
t = A n
soft - z = b + c + t.
```

The report retains the three squared-magnitude terms and all three signed
cross terms `2 Re(b c*)`, `2 Re(b t*)`, `2 Re(c t*)`. Their sum must reproduce
the empirical error energy. Negative cross terms are preserved. An analogous
nominal-power decomposition must reproduce the conditional model MSE.

The report compares empirical error, reported modelled variance and the
conditional propagated model on the same historical nominal-power scale,
with FDMA, Hybrid key and shared components retained. Complete-MU gamma
averages precede uncertainty estimates; coordinate counts are descriptive,
not independent confidence-interval observations.

If the conditional model approaches actual error while the reported model
remains low, a deployable CE bias/covariance or propagation calibration is a
well-motivated next design. That calibration must use independent data and
cannot supply true UL at inference. Remaining discrepancy can instead
motivate examining source means, correlations or conditional prior mismatch.
Neither outcome proves that correcting a reliability feature alone improves
downlink rate or that Hybrid must beat FDMA.

## Run on the new SSH machine after the code PR is merged

The real project path remains
`/home/r50037942/project/03_PRECODER`, login user `rong`, Conda environment
`precoder` under `/home/rong/miniforge3`. GitHub CA trust is already configured
for this repository, so the new machine can pull directly. The proxy3129 is
needed for pull/push; computation uses installed packages, local checkpoints
and locally generated channels and does not require the proxy.

Run this complete block on the new machine. It preserves unrelated work and
stops on conflicting source state, active03 Python processes or an existing
diagnostic. It never resets, cleans, deletes locks or terminates a process.

```bash
bash <<'UL_PROPAGATION' || printf '%s\n' '同步或启动停止，请回传错误。'
set -Ee -o pipefail
fail() { printf 'STOP: %s\n' "$*" >&2; exit 1; }
test "$(id -un)" = rong || fail '请在新机 rong 账号运行。'
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate precoder
cd -P /home/r50037942/project/03_PRECODER
test "$PWD" = /home/r50037942/project/03_PRECODER
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
test -f run_twc_v1_k8_ul_propagation.py || fail '条件传播诊断 PR 尚未合并。'
task_python="$CONDA_PREFIX/bin/python"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$task_python" -B -m unittest discover -s tests -p 'test_twc_v1_k8_ul_propagation_*.py' -v
"$task_python" -B -S run_twc_v1_k8_ul_propagation.py --dry-run
test ! -e runs/twc_v1_k8_ul_propagation_v1 \
  && test ! -L runs/twc_v1_k8_ul_propagation_v1 \
  || fail '诊断目录已存在，请先查看摘要。'
command -v tmux >/dev/null || fail '未找到 tmux。'
if tmux has-session -t '=k8_ul_propagation' 2>/dev/null; then
  fail 'k8_ul_propagation 会话已存在，请先查看。'
fi
mkdir -p terminal_logs
task_log="$PWD/terminal_logs/k8_ul_propagation_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
tmux new-session -d -s k8_ul_propagation \
  "cd '$PWD' && '$task_python' -B -u run_twc_v1_k8_ul_propagation.py --run --allow-shared-gpu --min-free-mib 12000 --max-gpu-util 80 > '$task_log' 2>&1"
printf 'Log: %s\n' "$task_log"
UL_PROPAGATION
```

Read and return the complete summary:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate precoder
cd -P /home/r50037942/project/03_PRECODER
"$CONDA_PREFIX/bin/python" -B run_twc_v1_k8_ul_propagation.py --summary
```

The runner exports an immutable local report under its own
`reports/attempt_.../` directory rather than a ZIP. An interrupted new batch
can use its own explicit `--run --resume` after the previous process exits;
earlier failed attempts and all older experiments are retained.

## Explicit public Git result handoff

After completion, turn the proxy on and run:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate precoder
cd -P /home/r50037942/project/03_PRECODER
"$CONDA_PREFIX/bin/python" -B publish_twc_v1_k8_ul_propagation.py --publish
```

The default selects the last completed local report; `--export-dir` can name
an explicit report. The publisher verifies its contents before creating a
separate `results/twc_v1_k8_ul_propagation_v1/...` branch. Only the explicit
scalar CSV/JSON whitelist and sanitized necessary provenance are published.
Weights, original channel tensors, private certificates, proxy credentials
and machine environment dumps are excluded. Computation never initiates a
network upload automatically.

The source checkout's HEAD, current branch, worktree and index stay in place;
results are not merged into main. The publisher returns a branch and commit
to send to the conversation. The assistant can read that branch directly
without a chat file upload. Deleting the exact result branch after review
does not delete the local report or promise that Git history, clones or caches
immediately erase the published data.

This repository is public. The result handoff is therefore treated as public
material from the outset, even if its result branch is later removed. Public
repository reads need no GitHub authentication, but publishing requires write
credentials. Use an existing authenticated Git helper or enter a repository-
scoped token only in the SSH credential prompt; do not send a token to chat.
If authentication is unavailable, upload fails visibly while the local report
remains intact. Do not disable certificate validation to address authentication.

### First write authentication on the new machine

Create a fine-grained personal access token in the GitHub account with write
access to this repository: Settings, Developer settings, Personal access
tokens, Fine-grained tokens. Select only `03_PRECODER` and give repository
Contents read/write permission, with an expiry. Do not put the token in a
command, Git configuration file, result artifact or this conversation.

In the new SSH terminal, configure only this repository to use Git's in-memory
credential cache. The empty helper first clears inherited helper lists; it
prevents an unrelated global plaintext `store` helper from also receiving the
credential. `useHttpPath` keeps this repository's credential entry distinct.

```bash
cd -P /home/r50037942/project/03_PRECODER
git config --local credential.https://github.com.useHttpPath true
git config --local credential.https://github.com.helper ''
git config --local --add credential.https://github.com.helper 'cache --timeout=86400'
```

Run the explicit publication command above from the interactive SSH terminal
after the diagnostic completes. At Git's Username prompt enter the GitHub
account name; at its Password prompt enter the token, not the account password.
The password input is not echoed. The cache lasts up to24 hours and can be
lost on restart; the public result files and CA trust do not depend on the
cache. Authentication must not be initiated inside an unattended tmux job.

Send only the printed result branch and commit to the conversation. After
the assistant has read and audited that immutable snapshot, delete the exact
result branch with the publisher's `--delete-remote --branch ...
--expected-commit ...` command if needed. This uses an atomic expected-commit
lease and cannot delete main or a result ref that has changed. Deletion
does not remove the server's local evidence or guarantee immediate erasure
of Git objects and caches.

## Verification boundary

Cloud CPU checks exercise formulas, complex receiver reconstruction, signed
cross terms, pairing, state preservation and Git-export isolation using small
fixtures. Their passing does not validate model performance. Actual pinned
weights, CUDA/Sionna execution and the scientific diagnostic are checked on
the SSH machine; no such execution or new training is performed by the cloud
code delivery itself. The PR reports the actual completed CPU check count.

## Completed cloud checks

The actual combined CPU regression passed **306 tests, zero skips**, in
46.172 seconds: the prior241 checks plus65 new propagation/export checks.
The new integration exercises real waveform/Fourier-LS/CE/normal recovery,
fixed-filter reconstruction, conditional and realized signed decompositions,
all scalar export schemas and frozen-state/pairing protections using tiny
explicit CPU fixtures. Those fixtures cannot pass production manifest checks.
Real temporary Git repositories and local bare remotes verify orphan result
publication, source HEAD/index/dirty-work preservation, non-overwrite, retry
and exact-lease deletion under a deliberately changed remote ref.

Cloud Torch is2.9.1+cpu. The new real-checkpoint CUDA/Sionna diagnostic and
real GitHub result upload have not run in cloud. Tests do not establish the
scientific cause, a rate improvement or model performance.
