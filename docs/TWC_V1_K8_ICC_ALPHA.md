# ICC-aligned K8: frozen global-alpha control

This is an inference-only control, `twc_v1_k8_icc_alpha_v1`, for the completed
CSI-NMSE and BS-rate experiments. It asks how much of the observed rate gain
can be matched by choosing one scalar RZF regularizer, without fitting a new
network. It does not change either completed experiment or require Hybrid to
outperform FDMA.

The rate-trained network changed its predicted CSI scale. For a positive
uniform scale c, normalized RZF obeys `W_alpha(c H) = W_(alpha/c^2)(H)`.
Actual channel geometry may also have changed, so this relationship motivates
a control; it does not prove that alpha alone explains the rate-trained result.

## Three frozen policies per physical arm

| Policy | Network checkpoint | RZF alpha |
|---|---|---|
| `source_fixed` | Original fixed20000-update codec | .03 |
| `source_calibrated` | The exact same original codec | One calibration-selected scalar |
| `rate_fixed` | Existing fixed5000 BS-rate terminal | .03 |

FDMA and Hybrid each have all three policies. Both physical layouts, paid
pilot budget, UE energy and noise remain unchanged: K8/Nt32/Nsc256,256RE with
64 pilots,Ep16+Ed48=64/UE, original CDL-C4x8 array and
`N0=.25*10**(-gamma/10)`. The DL SNR is25dB and every precoder keeps total power1
per tone. There is no per-user column normalization or new fairness constraint.

Every encoder, decoder, CE and historical prior is frozen, including buffers
and modes. No optimizer, backward pass, training update or EMA commit occurs.
The original CSI/effective-CSI outputs are reused; this is not a direct-W head.
For a given network/sample/gamma, generate H_hat once and reuse it for the
candidate alphas rather than repeating the neural forward nine times.

Original source files stay under `runs/twc_v1_k8_icc_aligned_savefix_v1/`,
from commit `9d345b2385d99eec1d047bd9c602638b568c84f8`. The existing rate files stay
under `runs/twc_v1_k8_icc_rate_v1/`, from commit
`957b730d56f07fae9c93cb2611f5e460f7566e67`. Configuration pins the original
checkpoint/receipt/calibration requirements and the21-file rate core plus both
rate checkpoint and terminal-receipt hashes. The server must check actual
bytes and strict state loading; pins alone do not establish server readiness.
Nothing rewrites a checkpoint, old manifest, monitor, result table or lock.

## Calibration is separate from final validation

Both arms use this same declared grid:

```text
0.003, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3
```

Each physical arm selects **one alpha shared across all six gamma values**.
There is no per-gamma, per-user, per-subcarrier or per-sample alpha selection.
For every calibration MU, average its six sum rates; average those complete-MU
values to score each candidate. Maximize that score. Exact score ties prefer
alpha.03, then the candidate closest to.03 in log distance, then the lower
alpha. The grid and rule are declared before reading calibration results.

The seed is20261121 with independent data/noise namespace
`twc_v1_k8_icc_alpha_v1`:

| Purpose | Underlying role and addresses | Complete samples per gamma |
|---|---|---:|
| Calibration |validation0–63,64 batches of4|256|
| Final validation |validation2048–2303,256 batches of4|1024|

The metadata checks verify actual calibration/final Sionna seed intervals are
disjoint from each other and both previous experiments' validation streams.
The independent namespace is retained across all policies, so the final
policies use the same actual channels, standard noise and pilot polarity.
True DL is only a scoring label, not a BS inference input. Test is never read.

The queue is `calibration → selection_lock → final_validation → result_package`.
Calibration writes an attempt-local `selection_candidate.json`. A separate
owned `selection.json` locks the chosen scalars, calibration evidence and
identities **before any final score**. Final evaluation requires that lock;
it cannot select or revise alpha from final outcomes. Existing altered locks
or evidence are rejected. Partial attempts remain for inspection, and explicit
resume may reuse only hash-verified completed work.

The calibrated policy wins on its calibration score by construction, since
.03 is in the grid. It can still lose on final validation. Keep that result,
all six gamma points and every signed difference. The earlier final result
package motivated this control but is not the alpha-selection dataset here.

## Reports and interpretation

On the same new final inputs, retain raw complete-MU rows, six-point means,
all eight fixed-user rates, each-MU weakest-user rate, desired/interference
power, I/S, output CSI energy, CSI NMSE and power checks. The method labels are
`source_fixed_fdma`, `source_calibrated_fdma`, `rate_fixed_fdma` and the matching
three `hybrid_g4` labels; these labels do not add independent user samples.

The primary residual readout is each arm's six-point
`rate_fixed − source_calibrated` sum rate, with the corresponding difference
between arms. Also retain `source_calibrated − source_fixed`,
`rate_fixed − source_fixed`, and Hybrid−FDMA for all three policies. Average
gamma inside the same complete8UE MU before sample/batch4 paired intervals.
Low-three and pointwise results are secondary. These intervals condition on
fixed trained models and the single selected alpha; they do not include
training-seed or repeated-calibration variation.

If simple scalar calibration matches most of the existing gain, prioritize
low-dimensional precoding adaptation. If a substantial residual remains,
a matched5000-update BS-NMSE control is still needed before attributing it
solely to the rate loss: the previous rate experiment also changed the frozen
boundary, learning rate and added updates. Neither outcome by itself proves
a direct-W Transformer is necessary or guarantees Hybrid wins.

## SSH execution after review and merge

Cloud work implements this control and tests small CPU paths. Actual source
weights, CUDA forward/memory and the calibrated results require the SSH
machine. Use the existing checkout and `precoder` environment; do not create
a Git worktree unless explicitly requested. The block below preserves
unrelated unstaged edits or stops on conflicts. No reset, clean, lock deletion
or process termination occurs. An external Bash prints failures and keeps the
interactive terminal open.

```bash
bash <<'ICC_ALPHA' || printf '%s\n' 'Alpha control stopped. Read the error above; the terminal stays open.'
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
test -f configs/twc_v1_k8_icc_alpha_config.py || fail 'The alpha control is not yet on main.'
CUDA_VISIBLE_DEVICES='' "$task_python" -B -m unittest discover -s tests -p 'test_twc_v1_k8_icc_alpha_*.py' -v
"$task_python" -B -S run_twc_v1_k8_icc_alpha.py --dry-run
test ! -e runs/twc_v1_k8_icc_alpha_v1 || fail 'Alpha output already exists; inspect its summary first.'
command -v tmux >/dev/null || fail 'tmux is not installed.'
if tmux has-session -t '=k8_icc_alpha' 2>/dev/null; then
  fail 'The k8_icc_alpha tmux session already exists.'
fi
mkdir -p terminal_logs
task_log="$PWD/terminal_logs/k8_icc_alpha_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
tmux new-session -d -s k8_icc_alpha \
  "cd '$PWD' && '$task_python' -B -u run_twc_v1_k8_icc_alpha.py --run --allow-shared-gpu --min-free-mib 12000 --max-gpu-util 80 > '$task_log' 2>&1"
printf 'Log: %s\n' "$task_log"
ICC_ALPHA
```

GPU policy is an initial admission check, not a continuous quota or automatic
pause mechanism. No model, sample budget or grid is silently reduced on
failure.

Read and return the compact summary:

```bash
conda activate precoder
cd -P ~/project/03_PRECODER
"$CONDA_PREFIX/bin/python" -B run_twc_v1_k8_icc_alpha.py --summary
```

After the previous alpha-control process has exited, resume only this new
inference queue with its existing lock and evidence:

```bash
"$CONDA_PREFIX/bin/python" -B -u run_twc_v1_k8_icc_alpha.py --run --resume --allow-shared-gpu --min-free-mib 12000 --max-gpu-util 80
```

This does not resume either old training batch. Full details remain available
through `--summary --details`; the normal summary should be sufficient for the
first result handoff. Actual cloud check counts and the delivered commit belong
in the PR. CUDA/source-checkpoint/calibration performance checks remain unrun
until the SSH queue executes.
