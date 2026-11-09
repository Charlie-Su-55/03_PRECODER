# Physical-split post-training finish

## Fixed failed batch and separate code identities

This is an independent post-training finish for the fixed original batch:

```
runs/twc_v1_k8_physical_split_v1/batches/batch_0626e5795f5941af8a5a67d2e465564a/
```

The training implementation is `914719b5b6e0323b6f5a07c1d875608ff6307cf1`. The reported stop was in `training_audit` while exporting mixed-layout monitor rows after the four training arms. FDMA and Hybrid rows do not have identical optional branch diagnostics; deriving CSV columns from the first row omitted later Hybrid fields. A CSV export failure does not itself establish a training/model failure.

The old batch, its failure record, partial CSVs, logs, cleanup reports, source locks, monitors and weights remain read-only. Completion of the four trained states must be verified on the server rather than assumed from the failure message. The finisher does not call the old controller's resume/training path, replay the 108-step segment, create or step an optimizer, clean caches or remove any old lock.

New outputs are isolated under `runs/twc_v1_k8_physical_split_finish_v1/`, with their own `last_finish.json`, `finishes/finish_<id>/finish_summary.json`, training-acceptance lock and new alpha-selection record. The finisher's new HEAD and hashes are separate from the training identity. All 56 original `SOURCE_FILES`, including STATUS, retain their historical identity; this task adds an independent record instead of updating STATUS or rewriting an old hash to match the new HEAD.

On the current Windows checkout, 16 of those files have LF→CRLF checkout bytes relative to the Git blobs. That is a line-ending difference, not a source-content edit or proof that local raw hashes equal the archived Linux training hashes. Source files are not rewritten/normalized to obtain acceptance. The server must still verify all 56 raw file hashes exactly against the 914719b Git identity; that real-server check has not been executed locally.

## Source validation and no training-state changes

The server accepts only the fixed original batch and its declared checkpoint paths. It verifies completion, source/configuration fingerprints, phase and input-address histories, original stage/file hashes, monitor evidence, shared-input/frozen-state checks and the complete historical source chain. Four new physical-split models must each have 6048 data iterations/head updates, 5940 original-BS updates, UE/CE zero updates and final data address 46288. Both optimizer/scheduler states and saved RNG are checked as evidence; they are not continued.

The known `training_audit` export stop is handled by a new, explicit postprocessing workflow, not a general ignore-errors/hash bypass. The source audit error must contain `dict contains fields not in fieldnames`; a different audit failure is rejected. CPU checkpoint verification is server work. The unchanged model classes then load the trained states strictly for frozen CUDA evaluation: no migration, zero-head initialization, missing-key fill, partial load or checkpoint rewriting is permitted. The trained physical heads/bridges stay intact. Model mode/buffers and checkpoint bytes are verified before/after evaluation.

The new `training_acceptance.json` is independent and non-overwriting. A finish-only fcntl mutex and persistent lease record active/inactive status and history; neither old nor new lock files are unlinked to obtain access. Explicit resume reuses only completed stages whose recorded evidence and hashes still pass. Original failed attempts and the source batch's status are not relabeled as though the old run had completed under new code.

## Complete-schema atomic CSV output

`physical_split_finish_io.write_union_csv(path, rows, compressed=None)` is standard-library-only. It snapshots and scans every row, takes the sorted union of all string field names and preserves input row order. Missing values and None become empty CSV fields. Actual 0/0.0/negative-zero values are preserved; an unavailable branch statistic is not imputed as 0. Nonmapping rows or invalid schemas fail explicitly.

The helper uses `restval=''` and `extrasaction='raise'`; it never suppresses later columns with `extrasaction='ignore'`. The same helper is used for reconstructed `monitor_full_grid.csv`/`monitor_all_six.csv` and all new evaluation CSV artifacts. CSV field ordering can change to the deterministic union; experiment values and statistic definitions do not change.

Plain and gzip outputs are supported, including explicit positional compressed=True calls. Gzip does not embed the random temporary filename or current time. The writer creates a unique temporary file in the destination's existing directory, writes/closes/flushes it completely and only then atomically replaces the destination. On failure, existing destination bytes remain intact and only this call's new temporary path is removed. Existing user temporary files, old source CSVs and permissions are not changed to force success. Empty input writes an empty file; nonempty input with no fields is rejected.

## Finish protocol and reporting

Rebuild the training audit/tables from verified original monitor/progress records in the new output tree. Do not recalculate their source training events or overwrite the partial tables in the failed batch. Preserve optional FDMA/Hybrid columns across all rows.

Frozen evaluation keeps the original physical-split protocol: seed 20261107, K8/Nt32/Nsc256, batch 4, DL SNR 25 dB, six gamma points 0/5/10/15/20/25, Ep 16, G4 ratios 1:3 and 1:1, and test sealed. Final-only global-alpha selection uses the prescribed 64-MU validation addresses 15360–15375 and unchanged nine-point grid; final scoring uses separate 256-MU addresses 16384–16447. Lock selection first. Fixed alpha 0.03 and the independently selected policy are both retained. No checkpoint/seed/SNR or alpha is selected using final results.

Compare the four completed new models and the same four frozen old controls. Retain raw physical NMSE, separately labeled semantic-latent diagnostics, all rates/power checks and sample/batch paired statistics, including negative results. Across-SNR statistics average inside each complete MU sample before computing uncertainty. CSV schema repair is not a new metric, data resampling, model update or performance claim.

The new controller is `finish_twc_v1_k8_physical_split.py`. It provides standard-library `--dry-run`, explicit `--run`, `--run --resume`, compact `--summary` and explicit `--summary --details`. Its stages are `training_acceptance`, `CPU_aggregation`, `alpha_calibration`, `frozen_validation` and separately verified `result_packaging`. It has no training or cleanup stage. Progress/errors, exact result paths and unknown/incomplete values remain visible; detailed monitor and raw statistics stay in files/details. Failure reporting prioritizes the finisher's own failing-stage exception/traceback rather than guessing the newest historical training log. Alpha selection uses its own explicit attempt path; a partial attempt is retained rather than silently reselected.

The commands below use the existing `precoder` environment and its absolute interpreter in `~/project/03_PRECODER`. Related same-user Python processes are checked before fetch/pull; none are killed or unlocked. This launches only the new finisher, not either training controller. No SSH connection or server command is run during local preparation.

```bash
set -eu
cd -P ~/project/03_PRECODER
conda activate precoder
task_python="$(python -c 'import sys; print(sys.executable)')"
case "$task_python" in /*) ;; *) echo 'Absolute Python path required'; exit 1 ;; esac
for task_pid in $(pgrep -u "$(id -u)" -x 'python([0-9.]*)?' || true); do
  task_cwd="$(readlink -f "/proc/$task_pid/cwd" 2>/dev/null || true)"
  case "$task_cwd/" in "$PWD/"*) echo "03 Python process still active: $task_pid"; exit 1 ;; esac
done
test "$(git branch --show-current)" = main
test "$(git remote get-url origin)" = https://github.com/Charlie-Su-55/03_PRECODER.git
git status --short --branch
test -z "$(git diff --cached --name-only)"
git fetch origin
test "$(git rev-list --count origin/main..HEAD)" -eq 0
git pull --ff-only origin main
git merge-base --is-ancestor 914719b5b6e0323b6f5a07c1d875608ff6307cf1 HEAD
"$task_python" -B -S finish_twc_v1_k8_physical_split.py --dry-run
mkdir -p terminal_logs
task_log="$PWD/terminal_logs/physical_split_finish_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
tmux new-session -d -s k8_physical_finish "cd '$PWD' && '$task_python' -B -u finish_twc_v1_k8_physical_split.py --run > '$task_log' 2>&1"
printf 'Finish log: %s\n' "$task_log"
```

Return the complete compact summary:

```bash
conda activate precoder
cd ~/project/03_PRECODER
task_python="$(python -c 'import sys; print(sys.executable)')"
"$task_python" -B finish_twc_v1_k8_physical_split.py --summary
```

Use `--summary --details` for full records. After the prior finish process has exited, explicitly continue the independent finisher:

```bash
"$task_python" -B -u finish_twc_v1_k8_physical_split.py --run --resume
```

## Actual local checks and server work remaining

The IO helper's six standard-library tests passed with 0 failures/errors/skips under `-B -S`. They cover FDMA-first and Hybrid-first full-schema exports, missing/None versus real zeros, quoted Unicode/newlines, plain/gzip round trips, deterministic gzip bytes, complete pre-scan failures, partial-write/replace failures preserving old bytes and unrelated user temporaries, and explicit strict DictWriter behavior. These tests use owned temporary fixtures, not source artifacts.

The final integrated suite ran 25 tests with 25 passes and 0 failures/errors/skips on CPU. A separate `-B -S` run passed 22 tests and explicitly skipped three requiring the real Torch/NumPy site packages. The seven new Python files passed Python 3.10-compatible AST/compile checks; both new CLI standard-library dry-runs and the no-start summary passed without creating the finish output root. Tests cover actual mixed-layout table aggregation, complete-schema/atomic IO, strict historical failure/identity rejection, saved optimizer/RNG evidence and small shared-module loading, hash-bound completed-stage reuse, first alpha-attempt guards, exact early-child failure traces, compact full-result summaries and persistent lease history. They do not load a real source checkpoint or execute the Linux fcntl mutex.

Before/after raw SHA256 values matched for all 64 protected local files: the 56 original `SOURCE_FILES` including STATUS, five manual plotting/figure files, `.gitignore` and the two existing cleanup receipts. The only pre-existing tracked modification remains the manual plotting script. This proves preservation of the local checkout bytes, not equality of Windows CRLF bytes to the archived Linux training files.

No true source checkpoint, complete codec/CE, CDL, training or GPU evaluation is run locally. Actual checkpoint CPU acceptance, Linux mutex behavior, strict frozen CUDA model loading/evaluation and numerical performance results remain server verification. No experiment result has been invented or inferred from the CSV repair.
