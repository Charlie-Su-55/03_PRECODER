#!/usr/bin/env bash
# Launch only on the user's new SSH; no package installation or automatic upload.
set -Ee -o pipefail
test "$(id -un)" = rong || { echo 'STOP: run this launcher on the new SSH rong account.' >&2; exit 1; }
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
task_root="$PWD"
test -n "${CONDA_PREFIX:-}" || { echo 'STOP: conda activate precoder first.' >&2; exit 1; }
task_python="$CONDA_PREFIX/bin/python"
test -x "$task_python"
command -v tmux >/dev/null
command -v nvidia-smi >/dev/null
mapfile -t task_core < <("$task_python" -B -S -c \
    'from configs.twc_v1_k8_ce_refine_config import CORE_FILES; print(*CORE_FILES, sep="\n")')
test "${#task_core[@]}" -gt 0 || { echo 'STOP: unable to read the CE scientific core.' >&2; exit 1; }
for task_file in "${task_core[@]}"; do
    git ls-files --error-unmatch -- "$task_file" >/dev/null 2>&1 || {
        echo "STOP: untracked or missing scientific source: $task_file" >&2
        exit 1
    }
done
if ! git diff --quiet -- "${task_core[@]}" || ! git diff --cached --quiet -- "${task_core[@]}"; then
    echo 'STOP: CE scientific core has local changes; all files are preserved.' >&2
    exit 1
fi
if tmux has-session -t '=k8_ce_refine' 2>/dev/null; then
    echo 'STOP: k8_ce_refine already exists; inspect its log/session before proceeding.' >&2
    exit 1
fi
if test -e runs/twc_v1_k8_ce_refine_v1; then
    "$task_python" -B run_twc_v1_k8_ce_refine.py --summary
    echo 'STOP: existing CE run is preserved; resume explicitly only after inspection.' >&2
    exit 1
fi
for task_file in checkpoint_latest.pt terminal_receipt.json manifest.json initialization.json; do
    test -f "runs/twc_v1_k8_icc_aligned_savefix_v1/training/ce/$task_file" || {
        echo "STOP: original CE5000 source is missing: $task_file" >&2
        exit 1
    }
done
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$task_python" -B -m unittest discover -s tests -p 'test_twc_v1_k8_ce_refine*.py' -v
"$task_python" -B -S run_twc_v1_k8_ce_refine.py --dry-run
mkdir -p terminal_logs
task_log="$task_root/terminal_logs/k8_ce_refine_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
printf -v task_command '%q ' "$task_python" -B -u run_twc_v1_k8_ce_refine.py \
    --run --allow-shared-gpu --min-free-mib 12000 --max-gpu-util 80
printf -v task_quoted_log '%q' "$task_log"
tmux new-session -d -s k8_ce_refine -c "$task_root" \
    "$task_command > $task_quoted_log 2>&1"
printf 'Log: %s\n' "$task_log"
printf '%s\n' 'Queued CE-only NMSE and geometry trajectories, 60000 NEW updates each; UE/BS/FDMA/EMA unchanged.'
