#!/usr/bin/env bash
# Run only after the diagnostic PR is merged/pulled and conda precoder active.
set -Ee -o pipefail
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
task_root="$PWD"
test -n "${CONDA_PREFIX:-}" || { echo 'STOP: conda activate precoder first.' >&2; exit 1; }
task_python="$CONDA_PREFIX/bin/python"
test -x "$task_python"
command -v tmux >/dev/null
if tmux has-session -t '=k8_rate_headroom' 2>/dev/null; then
    echo 'STOP: k8_rate_headroom exists; inspect its log/session.' >&2
    exit 1
fi
if test -e runs/twc_v1_k8_rate_headroom_v1; then
    "$task_python" -B diagnose_twc_v1_k8_rate_headroom.py --summary
    echo 'STOP: existing evidence retained; no rerun or overwrite.' >&2
    exit 1
fi
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$task_python" -B -m unittest discover -s tests -p 'test_twc_v1_k8_rate_headroom*.py' -v
"$task_python" -B -S diagnose_twc_v1_k8_rate_headroom.py --plan
mkdir -p terminal_logs
task_log="$task_root/terminal_logs/k8_rate_headroom_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
printf -v task_command '%q ' "$task_python" -B -u diagnose_twc_v1_k8_rate_headroom.py \
    --run --allow-shared-gpu --min-free-mib 12000 --max-gpu-util 80
printf -v task_quoted_log '%q' "$task_log"
tmux new-session -d -s k8_rate_headroom -c "$task_root" \
    "$task_command > $task_quoted_log 2>&1"
printf 'Log: %s\n' "$task_log"
printf '%s\n' 'Started frozen diagnostics only; no training or automatic upload.'
