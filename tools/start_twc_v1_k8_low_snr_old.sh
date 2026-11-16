#!/usr/bin/env bash
# User launches after merging/pulling main and activating precoder.
set -Ee -o pipefail
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
task_root="$PWD"
test -n "${CONDA_PREFIX:-}" || { echo 'STOP: conda activate precoder first.' >&2; exit 1; }
task_python="$CONDA_PREFIX/bin/python"
test -x "$task_python"
command -v tmux >/dev/null
if tmux has-session -t '=k8_low_snr_interference' 2>/dev/null; then
    echo 'STOP: k8_low_snr_interference already exists; inspect its session/log.' >&2
    exit 1
fi
if test -e runs/twc_v1_k8_low_snr_interference_v1; then
    "$task_python" -B run_twc_v1_k8_low_snr.py --summary
    echo 'STOP: existing run retained; use explicit resume only after inspection.' >&2
    exit 1
fi
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$task_python" -B -m unittest discover -s tests -p 'test_twc_v1_k8_low_snr*.py' -v
"$task_python" -B -S run_twc_v1_k8_low_snr.py --dry-run
mkdir -p terminal_logs
task_log="$task_root/terminal_logs/k8_low_snr_interference_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
printf -v task_command '%q ' "$task_python" -B -u run_twc_v1_k8_low_snr.py \
    --run --allow-shared-gpu --min-free-mib 12000 --max-gpu-util 80
printf -v task_quoted_log '%q' "$task_log"
tmux new-session -d -s k8_low_snr_interference -c "$task_root" \
    "$task_command > $task_quoted_log 2>&1"
printf 'Log: %s\n' "$task_log"
printf '%s\n' 'Two matched Hybrid trajectories queued (10000 each); FDMA/CE frozen; no automatic upload.'
