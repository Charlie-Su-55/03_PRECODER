#!/usr/bin/env bash
# Launch the final Hybrid training followed by the paper evaluation in tmux.
# Run on the new SSH machine (account rong) after `conda activate precoder`.
# Existing runs are never overwritten: if training already exists, the script
# resumes it only when RESUME=1 is set explicitly.
set -Eeuo pipefail
test "$(id -un)" = rong || { echo 'STOP: run on the new SSH machine (account rong).' >&2; exit 1; }
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
test -n "${CONDA_PREFIX:-}" || { echo 'STOP: conda activate precoder first.' >&2; exit 1; }
py="$CONDA_PREFIX/bin/python"
command -v tmux >/dev/null || { echo 'STOP: tmux not found.' >&2; exit 1; }
if tmux has-session -t '=k8_final' 2>/dev/null; then
  echo 'STOP: tmux session k8_final already exists: tmux attach -t k8_final' >&2; exit 1
fi
for f in configs/twc_v1_k8_final_config.py final_runtime.py train_twc_v1_k8_final.py \
         evaluate_twc_v1_k8_final.py twc_v1_k8_final_results.py; do
  git ls-files --error-unmatch -- "$f" >/dev/null 2>&1 || { echo "STOP: untracked $f" >&2; exit 1; }
  git diff --quiet -- "$f" || { echo "STOP: local changes in $f" >&2; exit 1; }
done
for d in runs/twc_v1_k8_low_snr_interference_v1/training/low_snr_interference \
         runs/twc_v1_k8_ce_refine_v1/training/nmse; do
  test -f "$d/terminal_receipt.json" || { echo "STOP: missing source $d" >&2; exit 1; }
done
test -f runs/twc_v1_k8_ce_refine_v1/reports/attempt_9e9d46c5e8f7/calibration.json \
  || { echo 'STOP: missing CE calibration.json' >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
train_args='--train'
if test -e runs/twc_v1_k8_final_hybrid_v1/training; then
  if test "${RESUME:-0}" = 1; then train_args='--train --resume'
  else echo 'STOP: training directory exists. Inspect it, then rerun with RESUME=1 to resume.' >&2; exit 1; fi
fi
mkdir -p runs/twc_v1_k8_final_hybrid_v1
log="runs/twc_v1_k8_final_hybrid_v1/console_$(date +%Y%m%d_%H%M%S).log"
tmux new-session -d -s k8_final \
  "set -o pipefail; '$py' -B -u train_twc_v1_k8_final.py $train_args 2>&1 | tee -a '$log' && \
   '$py' -B -u evaluate_twc_v1_k8_final.py --evaluate 2>&1 | tee -a '$log'; \
   echo EXIT=\$? | tee -a '$log'; exec bash"
echo "Started tmux session k8_final. Log: $log"
echo "Watch: tail -f $log    Attach: tmux attach -t k8_final (detach: Ctrl-b d)"
