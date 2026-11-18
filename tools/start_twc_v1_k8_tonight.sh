#!/usr/bin/env bash
# New SSH machine: start (1) training-free analog baselines evaluation and
# (2) pairing-aware Hybrid fine-tune followed by its evaluation, each in its own tmux session.
# FDMA is not trained. RESUME=1 resumes the fine-tune.
set -Eeuo pipefail
test "$(id -un)" = rong || { echo 'STOP: run on the new SSH machine (account rong).' >&2; exit 1; }
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
test -n "${CONDA_PREFIX:-}" || { echo 'STOP: conda activate precoder first.' >&2; exit 1; }
py="$CONDA_PREFIX/bin/python"
command -v tmux >/dev/null || { echo 'STOP: tmux not found.' >&2; exit 1; }
for f in evaluate_twc_v1_k8_baselines.py train_twc_v1_k8_paired_ft.py evaluate_twc_v1_k8_extended.py \
         evaluate_twc_v1_k8_final.py final_runtime.py train_twc_v1_k8_final.py twc_v1_k8_final_results.py \
         configs/twc_v1_k8_final_config.py; do
  git ls-files --error-unmatch -- "$f" >/dev/null 2>&1 || { echo "STOP: untracked $f" >&2; exit 1; }
  git diff --quiet -- "$f" || { echo "STOP: local changes in $f" >&2; exit 1; }
done
test -f runs/twc_v1_k8_final_hybrid_v1/training/terminal_receipt.json || { echo 'STOP: final terminal missing' >&2; exit 1; }
for s in k8_analog k8_paired_ft; do
  if tmux has-session -t "=$s" 2>/dev/null; then echo "STOP: tmux session $s exists (tmux attach -t $s)" >&2; exit 1; fi
done
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
stamp=$(date +%Y%m%d_%H%M%S)
mkdir -p runs/twc_v1_k8_analog_baselines_v1 runs/twc_v1_k8_paired_ft_v1
la="runs/twc_v1_k8_analog_baselines_v1/console_$stamp.log"
lf="runs/twc_v1_k8_paired_ft_v1/console_$stamp.log"
resume=""; test "${RESUME:-0}" = 1 && resume="--resume"
tmux new-session -d -s k8_analog \
  "set -o pipefail; '$py' -B -u evaluate_twc_v1_k8_baselines.py --evaluate 2>&1 | tee -a '$la'; echo EXIT=\$? | tee -a '$la'; exec bash"
tmux new-session -d -s k8_paired_ft \
  "set -o pipefail; '$py' -B -u train_twc_v1_k8_paired_ft.py --train $resume 2>&1 | tee -a '$lf' \
   && '$py' -B -u train_twc_v1_k8_paired_ft.py --evaluate 2>&1 | tee -a '$lf'; echo EXIT=\$? | tee -a '$lf'; exec bash"
echo "Started k8_analog (log $la) and k8_paired_ft (log $lf)."
