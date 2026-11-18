#!/usr/bin/env bash
# Start one decoder-comparison arm (joint or DETR-style decoder, paired Hybrid layout, from scratch): train then evaluate.
# Usage:  bash tools/start_twc_v1_k8_decoder_cmp.sh g4_detr      (or g4_joint)
#         RESUME=1 bash tools/start_twc_v1_k8_decoder_cmp.sh g4_detr
# Works on any machine/path; needs `conda activate precoder` and ~/precoder_shared_ce/nmse65000.
set -Eeuo pipefail
arm="${1:-}"
case "$arm" in g4_joint|g4_detr) ;; *) echo "usage: start_twc_v1_k8_decoder_cmp.sh g4_joint|g4_detr" >&2; exit 2;; esac
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
test -n "${CONDA_PREFIX:-}" || { echo 'STOP: conda activate precoder first.' >&2; exit 1; }
py="$CONDA_PREFIX/bin/python"
for f in configs/twc_v1_k8_decoder_cmp_config.py train_twc_v1_k8_decoder_cmp.py models/twc_v1_k8_detr_decoder.py configs/twc_v1_k8_grouped_config.py grouped_runtime.py train_twc_v1_k8_grouped.py final_runtime.py; do
  git ls-files --error-unmatch -- "$f" >/dev/null 2>&1 || { echo "STOP: untracked $f (git pull first)" >&2; exit 1; }
  git diff --quiet -- "$f" || { echo "STOP: local changes in $f" >&2; exit 1; }
done
for f in checkpoint_latest.pt manifest.json calibration.json; do
  test -f "$HOME/precoder_shared_ce/nmse65000/$f" || { echo "STOP: missing ~/precoder_shared_ce/nmse65000/$f" >&2; exit 1; }
done
run="runs/twc_v1_k8_decoder_cmp_v1/$arm"
args="--arm $arm --train"
if test -e "$run/training"; then
  if test "${RESUME:-0}" = 1; then args="$args --resume"
  else echo "STOP: $run/training exists. Rerun with RESUME=1 to resume." >&2; exit 1; fi
fi
mkdir -p "$run"
log="$run/console_$(date +%Y%m%d_%H%M%S).log"
cmd="'$py' -B -u train_twc_v1_k8_decoder_cmp.py $args && '$py' -B -u train_twc_v1_k8_decoder_cmp.py --arm $arm --evaluate; echo EXIT=\$?"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader || true
if command -v tmux >/dev/null; then
  tmux new-session -d -s "k8_$arm" "bash -c \"$cmd\" 2>&1 | tee -a '$log'; exec bash"
  echo "Started tmux session k8_$arm"
else
  nohup bash -c "$cmd" >>"$log" 2>&1 &
  echo "Started in background (nohup), PID $!"
fi
echo "Log: $log    Watch: tail -f $log"
