#!/usr/bin/env bash
# K=16 user-load study on one GPU machine (intended: H100): CE fine-tune -> FDMA and Hybrid
# trained in parallel with the same recipe -> joint evaluation. Resumes automatically.
# Needs `conda activate precoder` and ~/precoder_shared_ce/nmse65000 (K8 CE used as initialization).
set -Eeuo pipefail
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
test -n "${CONDA_PREFIX:-}" || { echo 'STOP: conda activate precoder first.' >&2; exit 1; }
py="$CONDA_PREFIX/bin/python"
for f in k16_runtime.py train_twc_v1_k16.py configs/twc_v1_k16_config.py models/twc_v1_k8_icc_aligned.py \
         utils/twc_v1_k8_replayable_data.py train_twc_v1_k8_final.py; do
  git ls-files --error-unmatch -- "$f" >/dev/null 2>&1 || { echo "STOP: untracked $f (git pull first)" >&2; exit 1; }
  git diff --quiet -- "$f" || { echo "STOP: local changes in $f" >&2; exit 1; }
done
for f in checkpoint_latest.pt manifest.json; do
  test -f "$HOME/precoder_shared_ce/nmse65000/$f" || { echo "STOP: missing ~/precoder_shared_ce/nmse65000/$f" >&2; exit 1; }
done
mkdir -p runs/twc_v1_k16_v1
log="runs/twc_v1_k16_v1/console_$(date +%Y%m%d_%H%M%S).log"
cmd="'$py' -B -u train_twc_v1_k16.py --pipeline; echo EXIT=\$?"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader || true
if command -v tmux >/dev/null; then
  if tmux has-session -t '=k16' 2>/dev/null; then echo 'STOP: tmux session k16 exists (tmux attach -t k16)' >&2; exit 1; fi
  tmux new-session -d -s k16 "bash -c \"$cmd\" 2>&1 | tee -a '$log'; exec bash"
  echo 'Started tmux session k16'
else
  nohup bash -c "$cmd" >>"$log" 2>&1 &
  echo "Started in background (nohup), PID $!"
fi
echo "Log: $log    Watch: tail -f $log"
