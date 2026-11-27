#!/usr/bin/env bash
# Start the review experiments on the H100 (account jupyter-r50037942), one background process per arm:
#   key-count ablation   q0 q2 q8              (train_twc_v1_k8_key_ablation.py)
#   geometry study       fdma_geo hybrid_geo   (train_twc_v1_k8_geometry.py)
# Usage: bash tools/start_twc_v1_k8_review_runs.sh arm [arm ...]   (arms must be named explicitly)
# Machine plan (2026-11-27): geometry arms and q8 on the H100; q0 and q2 on the new machine
# (tools/queue_twc_v1_k8_key_ablation_new.sh). The partial H100 runs of q0 and q2 (~1k updates) are not resumed.
# Every arm trains from scratch for 30k updates. An arm whose training directory already exists is
# resumed from its last checkpoint (same recipe, code and CE are verified by the trainer).
set -Eeuo pipefail
test "$(id -un)" = jupyter-r50037942 || { echo 'STOP: run on the H100 (account jupyter-r50037942).' >&2; exit 1; }
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
test -n "${CONDA_PREFIX:-}" || { echo 'STOP: conda activate precoder first.' >&2; exit 1; }
py="$CONDA_PREFIX/bin/python"
arms=("$@"); test ${#arms[@]} -gt 0 || { echo 'usage: bash tools/start_twc_v1_k8_review_runs.sh arm [arm ...]' >&2; exit 2; }
for f in train_twc_v1_k8_key_ablation.py train_twc_v1_k8_geometry.py keyed_runtime.py grouped_runtime.py \
         train_twc_v1_k8_grouped.py final_runtime.py evaluate_twc_v1_k8_channel_shift.py \
         configs/twc_v1_k8_key_ablation_config.py configs/twc_v1_k8_geometry_config.py \
         configs/twc_v1_k8_grouped_config.py configs/twc_v1_k8_final_config.py; do
  git ls-files --error-unmatch -- "$f" >/dev/null 2>&1 || { echo "STOP: untracked $f (git pull first)" >&2; exit 1; }
  git diff --quiet -- "$f" || { echo "STOP: local changes in $f" >&2; exit 1; }
done
for f in "$HOME/precoder_shared_ce/nmse65000/checkpoint_latest.pt" "$HOME/precoder_shared_ce/nmse65000/manifest.json" \
         "$HOME/precoder_shared_ce/nmse65000/calibration.json" \
         runs/twc_v1_k8_decoder_cmp_v1/g4_joint/training/terminal_receipt.json; do
  test -f "$f" || { echo "STOP: missing $f" >&2; exit 1; }
done
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
for arm in "${arms[@]}"; do
  case "$arm" in
    q0|q2|q8) script=train_twc_v1_k8_key_ablation.py; ns=twc_v1_k8_key_ablation_v1 ;;
    fdma_geo|hybrid_geo) script=train_twc_v1_k8_geometry.py; ns=twc_v1_k8_geometry_v1 ;;
    *) echo "STOP: unknown arm $arm" >&2; exit 1 ;;
  esac
  if pgrep -u "$(id -u)" -f "$script --arm $arm --train" >/dev/null; then
    echo "SKIP: $arm is already running."; continue
  fi
  resume=''
  test -d "runs/$ns/$arm/training" && resume='--resume'
  if test -f "runs/$ns/$arm/training/terminal_receipt.json"; then echo "SKIP: $arm already completed."; continue; fi
  mkdir -p "runs/$ns/logs"
  log="runs/$ns/logs/${arm}_$(date +%Y%m%d_%H%M%S).log"
  nohup "$py" -B -u "$script" --arm "$arm" --train $resume >>"$log" 2>&1 < /dev/null &
  echo "Started $arm ${resume:-(new)} PID $! log $log"
  sleep 20
done
echo 'Progress: tail -n 3 runs/twc_v1_k8_*_v1/logs/*.log'
