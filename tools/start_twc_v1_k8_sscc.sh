#!/usr/bin/env bash
# Forward-only idealized SSCC baseline (capacity-achieving code + RD-optimal quantizer) in tmux.
# Run on the new SSH machine (account rong) after `conda activate precoder`.
set -Eeuo pipefail
test "$(id -un)" = rong || { echo 'STOP: run on the new SSH machine (account rong).' >&2; exit 1; }
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
test -n "${CONDA_PREFIX:-}" || { echo 'STOP: conda activate precoder first.' >&2; exit 1; }
py="$CONDA_PREFIX/bin/python"
command -v tmux >/dev/null || { echo 'STOP: tmux not found.' >&2; exit 1; }
if tmux has-session -t '=k8_sscc' 2>/dev/null; then
  echo 'STOP: tmux session k8_sscc already exists: tmux attach -t k8_sscc' >&2; exit 1
fi
for f in evaluate_twc_v1_k8_sscc.py evaluate_twc_v1_k8_baselines.py evaluate_twc_v1_k8_extended.py evaluate_twc_v1_k8_final.py \
         final_runtime.py train_twc_v1_k8_final.py twc_v1_k8_final_results.py configs/twc_v1_k8_final_config.py; do
  git ls-files --error-unmatch -- "$f" >/dev/null 2>&1 || { echo "STOP: untracked $f" >&2; exit 1; }
  git diff --quiet -- "$f" || { echo "STOP: local changes in $f" >&2; exit 1; }
done
test -d runs/twc_v1_k8_final_hybrid_v1/training || { echo 'STOP: final training directory missing' >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
mkdir -p runs/twc_v1_k8_sscc_v1
log="runs/twc_v1_k8_sscc_v1/console_$(date +%Y%m%d_%H%M%S).log"
tmux new-session -d -s k8_sscc \
  "set -o pipefail; '$py' -B -u evaluate_twc_v1_k8_sscc.py --evaluate 2>&1 | tee -a '$log'; \
   echo EXIT=\$? | tee -a '$log'; exec bash"
echo "Started tmux session k8_sscc. Log: $log"
echo "Watch: tail -f $log"
