#!/usr/bin/env bash
# ONE-TIME sealed test evaluation of the K8 paper models in tmux (refuses if a test report exists).
# Run on the new SSH machine (account rong) after `conda activate precoder`.
set -Eeuo pipefail
test "$(id -un)" = rong || { echo 'STOP: run on the new SSH machine (account rong).' >&2; exit 1; }
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
test -n "${CONDA_PREFIX:-}" || { echo 'STOP: conda activate precoder first.' >&2; exit 1; }
py="$CONDA_PREFIX/bin/python"
command -v tmux >/dev/null || { echo 'STOP: tmux not found.' >&2; exit 1; }
if tmux has-session -t '=k8_test' 2>/dev/null; then
  echo 'STOP: tmux session k8_test already exists: tmux attach -t k8_test' >&2; exit 1
fi
for f in evaluate_twc_v1_k8_test.py evaluate_twc_v1_k8_baselines.py evaluate_twc_v1_k8_extended.py evaluate_twc_v1_k8_final.py \
         final_runtime.py train_twc_v1_k8_final.py twc_v1_k8_final_results.py configs/twc_v1_k8_final_config.py; do
  git ls-files --error-unmatch -- "$f" >/dev/null 2>&1 || { echo "STOP: untracked $f" >&2; exit 1; }
  git diff --quiet -- "$f" || { echo "STOP: local changes in $f" >&2; exit 1; }
done
test -d runs/twc_v1_k8_final_hybrid_v1/training || { echo 'STOP: final training directory missing' >&2; exit 1; }
test -z "$(ls runs/twc_v1_k8_test_v1/reports 2>/dev/null)" || { echo 'STOP: a sealed-test report already exists.' >&2; exit 1; }
test -f runs/twc_v1_k8_paired_ft_v1/reports/attempt_180e3d7891b1/provenance.json || { echo 'STOP: switch source report missing' >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
mkdir -p runs/twc_v1_k8_test_v1
log="runs/twc_v1_k8_test_v1/console_$(date +%Y%m%d_%H%M%S).log"
tmux new-session -d -s k8_test \
  "set -o pipefail; '$py' -B -u evaluate_twc_v1_k8_test.py --evaluate 2>&1 | tee -a '$log'; \
   echo EXIT=\$? | tee -a '$log'; exec bash"
echo "Started tmux session k8_test. Log: $log"
echo "Watch: tail -f $log"
