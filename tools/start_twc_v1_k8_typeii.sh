#!/usr/bin/env bash
# Forward-only 3GPP Type II (Rel-15, rank 1) baseline in tmux on the new SSH machine (account rong).
#   bash tools/start_twc_v1_k8_typeii.sh validate    validation frames (repeatable)
#   bash tools/start_twc_v1_k8_typeii.sh test        sealed test set, once (refuses if a report exists)
# Run after `conda activate precoder`.
set -Eeuo pipefail
mode="${1:-}"
case "$mode" in
  validate) flag=--validate; ns=twc_v1_k8_typeii_v1 ;;
  test) flag=--evaluate-once-test; ns=twc_v1_k8_typeii_test_v1 ;;
  *) echo 'usage: bash tools/start_twc_v1_k8_typeii.sh validate|test' >&2; exit 2 ;;
esac
test "$(id -un)" = rong || { echo 'STOP: run on the new SSH machine (account rong).' >&2; exit 1; }
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
test -n "${CONDA_PREFIX:-}" || { echo 'STOP: conda activate precoder first.' >&2; exit 1; }
py="$CONDA_PREFIX/bin/python"
command -v tmux >/dev/null || { echo 'STOP: tmux not found.' >&2; exit 1; }
session="k8_typeii_$mode"
if tmux has-session -t "=$session" 2>/dev/null; then
  echo "STOP: tmux session $session already exists: tmux attach -t $session" >&2; exit 1
fi
for f in evaluate_twc_v1_k8_typeii.py evaluate_twc_v1_k8_sscc.py evaluate_twc_v1_k8_channel_shift.py evaluate_twc_v1_k8_test.py \
         evaluate_twc_v1_k8_baselines.py evaluate_twc_v1_k8_extended.py evaluate_twc_v1_k8_final.py \
         final_runtime.py train_twc_v1_k8_final.py twc_v1_k8_final_results.py configs/twc_v1_k8_final_config.py; do
  git ls-files --error-unmatch -- "$f" >/dev/null 2>&1 || { echo "STOP: untracked $f (git pull first)" >&2; exit 1; }
  git diff --quiet -- "$f" || { echo "STOP: local changes in $f" >&2; exit 1; }
done
test -d runs/twc_v1_k8_final_hybrid_v1/training || { echo 'STOP: final training directory missing' >&2; exit 1; }
if [ "$mode" = test ]; then
  test -z "$(ls runs/$ns/reports 2>/dev/null)" || { echo 'STOP: a sealed-test Type II report already exists.' >&2; exit 1; }
  test -n "$(ls runs/twc_v1_k8_typeii_v1/reports 2>/dev/null)" || { echo 'STOP: run the validation mode first.' >&2; exit 1; }
fi
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
mkdir -p "runs/$ns"
log="runs/$ns/console_$(date +%Y%m%d_%H%M%S).log"
tmux new-session -d -s "$session" \
  "set -o pipefail; '$py' -B -u evaluate_twc_v1_k8_typeii.py $flag 2>&1 | tee -a '$log'; \
   echo EXIT=\$? | tee -a '$log'; exec bash"
echo "Started tmux session $session. Log: $log"
echo "Watch: tail -f $log"
