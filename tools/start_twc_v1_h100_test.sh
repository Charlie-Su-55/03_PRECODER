#!/usr/bin/env bash
# ONE-TIME sealed test of the H100 models in tmux: K8 arms (M=4, M=8, joint, DETR), then K=16.
# Run on the H100 (account jupyter-r50037942) after `conda activate precoder`. Refuses if any report exists.
set -Eeuo pipefail
test "$(id -un)" = jupyter-r50037942 || { echo 'STOP: run on the H100 (account jupyter-r50037942).' >&2; exit 1; }
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
test -n "${CONDA_PREFIX:-}" || { echo 'STOP: conda activate precoder first.' >&2; exit 1; }
py="$CONDA_PREFIX/bin/python"
command -v tmux >/dev/null || { echo 'STOP: tmux not found.' >&2; exit 1; }
if tmux has-session -t '=h100_test' 2>/dev/null; then
  echo 'STOP: tmux session h100_test already exists: tmux attach -t h100_test' >&2; exit 1
fi
for f in evaluate_twc_v1_h100_test.py evaluate_twc_v1_k8_test.py train_twc_v1_k8_grouped.py train_twc_v1_k8_decoder_cmp.py \
         grouped_runtime.py final_runtime.py k16_runtime.py train_twc_v1_k16.py models/twc_v1_k8_detr_decoder.py \
         configs/twc_v1_k8_grouped_config.py configs/twc_v1_k8_decoder_cmp_config.py configs/twc_v1_k16_config.py \
         configs/twc_v1_k8_final_config.py twc_v1_k8_final_results.py evaluate_twc_v1_k8_final.py; do
  git ls-files --error-unmatch -- "$f" >/dev/null 2>&1 || { echo "STOP: untracked $f (git pull first)" >&2; exit 1; }
  git diff --quiet -- "$f" || { echo "STOP: local changes in $f" >&2; exit 1; }
done
for f in runs/twc_v1_k8_grouped_v1/hybrid_g1/training/terminal_receipt.json \
         runs/twc_v1_k8_grouped_v1/hybrid_g2/training/terminal_receipt.json \
         runs/twc_v1_k8_decoder_cmp_v1/g4_joint/training/terminal_receipt.json \
         runs/twc_v1_k8_decoder_cmp_v1/g4_detr/training/terminal_receipt.json \
         runs/twc_v1_k16_v1/fdma/terminal_receipt.json runs/twc_v1_k16_v1/hybrid/terminal_receipt.json \
         runs/twc_v1_k16_v1/reports/attempt_63ef716343a8/provenance.json \
         "$HOME/precoder_shared_ce/nmse65000/checkpoint_latest.pt"; do
  test -f "$f" || { echo "STOP: missing $f" >&2; exit 1; }
done
test -z "$(ls runs/twc_v1_k8_test_h100_v1/reports 2>/dev/null)" || { echo 'STOP: an H100 K8 test report already exists.' >&2; exit 1; }
test -z "$(ls runs/twc_v1_k16_test_v1/reports 2>/dev/null)" || { echo 'STOP: a K16 test report already exists.' >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
mkdir -p runs/twc_v1_k8_test_h100_v1
log="runs/twc_v1_k8_test_h100_v1/console_$(date +%Y%m%d_%H%M%S).log"
tmux new-session -d -s h100_test \
  "set -o pipefail; '$py' -B -u evaluate_twc_v1_h100_test.py --evaluate-once k8 2>&1 | tee -a '$log' && \
   '$py' -B -u evaluate_twc_v1_h100_test.py --evaluate-once k16 2>&1 | tee -a '$log'; \
   echo EXIT=\$? | tee -a '$log'; exec bash"
echo "Started tmux session h100_test. Log: $log"
echo "Watch: tail -f $log"
