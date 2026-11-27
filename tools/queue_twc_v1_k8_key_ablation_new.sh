#!/usr/bin/env bash
# Key-count ablation on the new SSH machine (account rong): arms one after another in one tmux session.
#   bash tools/queue_twc_v1_k8_key_ablation_new.sh q0 q2
# Each arm trains (or resumes) to its fixed 30k terminal, then runs the validation evaluation
# (--evaluate). The sealed test (--evaluate-once-test) stays a separate, manual step. The queue
# stops at the first failure. Needs `conda activate precoder` and ~/precoder_shared_ce/nmse65000
# (bash tools/setup_shared_ce.sh). g4_joint is not needed here.
set -Eeuo pipefail
test "$(id -un)" = rong || { echo 'STOP: run on the new SSH machine (account rong).' >&2; exit 1; }
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
test -n "${CONDA_PREFIX:-}" || { echo 'STOP: conda activate precoder first.' >&2; exit 1; }
command -v tmux >/dev/null || { echo 'STOP: tmux not found.' >&2; exit 1; }
test $# -gt 0 || { echo 'usage: bash tools/queue_twc_v1_k8_key_ablation_new.sh q0 [q2 q8]' >&2; exit 2; }
for arm in "$@"; do case "$arm" in q0|q2|q8) ;; *) echo "STOP: unknown arm $arm" >&2; exit 1 ;; esac; done
for f in train_twc_v1_k8_key_ablation.py keyed_runtime.py grouped_runtime.py train_twc_v1_k8_grouped.py \
         final_runtime.py configs/twc_v1_k8_key_ablation_config.py configs/twc_v1_k8_grouped_config.py \
         configs/twc_v1_k8_final_config.py; do
  git ls-files --error-unmatch -- "$f" >/dev/null 2>&1 || { echo "STOP: untracked $f (git pull first)" >&2; exit 1; }
  git diff --quiet -- "$f" || { echo "STOP: local changes in $f" >&2; exit 1; }
done
bash tools/setup_shared_ce.sh >/dev/null || { echo 'STOP: shared CE missing; run bash tools/setup_shared_ce.sh' >&2; exit 1; }
session=k8_key_queue
if tmux has-session -t "=$session" 2>/dev/null; then
  echo "STOP: tmux session $session already exists: tmux attach -t $session" >&2; exit 1
fi
mkdir -p runs/twc_v1_k8_key_ablation_v1/logs
py="$CONDA_PREFIX/bin/python"
script=$(mktemp "${TMPDIR:-/tmp}/k8_key_queue.XXXXXX")
{
  echo 'set -uo pipefail'
  printf 'cd %q\n' "$PWD"
  for arm in "$@"; do
    printf 'arm=%q; log=runs/twc_v1_k8_key_ablation_v1/logs/${arm}_$(date +%%Y%%m%%d_%%H%%M%%S)_new.log\n' "$arm"
    echo 'if test -f runs/twc_v1_k8_key_ablation_v1/$arm/training/terminal_receipt.json; then echo "SKIP train: $arm complete"; else'
    echo '  resume=""; test -d runs/twc_v1_k8_key_ablation_v1/$arm/training && resume=--resume'
    printf '  %q -B -u train_twc_v1_k8_key_ablation.py --arm "$arm" --train $resume 2>&1 | tee -a "$log"\n' "$py"
    echo '  test "${PIPESTATUS[0]}" = 0 || { echo "QUEUE STOPPED: $arm training failed" | tee -a "$log"; exit 1; }'
    echo 'fi'
    printf '%q -B -u train_twc_v1_k8_key_ablation.py --arm "$arm" --evaluate 2>&1 | tee -a "$log"\n' "$py"
    echo 'test "${PIPESTATUS[0]}" = 0 || { echo "QUEUE STOPPED: $arm evaluation failed" | tee -a "$log"; exit 1; }'
  done
  echo 'echo "QUEUE DONE"'
} > "$script"
tmux new-session -d -s "$session" "bash $script; exec bash"
echo "Started tmux session $session with arms: $*"
echo "Watch: tail -n 2 runs/twc_v1_k8_key_ablation_v1/logs/*_new.log   (or: tmux attach -t $session)"
