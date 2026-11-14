#!/usr/bin/env bash
# Activate precoder first; this launcher is restricted to the old SSH account.
set -Ee -o pipefail
fail() { printf 'STOP: %s\n' "$*" >&2; exit 1; }
test "$(id -un)" = r50037942 || fail '请在旧机 r50037942 账号运行。'
test "${CONDA_DEFAULT_ENV:-}" = precoder || fail 'precoder 环境未激活。'
cd -P /home/r50037942/project/03_PRECODER
test "$(git branch --show-current)" = main || fail '当前不在 main。'
test "$(git remote get-url origin)" = https://github.com/Charlie-Su-55/03_PRECODER.git || fail 'origin 不符。'
git diff --cached --quiet || fail '存在已暂存修改，已保留。'
for task_pid in $(pgrep -u "$(id -u)" -x 'python([0-9.]*)?' || true); do
  task_cwd="$(readlink -f "/proc/$task_pid/cwd" 2>/dev/null || true)"
  case "$task_cwd/" in "$PWD/"*) fail "03 Python 进程仍在运行：$task_pid" ;; esac
done
git fetch origin main
test "$(git rev-list --count origin/main..HEAD)" -eq 0 || fail '存在本地独立提交，已保留。'
git pull --ff-only origin main
test -f run_twc_v1_k8_ul_calibration.py || fail '独立校准 PR 尚未合并。'
task_python="$CONDA_PREFIX/bin/python"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$task_python" -B -m unittest discover -s tests -p 'test_twc_v1_k8_ul_calibration_*.py'
"$task_python" -B -S run_twc_v1_k8_ul_calibration.py --dry-run
test ! -e runs/twc_v1_k8_ul_calibration_v1 \
  && test ! -L runs/twc_v1_k8_ul_calibration_v1 \
  || fail '本批目录已存在，请先查看摘要。'
command -v tmux >/dev/null || fail '未找到 tmux。'
if tmux has-session -t '=k8_ul_calibration' 2>/dev/null; then
  fail 'k8_ul_calibration 会话已存在，请先查看。'
fi
mkdir -p terminal_logs
task_log="$PWD/terminal_logs/k8_ul_calibration_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
tmux new-session -d -s k8_ul_calibration \
  "cd '$PWD' && '$task_python' -B -u run_twc_v1_k8_ul_calibration.py --run --allow-shared-gpu --min-free-mib 12000 --max-gpu-util 80 > '$task_log' 2>&1"
printf 'Log: %s\n' "$task_log"
