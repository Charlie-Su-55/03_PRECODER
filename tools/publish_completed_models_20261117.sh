#!/usr/bin/env bash
# Explicitly upload the two completed studies; no GPU or training commands.
set -Ee -o pipefail

task_profile="${1:-}"
case "$task_profile" in
  old)
    test "$(id -un)" = r50037942
    task_publisher=publish_twc_v1_k8_low_snr.py
    task_report=runs/twc_v1_k8_low_snr_interference_v1/reports/attempt_63d630146f51
    ;;
  new)
    test "$(id -un)" = rong
    task_publisher=publish_twc_v1_k8_ce_refine.py
    task_report=runs/twc_v1_k8_ce_refine_v1/reports/attempt_9e9d46c5e8f7
    ;;
  *) printf '%s\n' 'Usage: bash tools/publish_completed_models_20261117.sh old|new' >&2; exit 2 ;;
esac

cd -P "$(dirname -- "${BASH_SOURCE[0]}")/.."
test "$(git remote get-url origin)" = https://github.com/Charlie-Su-55/03_PRECODER.git
test "$(git branch --show-current)" = main
task_python="${CONDA_PREFIX:?Activate precoder first}/bin/python"
test -x "$task_python"

"$task_python" -B "$task_publisher" --publish --export-dir "$PWD/$task_report"
mkdir -p "$HOME/precoder_model_releases"
task_bundle="$HOME/precoder_model_releases/${task_profile}_20261117"
if test -e "$task_bundle"; then
  "$task_python" -B tools/model_release_03.py --verify --bundle-dir "$task_bundle"
else
  "$task_python" -B tools/model_release_03.py --prepare --profile "$task_profile" --output-dir "$task_bundle"
fi
"$task_python" -B tools/publish_03_model_release.py --publish --bundle-dir "$task_bundle"
printf 'Completed model archive: %s\n' "$task_bundle"
