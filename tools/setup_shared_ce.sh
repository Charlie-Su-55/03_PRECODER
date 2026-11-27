#!/usr/bin/env bash
# Create ~/precoder_shared_ce/nmse65000 from this machine's CE-refine run if it is missing.
#   bash tools/setup_shared_ce.sh          check only (prints what it would copy)
#   bash tools/setup_shared_ce.sh --copy   copy the three pinned files (never overwrites)
# The files are accepted only if their SHA256 equals the pins in configs/twc_v1_k8_grouped_config.py.
set -Eeuo pipefail
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
dest="$HOME/precoder_shared_ce/nmse65000"
pins=$(python3 -B -S - <<'EOF'
import re
text = open('configs/twc_v1_k8_grouped_config.py').read()
final = open('configs/twc_v1_k8_final_config.py').read()
ck = re.search(r"'checkpoint_latest.pt': '([0-9a-f]{64})'", text).group(1)
mf = re.search(r"'manifest.json': '([0-9a-f]{64})'", text).group(1)
cal = re.search(r"CE_CALIBRATION_SHA256 = '([0-9a-f]{64})'", final).group(1)
print(ck, mf, cal)
EOF
)
read -r pin_ck pin_mf pin_cal <<<"$pins"
check() { test -f "$1" && test "$(sha256sum "$1" | cut -d' ' -f1)" = "$2"; }
if check "$dest/checkpoint_latest.pt" "$pin_ck" && check "$dest/manifest.json" "$pin_mf" && check "$dest/calibration.json" "$pin_cal"; then
  echo "OK: $dest has the pinned CE nmse65000."; exit 0
fi
if test -e "$dest/checkpoint_latest.pt" || test -e "$dest/manifest.json" || test -e "$dest/calibration.json"; then
  echo "STOP: $dest exists but differs from the pins; not touching it." >&2; exit 1
fi
src_ck=''
for f in runs/twc_v1_k8_ce_refine_v1/training/*/checkpoint_latest.pt; do
  if check "$f" "$pin_ck"; then src_ck="$f"; break; fi
done
test -n "$src_ck" || { echo 'STOP: no CE-refine checkpoint with the pinned SHA256 on this machine.' >&2; exit 1; }
src_mf="$(dirname "$src_ck")/manifest.json"
src_cal=runs/twc_v1_k8_ce_refine_v1/reports/attempt_9e9d46c5e8f7/calibration.json
check "$src_mf" "$pin_mf" || { echo "STOP: $src_mf differs from the pin." >&2; exit 1; }
check "$src_cal" "$pin_cal" || { echo "STOP: $src_cal differs from the pin." >&2; exit 1; }
echo "Source: $src_ck, $src_mf, $src_cal"
if [ "${1:-}" != --copy ]; then echo 'Check only; rerun with --copy to create the shared CE directory.'; exit 0; fi
mkdir -p "$dest"
cp -n "$src_ck" "$dest/checkpoint_latest.pt"; cp -n "$src_mf" "$dest/manifest.json"; cp -n "$src_cal" "$dest/calibration.json"
check "$dest/checkpoint_latest.pt" "$pin_ck" && check "$dest/manifest.json" "$pin_mf" && check "$dest/calibration.json" "$pin_cal" \
  && echo "OK: created $dest" || { echo 'STOP: copy does not match the pins.' >&2; exit 1; }
