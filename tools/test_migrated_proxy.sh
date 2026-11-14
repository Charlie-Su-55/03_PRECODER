#!/usr/bin/env bash
# Run only on the destination SSH host. No old proxy or default Git setting changes.
set +x
set -Ee -o pipefail
for task_var in "${!GIT_TRACE@}"; do
    if test -n "$task_var"; then unset "$task_var"; fi
done
unset GIT_CURL_VERBOSE SSLKEYLOGFILE
fail() { printf 'STOP: %s\n' "$*" >&2; exit 1; }
test "$(id -un)" = rong || fail 'Run on the new rong account.'
test "$HOME" = /home/rong || fail 'Unexpected destination home.'
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate precoder
task_repo=/home/r50037942/project/03_PRECODER
cd -P "$task_repo"
test "$(git remote get-url origin)" = https://github.com/Charlie-Su-55/03_PRECODER.git || fail 'Unexpected origin.'
task_python="$CONDA_PREFIX/bin/python"
task_profile="$HOME/precoder_proxy_migrated_v1"
task_unit=precoder-proxy-migrated.service
task_unit_file="/etc/systemd/system/$task_unit"
task_unit_temp=''
task_started=0
task_success=0
cleanup() {
    task_rc=$?
    trap - EXIT
    if test -n "$task_unit_temp"; then rm -f -- "$task_unit_temp" || true; fi
    if test "$task_success" -ne 1 && test "$task_started" -eq 1; then
        sudo -n systemctl stop "$task_unit" || true
        printf '%s\n' 'Stopped only the new test service; original proxies/settings remain.'
    fi
    exit "$task_rc"
}
trap cleanup EXIT
if test -e "$task_profile" || test -L "$task_profile"; then
    "$task_python" -B tools/prepare_migrated_proxy.py --verify-only
else
    "$task_python" -B tools/prepare_migrated_proxy.py --prepare
fi
test -x /usr/sbin/cntlm || fail 'Native /usr/sbin/cntlm is unavailable.'
"$task_python" -B publish_twc_v1_k8_ul_propagation_rest.py --verify-only \
  --export-dir "$task_repo/runs/twc_v1_k8_ul_propagation_v1/reports/attempt_76f6300c2486"
sudo -v
task_unit_temp="$(mktemp /tmp/precoder-proxy-unit.XXXXXXXX)"
cat > "$task_unit_temp" <<'UNIT'
[Unit]
Description=Private migrated precoder proxy on loopback port 3130
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=rong
WorkingDirectory=/home/rong
ExecStart=/usr/sbin/cntlm -f -c /home/rong/precoder_proxy_migrated_v1/cntlm-3130.conf
Restart=on-failure
RestartSec=3
UMask=0077
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
UNIT
if sudo -n test -e "$task_unit_file" || sudo -n test -L "$task_unit_file"; then
    sudo -n cmp -s "$task_unit_temp" "$task_unit_file" || fail 'Existing service differs; preserved.'
else
    sudo -n install -o root -g root -m 644 "$task_unit_temp" "$task_unit_file"
fi
sudo -n systemctl daemon-reload
task_dropins="$(sudo -n systemctl show "$task_unit" --property=DropInPaths --value)"
test -z "$task_dropins" || fail 'Unexpected service overrides; preserved.'
if ! sudo -n systemctl is-active --quiet "$task_unit"; then
    "$task_python" -B - <<'PY'
import socket
with socket.socket() as probe:
    probe.bind(('127.0.0.1', 3130))
PY
    task_started=1
    sudo -n systemctl start "$task_unit"
fi
"$task_python" -B - <<'PY'
import socket, time
for attempt in range(40):
    try:
        with socket.create_connection(('127.0.0.1',3130), timeout=1):
            break
    except OSError:
        time.sleep(.1)
else:
    raise SystemExit('Port 3130 did not become ready.')
PY
"$task_python" -B - <<'PY'
import re, subprocess
unit='precoder-proxy-migrated.service'
subprocess.run(['sudo','-n','systemctl','is-active','--quiet',unit],check=True)
pid=subprocess.run(['sudo','-n','systemctl','show',unit,'--property=MainPID','--value'],
    check=True,capture_output=True,text=True).stdout.strip()
result=subprocess.run(['sudo','-n','ss','-H','-ltnp','( sport = :3130 )'],
    check=True,capture_output=True,text=True)
rows=[row.split() for row in result.stdout.splitlines()]
if (len(rows)!=1 or len(rows[0])<5 or rows[0][3]!='127.0.0.1:3130'
        or not pid.isdecimal() or int(pid)<1
        or pid not in re.findall(r'pid=(\d+)',result.stdout)):
    raise SystemExit('Unexpected listener scope; stopped.')
print('PASS: test service owns sole loopback listener 127.0.0.1:3130')
PY
unset GIT_SSL_NO_VERIFY GIT_SSL_CAPATH
export GIT_SSL_CAINFO="$task_profile/ca-bundle.pem"
export no_proxy=localhost,127.0.0.1,::1
export NO_PROXY="$no_proxy"
export GIT_CONFIG_COUNT=4
export GIT_CONFIG_KEY_0=http.https://github.com/.proxy
export GIT_CONFIG_VALUE_0=http://127.0.0.1:3130
export GIT_CONFIG_KEY_1=http.https://github.com/.sslCAInfo
export GIT_CONFIG_VALUE_1="$task_profile/ca-bundle.pem"
export GIT_CONFIG_KEY_2=http.https://github.com/.sslVerify
export GIT_CONFIG_VALUE_2=true
export GIT_CONFIG_KEY_3=remote.origin.proxy
export GIT_CONFIG_VALUE_3=http://127.0.0.1:3130
"$task_python" -B - <<'PY'
from pathlib import Path
import publish_twc_v1_k8_ul_propagation_rest as rest
settings=rest._network_config(rest.previous._config().ROOT)
expected=Path.home()/'precoder_proxy_migrated_v1/ca-bundle.pem'
if settings['proxy']!='http://127.0.0.1:3130' or settings['cafile']!=str(expected):
    raise SystemExit('Selected proxy/CA does not match the migrated test profile.')
print('PASS: test requests select port 3130 and the dedicated CA bundle')
PY
timeout 30s git ls-remote origin refs/heads/main
if ! "$task_python" -B - <<'PY'
import publish_twc_v1_k8_ul_propagation_rest as rest
try:
    rest._cached_credentials(rest.previous._config().ROOT)
except rest.RESTError:
    raise SystemExit(1)
PY
then
    printf '%s\n' 'GitHub credential cache unavailable: enter GitHub username/token at the Git prompt.'
    git push --dry-run origin \
      24209c2e2b4defdddfad0bfb0b594c9d28b898eb:refs/heads/results/twc_v1_k8_ul_propagation_v1/76f6300c2486
fi
"$task_python" -B publish_twc_v1_k8_ul_propagation_rest.py \
  --publish --export-dir "$task_repo/runs/twc_v1_k8_ul_propagation_v1/reports/attempt_76f6300c2486"
task_success=1
printf '%s\n' 'PASS: migrated proxy read and exact result upload. Test service stays running; default Git settings unchanged.'
