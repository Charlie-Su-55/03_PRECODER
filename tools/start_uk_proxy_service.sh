#!/usr/bin/env bash
# Destination-only UK proxy; original3129 survives all preparation/test failures.
set +x
set -Ee -o pipefail
fail() { printf 'STOP: %s\n' "$*" >&2; exit 1; }
test "$(id -un)" = rong || fail 'Run on the new rong account.'
test "$HOME" = /home/rong || fail 'Unexpected destination home.'
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate precoder
task_repo=/home/r50037942/project/03_PRECODER
cd -P "$task_repo"
test "$(git remote get-url origin)" = https://github.com/Charlie-Su-55/03_PRECODER.git || fail 'Official03 origin required.'
task_python="$CONDA_PREFIX/bin/python"
task_profile="$HOME/precoder_proxy_uk_v1"
task_unit=precoder-proxy-uk.service
task_unit_file="/etc/systemd/system/$task_unit"
task_temp=''
task_started=0
task_enabled=0
task_git_changed=0
task_backup=''
task_config=''
task_success=0
cleanup() {
    task_rc=$?
    trap - EXIT
    if test "$task_success" -ne 1; then
        if test "$task_git_changed" -eq 1; then
            "$task_python" -B - "$task_backup/git-config" "$task_config" <<'PY' || printf 'Git restore failed; private backup: %s\n' "$task_backup/git-config" >&2
from pathlib import Path
import os, sys, tempfile
source, target = map(Path, sys.argv[1:])
fd, name = tempfile.mkstemp(prefix='uk_proxy_restore_', dir=target.parent)
try:
    with os.fdopen(fd, 'wb') as stream:
        stream.write(source.read_bytes()); stream.flush(); os.fsync(stream.fileno())
    os.replace(name, target)
finally:
    if os.path.exists(name): os.unlink(name)
PY
        fi
        if test "$task_enabled" -eq 1; then sudo -n systemctl disable "$task_unit" >/dev/null || true; fi
        if test "$task_started" -eq 1; then sudo -n systemctl stop "$task_unit" || true; fi
        printf '%s\n' 'Test stopped. Original3129 and private profile remain; check the error above.'
    fi
    if test -n "$task_temp"; then rm -f -- "$task_temp" || true; fi
    exit "$task_rc"
}
trap cleanup EXIT
for task_var in "${!GIT_TRACE@}"; do test -z "$task_var" || unset "$task_var"; done
unset GIT_CURL_VERBOSE SSLKEYLOGFILE
sudo -v
if test -e "$task_profile" || test -L "$task_profile"; then
    "$task_python" -B tools/setup_uk_proxy.py --verify-only
else
    "$task_python" -B tools/setup_uk_proxy.py --prepare
fi
task_temp="$(mktemp /tmp/precoder-uk-unit.XXXXXXXX)"
cat > "$task_temp" <<'UNIT'
[Unit]
Description=Private precoder UK proxy on loopback3132
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=rong
WorkingDirectory=/home/rong
ExecStart=/usr/sbin/cntlm -f -c /home/rong/precoder_proxy_uk_v1/cntlm-3132.conf
Restart=on-failure
RestartSec=3
UMask=0077
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
UNIT
if sudo -n test -L "$task_unit_file"; then fail 'Linked service definition refused.'; fi
if sudo -n test -e "$task_unit_file"; then
    sudo -n cmp -s "$task_temp" "$task_unit_file" || fail 'Existing different service preserved.'
else
    sudo -n install -o root -g root -m 644 "$task_temp" "$task_unit_file"
fi
sudo -n systemctl daemon-reload
test -z "$(sudo -n systemctl show "$task_unit" --property=DropInPaths --value)" || fail 'Unexpected service overrides.'
if ! sudo -n systemctl is-active --quiet "$task_unit"; then
    "$task_python" -B - <<'PY'
import socket
with socket.socket() as probe: probe.bind(('127.0.0.1',3132))
PY
    task_started=1
    sudo -n systemctl start "$task_unit"
fi
"$task_python" -B - <<'PY'
import re, socket, subprocess, time
for _ in range(40):
    try:
        with socket.create_connection(('127.0.0.1',3132),timeout=1): break
    except OSError: time.sleep(.1)
else: raise SystemExit('3132 did not become ready.')
def read(args): return subprocess.run(args,capture_output=True,text=True,check=True).stdout
pid=read(['sudo','-n','systemctl','show','precoder-proxy-uk.service','--property=MainPID','--value']).strip()
rows=read(['sudo','-n','ss','-H','-ltnp','( sport = :3132 )'])
parts=[r.split() for r in rows.splitlines()]
if len(parts)!=1 or parts[0][3]!='127.0.0.1:3132' or not pid.isdecimal() or int(pid)<1 or pid not in re.findall(r'pid=(\d+)',rows):
    raise SystemExit('Unexpected3132 listener; original proxy preserved.')
print('PASS: independent service owns loopback3132')
PY
unset GIT_SSL_NO_VERIFY GIT_SSL_CAPATH
export GIT_SSL_CAINFO="$task_profile/ca-bundle.pem"
export no_proxy=localhost,127.0.0.1,::1
export NO_PROXY="$no_proxy"
task_ref="$(timeout 30s git -c http.https://github.com/.proxy=http://127.0.0.1:3132 \
    -c remote.origin.proxy=http://127.0.0.1:3132 \
    -c http.https://github.com/.sslCAInfo="$GIT_SSL_CAINFO" \
    -c http.https://github.com/.sslVerify=true ls-remote origin refs/heads/main)"
[[ "$task_ref" =~ ^[[:xdigit:]]{40}[[:space:]]+refs/heads/main$ ]] || fail 'GitHub main advertisement missing.'
printf 'PASS: strict TLS GitHub read %s\n' "$task_ref"
umask 077
task_backup="$(mktemp -d "$HOME/precoder_git_before_uk3132.XXXXXXXX")"
task_config="$(git rev-parse --git-path config)"
test -f "$task_config" && test ! -L "$task_config" || fail 'Regular repository config required.'
cp -- "$task_config" "$task_backup/git-config"
chmod 600 "$task_backup/git-config"
if ! sudo -n systemctl is-enabled --quiet "$task_unit"; then
    task_enabled=1
    sudo -n systemctl enable "$task_unit"
fi
task_git_changed=1
git config --local remote.origin.proxy http://127.0.0.1:3132
git config --local http.https://github.com/.proxy http://127.0.0.1:3132
git config --local http.https://github.com/.sslCAInfo "$task_profile/ca-bundle.pem"
git config --local http.https://github.com/.sslVerify true
task_success=1
printf 'PASS: UK service enabled;03 Git uses3132. Private config backup: %s\n' "$task_backup/git-config"
printf '%s\n' 'The manual3129 window may now be closed for03 Git. Other programs may still reference3129.'
printf '%s\n' 'Read access is verified; larger-upload403 is not resolved by this setup.'
