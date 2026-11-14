# Private snapshot of the old server's proxy environment

This utility exports the old server's CNTLM and related proxy/CA settings for
private transfer to the new server. It does not activate a proxy, execute an
old startup script, restart a service or change the network configuration.
It does not read model weights or change any experiment source or result.

The source is the existing loopback CNTLM listener on port3128, using the
default `/etc/cntlm.conf`. The observed service is a generated SysV unit;
`MainPID=0` does not establish who started the current listener. Capture both
the configuration file and the process invocation rather than inferring that
the service definition alone contains all active settings.

The tar archive is **private credential-bearing material**. It can contain
proxy passwords, NTLM credential hashes, environment values and private startup
configuration. Do not upload it or its extracted contents to GitHub or chat.
The public repository contains only the utility, tests and these instructions.
Transfer the real backup privately between machines through SSH/SCP or the
already-authenticated VS Code Remote file interface.

The normal-user context step captures only selected proxy and trust variables
in a private file. The sudo export step checks the original invoking account,
captures the specific CNTLM process/configuration and references, and creates
a mode0700 directory with a mode0600 tar and checksum. Source files remain
unchanged; links, unrelated users' homes, SSH keys and Git credential stores
are excluded. Certificates must be public certificate material. Configuration
and full process arguments are archive-only; terminal output contains paths,
checksums and a restricted status summary.

The archive is a snapshot, not a guarantee of complete in-memory credential
recovery. CNTLM0.92.3 overwrites a `-p` password argument with asterisks, and a
password entered with `-I` is not recoverable from argv. The export flags these
cases for local re-entry rather than treating stars as a working password.
The utility never invokes CNTLM's `-v` debug mode, which may print credentials.

This entry point is deliberately restricted to the known old account
`r50037942` and its home; it refuses to capture the new `rong` account.
The archive includes the system CA bundle, custom public certificates and
explicit CA-file references. CA-directory settings are recorded but their
directories are not generically enumerated; those settings were unset in the
observed old Git configuration. Review the private manifest's skipped entries
and any additional custom script paths before claiming a complete restoration.

After the tool PR is merged, the old server can retrieve its code using
`git fetch origin main` and `git show FETCH_HEAD:tools/export_old_network.py`.
This leaves the source worktree and existing dirty files in place. Run
`python -B export_old_network.py --capture-user-context` as the original user,
then `sudo python3 -B export_old_network.py --user-context <printed-path>`.
Both operations print only safe metadata. Return the terminal metadata, not
the tar or context file, to the conversation for the next migration step.

On the new server, first store and verify the private backup in a mode0700
directory. Keep the current3128/3129 proxies and existing GitHub credential
cache. An adapted proxy should first use a separate loopback port, such as
3130. No captured script, argv or systemd unit is executed automatically.
Do not overwrite the new NIC address, routes, SSH configuration, OS package
sources or Conda initialization with old-machine values. Network
Username/Domain/Workstation are distinct from the new Linux login `rong`;
an NTLMv2 hash is tied to its network identity and cannot be adapted by a
blanket username replacement.

Even an exact proxy configuration copy cannot promise that the upstream
gateway grants the same upload policy to the new machine. Test the same
GitHub request only after staging, checking source overrides and preparing
a reversible new-machine configuration. Read-only cloud fixture tests do not
establish access to either SSH server or success through their real gateways.

Cloud validation passed four unittest cases containing32 explicit fixture
checks (0.075s). These check private archive permissions and hashes, bounded
regular members, excluded credentials/SSH paths, source ownership, symlink,
hardlink and FIFO handling, no secret output, source-account guards and exact
CNTLM option consumption. The 46 diagnostic core files and REST uploader
remain unchanged. The tests used local synthetic source trees, not either
server's real private configuration.

## Stage and test the transferred profile on the new server

The inspected backup has SHA256
`e6ddc8ea0785d3b057b73d47db2ea19f57995b11dd3c4e5a36fe405f88d1071b`.
Its active configuration contains exactly one each of `Auth`, `Domain`,
`Gateway`, `Listen`, `NoProxy`, `PassNT`, `PassNTLMv2`, `Proxy` and `Username`.
The process arguments contain only the old Unix-user and PID-file flags
`-U` and `-P`; no command-line password needs recovery.

`tools/prepare_migrated_proxy.py --prepare` accepts that specific private
archive, only as the destination account `rong` with home `/home/rong`.
It verifies the outer SHA256, complete member inventory and individual hashes
without extracting or executing the archive. The default archive path is
`/home/rong/precoder_network_import_v1/old-network-private.tar`.

Preparation creates `/home/rong/precoder_proxy_migrated_v1` with mode0700 and
five mode0600 files: `cntlm-3130.conf`, `ca-bundle.pem`, `source-cntlm.conf`,
`source-cmdline.bin` and `migration.json`. The source configuration and argv
are private evidence. The derived configuration preserves the network
identity, authentication hashes, parent proxy, `Auth` and `NoProxy` bytes.
It replaces the old bind settings with exactly `Gateway no` and
`Listen 127.0.0.1:3130`. The old `-U`/`-P` arguments are never executed.
No source `Workstation` directive exists, so CNTLM uses the new machine's
native hostname rather than impersonating the old host.

The dedicated CA bundle combines the destination system's public CA
certificates with the archived system/custom/referenced public CAs, deduplicated
and sorted by certificate SHA256. It does not change global trust stores or
disable certificate/hostname verification. Missing optional archived CA-file
references are recorded privately; terminal output gives only their count.
Unknown source options, extra listeners, unsafe archive entries and an
occupied3130 port cause preparation to stop. Existing profiles are never
overwritten. `--verify-only` reconstructs and verifies an existing profile,
including permissions and exact file inventory, without starting, binding,
writing or making network requests.

After this PR is merged, run this block on **the new SSH server**. The earlier
successful private archive transfer and result export must remain in place.
The currently working3129 proxy may be needed for the initial pull.

```bash
bash <<'TEST_MIGRATED_PROXY' || printf '%s\n' '迁移测试停止，请回传终端错误；不要粘贴私有配置。'
set -Ee -o pipefail
test "$(id -un)" = rong
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate precoder
cd -P /home/r50037942/project/03_PRECODER
git pull --ff-only origin main
"$CONDA_PREFIX/bin/python" -B -m unittest discover -s tests -p 'test_prepare_migrated_proxy.py' -v
bash tools/test_migrated_proxy.sh
TEST_MIGRATED_PROXY
```

The launcher validates the profile and the original report export before
starting anything. It installs an isolated systemd unit named
`precoder-proxy-migrated.service`, running native `/usr/sbin/cntlm -f -c ...`
as `rong`. An existing different unit or any drop-in override causes a stop.
The launcher verifies that this unit's actual MainPID owns the sole loopback
listener on3130, then checks GitHub reading and retries the **same complete
report** through the existing REST uploader. Request-scoped Git settings
override both URL and origin proxy selection, and the selected CA/proxy are
checked before upload. Original3128/3129 processes and default Git settings
are preserved.

If the GitHub credential cache has expired, the launcher asks for the GitHub
username and token through a normal Git terminal prompt using `push --dry-run`.
That command sends no result objects or branch update. The token belongs in
that private terminal prompt, never in chat or a shell command. CNTLM debug
mode, Git HTTP tracing and TLS key logging are not used.

On failure, the launcher stops only a test service started by that invocation;
it preserves all reports, private files and pre-existing services. On success,
the test service stays running but is not enabled at boot, and default Git
settings remain unchanged. Success must include `Uploaded branch`, the
verified commit and `PASS: migrated proxy read and exact result upload`.
Return those terminal lines for the full result audit. This operation has zero
training updates and does not rerun the mechanism diagnostic.

## Make the tested proxy persistent

Only after the preceding full upload succeeds, this separate block checks its
receipt, backs up the current repository Git configuration privately, selects
3130 for this repository and enables the tested service at boot. It does not
rewrite `.bashrc`, APT, Conda or other projects' network settings.

```bash
bash <<'KEEP_MIGRATED_PROXY' || printf '%s\n' '持久化停止，请回传终端错误。'
set -Ee -o pipefail
test "$(id -un)" = rong
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate precoder
cd -P /home/r50037942/project/03_PRECODER
"$CONDA_PREFIX/bin/python" -B tools/prepare_migrated_proxy.py --verify-only
"$CONDA_PREFIX/bin/python" -B - <<'PY'
import json
from pathlib import Path
p = Path('runs/twc_v1_k8_ul_propagation_v1/uploads/76f6300c2486.json')
r = json.loads(p.read_text())
expected = {
    'status': 'uploaded', 'remote_ref_verified': True,
    'branch': 'results/twc_v1_k8_ul_propagation_v1/76f6300c2486',
    'commit': '24209c2e2b4defdddfad0bfb0b594c9d28b898eb',
    'export_manifest_sha256': 'b50fdaa80783b8add24cbdbda317fa1be27cfbb92f63fa6bc1382f40ed39250a',
}
if any(r.get(k) != v for k, v in expected.items()):
    raise SystemExit('STOP: full-upload success receipt is missing or differs.')
print('PASS: original complete upload receipt verified')
PY
sudo systemctl is-active --quiet precoder-proxy-migrated.service
umask 077
task_backup="$(mktemp -d "$HOME/precoder_git_before_3130.XXXXXXXX")"
cp -- "$(git rev-parse --git-path config)" "$task_backup/git-config"
chmod 600 "$task_backup/git-config"
printf 'Private Git configuration backup: %s\n' "$task_backup/git-config"
git config --local remote.origin.proxy http://127.0.0.1:3130
git config --local http.https://github.com/.proxy http://127.0.0.1:3130
git config --local http.https://github.com/.sslCAInfo "$HOME/precoder_proxy_migrated_v1/ca-bundle.pem"
git config --local http.https://github.com/.sslVerify true
sudo systemctl enable precoder-proxy-migrated.service
timeout 30s git ls-remote origin refs/heads/main
printf '%s\n' '03 Git now uses the tested independent service; the foreground3129 proxy is no longer needed by this repository.'
KEEP_MIGRATED_PROXY
```

If subsequent checks fail, keep the backup and the existing3128/3129 services.
Restore the saved repository Git configuration locally before stopping the
new unit; do not paste that backup into chat. Other programs that still select
3129 continue to require their existing proxy until separately configured.

The combined cloud suite passed21 tests (0.381s):17 preparation checks and
the four existing export cases. These use synthetic private archives and
local public test CAs. The 46 diagnostic core files, REST uploader and old
exporter are byte-identical to the main branch used for this change.
They cover preservation of authentication/parent settings, private permissions,
strict CA loading, malformed archives, occupied ports, read-only verification
and tamper detection. The launcher receives shell syntax and code review only;
the actual server account, sudo/systemd behavior, corporate authentication,
upstream upload policy and full report upload require the new-server execution
above. A successful profile preparation alone does not establish that the
gateway accepts larger uploads.
