# Keep the working UK proxy independent of Remote Desktop

The new server already pulls GitHub code through its current UK proxy3129.
Larger uploads receive a gateway403. The earlier DE test returned407, before
GitHub TLS/authentication. This utility addresses the foreground proxy's
lifetime only; it does not promise to change the UK gateway's upload policy.
There is no need to copy the old server's network account or its saved hashes.

Run only as `rong` on `admin1-ELITE`, in the existing03 repository at
`/home/r50037942/project/03_PRECODER`. Keep3129 running until all checks below
pass. This setup creates a separate loopback service on3132; it never stops,
restarts or modifies3129 and does not touch scientific code, weights or runs.

The Python preparer privately reads only the current3129 CNTLM command-line
identity/authentication settings, verifies UK/NTLMv2 and copies the repository's
current effective public CA bundle. It asks for the same network password
locally through `/dev/tty`, with echo disabled and restored in `finally`.
The password is passed to native CNTLM `-H` through a private echo-disabled
pseudoterminal, never argv, environment or a temporary file. Both native
stdout and stderr are captured: `-H` prints credential hashes and those outputs
must never be forwarded to the terminal or chat. Only its one native-format
`PassNTLMv2` value is retained. There is no `-v`, `-T`, old argv replay, old
password hash, LM hash or NT hash in the new configuration.

The source implementation's50-byte password buffer requires at most48 encoded
bytes, leaving room for newline and NUL; empty/multiline/NUL input is refused.
The setup does not claim to read the3129 process's password or recover its
in-memory authentication. You must enter the current network password locally.

The profile `/home/rong/precoder_proxy_uk_v1` has mode0700. Its three files
`cntlm-3132.conf`, `ca-bundle.pem` and `profile.json` have mode0600. The hash is
credential material; keep this directory private and do not upload it. A file
write failure removes only a directory newly created by that preparation.
An existing profile is never overwritten or silently re-prompted. Verification
checks its exact inventory, permissions, content hashes and fixed loopback/UK
settings; it does not need3129 to remain alive after successful installation.

After merging the operational PR, run on the new server:

```bash
bash <<'UK_SERVICE' || printf '%s\n' '后台代理配置停止，请回传终端错误，不要发送凭据。'
set -Ee -o pipefail
test "$(id -un)" = rong
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate precoder
cd -P /home/r50037942/project/03_PRECODER
git pull --ff-only origin main
"$CONDA_PREFIX/bin/python" -B -m unittest discover -s tests -p 'test_uk_proxy_setup.py'
bash tools/start_uk_proxy_service.sh
UK_SERVICE
```

At the local password prompt enter your company network password used for
CNTLM `-I`, rather than a GitHub token. No credentials belong in chat.

The shell installs an exact, independent `precoder-proxy-uk.service` definition
running native CNTLM as `rong`, checks the actual service PID owns only
`127.0.0.1:3132`, and reads the official GitHub main ref with strict certificate
and hostname verification. Only after that read succeeds does it enable the
service at boot and change this repository's URL/origin proxy to3132.
The previous Git configuration is backed up privately under
`/home/rong/precoder_git_before_uk3132.*/git-config` with mode0600.

Failure restores Git bytes if configuration changes had begun, reverses only
an enable performed by that invocation, and stops only a service it started.
A previously active3132 service is not stopped merely because a later check
fails. Private profiles and original3129 remain. If the file write itself
failed, no incomplete new profile is retained. Wrong-password407 or gateway
errors do not trigger an authentication loop or a protocol downgrade.

Success ends with `PASS: UK service enabled;03 Git uses3132`. Then the manual
3129 proxy window can be closed for03 Git usage. Other programs may still
have proxy environment variables pointing at3129; this does not rewrite
`.bashrc`, Conda, pip, APT, other repositories or system-wide networking.
Computation with installed packages/local data does not need a foreground proxy.

Check the persistent service and Git route without printing private files:

```bash
systemctl is-active precoder-proxy-uk.service
systemctl is-enabled precoder-proxy-uk.service
git -C /home/r50037942/project/03_PRECODER ls-remote origin refs/heads/main
```

`tools/setup_uk_proxy.py --verify-only` is a local, no-start/no-write profile
check. Re-running the full shell verifies and reuses an intact profile and
identical service definition. If a credential needs renewal, preserve the
current private files and inspect the failure before choosing an explicit
renewal step; the tool does not replace them automatically.

Cloud tests use real local PTYs and a synthetic native hash process, public
test CA data, temporary files and fake service/network interfaces. They check
secret output suppression, terminal restoration, private files, no overwrite,
read-only retry and actual Git-config rollback after simulated failures.
Real server CNTLM, sudo/systemd, password validity, TLS read and gateway upload
behavior are not executed in cloud. A successful read does not establish that
larger uploads now work, and no experiment performance is asserted.

The completed local suite passed15 tests in2.192 seconds, including a real
PTY synthetic hash process, actual terminal-echo restoration, restrictive
umask handling, partial-write cleanup, and real Bash/Git-config rollback with
simulated sudo/systemd/network interfaces. Shell syntax also passed. No real
corporate password or native server process was used in those tests.
