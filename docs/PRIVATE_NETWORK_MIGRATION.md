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
