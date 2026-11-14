#!/usr/bin/env python3
"""Private, read-only CNTLM/network snapshot. Never activate or execute a profile.

Normal user: --capture-user-context. Then sudo: --user-context PATH.
All source contents, argv and environment values remain inside a private tar.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import getopt
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import pwd
import re
import shlex
import shutil
import ssl
import stat
import subprocess
import tarfile
import tempfile

FORMAT = 'precoder_private_network_export_v1'
LIMIT = 32 * 1024 * 1024
SINGLE_LIMIT = 8 * 1024 * 1024
PROXY_KEYS = ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
    'all_proxy', 'ALL_PROXY', 'no_proxy', 'NO_PROXY', 'GIT_SSL_CAINFO',
    'GIT_SSL_CAPATH', 'SSL_CERT_FILE', 'REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE',
    'CONDA_SSL_VERIFY', 'PIP_CERT', 'GIT_SSL_NO_VERIFY')
EXPECTED_USER, EXPECTED_HOME = 'r50037942', '/home/r50037942'
# Exact documented short options from CNTLM 0.92.3. No binary is executed.
SHORT_OPTIONS = 'a:c:d:fghIl:p:r:su:vw:A:BD:F:G:HL:M:N:O:P:R:S:T:U:'
NETWORK = re.compile(rb'(?i)(?:proxy|cntlm|3128|3129|ssl|ca_bundle|cacert|certificate|\bcert\b)')
EXCLUDED = frozenset(('.ssh', '.git-credentials', '.netrc', '.git-credential-cache'))


class ExportError(Exception):
    pass


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise ExportError('INVALID_COMMAND_ARGUMENTS')


@dataclass(frozen=True)
class Owner:
    uid: int
    gid: int
    name: str
    home: str


def sha(data):
    return hashlib.sha256(data).hexdigest()


def public_ca(data):
    if b'PRIVATE KEY' in data:
        raise ExportError('PRIVATE_KEY_MATERIAL_FORBIDDEN')
    try:
        cadata = data.decode('ascii') if b'-----BEGIN CERTIFICATE-----' in data else data
        context = ssl.create_default_context(cadata=cadata)
        if not context.get_ca_certs():
            raise ValueError()
    except (ValueError, UnicodeError, ssl.SSLError):
        raise ExportError('PUBLIC_CA_CERTIFICATE_REQUIRED') from None


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True,
                       allow_nan=False)+'\n').encode()


def private_write(path, data, owner=None):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if owner is not None and os.geteuid() == 0:
        os.chown(path, owner.uid, owner.gid)
    os.chmod(path, 0o600)


def checked_path(root, logical):
    if not isinstance(logical, str) or '\x00' in logical or '\\' in logical:
        raise ExportError('UNSAFE_SOURCE_PATH')
    p = PurePosixPath(logical)
    if not p.is_absolute() or '..' in p.parts or str(p) != logical:
        raise ExportError('UNSAFE_SOURCE_PATH')
    actual = Path(root).joinpath(*p.parts[1:])
    for candidate in (actual, *actual.parents):
        if candidate.is_symlink():
            raise ExportError('LINKED_SOURCE_PATH')
        if candidate == Path(root):
            break
    return actual


def read_regular(root, logical, limit=SINGLE_LIMIT):
    actual = checked_path(root, logical)
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    fd = os.open(actual, flags)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
            raise ExportError('SOURCE_NOT_BOUNDED_REGULAR_FILE')
        data = stream.read(limit+1)
        after = os.fstat(stream.fileno())
    if len(data) > limit or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ExportError('SOURCE_CHANGED_OR_TOO_LARGE')
    return data, before


def run_private(arguments):
    try:
        p = subprocess.run(arguments, capture_output=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise ExportError('METADATA_COMMAND_FAILED') from None
    return dict(returncode=p.returncode, stdout=p.stdout.decode(errors='backslashreplace'),
                stderr=p.stderr.decode(errors='backslashreplace'))


def capture_user_context():
    account = pwd.getpwuid(os.getuid())
    if account.pw_uid == 0 or (account.pw_name, account.pw_dir) != (EXPECTED_USER, EXPECTED_HOME):
        raise ExportError('CAPTURE_ON_OLD_SOURCE_ACCOUNT_ONLY')
    home = Path(account.pw_dir)
    if home.resolve() != home:
        raise ExportError('LINKED_USER_HOME')
    output = Path(tempfile.mkdtemp(prefix='precoder_network_context.', dir=home))
    os.chmod(output, 0o700)
    value = dict(format=FORMAT+'_context', uid=account.pw_uid, home=str(home),
        proxy_environment={key: os.environ[key] for key in PROXY_KEYS if key in os.environ},
        conda_prefix=os.environ.get('CONDA_PREFIX'), git_https_settings={})
    repo = home/'project/03_PRECODER'
    git = shutil.which('git')
    if git and (repo/'.git/config').is_file():
        for key in ('sslCAInfo', 'sslCAPath', 'sslVerify', 'proxy'):
            result = run_private([git, '-C', str(repo), 'config', '--get-urlmatch',
                'http.'+key, 'https://github.com/Charlie-Su-55/03_PRECODER.git'])
            if result['returncode'] == 0:
                value['git_https_settings'][key] = result['stdout'].strip()
    destination = output/'user_context.json'
    private_write(destination, encoded(value))
    return destination


def sudo_owner():
    if os.geteuid() != 0:
        raise ExportError('SUDO_REQUIRED')
    try:
        uid = int(os.environ['SUDO_UID'])
        account = pwd.getpwuid(uid)
    except (KeyError, ValueError):
        raise ExportError('ORIGINAL_SUDO_USER_REQUIRED') from None
    if (uid == 0 or os.environ.get('SUDO_USER') != account.pw_name
            or (account.pw_name, account.pw_dir) != (EXPECTED_USER, EXPECTED_HOME)):
        raise ExportError('ORIGINAL_SUDO_USER_REQUIRED')
    return Owner(uid, account.pw_gid, account.pw_name, account.pw_dir)


def discover_listener():
    result = run_private(['ss', '-H', '-ltnp', 'sport = :3128'])
    if result['returncode']:
        raise ExportError('LISTENER_DISCOVERY_FAILED')
    ids = set(re.findall(r'\("cntlm",pid=(\d+),', result['stdout']))
    if len(ids) != 1:
        raise ExportError('ONE_CNTLM_3128_LISTENER_REQUIRED')
    return int(next(iter(ids))), result


def command_inventory(argv):
    try:
        options, remaining = getopt.gnu_getopt(argv[1:], SHORT_OPTIONS)
    except getopt.GetoptError:
        raise ExportError('UNSUPPORTED_CNTLM_ARGUMENT_LAYOUT') from None
    if any(option == '-c' for option, _ in options):
        raise ExportError('DEFAULT_CONFIG_CHANGED_REQUIRES_REVIEW')
    passwords = [value for option, value in options if option == '-p']
    interactive = any(option == '-I' for option, _ in options)
    return dict(flag_names=[option for option, _ in options],
        password_override_present=bool(passwords), interactive_password_present=interactive,
        allstar_password_masked=any(value and set(value) == {'*'} for value in passwords),
        auth_cli_requires_local_reentry=bool(passwords) or interactive,
        remaining_argument_count=len(remaining), argv_is_private_reference_not_executable=True)


class Collector:
    def __init__(self, root, owner, limit=LIMIT):
        self.root, self.owner, self.limit = Path(root), owner, limit
        self.files, self.sources, self.skipped = {}, {}, []

    def put(self, member, data):
        p = PurePosixPath(member)
        if (not p.parts or p.is_absolute() or '..' in p.parts or str(p) != member
                or '\x00' in member or '\\' in member or member in self.files):
            raise ExportError('UNSAFE_OR_DUPLICATE_MEMBER')
        if len(self.files) >= 1024 or sum(map(len, self.files.values()))+len(data) > self.limit:
            raise ExportError('EXPORT_SIZE_LIMIT')
        self.files[member] = data

    def add_file(self, logical, *, required=False, network_only=False, ca_only=False):
        if any(part in EXCLUDED for part in PurePosixPath(logical).parts):
            raise ExportError('EXCLUDED_CREDENTIAL_OR_SSH_PATH')
        home = PurePosixPath(self.owner.home)
        p = PurePosixPath(logical)
        if str(p).startswith(('/home/', '/root/')) and not p.is_relative_to(home):
            raise ExportError('OTHER_USER_PATH_FORBIDDEN')
        member = 'snapshots/'+logical.lstrip('/')
        if member in self.files:
            if ca_only:
                public_ca(self.files[member])
            return self.files[member]
        try:
            data, info = read_regular(self.root, logical)
        except (OSError, ExportError) as exc:
            if required:
                raise ExportError('REQUIRED_SOURCE_UNAVAILABLE') from None
            self.skipped.append(dict(path=logical,
                reason=str(exc) if isinstance(exc, ExportError) else type(exc).__name__))
            return None
        is_user_file = p.is_relative_to(home)
        if ((is_user_file and info.st_uid != self.owner.uid)
                or (not is_user_file and info.st_uid not in (0, self.owner.uid))):
            raise ExportError('SOURCE_OWNER_MISMATCH')
        if b'PRIVATE KEY' in data:
            raise ExportError('PRIVATE_KEY_MATERIAL_FORBIDDEN')
        if network_only and not NETWORK.search(data):
            return None
        if ca_only:
            public_ca(data)
        self.put(member, data)
        self.sources[member] = dict(path=logical, mode=stat.S_IMODE(info.st_mode),
            uid=info.st_uid, gid=info.st_gid, mtime_ns=info.st_mtime_ns)
        return data


def build_export(owner, user_context, *, root='/', pid=None, listener=None,
                 service=None, output_parent=None):
    """Fixture-friendly read-only export; root maps logical absolute paths."""
    collector = Collector(root, owner)
    home_actual = checked_path(root, owner.home)
    if not home_actual.is_dir() or home_actual.stat().st_uid != owner.uid:
        raise ExportError('USER_HOME_UNAVAILABLE')
    context_data, context_stat = read_regular(root, user_context, limit=1024*1024)
    if (not PurePosixPath(user_context).is_relative_to(PurePosixPath(owner.home))
            or context_stat.st_uid != owner.uid or stat.S_IMODE(context_stat.st_mode) & 0o077):
        raise ExportError('PRIVATE_OWNER_CONTEXT_REQUIRED')
    context = json.loads(context_data)
    if (context.get('format') != FORMAT+'_context' or context.get('uid') != owner.uid
            or context.get('home') != owner.home
            or set(context.get('proxy_environment', {}))-set(PROXY_KEYS)):
        raise ExportError('USER_CONTEXT_IDENTITY_MISMATCH')
    collector.put('reference/user_context.json', context_data)
    if pid is None:
        pid, listener = discover_listener()
    if type(pid) is not int or pid < 1:
        raise ExportError('INVALID_PROCESS_ID')
    cmdline, _ = read_regular(root, f'/proc/{pid}/cmdline', limit=1024*1024)
    parts = cmdline.split(b'\0')
    if parts and parts[-1] == b'':
        parts.pop()
    argv = [os.fsdecode(part) for part in parts]
    executable = os.readlink(Path(root)/f'proc/{pid}/exe')
    if executable != '/usr/sbin/cntlm' or not argv or Path(argv[0]).name != 'cntlm':
        raise ExportError('UNEXPECTED_CNTLM_EXECUTABLE')
    inventory = command_inventory(argv)
    collector.put('process/cmdline.bin', cmdline)
    process = dict(pid=pid, argv=argv, executable=executable,
        cwd=os.readlink(Path(root)/f'proc/{pid}/cwd'), inventory=inventory)
    for leaf in ('status', 'cgroup', 'stat'):
        value, _ = read_regular(root, f'/proc/{pid}/{leaf}', limit=1024*1024)
        collector.put('process/'+leaf+'.txt', value)
    collector.put('process/process.json', encoded(process))
    collector.put('reference/listener.json', encoded(listener or {}))
    collector.add_file('/etc/cntlm.conf', required=True)
    for logical in ('/etc/default/cntlm', '/etc/init.d/cntlm'):
        collector.add_file(logical)
    if service is None:
        service = run_private(['systemctl', 'show', 'cntlm.service', '--no-pager',
            '--property=FragmentPath,DropInPaths,EnvironmentFiles,MainPID,ActiveState,SubState,SourcePath,UnitFileState'])
    collector.put('reference/systemd_show.json', encoded(service))
    properties = dict(line.split('=', 1) for line in service.get('stdout', '').splitlines() if '=' in line)
    references = []
    for key in ('FragmentPath', 'SourcePath', 'DropInPaths'):
        references += shlex.split(properties.get(key, ''))
    references += re.findall(r'(-?/[^\s]+)\s+\(ignore_errors=(?:yes|no)\)', properties.get('EnvironmentFiles', ''))
    allowed_system = ('/etc/default/', '/etc/sysconfig/', '/etc/systemd/', '/run/systemd/',
                      '/lib/systemd/', '/usr/lib/systemd/', '/etc/cntlm')
    for logical in references:
        logical = logical.removeprefix('-')
        if not logical.startswith(allowed_system) or '%' in logical:
            collector.skipped.append(dict(path=logical, reason='unresolved_service_reference'))
            continue
        collector.add_file(logical)
    for pattern in ('/etc/systemd/system/cntlm.service.d/*.conf',
                    '/run/systemd/system/cntlm.service.d/*.conf',
                    '/etc/apt/apt.conf.d/*', '/etc/profile.d/*'):
        for actual in sorted(Path(root).glob(pattern.lstrip('/'))):
            logical = '/'+str(actual.relative_to(root))
            collector.add_file(logical, network_only='cntlm.service.d' not in logical)
    for logical in ('/etc/environment', '/etc/profile', '/etc/bash.bashrc', '/etc/apt/apt.conf'):
        collector.add_file(logical, network_only=True)
    for relative in ('.bashrc', '.bash_profile', '.bash_login', '.profile', '.zshrc', '.zprofile',
        '.pam_environment'):
        collector.add_file(owner.home+'/'+relative, network_only=True)
    for relative in ('.curlrc', '.wgetrc', '.condarc', '.config/pip/pip.conf',
        '.pip/pip.conf', '.config/git/config', '.gitconfig', 'project/03_PRECODER/.git/config'):
        collector.add_file(owner.home+'/'+relative)
    for logical in ('/etc/gitconfig', '/etc/pip.conf'):
        collector.add_file(logical)
    prefix = context.get('conda_prefix')
    if isinstance(prefix, str) and prefix.startswith('/'):
        try:
            actual_prefix = checked_path(root, prefix)
            if (not actual_prefix.is_dir() or actual_prefix.stat().st_uid != owner.uid
                    or str(PurePosixPath(prefix)).startswith(('/home/', '/root/'))
                    and not PurePosixPath(prefix).is_relative_to(PurePosixPath(owner.home))):
                raise ExportError('UNOWNED_CONDA_PREFIX')
            for actual in sorted((actual_prefix/'etc/conda/activate.d').glob('*')):
                collector.add_file('/'+str(actual.relative_to(root)), network_only=True)
        except (OSError, ExportError):
            collector.skipped.append(dict(path=prefix, reason='unavailable_or_unowned_conda_prefix'))
    collector.add_file('/etc/ssl/certs/ca-certificates.crt', required=True, ca_only=True)
    ca_root = checked_path(root, '/usr/local/share/ca-certificates')
    if ca_root.is_dir():
        for actual in sorted(ca_root.rglob('*')):
            if actual.suffix.lower() in ('.crt', '.pem', '.cer'):
                collector.add_file('/'+str(actual.relative_to(root)), ca_only=True)
    ca_references = [context.get('proxy_environment', {}).get(key) for key in
        ('GIT_SSL_CAINFO', 'SSL_CERT_FILE', 'REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE',
         'CONDA_SSL_VERIFY', 'PIP_CERT')]
    ca_references.append(context.get('git_https_settings', {}).get('sslCAInfo'))
    for logical in ca_references:
        if isinstance(logical, str) and logical.startswith('/'):
            collector.add_file(logical, ca_only=True)
    manifest = dict(format=FORMAT, private=True, activate=False,
        owner=dict(uid=owner.uid, gid=owner.gid, username=owner.name, home=owner.home),
        source_listener_port=3128, source_config='/etc/cntlm.conf', process_inventory=inventory,
        skipped=collector.skipped, source_metadata=collector.sources,
        files={name: dict(size=len(data), sha256=sha(data)) for name, data in collector.files.items()})
    collector.put('manifest.json', encoded(manifest))
    output = Path(tempfile.mkdtemp(prefix='precoder_private_network.', dir=output_parent or home_actual))
    os.chmod(output, 0o700)
    archive = output/'old-network-private.tar'
    fd = os.open(archive, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as raw, tarfile.open(fileobj=raw, mode='w') as tar:
        for name, data in collector.files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.uid, info.gid = len(data), 0o600, owner.uid, owner.gid
            tar.addfile(info, io.BytesIO(data))
    if archive.stat().st_size > LIMIT:
        archive.unlink()
        raise ExportError('ARCHIVE_SIZE_LIMIT')
    archive_hash = sha(archive.read_bytes())
    checksum = output/(archive.name+'.sha256')
    private_write(checksum, (archive_hash+'  '+archive.name+'\n').encode(), owner)
    if os.geteuid() == 0:
        os.chown(archive, owner.uid, owner.gid)
        os.chown(output, owner.uid, owner.gid)
    os.chmod(archive, 0o600)
    return dict(archive=str(archive), checksum_file=str(checksum), sha256=archive_hash,
        member_names=list(collector.files), safe_summary=dict(source_listener_port=3128,
            source_config='/etc/cntlm.conf', auth_cli_requires_local_reentry=inventory['auth_cli_requires_local_reentry'],
            password_override_present=inventory['password_override_present'],
            interactive_password_present=inventory['interactive_password_present'],
            allstar_password_masked=inventory['allstar_password_masked'], activate=False))


def main(argv=None):
    os.umask(0o077)
    parser = SafeParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--capture-user-context', action='store_true')
    action.add_argument('--user-context', type=Path)
    try:
        args = parser.parse_args(argv)
        if args.capture_user_context:
            print(capture_user_context())
        else:
            owner = sudo_owner()
            print(json.dumps(build_export(owner, str(args.user_context)), sort_keys=True))
        return 0
    except Exception as exc:
        code = str(exc) if isinstance(exc, ExportError) else type(exc).__name__
        print('EXPORT_FAILED: '+code, file=__import__('sys').stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
