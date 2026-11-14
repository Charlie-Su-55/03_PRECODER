#!/usr/bin/env python3
"""Prepare an isolated private CNTLM profile without starting any process.

This adapter accepts the already verified old-server snapshot. It does not
execute its argv or scripts, alter existing proxies, change trust stores or Git
configuration, or copy old Linux account names into the new service identity.
"""
from __future__ import annotations

import argparse
from collections import Counter
import getopt
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import pwd
import re
import shutil
import socket
import ssl
import stat
import tarfile


SOURCE_SHA256 = 'e6ddc8ea0785d3b057b73d47db2ea19f57995b11dd3c4e5a36fe405f88d1071b'
ARCHIVE_FORMAT = 'precoder_private_network_export_v1'
PREPARE_FORMAT = 'precoder_migrated_proxy_v1'
LIMIT = 32 * 1024 * 1024
SINGLE_LIMIT = 8 * 1024 * 1024
SHORT_OPTIONS = 'a:c:d:fghIl:p:r:su:vw:A:BD:F:G:HL:M:N:O:P:R:S:T:U:'
SOURCE_KEYS = frozenset(('Auth', 'Domain', 'Gateway', 'Listen', 'NoProxy',
    'PassNT', 'PassNTLMv2', 'Proxy', 'Username'))
DROP_KEYS = frozenset(value.lower() for value in
    ('Listen', 'Gateway', 'Tunnel', 'SOCKS5Proxy', 'SOCKS5User', 'SOCKS5Users'))
CA_ENV_KEYS = ('GIT_SSL_CAINFO', 'SSL_CERT_FILE', 'REQUESTS_CA_BUNDLE',
    'CURL_CA_BUNDLE', 'CONDA_SSL_VERIFY', 'PIP_CERT')
SAFE_KEYS = ('format', 'status', 'directory', 'config_path', 'ca_bundle_path',
    'private_manifest_path', 'source_config_path', 'source_cmdline_path', 'port',
    'started', 'existing_proxies_changed', 'strict_tls', 'workstation_policy',
    'missing_ca_reference_count')


class PrepareError(Exception):
    """Only fixed, non-sensitive error codes may reach terminal output."""


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise PrepareError('INVALID_COMMAND_ARGUMENTS')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(data):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise PrepareError('DUPLICATE_JSON_FIELD')
            value[key] = item
        return value
    try:
        return json.loads(data, object_pairs_hook=unique)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise PrepareError('INVALID_PRIVATE_METADATA') from None


def _regular_bytes(path, *, limit=SINGLE_LIMIT, require_owner=None, private=False):
    path = Path(path)
    if path.is_symlink():
        raise PrepareError('LINKED_INPUT_FORBIDDEN')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                    or before.st_size > limit or before.st_size < 0
                    or require_owner is not None and before.st_uid != require_owner
                    or private and stat.S_IMODE(before.st_mode) & 0o077):
                raise PrepareError('PRIVATE_REGULAR_INPUT_REQUIRED')
            data = stream.read(limit+1)
            after = os.fstat(stream.fileno())
    except OSError:
        raise PrepareError('LOCAL_INPUT_UNAVAILABLE') from None
    if (len(data) > limit or (before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino,
            after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
        raise PrepareError('LOCAL_INPUT_CHANGED_OR_TOO_LARGE')
    return data


def _member_name(name):
    if (not isinstance(name, str) or not name or '\x00' in name or '\\' in name
            or any(ord(char) < 32 for char in name)):
        raise PrepareError('UNSAFE_ARCHIVE_MEMBER')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name or name == '.':
        raise PrepareError('UNSAFE_ARCHIVE_MEMBER')
    return name


def _archive(archive, expected_sha, require_owner):
    if not isinstance(expected_sha, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_sha):
        raise PrepareError('INVALID_EXPECTED_ARCHIVE_HASH')
    raw = _regular_bytes(archive, limit=LIMIT, require_owner=require_owner, private=True)
    if _sha(raw) != expected_sha:
        raise PrepareError('ARCHIVE_HASH_MISMATCH')
    files, total = {}, 0
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as stream:
            for member in stream:
                name = _member_name(member.name)
                if (name in files or len(files) >= 1024
                        or member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
                        or member.linkname or member.sparse is not None
                        or set(member.pax_headers)-{'path'}
                        or member.pax_headers.get('path', name) != name
                        or member.mode != 0o600 or member.size < 0 or member.size > SINGLE_LIMIT):
                    raise PrepareError('UNSAFE_ARCHIVE_MEMBER')
                total += member.size
                if total > LIMIT:
                    raise PrepareError('ARCHIVE_CONTENT_SIZE_LIMIT')
                extracted = stream.extractfile(member)
                if extracted is None:
                    raise PrepareError('ARCHIVE_MEMBER_UNREADABLE')
                data = extracted.read(SINGLE_LIMIT+1)
                if len(data) != member.size or b'PRIVATE KEY' in data:
                    raise PrepareError('INVALID_OR_FORBIDDEN_ARCHIVE_CONTENT')
                files[name] = data
    except (tarfile.TarError, OSError, EOFError):
        raise PrepareError('INVALID_UNCOMPRESSED_ARCHIVE') from None
    manifest = _json(files.get('manifest.json', b''))
    if not isinstance(manifest, dict):
        raise PrepareError('INVALID_PRIVATE_MANIFEST')
    owner = manifest.get('owner', {})
    inventory = manifest.get('files')
    if (manifest.get('format') != ARCHIVE_FORMAT or manifest.get('private') is not True
            or manifest.get('activate') is not False or manifest.get('source_listener_port') != 3128
            or manifest.get('source_config') != '/etc/cntlm.conf'
            or not isinstance(owner, dict) or owner.get('username') != 'r50037942'
            or owner.get('home') != '/home/r50037942' or not isinstance(inventory, dict)
            or set(files) != set(inventory)|{'manifest.json'}):
        raise PrepareError('PRIVATE_MANIFEST_IDENTITY_OR_INVENTORY_MISMATCH')
    for name, record in inventory.items():
        _member_name(name)
        if (not isinstance(record, dict) or type(record.get('size')) is not int
                or record['size'] != len(files[name])
                or record.get('sha256') != _sha(files[name])):
            raise PrepareError('ARCHIVE_MEMBER_HASH_MISMATCH')
    return files, manifest


def _source_argv(files, manifest):
    raw = files.get('process/cmdline.bin', b'')
    if not raw.endswith(b'\0'):
        raise PrepareError('INVALID_SOURCE_ARGV')
    try:
        argv = [value.decode('utf-8') for value in raw[:-1].split(b'\0')]
        options, remaining = getopt.gnu_getopt(argv[1:], SHORT_OPTIONS)
    except (UnicodeError, getopt.GetoptError):
        raise PrepareError('UNSUPPORTED_SOURCE_ARGV') from None
    if (not argv or Path(argv[0]).name != 'cntlm'
            or [key for key, _ in options] != ['-U', '-P'] or remaining
            or any(not value or '\n' in value or '\r' in value for _, value in options)):
        raise PrepareError('SOURCE_ARGV_LAYOUT_REQUIRES_REVIEW')
    expected = dict(flag_names=['-U', '-P'], password_override_present=False,
        interactive_password_present=False, allstar_password_masked=False,
        auth_cli_requires_local_reentry=False, remaining_argument_count=0,
        argv_is_private_reference_not_executable=True)
    process = _json(files.get('process/process.json', b''))
    if (not isinstance(process, dict) or process.get('argv') != argv
            or process.get('executable') != '/usr/sbin/cntlm'
            or process.get('inventory') != expected or manifest.get('process_inventory') != expected):
        raise PrepareError('SOURCE_PROCESS_INVENTORY_MISMATCH')
    return raw


def _config_lines(raw):
    if not raw or b'\0' in raw:
        raise PrepareError('UNSUPPORTED_SOURCE_CONFIG_TEXT')
    try:
        raw.decode('utf-8')
    except UnicodeError:
        raise PrepareError('UNSUPPORTED_SOURCE_CONFIG_TEXT') from None
    records = []
    for line in raw.splitlines(keepends=True):
        stripped = line.lstrip()
        if not stripped.strip() or stripped.startswith((b'#', b';')):
            records.append((line, None, None))
            continue
        if stripped.startswith(b'['):
            # CNTLM's parser records section names but processes its keys in a
            # single option list. Preserve these inert headings byte for byte.
            records.append((line, None, None))
            continue
        match = re.match(rb'([A-Za-z0-9]+)(.*)', stripped.rstrip(b'\r\n'))
        if not match:
            raise PrepareError('SOURCE_CONFIG_LAYOUT_REQUIRES_REVIEW')
        key = match[1].decode('ascii').lower()
        value = match[2].lstrip()
        if value.startswith(b'"'):
            end = value.find(b'"', 1)
            if end < 0:
                raise PrepareError('SOURCE_CONFIG_LAYOUT_REQUIRES_REVIEW')
            value = value[1:end]
        else:
            value = value.split(b'#', 1)[0].rstrip()
        if not value:
            raise PrepareError('SOURCE_CONFIG_LAYOUT_REQUIRES_REVIEW')
        records.append((line, key, value))
    return records


def _derive_config(raw):
    records = _config_lines(raw)
    counts = Counter(key for _, key, _ in records if key is not None)
    if counts != Counter(key.lower() for key in SOURCE_KEYS):
        raise PrepareError('SOURCE_CONFIG_LAYOUT_REQUIRES_REVIEW')
    values = {key: value for _, key, value in records if key is not None}
    if (values['auth'].lower() not in (b'ntlmv2', b'ntlm2sr', b'nt', b'ntlm', b'lm')
            or any(not re.fullmatch(rb'[0-9A-Fa-f]{32}', values[key]) for key in ('passnt', 'passntlmv2'))):
        raise PrepareError('SOURCE_AUTH_LAYOUT_REQUIRES_REVIEW')
    # Keep every non-binding source line byte for byte, including its original
    # account/domain, hashes, parent proxy, NoProxy, comments and line endings.
    derived = b''.join(line for line, key, _ in records if key not in DROP_KEYS)
    derived += b'\n# Isolated migrated profile; original credentials and parent are unchanged.\n'
    derived += b'Gateway no\nListen 127.0.0.1:3130\n'
    checked = [(key, value) for _, key, value in _config_lines(derived) if key in DROP_KEYS]
    if checked != [('gateway', b'no'), ('listen', b'127.0.0.1:3130')]:
        raise PrepareError('DERIVED_LISTENER_VERIFICATION_FAILED')
    return derived


def _ca_certificates(data):
    if not data or b'PRIVATE KEY' in data:
        raise PrepareError('PUBLIC_CA_CERTIFICATES_REQUIRED')
    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations(cadata=data.decode('ascii') if b'-----BEGIN CERTIFICATE-----' in data else data)
        certificates = context.get_ca_certs(binary_form=True)
        if not certificates or not context.check_hostname or context.verify_mode != ssl.CERT_REQUIRED:
            raise ValueError()
    except (OSError, ValueError, UnicodeError, ssl.SSLError):
        raise PrepareError('PUBLIC_CA_CERTIFICATES_REQUIRED') from None
    return certificates


def _ca_bundle(files, system_ca):
    fresh = _regular_bytes(Path(system_ca).resolve(), limit=SINGLE_LIMIT)
    candidates = {'snapshots/etc/ssl/certs/ca-certificates.crt'}
    candidates.update(name for name in files if name.startswith('snapshots/usr/local/share/ca-certificates/')
        and Path(name).suffix.lower() in ('.crt', '.pem', '.cer'))
    context = _json(files.get('reference/user_context.json', b''))
    if not isinstance(context, dict) or context.get('format') != ARCHIVE_FORMAT+'_context':
        raise PrepareError('INVALID_SOURCE_USER_CONTEXT')
    environment, git_settings = context.get('proxy_environment', {}), context.get('git_https_settings', {})
    if not isinstance(environment, dict) or not isinstance(git_settings, dict):
        raise PrepareError('INVALID_SOURCE_USER_CONTEXT')
    references = [environment.get(key) for key in CA_ENV_KEYS]+[git_settings.get('sslCAInfo')]
    for reference in references:
        if isinstance(reference, str) and reference.startswith('/'):
            path = PurePosixPath(reference)
            if str(path) != reference or '..' in path.parts or '\\' in reference:
                raise PrepareError('UNSAFE_SOURCE_CA_REFERENCE')
            candidates.add('snapshots/'+reference.lstrip('/'))
    if 'snapshots/etc/ssl/certs/ca-certificates.crt' not in files:
        raise PrepareError('ARCHIVED_CA_REFERENCE_UNAVAILABLE')
    missing = sorted(name.removeprefix('snapshots') for name in candidates if name not in files)
    candidates.intersection_update(files)
    certificates = {}
    for data in [fresh, *(files[name] for name in sorted(candidates))]:
        for der in _ca_certificates(data):
            certificates.setdefault(_sha(der), der)
    bundle = ''.join(ssl.DER_cert_to_PEM_cert(certificates[key])
        for key in sorted(certificates)).encode('ascii')
    _ca_certificates(bundle)
    return bundle, sorted(candidates), len(certificates), missing


def _probe_port():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(('127.0.0.1', 3130))
    except OSError:
        raise PrepareError('PORT_3130_UNAVAILABLE') from None


def _write_private(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)


def _locations(archive, target, require_owner):
    archive, target = Path(archive).absolute(), Path(target).absolute()
    for path in (archive, target.parent):
        if any(candidate.is_symlink() for candidate in (path, *path.parents)):
            raise PrepareError('LINKED_INPUT_OR_TARGET_FORBIDDEN')
    if (not target.parent.is_dir() or require_owner is not None
            and target.parent.stat().st_uid != require_owner):
        raise PrepareError('OWNED_TARGET_PARENT_REQUIRED')
    return archive, target


def _profile_contents(archive, source_sha, system_ca, require_owner):
    files, manifest = _archive(archive, source_sha, require_owner)
    argv = _source_argv(files, manifest)
    original = files.get('snapshots/etc/cntlm.conf', b'')
    config = _derive_config(original)
    bundle, ca_sources, ca_count, missing_ca = _ca_bundle(files, system_ca)
    output = {'cntlm-3130.conf': config, 'ca-bundle.pem': bundle,
        'source-cntlm.conf': original, 'source-cmdline.bin': argv}
    metadata = dict(format=PREPARE_FORMAT, status='prepared_not_started', started=False,
        source_archive_path=str(archive), source_archive_sha256=source_sha,
        source_manifest_sha256=_sha(files['manifest.json']), ignored_source_flags=['-U', '-P'],
        source_argv_is_private_reference_not_executable=True, port=3130,
        existing_proxies_changed=False, strict_tls=True,
        workstation_policy='source_absent_native_default', ca_certificate_count=ca_count,
        missing_ca_reference_paths=missing_ca, missing_ca_reference_count=len(missing_ca),
        archived_ca_sources=ca_sources, files={name: dict(size=len(data), sha256=_sha(data))
            for name, data in output.items()})
    output['migration.json'] = (json.dumps(metadata, indent=2, sort_keys=True)+'\n').encode()
    return output


def _result(target, *, verified=False, missing_ca_reference_count=0):
    return dict(format=PREPARE_FORMAT,
        status='verified_not_started' if verified else 'prepared_not_started', directory=str(target),
        config_path=str(target/'cntlm-3130.conf'), ca_bundle_path=str(target/'ca-bundle.pem'),
        private_manifest_path=str(target/'migration.json'),
        source_config_path=str(target/'source-cntlm.conf'), source_cmdline_path=str(target/'source-cmdline.bin'),
        port=3130, started=False, existing_proxies_changed=False, strict_tls=True,
        workstation_policy='source_absent_native_default',
        missing_ca_reference_count=missing_ca_reference_count)


def build_profile(archive, *, target, source_sha, system_ca, require_owner=None, probe_port=True):
    """Internal fixture seam; production CLI fixes account, hash and target."""
    archive, target = _locations(archive, target, require_owner)
    if target.exists() or target.is_symlink():
        raise PrepareError('PREPARED_DIRECTORY_ALREADY_EXISTS')
    output = _profile_contents(archive, source_sha, system_ca, require_owner)
    if probe_port:
        _probe_port()
    created = False
    try:
        target.mkdir(mode=0o700)
        created = True
        target.chmod(0o700)
        for name, data in output.items():
            _write_private(target/name, data)
    except OSError:
        if created:
            shutil.rmtree(target)
        raise PrepareError('PRIVATE_PREPARATION_WRITE_FAILED') from None
    return _result(target, missing_ca_reference_count=_json(output['migration.json'])['missing_ca_reference_count'])


def verify_profile(archive, *, target, source_sha, system_ca, require_owner=None):
    """Reconstruct and check private files locally; no bind, write or startup."""
    archive, target = _locations(archive, target, require_owner)
    if (not target.is_dir() or target.is_symlink()
            or stat.S_IMODE(target.stat().st_mode) != 0o700
            or require_owner is not None and target.stat().st_uid != require_owner):
        raise PrepareError('PRIVATE_PREPARED_DIRECTORY_REQUIRED')
    expected = _profile_contents(archive, source_sha, system_ca, require_owner)
    if set(path.name for path in target.iterdir()) != set(expected):
        raise PrepareError('PREPARED_FILE_INVENTORY_MISMATCH')
    for name, data in expected.items():
        path = target/name
        if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise PrepareError('PRIVATE_PREPARED_FILE_REQUIRED')
        actual = _regular_bytes(path, require_owner=require_owner, private=True, limit=LIMIT)
        if actual != data:
            raise PrepareError('PREPARED_FILE_CONTENT_MISMATCH')
    return _result(target, verified=True,
        missing_ca_reference_count=_json(expected['migration.json'])['missing_ca_reference_count'])


def prepare(archive=None, *, verify_only=False):
    account = pwd.getpwuid(os.getuid())
    if (os.geteuid() != os.getuid() or account.pw_uid == 0
            or account.pw_name != 'rong' or account.pw_dir != '/home/rong'):
        raise PrepareError('NEW_RONG_ACCOUNT_REQUIRED')
    home = Path(account.pw_dir)
    action = verify_profile if verify_only else build_profile
    return action(archive or home/'precoder_network_import_v1/old-network-private.tar',
        target=home/'precoder_proxy_migrated_v1', source_sha=SOURCE_SHA256,
        system_ca='/etc/ssl/certs/ca-certificates.crt', require_owner=account.pw_uid)


def format_safe_status(result):
    return json.dumps({key: result[key] for key in SAFE_KEYS}, indent=2, sort_keys=True)


def main(argv=None):
    parser = SafeParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--verify-only', action='store_true')
    parser.add_argument('--archive', type=Path)
    try:
        args = parser.parse_args(argv)
        print(format_safe_status(prepare(args.archive, verify_only=args.verify_only)))
        return 0
    except PrepareError as exc:
        print('PREPARATION_STOPPED: '+str(exc), file=__import__('sys').stderr)
    except Exception:
        print('PREPARATION_STOPPED: LOCAL_VALIDATION_FAILED', file=__import__('sys').stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
