#!/usr/bin/env python3
"""Prepare a private UK CNTLM service profile without starting a service.

Only the working3129 account settings are adapted. Passwords enter through
the user's real terminal and the native hash utility's private pseudoterminal;
they never enter argv, environment or a temporary file. Hash output is captured.
"""
from __future__ import annotations

import argparse
import getopt
import hashlib
import json
import os
from pathlib import Path
import pty
import pwd
import re
import shutil
import socket
import ssl
import stat
import subprocess
import sys
import termios

REPO = Path('/home/r50037942/project/03_PRECODER')
PARENT = 'proxyuk.huawei.com:8080'
PROFILE = 'precoder_proxy_uk_v1'
FORMAT = 'precoder_uk_proxy_v1'
OPTIONS = 'a:c:d:fghIl:p:r:su:vw:A:BD:F:G:HL:M:N:O:P:R:S:T:U:'
FILES = {'cntlm-3132.conf', 'ca-bundle.pem', 'profile.json'}


class SetupError(Exception):
    """Fixed error codes only; never attach command output or private values."""


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise SetupError('INVALID_ARGUMENTS')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(args, *, binary=False):
    result = subprocess.run(args, capture_output=True, text=not binary)
    if result.returncode:
        raise SetupError('LOCAL_COMMAND_FAILED')
    return result.stdout


def account_home():
    user = pwd.getpwuid(os.getuid())
    if (user.pw_name != 'rong' or user.pw_dir != '/home/rong'
            or os.geteuid() != os.getuid() or os.getuid() == 0):
        raise SetupError('NEW_RONG_ACCOUNT_REQUIRED')
    return Path(user.pw_dir)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,49}', value):
        raise SetupError('NETWORK_IDENTITY_REQUIRES_REVIEW')
    return value


def working_profile():
    listing = run(['sudo', '-n', 'ss', '-H', '-ltnp', '( sport = :3129 )'])
    rows = [line.split() for line in listing.splitlines()]
    ids = set(re.findall(r'\("cntlm",pid=(\d+),', listing))
    if len(rows) != 1 or len(rows[0]) < 5 or rows[0][3] != '127.0.0.1:3129' or len(ids) != 1:
        raise SetupError('ONE_LOOPBACK_3129_CNTLM_REQUIRED')
    base = '/proc/'+ids.pop()
    if run(['sudo', '-n', 'readlink', base+'/exe']).strip() != '/usr/sbin/cntlm':
        raise SetupError('NATIVE_CNTLM_REQUIRED')
    raw = run(['sudo', '-n', 'cat', base+'/cmdline'], binary=True)
    if len(raw) > 1024*1024 or not raw.endswith(b'\0'):
        raise SetupError('WORKING_ARGUMENT_LAYOUT_CHANGED')
    try:
        options, remaining = getopt.gnu_getopt(raw[:-1].decode().split('\0')[1:], OPTIONS)
    except (UnicodeError, getopt.GetoptError):
        raise SetupError('WORKING_ARGUMENT_LAYOUT_CHANGED') from None
    flags = [key for key, value in options]
    allowed = {'-c', '-f', '-I', '-l', '-u', '-d', '-a', '-w', '-v', '-U', '-P'}
    if (set(flags)-allowed or '-I' not in flags
            or [v for k, v in options if k == '-c'] != ['/dev/null']
            or [v.lower() for v in remaining] not in
                (['proxyuk.huawei.com:8080'], ['proxyuk.huawei.com', '8080'])):
        raise SetupError('WORKING_UK_PROFILE_REQUIRES_REVIEW')
    values = dict(username='', domain='', auth='', workstation='')
    for key, value in options:
        if key == '-u':
            values['username'] = value.split('@', 1)[0]
            if '@' in value:
                values['domain'] = value.split('@', 1)[1]
        elif key in ('-d', '-a', '-w'):
            values[{'-d': 'domain', '-a': 'auth', '-w': 'workstation'}[key]] = value
    for field in ('username', 'domain'):
        identifier(values[field])
    if values['workstation']:
        identifier(values['workstation'])
    if values['auth'].lower() != 'ntlmv2':
        raise SetupError('WORKING_NTLMV2_REQUIRED')
    return values


def ca_bytes():
    origin = 'https://github.com/Charlie-Su-55/03_PRECODER.git'
    if run(['git', '-C', str(REPO), 'remote', 'get-url', 'origin']).strip() != origin:
        raise SetupError('OFFICIAL_03_ORIGIN_REQUIRED')
    verify = subprocess.run(['git', '-C', str(REPO), 'config', '--get-urlmatch',
        'http.sslVerify', origin], capture_output=True, text=True).stdout.strip().lower()
    if os.environ.get('GIT_SSL_NO_VERIFY', '').lower() not in ('', '0', 'false', 'no') or verify in ('false', '0', 'no'):
        raise SetupError('TLS_VERIFICATION_REQUIRED')
    configured = subprocess.run(['git', '-C', str(REPO), 'config', '--get-urlmatch',
        'http.sslCAInfo', origin], capture_output=True, text=True).stdout.strip()
    path = Path(os.environ.get('GIT_SSL_CAINFO') or configured or '/etc/ssl/certs/ca-certificates.crt').expanduser().resolve(strict=True)
    raw = path.read_bytes()
    check_ca(raw)
    return raw


def check_ca(raw):
    if not raw or len(raw) > 8*1024*1024 or b'PRIVATE KEY' in raw:
        raise SetupError('PUBLIC_CA_BUNDLE_REQUIRED')
    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations(cadata=raw.decode('ascii'))
        if not context.get_ca_certs() or not context.check_hostname or context.verify_mode != ssl.CERT_REQUIRED:
            raise ValueError()
    except (ValueError, UnicodeError, ssl.SSLError):
        raise SetupError('PUBLIC_CA_BUNDLE_REQUIRED') from None


def read_password():
    # Never fall back to stdin: it can be a pasted heredoc or a logged pipe.
    with open('/dev/tty', 'r+b', buffering=0) as terminal:
        fd = terminal.fileno()
        if not os.isatty(fd):
            raise SetupError('REAL_TERMINAL_REQUIRED')
        original = termios.tcgetattr(fd)
        hidden = original[:]
        hidden[3] &= ~(termios.ECHO | termios.ECHONL)
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, hidden)
            terminal.write(b'Current3129 network password (local input only): ')
            terminal.flush()
            value = terminal.readline()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, original)
            terminal.write(b'\n')
            terminal.flush()
    value = value.rstrip(b'\r\n')
    # CNTLM0.92.3 fgets uses a50-byte buffer. Leave room for LF and NUL.
    if not value or len(value) > 48 or any(mark in value for mark in (b'\0', b'\r', b'\n')):
        raise SetupError('PASSWORD_LENGTH_OR_FORMAT_UNSUPPORTED')
    return value


def native_hash(identity, *, password_reader=read_password, native='/usr/sbin/cntlm'):
    secret = bytearray(password_reader())
    if not secret or len(secret) > 48 or any(c in secret for c in (0, 10, 13)):
        raise SetupError('PASSWORD_LENGTH_OR_FORMAT_UNSUPPORTED')
    master, slave = pty.openpty()
    child = None
    try:
        attrs = termios.tcgetattr(slave)
        attrs[3] &= ~(termios.ECHO | termios.ECHONL)
        termios.tcsetattr(slave, termios.TCSANOW, attrs)
        args = [native, '-H', '-c', '/dev/null', '-u', identity['username'],
                '-d', identity['domain'], '-a', 'NTLMv2']
        child = subprocess.Popen(args, stdin=slave, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            close_fds=True, env={'PATH': '/usr/sbin:/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C'})
        os.close(slave)
        slave = None
        os.write(master, secret+b'\n')
        stdout, stderr = child.communicate(timeout=10)
        if child.returncode or len(stdout)+len(stderr) > 16384:
            raise SetupError('NATIVE_HASH_FAILED')
        matches = re.findall(rb'(?m)^PassNTLMv2[ \t]+([0-9A-Fa-f]{32})(?:[ \t]+#[^\r\n]*)?\r?$', stdout)
        if len(matches) != 1:
            raise SetupError('NATIVE_HASH_OUTPUT_CHANGED')
        return matches[0].decode('ascii').upper()
    finally:
        secret[:] = b'\0'*len(secret)
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=2)
        os.close(master)
        if slave is not None:
            os.close(slave)


def private_write(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'wb') as stream:
        os.fchmod(stream.fileno(),0o600)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def build_profile(identity, target, bundle, *, password_reader=read_password, native='/usr/sbin/cntlm', probe=True):
    target = Path(target)
    if target.exists() or target.is_symlink():
        raise SetupError('PROFILE_ALREADY_EXISTS_USE_VERIFY_ONLY')
    if any(p.is_symlink() for p in (target.parent, *target.parents)):
        raise SetupError('LINKED_PROFILE_PARENT_FORBIDDEN')
    for field in ('username', 'domain'):
        identifier(identity[field])
    if identity.get('workstation'):
        identifier(identity['workstation'])
    check_ca(bundle)
    if probe:
        with socket.socket() as test:
            test.bind(('127.0.0.1', 3132))
    value = native_hash(identity, password_reader=password_reader, native=native)
    lines = ['Username '+identity['username'], 'Domain '+identity['domain'],
        'Auth NTLMv2', 'PassNTLMv2 '+value, 'Proxy '+PARENT,
        'Gateway no', 'Listen 127.0.0.1:3132', 'NoProxy localhost,127.0.0.*']
    if identity.get('workstation'):
        lines.append('Workstation '+identity['workstation'])
    config = ('\n'.join(lines)+'\n').encode('ascii')
    created = False
    try:
        target.mkdir(mode=0o700)
        created = True
        target.chmod(0o700)
        private_write(target/'cntlm-3132.conf', config)
        private_write(target/'ca-bundle.pem', bundle)
        record = dict(format=FORMAT, port=3132, parent=PARENT, prepared=True,
            source_listener_port=3129, plaintext_password_stored=False,
            ntlmv2_hash_stored=True, strict_tls=True,
            files={'cntlm-3132.conf':digest(config), 'ca-bundle.pem':digest(bundle)})
        private_write(target/'profile.json', (json.dumps(record,indent=2,sort_keys=True)+'\n').encode())
        return verify_profile(target)
    except BaseException:
        if created:
            shutil.rmtree(target)
        raise


def verify_profile(target):
    target = Path(target)
    if (target.is_symlink() or not target.is_dir() or target.stat().st_uid != os.getuid()
            or any(p.is_symlink() for p in target.parents)
            or stat.S_IMODE(target.stat().st_mode) != 0o700
            or {p.name for p in target.iterdir()} != FILES):
        raise SetupError('PRIVATE_PROFILE_DIRECTORY_REQUIRED')
    values = {}
    for name in FILES:
        path = target/name
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > 8*1024*1024):
            raise SetupError('PRIVATE_PROFILE_FILE_REQUIRED')
        values[name] = path.read_bytes()
    def unique(pairs):
        result = {}
        for key,value in pairs:
            if key in result:
                raise SetupError('DUPLICATE_PROFILE_FIELD')
            result[key] = value
        return result
    record = json.loads(values['profile.json'], object_pairs_hook=unique)
    fixed = dict(format=FORMAT, port=3132, parent=PARENT, prepared=True,
        source_listener_port=3129, plaintext_password_stored=False,
        ntlmv2_hash_stored=True, strict_tls=True)
    if (set(record) != set(fixed)|{'files'} or any(record.get(k) != v
            or isinstance(v,bool) and type(record.get(k)) is not bool for k,v in fixed.items())):
        raise SetupError('PROFILE_IDENTITY_CHANGED')
    if record['files'] != {name:digest(values[name]) for name in ('cntlm-3132.conf','ca-bundle.pem')}:
        raise SetupError('PROFILE_BYTES_CHANGED')
    parsed = {}
    for line in values['cntlm-3132.conf'].decode('ascii').splitlines():
        key, value = line.split(None, 1)
        if key in parsed:
            raise SetupError('PROFILE_DIRECTIVES_CHANGED')
        parsed[key] = value
    required = {'Username','Domain','Auth','PassNTLMv2','Proxy','Gateway','Listen','NoProxy'}
    if (set(parsed)-{'Workstation'} != required or parsed['Auth'] != 'NTLMv2'
            or parsed['Proxy'] != PARENT or parsed['Gateway'] != 'no'
            or parsed['Listen'] != '127.0.0.1:3132' or parsed['NoProxy'] != 'localhost,127.0.0.*'
            or not re.fullmatch(r'[0-9A-F]{32}',parsed['PassNTLMv2'])):
        raise SetupError('PROFILE_DIRECTIVES_CHANGED')
    for name in ('Username','Domain'):
        identifier(parsed[name])
    if 'Workstation' in parsed:
        identifier(parsed['Workstation'])
    check_ca(values['ca-bundle.pem'])
    return dict(status='verified', directory=str(target), port=3132, parent=PARENT,
        started=False, existing_proxy_changed=False, strict_tls=True)


def main(argv=None):
    parser = Parser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--verify-only', action='store_true')
    try:
        args = parser.parse_args(argv)
        home = account_home()
        target = home/PROFILE
        if args.verify_only:
            result = verify_profile(target)
        else:
            if target.exists() or target.is_symlink():
                raise SetupError('PROFILE_ALREADY_EXISTS_USE_VERIFY_ONLY')
            result = build_profile(working_profile(), target, ca_bytes())
        print(json.dumps(result,indent=2,sort_keys=True))
        return 0
    except KeyboardInterrupt:
        print('UK_PROXY_STOPPED: CANCELLED',file=sys.stderr)
    except SetupError as exc:
        print('UK_PROXY_STOPPED: '+str(exc),file=sys.stderr)
    except Exception:
        print('UK_PROXY_STOPPED: LOCAL_PREPARATION_FAILED',file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
