"""Private CNTLM profile preparation with synthetic archives and local checks.

No real credentials, existing proxy configuration, SSH, or external HTTP is used.
The one port-conflict check only holds a local loopback socket while preparing.
"""
from contextlib import redirect_stderr, redirect_stdout
import errno
import getopt
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import socket
import ssl
import stat
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1]/'tools'/'prepare_migrated_proxy.py'
_spec = importlib.util.spec_from_file_location('_prepare_migrated_proxy_test', SOURCE)
profile = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = profile
_spec.loader.exec_module(profile)

FORMAT = 'precoder_private_network_export_v1'
OPTIONS = 'a:c:d:fghIl:p:r:su:vw:A:BD:F:G:HL:M:N:O:P:R:S:T:U:'
CONFIG = (b'# private fixture, never an installed profile\n'
    b'Username FAKE_SOURCE_USER\nDomain FAKE_SOURCE_DOMAIN\nAuth NTLMv2\n'
    b'PassNT 11111111111111111111111111111111\n'
    b'PassNTLMv2 22222222222222222222222222222222\n'
    b'Proxy old-parent.example.invalid:8080\nNoProxy localhost,127.0.0.*\n'
    b'Gateway yes\nListen 0.0.0.0:3128\n')


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2)+'\n').encode()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class PrepareMigratedProxyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='private-proxy-profile-test.')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive = self.root/'old-network-private.tar'
        self.target = self.root/'new-profile'
        ca_path = ssl.get_default_verify_paths().cafile or ssl.get_default_verify_paths().openssl_cafile
        blocks = re.findall(rb'-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----',
            Path(ca_path).read_bytes(), flags=re.S)
        self.assertGreaterEqual(len(blocks), 3)
        self.certificates = [block+b'\n' for block in blocks[:3]]
        self.system_ca = self.root/'new-system-ca.pem'
        self.system_ca.write_bytes(self.certificates[1])
        self.system_ca.chmod(0o600)
        self.write_archive()

    @staticmethod
    def inventory(argv):
        options, remaining = getopt.gnu_getopt(argv[1:], OPTIONS)
        passwords = [value for option, value in options if option == '-p']
        interactive = any(option == '-I' for option, _ in options)
        return dict(flag_names=[option for option, _ in options],
            password_override_present=bool(passwords), interactive_password_present=interactive,
            allstar_password_masked=any(value and set(value) == {'*'} for value in passwords),
            auth_cli_requires_local_reentry=bool(passwords) or interactive,
            remaining_argument_count=len(remaining), argv_is_private_reference_not_executable=True)

    def write_archive(self, *, changes=None, argv=None, manifest_change=None, special=None):
        argv = argv or ['/usr/sbin/cntlm', '-U', 'cntlm', '-P', '/run/source-cntlm.pid']
        inventory = self.inventory(argv)
        data = {
            'snapshots/etc/cntlm.conf': CONFIG,
            'snapshots/etc/ssl/certs/ca-certificates.crt': self.certificates[0],
            'snapshots/usr/local/share/ca-certificates/fixture-extra.crt': self.certificates[2],
            'process/cmdline.bin': b'\0'.join(value.encode() for value in argv)+b'\0',
            'process/process.json': encoded(dict(pid=4321, argv=argv,
                executable='/usr/sbin/cntlm', cwd='/', inventory=inventory)),
            'process/status.txt': b'Name:\tcntlm\n',
            'process/cgroup.txt': b'fixture cgroup\n',
            'process/stat.txt': b'fixture process\n',
            'reference/user_context.json': encoded(dict(format=FORMAT+'_context', uid=1000,
                home='/home/r50037942', proxy_environment={}, git_https_settings={})),
            'reference/listener.json': encoded(dict(returncode=0, stdout='', stderr='')),
            'reference/systemd_show.json': encoded(dict(returncode=0, stdout='', stderr='')),
        }
        data.update(changes or {})
        for name in [name for name, value in data.items() if value is None]:
            del data[name]
        manifest = dict(format=FORMAT, private=True, activate=False,
            owner=dict(uid=1000, gid=1000, username='r50037942', home='/home/r50037942'),
            source_listener_port=3128, source_config='/etc/cntlm.conf',
            process_inventory=inventory, skipped=[], source_metadata={},
            files={name: dict(size=len(value), sha256=hashlib.sha256(value).hexdigest())
                   for name, value in data.items()})
        if manifest_change:
            manifest_change(manifest)
        data['manifest.json'] = encoded(manifest)
        with tarfile.open(self.archive, 'w') as stream:
            for name, value in data.items():
                entry = tarfile.TarInfo(name)
                entry.mode, entry.uid, entry.gid = 0o600, 1000, 1000
                entry.size = len(value)
                if special and name in special:
                    entry.type = special[name]
                    entry.linkname = '../../outside'
                    if entry.type in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                        entry.size = 0
                stream.addfile(entry, io.BytesIO(value) if entry.size else None)
        self.archive.chmod(0o600)
        self.source_sha = digest(self.archive)
        self.contents = data

    def build(self, **overrides):
        archive = overrides.pop('archive', self.archive)
        options = dict(target=self.target, source_sha=self.source_sha, system_ca=self.system_ca,
            require_owner=os.getuid(), probe_port=False)
        options.update(overrides)
        return profile.build_profile(archive, **options)

    def verify(self):
        return profile.verify_profile(self.archive, target=self.target, source_sha=self.source_sha,
            system_ca=self.system_ca, require_owner=os.getuid())

    def denied(self, **overrides):
        before = digest(overrides.get('archive', self.archive))
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            with self.assertRaises(profile.PrepareError) as caught:
                self.build(**overrides)
        text = str(caught.exception)+output.getvalue()+errors.getvalue()
        for secret in ('FAKE_SOURCE_USER', 'FAKE_SOURCE_DOMAIN', '1111111111111111',
                       '2222222222222222', 'FAKE_CLI_SECRET'):
            self.assertNotIn(secret, text)
        self.assertEqual(digest(overrides.get('archive', self.archive)), before)
        self.assertFalse(self.target.exists())
        return caught.exception

    def test_private_profile_preserves_auth_parent_and_original_evidence_without_starting(self):
        original = self.archive.read_bytes()
        output, errors = io.StringIO(), io.StringIO()
        with mock.patch('subprocess.run', side_effect=AssertionError('must not run old argv')):
            with mock.patch('socket.socket', side_effect=AssertionError('fixture disabled port probe')):
                with redirect_stdout(output), redirect_stderr(errors):
                    result = self.build()
        self.assertEqual(result['status'], 'prepared_not_started')
        self.assertFalse(result['started'])
        self.assertFalse(result['existing_proxies_changed'])
        self.assertEqual(result['port'], 3130)
        self.assertEqual(self.archive.read_bytes(), original)
        self.assertEqual(Path(result['source_config_path']).read_bytes(), CONFIG)
        self.assertEqual(Path(result['source_cmdline_path']).read_bytes(), self.contents['process/cmdline.bin'])
        config = Path(result['config_path']).read_bytes()
        for line in CONFIG.splitlines(keepends=True):
            if line and not line.lower().startswith((b'gateway ', b'listen ')):
                self.assertIn(line, config)
        directives = [line.split(None, 1) for line in config.splitlines()
            if line.strip() and not line.lstrip().startswith(b'#')]
        listens = [fields[1] for fields in directives if fields[0].lower() == b'listen']
        self.assertEqual(listens, [b'127.0.0.1:3130'])
        self.assertEqual([fields[1] for fields in directives if fields[0].lower() == b'gateway'], [b'no'])
        self.assertFalse(any(fields[0].lower() in (b'tunnel', b'socks5proxy')
                             for fields in directives))
        self.assertNotIn(b'/run/source-cntlm.pid', config)
        self.assertEqual(stat.S_IMODE(self.target.stat().st_mode), 0o700)
        self.assertTrue(all(stat.S_IMODE(path.stat().st_mode) == 0o600
            for path in self.target.rglob('*') if path.is_file()))
        self.assertEqual(output.getvalue()+errors.getvalue(), '')
        self.assertFalse(any(value in json.dumps(result) for value in
            ('FAKE_SOURCE_USER', 'FAKE_SOURCE_DOMAIN', '1111111111111111', '2222222222222222')))

    def test_new_and_old_public_certificates_form_a_strict_dedicated_ssl_context(self):
        before = self.system_ca.read_bytes()
        result = self.build()
        bundle = Path(result['ca_bundle_path'])
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations(cafile=str(bundle))
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        actual = {hashlib.sha256(cert).digest() for cert in context.get_ca_certs(binary_form=True)}
        expected = {hashlib.sha256(ssl.PEM_cert_to_DER_cert(cert.decode())).digest()
            for cert in self.certificates}
        self.assertTrue(expected.issubset(actual))
        self.assertEqual(self.system_ca.read_bytes(), before)
        self.assertNotEqual(bundle.resolve(), self.system_ca.resolve())

    def test_wrong_outer_hash_and_manifest_hash_size_or_membership_are_rejected(self):
        self.denied(source_sha='0'*64)
        changes = [
            lambda m: m['files']['snapshots/etc/cntlm.conf'].update(sha256='0'*64),
            lambda m: m['files']['snapshots/etc/cntlm.conf'].update(size=len(CONFIG)+1),
            lambda m: m['files'].pop('snapshots/etc/cntlm.conf'),
            lambda m: m['files'].update({'missing.txt': dict(size=0, sha256=hashlib.sha256(b'').hexdigest())}),
        ]
        for change in changes:
            with self.subTest(change=changes.index(change)):
                self.write_archive(manifest_change=change)
                self.denied()

    def test_traversal_absolute_backslash_and_duplicate_members_are_rejected(self):
        for name in ('../escape', '/absolute', 'snapshots/../escape', 'bad\\name'):
            with self.subTest(name=name):
                self.write_archive(changes={name: b'x'})
                self.denied()
        self.write_archive()
        with tarfile.open(self.archive, 'a') as stream:
            entry = tarfile.TarInfo('snapshots/etc/cntlm.conf')
            entry.size = len(CONFIG)
            stream.addfile(entry, io.BytesIO(CONFIG))
        self.source_sha = digest(self.archive)
        self.denied()

    def test_symlink_hardlink_and_gnu_sparse_members_are_rejected(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.GNUTYPE_SPARSE):
            with self.subTest(kind=kind):
                self.write_archive(special={'snapshots/etc/cntlm.conf': kind})
                self.denied()

    def test_original_user_and_unexpected_source_cli_overrides_are_rejected(self):
        for key, value in (('username', 'rong'), ('home', '/home/rong')):
            self.write_archive(manifest_change=lambda m: m['owner'].update({key: value}))
            self.denied()
        for extra in (['-u', 'FAKE_CLI_SECRET'], ['-p', 'FAKE_CLI_SECRET'], ['-I'], ['-v'],
                      ['-T', '/tmp/source-trace'], ['-l', '0.0.0.0:9999'],
                      ['-L', '9999:example.invalid:80']):
            with self.subTest(flag=extra[0]):
                self.write_archive(argv=['/usr/sbin/cntlm', '-U', 'cntlm', '-P', '/run/source-cntlm.pid', *extra])
                self.denied()

    def test_unexpected_extra_listener_directive_and_duplicate_config_are_rejected(self):
        for extra in (b'Tunnel 9999:example.invalid:80\n', b'SOCKS5Proxy 9999\n',
                      b'Listen 127.0.0.1:9999\n', b'Gateway no\n', b'UnknownPrivate FAKE_CLI_SECRET\n'):
            self.write_archive(changes={'snapshots/etc/cntlm.conf': CONFIG+extra})
            self.denied()

    def test_ca_private_key_material_is_rejected_without_global_trust_changes(self):
        original = self.system_ca.read_bytes()
        self.system_ca.write_bytes(original+b'-----BEGIN PRIVATE KEY-----\nFAKE\n')
        self.denied()
        self.system_ca.write_bytes(original)
        self.write_archive(changes={'snapshots/usr/local/share/ca-certificates/fixture-extra.crt':
            self.certificates[2]+b'-----BEGIN PRIVATE KEY-----\nFAKE\n'})
        self.denied()

    def test_target_is_never_overwritten_and_source_owner_is_checked(self):
        self.denied(require_owner=os.getuid()+1)
        self.target.mkdir(mode=0o700)
        sentinel = self.target/'cntlm-3130.conf'
        sentinel.write_bytes(b'preserve existing profile\n')
        with self.assertRaises(profile.PrepareError):
            self.build()
        self.assertEqual(sentinel.read_bytes(), b'preserve existing profile\n')

    def test_a_real_occupied_loopback_port_is_rejected_before_any_profile_is_written(self):
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            try:
                holder.bind(('127.0.0.1', 3130))
                holder.listen(1)
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise
            self.denied(probe_port=True)
        finally:
            holder.close()

    def test_per_file_and_archive_caps_reject_large_payloads(self):
        self.write_archive(changes={'reference/oversized.txt': b'x'*(8*1024*1024+1)})
        self.assertEqual(str(self.denied()), 'UNSAFE_ARCHIVE_MEMBER')
        self.archive.write_bytes(b'x'*(32*1024*1024+1))
        self.source_sha = digest(self.archive)
        self.assertEqual(str(self.denied()), 'PRIVATE_REGULAR_INPUT_REQUIRED')

    def test_production_prepare_refuses_the_wrong_account_before_reading_or_writing(self):
        account = SimpleNamespace(pw_uid=os.getuid(), pw_name='wrong-account', pw_dir='/home/wrong-account')
        with mock.patch.object(profile.pwd, 'getpwuid', return_value=account):
            with mock.patch.object(profile, 'build_profile', side_effect=AssertionError('account guard missing')):
                with self.assertRaises(profile.PrepareError) as caught:
                    profile.prepare(self.archive)
        self.assertEqual(str(caught.exception), 'NEW_RONG_ACCOUNT_REQUIRED')

    def test_verify_only_checks_existing_profile_without_binding_executing_or_writing(self):
        prepared = self.build()
        before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns,
            path.stat().st_ctime_ns, stat.S_IMODE(path.stat().st_mode))
            for path in self.target.iterdir()}
        archive_before = digest(self.archive)
        with mock.patch('subprocess.run', side_effect=AssertionError('verify executed a command')):
            with mock.patch('socket.socket', side_effect=AssertionError('verify bound a port')):
                with mock.patch.object(profile, '_write_private', side_effect=AssertionError('verify wrote a file')):
                    verified = self.verify()
        self.assertEqual(verified['status'], 'verified_not_started')
        self.assertEqual(verified['config_path'], prepared['config_path'])
        self.assertFalse(verified['started'])
        self.assertEqual(before, {path.name: (path.read_bytes(), path.stat().st_mtime_ns,
            path.stat().st_ctime_ns, stat.S_IMODE(path.stat().st_mode))
            for path in self.target.iterdir()})
        self.assertEqual(digest(self.archive), archive_before)

    def test_verify_only_rejects_tampered_content_permissions_inventory_and_links(self):
        prepared = self.build()
        config = Path(prepared['config_path'])
        original = config.read_bytes()
        config.write_bytes(original+b'# changed after preparation\n')
        with self.assertRaises(profile.PrepareError):
            self.verify()
        self.assertEqual(config.read_bytes(), original+b'# changed after preparation\n')
        config.write_bytes(original)
        config.chmod(0o644)
        with self.assertRaises(profile.PrepareError):
            self.verify()
        self.assertEqual(stat.S_IMODE(config.stat().st_mode), 0o644)
        config.chmod(0o600)
        extra = self.target/'unexpected.txt'
        extra.write_bytes(b'do not delete user files\n')
        with self.assertRaises(profile.PrepareError):
            self.verify()
        self.assertEqual(extra.read_bytes(), b'do not delete user files\n')
        extra.unlink()
        config.unlink()
        config.symlink_to(Path(prepared['source_config_path']))
        with self.assertRaises(profile.PrepareError):
            self.verify()
        self.assertTrue(config.is_symlink())

    def test_invalid_command_arguments_do_not_echo_arbitrary_secret_values(self):
        output, errors = io.StringIO(), io.StringIO()
        with mock.patch.object(profile, 'prepare', side_effect=AssertionError('invalid CLI reached preparation')):
            with redirect_stdout(output), redirect_stderr(errors):
                result = profile.main(['--prepare', '--password', 'FAKE_CLI_SECRET'])
        self.assertEqual(result, 1)
        self.assertNotIn('FAKE_CLI_SECRET', output.getvalue()+errors.getvalue())

    def test_source_sections_missing_final_newline_and_legacy_gateway_are_preserved_safely(self):
        original = b'[Network]\n'+CONFIG.replace(b'Gateway yes\n', b'Gateway false\n').rstrip(b'\n')
        self.write_archive(changes={'snapshots/etc/cntlm.conf': original})
        result = self.build()
        self.assertEqual(Path(result['source_config_path']).read_bytes(), original)
        config = Path(result['config_path']).read_bytes()
        self.assertIn(b'[Network]\n', config)
        self.assertIn(b'Username FAKE_SOURCE_USER\n', config)
        self.assertIn(b'Proxy old-parent.example.invalid:8080\n', config)
        self.assertNotIn(b'Gateway false', config)
        self.assertEqual(config.count(b'Gateway no\n'), 1)
        self.assertEqual(config.count(b'Listen 127.0.0.1:3130\n'), 1)
        self.assertEqual(self.verify()['status'], 'verified_not_started')

    def test_missing_optional_ca_reference_is_private_metadata_and_never_reused(self):
        missing = '/home/r50037942/private-old-path/missing-ca.pem'
        context = json.loads(self.contents['reference/user_context.json'])
        context['proxy_environment']['GIT_SSL_CAINFO'] = missing
        self.write_archive(changes={'reference/user_context.json': encoded(context)})
        result = self.build()
        self.assertEqual(result['missing_ca_reference_count'], 1)
        metadata = json.loads(Path(result['private_manifest_path']).read_bytes())
        self.assertEqual(metadata['missing_ca_reference_paths'], [missing])
        self.assertNotIn(missing, profile.format_safe_status(result))
        self.assertNotIn(missing.encode(), Path(result['ca_bundle_path']).read_bytes())
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations(cafile=result['ca_bundle_path'])
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.assertEqual(self.verify()['missing_ca_reference_count'], 1)


if __name__ == '__main__':
    unittest.main()
