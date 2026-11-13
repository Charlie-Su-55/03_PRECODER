"""Four local fixture tests with 32 explicit export and rejection checks.

Only synthetic credentials and public system CA certificates are used. No SSH,
network requests, real network profiles, or source configuration writes occur.
"""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import ssl
import stat
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


def _module_path():
    if os.environ.get('MODULE_PATH'):
        return Path(os.environ['MODULE_PATH'])
    sibling = Path(__file__).with_name('export_old_network.py')
    return sibling if sibling.is_file() else Path(__file__).resolve().parents[1]/'tools'/'export_old_network.py'


SOURCE = _module_path()
_spec = importlib.util.spec_from_file_location('_private_network_export_under_test', SOURCE)
exporter = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = exporter
_spec.loader.exec_module(exporter)


class PrivateNetworkExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='network-export-test.')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.home = '/home/fixture'
        self.owner = exporter.Owner(os.getuid(), os.getgid(), 'fixture', self.home)
        (self.root/self.home.lstrip('/')).mkdir(parents=True, mode=0o700)
        self.context = self.home+'/private/user_context.json'
        self.put(self.context, exporter.encoded(dict(format=exporter.FORMAT+'_context',
            uid=self.owner.uid, home=self.home,
            proxy_environment={'https_proxy': 'http://FAKEUSER:FAKEPASS@old-proxy:3128'})))
        self.put('/etc/cntlm.conf', b'Proxy old-proxy:8080\nUsername FAKEUSER\n'
            b'PassNTLMv2 FAKEHASH\nListen 127.0.0.1:3128\n')
        ca = ssl.get_default_verify_paths().cafile or ssl.get_default_verify_paths().openssl_cafile
        self.put('/etc/ssl/certs/ca-certificates.crt', Path(ca).read_bytes())
        self.put('/proc/4321/cmdline',
            b'/usr/sbin/cntlm\x00-U\x00cntlm\x00-p\x00FAKE-ARGV-PASSWORD\x00')
        for leaf in ('status', 'stat', 'cgroup'):
            self.put('/proc/4321/'+leaf, b'fixture metadata\n')
        (self.root/'proc/4321/exe').symlink_to('/usr/sbin/cntlm')
        (self.root/'proc/4321/cwd').symlink_to('/tmp')

    def put(self, logical, content):
        path = self.root/logical.lstrip('/')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        path.chmod(0o600)
        return path

    def build(self, owner=None):
        return exporter.build_export(owner or self.owner, self.context, root=self.root,
            pid=4321, listener={}, service={})

    def denied(self, callback):
        with self.assertRaises(exporter.ExportError):
            callback()

    def test_private_archive_permissions_hashes_inventory_and_no_secret_output(self):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            result = self.build()
        archive, checksum = Path(result['archive']), Path(result['checksum_file'])
        self.assertEqual(output.getvalue()+errors.getvalue(), '')
        self.assertEqual(stat.S_IMODE(archive.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(checksum.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(archive.parent.stat().st_mode), 0o700)
        self.assertLessEqual(archive.stat().st_size, exporter.LIMIT)
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), result['sha256'])
        with tarfile.open(archive) as stream:
            members = stream.getmembers()
            data = {member.name: stream.extractfile(member).read() for member in members}
        self.assertTrue(all(member.isfile() and not member.issym() and not member.islnk()
            and not PurePosixPath(member.name).is_absolute()
            and '..' not in PurePosixPath(member.name).parts and member.mode == 0o600
            for member in members))
        self.assertTrue(b'FAKE-ARGV-PASSWORD' in data['process/cmdline.bin']
            and b'FAKEHASH' in data['snapshots/etc/cntlm.conf']
            and b'FAKEPASS' in data['reference/user_context.json'])
        self.assertFalse(any(value in json.dumps(result) for value in
            ('FAKEHASH', 'FAKEPASS', 'FAKEUSER', 'FAKE-ARGV-PASSWORD')))
        manifest = json.loads(data['manifest.json'])
        self.assertTrue(all(hashlib.sha256(data[name]).hexdigest() == record['sha256']
            and len(data[name]) == record['size'] for name, record in manifest['files'].items()))

    def test_member_names_and_total_byte_limit(self):
        for member in ('../escape', '/absolute', 'a/../escape', '.', 'a\\b', 'nul\x00name'):
            with self.subTest(member=repr(member)):
                self.denied(lambda: exporter.Collector(self.root, self.owner).put(member, b'x'))
        self.denied(lambda: exporter.Collector(self.root, self.owner, limit=4).put('valid', b'12345'))

    def test_excluded_links_fifo_ownership_and_private_key_sources(self):
        for logical in (self.home+'/.ssh/id_rsa', '/home/another/.gitconfig',
                        self.home+'/.git-credentials'):
            with self.subTest(source=logical):
                self.denied(lambda: exporter.Collector(self.root, self.owner).add_file(logical, required=True))
        (self.root/'home/fixture/link').symlink_to(self.root/'etc/cntlm.conf')
        self.denied(lambda: exporter.read_regular(self.root, self.home+'/link'))
        target = self.put(self.home+'/regular', b'abc')
        os.link(target, self.root/'home/fixture/hardlink')
        self.denied(lambda: exporter.read_regular(self.root, self.home+'/hardlink'))
        os.mkfifo(self.root/'home/fixture/fifo', 0o600)
        # A regression must fail promptly rather than block the entire suite.
        code = '''import importlib.util,sys
s=importlib.util.spec_from_file_location('fifo_export',sys.argv[1])
m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
try:m.read_regular(sys.argv[2],sys.argv[3])
except m.ExportError:sys.exit(0)
sys.exit(1)
'''
        child = subprocess.run([sys.executable, '-B', '-c', code, str(SOURCE),
            str(self.root), self.home+'/fifo'], capture_output=True, timeout=2, check=False)
        self.assertEqual(child.returncode, 0)
        foreign = exporter.Owner(self.owner.uid+1, self.owner.gid, 'foreign', self.home)
        self.denied(lambda: self.build(owner=foreign))
        info = list(os.stat(self.root/'etc/cntlm.conf'))
        info[4] = self.owner.uid+1
        with mock.patch.object(exporter, 'read_regular',
                return_value=(b'Proxy private\n', os.stat_result(info))):
            self.denied(lambda: exporter.Collector(self.root, self.owner).add_file(self.home+'/.bashrc'))
            self.denied(lambda: exporter.Collector(self.root, self.owner).add_file('/etc/cntlm.conf'))
        self.put(self.home+'/privatekey', b'-----BEGIN PRIVATE KEY-----\nFAKE\n')
        self.denied(lambda: exporter.Collector(self.root, self.owner).add_file(self.home+'/privatekey'))

    def test_old_account_guards_and_exact_cntlm_option_consumption(self):
        account = SimpleNamespace(pw_uid=self.owner.uid, pw_gid=self.owner.gid,
            pw_name='rong', pw_dir='/home/rong')
        with mock.patch.object(exporter.pwd, 'getpwuid', return_value=account):
            self.denied(exporter.capture_user_context)
            with mock.patch.object(exporter.os, 'geteuid', return_value=0):
                with mock.patch.dict(os.environ,
                        {'SUDO_UID': str(self.owner.uid), 'SUDO_USER': 'rong'}):
                    self.denied(exporter.sudo_owner)
        self.assertTrue(exporter.command_inventory(['cntlm', '-p', '-c/etc/not-config'])[
            'password_override_present'])
        for argv in (['cntlm', '-c', '/etc/cntlm.conf'], ['cntlm', '-c/etc/cntlm.conf']):
            with self.subTest(argv=argv):
                self.denied(lambda: exporter.command_inventory(argv))


if __name__ == '__main__':
    unittest.main()
