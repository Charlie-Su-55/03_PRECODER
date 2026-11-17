"""Release transaction tests use a fake GitHub API; no assets leave the machine."""
from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import model_release_03 as archive
import publish_03_model_release as pub
from test_model_release_03 import CE, SOURCE, digest, init_repo, terminal, write_json

ORIGIN = 'https://github.com/Charlie-Su-55/03_PRECODER.git'
API = 'https://api.github.com/repos/Charlie-Su-55/03_PRECODER'
UPLOAD = 'https://uploads.github.com/repos/Charlie-Su-55/03_PRECODER/releases/71/assets{?name,label}'


class FakeAPI:
    def __init__(self):
        self.release = None
        self.assets = []
        self.calls = []
        self.bad_digest = False
        self.no_digest = False
        self.fail_upload = False
        self.draft_tag_404 = False

    def request(self, method, url, payload=None, file_path=None):
        self.calls.append((method, url, payload, file_path))
        path = urlparse(url).path
        if method == 'GET' and '/releases/tags/' in path:
            if self.release is None or self.draft_tag_404 and self.release['draft']:
                raise pub.PublishError(status=404, github_request_id_present=True)
            return dict(self.release)
        if method == 'GET' and path.endswith('/releases'):
            return [dict(self.release)] if self.release else []
        if method == 'POST' and path.endswith('/releases'):
            self.release = dict(payload, id=71, upload_url=UPLOAD,
                                html_url='https://github.com/Charlie-Su-55/03_PRECODER/releases/tag/' + payload['tag_name'])
            return dict(self.release)
        if method == 'GET' and path.endswith('/releases/71/assets'):
            return list(self.assets)
        if method == 'POST' and path.endswith('/releases/71/assets'):
            if self.fail_upload:
                raise pub.PublishError(status=403)
            self.assert_file(file_path)
            name = parse_qs(urlparse(url).query)['name'][0]
            raw = Path(file_path).read_bytes()
            asset = dict(id=len(self.assets)+1, name=name, size=len(raw), state='uploaded',
                         digest='sha256:' + ('0' * 64 if self.bad_digest else digest(raw)))
            if self.no_digest:
                asset.pop('digest')
            self.assets.append(asset)
            return dict(asset)
        if method == 'DELETE' and '/releases/assets/' in path:
            asset_id = int(path.rsplit('/', 1)[1])
            self.assets = [asset for asset in self.assets if asset['id'] != asset_id]
            return None
        if method == 'PATCH' and path.endswith('/releases/71'):
            self.release.update(payload)
            return dict(self.release)
        raise AssertionError(f'Unexpected fake request: {method} {url}')

    @staticmethod
    def assert_file(file_path):
        if not isinstance(file_path, (str, Path)) or not Path(file_path).is_file():
            raise AssertionError('Assets must be streamed from file paths')


class ModelReleaseTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(redirect_stderr(io.StringIO()))
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name); self.root = self.base / 'repo'
        init_repo(self.root)
        self.git('remote', 'add', 'origin', ORIGIN)
        source = terminal(self.root, SOURCE, 'ce', 5000)
        reference = dict(path=str(source / 'checkpoint_latest.pt'),
                         sha256=digest((source / 'checkpoint_latest.pt').read_bytes()))
        write_json(self.root / 'runs' / CE / 'summary.json', {'status': 'completed', 'namespace': CE})
        for name in ('nmse', 'geometry'):
            terminal(self.root, CE, name, 60000, source_files=[reference])
        self.bundle = self.base / 'bundle'
        archive.prepare(self.root, 'new', self.bundle, part_bytes=16384)
        self.before = {p.name:p.read_bytes() for p in self.bundle.iterdir()}

    def git(self, *args):
        return subprocess.run(['git', *args], cwd=self.root, check=True,
                              capture_output=True, text=True).stdout.strip()

    def test_draft_upload_verified_assets_publish_and_idempotent_retry(self):
        api = FakeAPI()
        with redirect_stdout(io.StringIO()):
            first = pub.publish(self.bundle, root=self.root, api=api)
        self.assertFalse(api.release['draft'])
        uploads = [c for c in api.calls if c[0] == 'POST' and c[3] is not None]
        self.assertEqual({Path(c[3]).name for c in uploads}, set(self.before))
        patch_positions = [i for i,c in enumerate(api.calls) if c[0] == 'PATCH']
        upload_positions = [i for i,c in enumerate(api.calls) if c[3] is not None]
        self.assertGreater(min(patch_positions), max(upload_positions))
        self.assertTrue(any(c[2] and c[2].get('draft') is True for c in api.calls))
        before_uploads = len(uploads)
        with redirect_stdout(io.StringIO()):
            pub.publish(self.bundle, root=self.root, api=api)
        self.assertEqual(sum(c[0] == 'POST' and c[3] is not None for c in api.calls), before_uploads)
        self.assertEqual({p.name:p.read_bytes() for p in self.bundle.iterdir()}, self.before)

    def test_upload_failure_preserves_draft_and_local_checkpoint_bytes(self):
        api = FakeAPI(); api.fail_upload = True
        with self.assertRaises((pub.PublishError, ValueError)):
            pub.publish(self.bundle, root=self.root, api=api)
        self.assertTrue(api.release['draft'])
        self.assertFalse(any(c[0] == 'PATCH' for c in api.calls))
        self.assertEqual({p.name:p.read_bytes() for p in self.bundle.iterdir()}, self.before)

    def test_wrong_or_missing_server_digest_never_publishes(self):
        for mode in ('bad_digest', 'no_digest'):
            with self.subTest(mode=mode):
                api = FakeAPI(); setattr(api, mode, True)
                with self.assertRaises((pub.PublishError, ValueError)):
                    pub.publish(self.bundle, root=self.root, api=api)
                self.assertTrue(api.release['draft'])
                self.assertFalse(any(c[0] == 'PATCH' for c in api.calls))

    def test_conflicting_asset_is_never_overwritten(self):
        api = FakeAPI()
        with redirect_stdout(io.StringIO()):
            pub.publish(self.bundle, root=self.root, api=api)
        api.assets[0]['digest'] = 'sha256:' + '0' * 64
        calls = len(api.calls)
        with self.assertRaises((pub.PublishError, ValueError)):
            pub.publish(self.bundle, root=self.root, api=api)
        self.assertFalse(any(c[0] in ('POST', 'DELETE', 'PATCH') for c in api.calls[calls:]))

    def test_draft_resume_uses_authenticated_release_list_after_tag_404(self):
        api = FakeAPI(); api.fail_upload = True
        with self.assertRaises(pub.PublishError):
            pub.publish(self.bundle, root=self.root, api=api)
        api.fail_upload = False; api.draft_tag_404 = True
        pub.publish(self.bundle, root=self.root, api=api)
        self.assertFalse(api.release['draft'])
        self.assertEqual(sum(c[0] == 'POST' and urlparse(c[1]).path.endswith('/releases')
                             for c in api.calls), 1)

    def test_empty_failed_upload_placeholder_is_recovered_only_in_owned_draft(self):
        api = FakeAPI(); api.fail_upload = True
        with self.assertRaises(pub.PublishError):
            pub.publish(self.bundle, root=self.root, api=api)
        manifest = json.loads((self.bundle / 'manifest.json').read_text())
        api.assets = [dict(id=129, name=manifest['parts'][0]['name'], size=0,
                           digest=None, state='starter')]
        api.fail_upload = False
        pub.publish(self.bundle, root=self.root, api=api)
        deletes = [call for call in api.calls if call[0] == 'DELETE']
        self.assertEqual(len(deletes), 1)
        self.assertEqual(deletes[0][1], API + '/releases/assets/129')
        self.assertFalse(api.release['draft'])
        # The same-looking entry in a published release is not eligible for deletion.
        api.assets[0].update(size=0, digest=None, state='starter')
        calls = len(api.calls)
        with self.assertRaises(pub.PublishError):
            pub.publish(self.bundle, root=self.root, api=api)
        self.assertFalse(any(call[0] == 'DELETE' for call in api.calls[calls:]))

    def client(self, opener_factory):
        with mock.patch.object(pub, '_token', return_value='sensitive-test-token'):
            return pub.GitHubAPI(self.root, opener_factory=opener_factory)

    def test_real_request_streams_file_with_content_length_and_closes_it(self):
        asset = self.base / 'small-model.part'; asset.write_bytes(bytes(range(256)) * 40)
        captured, handlers = [], []
        case = self
        class Opener:
            def open(self, request, timeout):
                case.assertEqual(timeout, 120)
                case.assertNotIsInstance(request.data, (bytes, bytearray))
                case.assertTrue(hasattr(request.data, 'read'))
                case.assertEqual(request.get_header('Content-length'), str(asset.stat().st_size))
                case.assertEqual(request.get_header('Authorization'), 'Bearer sensitive-test-token')
                case.assertEqual(request.get_header('X-github-api-version'), '2022-11-28')
                hasher = hashlib.sha256()
                while True:
                    data = request.data.read(37)
                    if not data:
                        break
                    hasher.update(data)
                case.assertEqual(hasher.hexdigest(), digest(asset.read_bytes()))
                captured.append(request.data)
                return io.BytesIO(b'{"verified":true}')
        def factory(*selected):
            handlers.extend(selected)
            return Opener()
        client = self.client(factory)
        result = client.request('POST', API.replace('api.github.com','uploads.github.com') +
                                '/releases/71/assets?name=small-model.part', file_path=asset)
        self.assertEqual(result, {'verified': True})
        self.assertTrue(captured[0].closed)
        self.assertTrue(any(isinstance(h, pub._NoRedirect) for h in handlers))

    def test_authenticated_redirects_and_other_hosts_are_refused(self):
        with self.assertRaises(pub.PublishError):
            pub._NoRedirect().redirect_request(None, None, 302, 'Found', {}, 'https://evil.invalid/')
        for url in ('http://api.github.com/repos/Charlie-Su-55/03_PRECODER/releases',
                    'https://uploads.github.com.evil.invalid/repos/Charlie-Su-55/03_PRECODER/releases',
                    'https://token@api.github.com/repos/Charlie-Su-55/03_PRECODER/releases',
                    'https://api.github.com/repos/another/repo/releases'):
            with self.subTest(url=url), self.assertRaises(pub.PublishError):
                pub._checked_url(url)

    def test_remote_html_credentials_are_not_echoed_in_diagnostics(self):
        secret = 'never-echo-this-credential'
        class Opener:
            def open(self, request, timeout):
                raise urllib.error.HTTPError(request.full_url, 403, secret,
                    {'Content-Type': 'text/html'}, io.BytesIO(secret.encode()))
        client = self.client(lambda *handlers: Opener())
        with self.assertRaises(pub.PublishError) as context:
            client.request('GET', API + '/releases')
        self.assertNotIn(secret, str(context.exception))
        self.assertNotIn('sensitive-test-token', str(context.exception))
        self.assertEqual(context.exception.status, 403)
        self.assertEqual(context.exception.response_kind, 'html')
        self.assertFalse(context.exception.github_request_id_present)

    def test_ssl_key_logging_is_not_activated_by_environment(self):
        target = self.base / 'must-not-create-tls-secrets.log'
        with mock.patch.dict(os.environ, {'SSLKEYLOGFILE': str(target)}):
            self.client(lambda *handlers: object())
        self.assertFalse(target.exists())

    def test_verify_only_needs_no_token_or_network(self):
        with mock.patch.object(pub, 'GitHubAPI', side_effect=AssertionError('no network allowed')):
            self.assertEqual(pub.main(['--verify-only', '--bundle-dir', str(self.bundle)]), 0)

    def test_explicit_tls_bypass_is_refused(self):
        with mock.patch.dict(os.environ, {'GIT_SSL_NO_VERIFY': 'true'}):
            with self.assertRaises((pub.PublishError, ValueError)):
                pub._network_config(self.root)
        self.git('config', '--local', 'http.sslVerify', 'false')
        with self.assertRaises((pub.PublishError, ValueError)):
            pub._network_config(self.root)

    def test_repository_proxy_ca_and_environment_ca_precedence(self):
        cert = self.base / 'ca.pem'; cert.write_bytes(b'not opened by config selection')
        environment_cert = self.base / 'environment.pem'; environment_cert.write_bytes(b'not opened')
        self.git('config', '--local', 'http.https://github.com/.proxy', 'http://127.0.0.1:3132')
        self.git('config', '--local', 'http.https://github.com/.sslCAInfo', str(cert))
        # Keep PATH: real Git must remain callable inside production subprocesses.
        with mock.patch.dict(os.environ, {'GIT_SSL_CAINFO': str(environment_cert)}):
            network = pub._network_config(self.root)
        self.assertEqual(network['cafile'], str(environment_cert))
        self.assertEqual(network['proxy'], 'http://127.0.0.1:3132')
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('GIT_SSL_CAINFO', None)
            self.assertEqual(pub._network_config(self.root)['cafile'], str(cert))

    def test_wrong_origin_refuses_network(self):
        self.git('remote', 'set-url', 'origin', 'https://example.invalid/other.git')
        with self.assertRaises((pub.PublishError, ValueError)):
            pub._network_config(self.root)


if __name__ == '__main__':
    unittest.main()
