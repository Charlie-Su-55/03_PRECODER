"""REST transport checks with real local Git objects and fake HTTP responses.

These tests validate publication mechanics, not scientific model performance.
They never contact GitHub or publish to a network remote.
"""
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import ssl
import subprocess
import unittest
from unittest import mock

import publish_twc_v1_k8_ul_propagation as publication
import publish_twc_v1_k8_ul_propagation_rest as rest


# Import the fixture module rather than binding its TestCase in this module;
# unittest must not discover and run the original tests a second time.
_fixture_spec = importlib.util.spec_from_file_location(
    '_propagation_publication_fixture',
    Path(__file__).with_name('test_twc_v1_k8_ul_propagation_publish.py'))
_fixtures = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixtures)


class LocalGitDatabase:
    """A fake REST service whose object IDs come from actual bare Git plumbing."""

    def __init__(self, bare):
        self.bare = Path(bare)
        self.calls = []
        self.corrupt_at = None
        self.fail_once = None
        self.race_sha = None
        self.race_status = 422
        self.created_refs = []
        self.normalize_commit_lf = False

    def git(self, *args, data=None, check=True):
        return subprocess.run(['git', '--git-dir='+str(self.bare), *args],
            input=data, capture_output=True, check=check)

    def object(self, kind, content):
        return self.git('hash-object', '-w', '-t', kind, '--stdin', data=content).stdout.decode().strip()

    def read(self, kind, sha):
        return self.git('cat-file', kind, sha).stdout

    def ref(self, name):
        result = self.git('show-ref', '--verify', '--hash', name, check=False)
        return result.stdout.decode().strip() if result.returncode == 0 else None

    def other_commit(self):
        tree = self.git('mktree', data=b'').stdout.decode().strip()
        raw = ('tree '+tree+'\nauthor Other <other@example.invalid> 1 +0000\n'
               'committer Other <other@example.invalid> 1 +0000\n\nother result\n').encode()
        return self.object('commit', raw)

    @staticmethod
    def identity(value):
        date = datetime.fromisoformat(value['date'].replace('Z', '+00:00'))
        return (value['name']+' <'+value['email']+'> '+str(int(date.timestamp()))
                +' '+date.strftime('%z'))

    def _sha_response(self, kind, sha):
        return {'sha': 'f'*40 if self.corrupt_at == kind else sha}

    def request(self, method, path, payload=None):
        self.calls.append((method, path,
            json.loads(json.dumps(payload)) if payload is not None else None))
        kind = path.rsplit('/', 1)[-1]
        if self.fail_once == (method, kind):
            self.fail_once = None
            raise rest.RESTError(403, 'forbidden')
        if method == 'GET' and '/git/ref/heads/' in path:
            reference = 'refs/heads/'+path.split('/git/ref/heads/', 1)[1]
            sha = self.ref(reference)
            if sha is None:
                raise rest.RESTError(404, 'not_found')
            return {'ref': reference, 'object': {'type': 'commit', 'sha': sha}}
        if method == 'POST' and kind == 'blobs':
            assert payload['encoding'] == 'base64'
            sha = self.object('blob', base64.b64decode(payload['content'], validate=True))
            return self._sha_response('blob', sha)
        if method == 'POST' and kind == 'trees':
            assert 'base_tree' not in payload
            lines = []
            for entry in payload['tree']:
                assert entry['mode'] == '100644' and entry['type'] == 'blob'
                lines.append('100644 blob '+entry['sha']+'\t'+entry['path']+'\n')
            sha = self.git('mktree', data=''.join(lines).encode()).stdout.decode().strip()
            return self._sha_response('tree', sha)
        if method == 'POST' and kind == 'commits':
            assert payload.get('parents') == []
            message = payload['message']+('\n' if self.normalize_commit_lf else '')
            raw = ('tree '+payload['tree']+'\nauthor '+self.identity(payload['author'])
                +'\ncommitter '+self.identity(payload['committer'])+'\n\n'
                +message).encode('utf-8')
            sha = self.object('commit', raw)
            return self._sha_response('commit', sha)
        if method == 'POST' and kind == 'refs':
            reference, sha = payload['ref'], payload['sha']
            if self.race_sha is not None:
                self.git('update-ref', reference, self.race_sha)
                self.race_sha = None
                raise rest.RESTError(self.race_status, 'conflict')
            result = self.git('update-ref', reference, sha, '0'*40, check=False)
            if result.returncode:
                raise rest.RESTError(422, 'already_exists')
            self.created_refs.append(reference)
            return {'ref': reference, 'object': {'type': 'commit', 'sha': sha}}
        raise AssertionError('Unexpected fake API endpoint')


class RESTPublicationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = _fixtures.ResultPublicationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.git('remote', 'set-url', 'origin',
            'https://github.com/Charlie-Su-55/03_PRECODER.git')
        patch = mock.patch.object(publication, 'REPOSITORY',
            'https://github.com/Charlie-Su-55/03_PRECODER.git')
        patch.start()
        self.addCleanup(patch.stop)
        self.fixture.export()
        self.client = LocalGitDatabase(self.fixture.remote)
        self.before = {p.name: p.read_bytes() for p in self.fixture.report.iterdir()}

    def publish(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return rest.publish(self.fixture.report, client=self.client)

    def receipt(self):
        return self.fixture.runtime.read_json(self.fixture.output/'uploads'/'0123456789ab.json')

    def assert_preserved(self):
        self.fixture.assert_source_preserved()
        self.assertEqual(self.before,
            {p.name: p.read_bytes() for p in self.fixture.report.iterdir()})

    def test_actual_git_objects_exact_orphan_commit_and_dirty_source_preserved(self):
        receipt = self.publish()
        self.assertEqual(receipt['status'], 'uploaded')
        self.assertEqual(receipt['transport'], 'github_rest')
        self.assertEqual(receipt['rest_tool_sha256'],
            hashlib.sha256(Path(rest.__file__).read_bytes()).hexdigest())
        local = self.fixture.git('cat-file', 'commit', receipt['commit'])
        remote = self.client.read('commit', receipt['commit'])
        # cat-file through the text helper strips its final newline; compare raw bytes too.
        local_raw = subprocess.run(['git', 'cat-file', 'commit', receipt['commit']],
            cwd=self.fixture.root, capture_output=True, check=True).stdout
        self.assertEqual(remote, local_raw)
        self.assertTrue(local)
        self.assertTrue(remote.endswith(b'\n'))
        self.assertNotIn(b'\nparent ', remote)
        self.assertEqual(self.client.ref('refs/heads/'+receipt['branch']), receipt['commit'])
        for name, content in self.before.items():
            actual = self.client.git('show', receipt['commit']+':'+name).stdout
            self.assertEqual(actual, content)
        self.assert_preserved()

    def test_blob_tree_or_commit_sha_mismatch_never_creates_ref(self):
        for kind in ('blob', 'tree', 'commit'):
            with self.subTest(kind=kind):
                self.client.corrupt_at = kind
                self.client.calls.clear()
                with self.assertRaises((RuntimeError, ValueError)):
                    self.publish()
                self.assertFalse(any(method == 'POST' and path.endswith('/git/refs')
                    for method, path, _ in self.client.calls))
                if kind == 'commit':
                    self.assertEqual(sum(method == 'POST' and path.endswith('/git/commits')
                        for method, path, _ in self.client.calls), 2)
                self.assertEqual(self.client.created_refs, [])
                self.assert_preserved()

    def test_api_final_newline_normalization_only_accepts_exact_prepared_commit(self):
        self.client.normalize_commit_lf = True
        receipt = self.publish()
        attempts = [payload for method, path, payload in self.client.calls
            if method == 'POST' and path.endswith('/git/commits')]
        self.assertEqual(len(attempts), 2)
        local_raw = subprocess.run(['git', 'cat-file', 'commit', receipt['commit']],
            cwd=self.fixture.root, capture_output=True, check=True).stdout
        original_message = local_raw.split(b'\n\n', 1)[1].decode()
        self.assertTrue(original_message.endswith('\n'))
        self.assertEqual(attempts[0]['message'], original_message)
        self.assertEqual(attempts[1]['message'], original_message[:-1])
        self.assertEqual({key: value for key, value in attempts[0].items() if key != 'message'},
                         {key: value for key, value in attempts[1].items() if key != 'message'})
        self.assertEqual(self.client.read('commit', receipt['commit']), local_raw)
        self.assertEqual(self.client.created_refs, ['refs/heads/'+receipt['branch']])
        self.assertEqual(self.client.ref('refs/heads/'+receipt['branch']), receipt['commit'])
        self.assert_preserved()

    def test_existing_same_ref_is_idempotent_without_new_object_writes(self):
        receipt = self.publish()
        self.client.calls.clear()
        again = self.publish()
        self.assertEqual(receipt['commit'], again['commit'])
        self.assertTrue(self.client.calls)
        self.assertTrue(all(method == 'GET' for method, _, _ in self.client.calls))
        self.assertEqual(len(self.client.created_refs), 1)
        self.assert_preserved()

    def test_existing_different_ref_is_refused_without_object_writes(self):
        branch = publication._branch('0123456789ab')
        other = self.client.other_commit()
        self.client.git('update-ref', 'refs/heads/'+branch, other)
        with self.assertRaises((RuntimeError, ValueError)):
            self.publish()
        self.assertTrue(all(method == 'GET' for method, _, _ in self.client.calls))
        self.assertEqual(self.client.ref('refs/heads/'+branch), other)
        self.assert_preserved()

    def test_create_ref_race_requires_matching_commit_for_409_and_422(self):
        # First prepare the exact immutable local commit without sending an object.
        self.client.fail_once = ('POST', 'blobs')
        with self.assertRaises((RuntimeError, ValueError)):
            self.publish()
        expected = self.receipt()['commit']
        local_raw = subprocess.run(['git', 'cat-file', 'commit', expected],
            cwd=self.fixture.root, capture_output=True, check=True).stdout
        # The commit is materialized in the fake Git database by normal upload.
        for status in (409, 422):
            for matching in (True, False):
                with self.subTest(status=status, matching=matching):
                    reference = 'refs/heads/'+publication._branch('0123456789ab')
                    self.client.git('update-ref', '-d', reference)
                    self.client.race_status = status
                    self.client.race_sha = expected if matching else self.client.other_commit()
                    if matching:
                        result = self.publish()
                        self.assertEqual(result['commit'], expected)
                        self.assertEqual(self.client.read('commit', expected), local_raw)
                    else:
                        with self.assertRaises((RuntimeError, ValueError)):
                            self.publish()
                        self.assertNotEqual(self.client.ref(reference), expected)
                    self.assert_preserved()

    def test_api_403_keeps_failed_receipt_and_export_then_reuses_commit_on_retry(self):
        self.client.fail_once = ('POST', 'blobs')
        with self.assertRaises((RuntimeError, ValueError)):
            self.publish()
        failed = self.receipt()
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(failed['transport'], 'github_rest')
        self.assertEqual(self.client.created_refs, [])
        self.assert_preserved()
        retry = self.publish()
        self.assertEqual(retry['commit'], failed['commit'])
        self.assertEqual(retry['status'], 'uploaded')
        self.assert_preserved()

    def test_failed_git_transport_receipt_reuses_exact_commit_for_rest(self):
        with mock.patch.object(publication, '_push_commit', return_value=1):
            with self.assertRaisesRegex(RuntimeError, 'local result export is preserved'):
                publication.publish(self.fixture.report)
        prior = self.receipt()
        self.assertEqual(prior['status'], 'failed')
        self.assertNotIn('rest_tool_sha256', prior)
        receipt = self.publish()
        self.assertEqual(receipt['commit'], prior['commit'])
        self.assertEqual(receipt['tree'], prior['tree'])
        self.assertEqual(receipt['export_manifest_sha256'], prior['export_manifest_sha256'])
        self.assertEqual(receipt['transport'], 'github_rest')
        self.assertEqual(receipt['rest_tool_sha256'],
            hashlib.sha256(Path(rest.__file__).read_bytes()).hexdigest())
        self.assert_preserved()

    def test_verify_only_never_requests_credentials_or_constructs_network_client(self):
        objects = self.fixture.root/'.git'/'objects'
        before_objects = {p.relative_to(objects) for p in objects.rglob('*') if p.is_file()}
        with mock.patch.object(rest, '_cached_credentials',
                side_effect=AssertionError('verification requested a credential')) as credential:
            with mock.patch.object(rest, 'GitHubAPI',
                    side_effect=AssertionError('verification constructed a network client')) as client:
                with redirect_stdout(io.StringIO()):
                    rest.verify_only(self.fixture.report)
        credential.assert_not_called()
        client.assert_not_called()
        self.assertEqual(self.client.calls, [])
        self.assertEqual(before_objects,
            {p.relative_to(objects) for p in objects.rglob('*') if p.is_file()})
        self.assert_preserved()

    def test_unexpected_origin_fails_before_any_api_requests(self):
        self.fixture.git('remote', 'set-url', 'origin', 'https://example.invalid/other.git')
        with self.assertRaises((RuntimeError, ValueError)):
            self.publish()
        self.assertEqual(self.client.calls, [])
        self.assert_preserved()

    def test_raw_commit_timezone_message_and_final_newline_are_exact(self):
        tree = self.client.git('mktree', data=b'').stdout.decode().strip()
        # Use explicit raw identities so the test does not depend on machine TZ.
        raw = ('tree '+tree+'\nauthor User With Spaces <u@example.invalid> 1234567890 +0530\n'
            'committer Other Name <c@example.invalid> 1234567990 -0330\n\n'
            'first line\n\nsecond line\n').encode()
        parsed = rest.parse_raw_commit(raw)
        self.assertEqual(parsed['tree'], tree)
        self.assertEqual(LocalGitDatabase.identity(parsed['author']),
            'User With Spaces <u@example.invalid> 1234567890 +0530')
        self.assertEqual(LocalGitDatabase.identity(parsed['committer']),
            'Other Name <c@example.invalid> 1234567990 -0330')
        self.assertEqual(parsed['message'], 'first line\n\nsecond line\n')
        payload = dict(parsed, parents=[])
        sha = self.client.request('POST',
            '/repos/Charlie-Su-55/03_PRECODER/git/commits', payload)['sha']
        self.assertEqual(self.client.read('commit', sha), raw)
        self.assert_preserved()


class FakeResponse:
    def __init__(self, status=200, body=b'{"ok":true}', headers=None):
        self.status = status
        self.reason = 'private-server-reason-must-not-be-printed'
        self.body = body
        self.headers = {'Content-Type': 'application/json', **(headers or {})}

    def read(self, amount=None):
        return self.body if amount is None else self.body[:amount]

    def getheader(self, name, default=None):
        return next((value for key, value in self.headers.items()
                     if key.lower() == name.lower()), default)

    def getheaders(self):
        return list(self.headers.items())


class FakeConnection:
    def __init__(self, response):
        self.response = response
        self.requests = []
        self.tunnels = []
        self.closed = 0
        self.debuglevel = None

    def set_debuglevel(self, level):
        self.debuglevel = level

    def set_tunnel(self, host, port=None, headers=None):
        self.tunnels.append((host, port, headers))

    def request(self, method, url, body=None, headers=None, **kwargs):
        self.requests.append((method, url, body, headers or {}))

    def getresponse(self):
        return self.response

    def close(self):
        self.closed += 1


class HTTPClientSecurityTests(unittest.TestCase):
    """Local fake transport responses; no external HTTP or credential is used."""

    def setUp(self):
        self.fixture = _fixtures.ResultPublicationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.git('remote', 'set-url', 'origin',
            'https://github.com/Charlie-Su-55/03_PRECODER.git')
        self.secret = 'FAKE_TEST_TOKEN_NEVER_FROM_A_USER'
        self.factories = []
        self.ref_path = ('/repos/Charlie-Su-55/03_PRECODER/git/ref/heads/results/'
                         'fixture_propagation/0123456789ab')

    def api(self, connection, network=None):
        def factory(*args, **kwargs):
            self.factories.append((args, kwargs))
            return connection

        with mock.patch.object(rest, '_cached_credentials',
                return_value={'username': 'fixture', 'password': self.secret}):
            with mock.patch.object(rest, '_network_config', return_value=network or
                    {'cafile': None, 'capath': None, 'proxy': None}):
                return rest.GitHubAPI(self.fixture.root, connection_factory=factory)

    def test_authenticated_json_request_has_pinned_api_version_and_fixed_origin(self):
        connection = FakeConnection(FakeResponse())
        api = self.api(connection)
        result = api.request('POST', '/repos/Charlie-Su-55/03_PRECODER/git/blobs',
            {'content': 'c2NhbGFy', 'encoding': 'base64'})
        self.assertEqual(result, {'ok': True})
        self.assertEqual(len(connection.requests), 1)
        method, target, body, headers = connection.requests[0]
        headers = {key.lower(): value for key, value in headers.items()}
        self.assertEqual(method, 'POST')
        self.assertEqual(target, '/repos/Charlie-Su-55/03_PRECODER/git/blobs')
        self.assertEqual(headers['x-github-api-version'], '2022-11-28')
        self.assertEqual(headers['accept'], 'application/vnd.github+json')
        self.assertEqual(headers['authorization'], 'Bearer '+self.secret)
        self.assertNotIn(self.secret, body.decode() if isinstance(body, bytes) else body)
        args, kwargs = self.factories[0]
        self.assertEqual(args[0], 'api.github.com')
        self.assertIsInstance(kwargs['context'], ssl.SSLContext)
        self.assertEqual(kwargs['context'].verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(kwargs['context'].check_hostname)
        self.assertEqual(connection.debuglevel, 0)

    def test_http_proxy_connect_targets_only_api_github_with_tls_verification(self):
        connection = FakeConnection(FakeResponse())
        api = self.api(connection, network={
            'cafile': None, 'capath': None, 'proxy': 'http://proxy.example.invalid:3129'})
        api.request('GET', self.ref_path)
        args, kwargs = self.factories[0]
        self.assertEqual(args[0], 'proxy.example.invalid')
        self.assertEqual(connection.tunnels[0][:2], ('api.github.com', 443))
        self.assertEqual(kwargs['context'].verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(kwargs['context'].check_hostname)

    def test_redirect_is_not_followed_and_error_does_not_expose_private_headers_or_body(self):
        marker = self.secret+' https://private.invalid/credential-path'
        for status in (302, 403):
            with self.subTest(status=status):
                self.factories.clear()
                response = FakeResponse(status, json.dumps({'message': marker}).encode(), {
                    'Location': 'https://private.invalid/'+self.secret,
                    'Server': marker, 'X-GitHub-Request-Id': marker})
                connection = FakeConnection(response)
                api = self.api(connection)
                output, errors = io.StringIO(), io.StringIO()
                with redirect_stdout(output), redirect_stderr(errors):
                    with self.assertRaises(rest.RESTError) as caught:
                        api.request('GET', self.ref_path)
                self.assertEqual(caught.exception.status, status)
                safe = str(caught.exception)+'\n'+output.getvalue()+errors.getvalue()
                for forbidden in (self.secret, 'private.invalid', response.reason):
                    self.assertNotIn(forbidden, safe)
                self.assertEqual(len(connection.requests), 1)
                self.assertEqual(len(self.factories), 1)
                self.assertGreaterEqual(connection.closed, 1)

    def test_malformed_error_message_remains_a_sanitized_http_error(self):
        for message in (None, 17, {'private': self.secret}, [self.secret]):
            with self.subTest(message_type=type(message).__name__):
                connection = FakeConnection(FakeResponse(403,
                    json.dumps({'message': message}).encode()))
                api = self.api(connection)
                with self.assertRaises(rest.RESTError) as caught:
                    api.request('GET', self.ref_path)
                self.assertEqual(caught.exception.status, 403)
                self.assertNotIn(self.secret, str(caught.exception))
                self.assertEqual(len(connection.requests), 1)

    def test_proxy_authentication_is_confined_to_connect_tunnel(self):
        connection = FakeConnection(FakeResponse())
        api = self.api(connection, network={'cafile': None, 'capath': None,
            'proxy': 'http://fake-proxy-user:FAKE-PROXY-PASSWORD@proxy.example.invalid:3129'})
        api.request('GET', self.ref_path)
        tunnel_headers = connection.tunnels[0][2]
        self.assertEqual(tunnel_headers['Proxy-Authorization'], 'Basic '+base64.b64encode(
            b'fake-proxy-user:FAKE-PROXY-PASSWORD').decode())
        self.assertFalse(any(self.secret in value for value in tunnel_headers.values()))
        request_headers = {key.lower(): value for key, value in connection.requests[0][3].items()}
        self.assertNotIn('proxy-authorization', request_headers)
        self.assertFalse(any('FAKE-PROXY-PASSWORD' in value for value in request_headers.values()))

    def test_foreign_or_absolute_api_paths_are_rejected_before_request(self):
        connection = FakeConnection(FakeResponse())
        api = self.api(connection)
        for path in ('https://evil.invalid/repos/Charlie-Su-55/03_PRECODER/git/refs',
                     '//evil.invalid/path', '/repos/other/project/git/refs'):
            with self.subTest(path=path):
                with self.assertRaises((ValueError, rest.RESTError)):
                    api.request('GET', path)
        self.assertEqual(connection.requests, [])
        self.assertEqual(self.factories, [])

    def test_git_url_ca_and_proxy_are_selected_with_environment_ca_precedence(self):
        ca = ssl.get_default_verify_paths().cafile
        self.assertIsNotNone(ca)
        self.fixture.git('config', 'http.https://github.com/.proxy', 'http://127.0.0.1:3129')
        self.fixture.git('config', 'http.https://github.com/.sslCAInfo', ca)
        with mock.patch.dict(os.environ, {}, clear=True):
            selected = rest._network_config(self.fixture.root)
        self.assertEqual(selected['proxy'], 'http://127.0.0.1:3129')
        self.assertEqual(selected['cafile'], ca)
        override = self.fixture.root/'local-ca.pem'
        override.write_bytes(Path(ca).read_bytes())
        with mock.patch.dict(os.environ, {'GIT_SSL_CAINFO': str(override)}, clear=True):
            selected = rest._network_config(self.fixture.root)
        self.assertEqual(selected['cafile'], str(override))

    def test_explicit_tls_bypass_is_refused(self):
        with mock.patch.dict(os.environ, {'GIT_SSL_NO_VERIFY': '1'}, clear=True):
            with self.assertRaises((ValueError, rest.RESTError)):
                rest._network_config(self.fixture.root)
        self.fixture.git('config', 'http.https://github.com/.sslVerify', 'false')
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises((ValueError, rest.RESTError)):
                rest._network_config(self.fixture.root)

    def test_cached_credential_is_memory_only_silent_and_disables_prompt_and_trace(self):
        calls = []
        real_run = subprocess.run

        def run(command, *args, **kwargs):
            if 'credential' in command and 'fill' in command:
                calls.append((command, kwargs))
                return subprocess.CompletedProcess(command, 0,
                    ('protocol=https\nhost=github.com\nusername=fixture\npassword='
                     +self.secret+'\n\n').encode(), b'')
            return real_run(command, *args, **kwargs)

        output, errors = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {'GIT_TRACE_CURL': '1', 'GIT_TRACE2_EVENT': '1',
                'GIT_CURL_VERBOSE': '1', 'GIT_ASKPASS': 'must-not-run'}, clear=False):
            with mock.patch.object(rest.subprocess, 'run', side_effect=run):
                with redirect_stdout(output), redirect_stderr(errors):
                    credential = rest._cached_credentials(self.fixture.root)
        self.assertEqual(credential, {'username': 'fixture', 'password': self.secret})
        self.assertEqual(output.getvalue()+errors.getvalue(), '')
        self.assertEqual(len(calls), 1)
        _, kwargs = calls[0]
        self.assertTrue(kwargs['capture_output'])
        environment = kwargs['env']
        self.assertEqual(environment['GIT_TERMINAL_PROMPT'], '0')
        self.assertEqual(environment['GIT_ASKPASS'], '/bin/false')
        self.assertFalse(any(key.startswith('GIT_TRACE') for key in environment))
        self.assertNotIn('GIT_CURL_VERBOSE', environment)
        supplied = kwargs['input']
        supplied = supplied.decode() if isinstance(supplied, bytes) else supplied
        self.assertIn('protocol=https\n', supplied)
        self.assertIn('host=github.com\n', supplied)
        self.assertNotIn(self.secret, supplied)
        self.fixture.assert_source_preserved()

    def test_failed_helper_and_network_exceptions_do_not_expose_sensitive_text(self):
        marker = self.secret+' https://private.invalid/credential-path'
        failed = subprocess.CompletedProcess(['git', 'credential', 'fill'], 1,
            marker.encode(), marker.encode())
        output, errors = io.StringIO(), io.StringIO()
        with mock.patch.object(rest.subprocess, 'run', return_value=failed):
            with redirect_stdout(output), redirect_stderr(errors):
                with self.assertRaises(rest.RESTError) as caught:
                    rest._cached_credentials(self.fixture.root)
        self.assertNotIn(self.secret, str(caught.exception))
        self.assertNotIn('private.invalid', str(caught.exception))
        self.assertEqual(output.getvalue()+errors.getvalue(), '')
        connection = FakeConnection(FakeResponse())
        connection.request = mock.Mock(side_effect=OSError(marker))
        api = self.api(connection)
        with redirect_stdout(output), redirect_stderr(errors):
            with self.assertRaises(rest.RESTError) as caught:
                api.request('GET', self.ref_path)
        self.assertNotIn(self.secret, str(caught.exception))
        self.assertNotIn('private.invalid', str(caught.exception))
        self.assertEqual(output.getvalue()+errors.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
