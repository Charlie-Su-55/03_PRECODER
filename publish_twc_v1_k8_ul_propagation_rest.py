#!/usr/bin/env python3
"""Create the exact prepared scalar-result Git objects over GitHub REST.

This is an additive transport adapter. The completed 46-file diagnostic core,
its report bytes and original prepared orphan commit remain unchanged. Tokens
come only from the existing Git credential helper, stay in memory, and are
never accepted as command-line arguments or written to receipts.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import socket
import ssl
import subprocess
from urllib.parse import unquote, urlsplit

import publish_twc_v1_k8_ul_propagation as previous

API_HOST = 'api.github.com'
API_PREFIX = '/repos/Charlie-Su-55/03_PRECODER'
OFFICIAL_ORIGIN = 'https://github.com/Charlie-Su-55/03_PRECODER.git'
MAX_RESPONSE_BYTES = 2*1024**2


class RESTError(RuntimeError):
    """A deliberately sanitized failure: never retain arbitrary body/headers."""
    def __init__(self, status=None, category='rest_failure', *, response_kind=None,
                 github_request_id_present=False, known_scope_denied=False,
                 retry_after_seconds=None):
        self.status = status
        self.category = category
        self.response_kind = response_kind
        self.github_request_id_present = bool(github_request_id_present)
        self.known_scope_denied = bool(known_scope_denied)
        self.retry_after_seconds = retry_after_seconds
        super().__init__('GitHub REST '+json.dumps(self.safe_details(), sort_keys=True))

    def safe_details(self):
        return dict(status=self.status, category=self.category,
            response_kind=self.response_kind,
            github_request_id_present=self.github_request_id_present,
            known_scope_denied=self.known_scope_denied,
            retry_after_seconds=self.retry_after_seconds)


def _git_config(root, key, *, urlmatch=False):
    args = ['git', 'config']
    args += ['--get-urlmatch', key, OFFICIAL_ORIGIN] if urlmatch else ['--get', key]
    try:
        result = subprocess.run(args, cwd=root, capture_output=True, timeout=15,
            check=False, env=previous._git_environment())
    except (OSError, subprocess.TimeoutExpired):
        raise RESTError(category='local_git_configuration_unavailable') from None
    if result.returncode == 1:
        return None
    if result.returncode:
        raise RESTError(category='local_git_configuration_unavailable')
    return result.stdout.decode('utf-8').strip()


def _network_config(root):
    """Use the Git origin's effective CA/proxy, without exposing their values."""
    if previous._git(root, ['remote', 'get-url', 'origin']) != OFFICIAL_ORIGIN:
        raise RESTError(category='official_origin_required')
    disabled = os.environ.get('GIT_SSL_NO_VERIFY', '').strip().lower()
    configured = _git_config(root, 'http.sslVerify', urlmatch=True)
    if (disabled not in ('', '0', 'false', 'no', 'off')
            or configured is not None and configured.lower() in ('0', 'false', 'no', 'off')):
        raise RESTError(category='tls_verification_disabled_in_git_configuration')
    cafile = os.environ.get('GIT_SSL_CAINFO') or _git_config(root, 'http.sslCAInfo', urlmatch=True)
    capath = os.environ.get('GIT_SSL_CAPATH') or _git_config(root, 'http.sslCAPath', urlmatch=True)
    proxy = _git_config(root, 'remote.origin.proxy')
    if proxy is None:
        proxy = _git_config(root, 'http.proxy', urlmatch=True)
    if proxy is None:
        proxy = (os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
                 or os.environ.get('all_proxy') or os.environ.get('ALL_PROXY'))
    return dict(cafile=str(Path(cafile).expanduser()) if cafile else None,
        capath=str(Path(capath).expanduser()) if capath else None, proxy=proxy or None)


def _cached_credentials(root):
    """Read only an existing cached/helper credential; never ask the terminal."""
    environment = previous._git_environment()
    environment.update(GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never',
        GIT_ASKPASS='/bin/false', SSH_ASKPASS='/bin/false')
    request = b'protocol=https\nhost=github.com\npath=Charlie-Su-55/03_PRECODER.git\n\n'
    try:
        result = subprocess.run(['git', '-c', 'credential.interactive=false', 'credential', 'fill'],
            cwd=root, input=request, capture_output=True, env=environment,
            timeout=15, check=False)
    except subprocess.TimeoutExpired:
        raise RESTError(category='credential_helper_timeout') from None
    except OSError:
        raise RESTError(category='credential_helper_unavailable') from None
    if result.returncode:
        raise RESTError(category='cached_git_credentials_unavailable')
    try:
        fields = dict(line.split('=', 1) for line in result.stdout.decode('utf-8').splitlines() if '=' in line)
    except UnicodeError:
        raise RESTError(category='cached_git_credentials_invalid') from None
    password = fields.get('password', '')
    if not password or any(char in password for char in '\r\n\0'):
        raise RESTError(category='cached_git_credentials_unavailable')
    try:
        password.encode('ascii')
    except UnicodeError:
        raise RESTError(category='cached_git_credentials_invalid') from None
    return dict(username=fields.get('username', ''), password=password)


def _request_path(method, path):
    if not isinstance(path, str) or not path.startswith(API_PREFIX+'/git/'):
        raise RESTError(category='api_target_rejected')
    if any(value in path for value in ('?', '#', '\\', '..', '://')):
        raise RESTError(category='api_target_rejected')
    allowed_post = {API_PREFIX+'/git/'+name for name in ('blobs', 'trees', 'commits', 'refs')}
    prefix = API_PREFIX+'/git/ref/heads/results/'+previous._config().NAMESPACE+'/'
    if method == 'POST' and path in allowed_post:
        return path
    if method == 'GET' and path.startswith(prefix) and re.fullmatch(r'[0-9a-f]{12,32}', path[len(prefix):]):
        return path
    raise RESTError(category='api_method_or_path_rejected')


def _request_url(path, method='GET'):
    return 'https://'+API_HOST+_request_path(method, path)


def _http_failure(status, content_type, body, request_id, retry_after=None):
    lowered = body.lstrip().lower()
    kind = ('html' if 'html' in content_type.lower() or lowered.startswith((b'<!doctype html', b'<html'))
            else 'json' if 'json' in content_type.lower() or lowered.startswith((b'{', b'[')) else 'other')
    message = ''
    if kind == 'json':
        try:
            parsed = json.loads(body)
            raw_message = parsed.get('message', '') if isinstance(parsed, dict) else ''
            message = raw_message.lower() if isinstance(raw_message, str) else ''
        except (ValueError, UnicodeError):
            pass
    scope = any(marker in message for marker in
        ('resource not accessible by personal access token', 'resource not accessible by integration',
         'insufficient permission', 'insufficient scope'))
    if 300 <= status < 400:
        category = 'redirect_rejected'
    elif kind == 'html':
        category = 'gateway_or_non_github_html_response'
    elif status == 401:
        category = 'authentication_rejected'
    elif status == 429 or 'rate limit' in message:
        category = 'rate_limited'
    elif status == 403 and scope:
        category = 'known_scope_or_permission_denied'
    elif status == 403:
        category = 'forbidden_unclassified'
    elif status == 404:
        category = 'not_found'
    elif status in (409, 422):
        category = 'conflict_or_validation_rejected'
    else:
        category = 'http_failure'
    seconds = None
    try:
        parsed_retry = int(retry_after)
        if 0 <= parsed_retry <= 86400:
            seconds = parsed_retry
    except (ValueError, TypeError):
        pass
    return RESTError(status, category, response_kind=kind,
        github_request_id_present=bool(request_id), known_scope_denied=scope,
        retry_after_seconds=seconds)


class GitHubAPI:
    """Strict HTTPS only; explicit HTTP proxy tunnel, no redirect mechanism."""
    def __init__(self, root=None, *, timeout=30, connection_factory=None):
        root = previous._config().ROOT if root is None else root
        settings = _network_config(root)
        try:
            self._context = ssl.create_default_context(cafile=settings['cafile'], capath=settings['capath'])
        except (OSError, ssl.SSLError):
            raise RESTError(category='trusted_ca_configuration_unavailable') from None
        self._context.check_hostname = True
        self._context.verify_mode = ssl.CERT_REQUIRED
        # Python may inherit SSLKEYLOGFILE. Never record this credential-bearing
        # API session's TLS secrets, even when another process enables tracing.
        if hasattr(self._context, 'keylog_filename'):
            self._context.keylog_filename = None
        self._timeout = timeout
        self._factory = connection_factory or http.client.HTTPSConnection
        self._proxy = None
        self._proxy_headers = {}
        if settings['proxy']:
            try:
                parsed = urlsplit(settings['proxy'])
            except ValueError:
                raise RESTError(category='unsupported_proxy_configuration') from None
            if parsed.scheme != 'http' or not parsed.hostname or parsed.path not in ('', '/') or parsed.query or parsed.fragment:
                raise RESTError(category='unsupported_proxy_configuration')
            try:
                self._proxy = (parsed.hostname, parsed.port or 80)
            except ValueError:
                raise RESTError(category='unsupported_proxy_configuration') from None
            if parsed.username is not None:
                credential = unquote(parsed.username)+':'+unquote(parsed.password or '')
                self._proxy_headers['Proxy-Authorization'] = 'Basic '+base64.b64encode(credential.encode()).decode()
        cached = _cached_credentials(root)
        self._authorization = 'Bearer '+cached['password']

    def _connection(self):
        host, port = self._proxy if self._proxy else (API_HOST, 443)
        connection = self._factory(host, port, timeout=self._timeout, context=self._context)
        connection.set_debuglevel(0)
        if self._proxy:
            connection.set_tunnel(API_HOST, 443, headers=self._proxy_headers)
        return connection

    def request(self, method, path, payload=None):
        path = _request_path(method, path)
        if method == 'GET' and payload is not None:
            raise RESTError(category='unexpected_get_payload')
        body = None if payload is None else json.dumps(payload, ensure_ascii=True, allow_nan=False).encode('utf-8')
        headers = dict(Accept='application/vnd.github+json', Authorization=self._authorization,
            **{'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': '03-precoder-scalar-results-rest',
               'Content-Type': 'application/json'})
        connection = None
        try:
            connection = self._connection()
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            data = response.read(MAX_RESPONSE_BYTES+1)
            if len(data) > MAX_RESPONSE_BYTES:
                raise RESTError(response.status, 'response_size_limit_exceeded')
            content_type = response.getheader('Content-Type', '')
            request_id = response.getheader('X-GitHub-Request-Id', '')
            if not 200 <= response.status < 300:
                raise _http_failure(response.status, content_type, data, request_id,
                    response.getheader('Retry-After'))
            try:
                result = json.loads(data)
            except (ValueError, UnicodeError):
                raise RESTError(response.status, 'non_json_success_response',
                    response_kind='html' if b'<html' in data[:256].lower() else 'other',
                    github_request_id_present=bool(request_id)) from None
            if not isinstance(result, dict):
                raise RESTError(response.status, 'unexpected_success_payload')
            return result
        except RESTError:
            raise
        except ssl.SSLCertVerificationError:
            raise RESTError(category='tls_certificate_verification_failed') from None
        except ssl.SSLError:
            raise RESTError(category='tls_connection_failed') from None
        except (socket.timeout, TimeoutError):
            raise RESTError(category='network_timeout') from None
        except (OSError, http.client.HTTPException):
            raise RESTError(category='network_or_proxy_connection_failed') from None
        finally:
            if connection is not None:
                connection.close()


def _object_sha(kind, data):
    return hashlib.sha1(kind.encode()+b' '+str(len(data)).encode()+b'\0'+data,
        usedforsecurity=False).hexdigest()


def _tree_description(files):
    records, raw = [], bytearray()
    for name, path in sorted(files.items()):
        if Path(name).name != name or any(char in name for char in '\n\t\0'):
            raise RESTError(category='unsafe_public_filename')
        data = Path(path).read_bytes()
        sha = _object_sha('blob', data)
        records.append(dict(path=name, mode='100644', type='blob', sha=sha))
        raw.extend(b'100644 '+name.encode('utf-8')+b'\0'+bytes.fromhex(sha))
    return records, _object_sha('tree', bytes(raw))


def _identity(raw):
    match = re.fullmatch(rb'(.+) <([^<>\n]+)> (-?[0-9]+) ([+-][0-9]{4})', raw)
    if not match:
        raise RESTError(category='unsupported_prepared_commit_identity')
    name, email, timestamp, offset = match.groups()
    try:
        offset = offset.decode('ascii')
        hours, minutes = int(offset[1:3]), int(offset[3:5])
        if hours > 23 or minutes > 59:
            raise ValueError()
        shift = (1 if offset[0] == '+' else -1)*(hours*60+minutes)
        local = datetime.fromtimestamp(int(timestamp), timezone(timedelta(minutes=shift)))
        date = local.strftime('%Y-%m-%dT%H:%M:%S')+offset[:3]+':'+offset[3:]
        return dict(name=name.decode('utf-8'), email=email.decode('utf-8'), date=date)
    except (ValueError, OverflowError, UnicodeError):
        raise RESTError(category='unsupported_prepared_commit_identity') from None


def parse_raw_commit(raw):
    """Preserve both identity dates/offsets and the exact final message newline."""
    if not isinstance(raw, bytes) or b'\0' in raw or b'\n\n' not in raw:
        raise RESTError(category='invalid_prepared_commit')
    headers, message = raw.split(b'\n\n', 1)
    values = {}
    for line in headers.split(b'\n'):
        key, separator, value = line.partition(b' ')
        if not separator or key not in (b'tree', b'author', b'committer') or key in values:
            raise RESTError(category='unsupported_prepared_commit_headers')
        values[key] = value
    if set(values) != {b'tree', b'author', b'committer'} or not re.fullmatch(rb'[0-9a-f]{40}', values[b'tree']):
        raise RESTError(category='invalid_prepared_orphan_commit')
    try:
        decoded = message.decode('utf-8')
    except UnicodeError:
        raise RESTError(category='unsupported_prepared_commit_encoding') from None
    return dict(tree=values[b'tree'].decode(), parents=[], author=_identity(values[b'author']),
        committer=_identity(values[b'committer']), message=decoded)


def _raw_commit(root, commit):
    try:
        result = subprocess.run(['git', 'cat-file', 'commit', commit], cwd=root,
            capture_output=True, check=False, env=previous._git_environment(), timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        raise RESTError(category='prepared_commit_unavailable') from None
    if result.returncode or _object_sha('commit', result.stdout) != commit:
        raise RESTError(category='prepared_commit_unavailable')
    return result.stdout


def _local_export(path=None):
    cfg, runtime = previous._config(), previous._runtime()
    output = previous._completed_export(path)
    manifest = previous.verify_export_directory(output)
    if previous._git(cfg.ROOT, ['remote', 'get-url', 'origin']) != OFFICIAL_ORIGIN:
        raise RESTError(category='official_origin_required')
    branch = previous._branch(manifest['export_id'])
    receipt_path = runtime.inside(cfg.OUTPUT_ROOT/'uploads'/(manifest['export_id']+'.json'))
    files = {name: output/name for name in (*previous.PUBLIC_FILES, 'export_manifest.json')}
    entries, tree = _tree_description(files)
    manifest_sha = runtime.sha256(output/'export_manifest.json')
    receipt = runtime.read_json(receipt_path) if receipt_path.exists() else None
    if receipt:
        if (receipt.get('branch') != branch or receipt.get('export_id') != manifest['export_id']
                or receipt.get('export_manifest_sha256') != manifest_sha or receipt.get('tree') != tree
                or not re.fullmatch(r'[0-9a-f]{40}', receipt.get('commit', ''))):
            raise RESTError(category='prepared_publication_identity_changed')
        raw = _raw_commit(cfg.ROOT, receipt['commit'])
        parsed = parse_raw_commit(raw)
        if parsed['tree'] != tree:
            raise RESTError(category='prepared_publication_tree_changed')
    return dict(output=output, manifest=manifest, branch=branch, receipt_path=receipt_path,
        files=files, entries=entries, tree=tree, manifest_sha=manifest_sha, receipt=receipt)


def verify_only(path=None):
    """Fully local: no credentials, API client or new Git objects."""
    context = _local_export(path)
    return dict(export_dir=str(context['output']), export_id=context['manifest']['export_id'],
        branch=context['branch'], export_manifest_sha256=context['manifest_sha'],
        prepared_commit=context['receipt']['commit'] if context['receipt'] else None,
        tree=context['tree'], verified=True, network_requests=0)


def _reference_path(branch):
    return API_PREFIX+'/git/ref/heads/'+branch


def _reference_sha(client, branch):
    try:
        result = client.request('GET', _reference_path(branch))
    except RESTError as exc:
        if exc.status == 404 and exc.category == 'not_found' and exc.response_kind in (None, 'json'):
            return None
        raise
    value = result.get('object', {})
    if (result.get('ref') != 'refs/heads/'+branch or value.get('type') != 'commit'
            or not re.fullmatch(r'[0-9a-f]{40}', value.get('sha', ''))):
        raise RESTError(category='unexpected_result_ref_payload')
    return value['sha']


def _exact_sha(response, expected, kind):
    if response.get('sha') != expected:
        raise RESTError(category=kind+'_sha_mismatch')


def _upload_objects(client, context, commit):
    expected_blobs = {entry['path']: entry['sha'] for entry in context['entries']}
    names = sorted(context['files'], key=lambda value: (value != 'README.md', value))
    for index, name in enumerate(names, 1):
        print(f'REST blob {index}/{len(names)}: {name}', flush=True)
        data = context['files'][name].read_bytes()
        if _object_sha('blob', data) != expected_blobs[name]:
            raise RESTError(category='local_export_changed_during_upload')
        response = client.request('POST', API_PREFIX+'/git/blobs',
            dict(content=base64.b64encode(data).decode('ascii'), encoding='base64'))
        _exact_sha(response, expected_blobs[name], 'blob')
    response = client.request('POST', API_PREFIX+'/git/trees', dict(tree=context['entries']))
    _exact_sha(response, context['tree'], 'tree')
    payload = parse_raw_commit(_raw_commit(previous._config().ROOT, commit))
    if payload['tree'] != context['tree']:
        raise RESTError(category='prepared_publication_tree_changed')
    response = client.request('POST', API_PREFIX+'/git/commits', payload)
    # GitHub must reproduce the exact prepared object, including both timezones
    # and final message LF. Some Git backends append one final LF to a supplied
    # message. Try that representation once, retaining every other field, then
    # still require the original prepared SHA before any ref can be created.
    if response.get('sha') != commit and payload['message'].endswith('\n'):
        payload = dict(payload, message=payload['message'][:-1])
        response = client.request('POST', API_PREFIX+'/git/commits', payload)
    # Never substitute a server-normalized object or relax the identity check.
    _exact_sha(response, commit, 'commit')


def publish(path=None, client=None):
    cfg, runtime = previous._config(), previous._runtime()
    with runtime.process_lock(cfg.OUTPUT_ROOT/'publication.lock'):
        context = _local_export(path)
        before = previous._git_state(cfg.ROOT)
        old = context['receipt']
        commit, tree = previous._orphan_commit(cfg.ROOT, context['files'],
            commit=old.get('commit') if old else None)
        if tree != context['tree'] or previous._git_state(cfg.ROOT) != before:
            raise RESTError(category='source_state_or_prepared_tree_changed')
        receipt = dict(old or {}, format=cfg.NAMESPACE+'_publication_v1', status='prepared',
            transport='github_rest', branch=context['branch'], commit=commit, tree=tree,
            export_manifest_sha256=context['manifest_sha'], export_id=context['manifest']['export_id'],
            public_repository=True, url=previous.WEB_REPOSITORY+'/tree/'+context['branch'],
            rest_tool_sha256=previous._sha(Path(__file__).resolve()), remote_ref_verified=False)
        if 'git_exit_code' in receipt:
            receipt['prior_git_exit_code'] = receipt.pop('git_exit_code')
        receipt.pop('rest_error', None)
        runtime.atomic_json(context['receipt_path'], receipt)
        try:
            client = GitHubAPI(cfg.ROOT) if client is None else client
            existing = _reference_sha(client, context['branch'])
            if existing not in (None, commit):
                raise RESTError(409, 'result_ref_conflict')
            if existing is None:
                _upload_objects(client, context, commit)
                current = _reference_sha(client, context['branch'])
                if current not in (None, commit):
                    raise RESTError(409, 'result_ref_conflict')
                if current is None:
                    try:
                        client.request('POST', API_PREFIX+'/git/refs',
                            dict(ref='refs/heads/'+context['branch'], sha=commit))
                    except RESTError as exc:
                        if exc.status not in (409, 422) or _reference_sha(client, context['branch']) != commit:
                            raise
            if _reference_sha(client, context['branch']) != commit:
                raise RESTError(category='result_ref_confirmation_mismatch')
            previous.verify_export_directory(context['output'])
            if previous._git_state(cfg.ROOT) != before:
                raise RESTError(category='source_head_or_index_changed_during_rest_upload')
        except RESTError as exc:
            receipt.update(status='failed', rest_error=exc.safe_details())
            runtime.atomic_json(context['receipt_path'], receipt)
            raise
        except Exception:
            # Never propagate an unexpected transport exception that may embed
            # a credential-bearing request, response body or URL.
            failure = RESTError(category='rest_upload_failed_local_evidence_preserved')
            receipt.update(status='failed', rest_error=failure.safe_details())
            runtime.atomic_json(context['receipt_path'], receipt)
            raise failure from None
        receipt.update(status='uploaded', remote_ref_verified=True)
        runtime.atomic_json(context['receipt_path'], receipt)
    print('Uploaded branch: '+context['branch'])
    print('Result commit: '+commit)
    print('Transport: github_rest; exact prepared commit preserved')
    print('Read URL: '+receipt['url'])
    print('Deleting the result branch later does not guarantee Git-history erasure.')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--publish', action='store_true')
    actions.add_argument('--verify-only', action='store_true')
    parser.add_argument('--export-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            print(json.dumps(verify_only(args.export_dir), indent=2, sort_keys=True))
        else:
            publish(args.export_dir)
    except RESTError as exc:
        parser.exit(1, str(exc)+'; local export and prepared commit are preserved.\n')
    except ImportError:
        parser.exit(1, str(RESTError(category='local_runtime_unavailable'))+'\n')
    except Exception:
        # Legacy local guards can raise ValueError/OSError. Do not turn their
        # potentially machine-specific text into a traceback in shared logs.
        parser.exit(1, str(RESTError(category='local_validation_failed'))
            +'; local export and prepared commit are preserved.\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
