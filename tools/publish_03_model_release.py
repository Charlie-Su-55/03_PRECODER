#!/usr/bin/env python3
"""Publish a verified 03 model archive as immutable GitHub Release assets.

Files stream over strict HTTPS. A release remains a draft until GitHub reports
SHA256 digests for every expected asset. Repeat a failed command to resume.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

import model_release_03 as archive

ROOT = Path(__file__).resolve().parent.parent
ORIGIN = 'https://github.com/Charlie-Su-55/03_PRECODER.git'
API = 'https://api.github.com/repos/Charlie-Su-55/03_PRECODER'
UPLOADS = 'https://uploads.github.com/repos/Charlie-Su-55/03_PRECODER'
MAX_RESPONSE_BYTES = 4 * 1024**2
MAX_ASSET_BYTES = 2 * 1024**3 - 1
FALSE_VALUES = ('0', 'false', 'no', 'off')


class PublishError(RuntimeError):
    """Only classifications, never arbitrary remote bodies or secrets."""
    def __init__(self, status=None, category='publication_failed', *,
                 response_kind=None, github_request_id_present=False):
        self.status = status
        self.category = category
        self.response_kind = response_kind
        self.github_request_id_present = bool(github_request_id_present)
        super().__init__('GitHub model release: ' + json.dumps(dict(
            status=status, category=category, response_kind=response_kind,
            github_request_id_present=self.github_request_id_present), sort_keys=True))


def _environment():
    return {k: v for k, v in os.environ.items()
            if not k.startswith('GIT_TRACE') and k not in ('GIT_CURL_VERBOSE', 'SSLKEYLOGFILE')}


def _git(root, *args, data=None, optional=False):
    try:
        result = subprocess.run(['git', *args], cwd=root, input=data,
            capture_output=True, env=_environment(), timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise PublishError(category='local_git_unavailable') from None
    if optional and result.returncode == 1:
        return None
    if result.returncode:
        raise PublishError(category='local_git_command_failed')
    try:
        return result.stdout.decode('utf-8').strip()
    except UnicodeError:
        raise PublishError(category='local_git_output_invalid') from None


def _config(root, key, url=None):
    args = ('config', '--get-urlmatch', key, url) if url else ('config', '--get', key)
    return _git(root, *args, optional=True)


def _network_config(root, url=API):
    if _git(root, 'remote', 'get-url', 'origin') != ORIGIN:
        raise PublishError(category='official_origin_required')
    disabled = os.environ.get('GIT_SSL_NO_VERIFY', '').strip().lower()
    checks = [_config(root, 'http.sslVerify', target) for target in (ORIGIN, url)]
    if (disabled not in ('', *FALSE_VALUES)
            or any(value is not None and value.lower() in FALSE_VALUES for value in checks)):
        raise PublishError(category='tls_verification_disabled')
    def choose(key, environment):
        return (os.environ.get(environment) or _config(root, key, url)
                or _config(root, key, ORIGIN) or None)
    cafile = choose('http.sslCAInfo', 'GIT_SSL_CAINFO')
    capath = choose('http.sslCAPath', 'GIT_SSL_CAPATH')
    proxy = _config(root, 'http.proxy', url)
    if proxy is None:
        proxy = _config(root, 'remote.origin.proxy')
    if proxy is None:
        proxy = _config(root, 'http.proxy', ORIGIN)
    if proxy is None:
        proxy = (os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
                 or os.environ.get('all_proxy') or os.environ.get('ALL_PROXY'))
    if proxy:
        try:
            parsed = urllib.parse.urlsplit(proxy)
            if (parsed.scheme != 'http' or not parsed.hostname
                    or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
                raise ValueError()
            parsed.port
        except ValueError:
            raise PublishError(category='unsupported_proxy_configuration') from None
    return dict(cafile=str(Path(cafile).expanduser()) if cafile else None,
                capath=str(Path(capath).expanduser()) if capath else None, proxy=proxy or None)


def _token(root):
    environment = _environment()
    environment.update(GIT_TERMINAL_PROMPT='0', GIT_ASKPASS='/bin/false',
                       SSH_ASKPASS='/bin/false', GCM_INTERACTIVE='never')
    try:
        result = subprocess.run(['git', '-c', 'credential.interactive=false', 'credential', 'fill'],
            cwd=root, input=b'protocol=https\nhost=github.com\npath=Charlie-Su-55/03_PRECODER.git\n\n',
            capture_output=True, env=environment, timeout=15, check=False)
        raw = result.stdout.decode('utf-8') if result.returncode == 0 else ''
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        raw = ''
    values = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
    token = values.get('password', '')
    if not token and (sys.stdin.isatty() or sys.stderr.isatty()):
        try:
            token = getpass.getpass('GitHub token (local terminal only; never the proxy/sudo password): ')
        except (EOFError, KeyboardInterrupt):
            raise PublishError(category='git_credential_unavailable') from None
    if not token or any(c in token for c in '\r\n\0'):
        raise PublishError(category='git_credential_unavailable')
    try:
        token.encode('ascii')
    except UnicodeError:
        raise PublishError(category='git_credential_invalid') from None
    return token


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        raise PublishError(status=code, category='authenticated_redirect_rejected')


def _checked_url(url):
    try:
        parsed = urllib.parse.urlsplit(url)
        if (parsed.scheme != 'https' or parsed.username is not None or parsed.password is not None
                or parsed.port not in (None, 443) or parsed.fragment
                or parsed.hostname not in ('api.github.com', 'uploads.github.com')
                or not parsed.path.startswith('/repos/Charlie-Su-55/03_PRECODER/releases')):
            raise ValueError()
    except ValueError:
        raise PublishError(category='api_target_rejected') from None
    return parsed


class GitHubAPI:
    def __init__(self, root=ROOT, *, timeout=120, opener_factory=None):
        self.root = Path(root)
        self.timeout = timeout
        self._openers = {}
        for host in ('api.github.com', 'uploads.github.com'):
            settings = _network_config(root, 'https://' + host + '/')
            try:
                # Construct explicitly: create_default_context() honors
                # SSLKEYLOGFILE before a caller can disable TLS key logging.
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                if settings['cafile'] or settings['capath']:
                    context.load_verify_locations(cafile=settings['cafile'], capath=settings['capath'])
                else:
                    context.load_default_certs()
                context.check_hostname = True
                context.verify_mode = ssl.CERT_REQUIRED
                if hasattr(context, 'keylog_filename'):
                    context.keylog_filename = None
            except (OSError, ssl.SSLError):
                raise PublishError(category='trusted_ca_configuration_unavailable') from None
            handlers = [urllib.request.ProxyHandler({'https': settings['proxy']} if settings['proxy'] else {}),
                        urllib.request.HTTPSHandler(context=context), _NoRedirect()]
            self._openers[host] = (opener_factory or urllib.request.build_opener)(*handlers)
        self._token = _token(root)

    def request(self, method, url, payload=None, file_path=None):
        parsed = _checked_url(url)
        if method not in ('GET', 'POST', 'PATCH', 'DELETE') or (payload is not None and file_path is not None):
            raise PublishError(category='api_request_rejected')
        if method == 'DELETE' and (parsed.hostname != 'api.github.com' or parsed.query
                or not re.fullmatch(r'/repos/Charlie-Su-55/03_PRECODER/releases/assets/[0-9]+', parsed.path)
                or payload is not None or file_path is not None):
            raise PublishError(category='api_request_rejected')
        headers = {'Authorization': 'Bearer ' + self._token,
                   'Accept': 'application/vnd.github+json',
                   'X-GitHub-Api-Version': '2022-11-28',
                   'User-Agent': '03-PRECODER-model-release'}
        stream = None
        try:
            if file_path is not None:
                path = Path(file_path)
                if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_ASSET_BYTES:
                    raise PublishError(category='asset_file_or_size_rejected')
                stream = path.open('rb')
                data = stream
                headers.update({'Content-Type': 'application/octet-stream',
                                'Content-Length': str(path.stat().st_size)})
            elif payload is not None:
                data = json.dumps(payload, allow_nan=False, separators=(',', ':')).encode()
                headers['Content-Type'] = 'application/json'
            else:
                data = None
            request = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with self._openers[parsed.hostname].open(request, timeout=self.timeout) as response:
                    if method == 'DELETE':
                        if response.status != 204:
                            raise PublishError(category='asset_placeholder_deletion_not_confirmed')
                        return None
                    raw = response.read(MAX_RESPONSE_BYTES + 1)
                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise PublishError(category='api_response_too_large')
                    try:
                        return json.loads(raw)
                    except (ValueError, UnicodeError):
                        raise PublishError(category='api_response_not_json') from None
            except urllib.error.HTTPError as error:
                kind = 'html' if 'html' in error.headers.get('Content-Type', '').lower() else 'other'
                request_id = bool(error.headers.get('X-GitHub-Request-Id'))
                if 'json' in error.headers.get('Content-Type', '').lower():
                    kind = 'json'
                error.close()
                raise PublishError(status=error.code, category='http_rejected', response_kind=kind,
                                   github_request_id_present=request_id) from None
            except (OSError, urllib.error.URLError):
                raise PublishError(category='network_or_tls_failure') from None
        finally:
            if stream is not None:
                stream.close()


def _asset_record(asset, expected):
    if (asset.get('name') != expected['name'] or asset.get('size') != expected['size']
            or asset.get('state') != 'uploaded'
            or asset.get('digest') != 'sha256:' + expected['sha256']):
        raise PublishError(category='remote_asset_digest_or_size_mismatch')


def _assets(api, release_id):
    result = {}
    for page in range(1, 101):
        rows = api.request('GET', f'{API}/releases/{release_id}/assets?per_page=100&page={page}')
        if not isinstance(rows, list):
            raise PublishError(category='remote_asset_list_invalid')
        for row in rows:
            if not isinstance(row, dict) or row.get('name') in result:
                raise PublishError(category='remote_asset_list_invalid')
            result[row.get('name')] = row
        if len(rows) < 100:
            return result
    raise PublishError(category='remote_asset_list_too_large')


def _find_draft_or_release(api, tag):
    # Some GitHub installations omit drafts from the tag lookup endpoint.
    # Authenticated release listing includes them and makes retries safe.
    found = []
    for page in range(1, 101):
        rows = api.request('GET', f'{API}/releases?per_page=100&page={page}')
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise PublishError(category='remote_release_list_invalid')
        found.extend(row for row in rows if row.get('tag_name') == tag)
        if len(found) > 1:
            raise PublishError(category='duplicate_remote_release_tag')
        if len(rows) < 100:
            return found[0] if found else None
    raise PublishError(category='remote_release_list_too_large')


def publish(bundle_dir, root=None, api=None):
    root = ROOT if root is None else Path(root)
    directory = Path(bundle_dir)
    manifest = archive.verify(directory)
    commit = manifest.get('source_commit')
    if not isinstance(commit, str) or not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise PublishError(category='archive_source_commit_required')
    manifest_bytes = (directory / 'manifest.json').read_bytes()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    tag = 'models-03-' + manifest['profile'] + '-' + digest[:12]
    marker = '03 model archive manifest SHA256: ' + digest
    expected = [dict(row) for row in manifest['parts']] + [dict(
        name='manifest.json', size=len(manifest_bytes), sha256=digest)]
    if any(row['size'] > MAX_ASSET_BYTES for row in expected):
        raise PublishError(category='asset_exceeds_github_size_limit')
    if api is None:
        api = GitHubAPI(root)
    try:
        release = api.request('GET', API + '/releases/tags/' + urllib.parse.quote(tag, safe=''))
    except PublishError as error:
        if error.status != 404 or not error.github_request_id_present:
            raise
        release = _find_draft_or_release(api, tag)
        if release is None:
            release = api.request('POST', API + '/releases', dict(
                tag_name=tag, target_commitish=commit, name=tag, draft=True, prerelease=False,
                body=marker + '\n\nOriginal complete checkpoints and provenance. '
                     'Keep all parts and manifest.json together; no training was started by this uploader.'))
    if (not isinstance(release, dict) or release.get('tag_name') != tag
            or release.get('target_commitish') != commit
            or marker not in release.get('body', '').splitlines()
            or type(release.get('id')) is not int or type(release.get('draft')) is not bool):
        raise PublishError(category='release_identity_mismatch')
    release_id = release['id']
    existing = _assets(api, release_id)
    allowed = {row['name'] for row in expected}
    if set(existing) - allowed:
        raise PublishError(category='unexpected_remote_assets')
    for row in expected:
        if row['name'] in existing:
            asset = existing[row['name']]
            if (release['draft'] and asset.get('state') == 'starter' and asset.get('size') == 0
                    and asset.get('digest') in (None, '') and type(asset.get('id')) is int):
                # GitHub can leave this empty placeholder after an upload 502.
                # Only this archive's verified draft and expected empty asset
                # qualify; completed or conflicting remote files never do.
                api.request('DELETE', f'{API}/releases/assets/{asset["id"]}')
                del existing[row['name']]
            else:
                _asset_record(asset, row)
        if row['name'] not in existing:
            if not release['draft']:
                raise PublishError(category='published_release_missing_assets')
            print('Uploading asset: ' + row['name'], flush=True)
            result = api.request('POST', f'{UPLOADS}/releases/{release_id}/assets?name='
                                 + urllib.parse.quote(row['name'], safe=''), file_path=directory / row['name'])
            _asset_record(result, row)
    remote = _assets(api, release_id)
    if set(remote) != allowed:
        raise PublishError(category='remote_asset_set_mismatch')
    for row in expected:
        _asset_record(remote[row['name']], row)
    if release['draft']:
        release = api.request('PATCH', f'{API}/releases/{release_id}', dict(draft=False, make_latest='false'))
    if release.get('draft') is not False or release.get('tag_name') != tag:
        raise PublishError(category='release_publication_not_confirmed')
    url = 'https://github.com/Charlie-Su-55/03_PRECODER/releases/tag/' + tag
    return dict(url=url, tag=tag, manifest_sha256=digest, assets_verified=len(expected))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--publish', action='store_true')
    action.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = (archive.verify(args.bundle_dir) if args.verify_only
                  else publish(args.bundle_dir))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (PublishError, ValueError, OSError) as error:
        print(str(error) if isinstance(error, PublishError) else
              'Local archive verification failed: ' + type(error).__name__)
        print('Local files are preserved. A partial remote release remains a draft; repeat the same command to resume.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
