"""Serve immutable public HTTP fixture bytes inside an offline task image."""
import hashlib
import io
import json
from email.message import Message
from pathlib import Path
import urllib.error
import urllib.request
import urllib.response
import zipfile

_ORIGINAL = urllib.request.urlopen
# Read into memory and close the archive: Pandas checks open-file descriptors
# around each test, so a retained archive/file handle would change its result.
with zipfile.ZipFile(io.BytesIO(Path('/opt/hapt-pandas-http.zip').read_bytes())) as archive:
    _INDEX = json.loads(archive.read('index.json'))
    _BODIES = {url: archive.read(spec['body']) for url, spec in _INDEX.items()}
for _url, _spec in _INDEX.items():
    if hashlib.sha256(_BODIES[_url]).hexdigest() != _spec['sha256']:
        raise RuntimeError('public HTTP fixture checksum mismatch')


class CachedHTTPResponse(urllib.response.addinfourl):
    # Python 3.8's addinfourl lacks HTTPResponse.status.
    @property
    def status(self):
        return self.code


def cached_urlopen(request, *args, **kwargs):
    url = request.full_url if isinstance(request, urllib.request.Request) else str(request)
    spec = _INDEX.get(url)
    if spec is None:
        return _ORIGINAL(request, *args, **kwargs)
    headers = Message()
    for key, value in spec['headers'].items():
        headers[key] = value
    body = io.BytesIO(_BODIES[url])
    if spec['status'] >= 400:
        raise urllib.error.HTTPError(url, spec['status'], 'Not Found', headers, body)
    return CachedHTTPResponse(body, headers, url, spec['status'])


# Install before conftest imports Pandas and captures urllib function aliases.
urllib.request.urlopen = cached_urlopen


def pytest_unconfigure(config):
    urllib.request.urlopen = _ORIGINAL
