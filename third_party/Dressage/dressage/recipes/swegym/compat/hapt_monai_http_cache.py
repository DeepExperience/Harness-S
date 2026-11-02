"""Retrieve an immutable official MONAI test image without external network."""
import hashlib
from email.message import Message
from pathlib import Path
import shutil
import tempfile
import urllib.request

_URL = 'https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/Prostate_T2W_AX_1.nii'
_PATH = Path('/opt/hapt-test-data/Prostate_T2W_AX_1.nii')
_SHA256 = 'a14231f539c0f365a5f83f2a046969a9b9870e56ffd126fd8e7242364d25938a'
_ORIGINAL = urllib.request.urlretrieve
if hashlib.sha256(_PATH.read_bytes()).hexdigest() != _SHA256:
    raise RuntimeError('official MONAI test image checksum mismatch')


def cached_urlretrieve(url, filename=None, reporthook=None, data=None):
    if url != _URL or data is not None:
        return _ORIGINAL(url, filename, reporthook, data)
    if filename is None:
        with tempfile.NamedTemporaryFile(delete=False) as temporary:
            filename = temporary.name
    shutil.copyfile(_PATH, filename)
    size = _PATH.stat().st_size
    headers = Message(); headers['Content-Length'] = str(size)
    if reporthook:
        reporthook(0, max(size, 1), size)
        reporthook(1, max(size, 1), size)
    return filename, headers


urllib.request.urlretrieve = cached_urlretrieve


def pytest_unconfigure(config):
    urllib.request.urlretrieve = _ORIGINAL
