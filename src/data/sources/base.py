"""Base interfaces and network utilities for data sourcing."""

import socket
import urllib.request
import ssl
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List

# Known DNS fallbacks for servers that may have local router DNS lookup failures
DNS_CACHE = {
    "www.football-data.co.uk": "217.160.0.246",
    "football-data.co.uk": "217.160.0.246"
}

_orig_getaddrinfo = socket.getaddrinfo


def _custom_getaddrinfo(host, port, *args, **kwargs):
    if host in DNS_CACHE:
        host = DNS_CACHE[host]
    return _orig_getaddrinfo(host, port, *args, **kwargs)


socket.getaddrinfo = _custom_getaddrinfo


def download_file(url: str, dest_path: Path, headers: Dict[str, str] = None) -> Path:
    """Download a remote file reliably to destination path."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url, headers=req_headers)
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        with open(dest_path, "wb") as f:
            while chunk := resp.read(16384):
                f.write(chunk)
    return dest_path


class BaseDataSource(ABC):
    """Abstract base class for all sport-specific data sources."""

    @abstractmethod
    def download(self, target_dir: Path) -> List[Path]:
        """Download raw data files to target directory."""
        pass

    @abstractmethod
    def parse_matches(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse raw file into standardized match dictionaries."""
        pass
