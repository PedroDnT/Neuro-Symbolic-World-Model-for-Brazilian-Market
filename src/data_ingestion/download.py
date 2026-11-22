"""Paths and a download helper with a local cache and exponential backoff."""
from __future__ import annotations

import time
from pathlib import Path

import requests

from src.protocol import data_fetch_error

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = ROOT / "results"

CVM_BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA"


def download(url: str, dest: Path, module: str, retries: int = 4, timeout: int = 900) -> Path:
    """Return `dest`. Download it first when it is not in the cache.

    A failed download raises ERR_DATA_FETCH. It never returns a partial file.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with open(part, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        handle.write(chunk)
            if part.stat().st_size == 0:
                raise ValueError("empty response")
            part.replace(dest)
            return dest
        except (requests.RequestException, ValueError, OSError) as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise data_fetch_error(module, last_error) from last_error
