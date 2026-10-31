"""The creator's secret key and the ledger of copies issued with it.

The key decides every tile, so whoever holds it can both mark and accuse. It lives in a file readable only
by its owner. The ledger is a plain JSON-lines log kept beside it: who got which copy, when, and the hashes.
It is a convenience (tiles are derived from key + label, so a list of handles also works), never the evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import time
from pathlib import Path

ENV_KEY = "INVISIMARK_KEY"
ENV_DIR = "INVISIMARK_HOME"


class KeyError_(RuntimeError):
    pass


def home() -> Path:
    return Path(os.environ.get(ENV_DIR) or Path.home() / ".config" / "invisimark")


def key_path(explicit: str | None = None) -> Path:
    return Path(explicit or os.environ.get(ENV_KEY) or home() / "key")


def ledger_path(explicit: str | None = None) -> Path:
    return Path(explicit) if explicit else home() / "ledger.jsonl"


def create_key(path: Path, force: bool = False) -> bytes:
    if path.exists() and not force:
        raise KeyError_(f"{path} already exists; pass --force to replace it (copies issued with the old key can no longer be traced)")
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(key.hex() + "\n")
    return key


def load_key(path: Path) -> bytes:
    try:
        text = path.read_text().strip()
    except FileNotFoundError:
        raise KeyError_(f"no key at {path}; run `invisimark keygen` first") from None
    try:
        key = bytes.fromhex(text)
    except ValueError:
        raise KeyError_(f"{path} is not a hex key file") from None
    if len(key) < 16:
        raise KeyError_(f"{path} holds a key shorter than 16 bytes")
    return key


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", label.strip().casefold()).strip("-") or "recipient"


def unique_slugs(labels: list[str]) -> dict[str, str]:
    """A filename-safe, collision-free stem per label: 'Bob!' and 'bob?' must not overwrite each other's copy."""
    out: dict[str, str] = {}
    taken: set[str] = set()
    for label in labels:
        stem = slug(label)
        if stem in taken:
            stem = f"{stem}-{hashlib.sha256(label.encode()).hexdigest()[:6]}"
        taken.add(stem)
        out[label] = stem
    return out


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record_issue(ledger: Path, **fields) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields}
    with open(ledger, "a") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def read_ledger(ledger: Path) -> list[dict]:
    if not ledger.exists():
        return []
    rows = []
    for line in ledger.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def recipients_in(ledger: Path) -> list[str]:
    seen: dict[str, None] = {}
    for row in read_ledger(ledger):
        seen.setdefault(row["recipient"], None)
    return list(seen)
