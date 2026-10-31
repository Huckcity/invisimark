"""Keyed watermark tiles.

Every mark is the sum of two periodic, band-limited noise tiles:

* the *sync* tile, shared by every recipient of one key, lets the detector recover scale and crop;
* the *id* tile, derived from the key and the recipient's label, says who the copy went to.

Tiles are pseudo-random but fully determined by (key, label), so a ledger is a convenience, not a
necessity: anyone holding the key and a list of candidate labels can re-derive every mark.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

TILE = 128


@dataclass(frozen=True)
class Profile:
    """Shape of the mark. Alphas are only a ratio: the embedder scales them so the added pattern has a chosen RMS.

    Defaults come from the equal-distortion variant study in bench/: the sync tile sits in the low band, which
    survives compression and blur best, the id tile in the mid band, and the mask gives textured areas most of
    the energy because that is where both the eye and the encoder let it through.
    """

    alpha_sync: float = 2.0
    alpha_id: float = 1.8
    band_lo: float = 3.0
    band_hi: float = 18.0
    sync_lo: float | None = 1.5      # the sync tile may sit in its own band; None means the shared band
    sync_hi: float | None = 8.0
    flat: bool = True                # equal-magnitude spectrum with random phase: every key is equally detectable
    mask_floor: float = 0.15
    mask_ceil: float = 2.6
    mask_ref: float = 30.0


DEFAULT = Profile()


@lru_cache(maxsize=8)
def _band(lo: float, hi: float) -> np.ndarray:
    fy = np.fft.fftfreq(TILE) * TILE
    fx = np.fft.rfftfreq(TILE) * TILE
    r = np.hypot(fy[:, None], fx[None, :])
    # raised-cosine shoulders, one cycle wide, so the tile has no ringing
    up = np.clip((r - (lo - 1.0)) / 1.0, 0.0, 1.0)
    down = np.clip(((hi + 1.0) - r) / 1.0, 0.0, 1.0)
    band = 0.5 - 0.5 * np.cos(np.pi * up)
    band *= 0.5 - 0.5 * np.cos(np.pi * down)
    band[0, 0] = 0.0
    return band.astype(np.float32)


def _seed(key: bytes, label: bytes) -> int:
    return int.from_bytes(hmac.new(key, label, hashlib.sha256).digest()[:8], "little")


def make_tile(key: bytes, label: bytes, profile: Profile = DEFAULT, band: tuple[float, float] | None = None) -> np.ndarray:
    """A zero-mean, unit-variance, band-limited, TILE x TILE periodic pattern."""
    lo, hi = band or (profile.band_lo, profile.band_hi)
    rng = np.random.Generator(np.random.PCG64(_seed(key, label)))
    noise = rng.standard_normal((TILE, TILE)).astype(np.float32)
    spec = np.fft.rfft2(noise)
    if profile.flat:
        # A Gaussian spectrum leaves some keys with their energy in a few components, which makes them poorer
        # matched filters. Keep the random phases, give every component the same magnitude.
        spec = spec / np.maximum(np.abs(spec), 1e-9)
    spec = spec * _band(lo, hi)
    tile = np.fft.irfft2(spec, s=(TILE, TILE)).astype(np.float32)
    tile -= tile.mean()
    tile /= tile.std()
    return tile


def sync_tile(key: bytes, profile: Profile = DEFAULT) -> np.ndarray:
    band = (profile.sync_lo, profile.sync_hi) if profile.sync_lo is not None and profile.sync_hi is not None else None
    return make_tile(key, b"sync", profile, band)


def id_tile(key: bytes, recipient: str, profile: Profile = DEFAULT) -> np.ndarray:
    return make_tile(key, b"id:" + recipient.encode("utf-8"), profile)


def pattern(key: bytes, recipient: str, profile: Profile = DEFAULT) -> np.ndarray:
    """The tile actually added to the luma plane (before the perceptual mask)."""
    return profile.alpha_sync * sync_tile(key, profile) + profile.alpha_id * id_tile(key, recipient, profile)


def tile_to_plane(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    reps = (-(-height // TILE), -(-width // TILE))
    return np.tile(tile, reps)[:height, :width]
