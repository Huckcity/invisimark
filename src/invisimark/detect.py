"""Detection: recover scale, crop and orientation from the sync tile, then score every candidate recipient.

A leaked copy has been re-encoded, probably rescaled and cropped, possibly re-filmed off a screen. The
mark is a periodic pattern, so the detector does not need the original. It works in four steps:

1. Band-pass and soft-clip each sampled frame (so a few razor-sharp edges cannot outvote the mark), then
   average the frames. The mark is the same in every frame; the picture is not.
2. For each candidate scale, resize the average until the pattern's period is TILE pixels again and fold it onto
   one TILE x TILE cell, which averages the picture away and keeps the pattern. Flips and transposes of the image
   are flips and transposes of that cell, so all eight orientations cost almost nothing.
3. Correlate the cell with the sync tile (whitened, so busy low frequencies stop drowning the mark). The scale,
   orientation and shift that peak are the leaked copy's geometry.
4. At that geometry, correlate with every candidate recipient's tile and compare against a null of decoys.

A trusted-area mask keeps letterbox bars, a static desktop around a screen capture and the video's own border
out of every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import media
from .tile import DEFAULT, TILE, Profile, id_tile, sync_tile

DECOYS = 1500
SYNC_GATE = 5.5                 # below this the alignment is noise and no candidate is scored
SYNC_SURE = 7.5                 # sync alone this strong means marked; a null of 260 runs peaked at 6.6 (mean 5.0, sd 0.64)
SIGMA_LO = 0.8
COARSE_CANON = 1024             # canonical pixels (eight tiles) of picture the coarse scale walk looks at
REFINE_CANON = 2048             # and the fine alignment
SIGMAS = (6.0, 12.0, 24.0)      # band-pass widths, in leaked pixels, for scales near 0.5, 1 and 2


@dataclass
class Match:
    recipient: str
    z: float


def id_threshold(n_candidates: int) -> float:
    """z needed to name one of n candidates: a Gaussian tail of about 1e-7 per candidate, never below 6."""
    return max(6.0, float(np.sqrt(2.0 * np.log(max(n_candidates, 2) * 1e7))) - 0.6)


@dataclass
class Report:
    marked: bool                     # this key's mark is present: sync alone above SYNC_SURE, or sync above the gate plus a named candidate
    sync_z: float
    scale: float                     # leaked pixels per original pixel
    shift: tuple[int, int]
    matches: list[Match] = field(default_factory=list)   # best first
    frames: int = 0
    orientation: int = 0             # 0 as filmed; bit 0 mirrored left-right, bit 1 upside down, bit 2 transposed
    candidates: int = 0

    @property
    def mirrored(self) -> bool:
        return self.orientation != 0

    @property
    def best(self) -> Match | None:
        return self.matches[0] if self.matches else None

    def lead(self, z_min: float = 4.5, margin: float = 1.5) -> str | None:
        """A hint short of a verdict: the top candidate when it clearly leads but cannot be named."""
        if not self.marked or not self.matches or self.verdict() is not None:
            return None
        if self.matches[0].z < z_min:
            return None
        if len(self.matches) > 1 and self.matches[0].z - self.matches[1].z < margin:
            return None
        return self.matches[0].recipient

    def verdict(self, z_min: float | None = None, margin: float = 2.0) -> str | None:
        """The accused recipient, or None when the evidence is not strong enough to name anyone."""
        if not self.marked or not self.matches:
            return None
        if z_min is None:
            z_min = id_threshold(self.candidates)
        if self.matches[0].z < z_min:
            return None
        if len(self.matches) > 1 and self.matches[0].z - self.matches[1].z < margin:
            return None
        return self.matches[0].recipient


# ---------------------------------------------------------------------------------------------
# trusted area


def _letterbox(img: np.ndarray) -> np.ndarray:
    """0 for dead flat dark areas (letterbox bars), 1 elsewhere."""
    k = np.ones((9, 9), np.uint8)
    flat = (cv2.dilate(img, k) - cv2.erode(img, k)) <= 1.5
    dark = cv2.blur(img, (9, 9)) < 22
    return (~(flat & dark)).astype(np.float32)


def build_weights(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Where to trust the picture: not letterbox, and (for a screen capture) not the static desktop around the video.

    Pixels that never change across the sampled frames are treated as chrome, then the survivors are closed over
    (flat sky inside the video is also static) and eroded so the video's own border cannot leak into the fold.
    """
    base = [_letterbox(f) for f in frames]
    moving = np.ones(frames[0].shape, bool)
    if len(frames) >= 4:
        spread = cv2.blur(np.std(np.stack(frames), axis=0).astype(np.float32), (7, 7))
        cand = spread > 2.5
        if 0.03 < cand.mean() < 0.999:
            closed = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
            moving = closed.astype(bool)
    gate = cv2.erode(moving.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(np.float32)
    return [b * gate if gate.any() else b for b in base]


# ---------------------------------------------------------------------------------------------
# averaging, resizing and folding


def prep(img: np.ndarray, wt: np.ndarray | None, sigma_hi: float, sigma_lo: float = SIGMA_LO) -> np.ndarray:
    """Band-pass, then soft-clip at three robust standard deviations."""
    x = cv2.GaussianBlur(img, (0, 0), sigma_lo) - cv2.GaussianBlur(img, (0, 0), sigma_hi)
    probe = np.abs(x[::3, ::3])
    if wt is not None:
        keep = wt[::3, ::3] > 0.5
        probe = probe[keep] if keep.any() else probe
    tau = 3.0 * (float(np.median(probe)) * 1.4826 + 1e-3)
    return (tau * np.tanh(x / tau)).astype(np.float32)


@dataclass
class Stack:
    """Frames prepped at one band-pass width and averaged: a weighted numerator and its weight."""

    num: np.ndarray
    den: np.ndarray
    half_num: np.ndarray | None = None
    half_den: np.ndarray | None = None

    _uniform: bool | None = None

    @property
    def uniform(self) -> bool:
        if self._uniform is None:
            self._uniform = bool(self.den.min() == self.den.max() > 0)
        return self._uniform

    def half(self) -> "Stack":
        if self.half_num is None:
            self.half_num = cv2.resize(self.num, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            self.half_den = cv2.resize(self.den, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        return Stack(self.half_num, self.half_den, _uniform=self.uniform)


def build_stack(frames, weights, sigma_hi: float) -> Stack:
    num = np.zeros(frames[0].shape, np.float32)
    den = np.zeros(frames[0].shape, np.float32)
    for f, w in zip(frames, weights):
        num += prep(f, w, sigma_hi) * w
        den += w
    return Stack(num, den)


def _fold_sum(img: np.ndarray) -> np.ndarray:
    """Sum an image onto one TILE x TILE cell. The whole tiles go through two contiguous reductions; the ragged
    right and bottom edges are padded and folded separately, so nothing is thrown away and nothing big is copied."""
    h, w = img.shape
    ny, nx = h // TILE, w // TILE
    cell = np.zeros((TILE, TILE), np.float32)
    if ny and nx:
        body = img[: ny * TILE, : nx * TILE]
        rows = body.reshape(ny, TILE, nx * TILE).sum(axis=0)
        cell += rows.reshape(TILE, nx, TILE).sum(axis=1)
    if ny and w > nx * TILE:                            # right strip beside the whole tiles
        strip = np.zeros((ny * TILE, TILE), np.float32)
        part = img[: ny * TILE, nx * TILE :]
        strip[:, : part.shape[1]] = part
        cell += strip.reshape(-1, TILE, TILE).sum(axis=0)
    if h > ny * TILE:                                   # bottom strip, full width
        part = img[ny * TILE :, :]
        padded = np.zeros((TILE, -(-w // TILE) * TILE), np.float32)
        padded[: part.shape[0], : w] = part
        cell += padded.reshape(TILE, -1, TILE).sum(axis=1)
    return cell


@lru_cache(maxsize=256)
def _count_cell(h: int, w: int) -> np.ndarray:
    return _fold_sum(np.ones((h, w), np.float32))


def cell_of(num: np.ndarray, den: np.ndarray | None, scale: float) -> np.ndarray:
    """Undo a scale of `scale` (leaked px per original px) so the period is TILE again, then fold to one cell.
    den None means every pixel counts equally."""
    interp = cv2.INTER_AREA if scale > 1.0 else cv2.INTER_LINEAR
    resized = abs(scale - 1.0) > 1e-6
    if resized:
        num = cv2.resize(num, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=interp)
    if den is None:
        bottom = _count_cell(*num.shape)
    else:
        if resized:
            den = cv2.resize(den, (num.shape[1], num.shape[0]), interpolation=cv2.INTER_LINEAR)
        bottom = _fold_sum(np.ascontiguousarray(den))
    top = _fold_sum(np.ascontiguousarray(num))
    return np.where(bottom > 1e-3, top / np.maximum(bottom, 1e-3), 0.0).astype(np.float32)


def window(num: np.ndarray, den: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray]:
    """At most side x side pixels, centred on where the trusted weight is, so a huge picture costs no more than a small one."""
    h, w = den.shape
    side = max(side, 64)
    if h <= side and w <= side:
        return num, den
    rows, cols = den.sum(axis=1), den.sum(axis=0)
    cy = int(np.dot(np.arange(h), rows) / max(rows.sum(), 1e-9))
    cx = int(np.dot(np.arange(w), cols) / max(cols.sum(), 1e-9))
    y0 = int(np.clip(cy - side // 2, 0, max(h - side, 0)))
    x0 = int(np.clip(cx - side // 2, 0, max(w - side, 0)))
    return num[y0 : y0 + side, x0 : x0 + side], den[y0 : y0 + side, x0 : x0 + side]


def orient(cell: np.ndarray, o: int) -> np.ndarray:
    out = cell
    if o & 1:
        out = out[:, ::-1]
    if o & 2:
        out = out[::-1, :]
    if o & 4:
        out = out.T
    return np.ascontiguousarray(out)


# ---------------------------------------------------------------------------------------------
# correlation


def whitener(cell: np.ndarray) -> np.ndarray:
    """Per-frequency weights that flatten the picture's own spectrum, so busy low frequencies stop drowning the mark."""
    power = np.abs(np.fft.rfft2(cell)) ** 2
    fy = np.fft.fftfreq(TILE) * TILE
    fx = np.fft.rfftfreq(TILE) * TILE
    radius = np.rint(np.hypot(fy[:, None], fx[None, :])).astype(int)
    prof = np.bincount(radius.ravel(), weights=power.ravel()) / np.maximum(np.bincount(radius.ravel()), 1)
    prof = np.convolve(np.pad(prof, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    return (1.0 / np.sqrt(np.maximum(prof[radius], 1e-9))).astype(np.float32)


def _peak(surface: np.ndarray) -> tuple[float, tuple[int, int]]:
    """z-score of the highest point of a correlation surface, and where it is.

    The baseline is the median and MAD, not the mean and standard deviation: a strong peak inflates the latter
    and would cap z at about 9 however good the mark is, which makes the statistic useless for judging margin.
    """
    idx = np.unravel_index(int(np.argmax(surface)), surface.shape)
    med = float(np.median(surface))
    sd = float(np.median(np.abs(surface - med))) * 1.4826
    return ((float(surface[idx]) - med) / sd if sd > 0 else 0.0), (int(idx[0]), int(idx[1]))


def sync_peak(cell: np.ndarray, sync_f: np.ndarray, orientations=(0,)) -> tuple[float, int, tuple[int, int]]:
    """Best (z, orientation, shift) of the whitened sync correlation over the given orientations."""
    w = whitener(cell)
    tile_f = sync_f * w
    best = (-1.0, 0, (0, 0))
    for o in orientations:
        c = cell if o == 0 else orient(cell, o)
        surf = np.fft.irfft2(np.fft.rfft2(c) * w * np.conj(tile_f), s=(TILE, TILE))
        z, shift = _peak(surf)
        if z > best[0]:
            best = (z, o, shift)
    return best


# ---------------------------------------------------------------------------------------------
# scale search


def _stack_for(stacks: dict[float, Stack], frames, weights, scale: float) -> Stack:
    sigma = min(SIGMAS, key=lambda s: abs(np.log(s / (12.0 * scale))))
    if sigma not in stacks:
        stacks[sigma] = build_stack(frames, weights, sigma)
    return stacks[sigma]


def coarse_search(stacks, frames, weights, sync_f, lo=0.3, hi=3.2, step=0.002, topk=3):
    """Walk the scale axis (at half resolution when the picture is big enough); return the best few
    well-separated (z, scale, orientation)."""
    use_half = min(frames[0].shape) >= 240
    scales = np.exp(np.arange(np.log(lo), np.log(hi), step))
    scored = []
    for r in scales:
        st = _stack_for(stacks, frames, weights, r)
        if use_half:
            st, r_eff = st.half(), r / 2.0
        else:
            r_eff = r
        num, den = window(st.num, st.den, int(COARSE_CANON * r_eff))
        z, o, _ = sync_peak(cell_of(num, None if st.uniform else den, r_eff), sync_f, range(8))
        scored.append((z, float(r), o))
    scored.sort(reverse=True)
    out: list[tuple[float, float, int]] = []
    for z, r, o in scored:
        if all(abs(np.log(r / q)) > 0.02 for _, q, _ in out):
            out.append((z, r, o))
        if len(out) == topk:
            break
    return out


def refine(stacks, frames, weights, sync_f, centre, orientation, span, step):
    best = (-1.0, centre, (0, 0))
    for s in centre * np.exp(np.arange(-span, span + 1e-12, step)):
        st = _stack_for(stacks, frames, weights, s)
        num, den = window(st.num, st.den, int(REFINE_CANON * s))
        z, _, shift = sync_peak(cell_of(num, None if st.uniform else den, s), sync_f, (orientation,))
        if z > best[0]:
            best = (z, float(s), shift)
    return best


# ---------------------------------------------------------------------------------------------
# the detector


def detect_frames(
    frames: list[np.ndarray],
    key: bytes,
    candidates: list[str],
    profile: Profile = DEFAULT,
    scale_hint: float | None = None,
    sync_min_z: float = SYNC_GATE,
    search: np.ndarray | None = None,
) -> Report:
    """`search` aligns on that tile in place of the sync tile, for marks whose whole pattern is already known."""
    sync_f = np.fft.rfft2(sync_tile(key, profile) if search is None else search)
    weights = build_weights(frames)
    stacks: dict[float, Stack] = {}

    if scale_hint is not None:
        st = _stack_for(stacks, frames, weights, scale_hint)
        scored = [(sync_peak(cell_of(st.num, None if st.uniform else st.den, scale_hint), sync_f, (o,))[0], scale_hint, o)
                  for o in range(8)]
        peaks = sorted(scored, reverse=True)[:1]
        span, step = 0.004, 0.0004
    else:
        peaks = coarse_search(stacks, frames, weights, sync_f)
        span, step = 0.006, 0.0004

    best = (-1.0, 1.0, 0)
    for _, r, o in peaks:
        _, r1, _ = refine(stacks, frames, weights, sync_f, r, o, span, step)
        z, r2, _ = refine(stacks, frames, weights, sync_f, r1, o, 0.0004, 0.00008)
        if z > best[0]:
            best = (z, r2, o)
    _, scale, o = best

    st = _stack_for(stacks, frames, weights, scale)
    cell = orient(cell_of(st.num, None if st.uniform else st.den, scale), o)
    sync_z, _, shift = sync_peak(cell, sync_f)
    report = Report(
        marked=sync_z >= SYNC_SURE, sync_z=sync_z, scale=scale, shift=shift, frames=len(frames),
        orientation=o, candidates=len(candidates),
    )
    if candidates and sync_z >= sync_min_z:
        report.matches = score_candidates(cell, key, candidates, profile, shift)
        # A candidate that stands six deviations clear of a null of decoys is evidence of the mark in itself.
        if report.matches and report.matches[0].z >= id_threshold(len(candidates)):
            report.marked = True
    return report


def score_candidates(cell, key, candidates, profile, shift, decoys: int = DECOYS):
    """z-score every candidate against the folded cell at the recovered alignment."""
    w = whitener(cell)
    cell_f = np.fft.rfft2(cell) * w
    sync_f = np.fft.rfft2(sync_tile(key, profile)) * w
    # remove the sync component so it cannot leak into the id correlation
    coef = (cell_f * np.conj(sync_f)).real.sum() / max((np.abs(sync_f) ** 2).sum(), 1e-9)
    cell_f = cell_f - coef * sync_f
    dy, dx = shift
    offsets = [((dy + a) % TILE, (dx + b) % TILE) for a in (-1, 0, 1) for b in (-1, 0, 1)]

    def stat(labels: list[str]) -> np.ndarray:
        out = np.empty(len(labels), np.float64)
        for start in range(0, len(labels), 500):
            chunk = labels[start : start + 500]
            tiles = np.stack([id_tile(key, lab, profile) for lab in chunk])
            tf = np.fft.rfft2(tiles) * w
            surf = np.fft.irfft2(cell_f[None] * np.conj(tf), s=(TILE, TILE))
            out[start : start + len(chunk)] = np.max([surf[:, y, x] for y, x in offsets], axis=0)
        return out

    null = stat([f"\x00decoy{i}" for i in range(decoys)])
    mu, sd = float(null.mean()), float(null.std())
    z = (stat(list(candidates)) - mu) / sd
    order = np.argsort(z)[::-1]
    return [Match(candidates[i], float(z[i])) for i in order[:10]]


def _affine(theta: float, sx: float, sy: float) -> np.ndarray:
    """Leaked pixels to canonical pixels: undo a rotation by theta degrees, then a scale of (sx, sy)."""
    c, n = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    return np.array([[c / sx, n / sx], [-n / sy, c / sy]], dtype=np.float64)


def _fold_affine(x: np.ndarray, xx: np.ndarray, yy: np.ndarray, m: np.ndarray) -> np.ndarray:
    u = np.floor(m[0, 0] * xx + m[0, 1] * yy).astype(np.int64) % TILE
    v = np.floor(m[1, 0] * xx + m[1, 1] * yy).astype(np.int64) % TILE
    index = v * TILE + u
    total = np.bincount(index, x, TILE * TILE)
    count = np.bincount(index, minlength=TILE * TILE)
    cell = (total / np.maximum(count, 1)).reshape(TILE, TILE).astype(np.float32)
    return cell - cell.mean()


def geometry_grid() -> list[tuple[float, float, float]]:
    """(theta, sx, sy) to try: small tilts at about the same size, and one axis stretched with the other left alone."""
    scales = np.exp(np.arange(-24, 18) * 0.015)        # includes exactly 1: most tilted copies were not also resized
    tilts = [t for t in np.arange(-15.0, 15.01, 0.5) if abs(t) >= 0.25]
    stretch = np.exp(np.arange(np.log(0.45), np.log(2.2) + 0.01, 0.01))
    grid = [(float(t), float(k), float(k)) for t in tilts for k in scales]
    grid += [(0.0, float(k), 1.0) for k in stretch] + [(0.0, 1.0, float(k)) for k in stretch]
    return grid


def search_geometry(frames: list[np.ndarray], search: np.ndarray, topk: int = 4) -> tuple[float, tuple[float, float, float]]:
    """Find a rotation or a change of aspect ratio by folding at each candidate geometry. Returns (best score, (theta, sx, sy)).

    The pattern is known, so this tries it directly; the picture's own autocorrelation does not show the period at this strength.
    """
    weights = build_weights(frames)
    stack = build_stack(frames, weights, SIGMAS[1])
    x = stack.num / np.maximum(stack.den, 1e-6)
    h, w = x.shape
    sync_f = np.fft.rfft2(search)
    side = 640
    y0, x0 = max(0, (h - side) // 2), max(0, (w - side) // 2)
    yy, xx = np.mgrid[y0:min(h, y0 + side):2, x0:min(w, x0 + side):2]
    coarse_x, coarse_yy, coarse_xx = x[yy, xx].ravel().astype(np.float64), yy.ravel().astype(np.float64), xx.ravel().astype(np.float64)
    upright = (0, 1)                                    # as is, or mirrored; the final check with detect_frames tries all eight
    scored = []
    for geo in geometry_grid():
        scored.append((sync_peak(_fold_affine(coarse_x, coarse_xx, coarse_yy, _affine(*geo)), sync_f, upright)[0], geo))
    scored.sort(key=lambda row: -row[0])

    fy, fx = np.mgrid[0:h, 0:w]
    full_x, full_yy, full_xx = x.ravel().astype(np.float64), fy.ravel().astype(np.float64), fx.ravel().astype(np.float64)

    def score(geo):
        return sync_peak(_fold_affine(full_x, full_xx, full_yy, _affine(*geo)), sync_f, upright)[0]

    best = (-1.0, (0.0, 1.0, 1.0))
    for _, geo in scored[:topk]:
        now, geo = score(geo), list(geo)
        for d_theta, d_scale in ((0.25, 0.006), (0.1, 0.003), (0.04, 0.0012)):
            for i, step in ((0, d_theta), (1, d_scale), (2, d_scale)):
                for sign in (1, -1):
                    while True:
                        trial = list(geo)
                        trial[i] += sign * step
                        z = score(trial)
                        if z <= now:
                            break
                        now, geo = z, trial
        if now > best[0]:
            best = (now, tuple(geo))
    return best


def rectify(frames: list[np.ndarray], geo: tuple[float, float, float]) -> list[np.ndarray]:
    """The frames with that geometry undone, on a canvas big enough to hold them."""
    m = _affine(*geo)
    h, w = frames[0].shape
    corners = m @ np.array([[0, w, 0, w], [0, 0, h, h]], dtype=np.float64)
    lo, hi = corners.min(axis=1), corners.max(axis=1)
    size = (int(np.ceil(hi[0] - lo[0])), int(np.ceil(hi[1] - lo[1])))
    # whole tiles of offset, so the pattern stays in phase with the fold the search scored
    off = np.ceil(-lo / TILE) * TILE
    warp = np.hstack([m, off[:, None]]).astype(np.float64)
    size = (int(size[0] + TILE), int(size[1] + TILE))
    return [cv2.warpAffine(f, warp, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0) for f in frames]


def load_frames(path: Path | str, frames: int = 32) -> list[np.ndarray]:
    if media.is_image(path):
        return [media.read_image_gray(path)]
    return media.sample_frames(path, media.probe(path), frames)


def detect(
    path: Path | str,
    key: bytes,
    candidates: list[str],
    profile: Profile = DEFAULT,
    frames: int = 32,
    scale_hint: float | None = None,
    sync_min_z: float = SYNC_GATE,
    search: np.ndarray | None = None,
) -> Report:
    return detect_frames(load_frames(path, frames), key, candidates, profile, scale_hint, sync_min_z, search)
