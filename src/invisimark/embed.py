"""Embedding: add the recipient's pattern to the luma plane, scaled by a perceptual mask."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import media
from .tile import DEFAULT, Profile, pattern, tile_to_plane


def activity_mask(y: np.ndarray, profile: Profile = DEFAULT) -> np.ndarray:
    """Per-pixel strength in [floor, ceil]: busy regions hide the mark, flat ones show it."""
    smooth = cv2.GaussianBlur(y, (0, 0), 1.0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    activity = cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), 6.0)
    level = np.clip(activity / profile.mask_ref, 0.0, 1.0)
    mask = profile.mask_floor + (profile.mask_ceil - profile.mask_floor) * level
    return mask.astype(np.float32)


def pattern_rms(lumas: list[np.ndarray], plane: np.ndarray, profile: Profile) -> float:
    """RMS of the pattern (alphas as given, mask applied) over some sample frames."""
    return float(np.sqrt(np.mean([np.mean((plane * activity_mask(y, profile)) ** 2) for y in lumas])))


def embed_plane(
    y: np.ndarray, plane: np.ndarray, profile: Profile, lo: float, hi: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Add the masked pattern. With an rng the rounding is dithered, so a pattern smaller than one code value
    survives in expectation instead of being rounded away."""
    out = y + plane * activity_mask(y, profile)
    if rng is not None:
        out = out + rng.random(out.shape, dtype=np.float32) - 0.5
    return np.clip(np.rint(out), lo, hi)


def embed_image(
    src: Path | str, dst: Path | str, key: bytes, recipient: str, profile: Profile = DEFAULT, rms: float = 1.2,
    tile: np.ndarray | None = None,
) -> None:
    """Mark a still. `tile` replaces the tile derived from (key, recipient). An alpha channel is carried through."""
    raw = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise media.MediaError(f"{src}: not a readable image")
    alpha = raw[..., 3] if raw.ndim == 3 and raw.shape[2] == 4 and raw.dtype == np.uint8 else None
    img = raw[..., :3] if alpha is not None else cv2.imread(str(src), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    rgb = img.astype(np.float32)
    luma = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    plane = tile_to_plane(pattern(key, recipient, profile) if tile is None else tile, h, w)
    k = rms / max(pattern_rms([luma], plane, profile), 1e-6)
    delta = plane * k * activity_mask(luma, profile)
    dither = np.random.default_rng(7).random(rgb.shape, dtype=np.float32) - 0.5
    out = np.clip(np.rint(rgb + delta[..., None] + dither), 0, 255).astype(np.uint8)
    if alpha is not None:
        out = np.dstack([out, alpha])
    # JPEG is the usual leak format; keep the quality high so the mark is not our own casualty.
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if Path(dst).suffix.lower() in {".jpg", ".jpeg"} else []
    if not cv2.imwrite(str(dst), out, params):
        raise media.MediaError(f"{dst}: could not write image")


def embed_video(
    src: Path | str,
    dst: Path | str,
    key: bytes,
    recipient: str,
    profile: Profile = DEFAULT,
    rms: float = 1.2,
    crf: int = 18,
    preset: str = "medium",
    keyframe_every: float | None = None,
    window: tuple[float, float] | None = None,
    tile: np.ndarray | None = None,
) -> None:
    """Mark a video. `keyframe_every` forces a keyframe at every multiple of that many seconds, so
    separately marked renders can be cut and rejoined at those points without re-encoding.
    `window=(start, seconds)` marks only that stretch, as a silent clip that starts on a keyframe.
    `tile` replaces the tile derived from (key, recipient)."""
    info = media.probe(src)
    w, h = info.width - info.width % 2, info.height - info.height % 2
    plane = tile_to_plane(pattern(key, recipient, profile) if tile is None else tile, h, w)
    frame_bytes = w * h * 3 // 2
    lo, hi = (0.0, 255.0) if info.color_range == "pc" else (16.0, 235.0)
    if window:
        t0, span = window
        grabbed = [media.grab_gray(src, t0 + span * (i + 0.5) / 8, info.width, info.height) for i in range(8)]
        samples = [f[:h, :w] for f in grabbed if f is not None] or [f[:h, :w] for f in media.sample_frames(src, info, 8)]
    else:
        samples = [f[:h, :w] for f in media.sample_frames(src, info, 8)]
    plane = plane * (rms / max(pattern_rms(samples, plane, profile), 1e-6))
    rng = np.random.default_rng(11)

    decoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", *(["-ss", f"{window[0]:.6f}", "-t", f"{window[1]:.6f}"] if window else []),
         "-i", str(src), "-vf", f"crop={w}:{h}:0:0", "-f", "rawvideo",
         "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    color_args: list[str] = []
    if info.color_range in {"tv", "pc"}:
        color_args += ["-color_range", info.color_range]
    for flag, value in (("-colorspace", info.color_space), ("-color_primaries", info.color_primaries),
                        ("-color_trc", info.color_transfer)):
        if value and value != "unknown":
            color_args += [flag, value]
    key_args = ["-force_key_frames", f"expr:gte(t,n_forced*{keyframe_every})"] if keyframe_every else []
    encoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}",
         "-r", f"{info.fps:.5f}", "-i", "-", *([] if window else ["-i", str(src)]), "-map", "0:v", *([] if window else ["-map", "1:a?"]),
         "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p", *key_args, *color_args,
         *([] if window else ["-c:a", "copy", "-shortest"]), "-movflags", "+faststart", str(dst)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert decoder.stdout is not None and encoder.stdin is not None
    broken = False
    try:
        while True:
            raw = decoder.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            buf = np.frombuffer(raw, np.uint8).copy()
            y = buf[: w * h].reshape(h, w).astype(np.float32)
            buf[: w * h] = embed_plane(y, plane, profile, lo, hi, rng).astype(np.uint8).ravel()
            encoder.stdin.write(buf.tobytes())
    except BrokenPipeError:
        broken = True  # the encoder died; its stderr says why
    finally:
        try:
            encoder.stdin.close()
        except BrokenPipeError:
            broken = True
        decoder.stdout.close()
        enc_err = encoder.stderr.read().decode("utf-8", "replace") if encoder.stderr else ""
        dec_err = decoder.stderr.read().decode("utf-8", "replace") if decoder.stderr else ""
        decoder.wait()
        encoder.wait()
    if broken or decoder.returncode or encoder.returncode:
        raise media.MediaError(f"{src}: ffmpeg failed: {(enc_err or dec_err).strip()[-500:]}")
