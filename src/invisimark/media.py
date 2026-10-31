"""Thin ffmpeg/ffprobe wrappers: probing, single-frame grabs, and raw frame pipes."""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class MediaError(RuntimeError):
    pass


# Uploads are untrusted. A file named .mp4 can really be an HLS or concat playlist that tells ffmpeg to open other files
# or URLs, so demuxers and protocols are limited to plain local media.
GUARD = ["-protocol_whitelist", "file",
         "-format_whitelist", "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,image2,png_pipe,jpeg_pipe,webp_pipe,bmp_pipe,tiff_pipe"]


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool
    color_space: str | None
    color_range: str | None
    color_primaries: str | None
    color_transfer: str | None


def _run(cmd: list[str]) -> bytes:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise MediaError(f"{cmd[0]} failed: {proc.stderr.decode('utf-8', 'replace')[-600:]}")
    return proc.stdout


def _rate(text: str) -> float:
    num, _, den = text.partition("/")
    return float(num) / float(den or 1) if float(den or 1) else 0.0


def probe(path: Path | str) -> VideoInfo:
    out = _run(["ffprobe", "-v", "error", *GUARD, "-print_format", "json", "-show_streams", "-show_format", str(path)])
    data = json.loads(out)
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise MediaError(f"{path}: no video stream")
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0.0)
    fps = _rate(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1") or 30.0
    return VideoInfo(
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        duration=duration,
        has_audio=any(s["codec_type"] == "audio" for s in data["streams"]),
        color_space=video.get("color_space"),
        color_range=video.get("color_range"),
        color_primaries=video.get("color_primaries"),
        color_transfer=video.get("color_transfer"),
    )


def grab_gray(path: Path | str, t: float, width: int, height: int) -> np.ndarray | None:
    """One luma frame near time t, as float32 (height, width). None when the seek runs past the end."""
    cmd = [
        "ffmpeg", "-v", "error", *GUARD, "-ss", f"{t:.3f}", "-i", str(path),
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True).stdout
    if len(raw) != width * height:
        return None
    return np.frombuffer(raw, np.uint8).reshape(height, width).astype(np.float32)


def sample_frames(path: Path | str, info: VideoInfo, count: int = 16) -> list[np.ndarray]:
    """Evenly spread luma frames, skipping the first and last 5% (fades, title cards)."""
    if info.duration <= 0:
        times = [0.0]
    else:
        lo, hi = info.duration * 0.05, info.duration * 0.95
        times = [lo + (hi - lo) * (i + 0.5) / count for i in range(count)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = list(pool.map(lambda t: grab_gray(path, t, info.width, info.height), times))
    got = [f for f in frames if f is not None]
    if not got:
        raise MediaError(f"{path}: could not decode any frames")
    return got


def read_image_gray(path: Path | str) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise MediaError(f"{path}: not a readable image")
    return img.astype(np.float32)


def is_image(path: Path | str) -> bool:
    return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
