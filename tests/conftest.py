import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

KEY = b"unit-test-key-0123456789"


def photo_like(h: int = 720, w: int = 1280, seed: int = 1) -> np.ndarray:
    """A stand-in photograph: 1/f texture, soft gradients and a few hard edges."""
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    spec = (rng.standard_normal((h, w // 2 + 1)) + 1j * rng.standard_normal((h, w // 2 + 1))) / np.maximum(
        np.hypot(fy, fx), 1.0 / max(h, w)
    ) ** 1.0
    base = np.fft.irfft2(spec, s=(h, w))
    base = (base - base.mean()) / base.std()
    yy, xx = np.mgrid[0:h, 0:w]
    plane = 128 + 38 * base + 40 * np.sin(xx / 170.0) * np.cos(yy / 120.0)
    img = np.stack([plane * 0.95 + 8, plane, plane * 1.05 - 6], axis=-1)
    cv2.rectangle(img, (w // 5, h // 4), (w // 2, h // 2), (210, 90, 60), -1)
    cv2.circle(img, (int(w * 0.72), int(h * 0.6)), h // 5, (40, 180, 220), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture(scope="session")
def photo(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("media") / "photo.png"
    cv2.imwrite(str(path), photo_like())
    return path


def pan_clip(path: Path, width: int = 960, height: int = 540, frames: int = 72, fps: int = 24) -> Path:
    """A slow pan across a natural-statistics texture, with a drifting bright blob for foreground motion."""
    canvas = photo_like(height + 40, width + frames * 2 + 40, seed=5)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
         "-i", "-", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE,
    )
    for i in range(frames):
        frame = canvas[20 : 20 + height, 2 * i : 2 * i + width].copy()
        cv2.circle(frame, (width // 4 + 6 * i, height // 2 + int(40 * np.sin(i / 9))), 60, (230, 235, 245), -1)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    return path


@pytest.fixture(scope="session")
def clip(tmp_path_factory) -> Path:
    return pan_clip(tmp_path_factory.mktemp("media") / "clip.mp4")


@pytest.fixture
def fixed_key(monkeypatch):
    """`keygen` normally draws a random key; a few keys are marginally weaker, so tests pin one."""
    from invisimark import keys

    monkeypatch.setattr(keys.secrets, "token_bytes", lambda n: bytes(range(n)))
