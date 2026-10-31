"""False-positive study: how high does the sync statistic get when there is nothing to find?

    uv run python bench/null_study.py VIDEO [VIDEO ...] [--marked MARKED.mp4 ...] [--workers 6]

Unmarked videos are run through a few attacks and traced with a key that never touched them; marked videos
are traced with the wrong key. The largest sync z seen sets how far below it the decision threshold must sit.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from invisimark import detect

cv2.setNumThreads(1)
sys.path.insert(0, str(Path(__file__).parent))
from attacks import ATTACKS  # noqa: E402

WRONG_KEY = b"a-key-that-never-touched-these"
ATTACK_SET = ["none", "crf34", "360p", "up1080p", "zoom80", "screenrec", "blur2", "grain"]
CANDIDATES = [f"user{i}" for i in range(200)]


def video_job(args: tuple[str, str, str]) -> tuple[str, float, bool, float, float]:
    src, attack, work = args
    leaked = Path(work) / f"{Path(src).stem}.{attack}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, *ATTACKS[attack], "-an", str(leaked)], check=True)
    rep = detect.detect(leaked, WRONG_KEY + f'{src}{attack}'.encode(), CANDIDATES, sync_min_z=-1.0)
    return f"{Path(src).stem}/{attack}", rep.sync_z, rep.marked, rep.scale, rep.matches[0].z


def image_job(args: tuple[str, int, int]) -> tuple[str, float, bool, float, float]:
    name, size, key_seed = args
    from skimage import data

    try:
        img = getattr(data, name)()
    except Exception:
        return f"{name}@{size}", 0.0, False, 1.0, 0.0     # needs a download we do not have; skip
    img = np.asarray(img)
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    elif img.dtype != np.uint8:
        lo, hi = float(img.min()), float(img.max())
        img = ((img - lo) / max(hi - lo, 1e-9) * 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    img = img[..., :3]
    if img.ndim != 3 or min(img.shape[:2]) < 16:
        return f"{name}@{size}", 0.0, False, 1.0, 0.0
    img = cv2.resize(img, None, fx=size / max(img.shape[:2]), fy=size / max(img.shape[:2]))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    key = WRONG_KEY + key_seed.to_bytes(2, "little")    # a different key each time: tiles are random, so detectability is too
    rep = detect.detect_frames([gray], key, CANDIDATES, sync_min_z=-1.0)
    return f"{name}@{size}", rep.sync_z, rep.marked, rep.scale, rep.matches[0].z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--marked", nargs="*", default=[], help="marked videos to trace with the wrong key")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--key-repeats", type=int, default=2, help="wrong keys tried per image and size")
    opts = ap.parse_args()
    work = tempfile.mkdtemp(prefix="invisimark-null-")
    jobs = [(v, a, work) for v in list(opts.videos) + list(opts.marked) for a in ATTACK_SET]
    names = ["astronaut", "camera", "coffee", "chelsea", "rocket", "brick", "grass", "gravel", "moon", "page", "text",
             "retina", "immunohistochemistry", "colorwheel", "coins", "clock", "horse", "cell", "checkerboard", "logo",
             "motorcycle_left", "microaneurysms", "lily", "eagle", "hubble_deep_field"]
    image_jobs = [(n, s, i * 7 + j) for i, n in enumerate(names) for j, s in enumerate((480, 640, 960, 1280, 1920))
                  for _ in range(opts.key_repeats)]
    rows = []
    with ProcessPoolExecutor(opts.workers) as pool:
        for row in pool.map(video_job, jobs):
            rows.append(row)
            print(f"{row[0]:34s} sync z {row[1]:5.2f}  best id z {row[4]:5.2f}  scale {row[3]:.3f}", flush=True)
        for row in pool.map(image_job, image_jobs):
            rows.append(row)
            print(f"{row[0]:34s} sync z {row[1]:5.2f}  best id z {row[4]:5.2f}  scale {row[3]:.3f}", flush=True)
    rows = [r for r in rows if r[4] > 0 or r[1] > 0]
    z = np.array([r[1] for r in rows])
    ids = np.array([r[4] for r in rows])
    print(f"\n{len(z)} runs. sync z: max {z.max():.2f}, mean {z.mean():.2f}, std {z.std():.2f}; "
          f"above {detect.SYNC_SURE}: {int((z >= detect.SYNC_SURE).sum())}")
    print(f"best id z of {len(CANDIDATES)} wrong candidates: max {ids.max():.2f}, mean {ids.mean():.2f}, "
          f"std {ids.std():.2f}; above {detect.id_threshold(len(CANDIDATES)):.2f}: {int((ids >= detect.id_threshold(len(CANDIDATES))).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
