"""Image robustness benchmark: mark a still, mangle it the way chat apps and screenshots do, and trace it.

    uv run python bench/images.py IMAGE [IMAGE ...] [--strength 1.0] [--out results.md]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from invisimark import calibrate, detect, embed

cv2.setNumThreads(1)

KEY = b"bench-key"
RECIPIENT = "alice"
CANDIDATES = [f"user{i}" for i in range(500)] + [RECIPIENT]


def jpeg(img: np.ndarray, q: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def resize(img: np.ndarray, f: float) -> np.ndarray:
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, None, fx=f, fy=f, interpolation=interp)


def crop(img: np.ndarray, frac: float, ox: int = 37, oy: int = 21) -> np.ndarray:
    h, w = img.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    return img[oy : oy + ch, ox : ox + cw]


def grain(img: np.ndarray, sigma: float) -> np.ndarray:
    rng = np.random.default_rng(3)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


ATTACKS = {
    "none": lambda i: i,
    "jpeg90": lambda i: jpeg(i, 90),
    "jpeg70": lambda i: jpeg(i, 70),
    "jpeg50": lambda i: jpeg(i, 50),
    "jpeg30": lambda i: jpeg(i, 30),
    "half": lambda i: resize(i, 0.5),
    "half_jpeg60": lambda i: jpeg(resize(i, 0.5), 60),
    "chat_app": lambda i: jpeg(resize(i, 0.75), 65),
    "up1.5": lambda i: jpeg(resize(i, 1.5), 80),
    "crop80": lambda i: jpeg(crop(i, 0.8), 80),
    "crop60": lambda i: jpeg(crop(i, 0.6), 80),
    "screenshot": lambda i: cv2.GaussianBlur(resize(i, 0.82), (0, 0), 0.5),
    "contrast": lambda i: cv2.convertScaleAbs(i, alpha=1.2, beta=12),
    "blur1.5": lambda i: jpeg(cv2.GaussianBlur(i, (0, 0), 1.5), 80),
    "grain": lambda i: jpeg(grain(i, 5.0), 80),
    "hflip": lambda i: jpeg(i[:, ::-1].copy(), 80),
    "worst": lambda i: jpeg(cv2.GaussianBlur(crop(resize(i, 0.6), 0.85), (0, 0), 0.8), 50),
}


def run_one(args: tuple[str, str, float]) -> str:
    marked, name, strength = args
    img = cv2.imread(marked, cv2.IMREAD_COLOR)
    started = time.time()
    leaked = np.ascontiguousarray(ATTACKS[name](img))
    gray = cv2.cvtColor(leaked, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rep = detect.detect_frames([gray], KEY, CANDIDATES)
    top = rep.matches[:2]
    named = rep.verdict() == RECIPIENT
    return (
        f"| {name} | {'yes' if rep.marked else 'no'} | {rep.sync_z:.1f} | {rep.scale:.3f} | "
        f"{top[0].recipient if top else '-'} | {top[0].z if top else 0:.1f} | {top[1].z if len(top) > 1 else 0:.1f} | "
        f"{'**YES**' if named else 'no'} | {time.time() - started:.0f}s |"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--strength", type=float, default=0.0, help="RMS of the added pattern; 0 calibrates per image")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=6)
    opts = ap.parse_args()
    names = [n for n in (opts.only.split(",") if opts.only else ATTACKS) if n]
    lines: list[str] = []
    work = Path(opts.out).parent if opts.out else Path(".")
    for image in opts.images:
        src = Path(image)
        marked = work / f"{src.stem}.marked.png"
        rms, note = opts.strength, ""
        if rms <= 0:
            cal = calibrate.calibrate(src, KEY)
            rms, note = cal.rms, f"auto: {cal.summary()}. "
        embed.embed_image(src, marked, KEY, RECIPIENT, rms=rms)
        a, b = cv2.imread(str(src)).astype(np.float32), cv2.imread(str(marked)).astype(np.float32)
        psnr = 10 * np.log10(255.0**2 / max(float(np.mean((a - b) ** 2)), 1e-9))
        h, w = a.shape[:2]
        lines += [f"\n### {src.name} {w}x{h} (RMS {rms:g}, PSNR {psnr:.1f} dB)\n{note}",
                  "| attack | marked | sync z | scale | accused | z | runner-up | named | time |",
                  "|---|---|---|---|---|---|---|---|---|"]
        with ProcessPoolExecutor(opts.workers) as pool:
            for row in pool.map(run_one, [(str(marked), n, opts.strength) for n in names]):
                lines.append(row)
                print(row, flush=True)
    text = "\n".join(lines)
    print(text)
    if opts.out:
        Path(opts.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
