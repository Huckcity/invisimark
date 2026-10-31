"""AI-marking benchmark: reliability on untouched files, robustness to everyday changes, and false positives on unmarked files.

    uv run python bench/ai.py robust [--rms 1.75] [--size 1024] [--out bench/results/ai.md]
    uv run python bench/ai.py null   [--out bench/results/ai-null.txt]

Uses the public key and the same decision as `invisimark check`. Corpus: the photographs bundled with scikit-image, which
are natural images, not model output. Attacks follow the categories in Measure 3.3 of the Code of Practice.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from invisimark import ai, detect, embed

cv2.setNumThreads(1)
sys.path.insert(0, str(Path(__file__).parent))
from images import ATTACKS as BASE, crop, jpeg, resize  # noqa: E402

CORPUS = ["astronaut", "coffee", "chelsea", "rocket", "camera", "brick", "grass", "gravel", "cat", "hubble_deep_field", "immunohistochemistry", "retina"]
PROVIDER_KEY = b"bench-provider-key-0123456789abc"
NULL_CORPUS = CORPUS + ["moon", "page", "text", "colorwheel", "coins", "clock", "horse", "cell", "checkerboard", "logo", "motorcycle_left", "lily", "eagle"]


def rotate(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def blur_region(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    y0, y1, x0, x1 = h // 4, h // 2, w // 3, 2 * w // 3
    out[y0:y1, x0:x1] = cv2.GaussianBlur(out[y0:y1, x0:x1], (0, 0), 12)
    return out


def squeeze(img: np.ndarray, fx: float) -> np.ndarray:
    return cv2.resize(img, None, fx=fx, fy=1.0, interpolation=cv2.INTER_AREA)


ATTACKS = {
    **BASE,
    "webp70": lambda i: cv2.imdecode(cv2.imencode(".webp", i, [cv2.IMWRITE_WEBP_QUALITY, 70])[1], cv2.IMREAD_COLOR),
    "local_blur": blur_region,
    "rot90": lambda i: jpeg(cv2.rotate(i, cv2.ROTATE_90_CLOCKWISE), 80),
    "rot1": lambda i: jpeg(rotate(i, 1.0), 80),
    "rot5": lambda i: jpeg(rotate(i, 5.0), 80),
    "rot3": lambda i: jpeg(rotate(i, 3.0), 80),
    "rot10": lambda i: jpeg(rotate(i, 10.0), 80),
    "rot25": lambda i: jpeg(rotate(i, 25.0), 80),
    "rot5_half": lambda i: jpeg(resize(rotate(i, 5.0), 0.5), 80),
    "aspect_4_3": lambda i: jpeg(squeeze(i, 0.75), 80),
    "aspect_16_9": lambda i: jpeg(squeeze(i, 16 / 9), 80),
    "aspect_both_axes": lambda i: jpeg(cv2.resize(i, None, fx=0.75, fy=0.6, interpolation=cv2.INTER_AREA), 80),
    "half_crop": lambda i: jpeg(crop(resize(i, 0.5), 0.8), 75),
}


def load(name: str, size: int) -> np.ndarray | None:
    from skimage import data

    try:
        img = np.asarray(getattr(data, name)())
    except Exception:
        return None                                  # needs a download that failed: reported as missing, not as a pass
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    elif img.dtype != np.uint8:
        lo, hi = float(img.min()), float(img.max())
        img = ((img - lo) / max(hi - lo, 1e-9) * 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    img = img[..., :3][..., ::-1]
    if min(img.shape[:2]) < 16:
        return None
    f = size / max(img.shape[:2])
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)


def _decide(img: np.ndarray, provider_key: bytes | None = None) -> tuple[bool, float, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rep = ai.find_mark([gray], provider_key)
    return rep.sync_z >= ai.PUBLIC_SURE, rep.sync_z, rep.matches[0].z if rep.matches else 0.0


def robust_job(args: tuple[str, str, bytes | None]) -> tuple[str, str, bool, float, float]:
    path, attack, provider_key = args
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    marked, sync, idz = _decide(np.ascontiguousarray(ATTACKS[attack](img)), provider_key)
    return Path(path).stem, attack, marked, sync, idz


def null_job(args: tuple[str, int, str]) -> tuple[str, bool, float, float]:
    name, size, attack = args
    img = load(name, size)
    if img is None:
        return f"{name}@{size}/{attack}", False, -1.0, -1.0
    marked, sync, idz = _decide(np.ascontiguousarray(ATTACKS[attack](img)))
    return f"{name}@{size}/{attack}", marked, sync, idz


def cmd_robust(opts) -> int:
    provider_key = PROVIDER_KEY if opts.provider else None
    what = "Private tile, looked for with its own key, in a file that also has the public mark" if opts.provider else "Public key"
    lines, rows = [f"{what}. Strength RMS {opts.rms:g}. Longest side {opts.size} px. {time.strftime('%Y-%m-%d')}.\n"], {}
    with tempfile.TemporaryDirectory() as tmp, ProcessPoolExecutor(opts.workers) as pool:
        marked, psnr = [], []
        for name in CORPUS:
            img = load(name, opts.size)
            if img is None:
                print(f"missing: {name}", flush=True)
                continue
            src, out = Path(tmp) / f"{name}.src.png", Path(tmp) / f"{name}.png"
            cv2.imwrite(str(src), img)
            embed.embed_image(src, out, ai.AI_KEY, ai.AI_LABEL, rms=opts.rms, tile=ai.ai_tile(provider_key))
            diff = np.mean((img.astype(np.float32) - cv2.imread(str(out)).astype(np.float32)) ** 2)
            psnr.append(10 * np.log10(255.0**2 / max(float(diff), 1e-9)))
            marked.append(str(out))
        names = opts.only.split(",") if opts.only else list(ATTACKS)
        for stem, attack, ok, sync, idz in pool.map(robust_job, [(m, a, provider_key) for m in marked for a in names]):
            rows.setdefault(attack, []).append((stem, ok, sync, idz))
    lines.append(f"{len(marked)} images. PSNR against the unmarked image: mean {np.mean(psnr):.1f} dB, lowest {min(psnr):.1f} dB.\n")
    lines += ["| change | detected | lowest score | missed |", "|---|---|---|---|"]
    total = hits = 0
    for attack in names:
        got = rows[attack]
        ok = sum(r[1] for r in got)
        total, hits = total + len(got), hits + ok
        lines.append(f"| {attack} | {ok} / {len(got)} | {min(r[2] for r in got):.1f} | {', '.join(r[0] for r in got if not r[1]) or '-'} |")
    lines.append(f"\nOverall {hits} / {total}.")
    text = "\n".join(lines)
    print(text)
    if opts.out:
        Path(opts.out).write_text(text + "\n")
    return 0


def cmd_null(opts) -> int:
    attacks = ["none", "jpeg50", "half", "chat_app", "screenshot", "blur1.5", "grain", "contrast"]
    jobs = [(n, s, a) for n in NULL_CORPUS for s in (512, 768, 1024, 1536) for a in attacks]
    rows = []
    with ProcessPoolExecutor(opts.workers) as pool:
        for row in pool.map(null_job, jobs):
            if row[2] >= 0:
                rows.append(row)
    sync, ids = np.array([r[2] for r in rows]), np.array([r[3] for r in rows])
    false = [r[0] for r in rows if r[1]]
    n = len(rows)
    text = (f"{n} unmarked image variants, {len({r[0].split('@')[0] for r in rows})} source images, {time.strftime('%Y-%m-%d')}.\n"
            f"False positives: {len(false)} {false[:10]}\n"
            f"Upper 95% bound on the false-positive rate (rule of three when none were seen): {3 / n:.2%}\n"
            f"alignment score: mean {sync.mean():.2f}, sd {sync.std():.2f}, max {sync.max():.2f} (a mark needs {ai.PUBLIC_SURE})\n"
            f"id score, which does not decide anything: mean {ids.mean():.2f}, sd {ids.std():.2f}, max {ids.max():.2f}\n"
            "Note: variants of one photograph are not independent, so the true sample is smaller than the count suggests.\n")
    print(text)
    if opts.out:
        Path(opts.out).write_text(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["robust", "null"])
    ap.add_argument("--rms", type=float, default=ai.DEFAULT_RMS)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--only", default="")
    ap.add_argument("--provider", action="store_true", help="add a private tile and measure that instead of the public mark")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    opts = ap.parse_args()
    return cmd_robust(opts) if opts.what == "robust" else cmd_null(opts)


if __name__ == "__main__":
    sys.exit(main())
