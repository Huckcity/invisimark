"""Attacks by someone trying to defeat the public mark, as Measure 3.3 of the Code of Practice asks. They know the public key.

    uv run python bench/adversarial.py [--workers 4] [--out bench/results/ai-adversarial.md]

Copy: lift the pattern from one marked file and add it to an unmarked one. Subtract: remove the known tile at the alignment the
detector finds. Regenerate: a stand-in for passing a file through a model, since no generative model is run here: heavy denoising
and resampling. A result counts as "still detected" when `invisimark check` would say so, using the same decision.
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

from invisimark import ai, embed
from invisimark.tile import DEFAULT, TILE, tile_to_plane

cv2.setNumThreads(1)
sys.path.insert(0, str(Path(__file__).parent))
from ai import CORPUS, load  # noqa: E402

PROVIDER_KEY = b"adversarial-bench-provider-key-0"


def gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return 10 * np.log10(255.0**2 / max(float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)), 1e-9))


def lift(marked: np.ndarray) -> np.ndarray:
    """What a forger can get from one marked file: the high-passed residual, folded onto one period."""
    g = gray(marked)
    residual = g - cv2.cvtColor(cv2.medianBlur(marked, 5), cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    cell = np.zeros((TILE, TILE), np.float64)
    count = np.zeros((TILE, TILE), np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    np.add.at(cell, (yy % TILE, xx % TILE), residual)
    np.add.at(count, (yy % TILE, xx % TILE), 1)
    return (cell / np.maximum(count, 1)).astype(np.float32)


def copy_onto(target: np.ndarray, cell: np.ndarray, rms: float) -> np.ndarray:
    plane = tile_to_plane(cell, *target.shape[:2])
    plane = plane * rms / max(float(plane.std()), 1e-6)
    return np.clip(np.rint(target.astype(np.float32) + plane[..., None]), 0, 255).astype(np.uint8)


def subtract(marked: np.ndarray, gain: float) -> np.ndarray:
    """Subtract the published tile. Marking starts at the top-left pixel, so an uncropped file needs no alignment. The mask is
    recomputed on the marked picture, which is what an attacker has."""
    g = gray(marked)
    tile = ai.ai_tile()
    plane = tile_to_plane(tile, *g.shape) * embed.activity_mask(g, DEFAULT)
    k = gain * ai.DEFAULT_RMS / max(float(np.sqrt(np.mean(plane**2))), 1e-6)
    return np.clip(np.rint(marked.astype(np.float32) - k * plane[..., None]), 0, 255).astype(np.uint8)


def remove_lifted(marked: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """No key needed: subtract what `lift` recovered, shaped by the mask."""
    g = gray(marked)
    plane = tile_to_plane(cell, *g.shape) * embed.activity_mask(g, DEFAULT)
    plane = plane * ai.DEFAULT_RMS / max(float(np.sqrt(np.mean(plane**2))), 1e-6)
    return np.clip(np.rint(marked.astype(np.float32) - plane[..., None]), 0, 255).astype(np.uint8)


def regenerate(img: np.ndarray, how: str) -> np.ndarray:
    h, w = img.shape[:2]
    if how == "nlmeans":
        return cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
    if how == "bilateral":
        return cv2.bilateralFilter(img, 9, 40, 9)
    if how == "median5":
        return cv2.medianBlur(img, 5)
    if how == "quarter_and_back":
        return cv2.resize(cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_CUBIC)
    if how == "blur3_sharpen":
        soft = cv2.GaussianBlur(img, (0, 0), 3)
        return np.clip(soft.astype(np.float32) * 1.8 - cv2.GaussianBlur(soft, (0, 0), 2).astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
    if how == "noise12_median3":
        noisy = img.astype(np.float32) + np.random.default_rng(3).normal(0, 12, img.shape)
        return cv2.medianBlur(np.clip(noisy, 0, 255).astype(np.uint8), 3)
    raise ValueError(how)


def job(args: tuple[str, str, int]) -> list[tuple[str, str, bool, float, float | None]]:
    name, other, size = args
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        src, out, own = Path(tmp) / "s.png", Path(tmp) / "m.png", Path(tmp) / "p.png"
        img = load(name, size)
        cv2.imwrite(str(src), img)
        ai.mark(src, out)
        ai.mark(src, own, provider_key=PROVIDER_KEY)
        marked, withkey = cv2.imread(str(out)), cv2.imread(str(own))
        target = load(other, size)

        def score(result: np.ndarray, reference: np.ndarray | None, key: bytes | None = None) -> tuple[bool, float, bool | None]:
            path = Path(tmp) / "t.png"
            cv2.imwrite(str(path), result)
            wm = ai.read_watermark(path, key)
            return wm["present"], (psnr(result, reference) if reference is not None else 0.0), wm.get("provider_verified")

        for gain in (0.5, 1.0, 1.5, 2.0):
            ok, p, _ = score(subtract(marked, gain), img)
            rows.append((name, f"subtract x{gain:g}", ok, p, None))
        for how in ("nlmeans", "bilateral", "median5", "quarter_and_back", "blur3_sharpen", "noise12_median3"):
            ok, p, _ = score(regenerate(marked, how), img)
            rows.append((name, f"regenerate {how}", ok, p, None))
        # the file carries a private tile too, so the public part is weaker: the attacker tries several gains and keeps the best
        def public_score(g: float) -> float:
            return ai.find_mark([gray(subtract(withkey, g))]).sync_z

        best = min((0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), key=public_score)
        ok, p, verified = score(subtract(withkey, best), img, PROVIDER_KEY)
        rows.append((name, "subtract public tile from a file with a private tile", ok, p, verified))
        cell = lift(withkey)
        ok, p, verified = score(remove_lifted(withkey, cell), img, PROVIDER_KEY)
        rows.append((name, "lift the whole pattern from one file and subtract it", ok, p, verified))
        ok, p, _ = score(copy_onto(target, lift(marked), ai.DEFAULT_RMS), target)
        rows.append((name, "copy public mark onto another image", ok, p, None))
        ok, p, verified = score(copy_onto(target, lift(withkey), ai.DEFAULT_RMS * 1.3), target, PROVIDER_KEY)
        rows.append((name, "copy mark and private tile onto another image", ok, p, verified))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    opts = ap.parse_args()
    jobs = [(CORPUS[i], CORPUS[(i + 1) % len(CORPUS)], opts.size) for i in range(len(CORPUS))]
    table: dict[str, list] = {}
    with ProcessPoolExecutor(opts.workers) as pool:
        for rows in pool.map(job, jobs):
            for name, attack, ok, p, verified in rows:
                table.setdefault(attack, []).append((name, ok, p, verified))
    lines = [f"Public key. Strength RMS {ai.DEFAULT_RMS:g}. Longest side {opts.size} px. {time.strftime('%Y-%m-%d')}. No generative model was run: "
             "\"regenerate\" rows are denoising and resampling stand-ins.\n",
             "| attack | still detected | mean PSNR against the original or target | private tile still verifies |", "|---|---|---|---|"]
    for attack, got in table.items():
        ver = [g[3] for g in got if g[3] is not None]
        lines.append(f"| {attack} | {sum(g[1] for g in got)} / {len(got)} | {np.mean([g[2] for g in got]):.1f} dB | {f'{sum(ver)} / {len(ver)}' if ver else '-'} |")
    text = "\n".join(lines)
    print(text)
    if opts.out:
        Path(opts.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
