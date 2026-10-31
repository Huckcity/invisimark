"""Closed-loop strength: find the weakest mark that still survives a panel of realistic mangling.

How much mark a picture can carry depends on the picture: texture hides it and lets it through the encoder, flat
animation and dark water do not. So instead of a fixed strength, mark an excerpt at a few strengths, push each
through the panel below, trace it, and keep the lowest strength whose worst case clears the detection thresholds
with room to spare. If even the ceiling does not, say so, because then the copy is only traceable when it is
shared without heavy compression.
"""

from __future__ import annotations

import multiprocessing
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import detect, embed, media
from .detect import SYNC_GATE, id_threshold

LABEL = "\x00calibration"
GRID = (0.7, 0.85, 1.0, 1.2, 1.45, 1.75, 2.1, 2.6, 3.2)
START = 3                      # index of the default strength
SYNC_NEED = SYNC_GATE + 1.0    # room above the gate; the candidate score supplies the rest of the evidence
ID_NEED = id_threshold(100) + 3.0

# (name, ffmpeg output args). The scale is measured from the result, not assumed.
PANEL: list[tuple[str, list[str]]] = [
    ("recompress", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "32"]),
    ("half-size", ["-vf", "scale=trunc(iw/4)*2:trunc(ih/4)*2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30"]),
    ("screen-capture", ["-vf", "scale=trunc(iw*0.37)*2:trunc(ih*0.37)*2,gblur=sigma=0.6",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28"]),
]
IMAGE_PANEL = ("jpeg50", "half+jpeg60", "chat-app")


@dataclass
class Trial:
    attack: str
    sync_z: float
    id_z: float

    @property
    def ok(self) -> bool:
        return self.sync_z >= SYNC_NEED and self.id_z >= ID_NEED


@dataclass
class Calibration:
    rms: float
    passed: bool
    trials: list[Trial] = field(default_factory=list)
    tried: list[float] = field(default_factory=list)

    def summary(self) -> str:
        worst = min(self.trials, key=lambda t: min(t.sync_z / SYNC_NEED, t.id_z / ID_NEED)) if self.trials else None
        head = f"strength {self.rms:g} ({'robust' if self.passed else 'WEAK: content cannot carry a sturdier mark'})"
        if worst:
            head += f"; tightest case {worst.attack}: sync z {worst.sync_z:.1f}, id z {worst.id_z:.1f}"
        return head


# ---------------------------------------------------------------------------------------------
# video


def _activity(frame: np.ndarray) -> float:
    return float(embed.activity_mask(frame).mean())


def pick_excerpt(src: Path, seconds: float = 5.0) -> tuple[float, float]:
    """A typical stretch: the one whose texture sits at the median of twelve samples."""
    info = media.probe(src)
    if info.duration <= seconds * 1.2:
        return 0.0, info.duration
    starts = np.linspace(0, info.duration - seconds, 12)
    scores = []
    for t in starts:
        f = media.grab_gray(src, float(t + seconds / 2), info.width, info.height)
        scores.append(_activity(f) if f is not None else 0.0)
    order = np.argsort(scores)
    return float(starts[order[len(order) // 2]]), seconds


def _one_attack(args: tuple[str, str, list[str], bytes, float, str]) -> Trial:
    marked, name, ff_args, key, src_width, label = args
    with tempfile.TemporaryDirectory() as tmp:
        leaked = Path(tmp) / "leaked.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", marked, *ff_args, "-an", str(leaked)], check=True)
        hint = media.probe(leaked).width / src_width
        rep = detect.detect(leaked, key, [label], scale_hint=hint)
    return Trial(name, rep.sync_z, rep.matches[0].z if rep.matches else 0.0)


def _evaluate_video(excerpt: Path, key: bytes, rms: float, width: int, workdir: Path, label: str = LABEL) -> list[Trial]:
    marked = workdir / f"marked-{rms:g}.mp4"
    embed.embed_video(excerpt, marked, key, label, rms=rms, preset="veryfast")
    jobs = [(str(marked), name, args, key, float(width), label) for name, args in PANEL]
    with ProcessPoolExecutor(len(jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(_one_attack, jobs))


def _search(evaluate) -> Calibration:
    """Lowest passing strength on the grid: start at the default, walk down while it keeps passing, else walk up."""
    results: dict[int, list[Trial]] = {}

    def passes(i: int) -> bool:
        if i not in results:
            results[i] = evaluate(GRID[i])
        return all(t.ok for t in results[i])

    if passes(START):
        best = START
        while best > 0 and passes(best - 1):
            best -= 1
        ok = True
    else:
        best, ok = len(GRID) - 1, False
        for i in range(START + 1, len(GRID)):
            if passes(i):
                best, ok = i, True
                break
        passes(best)
    return Calibration(rms=GRID[best], passed=ok, trials=results[best], tried=[GRID[i] for i in sorted(results)])


def calibrate_video(src: Path | str, key: bytes, max_rms: float = 2.6, seconds: float = 5.0, label: str = LABEL) -> Calibration:
    src = Path(src)
    info = media.probe(src)
    start, length = pick_excerpt(src, seconds)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        excerpt = work / "excerpt.mp4"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{length:.2f}", "-i", str(src),
             "-an", "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(excerpt)],
            check=True,
        )
        cal = _search(lambda rms: _evaluate_video(excerpt, key, rms, info.width, work, label))
    return _cap(cal, max_rms)


def _cap(cal: Calibration, max_rms: float) -> Calibration:
    if cal.rms > max_rms:
        return Calibration(rms=max_rms, passed=False, trials=cal.trials, tried=cal.tried)
    return cal


# ---------------------------------------------------------------------------------------------
# images


def _image_attack(img: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    def jpeg(x, q):
        return cv2.imdecode(cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)

    if name == "jpeg50":
        return jpeg(img, 50), 1.0
    if name == "half+jpeg60":
        return jpeg(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), 60), 0.5
    return jpeg(cv2.resize(img, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA), 65), 0.75


def _evaluate_image(src: Path, key: bytes, rms: float, workdir: Path, label: str = LABEL) -> list[Trial]:
    marked = workdir / f"marked-{rms:g}.png"
    embed.embed_image(src, marked, key, label, rms=rms)
    img = cv2.imread(str(marked), cv2.IMREAD_COLOR)
    trials = []
    for name in IMAGE_PANEL:
        leaked, hint = _image_attack(img, name)
        gray = cv2.cvtColor(leaked, cv2.COLOR_BGR2GRAY).astype(np.float32)
        rep = detect.detect_frames([gray], key, [label], scale_hint=hint)
        trials.append(Trial(name, rep.sync_z, rep.matches[0].z if rep.matches else 0.0))
    return trials


def calibrate_image(src: Path | str, key: bytes, max_rms: float = 2.6, label: str = LABEL) -> Calibration:
    with tempfile.TemporaryDirectory() as tmp:
        cal = _search(lambda rms: _evaluate_image(Path(src), key, rms, Path(tmp), label))
    return _cap(cal, max_rms)


def calibrate(src: Path | str, key: bytes, max_rms: float = 2.6, label: str = LABEL) -> Calibration:
    if media.is_image(src):
        return calibrate_image(src, key, max_rms, label)
    return calibrate_video(src, key, max_rms, label=label)
