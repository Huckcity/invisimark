"""Course mode: A/B segment watermarking for sellers who hand over video files.

Marking costs about one real-time render per recipient, which is hopeless for 300 students x 40 lessons. Here each
lesson is rendered twice instead: variant A and variant B, cut into short segments that each start on a keyframe.
A student's copy takes every segment from A or from B according to a bit string derived from the secret key, the
student and the lesson, and the segments are joined with ``-c copy``. Making a copy costs a file concatenation.

To trace, the leaked file is aligned once (scale, orientation, shift) with the ordinary detector. Then each segment
window is scored against its lesson's A and B tiles, which reads one soft bit per segment. A student's score is the
correlation between those soft bits and the bits that student was given. A segment that cannot be read counts for
nothing instead of voting wrongly, and two students who swap segments both stand out.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import binom

from . import detect, embed, media
from .tile import DEFAULT

SEP = "\x1f"
VARIANTS = "AB"
SEGMENT_SECONDS = 5.0
WINDOW_DECOYS = 128          # the null for one segment window; the full 1,500 is for whole files
SOFT_SCALE = 4.0             # z difference at which a segment's vote is about three quarters of full weight
CONFIDENT = 2.0              # |z(A) - z(B)| a segment needs to count as a vote
NAME_ODDS = 1e-6             # chance of accusing anyone innocent, across the whole roster, that we accept
LEAD_ODDS = 1e-2             # a hint worth a look, not an accusation
LESSON_Z = 3.5               # which lesson is this? A leak is half A and half B, so each tile scores about half as high;
                             # guessing the wrong lesson only reads noise and cannot name anyone


def variant_label(lesson: str, variant: str) -> str:
    return f"course{SEP}{lesson}{SEP}{variant}"


def student_bits(key: bytes, student: str, lesson: str, count: int) -> list[int]:
    """The A(0)/B(1) choice for each segment: keyed, so no code book has to be stored."""
    out = []
    for i in range(count):
        digest = hmac.new(key, f"course-bit{SEP}{student}{SEP}{lesson}{SEP}{i}".encode(), hashlib.sha256).digest()
        out.append(digest[0] & 1)
    return out


@dataclass
class Plan:
    lesson: str
    src: str
    fps: float
    seg_frames: int
    n_seg: int
    total_frames: int

    @property
    def seg_seconds(self) -> float:
        return self.seg_frames / self.fps

    def window(self, i: int) -> tuple[float, float]:
        """(start, span) of segment i. The span stops half a frame short so the decoder yields exactly the frames
        that belong to this segment and the next one starts on the following frame."""
        first = i * self.seg_frames
        frames = min(self.seg_frames, self.total_frames - first)
        return first / self.fps, frames / self.fps - 0.5 / self.fps

    def to_row(self) -> dict:
        return {"kind": "course-lesson", "lesson": self.lesson, "src": self.src, "fps": self.fps,
                "seg_frames": self.seg_frames, "n_seg": self.n_seg, "total_frames": self.total_frames}

    @classmethod
    def from_row(cls, row: dict) -> "Plan":
        return cls(row["lesson"], row["src"], row["fps"], row["seg_frames"], row["n_seg"], row["total_frames"])


def make_plan(src: Path, lesson: str, segment_seconds: float = SEGMENT_SECONDS) -> Plan:
    info = media.probe(src)
    seg_frames = max(1, round(segment_seconds * info.fps))
    total = max(1, round(info.duration * info.fps))
    return Plan(lesson, str(src), info.fps, seg_frames, math.ceil(total / seg_frames), total)


def segment_path(work: Path, variant: str, i: int) -> Path:
    return work / f"{variant}{i:05d}.mp4"


def _render(job: tuple) -> str:
    src, dst, key_hex, label, rms, crf, window = job
    embed.embed_video(src, dst, bytes.fromhex(key_hex), label, rms=rms, crf=crf, window=window)
    return dst


def render_lesson(plan: Plan, key: bytes, work: Path, rms: float, crf: int = 18, jobs: int = 8, progress=None) -> None:
    """Render every segment in both variants. Existing segment files are kept, so an interrupted run resumes."""
    work.mkdir(parents=True, exist_ok=True)
    todo = []
    for i in range(plan.n_seg):
        for v in VARIANTS:
            dst = segment_path(work, v, i)
            if not dst.exists():
                todo.append((plan.src, str(dst) + ".part.mp4", key.hex(), variant_label(plan.lesson, v), rms, crf, plan.window(i)))
    if not todo:
        return
    done = 0
    with ProcessPoolExecutor(max(1, jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        for part in pool.map(_render, todo):
            Path(part).rename(part.removesuffix(".part.mp4"))
            done += 1
            if progress:
                progress(done, len(todo))


def assemble(plan: Plan, key: bytes, student: str, work: Path, dst: Path) -> list[int]:
    """Join one student's segments without re-encoding and keep the source's own audio. Returns their bits."""
    bits = student_bits(key, student, plan.lesson, plan.n_seg)
    listing = work / f"{hashlib.sha1(student.encode()).hexdigest()[:12]}.concat.txt"
    listing.write_text("".join(f"file '{segment_path(work, VARIANTS[b], i).resolve()}'\n" for i, b in enumerate(bits)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    base = ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-i", plan.src,
            "-map", "0:v", "-map", "1:a?", "-c:v", "copy"]
    tail = ["-movflags", "+faststart", str(dst)]      # no -shortest: it trims the last few video frames to the audio length
    run = subprocess.run([*base, "-c:a", "copy", *tail], capture_output=True)
    if run.returncode:     # audio the target container cannot hold as-is (PCM in .mov, say)
        run = subprocess.run([*base, "-c:a", "aac", "-b:a", "192k", *tail], capture_output=True)
    listing.unlink(missing_ok=True)
    if run.returncode:
        raise media.MediaError(f"{dst}: ffmpeg failed: {run.stderr.decode('utf-8', 'replace').strip()[-400:]}")
    return bits


# ------------------------------------------------------------------------------------------------- tracing


@dataclass
class Reading:
    lesson: str
    sync_z: float
    scale: float
    orientation: int
    soft: list[float | None] = field(default_factory=list)     # z(A) - z(B) per segment; None = unreadable


@dataclass
class Score:
    student: str
    z: float                 # soft correlation, for ranking
    agree: int               # confident segments that match this student's bits
    disagree: int
    p: float                 # chance an unrelated student agrees at least this well by luck
    p_roster: float          # the same, for "somebody on the roster does"


@dataclass
class Result:
    reading: Reading | None
    scores: list[Score]
    readable: int
    verdict: list[str]
    lead: str | None = None


def _grab(args: tuple) -> np.ndarray | None:
    return media.grab_gray(*args)


def read_lesson(leak: Path, key: bytes, plan: Plan, align: detect.Report, trim: float = 0.0, frames: int = 12) -> Reading:
    """`trim` is how many seconds of the lesson are missing from the start of the leak."""
    info = media.probe(leak)
    labels = [variant_label(plan.lesson, v) for v in VARIANTS]
    soft: list[float | None] = []
    seg_s = plan.seg_seconds
    with ThreadPoolExecutor(8) as pool:
        for i in range(plan.n_seg):
            lo = max(i * seg_s - trim, 0.0) + 0.2
            hi = min((i + 1) * seg_s - trim, info.duration) - 0.2
            if hi - lo < 0.5:
                soft.append(None)
                continue
            times = [lo + (hi - lo) * (k + 0.5) / frames for k in range(frames)]
            got = [f for f in pool.map(_grab, [(leak, t, info.width, info.height) for t in times]) if f is not None]
            if len(got) < 3:
                soft.append(None)
                continue
            weights = detect.build_weights(got)
            stack = detect._stack_for({}, got, weights, align.scale)
            cell = detect.orient(detect.cell_of(stack.num, None if stack.uniform else stack.den, align.scale), align.orientation)
            scored = {m.recipient: m.z for m in detect.score_candidates(cell, key, labels, DEFAULT, align.shift, decoys=WINDOW_DECOYS)}
            soft.append(scored[labels[0]] - scored[labels[1]])
    return Reading(plan.lesson, align.sync_z, align.scale, align.orientation, soft)


def score_students(key: bytes, reading: Reading, students: list[str]) -> list[Score]:
    """Rank students by how well their bits match the leak. Each student's bits come from an HMAC, so an unrelated
    student matches each confidently read segment with probability exactly 1/2: the odds below are a binomial tail."""
    soft = np.array([0.0 if s is None else s for s in reading.soft])
    w = np.tanh(soft / SOFT_SCALE)
    norm = math.sqrt(float((w ** 2).sum())) or 1.0
    vote = np.abs(soft) >= CONFIDENT
    out = []
    for student in students:
        sign = 1.0 - 2.0 * np.array(student_bits(key, student, reading.lesson, len(w)))      # A -> +1, B -> -1
        agree = int((vote & ((soft * sign) > 0)).sum())
        disagree = int((vote & ((soft * sign) < 0)).sum())
        n = agree + disagree
        p = float(binom.sf(agree - 1, n, 0.5)) if n else 1.0
        out.append(Score(student, float((w * sign).sum() / norm), agree, disagree, p, min(1.0, p * len(students))))
    out.sort(key=lambda s: (s.p, -s.z))
    return out


def trace(leak: Path, key: bytes, plans: list[Plan], students: list[str], trim: float = 0.0,
          frames: int = 12, align_frames: int = 32) -> Result:
    """Name the students whose A/B pattern the leak carries, or say that nobody can be named."""
    labels = [variant_label(p.lesson, v) for p in plans for v in VARIANTS]
    align = detect.detect(leak, key, labels, frames=align_frames)
    top = align.best
    if not align.marked or top is None or top.z < LESSON_Z:
        return Result(None, [], 0, [])
    lesson = top.recipient.split(SEP)[1]
    plan = next(p for p in plans if p.lesson == lesson)
    reading = read_lesson(leak, key, plan, align, trim=trim, frames=frames)
    scores = score_students(key, reading, students)
    readable = sum(s is not None for s in reading.soft)
    named = [s.student for s in scores if s.p_roster <= NAME_ODDS]
    lead = scores[0].student if not named and scores and scores[0].p_roster <= LEAD_ODDS else None
    return Result(reading, scores, readable, named, lead)
