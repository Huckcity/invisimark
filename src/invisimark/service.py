"""The hosted, anonymous version: upload a file, get marked copies and a case file; upload a leak and the case file, get a name.

Design rules, all of them about not holding what we do not need:

* No accounts and no database. The secret key for a job is generated for that job, goes into the case file the
  creator downloads, and is forgotten by the server when the job is deleted. Tracing needs the case file back.
* Uploads are deleted as soon as the job has used them. Results are kept for a short time (``ttl``) or until the
  creator deletes the job, then removed. Nothing is written anywhere but the job's own folder.
* The work is CPU-bound and free, so it is capped hard: bytes, megapixel-frames, recipients, queue length, and one
  active job per client.
* Videos are marked as two variants (A/B segments), so the cost depends on the video and not on how many recipients
  there are. Images and very short videos are marked per recipient.

Everything is configured by environment variables (see ``Limits.from_env``) so a host can tune it without code changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from . import ai, calibrate, course, detect, embed, keys, media
from .site import PAGES, Rendered, Site, pack

STATIC = Path(__file__).parent / "static" / "site"
ASSET_TYPES = {"app.js": "text/javascript; charset=utf-8", "check.js": "text/javascript; charset=utf-8", "style.css": "text/css; charset=utf-8", "favicon.svg": "image/svg+xml",
               "og.png": "image/png", "apple-touch-icon.png": "image/png"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
CASE_FORMAT = "invisimark-case"
CASE_VERSION = 1
MAX_LABEL = 80
MIN_SIDE = 160                                    # the mark repeats every 128 px, so a smaller picture cannot carry it
JOB_DIR = re.compile(r"[A-Za-z0-9_-]{22}")        # secrets.token_urlsafe(16)

# What happened to each job, one line per event, for the operator. These lines never carry a client address, a
# file name, a recipient's name or a whole job id (the id is what lets someone download the results): the privacy
# page promises no access log and no address on disk, and that stays true.
log = logging.getLogger("invisimark")


def log_to_stdout() -> None:
    """Send the event lines to stdout, where the host's log collector reads them. Called by `invisimark host`."""
    if not any(getattr(h, "_invisimark", False) for h in log.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._invisimark = True
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def _env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


@dataclass
class Limits:
    max_bytes: int = 300 * 1024 * 1024     # one upload
    max_work: float = 20000.0              # megapixel-frames in one video (1080p30 for about 5 minutes)
    max_image_pixels: float = 50e6
    max_recipients: int = 100
    max_queue: int = 6                     # jobs waiting or running, in total
    ttl: int = 30 * 60                     # seconds a finished job is kept
    render_jobs: int = max(1, (os.cpu_count() or 2) // 2)
    copies_mode_max: int = 8               # per-recipient marking is only used up to this many recipients
    keep_uploads: bool = False             # test mode: keep every upload (and note it on the pages)
    keep_dir: Path | None = None
    notify_url: str = ""                   # where to send "an upload arrived" messages; a secret, never shown to users
    notify_detail: bool = False            # include the uploaded file's name (it goes to whoever runs that service)
    allow_manual_strength: bool = False    # a fixed strength skips the per-key calibration, and a weak key can then fail even untouched

    @classmethod
    def from_env(cls) -> "Limits":
        base = cls()
        keep = os.environ.get("IM_KEEP_UPLOADS", "").strip().lower() in {"1", "true", "yes", "on"}
        data_home = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
        return cls(
            max_bytes=int(_env("IM_MAX_MB", base.max_bytes / 2**20) * 2**20),
            max_work=_env("IM_MAX_WORK", base.max_work),
            max_image_pixels=_env("IM_MAX_IMAGE_PIXELS", base.max_image_pixels),
            max_recipients=int(_env("IM_MAX_RECIPIENTS", base.max_recipients)),
            max_queue=int(_env("IM_MAX_QUEUE", base.max_queue)),
            ttl=int(_env("IM_TTL_SECONDS", 24 * 3600 if keep else base.ttl)),
            render_jobs=int(_env("IM_RENDER_JOBS", base.render_jobs)),
            copies_mode_max=int(_env("IM_COPIES_MODE_MAX", base.copies_mode_max)),
            notify_url=os.environ.get("IM_NOTIFY_URL", "").strip(),
            notify_detail=os.environ.get("IM_NOTIFY_DETAIL", "").strip().lower() in {"1", "true", "yes", "on"},
            allow_manual_strength=os.environ.get("IM_ALLOW_MANUAL_STRENGTH", "").strip().lower() in {"1", "true", "yes", "on"},
            keep_uploads=keep,
            keep_dir=Path(os.environ.get("IM_KEEP_DIR") or data_home / "invisimark" / "kept") if keep else None,
        )

    def public(self) -> dict:
        return {
            "max_mb": int(self.max_bytes / 2**20), "max_recipients": self.max_recipients, "ttl_minutes": round(self.ttl / 60),
            "max_minutes_1080p30": round(self.max_work / (2.0736 * 30) / 60, 1),
            "keep_uploads": self.keep_uploads,
        }

    def ttl_text(self) -> str:
        return f"{round(self.ttl / 3600)} hours" if self.ttl >= 2 * 3600 else f"{round(self.ttl / 60)} minutes"


@dataclass
class Job:
    id: str
    kind: str                                   # mark | trace | check
    ip: str
    status: str = "queued"                      # queued | running | done | error
    stage: str = "waiting for a free worker"
    progress: float = 0.0
    result: dict = field(default_factory=dict)
    error: str = ""
    files: dict[str, Path] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    finished: float = 0.0
    filename: str = ""                          # the name the browser sent; used only in test mode's notes and, if enabled, notifications
    info: dict = field(default_factory=dict)    # media kind, size, recipient count: what a notification may say
    kept: str = ""                              # name of the folder the upload was kept in, in test mode

    def public(self, position: int = 0) -> dict:
        return {"id": self.id, "kind": self.kind, "status": self.status, "stage": self.stage,
                "progress": round(self.progress, 3), "result": self.result, "error": self.error,
                "files": sorted(self.files), "queue_position": position}


# ------------------------------------------------------------------------------------------------ the work


def new_key() -> bytes:
    """A fresh secret for one job. A function so tests can pin it: a few random keys are marginally weaker."""
    return secrets.token_bytes(32)


def clean_label(raw: str) -> str:
    return "".join(ch for ch in raw.strip() if ch.isprintable())[:MAX_LABEL].strip()


def parse_recipients(text: str, limit: int) -> list[str]:
    names = list(dict.fromkeys(n for n in (clean_label(line) for line in text.splitlines()) if n))
    if not names:
        raise ValueError("name at least one recipient (one per line)")
    if len(names) > limit:
        raise ValueError(f"at most {limit} recipients per run")
    return names


def inspect_upload(path: Path, suffix: str, limits: Limits) -> dict:
    """Check what was uploaded, from its contents and not its name. Raises ValueError with a message for the page."""
    try:
        info = media.probe(path)
    except (media.MediaError, KeyError, ValueError, OSError):
        raise ValueError("that file could not be read as an image or a video") from None
    if info.width <= 0 or info.height <= 0:
        raise ValueError("that file could not be read as an image or a video")      # ffprobe reports 0x0 for a fake .png
    if min(info.width, info.height) < MIN_SIDE:
        raise ValueError(f"that picture is too small to carry a mark (at least {MIN_SIDE} pixels each way)")
    pixels = info.width * info.height
    if suffix in IMAGE_SUFFIXES:
        if pixels > limits.max_image_pixels:
            raise ValueError(f"that image is {pixels / 1e6:.0f} megapixels; the limit is {limits.max_image_pixels / 1e6:.0f}")
        return {"kind": "image", "width": info.width, "height": info.height, "duration": 0.0, "fps": 0.0}
    if info.duration <= 0 or info.fps <= 0:
        raise ValueError("that video has no readable duration")
    if info.fps > 61:
        raise ValueError("frame rates above 60 are not supported")
    work = pixels / 1e6 * info.duration * info.fps
    if work > limits.max_work:
        raise ValueError(f"that video is too long or too large for this free service (limit about "
                         f"{limits.public()['max_minutes_1080p30']} minutes of 1080p30). Trim it or scale it down.")
    return {"kind": "video", "width": info.width, "height": info.height, "duration": info.duration, "fps": info.fps}


def choose_mode(meta: dict, n: int, limits: Limits) -> dict:
    """Per-recipient copies for images and tiny jobs, A/B segments otherwise, with a note when the clip is too short
    for segments to name anyone."""
    if meta["kind"] == "image":
        return {"mode": "copies", "note": ""}
    need = math.ceil(math.log2(max(n, 2) / course.NAME_ODDS))               # clear segments needed to name one of n
    seg = min(5.0, max(1.0, meta["duration"] / (need + 6)))
    n_seg = math.ceil(meta["duration"] * meta["fps"] / max(1, round(seg * meta["fps"])))
    if n <= 2 or (n_seg < need and n <= limits.copies_mode_max):
        return {"mode": "copies", "note": ""}
    note = ""
    if n_seg < need:
        note = (f"This clip is short for {n} recipients: it can be split into {n_seg} segments and naming someone "
                f"needs about {need}, so a leak is likely to give a lead, not a name. Use fewer recipients or a longer video.")
    return {"mode": "segments", "segment_seconds": round(seg, 3), "segments": n_seg, "needed": need, "note": note}


def describe_copies(report: detect.Report, candidates: int) -> dict:
    verdict = report.verdict()
    lead = report.lead()
    if not report.marked:
        text = "No mark from this case was found in that file. It may not come from these copies, or it was changed too much."
    elif verdict:
        text = f"Traced to {verdict}."
    elif lead:
        text = f"A weak lead: {lead}. That is a reason to look, not enough to name anyone."
    else:
        text = "The mark is there, but nobody on the list stands out. Do not accuse anyone on this evidence."
    return {"mode": "copies", "marked": report.marked, "verdict": [verdict] if verdict else [], "lead": lead,
            "summary": text, "candidates": candidates,
            "matches": [{"recipient": m.recipient, "z": round(m.z, 1)} for m in report.matches[:6]]}


def describe_segments(result: course.Result, students: int) -> dict:
    if result.reading is None:
        return {"mode": "segments", "marked": False, "verdict": [], "lead": None, "candidates": students, "matches": [],
                "summary": "No mark from this case was found in that file. It may not come from these copies, or it was changed too much."}
    top = result.scores[0]
    if len(result.verdict) == 1:
        text = f"Traced to {result.verdict[0]}."
    elif result.verdict:
        text = "Traced to several recipients: " + ", ".join(result.verdict) + ". Did they share or swap parts of the video?"
    elif result.lead:
        text = (f"A weak lead: {result.lead}. The odds of an unrelated recipient matching this well anywhere on the list are "
                f"1 in {1 / max(top.p_roster, 1e-300):,.0f}, short of the 1 in {1 / course.NAME_ODDS:,.0f} needed to name anyone. "
                "A longer leaked section would settle it.")
    else:
        text = "The mark is there, but nobody on the list can be named on this evidence. Do not accuse anyone."
    return {"mode": "segments", "marked": True, "verdict": result.verdict, "lead": result.lead, "summary": text,
            "candidates": students, "segments_read": result.readable, "segments_total": len(result.reading.soft),
            "matches": [{"recipient": s.student, "agree": s.agree, "clear": s.agree + s.disagree,
                         "odds": round(1 / max(s.p, 1e-300), 1)} for s in result.scores[:6]]}


README_TEXT = """These copies are individually marked with an invisible watermark. Each recipient's copy looks the same as the
others but carries a mark only the case file can read.

KEEP invisimark-case.json PRIVATE AND SAFE.
  * Without it, a leaked copy cannot be traced. The service has already forgotten it.
  * Anyone who has it can make accusations look convincing. Do not share it or upload it anywhere but the Trace page.

To find out who leaked a copy: open the Trace page, upload the leaked file and this case file.

A result is evidence, not proof. Check it against how and when you sent each copy before acting on it. Tell recipients
their copy is individually marked: that is most of the deterrent.
"""


class Notifier:
    """Tell the operator an upload arrived. Best effort: it runs in a thread, times out quickly, never raises, and is
    rate limited so a flood of uploads cannot turn into a flood of messages."""

    def __init__(self, url: str, per_hour: int = 20):
        self.url, self.per_hour = url, per_hour
        self.sent: deque[float] = deque()
        self.suppressed = 0
        self.lock = threading.Lock()

    def _request(self, text: str) -> urllib.request.Request:
        host = self.url.split("/")[2] if self.url.count("/") >= 2 else ""
        if "discord.com" in host or "discordapp.com" in host:
            return urllib.request.Request(self.url, json.dumps({"content": text[:1900]}).encode(),
                                          {"Content-Type": "application/json", "User-Agent": "invisimark"})
        if "slack.com" in host:
            return urllib.request.Request(self.url, json.dumps({"text": text}).encode(),
                                          {"Content-Type": "application/json", "User-Agent": "invisimark"})
        # anything else (ntfy and friends): the message is the body; Title is honoured by ntfy and ignored elsewhere
        return urllib.request.Request(self.url, text.encode("utf-8"),
                                      {"Title": "Invisimark", "Tags": "inbox_tray", "User-Agent": "invisimark"})

    def send(self, text: str) -> bool:
        """Returns True when a message was handed off (the delivery itself happens in a thread)."""
        now = time.time()
        with self.lock:
            while self.sent and now - self.sent[0] > 3600:
                self.sent.popleft()
            if len(self.sent) >= self.per_hour:
                self.suppressed += 1
                return False
            self.sent.append(now)
            if self.suppressed:
                text += f" (+{self.suppressed} more upload{'s' if self.suppressed != 1 else ''} not announced this hour)"
                self.suppressed = 0
        threading.Thread(target=self._deliver, args=(text,), daemon=True).start()
        return True

    def _deliver(self, text: str) -> None:
        try:
            with urllib.request.urlopen(self._request(text), timeout=8):
                pass
        except (urllib.error.URLError, OSError, ValueError):
            pass


def describe_upload(job: "Job", status: str, reason: str, detail: bool) -> str:
    bits = [f"{job.kind} {status}"]
    media_kind, size = job.info.get("media"), job.info.get("bytes")
    if media_kind or size:
        bits.append(" ".join(x for x in (media_kind, f"{size / 2**20:.1f} MB" if size else "") if x))
    if job.info.get("recipients"):
        bits.append(f"{job.info['recipients']} recipients")
    if job.result.get("mode"):
        bits.append(str(job.result["mode"]))
    if status == "done" and job.finished:
        bits.append(f"{job.finished - job.created:.0f}s")
    elif reason:
        bits.append(reason[:120])
    if detail and job.filename:
        bits.append(f'"{job.filename[:80]}"')
    if job.kept:
        bits.append(f"kept as {job.kept}")
    return " \u00b7 ".join(bits)


class Service:
    def __init__(self, limits: Limits | None = None, workdir: Path | None = None):
        self.limits = limits or Limits.from_env()
        self.work = Path(workdir or os.environ.get("IM_WORK") or tempfile.mkdtemp(prefix="invisimark-svc-"))
        self.work.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1)    # one heavy job at a time; a job fans out over the cores itself
        self.order: list[str] = []                       # queue order, for "you are number N"
        self.stop = threading.Event()
        self.notifier = Notifier(self.limits.notify_url) if self.limits.notify_url else None
        self.report_key = os.environ.get("IM_REPORT_KEY", "").strip()    # signs detection results; a secret, never shown to users
        anchors = os.environ.get("IM_TRUST_ANCHORS_FILE", "")            # PEM roots, such as the C2PA trust list
        self.trust_anchors = Path(anchors).read_text() if anchors else None

    # -- housekeeping ---------------------------------------------------------------------

    def wipe_everything(self) -> None:
        """Remove job folders left by an earlier run. Only folders named like our job ids are touched."""
        for child in self.work.iterdir():
            if child.is_dir() and JOB_DIR.fullmatch(child.name):
                shutil.rmtree(child, ignore_errors=True)

    def sweep(self) -> None:
        now = time.time()
        with self.lock:
            for job in list(self.jobs.values()):
                done_age = now - job.finished if job.finished else 0
                if (job.finished and done_age > self.limits.ttl) or now - job.created > 3 * 3600:
                    log.info("%s results expired and removed [%s]", job.kind, job.id[:6])
                    self.drop(job.id)

    def drop(self, job_id: str) -> None:
        job = self.jobs.pop(job_id, None)
        shutil.rmtree(self.work / job_id, ignore_errors=True)
        if job_id in self.order:
            self.order.remove(job_id)

    def sweeper(self) -> None:
        while not self.stop.wait(30):
            self.sweep()

    # -- admission ------------------------------------------------------------------------

    def admit(self, kind: str, ip: str) -> Job:
        with self.lock:
            active = [j for j in self.jobs.values() if j.status in {"queued", "running"}]
            if len(active) >= self.limits.max_queue:
                raise HTTPException(503, "The service is busy; please try again in a few minutes.", headers={"Retry-After": "120"})
            if any(j.ip == ip for j in active):
                raise HTTPException(429, "You already have a job running. Wait for it to finish.", headers={"Retry-After": "60"})
            job = Job(id=secrets.token_urlsafe(16), kind=kind, ip=ip)
            self.jobs[job.id] = job
            self.order.append(job.id)
            (self.work / job.id).mkdir(parents=True)
            return job

    def position(self, job: Job) -> int:
        if job.status != "queued":
            return 0
        waiting = [i for i in self.order if self.jobs.get(i) and self.jobs[i].status == "queued"]
        return waiting.index(job.id) + 1 if job.id in waiting else 0

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "No such job. It may have expired and been deleted.")
        return job

    def archive(self, job: Job, status: str, reason: str = "") -> None:
        """Test mode only: copy the upload somewhere that outlives the job, with a note about what happened to it."""
        keep, src = self.limits.keep_dir, self.work / job.id / "in"
        if not self.limits.keep_uploads or keep is None or not src.is_dir() or job.kind == "check":
            return                                                         # a file sent for checking is never kept, even in test mode
        try:
            dest = keep / f"{time.strftime('%Y%m%d-%H%M%S')}-{job.kind}-{job.id[:8]}"
            dest.mkdir(parents=True, exist_ok=True, mode=0o700)
            for item in src.iterdir():
                shutil.copy2(item, dest / item.name)
                os.chmod(dest / item.name, 0o600)
            note = {"job": job.id, "kind": job.kind, "status": status, "reason": reason,
                    "original_filename": job.filename, "kept_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            (dest / "meta.json").write_text(json.dumps(note, indent=2))
            job.kept = dest.name
        except OSError:
            pass                                                           # keeping a copy must never break the job

    def notify(self, job: Job, status: str, reason: str = "") -> None:
        if self.notifier is not None:
            try:
                self.notifier.send(describe_upload(job, status, reason, self.limits.notify_detail))
            except Exception:
                pass                                                       # telling the operator must never break a job

    def record(self, job: Job, status: str, reason: str = "") -> None:
        """One log line for the operator. Never the file name (detail=False), whatever IM_NOTIFY_DETAIL says."""
        text = describe_upload(job, status, reason, False)
        if job.kind == "trace" and status == "done":
            named = len(job.result.get("verdict") or [])
            text += f" \u00b7 {named} named" if named else " \u00b7 nobody named"
        log.log(logging.WARNING if status == "error" else logging.INFO, "%s [%s]", text, job.id[:6])

    def finish(self, job: Job, status: str, error: str = "") -> None:
        job.status, job.error, job.finished = status, error, time.time()
        job.stage = "finished" if status == "done" else "failed"
        if status == "done":
            job.progress = 1.0
        self.archive(job, status, error)
        self.notify(job, status, error)
        self.record(job, status, error)
        shutil.rmtree(self.work / job.id / "in", ignore_errors=True)      # the upload is deleted unless test mode kept a copy
        shutil.rmtree(self.work / job.id / "tmp", ignore_errors=True)
        if status == "error":
            shutil.rmtree(self.work / job.id, ignore_errors=True)         # nothing from a failed run is kept either

    # -- jobs -----------------------------------------------------------------------------

    def run_mark(self, job: Job, src: Path, meta: dict, recipients: list[str], strength: str) -> None:
        job.status = "running"
        try:
            home = self.work / job.id
            key = new_key()
            plan_info = choose_mode(meta, len(recipients), self.limits)
            rms: float
            if strength == "auto":
                job.stage = ("choosing how strong the mark has to be (this step can take a couple of minutes for video)"
                             if meta["kind"] == "video" else "choosing how strong the mark has to be")
                cal = calibrate.calibrate(src, key, max_rms=2.6)
                rms = cal.rms
                job.result["robust"] = cal.passed
                job.result["calibration"] = cal.summary()
            else:
                rms = {"subtle": 0.8, "standard": 1.2, "strong": 2.0}.get(strength) or float(strength)
            job.result.update({"mode": plan_info["mode"], "note": plan_info.get("note", ""), "strength": round(rms, 2)})
            case = {
                "format": CASE_FORMAT, "version": CASE_VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "key": key.hex(), "kind": meta["kind"], "mode": plan_info["mode"], "recipients": recipients,
                "strength": rms, "source": {"sha256": keys.sha256_of(src), "duration": meta["duration"]},
            }
            stems = keys.unique_slugs(recipients)
            out = home / "copies.zip"
            suffix = ".mp4" if meta["kind"] == "video" else (src.suffix.lower() if src.suffix.lower() in {".jpg", ".jpeg"} else ".png")
            tmp = home / "tmp"
            tmp.mkdir(exist_ok=True)
            with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
                if plan_info["mode"] == "segments":
                    plan = course.make_plan(src, "video", plan_info["segment_seconds"])
                    case["plan"] = {**plan.to_row(), "src": ""}
                    job.stage = f"rendering two versions of the video ({plan.n_seg} segments each)"
                    course.render_lesson(plan, key, tmp / "work", rms, 18, self.limits.render_jobs,
                                         progress=lambda d, n: setattr(job, "progress", 0.9 * d / n))
                    for i, who in enumerate(recipients):
                        job.stage = f"building a copy for {who} ({i + 1} of {len(recipients)})"
                        dst = tmp / f"{stems[who]}{suffix}"
                        course.assemble(plan, key, who, tmp / "work", dst)
                        zf.write(dst, f"{stems[who]}/copy{suffix}")
                        dst.unlink()
                        job.progress = 0.9 + 0.1 * (i + 1) / len(recipients)
                else:
                    for i, who in enumerate(recipients):
                        job.stage = f"marking a copy for {who} ({i + 1} of {len(recipients)})"
                        dst = tmp / f"{stems[who]}{suffix}"
                        if meta["kind"] == "image":
                            embed.embed_image(src, dst, key, who, rms=rms)
                        else:
                            embed.embed_video(src, dst, key, who, rms=rms)
                        zf.write(dst, f"{stems[who]}/copy{suffix}")
                        dst.unlink()
                        job.progress = (i + 1) / len(recipients)
                case_text = json.dumps(case, indent=2)
                zf.writestr("invisimark-case.json", case_text)
                zf.writestr("README.txt", README_TEXT)
            (home / "invisimark-case.json").write_text(case_text)
            job.files = {"copies.zip": out, "invisimark-case.json": home / "invisimark-case.json"}
            job.result["recipients"] = recipients
            self.finish(job, "done")
        except Exception as exc:        # shown to the page; the upload is deleted in finish()
            log.warning("mark failed with %s [%s]", type(exc).__name__, job.id[:6])   # the class only: a message can hold a name
            self.finish(job, "error", _friendly(exc))

    def run_trace(self, job: Job, src: Path, case: dict, trim: float) -> None:
        job.status = "running"
        try:
            key = bytes.fromhex(case["key"])
            recipients = list(case["recipients"])
            job.stage = "looking for the mark"
            if case["mode"] == "segments":
                plan = course.Plan.from_row(case["plan"])
                result = course.trace(src, key, [plan], recipients, trim=trim)
                job.result = describe_segments(result, len(recipients))
            else:
                report = detect.detect(src, key, recipients)
                job.result = describe_copies(report, len(recipients))
            self.finish(job, "done")
        except Exception as exc:
            log.warning("trace failed with %s [%s]", type(exc).__name__, job.id[:6])
            self.finish(job, "error", _friendly(exc))


    def run_check(self, job: Job, src: Path) -> None:
        job.status = "running"
        try:
            job.stage = "looking for marks"
            result = ai.check(src, trust_anchors=self.trust_anchors)
            job.result = result.public()
            if self.report_key:
                signed = self.work / job.id / "invisimark-result.json"
                detector = os.environ.get("IM_PUBLIC_URL", "").strip().rstrip("/") or ai.DETECTOR
                signed.write_text(json.dumps(ai.signed_report(result, self.report_key, detector), indent=2, ensure_ascii=False) + "\n")
                job.files = {signed.name: signed}
            self.finish(job, "done")
        except Exception as exc:
            log.warning("check failed with %s [%s]", type(exc).__name__, job.id[:6])
            self.finish(job, "error", _friendly(exc))


def _friendly(exc: Exception) -> str:
    if isinstance(exc, media.MediaError):
        return "That file could not be processed. Is it a normal video or image?"
    return "Something went wrong while processing that file. Nothing was kept."


def load_case(raw: bytes, limits: Limits) -> dict:
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("that does not look like a case file")
    def need(ok: bool) -> None:
        if not ok:
            raise ValueError

    try:
        case = json.loads(raw)
        need(case["format"] == CASE_FORMAT and case["version"] == CASE_VERSION)
        need(case["mode"] in {"copies", "segments"} and len(bytes.fromhex(case["key"])) == 32)
        recipients = case["recipients"]
        need(isinstance(recipients, list) and 1 <= len(recipients) <= limits.max_recipients)
        need(all(isinstance(r, str) and 0 < len(r) <= MAX_LABEL for r in recipients))
        if case["mode"] == "segments":
            course.Plan.from_row(case["plan"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        raise ValueError("that does not look like an Invisimark case file") from None
    return case


# ------------------------------------------------------------------------------------------------ web layer


class BodyLimit:
    """Refuse oversized request bodies before they are stored, counting as the bytes arrive."""

    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        declared = dict(scope["headers"]).get(b"content-length")
        if declared and declared.isdigit() and int(declared) > self.limit:
            return await self._refuse(send)
        seen = 0
        started = False

        async def counted():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.limit:
                    raise _TooBig
            return message

        async def tracking_send(message):
            nonlocal started
            started = True
            await send(message)

        try:
            await self.app(scope, counted, tracking_send)
        except _TooBig:
            if not started:
                await self._refuse(send)

    @staticmethod
    async def _refuse(send):
        body = json.dumps({"detail": "That upload is larger than this service accepts."}).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


class NoIndexApi:
    """Keep crawlers out of the API and job pages, whatever robots.txt says."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/api/"):
            return await self.app(scope, receive, send)

        async def tagged(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).append((b"x-robots-tag", b"noindex"))
            await send(message)

        await self.app(scope, receive, tagged)


class _TooBig(Exception):
    pass


def _ad_block(name: str) -> str:
    """Raw HTML from the operator (an ad network's tag), or nothing. Read from $IM_AD_<NAME> or the file $IM_AD_<NAME>_FILE."""
    path = os.environ.get(f"IM_AD_{name}_FILE")
    if path and Path(path).is_file():
        return Path(path).read_text()
    return os.environ.get(f"IM_AD_{name}", "")


def make_app(limits: Limits | None = None, workdir: Path | None = None) -> Starlette:
    svc = Service(limits, workdir)
    contact = os.environ.get("IM_CONTACT", "")
    ads = {"top": _ad_block("TOP"), "bottom": _ad_block("BOTTOM")}
    csp_ads = bool(ads["top"] or ads["bottom"])

    def headers(extra: dict | None = None) -> dict:
        script = "script-src 'self' https: 'unsafe-inline'" if csp_ads else "script-src 'self'"
        frame = "frame-src https:" if csp_ads else "frame-src 'none'"
        img = "img-src 'self' data: https:" if csp_ads else "img-src 'self' data:"
        base = {
            "content-security-policy": f"default-src 'self'; {script}; {frame}; {img}; style-src 'self' 'unsafe-inline'; "
                                       "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'",
            "x-content-type-options": "nosniff", "referrer-policy": "no-referrer", "x-frame-options": "DENY",
        }
        return {**base, **(extra or {})}

    def client_ip(request: Request) -> str:
        return request.headers.get("fly-client-ip") or (request.client.host if request.client else "unknown")

    keep = svc.limits.keep_uploads
    source = os.environ.get("IM_SOURCE_URL", "").strip()
    public = os.environ.get("IM_PUBLIC_URL", "").strip().rstrip("/")
    public_host = public.split("//", 1)[-1].split("/")[0] if public else ""
    ad_html = {k: (f'<aside class="ad" aria-label="Advertisement">{v}</aside>' if v else "") for k, v in ads.items()}
    site = Site(
        {
            "retention_paragraph": (
                f"The results (your marked copies and the case file) are kept for {svc.limits.ttl_text()} "
                "and are then removed. Do not upload anything you would not want kept.") if keep else
                ("Files you upload are processed on the server and deleted as soon as the job finishes, or if it fails. "
                 f"The results (your marked copies and the case file) are kept for {svc.limits.ttl_text()} so you can download them, "
                 "and are then removed. Nothing is written anywhere else. We do not look at your files."),
            "source_link": f'Source code: <a href="{source}">{source}</a>' if source else "",
            "contact": contact or "the site operator",
            "contact_link": f'<a href="mailto:{contact}">{contact}</a>' if contact else "the site operator",
            "ttl": svc.limits.ttl_text(), "max_mb": str(int(svc.limits.max_bytes / 2**20)),
            "max_recipients": str(svc.limits.max_recipients),
            "max_minutes": str(svc.limits.public()["max_minutes_1080p30"]),
        },
        base=public, ad_top=ad_html["top"], ad_bottom=ad_html["bottom"],
    )
    assets: dict[str, tuple[Rendered, str]] = {}
    for name, media_type in ASSET_TYPES.items():
        path = STATIC / name
        if path.is_file():
            raw = path.read_bytes()
            assets[name] = (pack(raw.decode("utf-8")) if media_type.startswith(("text/", "image/svg")) else
                            Rendered(raw, b"", '"' + hashlib.sha256(raw).hexdigest()[:20] + '"'), media_type)

    def serve(request: Request, rendered: Rendered, media_type: str, cache: str) -> Response:
        extra = {"cache-control": cache, "etag": rendered.etag, "vary": "Accept-Encoding"}
        if public_host and request.headers.get("host", "").split(":")[0] != public_host.split(":")[0]:
            extra["x-robots-tag"] = "noindex"                  # fly.dev and other hostnames must not compete with the real domain
        if rendered.status == 200 and request.headers.get("if-none-match") == rendered.etag:
            return Response(status_code=304, headers=headers(extra))
        body = rendered.body
        if rendered.gz and "gzip" in request.headers.get("accept-encoding", ""):
            body, extra["content-encoding"] = rendered.gz, "gzip"
        return Response(body, status_code=rendered.status, media_type=media_type, headers=headers(extra))

    def page(path: str):
        async def handler(request: Request) -> Response:
            return serve(request, site.pages[path], "text/html; charset=utf-8", "public, max-age=600")
        return handler

    async def asset(request: Request) -> Response:
        found = assets.get(request.path_params["name"])
        if found is None:
            raise HTTPException(404, "not found")
        return serve(request, found[0], found[1], "public, max-age=3600")

    async def robots(request: Request) -> Response:
        return Response(site.robots(str(request.base_url)), media_type="text/plain", headers=headers({"cache-control": "public, max-age=3600"}))

    async def sitemap(request: Request) -> Response:
        return Response(site.sitemap(str(request.base_url)), media_type="application/xml", headers=headers({"cache-control": "public, max-age=3600"}))

    async def healthz(request: Request) -> Response:
        return PlainTextResponse("ok")

    async def limits_(request: Request) -> Response:
        return JSONResponse(svc.limits.public())

    async def read_form(request: Request):
        try:
            return await request.form(max_files=2, max_fields=8, max_part_size=svc.limits.max_bytes)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(400, "That request could not be read.") from None

    def save(upload, job: Job, suffixes: set[str]) -> tuple[Path, str]:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in suffixes:
            raise HTTPException(400, f"Unsupported file type {suffix or '(none)'}.")
        dest = svc.work / job.id / "in" / f"input{suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        return dest, suffix

    def fail(job: Job, status: int, message: str):
        svc.archive(job, "rejected", message)
        svc.notify(job, "rejected", message)
        svc.record(job, "rejected", message)
        svc.drop(job.id)
        raise HTTPException(status, message)

    async def mark(request: Request) -> Response:
        form = await read_form(request)
        upload = form.get("file")
        if upload is None or not hasattr(upload, "file"):
            raise HTTPException(400, "Attach a file.")
        try:
            recipients = parse_recipients(str(form.get("recipients", "")), svc.limits.max_recipients)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        strength = str(form.get("strength", "auto"))
        if strength not in {"auto", "subtle", "standard", "strong"}:
            raise HTTPException(400, "Unknown strength.")
        if strength != "auto" and not svc.limits.allow_manual_strength:
            raise HTTPException(400, "The strength of the mark is chosen automatically for each file.")
        job = svc.admit("mark", client_ip(request))
        job.filename = upload.filename or ""
        try:
            src, suffix = await run_in_threadpool(save, upload, job, VIDEO_SUFFIXES | IMAGE_SUFFIXES)
            job.info["bytes"] = src.stat().st_size
            meta = await run_in_threadpool(inspect_upload, src, suffix, svc.limits)
            job.info.update(media=meta["kind"], recipients=len(recipients))
        except HTTPException as exc:
            fail(job, exc.status_code, exc.detail)
        except ValueError as exc:
            fail(job, 400, str(exc))
        svc.record(job, "accepted")
        svc.pool.submit(svc.run_mark, job, src, meta, recipients, strength)
        return JSONResponse(job.public(svc.position(job)), status_code=202)

    async def trace(request: Request) -> Response:
        form = await read_form(request)
        upload, case_file = form.get("file"), form.get("case")
        if upload is None or not hasattr(upload, "file") or case_file is None:
            raise HTTPException(400, "Attach the leaked file and the case file.")
        raw = (await case_file.read()) if hasattr(case_file, "read") else str(case_file).encode()
        try:
            case = load_case(raw, svc.limits)
            trim = float(str(form.get("trim", "0") or "0"))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        if not 0 <= trim <= 6 * 3600:
            raise HTTPException(400, "Trim must be between 0 and 6 hours.")
        job = svc.admit("trace", client_ip(request))
        job.filename = upload.filename or ""
        try:
            src, suffix = await run_in_threadpool(save, upload, job, VIDEO_SUFFIXES | IMAGE_SUFFIXES)
            job.info["bytes"] = src.stat().st_size
            seen = await run_in_threadpool(inspect_upload, src, suffix, svc.limits)
            job.info.update(media=seen["kind"])
        except HTTPException as exc:
            fail(job, exc.status_code, exc.detail)
        except ValueError as exc:
            fail(job, 400, str(exc))
        svc.record(job, "accepted")
        svc.pool.submit(svc.run_trace, job, src, case, trim)
        return JSONResponse(job.public(svc.position(job)), status_code=202)

    async def check(request: Request) -> Response:
        form = await read_form(request)
        upload = form.get("file")
        if upload is None or not hasattr(upload, "file"):
            raise HTTPException(400, "Attach a file.")
        job = svc.admit("check", client_ip(request))              # no job.filename: nothing about the file's name is kept or sent
        try:
            src, suffix = await run_in_threadpool(save, upload, job, VIDEO_SUFFIXES | IMAGE_SUFFIXES)
            job.info["bytes"] = src.stat().st_size
            seen = await run_in_threadpool(inspect_upload, src, suffix, svc.limits)
            job.info.update(media=seen["kind"])
        except HTTPException as exc:
            fail(job, exc.status_code, exc.detail)
        except ValueError as exc:
            fail(job, 400, str(exc))
        svc.record(job, "accepted")
        svc.pool.submit(svc.run_check, job, src)
        return JSONResponse(job.public(svc.position(job)), status_code=202)

    async def report_key(request: Request) -> Response:
        if not svc.report_key:
            raise HTTPException(404, "This service does not sign its results.")
        return JSONResponse({"alg": "Ed25519", "public_key": ai.public_key_of(svc.report_key)})

    async def job_status(request: Request) -> Response:
        job = svc.get(request.path_params["job"])
        return JSONResponse(job.public(svc.position(job)), headers={"cache-control": "no-store"})

    async def download(request: Request) -> Response:
        job = svc.get(request.path_params["job"])
        path = job.files.get(request.path_params["name"])
        if path is None or not path.exists():
            raise HTTPException(404, "No such file.")
        return FileResponse(path, filename=path.name, headers={"cache-control": "no-store"})

    async def delete(request: Request) -> Response:
        job = svc.get(request.path_params["job"])
        if job.status in {"queued", "running"}:
            raise HTTPException(409, "That job is still running.")
        with svc.lock:
            svc.drop(job.id)
        log.info("%s results deleted on request [%s]", job.kind, job.id[:6])
        message = ("Results deleted. The uploaded file is kept for testing." if svc.limits.keep_uploads
                   else "Deleted. Nothing from this job remains on the server.")
        return JSONResponse({"deleted": True, "message": message})

    async def json_error(request: Request, exc: HTTPException) -> Response:
        if exc.status_code == 404 and not request.url.path.startswith("/api/"):
            return serve(request, site.not_found, "text/html; charset=utf-8", "no-store")
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)

    @asynccontextmanager
    async def lifespan(app):
        svc.wipe_everything()
        threading.Thread(target=svc.sweeper, daemon=True).start()
        log.info("started \u00b7 results last %s \u00b7 uploads up to %d MB \u00b7 %s", svc.limits.ttl_text(), svc.limits.max_bytes / 2**20,
                 "test mode: every upload is kept" if svc.limits.keep_uploads else "uploads are deleted when a job finishes")
        yield
        log.info("stopping")
        svc.stop.set()
        svc.wipe_everything()

    app = Starlette(
        routes=[
            *[Route(p.path, page(p.path)) for p in PAGES],
            Route("/robots.txt", robots),
            Route("/sitemap.xml", sitemap),
            Route("/static/{name}", asset),
            Route("/healthz", healthz),
            Route("/api/limits", limits_),
            Route("/api/mark", mark, methods=["POST"]),
            Route("/api/trace", trace, methods=["POST"]),
            Route("/api/check", check, methods=["POST"]),
            Route("/api/report-key", report_key),
            Route("/api/jobs/{job}", job_status),
            Route("/api/jobs/{job}", delete, methods=["DELETE"]),
            Route("/api/jobs/{job}/files/{name}", download),
        ],
        lifespan=lifespan,
        exception_handlers={HTTPException: json_error},
    )
    app.add_middleware(NoIndexApi)
    app.add_middleware(BodyLimit, limit=svc.limits.max_bytes + 4 * 1024 * 1024)
    app.state.service = svc
    return app


def create_app() -> Starlette:
    """Entry point for `uvicorn invisimark.service:create_app --factory`."""
    return make_app()
