"""A local web app for the same two jobs as the CLI: mark copies for recipients, and trace a leak.

Everything runs on this machine. The server binds to 127.0.0.1, every request must carry the launch token (so
another web page open in your browser cannot drive it), and uploaded files live in a work directory that is
cleared when the server stops.
"""

from __future__ import annotations

import io
import secrets
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from . import calibrate, detect, embed, keys, media
from .cli import STRENGTHS

STATIC = Path(__file__).parent / "static"
MAX_RECIPIENTS = 200
JOB_TTL = 6 * 3600          # finished jobs and their files are dropped after six hours
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"              # queued | running | done | error
    stage: str = "waiting"
    progress: float = 0.0
    result: dict = field(default_factory=dict)
    error: str = ""
    files: dict[str, Path] = field(default_factory=dict)
    created: float = field(default_factory=time.time)

    def public(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "status": self.status, "stage": self.stage,
            "progress": round(self.progress, 3), "result": self.result, "error": self.error,
            "files": sorted(self.files),
        }


class App:
    def __init__(self, key_file: Path, ledger_file: Path, token: str | None = None, workdir: Path | None = None):
        self.key_file, self.ledger_file = key_file, ledger_file
        self.token = token or secrets.token_urlsafe(24)
        self.work = Path(workdir or tempfile.mkdtemp(prefix="invisimark-web-"))
        self.work.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1)   # the work is CPU-bound; one job at a time

    # -- plumbing -------------------------------------------------------------------------

    def _job(self, kind: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        with self.lock:
            stale = [j for j in self.jobs.values() if time.time() - j.created > JOB_TTL and j.status in {"done", "error"}]
            for old in stale:
                del self.jobs[old.id]
                shutil.rmtree(self.work / old.id, ignore_errors=True)
            self.jobs[job.id] = job
        return job

    def _get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job

    def _save_upload(self, upload, job: Job) -> Path:
        name = Path(upload.filename or "upload").name
        suffix = Path(name).suffix.lower()
        if suffix not in VIDEO_SUFFIXES | IMAGE_SUFFIXES:
            raise HTTPException(400, f"unsupported file type {suffix or '(none)'}")
        dest = self.work / job.id / "in" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        return dest

    # -- jobs -----------------------------------------------------------------------------

    def run_issue(self, job: Job, src: Path, recipients: list[str], strength: str, max_strength: float) -> None:
        job.status = "running"
        try:
            key = keys.load_key(self.key_file)
            rms: float | None
            if strength == "auto":
                job.stage = "calibrating against recompression, downscaling and screen capture"
                cal = calibrate.calibrate(src, key, max_rms=max_strength)
                rms = cal.rms
                job.result["calibration"] = {
                    "rms": cal.rms, "robust": cal.passed, "summary": cal.summary(),
                    "trials": [{"attack": t.attack, "sync_z": round(t.sync_z, 1), "id_z": round(t.id_z, 1)} for t in cal.trials],
                }
            else:
                rms = STRENGTHS[strength] if strength in STRENGTHS else float(strength)
            out_dir = self.work / job.id / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            master_hash = keys.sha256_of(src)
            copies = []
            stems = keys.unique_slugs(recipients)
            for i, who in enumerate(recipients):
                job.stage = f"marking a copy for {who} ({i + 1} of {len(recipients)})"
                dst = out_dir / f"{src.stem}.{stems[who]}{src.suffix}"
                if media.is_image(src):
                    embed.embed_image(src, dst, key, who, rms=rms)
                else:
                    embed.embed_video(src, dst, key, who, rms=rms)
                keys.record_issue(
                    self.ledger_file, master=src.name, master_sha256=master_hash, strength=rms, recipient=who,
                    output=dst.name, output_sha256=keys.sha256_of(dst),
                    calibration=job.result.get("calibration", {}).get("summary"),
                )
                job.files[dst.name] = dst
                copies.append({"recipient": who, "file": dst.name})
                job.progress = (i + 1) / len(recipients)
            job.result["copies"] = copies
            job.result["strength"] = rms
            job.status, job.stage = "done", "finished"
        except Exception as exc:  # surfaced to the page, not swallowed
            job.status, job.error = "error", str(exc)

    def run_trace(self, job: Job, src: Path, extra: list[str]) -> None:
        job.status = "running"
        try:
            key = keys.load_key(self.key_file)
            candidates = keys.recipients_in(self.ledger_file)
            candidates += [c for c in extra if c not in candidates]
            job.stage = "searching scale, crop and orientation, then comparing recipients"
            rep = detect.detect(src, key, candidates)
            verdict = rep.verdict()
            job.result = {
                "marked": rep.marked, "sync_z": round(rep.sync_z, 1), "scale": round(rep.scale, 3),
                "orientation": rep.orientation, "mirrored": rep.mirrored, "verdict": verdict, "lead": rep.lead(),
                "candidates": len(candidates), "threshold": round(detect.id_threshold(len(candidates)), 1),
                "matches": [{"recipient": m.recipient, "z": round(m.z, 1)} for m in rep.matches[:6]],
            }
            job.progress = 1.0
            job.status, job.stage = "done", "finished"
        except Exception as exc:
            job.status, job.error = "error", str(exc)


def make_app(key_file: Path, ledger_file: Path, token: str | None = None, workdir: Path | None = None) -> Starlette:
    state = App(key_file, ledger_file, token, workdir)

    def guard(request: Request) -> None:
        host = request.headers.get("host", "").split(":")[0]
        if host not in {"127.0.0.1", "localhost", "testserver"}:
            raise HTTPException(403, "bad host")
        supplied = request.headers.get("x-token") or request.cookies.get("iv_token") or request.query_params.get("t")
        if not supplied or not secrets.compare_digest(supplied, state.token):
            raise HTTPException(403, "missing or wrong token")

    async def index(request: Request) -> Response:
        guard(request)
        page = HTMLResponse((STATIC / "index.html").read_text())
        page.set_cookie("iv_token", state.token, httponly=True, samesite="strict")
        return page

    async def status(request: Request) -> Response:
        guard(request)
        return JSONResponse({
            "key": state.key_file.exists(),
            "issued": len(keys.read_ledger(state.ledger_file)),
            "recipients": keys.recipients_in(state.ledger_file),
        })

    async def make_key(request: Request) -> Response:
        guard(request)
        if state.key_file.exists():
            raise HTTPException(409, "a key already exists")
        keys.create_key(state.key_file)
        return JSONResponse({"key": True})

    async def issue(request: Request) -> Response:
        guard(request)
        if not state.key_file.exists():
            raise HTTPException(409, "create a key first")
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "file"):
            raise HTTPException(400, "attach a file")
        recipients = [r.strip() for r in str(form.get("recipients", "")).splitlines() if r.strip()]
        recipients = list(dict.fromkeys(recipients))
        if not recipients:
            raise HTTPException(400, "name at least one recipient")
        if len(recipients) > MAX_RECIPIENTS:
            raise HTTPException(400, f"at most {MAX_RECIPIENTS} recipients per run")
        strength = str(form.get("strength", "auto"))
        if strength != "auto" and strength not in STRENGTHS:
            try:
                float(strength)
            except ValueError:
                raise HTTPException(400, "unknown strength") from None
        job = state._job("issue")
        src = state._save_upload(upload, job)
        state.pool.submit(state.run_issue, job, src, recipients, strength, 2.6)
        return JSONResponse(job.public(), status_code=202)

    async def trace(request: Request) -> Response:
        guard(request)
        if not state.key_file.exists():
            raise HTTPException(409, "create a key first")
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "file"):
            raise HTTPException(400, "attach a file")
        extra = [r.strip() for r in str(form.get("candidates", "")).splitlines() if r.strip()]
        job = state._job("trace")
        src = state._save_upload(upload, job)
        state.pool.submit(state.run_trace, job, src, extra)
        return JSONResponse(job.public(), status_code=202)

    async def job_status(request: Request) -> Response:
        guard(request)
        return JSONResponse(state._get(request.path_params["job"]).public())

    async def download(request: Request) -> Response:
        guard(request)
        job = state._get(request.path_params["job"])
        name = request.path_params["name"]
        path = job.files.get(name)
        if path is None:
            raise HTTPException(404, "no such file")
        return FileResponse(path, filename=name)

    async def download_zip(request: Request) -> Response:
        guard(request)
        job = state._get(request.path_params["job"])
        if not job.files:
            raise HTTPException(404, "nothing to download yet")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            for name, path in job.files.items():
                zf.write(path, name)
        return Response(buf.getvalue(), media_type="application/zip",
                        headers={"content-disposition": f'attachment; filename="invisimark-{job.id}.zip"'})

    async def ledger(request: Request) -> Response:
        guard(request)
        return JSONResponse(keys.read_ledger(state.ledger_file)[-200:][::-1])

    app = Starlette(routes=[
        Route("/", index),
        Route("/api/status", status),
        Route("/api/key", make_key, methods=["POST"]),
        Route("/api/issue", issue, methods=["POST"]),
        Route("/api/trace", trace, methods=["POST"]),
        Route("/api/jobs/{job}", job_status),
        Route("/api/jobs/{job}/files/{name}", download),
        Route("/api/jobs/{job}/zip", download_zip),
        Route("/api/ledger", ledger),
    ])
    app.state.invisimark = state
    return app


def serve(key_file: Path, ledger_file: Path, port: int = 8765, open_browser: bool = True) -> None:
    import uvicorn

    app = make_app(key_file, ledger_file)
    state: App = app.state.invisimark
    url = f"http://127.0.0.1:{port}/?t={state.token}"
    print(f"invisimark is running at {url}\nEverything stays on this machine. Press Ctrl+C to stop.", flush=True)
    if open_browser:
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    finally:
        shutil.rmtree(state.work, ignore_errors=True)
