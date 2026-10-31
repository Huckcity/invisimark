import io
import json
import time
import zipfile
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from invisimark import service

from .conftest import pan_clip


class NeverRuns:
    """A pool that accepts jobs and leaves them queued, so admission rules can be tested without timing games."""

    def submit(self, *args, **kwargs):
        return None


@pytest.fixture(autouse=True)
def pinned_key(monkeypatch):
    """The first job gets a known-good key; later jobs get different ones, so keys really do differ per job."""
    calls = iter(range(1000))
    monkeypatch.setattr(service, "new_key", lambda: bytes((i + next(calls)) % 256 for i in range(32)))


def make(tmp_path, **limits):
    base = dict(max_bytes=100 * 1024, render_jobs=4, allow_manual_strength=True)
    base.update(limits)
    app = service.make_app(service.Limits(**base), tmp_path / "work")
    return app


def wait(client, job_id, timeout=240):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"done", "error"}:
            return job
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def post_mark(client, path: Path, names="alice\nbob", strength="strong", ip="1.1.1.1"):
    return client.post("/api/mark", files={"file": (path.name, path.read_bytes(), "application/octet-stream")},
                       data={"recipients": names, "strength": strength}, headers={"fly-client-ip": ip})


def test_pages_render_and_are_locked_down(tmp_path):
    with TestClient(make(tmp_path)) as client:
        for path in ("/", "/about", "/privacy", "/terms"):
            r = client.get(path)
            assert r.status_code == 200 and "{{" not in r.text
            assert "script-src 'self'" in r.headers["content-security-policy"]
            assert r.headers["x-frame-options"] == "DENY"
        assert client.get("/healthz").text == "ok"
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/../service.py").status_code == 404
        assert client.get("/static/index.html").status_code == 404
        assert client.get("/api/limits").json()["max_recipients"] == 100


def test_ad_code_is_shown_only_when_configured_and_loosens_the_policy_only_then(tmp_path, monkeypatch):
    with TestClient(make(tmp_path)) as client:
        assert "Advertisement" not in client.get("/").text
    monkeypatch.setenv("IM_AD_TOP", '<div id="slot">AD</div>')
    with TestClient(make(tmp_path)) as client:
        r = client.get("/")
        assert '<aside class="ad" aria-label="Advertisement"><div id="slot">AD</div></aside>' in r.text
        assert "https:" in r.headers["content-security-policy"]


def test_image_flow_makes_copies_forgets_the_upload_and_traces_the_leak(photo, tmp_path):
    app = make(tmp_path, max_bytes=20 * 1024 * 1024)
    with TestClient(app) as client:
        r = post_mark(client, photo, names="alice\nbob\n  carol  \nalice")
        assert r.status_code == 202, r.text
        job = wait(client, r.json()["id"])
        assert job["status"] == "done" and job["result"]["mode"] == "copies"
        assert job["result"]["recipients"] == ["alice", "bob", "carol"]
        assert not (tmp_path / "work" / job["id"] / "in").exists()            # the upload is gone

        zipped = client.get(f"/api/jobs/{job['id']}/files/copies.zip")
        z = zipfile.ZipFile(io.BytesIO(zipped.content))
        names = z.namelist()
        assert "invisimark-case.json" in names and "README.txt" in names
        assert sum(n.endswith("/copy.png") for n in names) == 3
        case_text = client.get(f"/api/jobs/{job['id']}/files/invisimark-case.json").content
        assert json.loads(case_text)["recipients"] == ["alice", "bob", "carol"]
        assert z.read("invisimark-case.json") == case_text

        leak = next(n for n in names if n.startswith("bob") and n.endswith(".png"))
        r = client.post("/api/trace", files={"file": ("leak.png", z.read(leak), "image/png"),
                                             "case": ("case.json", case_text, "application/json")},
                        headers={"fly-client-ip": "2.2.2.2"})
        assert r.status_code == 202, r.text
        traced = wait(client, r.json()["id"])
        assert traced["status"] == "done" and traced["result"]["verdict"] == ["bob"], traced

        # a case file from another job cannot name anyone for this leak
        other = wait(client, post_mark(client, photo, ip="3.3.3.3").json()["id"])
        other_case = client.get(f"/api/jobs/{other['id']}/files/invisimark-case.json").content
        r = client.post("/api/trace", files={"file": ("leak.png", z.read(leak), "image/png"),
                                             "case": ("case.json", other_case, "application/json")},
                        headers={"fly-client-ip": "4.4.4.4"})
        assert wait(client, r.json()["id"])["result"]["verdict"] == []

        assert client.delete(f"/api/jobs/{job['id']}").json()["deleted"] is True
        assert client.get(f"/api/jobs/{job['id']}").status_code == 404
        assert not (tmp_path / "work" / job["id"]).exists()


def test_bad_requests_are_refused_and_leave_nothing_behind(photo, tmp_path):
    with TestClient(make(tmp_path, max_recipients=3, max_bytes=20 * 1024 * 1024)) as client:
        files = {"file": ("a.png", photo.read_bytes(), "image/png")}
        assert client.post("/api/mark", files=files, data={"recipients": "", "strength": "standard"}).status_code == 400
        assert client.post("/api/mark", files=files, data={"recipients": "a\nb\nc\nd"}).status_code == 400
        assert client.post("/api/mark", files=files, data={"recipients": "a", "strength": "-1; rm"}).status_code == 400
        r = client.post("/api/mark", files={"file": ("a.exe", b"MZ", "x")}, data={"recipients": "a"})
        assert r.status_code == 400 and "Unsupported" in r.json()["detail"]
        r = client.post("/api/mark", files={"file": ("a.png", b"not an image at all", "x")}, data={"recipients": "a"})
        assert r.status_code == 400 and "could not be read" in r.json()["detail"]
        r = client.post("/api/trace", files={"file": ("a.png", photo.read_bytes(), "x"),
                                             "case": ("c.json", b'{"format":"nope"}', "x")})
        assert r.status_code == 400 and "case file" in r.json()["detail"]
        assert list((tmp_path / "work").iterdir()) == []                          # every refused job removed its folder


def test_oversized_upload_is_cut_off(tmp_path):
    with TestClient(make(tmp_path)) as client:
        big = b"\0" * (5 * 1024 * 1024)
        r = client.post("/api/mark", files={"file": ("a.png", big, "x")}, data={"recipients": "a"})
        assert r.status_code == 413


def test_one_job_per_client_and_a_bounded_queue(photo, tmp_path):
    app = make(tmp_path, max_queue=2, max_bytes=20 * 1024 * 1024)
    app.state.service.pool = NeverRuns()
    with TestClient(app) as client:
        assert post_mark(client, photo, ip="1.1.1.1").status_code == 202
        again = post_mark(client, photo, ip="1.1.1.1")
        assert again.status_code == 429 and again.headers["retry-after"]
        second = post_mark(client, photo, ip="2.2.2.2")
        assert second.status_code == 202 and second.json()["queue_position"] == 2
        busy = post_mark(client, photo, ip="3.3.3.3")
        assert busy.status_code == 503 and busy.headers["retry-after"]
        assert client.delete(f"/api/jobs/{second.json()['id']}").status_code == 409     # cannot delete a job in flight


def test_finished_jobs_expire_and_stale_folders_are_wiped_at_start(photo, tmp_path):
    work = tmp_path / "work"
    stale = work / ("a" * 22)
    stale.mkdir(parents=True)
    (stale / "left-over.bin").write_bytes(b"x")
    (work / "keep-me").mkdir()                                                  # not a job folder: never touched
    app = make(tmp_path, ttl=0, max_bytes=20 * 1024 * 1024)
    with TestClient(app) as client:
        assert not stale.exists() and (work / "keep-me").exists()
        job = wait(client, post_mark(client, photo).json()["id"])
        assert client.get(f"/api/jobs/{job['id']}/files/copies.zip").status_code == 200
        app.state.service.sweep()
        assert client.get(f"/api/jobs/{job['id']}").status_code == 404


def test_mode_choice_is_honest_about_short_videos():
    lim = service.Limits()
    image = {"kind": "image", "duration": 0.0, "fps": 0.0}
    assert service.choose_mode(image, 50, lim)["mode"] == "copies"
    clip = {"kind": "video", "duration": 20.0, "fps": 24.0}
    assert service.choose_mode(clip, 2, lim)["mode"] == "copies"
    assert service.choose_mode(clip, 6, lim)["mode"] == "copies"                  # too short for segments, few enough to copy
    chosen = service.choose_mode(clip, 40, lim)
    assert chosen["mode"] == "segments" and chosen["note"] and chosen["segments"] < chosen["needed"]
    long = {"kind": "video", "duration": 600.0, "fps": 30.0}
    chosen = service.choose_mode(long, 100, lim)
    assert chosen["mode"] == "segments" and chosen["note"] == "" and chosen["segment_seconds"] == 5.0


def test_video_in_segment_mode_end_to_end(tmp_path):
    clip = pan_clip(tmp_path / "talk.mp4", frames=480, fps=24)                  # 20 s
    names = "\n".join(f"fan{i}" for i in range(10))                             # more than copies_mode_max: segments
    app = make(tmp_path, max_bytes=50 * 1024 * 1024, copies_mode_max=8)
    with TestClient(app) as client:
        job = wait(client, post_mark(client, clip, names=names, strength="strong", ip="9.9.9.9").json()["id"])
        assert job["status"] == "done", job
        assert job["result"]["mode"] == "segments" and "short for 10 recipients" in job["result"]["note"]
        z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{job['id']}/files/copies.zip").content))
        case = z.read("invisimark-case.json")
        assert json.loads(case)["plan"]["src"] == ""                             # no server path leaks into the case file
        leak = z.read("fan7/copy.mp4")
        r = client.post("/api/trace", files={"file": ("leak.mp4", leak, "video/mp4"), "case": ("c.json", case, "application/json")},
                        headers={"fly-client-ip": "8.8.8.8"})
        traced = wait(client, r.json()["id"])
        assert traced["status"] == "done", traced
        result = traced["result"]
        assert result["mode"] == "segments" and result["matches"][0]["recipient"] == "fan7"
        assert result["verdict"] in ([], ["fan7"]) and (result["verdict"] or result["lead"] == "fan7")


def test_test_mode_keeps_uploads_and_the_pages_say_so(photo, tmp_path):
    keep = tmp_path / "kept"
    app = make(tmp_path, max_bytes=20 * 1024 * 1024, keep_uploads=True, keep_dir=keep, ttl=86400)
    with TestClient(app) as client:
        home, privacy = client.get("/").text, client.get("/privacy").text
        assert "Test mode" not in home and "deleted when the job finishes" not in home      # the home page claims nothing either way
        assert "test period" not in privacy and "deleted as soon as" not in privacy      # nothing false is said about deletion
        assert "Do not upload anything you would not want kept." in privacy
        assert client.get("/api/limits").json()["keep_uploads"] is True

        job = wait(client, post_mark(client, photo).json()["id"])
        assert job["status"] == "done"
        (kept,) = list(keep.iterdir())
        assert (kept / "input.png").read_bytes() == photo.read_bytes()
        meta = json.loads((kept / "meta.json").read_text())
        assert meta["status"] == "done" and meta["kind"] == "mark" and meta["original_filename"] == photo.name
        assert oct(kept.stat().st_mode & 0o777) == "0o700" and oct((kept / "input.png").stat().st_mode & 0o777) == "0o600"
        assert not (tmp_path / "work" / job["id"] / "in").exists()                  # the job's own copy is still cleaned up

        gone = client.delete(f"/api/jobs/{job['id']}")
        assert "kept for testing" in gone.json()["message"]
        assert (kept / "input.png").exists()                                       # deleting results does not touch the kept upload

        refused = client.post("/api/mark", files={"file": ("fake.png", b"not an image at all", "x")}, data={"recipients": "a"},
                              headers={"fly-client-ip": "5.5.5.5"})
        assert refused.status_code == 400
        notes = [json.loads((d / "meta.json").read_text()) for d in keep.iterdir() if d != kept]
        assert len(notes) == 1 and notes[0]["status"] == "rejected" and "could not be read" in notes[0]["reason"]


def test_default_mode_keeps_nothing_and_says_so(photo, tmp_path):
    assert service.Limits().keep_uploads is False and service.Limits().keep_dir is None
    with TestClient(make(tmp_path, max_bytes=20 * 1024 * 1024)) as client:
        assert "deleted as soon as the job finishes" in client.get("/privacy").text and "test period" not in client.get("/privacy").text
        assert "Test mode" not in client.get("/").text
        assert client.get("/api/limits").json()["keep_uploads"] is False
        job = wait(client, post_mark(client, photo).json()["id"])
        assert client.delete(f"/api/jobs/{job['id']}").json()["message"].startswith("Deleted. Nothing")


def test_keep_mode_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("IM_KEEP_UPLOADS", "1")
    monkeypatch.setenv("IM_KEEP_DIR", str(tmp_path / "somewhere"))
    lim = service.Limits.from_env()
    assert lim.keep_uploads and lim.keep_dir == tmp_path / "somewhere" and lim.ttl == 24 * 3600
    monkeypatch.setenv("IM_TTL_SECONDS", "600")
    assert service.Limits.from_env().ttl == 600                                    # an explicit value still wins


# ----------------------------------------------------------------------------------------------- notifications


def capture_server():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    got = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("content-length", 0)))
            got.append({"path": self.path, "headers": dict(self.headers), "body": body.decode()})
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, got


def until(got, count, timeout=10):
    end = time.time() + timeout
    while time.time() < end and len(got) < count:
        time.sleep(0.05)
    return got


def test_notifier_speaks_ntfy_discord_and_slack():
    server, got = capture_server()
    try:
        plain = service.Notifier(f"http://127.0.0.1:{server.server_port}/secret-topic")
        assert plain.send("hello")
        until(got, 1)
        assert got[0]["path"] == "/secret-topic" and got[0]["body"] == "hello" and got[0]["headers"]["Title"] == "Invisimark"
    finally:
        server.shutdown()
    discord = service.Notifier("https://discord.com/api/webhooks/1/abc")._request("a message")
    assert json.loads(discord.data) == {"content": "a message"} and discord.get_header("Content-type") == "application/json"
    slack = service.Notifier("https://hooks.slack.com/services/T/B/x")._request("a message")
    assert json.loads(slack.data) == {"text": "a message"}


def test_notifier_is_rate_limited_and_says_what_it_held_back():
    server, got = capture_server()
    try:
        n = service.Notifier(f"http://127.0.0.1:{server.server_port}/t", per_hour=2)
        assert n.send("one") and n.send("two")
        assert n.send("three") is False and n.send("four") is False
        n.sent.clear()                                                              # an hour later
        assert n.send("five")
        until(got, 3)
        assert any(g["body"] == "five (+2 more uploads not announced this hour)" for g in got)
    finally:
        server.shutdown()


def test_uploads_are_announced_without_names_unless_asked(photo, tmp_path):
    server, got = capture_server()
    url = f"http://127.0.0.1:{server.server_port}/very-secret-topic"
    try:
        app = make(tmp_path, max_bytes=20 * 1024 * 1024, notify_url=url, keep_uploads=True, keep_dir=tmp_path / "kept")
        with TestClient(app) as client:
            wait(client, post_mark(client, photo, names="alice\nbob").json()["id"])
            until(got, 1)
            message = got[0]["body"]
            assert message.startswith("mark done · image ") and "MB" in message and "2 recipients" in message
            assert "copies" in message and "kept as " in message
            assert photo.name not in message and "alice" not in message                 # no file names, no recipient names
            assert "very-secret-topic" not in client.get("/").text and "very-secret-topic" not in client.get("/api/limits").text

            client.post("/api/mark", files={"file": ("fake.png", b"not an image at all", "x")}, data={"recipients": "a"},
                        headers={"fly-client-ip": "6.6.6.6"})
            until(got, 2)
            assert got[1]["body"].startswith("mark rejected") and "could not be read" in got[1]["body"]
    finally:
        server.shutdown()
    server, got = capture_server()
    try:
        app = make(tmp_path / "b", max_bytes=20 * 1024 * 1024, notify_url=f"http://127.0.0.1:{server.server_port}/t", notify_detail=True)
        with TestClient(app) as client:
            wait(client, post_mark(client, photo).json()["id"])
            until(got, 1)
            assert f'"{photo.name}"' in got[0]["body"]
    finally:
        server.shutdown()


def test_a_dead_notification_endpoint_never_breaks_a_job(photo, tmp_path):
    app = make(tmp_path, max_bytes=20 * 1024 * 1024, notify_url="http://127.0.0.1:9/nobody-home")
    with TestClient(app) as client:
        assert wait(client, post_mark(client, photo).json()["id"])["status"] == "done"


def test_notification_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("IM_NOTIFY_URL", "https://ntfy.sh/abc ")
    monkeypatch.setenv("IM_NOTIFY_DETAIL", "yes")
    lim = service.Limits.from_env()
    assert lim.notify_url == "https://ntfy.sh/abc" and lim.notify_detail is True
    assert "ntfy" not in json.dumps(lim.public())


def test_public_service_always_calibrates_the_strength(photo, tmp_path):
    with TestClient(make(tmp_path, max_bytes=20 * 1024 * 1024, allow_manual_strength=False)) as client:
        assert "Strength of the mark" not in client.get("/").text and 'name="strength"' not in client.get("/").text
        r = post_mark(client, photo, strength="strong")
        assert r.status_code == 400 and "chosen automatically" in r.json()["detail"]
        assert list((tmp_path / "work").iterdir()) == []
    assert service.Limits().allow_manual_strength is False


def test_the_results_offer_no_delete_button_and_the_privacy_page_promises_none(tmp_path):
    for keep in (False, True):
        app = make(tmp_path / str(keep), keep_uploads=keep, keep_dir=tmp_path / "kept")
        with TestClient(app) as client:
            script = client.get("/static/app.js").text
            assert "Delete everything" not in script and "'DELETE'" not in script
            assert "press delete" not in client.get("/privacy").text and "are then removed" in client.get("/privacy").text


def test_job_events_are_logged_without_addresses_names_file_names_or_whole_job_ids(photo, tmp_path, caplog):
    caplog.set_level("INFO", logger="invisimark")
    with TestClient(make(tmp_path, max_bytes=20 * 1024 * 1024)) as client:
        job = wait(client, post_mark(client, photo, names="alice\nbob", ip="9.8.7.6").json()["id"])
        zipped = zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{job['id']}/files/copies.zip").content))
        leak = next(n for n in zipped.namelist() if n.startswith("bob") and n.endswith(".png"))
        traced = client.post("/api/trace", files={"file": ("leaked-holiday.png", zipped.read(leak), "image/png"),
                                                  "case": ("case.json", zipped.read("invisimark-case.json"), "application/json")},
                             headers={"fly-client-ip": "9.8.7.5"})
        assert wait(client, traced.json()["id"])["result"]["verdict"] == ["bob"]
        bad = client.post("/api/mark", files={"file": ("holiday.exe", b"x", "application/octet-stream")},
                          data={"recipients": "alice", "strength": "strong"}, headers={"fly-client-ip": "9.8.7.7"})
        assert bad.status_code == 400
    lines = [r.getMessage() for r in caplog.records if r.name == "invisimark"]
    assert lines[0].startswith("started") and lines[-1] == "stopping"
    assert any(x.startswith("mark accepted") and "image" in x and "2 recipients" in x for x in lines)
    assert any(x.startswith("mark done") and "2 recipients" in x and x.endswith(f"[{job['id'][:6]}]") for x in lines)
    assert any(x.startswith("trace done") and "1 named" in x for x in lines)
    assert any(x.startswith("mark rejected") and "Unsupported file type" in x for x in lines)
    text = "\n".join(lines)
    for private in ("9.8.7.", "alice", "bob", photo.name, "holiday", job["id"]):
        assert private not in text, private


def post_check(client, name="photo.png", data=None):
    return client.post("/api/check", files={"file": (name, data, "application/octet-stream")})


def check_image(tmp_path):
    import cv2

    from invisimark import ai

    from .conftest import photo_like

    src, out = tmp_path / "src.png", tmp_path / "out.png"
    cv2.imwrite(str(src), photo_like(512, 512))
    ai.mark(src, out)
    return out.read_bytes()


def test_check_finds_the_public_mark_and_keeps_no_upload_even_in_test_mode(tmp_path):
    kept = tmp_path / "kept"
    with TestClient(make(tmp_path, max_bytes=20 * 2**20, keep_uploads=True, keep_dir=kept)) as client:
        r = post_check(client, data=check_image(tmp_path))
        assert r.status_code == 202
        job = wait(client, r.json()["id"])
        assert job["status"] == "done" and job["result"]["ai_marked"] and job["result"]["watermark"]["present"]
        assert job["files"] == []
        assert not (tmp_path / "work" / job["id"] / "in").exists()
    assert not kept.exists()


def test_check_signs_its_result_when_given_a_key(tmp_path, monkeypatch):
    from invisimark import ai

    private, public = ai.report_keypair()
    monkeypatch.setenv("IM_REPORT_KEY", private)
    monkeypatch.setenv("IM_PUBLIC_URL", "https://invisimark.example")
    data = check_image(tmp_path)
    with TestClient(make(tmp_path, max_bytes=20 * 2**20)) as client:
        assert client.get("/api/report-key").json() == {"alg": "Ed25519", "public_key": public}
        job = wait(client, post_check(client, data=data).json()["id"])
        report = client.get(f"/api/jobs/{job['id']}/files/invisimark-result.json").json()
    assert ai.verify_report(report, public)
    assert report["result"]["detector"] == "https://invisimark.example"
    assert report["result"]["content_sha256"] == job["result"]["sha256"]


def test_check_without_a_report_key_offers_no_signed_result(tmp_path, monkeypatch):
    monkeypatch.delenv("IM_REPORT_KEY", raising=False)
    with TestClient(make(tmp_path)) as client:
        assert client.get("/api/report-key").status_code == 404


def test_check_refuses_what_it_cannot_read(tmp_path):
    with TestClient(make(tmp_path)) as client:
        assert post_check(client, "notes.txt", b"hello").status_code == 400
        r = post_check(client, "fake.mp4", b"#EXTM3U\nfile:///etc/passwd\n")
        assert r.status_code == 400 and "could not be read" in r.json()["detail"]
