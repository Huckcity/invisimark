import time

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from invisimark import web


@pytest.fixture
def client(tmp_path, fixed_key):
    app = web.make_app(tmp_path / "key", tmp_path / "ledger.jsonl", token="test-token", workdir=tmp_path / "work")
    with TestClient(app) as c:
        c.headers["x-token"] = "test-token"
        yield c


def wait(client, job: dict, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while job["status"] in {"queued", "running"} and time.time() < deadline:
        time.sleep(0.2)
        job = client.get(f"/api/jobs/{job['id']}").json()
    return job


def test_every_route_needs_the_token(client):
    anon = TestClient(client.app)
    assert anon.get("/").status_code == 403
    assert anon.get("/api/status").status_code == 403
    assert anon.get("/api/ledger").status_code == 403
    assert anon.get("/?t=wrong").status_code == 403
    page = anon.get("/?t=test-token")
    assert page.status_code == 200 and "invisimark" in page.text
    assert "iv_token" in page.headers["set-cookie"]


def test_foreign_host_is_refused(client):
    assert client.get("/api/status", headers={"host": "evil.example"}).status_code == 403


def test_key_is_created_once(client):
    assert client.get("/api/status").json()["key"] is False
    assert client.post("/api/key").status_code == 200
    assert client.get("/api/status").json()["key"] is True
    assert client.post("/api/key").status_code == 409


def test_issue_needs_a_key_a_file_and_names(client, photo):
    assert client.post("/api/issue", data={"recipients": "a"}, files={"file": ("p.png", photo.read_bytes())}).status_code == 409
    client.post("/api/key")
    assert client.post("/api/issue", data={"recipients": "a"}).status_code == 400
    assert client.post("/api/issue", data={"recipients": "  \n"}, files={"file": ("p.png", photo.read_bytes())}).status_code == 400
    assert client.post("/api/issue", data={"recipients": "a"}, files={"file": ("notes.txt", b"hi")}).status_code == 400
    assert client.post("/api/issue", data={"recipients": "a", "strength": "loud"},
                       files={"file": ("p.png", photo.read_bytes())}).status_code == 400


def test_issue_then_trace_an_image(client, photo):
    client.post("/api/key")
    started = client.post(
        "/api/issue", data={"recipients": "alice\nbob\nalice", "strength": "2.2"},
        files={"file": ("photo.png", photo.read_bytes())},
    )
    assert started.status_code == 202
    job = wait(client, started.json())
    assert job["status"] == "done", job
    assert [c["recipient"] for c in job["result"]["copies"]] == ["alice", "bob"]   # duplicates dropped

    copy = client.get(f"/api/jobs/{job['id']}/files/photo.bob.png")
    assert copy.status_code == 200
    img = cv2.imdecode(np.frombuffer(copy.content, np.uint8), cv2.IMREAD_COLOR)
    ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])

    zipped = client.get(f"/api/jobs/{job['id']}/zip")
    assert zipped.status_code == 200 and zipped.headers["content-type"] == "application/zip"
    assert client.get(f"/api/jobs/{job['id']}/files/../../etc/passwd").status_code == 404

    traced = wait(client, client.post("/api/trace", data={}, files={"file": ("leak.jpg", jpeg.tobytes())}).json())
    assert traced["status"] == "done", traced
    assert traced["result"]["verdict"] == "bob"
    assert traced["result"]["matches"][0]["recipient"] == "bob"

    ledger = client.get("/api/ledger").json()
    assert {r["recipient"] for r in ledger} == {"alice", "bob"}


def test_trace_of_an_unmarked_file_says_so(client, photo):
    client.post("/api/key")
    traced = wait(client, client.post("/api/trace", data={}, files={"file": ("x.png", photo.read_bytes())}).json())
    assert traced["status"] == "done"
    assert traced["result"]["marked"] is False and traced["result"]["verdict"] is None


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


def test_finished_jobs_are_dropped_after_their_time(client, monkeypatch):
    state = client.app.state.invisimark
    old = state._job("trace")
    old.status, old.created = "done", time.time() - web.JOB_TTL - 5
    (state.work / old.id).mkdir()
    state._job("trace")
    assert old.id not in state.jobs and not (state.work / old.id).exists()
