import subprocess
from pathlib import Path

import cv2
import numpy as np

from invisimark import cli, detect, embed

from .conftest import KEY, pan_clip, photo_like

CANDIDATES = [f"user{i}" for i in range(100)] + ["alice", "bob"]
RMS = 1.6


def gray(path: Path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32)


def test_image_survives_jpeg_resize_and_crop(photo, tmp_path):
    marked = tmp_path / "marked.png"
    embed.embed_image(photo, marked, KEY, "alice", rms=RMS)
    img = cv2.imread(str(marked))
    leaked = cv2.resize(img, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)[20:560, 30:980]
    ok, buf = cv2.imencode(".jpg", leaked, [cv2.IMWRITE_JPEG_QUALITY, 60])
    leaked = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    report = detect.detect_frames([leaked], KEY, CANDIDATES)
    assert report.marked
    assert report.verdict() == "alice"
    assert abs(report.scale - 0.8) < 0.01


def test_mirrored_image_is_still_traced(photo, tmp_path):
    marked = tmp_path / "marked.png"
    embed.embed_image(photo, marked, KEY, "bob", rms=RMS)
    leaked = np.ascontiguousarray(gray(marked)[:, ::-1])
    report = detect.detect_frames([leaked], KEY, CANDIDATES)
    assert report.verdict() == "bob"
    assert report.mirrored


def test_unmarked_image_accuses_nobody(photo):
    report = detect.detect_frames([gray(photo)], KEY, CANDIDATES)
    assert not report.marked
    assert report.verdict() is None


def test_wrong_key_finds_nothing(photo, tmp_path):
    marked = tmp_path / "marked.png"
    embed.embed_image(photo, marked, KEY, "alice", rms=RMS)
    report = detect.detect_frames([gray(marked)], b"somebody-elses-key-0000", CANDIDATES)
    assert not report.marked
    assert report.verdict() is None


def test_unknown_recipient_is_not_pinned_on_a_candidate(photo, tmp_path):
    marked = tmp_path / "marked.png"
    embed.embed_image(photo, marked, KEY, "carol", rms=3.0)
    report = detect.detect_frames([gray(marked)], KEY, CANDIDATES)
    assert report.marked                       # the mark is plainly there ...
    assert report.verdict() is None            # ... but carol is not on the list, so nobody is named
    assert report.lead() is None


def test_two_colluders_averaging_copies_are_both_named(photo, tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    embed.embed_image(photo, a, KEY, "alice", rms=RMS)
    embed.embed_image(photo, b, KEY, "bob", rms=RMS)
    avg = (gray(a) + gray(b)) / 2
    report = detect.detect_frames([avg], KEY, CANDIDATES)
    assert report.marked
    assert {m.recipient for m in report.matches[:2]} == {"alice", "bob"}


def test_video_survives_reencode_and_rescale(clip, tmp_path):
    marked, leaked = tmp_path / "marked.mp4", tmp_path / "leaked.mp4"
    embed.embed_video(clip, marked, KEY, "alice", rms=2.5, preset="ultrafast")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(marked), "-vf", "scale=720:406,crop=680:370:20:15",
         "-c:v", "libx264", "-crf", "28", "-an", str(leaked)],
        check=True,
    )
    report = detect.detect(leaked, KEY, CANDIDATES)
    assert report.verdict() == "alice"
    assert abs(report.scale - 0.75) < 0.01


def test_cli_issue_then_trace(photo, tmp_path, monkeypatch, capsys, fixed_key):
    monkeypatch.setenv("INVISIMARK_HOME", str(tmp_path / "home"))
    assert cli.main(["keygen"]) == 0
    assert cli.main(["issue", str(photo), "--to", "alice", "--to", "bob", "--strength", "2.2",
                     "--out", str(tmp_path / "out"), "--jobs", "1"]) == 0
    capsys.readouterr()
    copy = tmp_path / "out" / "photo.bob.png"
    assert copy.is_file()
    leaked = tmp_path / "leak.jpg"
    cv2.imwrite(str(leaked), cv2.imread(str(copy)), [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert cli.main(["trace", str(leaked)]) == 0
    assert "traced to: bob" in capsys.readouterr().out
    assert cli.main(["trace", str(photo)]) == 1  # the clean master carries no mark


def test_photo_like_is_deterministic():
    assert np.array_equal(photo_like(), photo_like())


def test_audio_track_survives_marking(tmp_path):
    src, marked = tmp_path / "with_audio.mp4", tmp_path / "marked.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=2,noise=alls=12:allf=t+u",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:v", "libx264", "-crf", "16", "-c:a", "aac",
         "-shortest", str(src)],
        check=True,
    )
    embed.embed_video(src, marked, KEY, "alice", rms=1.6, preset="ultrafast")
    streams = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(marked)],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    assert sorted(streams) == ["audio", "video"]


def test_video_size_that_is_not_a_multiple_of_anything_still_traces(tmp_path):
    src, marked = tmp_path / "odd.mp4", tmp_path / "marked.mp4"
    pan_clip(src, width=702, height=398, frames=48)
    embed.embed_video(src, marked, KEY, "alice", rms=2.5, preset="ultrafast")
    assert detect.detect(marked, KEY, CANDIDATES).verdict() == "alice"
