import subprocess
from pathlib import Path

import pytest

from invisimark import course

from .conftest import KEY, pan_clip

STUDENTS = ["alice", "bob", "carol", "dave", "erin", "frank"]


def frames_in(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
         "stream=nb_read_frames", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return int(out.stdout.strip().strip(","))


def test_bits_are_keyed_deterministic_and_distinct():
    a = course.student_bits(KEY, "alice", "lesson1", 64)
    assert a == course.student_bits(KEY, "alice", "lesson1", 64)
    assert a != course.student_bits(KEY, "bob", "lesson1", 64)
    assert a != course.student_bits(KEY, "alice", "lesson2", 64)
    assert a != course.student_bits(b"another-key-0123456789", "alice", "lesson1", 64)
    assert 20 < sum(a) < 44


@pytest.mark.parametrize("fps,seconds,segment", [(30.0, 52.55, 5.0), (29.97, 33.0, 5.0), (24.0, 3.0, 1.0)])
def test_windows_tile_the_video_without_gaps_or_overlap(fps, seconds, segment):
    total = round(seconds * fps)
    seg_frames = round(segment * fps)
    plan = course.Plan("l", "x.mp4", fps, seg_frames, -(-total // seg_frames), total)
    covered = 0
    for i in range(plan.n_seg):
        start, span = plan.window(i)
        first = round(start * fps)
        assert first == covered                       # starts exactly where the last one stopped
        # frames a decoder yields for [start, start+span): those whose time t satisfies start <= t < start + span
        count = sum(1 for j in range(seg_frames + 2) if first / fps + j / fps < start + span - 1e-9)
        covered += count
    assert covered == total


def test_course_roundtrip_names_the_right_student(tmp_path):
    video = pan_clip(tmp_path / "lesson.mp4", frames=144, fps=24)           # 6 s, two kinds of segment
    plan = course.make_plan(video, "lesson", 1.0)
    assert plan.n_seg == 6
    work = tmp_path / "work"
    course.render_lesson(plan, KEY, work, rms=2.0, jobs=4)
    course.render_lesson(plan, KEY, work, rms=2.0, jobs=4)                   # resumes: nothing left to do

    copies = {}
    for student in ("alice", "dave"):
        dst = tmp_path / f"{student}.mp4"
        bits = course.assemble(plan, KEY, student, work, dst)
        assert bits == course.student_bits(KEY, student, "lesson", plan.n_seg)
        assert frames_in(dst) == plan.total_frames                            # a lossless join keeps every frame
        copies[student] = dst

    result = course.trace(copies["dave"], KEY, [plan], STUDENTS, frames=8)
    assert result.reading is not None and result.reading.lesson == "lesson"
    assert result.readable == 6
    top = result.scores[0]
    assert top.student == "dave"
    assert top.agree >= 5 and top.agree > result.scores[1].agree
    assert top.p_roster > course.NAME_ODDS or result.verdict == ["dave"]      # six segments cannot over-claim

    other = course.trace(copies["alice"], KEY, [plan], [s for s in STUDENTS if s != "alice"], frames=8)
    assert other.verdict == []                                                # the real leaker is not on the roster


def test_an_unmarked_video_traces_to_nobody(tmp_path):
    video = pan_clip(tmp_path / "plain.mp4", frames=96, fps=24)
    plan = course.make_plan(video, "lesson", 1.0)
    result = course.trace(video, KEY, [plan], STUDENTS, frames=8)
    assert result.reading is None and result.verdict == []


def test_score_students_uses_exact_odds():
    reading = course.Reading("l", 10.0, 1.0, 0, soft=[8.0] * 20 + [None] * 4)
    bits = course.student_bits(KEY, "alice", "l", 24)
    reading.soft = [(8.0 if b == 0 else -8.0) if i < 20 else None for i, b in enumerate(bits)]
    scores = course.score_students(KEY, reading, STUDENTS)
    assert scores[0].student == "alice" and scores[0].agree == 20 and scores[0].disagree == 0
    assert scores[0].p == pytest.approx(2.0 ** -20)
    assert scores[0].p_roster == pytest.approx(6 * 2.0 ** -20)
    assert all(s.p > 1e-3 for s in scores[1:])
