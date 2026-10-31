"""Command line: keygen, issue, trace, inspect, course, and for AI marking: mark, check, dev-cert, report-key, verify-report."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from . import ai, calibrate, course, embed, keys, media
from . import detect as detect_mod

# RMS of the added pattern, in 8-bit code values. A plain x264 re-encode at crf 18 already costs about 1.6.
STRENGTHS = {"subtle": 0.8, "standard": 1.2, "strong": 2.0}


def _strength(text: str) -> float | None:
    """RMS in code values, or None for 'auto' (calibrate against the attack panel)."""
    if text == "auto":
        return None
    return STRENGTHS[text] if text in STRENGTHS else float(text)


def _key(args) -> bytes:
    return keys.load_key(keys.key_path(args.key))


def _issue_one(job: tuple) -> dict:
    src, dst, recipient, key_hex, strength, crf = job
    key = bytes.fromhex(key_hex)
    if media.is_image(src):
        embed.embed_image(src, dst, key, recipient, rms=strength)
    else:
        embed.embed_video(src, dst, key, recipient, rms=strength, crf=crf)
    return {"recipient": recipient, "output": str(dst), "output_sha256": keys.sha256_of(Path(dst))}


def cmd_keygen(args) -> int:
    path = keys.key_path(args.key)
    keys.create_key(path, force=args.force)
    print(f"wrote {path} (mode 600). Back it up: without it, issued copies cannot be traced.")
    return 0


def cmd_issue(args) -> int:
    key = _key(args)
    src = Path(args.master)
    if not src.is_file():
        print(f"{src}: no such file", file=sys.stderr)
        return 1
    recipients = list(args.to or [])
    if args.to_file:
        recipients += [ln.strip() for ln in Path(args.to_file).read_text().splitlines() if ln.strip()]
    if not recipients:
        print("name at least one recipient with --to or --to-file", file=sys.stderr)
        return 1
    out_dir = Path(args.out) if args.out else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    strength = _strength(args.strength)
    ledger = keys.ledger_path(args.ledger)
    master_hash = keys.sha256_of(src)
    calibration = None
    if strength is None:
        print("calibrating strength against recompression, downscaling and screen capture ...", flush=True)
        cal = calibrate.calibrate(src, key, max_rms=args.max_strength)
        strength, calibration = cal.rms, cal.summary()
        print(calibration)
        if not cal.passed:
            print("warning: this picture is too flat or dark to carry a mark that survives heavy mangling at an "
                  "invisible strength. Copies are traceable when shared as-is or lightly recompressed.", file=sys.stderr)
    recipients = list(dict.fromkeys(recipients))
    stems = keys.unique_slugs(recipients)
    jobs = [
        (str(src), str(out_dir / f"{src.stem}.{stems[r]}{src.suffix}"), r, key.hex(), strength, args.crf)
        for r in recipients
    ]
    workers = max(1, min(len(jobs), args.jobs))
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for job, done in zip(jobs, pool.map(_issue_one, jobs)):
            keys.record_issue(
                ledger, master=str(src), master_sha256=master_hash, strength=strength, calibration=calibration, **done
            )
            print(f"{done['recipient']}: {done['output']}")
    return 0


def cmd_trace(args) -> int:
    key = _key(args)
    leak = Path(args.leak)
    if not leak.is_file():
        print(f"{leak}: no such file", file=sys.stderr)
        return 1
    candidates = list(keys.recipients_in(keys.ledger_path(args.ledger)))
    if args.candidates:
        extra = [ln.strip() for ln in Path(args.candidates).read_text().splitlines() if ln.strip()]
        candidates += [c for c in extra if c not in candidates]
    report = detect_mod.detect(leak, key, candidates, frames=args.frames)
    verdict = report.verdict()
    if args.json:
        print(json.dumps({
            "marked": report.marked, "sync_z": report.sync_z, "scale": report.scale, "mirrored": report.mirrored,
            "verdict": verdict, "lead": report.lead(), "threshold": detect_mod.id_threshold(len(candidates)),
            "orientation": report.orientation, "matches": [{"recipient": m.recipient, "z": m.z} for m in report.matches],
        }, indent=2))
    else:
        if not report.marked:
            print(f"no mark from this key found (sync z {report.sync_z:.1f}; {detect_mod.SYNC_SURE:g} on its own, or "
                  f"{detect_mod.SYNC_GATE:g} plus a candidate that stands out).")
        else:
            note = " (mirrored)" if report.mirrored else ""
            print(f"marked by this key: sync z {report.sync_z:.1f}, scale x{report.scale:.3f}{note}")
            if not candidates:
                print("no candidates: issue copies with this key (ledger) or pass --candidates FILE.")
            for m in report.matches[:5]:
                print(f"  {m.recipient:<30} z {m.z:6.1f}")
            if verdict:
                print(f"=> traced to: {verdict}")
            elif report.lead():
                print(f"=> weak lead: {report.lead()} (z {report.matches[0].z:.1f}, below the {detect_mod.id_threshold(len(candidates)):.1f} "
                      "needed to name anyone). Treat it as a hint, not evidence.")
            else:
                print("=> marked, but no candidate stands out; do not accuse anyone on this evidence.")
    if verdict:
        return 0
    return 2 if report.marked else 1


def _course_ledger(args) -> Path:
    return keys.ledger_path(args.ledger).with_name("course-ledger.jsonl")


def _read_names(path: str | None) -> list[str]:
    if not path:
        return []
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def cmd_course_issue(args) -> int:
    key = _key(args)
    students = list(dict.fromkeys(_read_names(args.roster) + list(args.to or [])))
    if not students:
        print("name the students with --roster FILE or --to NAME", file=sys.stderr)
        return 1
    out_dir = Path(args.out)
    work = Path(args.work) if args.work else out_dir / ".work"
    ledger = _course_ledger(args)
    stems = keys.unique_slugs(students)
    lessons = [Path(p) for p in args.lessons]
    names = [p.stem for p in lessons]
    if len(set(names)) != len(names):
        print("lesson file names must be unique (they label the lessons)", file=sys.stderr)
        return 1
    for src in lessons:
        if not src.is_file() or media.is_image(src):
            print(f"{src}: not a video file", file=sys.stderr)
            return 1
    for src in lessons:
        plan = course.make_plan(src, src.stem, args.segment)
        lesson_work = work / src.stem
        strength = _strength(args.strength)
        if strength is None:
            print(f"{src.name}: calibrating strength ...", flush=True)
            cal = calibrate.calibrate(src, key, max_rms=args.max_strength)
            strength = cal.rms
            print(cal.summary())
            if not cal.passed:
                print("warning: this footage is too flat or dark to carry a mark that survives heavy mangling at an "
                      "invisible strength. Copies are traceable when shared as-is or lightly recompressed.", file=sys.stderr)
        print(f"{src.name}: {plan.n_seg} segments x 2 variants, strength {strength:g}", flush=True)
        course.render_lesson(plan, key, lesson_work, strength, args.crf, args.jobs,
                             progress=lambda d, n: print(f"\r  rendered {d}/{n}", end="", flush=True))
        print()
        keys.record_issue(ledger, **plan.to_row(), strength=strength, master_sha256=keys.sha256_of(src))
        for student in students:
            dst = out_dir / stems[student] / f"{src.stem}{src.suffix if src.suffix.lower() in {'.mp4', '.m4v'} else '.mp4'}"
            course.assemble(plan, key, student, lesson_work, dst)
            keys.record_issue(ledger, kind="course-copy", recipient=student, lesson=src.stem, output=str(dst),
                              output_sha256=keys.sha256_of(dst))
        print(f"  {len(students)} copies written under {out_dir}")
    return 0


def cmd_course_trace(args) -> int:
    key = _key(args)
    leak = Path(args.leak)
    if not leak.is_file():
        print(f"{leak}: no such file", file=sys.stderr)
        return 1
    rows = keys.read_ledger(_course_ledger(args))
    plans = {r["lesson"]: course.Plan.from_row(r) for r in rows if r.get("kind") == "course-lesson"}
    if not plans:
        print("no course lessons in the ledger: run `invisimark course issue` first", file=sys.stderr)
        return 1
    students = list(dict.fromkeys([r["recipient"] for r in rows if r.get("kind") == "course-copy"] + _read_names(args.roster)))
    result = course.trace(leak, key, list(plans.values()), students, trim=args.trim, frames=args.frames)
    if args.json:
        print(json.dumps({
            "lesson": result.reading.lesson if result.reading else None, "named": result.verdict,
            "lead": result.lead, "readable_segments": result.readable,
            "segments": len(result.reading.soft) if result.reading else 0,
            "scores": [{"student": s.student, "z": s.z, "agree": s.agree, "disagree": s.disagree, "p": s.p, "p_roster": s.p_roster}
                       for s in result.scores[:10]],
        }, indent=2))
    elif result.reading is None:
        print("no mark from this key found in that file (or none from any lesson in the ledger).")
    else:
        r = result.reading
        print(f"lesson {r.lesson}: sync z {r.sync_z:.1f}, scale x{r.scale:.3f}; "
              f"read {result.readable} of {len(r.soft)} segments")
        for s in result.scores[:5]:
            print(f"  {s.student:<30} agrees on {s.agree} of {s.agree + s.disagree} clear segments; "
                  f"an unrelated student would do that by luck 1 time in {1 / max(s.p, 1e-300):,.0f}")
        if len(result.verdict) == 1:
            print(f"=> traced to: {result.verdict[0]}")
        elif result.verdict:
            print(f"=> traced to several students (shared or swapped segments?): {', '.join(result.verdict)}")
        elif result.lead:
            print(f"=> weak lead: {result.lead}. Odds of this by luck anywhere on the roster are 1 in "
                  f"{1 / max(result.scores[0].p_roster, 1e-300):,.0f}, short of the 1 in {1 / course.NAME_ODDS:,.0f} needed to name "
                  "anyone. More leaked footage (or shorter segments next time) would settle it.")
        else:
            print("=> nobody can be named on this evidence; do not accuse anyone.")
    if result.verdict:
        return 0
    return 2 if result.reading else 1


def cmd_host(args) -> int:
    import uvicorn

    from . import service

    if args.keep_uploads:
        os.environ["IM_KEEP_UPLOADS"] = "1"
        if args.keep_dir:
            os.environ["IM_KEEP_DIR"] = args.keep_dir
    service.log_to_stdout()          # job events only (no addresses, names or file names); the access log stays off below
    app = service.create_app()
    limits = app.state.service.limits
    print(f"hosted service on http://{args.host}:{args.port}  (limits come from IM_* environment variables)", flush=True)
    if limits.keep_uploads:
        print(f"TEST MODE: every upload is kept in {limits.keep_dir} (the pages say so); results last {limits.ttl_text()}.", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)
    return 0


def cmd_serve(args) -> int:
    from . import web

    web.serve(keys.key_path(args.key), keys.ledger_path(args.ledger), args.port, not args.no_browser)
    return 0


def cmd_inspect(args) -> int:
    key = _key(args)
    report = detect_mod.detect(Path(args.file), key, [], frames=args.frames)
    print(f"{'MARKED' if report.marked else 'not marked'}: sync z {report.sync_z:.1f}, scale x{report.scale:.3f}")
    return 0 if report.marked else 1


def _provider_key(args) -> bytes | None:
    return _key(args) if args.provider_key else None


def cmd_mark(args) -> int:
    signing = None
    if args.cert or args.signing_key:
        if not (args.cert and args.signing_key):
            print("signing needs both --cert and --signing-key", file=sys.stderr)
            return 1
        signing = ai.Signing(args.cert, args.signing_key, None if args.no_timestamp else args.timestamper)
    else:
        print("warning: no signing identity, so only the watermark layer was added. The Code of Practice asks for signed metadata too; "
              "see --cert and --signing-key.", file=sys.stderr)
    strength = "auto" if args.strength == "auto" else _strength(args.strength)
    done = ai.mark(args.input, args.output, signing=signing, generated=not args.edited, provider=args.provider, system=args.system,
                   model=args.model, strength=strength, provider_key=_provider_key(args), crf=args.crf, log=args.log)
    if args.json:
        print(json.dumps(asdict(done), indent=2))
    else:
        print(f"{done.output}: watermark at strength {done.strength:g}, {'signed' if done.signed else 'not signed'}")
        if done.calibration:
            print(done.calibration)
        if done.robust is False:
            print("warning: this file is too flat or dark to carry a mark that survives heavy changes at an invisible strength.", file=sys.stderr)
    return 0


def cmd_check(args) -> int:
    anchors = Path(args.trust_anchors).read_text() if args.trust_anchors else None
    result = ai.check(args.file, _provider_key(args), anchors, frames=args.frames)
    if args.report_key:
        if not Path(args.report_key).is_file():
            print(f"{args.report_key}: no such file", file=sys.stderr)
            return 1
        report = ai.signed_report(result, Path(args.report_key).read_text().strip())
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if args.json:
        print(json.dumps(result.public(), indent=2, ensure_ascii=False))
    else:
        print("\n".join(result.summary))
    return 0 if result.ai_marked else 1


def cmd_dev_cert(args) -> int:
    made = ai.dev_certificate(args.out)
    print(f"wrote {made.cert} and {made.key}. For testing only: anything signed with them validates as untrusted.")
    return 0


def cmd_report_key(args) -> int:
    path = Path(args.out)
    if path.exists():
        print(f"{path} already exists; refusing to overwrite it", file=sys.stderr)
        return 1
    private, public = ai.report_keypair()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(private + "\n")
    print(f"wrote {path} (mode 600). Public key, for people verifying your results:\n{public}")
    return 0


def cmd_verify_report(args) -> int:
    ok = ai.verify_report(json.loads(Path(args.report).read_text()), args.public_key)
    if not ok:
        print("signature does not match")
        return 1
    print("signature matches" + ("" if args.public_key else ", but with no --public-key this only shows the report is intact, not who signed it"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="invisimark", description="Invisible per-recipient watermarks for images and video.")
    ap.add_argument("--key", help="key file (default: $INVISIMARK_KEY or ~/.config/invisimark/key)")
    ap.add_argument("--ledger", help="ledger file (default: next to the key)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("keygen", help="create your secret key")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_keygen)

    p = sub.add_parser("issue", help="make a uniquely marked copy for each recipient")
    p.add_argument("master", help="image or video to mark")
    p.add_argument("--to", action="append", help="recipient label (repeatable)")
    p.add_argument("--to-file", help="file with one recipient label per line")
    p.add_argument("--out", help="output directory (default: beside the master)")
    p.add_argument("--strength", default="auto",
                   help="auto (calibrate per master) | subtle | standard | strong | RMS in code values")
    p.add_argument("--max-strength", type=float, default=2.6, help="visibility ceiling for auto, in RMS code values")
    p.add_argument("--crf", type=int, default=18, help="x264 quality for video output (lower is better)")
    p.add_argument("--jobs", type=int, default=2, help="copies to encode in parallel")
    p.set_defaults(fn=cmd_issue)

    p = sub.add_parser("trace", help="find out which recipient a leaked copy came from")
    p.add_argument("leak")
    p.add_argument("--candidates", help="file with one recipient label per line (added to the ledger)")
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_trace)

    p = sub.add_parser("course", help="per-student copies of whole courses (A/B segment marking)")
    csub = p.add_subparsers(dest="course_cmd", required=True)
    q = csub.add_parser("issue", help="make one marked copy of every lesson for every student")
    q.add_argument("lessons", nargs="+", help="lesson video files")
    q.add_argument("--roster", help="file with one student per line")
    q.add_argument("--to", action="append", help="student label (repeatable)")
    q.add_argument("--out", required=True, help="output directory: one folder per student")
    q.add_argument("--work", help="where the rendered segments live (default: OUT/.work; keep it to resume or add students)")
    q.add_argument("--segment", type=float, default=course.SEGMENT_SECONDS, help="segment length in seconds")
    q.add_argument("--strength", default="auto")
    q.add_argument("--max-strength", type=float, default=2.6)
    q.add_argument("--crf", type=int, default=18)
    q.add_argument("--jobs", type=int, default=max(1, (multiprocessing.cpu_count() or 2) // 2), help="segments rendered in parallel")
    q.set_defaults(fn=cmd_course_issue)
    q = csub.add_parser("trace", help="find out which student a leaked lesson came from")
    q.add_argument("leak")
    q.add_argument("--roster", help="extra students to consider, one per line")
    q.add_argument("--trim", type=float, default=0.0, help="seconds of the lesson missing from the start of the leak")
    q.add_argument("--frames", type=int, default=12, help="frames sampled per segment")
    q.add_argument("--json", action="store_true")
    q.set_defaults(fn=cmd_course_trace)

    p = sub.add_parser("mark", help="mark an AI-generated image or video: public watermark plus signed C2PA metadata")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--edited", action="store_true", help="the file was AI-manipulated, not generated from nothing")
    p.add_argument("--cert", help="PEM certificate chain for signing, leaf first")
    p.add_argument("--signing-key", help="PEM private key for signing")
    p.add_argument("--timestamper", default=ai.DEFAULT_TIMESTAMPER, help="RFC 3161 timestamp authority")
    p.add_argument("--no-timestamp", action="store_true", help="for machines that cannot reach a timestamp authority")
    p.add_argument("--provider", help="who provides the AI system, recorded in the signed metadata")
    p.add_argument("--system", help="the AI system's name")
    p.add_argument("--model", help="the model's name")
    p.add_argument("--strength", default=str(ai.DEFAULT_RMS), help="RMS in code values, or auto to measure it on this file (slower)")
    p.add_argument("--provider-key", action="store_true", help="also add a private tile from --key, so you can tell your output from a copy")
    p.add_argument("--crf", type=int, default=18, help="x264 quality for video output (lower is better)")
    p.add_argument("--log", help="append one line per marked file to this ledger")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_mark)

    p = sub.add_parser("check", help="does this file carry an AI marking? Exit 0 if so, 1 if not")
    p.add_argument("file")
    p.add_argument("--provider-key", action="store_true", help="also look for the private tile from --key")
    p.add_argument("--trust-anchors", help="PEM file of trusted roots, such as the C2PA trust list")
    p.add_argument("--report", default="invisimark-result.json", help="where --report-key writes the signed result")
    p.add_argument("--report-key", help="file holding the private key from `report-key`: also write a signed result")
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("dev-cert", help="make a throwaway signing certificate for testing")
    p.add_argument("--out", default="invisimark-dev-cert")
    p.set_defaults(fn=cmd_dev_cert)

    p = sub.add_parser("report-key", help="make a key for signing detection results")
    p.add_argument("--out", default="invisimark-report-key")
    p.set_defaults(fn=cmd_report_key)

    p = sub.add_parser("verify-report", help="check the signature on a detection result")
    p.add_argument("report")
    p.add_argument("--public-key", help="the detector's published public key")
    p.set_defaults(fn=cmd_verify_report)

    p = sub.add_parser("host", help="run the public, anonymous, capped service (see docs/self-hosting.md)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--keep-uploads", action="store_true", help="test mode: keep every upload (the pages say so)")
    p.add_argument("--keep-dir", help="where kept uploads go (default ~/.local/share/invisimark/kept)")
    p.set_defaults(fn=cmd_host)

    p = sub.add_parser("serve", help="open the local web app")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("inspect", help="is this file marked with my key at all?")
    p.add_argument("file")
    p.add_argument("--frames", type=int, default=32)
    p.set_defaults(fn=cmd_inspect)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (keys.KeyError_, media.MediaError, ai.MarkError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
