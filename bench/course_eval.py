"""Attack matrix for course tracing: student copy -> attack -> course.trace."""
import subprocess, sys, time, json
from pathlib import Path
from invisimark import course, keys

S = Path(__file__).parent
C = S / "course"
key = keys.load_key(S / "key")
rows = keys.read_ledger(S / "home" / "course-ledger.jsonl")
plans = {}
for r in rows:
    if r.get("kind") == "course-lesson":
        plans[r["lesson"]] = course.Plan.from_row(r)
students = list(dict.fromkeys(r["recipient"] for r in rows if r.get("kind") == "course-copy"))
lesson = sys.argv[1]
who = sys.argv[2].split(",")
ATT = {
    "none": ([], 0.0),
    "crf28": (["-c:v", "libx264", "-crf", "28"], 0.0),
    "crf34": (["-c:v", "libx264", "-crf", "34"], 0.0),
    "scale0.75+crf28": (["-vf", "scale=iw*0.75:-2", "-c:v", "libx264", "-crf", "28"], 0.0),
    "trim12+crf23": (["-ss", "12", "-c:v", "libx264", "-crf", "23"], 12.0),
}
which = sys.argv[3].split(",") if len(sys.argv) > 3 else list(ATT)
for name in who:
    src = C / "out" / name / f"{lesson}.mp4"
    for a in which:
        args, trim = ATT[a]
        leak = src if not args else C / f"leak_{name}_{a.replace('+','_')}.mp4"
        if args:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, *args, "-an", leak], check=True)
        t0 = time.time()
        res = course.trace(leak, key, list(plans.values()), students, trim=trim)
        top = res.scores[0] if res.scores else None
        out = "NO MARK" if res.reading is None else (
            f"readable {res.readable}/{len(res.reading.soft)}  top {top.student} agree {top.agree}/{top.agree+top.disagree}  "
            f"roster-odds 1 in {1/max(top.p_roster,1e-300):,.0f}  next {res.scores[1].student} {res.scores[1].agree}/{res.scores[1].agree+res.scores[1].disagree}")
        verdict = "NAMED " + ",".join(res.verdict) if res.verdict else ("lead " + res.lead if res.lead else "-")
        ok = "OK " if (res.verdict == [name]) else ("LEAD-ok" if res.lead == name else "MISS")
        print(f"{name:6s} {a:16s} {ok:8s} {verdict:16s} {out}  [{time.time()-t0:.0f}s]", flush=True)
