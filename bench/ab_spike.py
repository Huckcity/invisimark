"""Spike: A/B segment watermarking. Render a lesson twice (variant A, variant B), give each student a
different A/B choice per segment, assemble with `-c copy`, attack, then read the choices back."""
import subprocess, sys, time
from pathlib import Path
import numpy as np
from invisimark import embed, media, detect, keys

S = Path(__file__).parent
SRC = S / sys.argv[1]                      # e.g. screencast.mp4
SEG = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
RMS = float(sys.argv[3]) if len(sys.argv) > 3 else 2.6
FRAMES = int(sys.argv[4]) if len(sys.argv) > 4 else 12
key = keys.load_key(S / "key")
work = S / f"ab_{SRC.stem}"
work.mkdir(exist_ok=True)

info = media.probe(SRC)
n_seg = int(info.duration // SEG)
print(f"{SRC.name}: {info.width}x{info.height} {info.duration:.1f}s -> {n_seg} segments of {SEG}s, rms {RMS}")

# 1. render the two variants (the only expensive step; once per lesson, not per student)
t0 = time.time()
for v in "AB":
    if not (work / f"{v}.mp4").exists():
        embed.embed_video(SRC, work / f"{v}.mp4", key, v, rms=RMS, crf=18, keyframe_every=SEG)
print(f"render A+B: {time.time()-t0:.0f}s")

# 2. cut each variant at the segment boundaries (stream copy)
times = ",".join(f"{SEG*i:g}" for i in range(1, n_seg))
for v in "AB":
    for old in work.glob(f"{v}_seg*.mp4"):
        old.unlink()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", work / f"{v}.mp4", "-c", "copy", "-f", "segment",
                    "-segment_times", times, "-reset_timestamps", "1", work / f"{v}_seg%d.mp4"], check=True)
segs = {v: sorted(work.glob(f"{v}_seg*.mp4"), key=lambda p: int(p.stem.split("seg")[1])) for v in "AB"}
print("segments cut:", {v: [round(media.probe(p).duration, 2) for p in segs[v]] for v in "AB"})

# 3. per-student assembly
rng = np.random.default_rng(5)
students = {f"student{i}": rng.integers(0, 2, n_seg).tolist() for i in range(4)}
copies = {}
for name, bits in students.items():
    lst = work / f"{name}.txt"
    lst.write_text("".join(f"file '{segs['AB'[b]][i]}'\n" for i, b in enumerate(bits)))
    out = work / f"{name}.mp4"
    t0 = time.time()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out], check=True)
    copies[name] = out
    print(f"assembled {name} bits={''.join(map(str, bits))} in {time.time()-t0:.2f}s  ({media.probe(out).duration:.1f}s)")

# 4. attacks
ATTACKS = {
    "none": [],
    "crf28": ["-c:v", "libx264", "-crf", "28"],
    "half+crf30": ["-vf", "scale=iw/2:-2", "-c:v", "libx264", "-crf", "30"],
}

def read_bits(path: Path):
    inf = media.probe(path)
    out = []
    for i in range(n_seg):
        lo, hi = i * SEG + 0.25, (i + 1) * SEG - 0.25
        ts = [lo + (hi - lo) * (k + 0.5) / FRAMES for k in range(FRAMES)]
        fr = [f for f in (media.grab_gray(path, t, inf.width, inf.height) for t in ts) if f is not None]
        rep = detect.detect_frames(fr, key, ["A", "B"])
        z = {m.recipient: m.z for m in rep.matches}
        out.append((rep.marked, rep.sync_z, z.get("A"), z.get("B")))
    return out

ok = tot = 0
for aname, args in ATTACKS.items():
    for name, path in copies.items():
        att = path if not args else work / f"{name}.{aname}.mp4"
        if args:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, *args, "-an", att], check=True)
        res = read_bits(att)
        dec, line = [], []
        for marked, sz, za, zb in res:
            if not marked or za is None or zb is None:
                dec.append("?"); line.append("  -/-  ")
            else:
                dec.append("A" if za > zb else "B"); line.append(f"{za:+5.1f}/{zb:+5.1f}")
        truth = "".join("AB"[b] for b in students[name])
        good = sum(d == t for d, t in zip(dec, truth))
        ok += good; tot += n_seg
        print(f"{aname:11s} {name}: truth {truth} read {''.join(dec)}  correct {good}/{n_seg}   z(A/B) per seg: " + " ".join(line))
print(f"TOTAL segment bits correct: {ok}/{tot}")
