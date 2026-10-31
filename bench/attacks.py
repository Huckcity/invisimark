"""Robustness benchmark: mark a clip, mangle it the ways a leak gets mangled, and see who the detector names.

    uv run python bench/attacks.py CLIP.mp4 [CLIP.mp4 ...] [--only name,name] [--out results.md]

Every attack is an ffmpeg recipe applied to the *marked* file. "screenrec" is the one to watch: it models
a phone-free screen capture (player scaled down, window offset inside a larger desktop, capture re-encode).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

from invisimark import calibrate, detect, embed

cv2.setNumThreads(1)

KEY = b"bench-key"
RECIPIENT = "alice"
CANDIDATES = [f"user{i}" for i in range(500)] + [RECIPIENT]

X264 = ["-c:v", "libx264", "-preset", "fast"]

ATTACKS: dict[str, list[str]] = {
    "none": ["-c:v", "copy"],
    "crf23": [*X264, "-crf", "23"],
    "crf28": [*X264, "-crf", "28"],
    "crf34": [*X264, "-crf", "34"],
    "720p_2M": ["-vf", "scale=1280:720", *X264, "-b:v", "2M", "-maxrate", "2M", "-bufsize", "4M"],
    "480p_1M": ["-vf", "scale=854:480", *X264, "-b:v", "1M", "-maxrate", "1M", "-bufsize", "2M"],
    "360p": ["-vf", "scale=640:360", *X264, "-crf", "30"],
    "240p": ["-vf", "scale=426:240", *X264, "-crf", "30"],
    "up1080p": ["-vf", "scale=1920:1080:flags=bicubic", *X264, "-crf", "26"],
    "hevc_crf30": ["-c:v", "libx265", "-preset", "fast", "-crf", "30", "-tag:v", "hvc1"],
    "vp9_crf40": ["-c:v", "libvpx-vp9", "-crf", "40", "-b:v", "0", "-deadline", "realtime", "-cpu-used", "8"],
    "fps24": ["-vf", "fps=24", *X264, "-crf", "28"],
    "fps15": ["-vf", "fps=15", *X264, "-crf", "28"],
    "crop10": ["-vf", "crop=1152:648:57:33", *X264, "-crf", "26"],
    "crop25": ["-vf", "crop=960:540:213:97", *X264, "-crf", "26"],
    "zoom80": ["-vf", "crop=1024:576:101:57,scale=1280:720", *X264, "-crf", "26"],
    "vertical": ["-vf", "crop=405:720:437:0,scale=1080:1920", *X264, "-crf", "26"],
    "bright": ["-vf", "eq=brightness=0.06:contrast=1.12:saturation=1.2:gamma=1.1", *X264, "-crf", "26"],
    "blur1": ["-vf", "gblur=sigma=1.0", *X264, "-crf", "26"],
    "blur2": ["-vf", "gblur=sigma=2.0", *X264, "-crf", "26"],
    "sharpen": ["-vf", "unsharp=5:5:1.5", *X264, "-crf", "26"],
    "grain": ["-vf", "noise=alls=12:allf=t+u", *X264, "-crf", "26"],
    "hflip": ["-vf", "hflip", *X264, "-crf", "26"],
    "screenrec": [
        "-vf", "scale=1000:562,gblur=sigma=0.6,eq=contrast=1.04:brightness=0.02,pad=1920:1080:101:57:black",
        *X264, "-crf", "26",
    ],
    "screenrec_small": [
        "-vf", "scale=720:406,gblur=sigma=0.7,pad=1920:1080:333:211:0x202020", *X264, "-crf", "28",
    ],
    "trim2s": ["-t", "2", *X264, "-crf", "26"],
    "worst": [
        "-vf", "scale=640:360,crop=560:300:30:20,gblur=sigma=0.8,eq=brightness=0.04:contrast=1.08,scale=1120:600",
        *X264, "-crf", "32",
    ],
}


def attack(marked: Path, name: str, workdir: Path) -> Path:
    out = workdir / f"{marked.stem}.{name}.{'webm' if 'vp9' in name else 'mp4'}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(marked), *ATTACKS[name], "-an", str(out)], check=True)
    return out


def run_one(args: tuple[str, str, str, float]) -> dict:
    marked, name, workdir, strength = args
    started = time.time()
    try:
        leaked = attack(Path(marked), name, Path(workdir))
        rep = detect.detect(leaked, KEY, CANDIDATES)
    except Exception as exc:  # a failed attack is a result, not a crash
        return {"attack": name, "error": str(exc)[:80]}
    top = rep.matches[:2]
    return {
        "attack": name,
        "marked": rep.marked,
        "sync_z": rep.sync_z,
        "scale": rep.scale,
        "top": top[0].recipient if top else "-",
        "z1": top[0].z if top else 0.0,
        "z2": top[1].z if len(top) > 1 else 0.0,
        "ok": rep.verdict() == RECIPIENT,
        "secs": time.time() - started,
    }


def quality(orig: Path, marked: Path) -> dict:
    """PSNR/SSIM of the marked file against the original, and of a plain re-encode as the control."""
    def measure(a: Path, b: Path) -> tuple[float, float]:
        proc = subprocess.run(
            ["ffmpeg", "-i", str(a), "-i", str(b), "-lavfi", "[0:v][1:v]ssim;[0:v][1:v]psnr", "-f", "null", "-"],
            capture_output=True, text=True,
        )
        ssim = float(proc.stderr.split("SSIM")[-1].split("All:")[1].split()[0]) if "SSIM" in proc.stderr else 0.0
        psnr = float(proc.stderr.split("average:")[-1].split()[0]) if "average:" in proc.stderr else 0.0
        return psnr, ssim

    with tempfile.TemporaryDirectory() as tmp:
        control = Path(tmp) / "control.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(orig), "-c:v", "libx264", "-crf", "18",
                        "-preset", "medium", "-an", str(control)], check=True)
        p_m, s_m = measure(marked, orig)
        p_c, s_c = measure(control, orig)
    return {"psnr_marked": p_m, "ssim_marked": s_m, "psnr_control": p_c, "ssim_control": s_c}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--strength", type=float, default=0.0, help="RMS of the added pattern; 0 calibrates per clip")
    ap.add_argument("--keep", default="", help="directory to keep marked and attacked files in")
    opts = ap.parse_args()
    names = [n for n in (opts.only.split(",") if opts.only else ATTACKS) if n]

    lines: list[str] = []
    for clip in opts.clips:
        clip_path = Path(clip)
        workdir = Path(opts.keep) if opts.keep else Path(tempfile.mkdtemp(prefix="invisimark-"))
        workdir.mkdir(parents=True, exist_ok=True)
        marked = workdir / f"{clip_path.stem}.marked.mp4"
        started = time.time()
        rms = opts.strength
        note = ""
        if rms <= 0:
            cal = calibrate.calibrate(clip_path, KEY)
            rms, note = cal.rms, f"auto: {cal.summary()}; "
        embed.embed_video(clip_path, marked, KEY, RECIPIENT, rms=rms)
        q = quality(clip_path, marked)
        lines.append(f"\n### {clip_path.name} (RMS {rms:g})\n")
        lines.append(
            f"{note}embed {time.time() - started:.0f}s; vs original: marked PSNR {q['psnr_marked']:.1f} dB / SSIM "
            f"{q['ssim_marked']:.4f}; plain crf18 re-encode PSNR {q['psnr_control']:.1f} dB / SSIM {q['ssim_control']:.4f}\n"
        )
        lines.append("| attack | marked | sync z | scale | accused | z | runner-up | named |")
        lines.append("|---|---|---|---|---|---|---|---|")
        with ProcessPoolExecutor(opts.workers) as pool:
            for res in pool.map(run_one, [(str(marked), n, str(workdir), opts.strength) for n in names]):
                if "error" in res:
                    lines.append(f"| {res['attack']} | error | | | {res['error']} | | | |")
                    continue
                lines.append(
                    f"| {res['attack']} | {'yes' if res['marked'] else 'no'} | {res['sync_z']:.1f} | {res['scale']:.3f} | "
                    f"{res['top']} | {res['z1']:.1f} | {res['z2']:.1f} | {'**YES**' if res['ok'] else 'no'} |"
                )
                print(lines[-1], flush=True)
    text = "\n".join(lines)
    print(text)
    if opts.out:
        Path(opts.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
