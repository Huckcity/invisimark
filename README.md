# invisimark

Invisible per-recipient watermarks for images and video. Give every recipient a copy that looks identical to the
original but carries a mark only you can read, then, when a copy leaks, upload it and find out whose it was.

Everything runs on your machine. There is no server, no account and no upload.

It also [marks AI-generated images and video](#marking-ai-generated-images-and-video) with a public watermark and signed C2PA metadata, for apps preparing for EU AI Act Article 50.

```
$ invisimark keygen
$ invisimark issue master.mp4 --to alice --to bob --to order-1093
calibrating strength against recompression, downscaling and screen capture ...
strength 2.1 (robust); tightest case half-size: sync z 7.6, id z 10.7
alice: master.alice.mp4
bob: master.bob.mp4
order-1093: master.order-1093.mp4

$ invisimark trace screen-recording-from-a-forum.mp4
marked by this key: sync z 8.2, scale x0.565
  bob                            z   14.4
  alice                          z    2.9
  order-1093                     z    2.6
=> traced to: bob
```

Or run `invisimark serve` for a local web app that does the same with drag and drop.

## What it survives

Measured, not claimed. `bench/` marks a clip, mangles the marked file the way leaks get mangled, then traces it
with 500 wrong names plus the right one in the list. "Named" means the right recipient was accused, at the
thresholds described below, with nobody else near.

**Video** (720p, 10 s clips, 27 attacks each; `bench/results/video.md`)

| clip | picture | strength | PSNR vs original | named |
|---|---|---|---|---|
| `bbb` | textured animation | 2.1 (auto) | 40.8 dB (plain re-encode: 43.6) | **27 / 27** |
| `jelly` | natural footage | 1.75 (auto) | 43.3 dB (plain re-encode: 47.1) | **26 / 27**, misses 426x240 |
| `sintel` | flat shading, dark scenes | 2.6 (ceiling) | 41.2 dB (plain re-encode: 50.7) | 21 / 27, flagged **weak** |

The 27 attacks: x264 at crf 23, 28 and 34; H.265 crf 30; VP9 crf 40; downscale to 720p@2 Mbps, 480p@1 Mbps, 360p
and 240p; upscale to 1080p; 24 and 15 fps; crop 10% and 25%; zoom to 80%; reframe to 9:16 vertical; brightness,
contrast, saturation and gamma shifts; Gaussian blur at sigma 1 and 2; sharpen; film grain; left-right mirror;
a screen recording of the player inside a 1920x1080 desktop, at two sizes; a 2 s trim; and a worst case that
stacks downscale, crop, blur, a brightness shift and crf 32.

These are three clips from one source of open test footage, so read them as a sketch of the range, not a guarantee
for your material. `issue` re-measures on *your* file before it marks anything.

**Images** (17 attacks each; `bench/results/images.md`). 1280x720 stills cut from the clips: `jelly` and `sintel` named
**17 / 17**, `bbb` 16 / 17 (it misses a 60% crop at the very low strength of 0.85, PSNR 48.6 dB). The attacks include
JPEG quality 30, 80% and 60% crops, half size, a chat-app resize-and-recompress, a screenshot, mirroring and a combined
worst case. Small stock photos work too: 600x400 named 15 / 17 and 512x512 named 15 / 17 (flagged weak); both miss the
60% crop and the stacked worst case.

**False accusations** (`bench/results/null_study.txt`): 260 runs on unmarked images and videos, and on marked videos
traced with a wrong key, a different random key each time. The sync statistic had a mean of 5.0, a standard deviation
of 0.6 and a maximum of 6.6; it needs 7.5 to count as a mark on its own, and nothing got there. The best of 200 wrong
names scored 4.4 at most against a threshold of 6.0. Nobody was named, and nothing was called marked.

**Speed**: marking a 10 s 720p clip takes about 8-10 s per recipient, plus 35-50 s of calibration once per master.
Tracing a video typically takes 10-30 s. A 12 megapixel photo marks in about a second and traces in about 9.

## What it does not survive

Be honest with yourself about these before you rely on it.

* **Flat or dark pictures.** The mark lives where the picture has texture, because that is where the eye and the
  encoder both let it through. Cartoon-flat shading, black backgrounds and sky carry little. `issue` measures this and
  prints `WEAK`; then copies are traceable when shared as they are or lightly recompressed, not through heavy
  recompression. Do not promise yourself more than the calibration says.
* **Small pictures and heavy crops.** The mark repeats every 128 px and detection needs a few dozen repeats. A 512 px
  picture survives recompression and resizing but not a 60% crop. Mark the full-size original, not a thumbnail.
* **Very small copies.** 240p (a third of 720p) is the edge for video.
* **A phone filming a screen.** Perspective, moire and rotation by any angle other than a multiple of 90 degrees are
  not handled and not tested. Screen *recordings* are tested.
* **Generative reprocessing.** AI upscaling, denoising or "enhance" passes, and frame interpolation, are untested and
  should be assumed to strip it.
* **Someone who knows it is there.** Strong noise plus blur, or averaging several recipients' copies together, damages
  the mark. In the tests two colluders' averaged copy ranks both of them first and second; with many colluders you
  should expect no one to stand out. Treat this as deterrence plus evidence, not a lock.
* **Video format limits.** 8-bit SDR only. Variable-frame-rate sources are re-timed to constant. HDR is untested.

## Reading a result

* **traced to X**: X scored above the threshold for the size of your candidate list (at least 6, higher for long
  lists) and clear of the runner-up. The threshold is set so an unrelated name should pass about once in ten million;
  that assumes the null stays Gaussian in the tail, which was checked only out to about 4.5.
* **weak lead: X**: X is clearly ahead but below the threshold. A reason to look, not a reason to accuse.
* **marked, but no one stands out**: your mark is present (the alignment signal is strong on its own) but the copy was
  changed too much to tell recipients apart, or the leaker is not on your list.
* **no mark from this key**: either the file did not come from you or it was changed too much. Absence is not proof
  of innocence. "Marked" needs either a strong alignment signal on its own (7.5, against a null that peaks near 6.6) or
  a weaker one together with a candidate that clears the naming threshold.

A match is evidence, not proof. Check it against how and when you sent each copy before acting on it.

## How it works

Each copy gets two band-limited noise tiles added to its brightness, repeated every 128 px and scaled by a
perceptual mask (more where the picture is busy). Both are derived from your secret key with HMAC, so no database is
needed to re-derive them:

* the **sync tile** is the same for every copy of a key and sits in the low bands, which survive compression and blur
  best. It lets the detector recover scale, crop and orientation;
* the **id tile** is unique to the recipient's label and sits in the mid bands.

To trace, the detector band-passes and soft-clips sampled frames (so a few sharp edges cannot outvote the mark),
averages them, then walks a range of scales, resizing the average until the pattern's period is 128 px again and
folding it onto a single 128x128 cell. Flips and transposes of the image are flips and transposes of that cell, so
mirrored and 90-degree-rotated copies cost almost nothing to test. The cell is correlated with the sync tile
(whitened, so busy low frequencies stop drowning the mark) to find scale, orientation and shift. At that alignment it
is correlated with every candidate's id tile and each score is compared with a null built from 1,500 decoy names.

Recipients are found by *matching* against a list, not by decoding a payload. That is why it tolerates so much damage,
and why you need to know who might have the file: the ledger from `issue` supplies the list, and `trace
--candidates names.txt` adds more.

**Strength is chosen per master.** The same mark that is invisible on foliage shows on a flat sky and is erased from
it by the encoder, so `issue` defaults to `--strength auto`: it marks a typical 5 s excerpt at a few strengths, pushes
each through recompression, half-size and screen-capture panels, traces it, and keeps the weakest strength that
clears the detection thresholds with margin. Strength is the RMS of the added pattern in 8-bit code values; for
scale, a plain x264 re-encode at crf 18 already costs about 0.7 to 1.7 depending on the picture.

## Courses: one render per lesson, not one per student

`issue` costs about one real-time render per recipient, which is hopeless for 300 students and 40 lessons. `course`
renders each lesson **twice** (variant A and variant B, in 5 s segments that each start on a keyframe), then builds a
student's copy by taking every segment from A or from B according to a bit string derived from your key, the student and
the lesson, joined with `ffmpeg -c copy`. A copy is a file concatenation: well under a second. The source's own audio
is kept as it is (it is not marked).

```
invisimark course issue lessons/*.mp4 --roster students.txt --out copies/    # copies/<student>/<lesson>.mp4
invisimark course trace leaked-lesson.mp4 [--trim SECONDS] [--roster more.txt]
```

Rendered segments stay in `copies/.work`, so an interrupted run resumes, and a student added later costs seconds.
Course copies are recorded in `course-ledger.jsonl` beside the ledger; plain `trace` knows nothing about them.

**How a leak is read.** The file is aligned once (scale, crop, mirroring) by the ordinary detector. Each 5 s window is then
scored against that lesson's A and B tiles, which gives one bit per segment, or nothing when the window is unreadable.
Because each student's bits come from an HMAC, an unrelated student matches each clear segment with probability exactly one
half, so the report is an exact binomial tail rather than a tuned threshold:

```
lesson lesson3_phone: sync z 14.3, scale x0.999; read 51 of 51 segments
  alice                          agrees on 48 of 48 clear segments; an unrelated student would do that by luck 1 time in 281,474,976,710,656
  dave                           agrees on 30 of 48 clear segments; an unrelated student would do that by luck 1 time in 18
  erin                           agrees on 25 of 48 clear segments; an unrelated student would do that by luck 1 time in 2
  carol                          agrees on 23 of 48 clear segments; an unrelated student would do that by luck 1 time in 1
  frank                          agrees on 23 of 48 clear segments; an unrelated student would do that by luck 1 time in 1
=> traced to: alice
```

Someone is named when the odds of *any* innocent student on the roster doing as well are 1 in a million or better. Short of
that you get a "weak lead" with its odds. Two students who swap segments both stand out.

**Measured** (`bench/course_eval.py`; one 4-minute, 720x1558 screen recording, 51 segments, six students, strength 2.6,
which the tool flags `WEAK` on this flat footage). "Named" means the right student, with no one else near:

| leak | result |
|---|---|
| as delivered | named, 1 in 5x10^13 |
| x264 crf 28 | named, 1 in 3x10^9 and 1 in 1x10^10 (two students) |
| crf 34 | weak lead on the right student, 1 in 3x10^4 and 1 in 6x10^4 (two students) |
| 75% size + crf 28 | named (1 in 4x10^8) for one student, weak lead (1 in 4x10^5) for the other |
| first 12 s cut off, crf 23, `--trim 12` | named, 1 in 7x10^11 and 1 in 2x10^11 |
| true leaker not on the roster | nobody named (best unrelated student 30 of 48) |
| unmarked original, unrelated video | no mark found |
| 300-student roster | alice named, 1 in 9x10^11 |

**What it costs.** Rendering the 4-minute lesson in both variants took 3 min 35 s on 16 cores (about 0.85 s per second of
video, and it does not grow with the number of students). Copies came out 6 to 76 percent larger than their sources
(18 MB to 19 MB for camera footage, 17 MB to 30 MB for the screen recording): the mark is noise, and noise is not free to
encode. The segments take about 110 MB for three lessons.

**What it does not do.**
* **It needs enough footage.** To name someone among six students you need about 23 clear segments agreeing, among 300 about
  29: two to two and a half minutes of a leaked lesson at 5 s. A 50 s leak can only give a weak lead (1 in 341 for six
  students). Shorter segments help evidence per minute and cost bitrate; there is no setting here that beats having more
  of the lesson. Evidence from several leaked lessons is not yet combined.
* **A trimmed start needs `--trim`.** There is no automatic search for the cut point yet. A leak cut mid-segment still reads
  from the segments it fully contains.
* **Flat footage is the weak case**, as above, and so are screen recordings of the player and phone filming a screen:
  untested in this mode.
* **Variable frame rate sources** are re-timed to constant, so segment edges can drift by a frame. Convert first.
* **Collusion** is only handled in the simple case: students who swap segments are all named. Students who average their
  copies, or many colluders, are not modelled.

## Marking AI-generated images and video

A second use, separate from tracing leaks: mark what an AI system generates, so anyone can check it. It follows the two layers in Measure 1.1 of the EU Code of Practice on Transparency of AI-Generated Content: an imperceptible watermark from a published key, and a signed, time-stamped C2PA manifest.

```
$ invisimark mark output.png marked.png --cert chain.pem --signing-key key.pem --provider "Acme" --system "AcmeGen"
marked.png: watermark at strength 1.75, signed

$ invisimark check marked.png
Watermark: this file carries the public Invisimark AI-marking watermark. ...
Metadata: a C2PA manifest signed by Acme. The signature is intact, ...
```

`invisimark dev-cert` makes a throwaway certificate for trying this out. `invisimark report-key`, `check --report-key` and `verify-report` produce and verify signed detection results. The format is in `docs/spec.md`, so anyone can write their own detector.

Measured at the default strength (`bench/results/ai.md`, PSNR about 43 dB), on 12 photographs at 1024 px:

| change | detected |
|---|---|
| untouched, JPEG down to quality 30, WebP, half size, 1.5x, chat-app recompression, screenshot, contrast, blur, grain, mirror, 90-degree turn, blurred region, crops to 80% and 60% | **12 / 12** each |
| tilts of 1, 3, 5 and 10 degrees; one axis stretched to 4:3 or 16:9 | **12 / 12** each |
| stacked worst case | 11 / 12 |
| **tilt of 25 degrees; tilt plus half size; both axes scaled differently** | **0 / 12** |

The search for tilts covers up to 15 degrees at 0.7x to 1.3x size, and one-axis stretches from 0.45x to 2.2x. It adds about 8 seconds to any check that does not find a mark straight away.

**Against someone trying to defeat it** (`bench/results/ai-adversarial.md`, 12 images):

| attack | public mark still detected | private tile still verifies |
|---|---|---|
| **subtract the published tile** (53.9 dB, effectively invisible) | **0 / 12** | 12 / 12, when the file has one |
| lift the pattern from the file and subtract it, no key needed (42.8 dB) | 7 / 12 | 7 / 12 |
| **copy the pattern onto another image** | **11 / 12: a false "marked"** | **12 / 12: a false "verified"** |
| heavy denoising and resampling, as a stand-in for regeneration | 11 or 12 / 12 | not measured |

So the public mark helps with accidental loss, such as recompression, not with someone determined. Its key is published, which is what lets anyone check it and also what makes removal this easy. No generative model was run, so real regeneration may do much better than the stand-ins.

**False positives** (`bench/results/ai-null.txt`): none on 768 variants of 24 unmarked images. The highest score was 6.81 against a threshold of 8.0. The variants are not independent, so read the 0.39% upper bound as optimistic.

Other limits:

* The corpus is natural photographs, not model output.
* The mark says "marked as AI", not who made the file. Only signed metadata shows origin, and most platforms strip it. A private tile added with `--provider-key` lets a provider recognise its own output after the public tile is subtracted, but it can be copied too, so it doesn't prove origin. It holds up about as well as the public mark against everyday changes, though it's weaker on heavy crops and stacked changes (`bench/results/ai-provider.md`).
* Audio and text are not covered.

Using this does not by itself make anyone compliant with the AI Act.

## Install

Python 3.13, with `ffmpeg` and `ffprobe` on the path.

```
uv tool install .          # from a checkout; gives you the `invisimark` command
# or
uv sync && uv run invisimark --help
```

## Using it

```
invisimark keygen                                   # once; back up the key file
invisimark issue FILE --to alice --to bob           # or --to-file names.txt
invisimark issue FILE --to alice --strength subtle  # auto | subtle | standard | strong | an RMS number
invisimark trace LEAKED [--candidates names.txt] [--json]
invisimark inspect FILE                             # is this marked with my key at all?
invisimark serve                                    # local web app on 127.0.0.1
invisimark course issue|trace ...                   # whole courses, see below
invisimark mark IN OUT [--cert C --signing-key K]   # mark AI output, see below
invisimark check FILE                               # look for AI marks
invisimark host                                     # public, anonymous, capped service: docs/self-hosting.md
```

`trace` exits 0 when it names someone, 2 when the mark is present but nobody can be named, 1 when there is no mark.

The key lives at `~/.config/invisimark/key` (override with `--key` or `$INVISIMARK_KEY`), mode 600. **Back it up.**
Without it nothing can be traced. Anyone with it can mark files and accuse, so keep it private. The ledger
(`ledger.jsonl` beside it) records who got which file when, with hashes; it is a convenience, the evidence is the
mark. Labels are exact and case-sensitive: `Bob` and `bob` are different recipients.

The web app binds to 127.0.0.1, requires a per-launch token on every request (so another page open in your browser
cannot drive it), keeps uploads in a temporary folder that is cleared when you stop it, and never talks to the network.

## Using it responsibly

Mark files you have the right to share, and tell recipients their copy is individually marked; that is most of the
deterrent. The mark itself holds no personal data, but the ledger does (names, dates), so treat it as personal data.

## Development

```
uv run pytest                                  # 59 tests, about three minutes
uv run python bench/attacks.py CLIP.mp4        # the 27-attack video matrix, strength auto
uv run python bench/images.py STILL.png        # the 17-attack image matrix
uv run python bench/null_study.py CLIP.mp4     # how high does the statistic get when there is nothing to find
```

| file | job |
|---|---|
| `tile.py` | keyed tiles and the mark's shape |
| `embed.py` | adds the mark to an image or, through ffmpeg pipes, a video |
| `detect.py` | scale, orientation and shift recovery; candidate scoring; thresholds |
| `calibrate.py` | closed-loop strength |
| `course.py` | A/B segment marking for sets of lessons: render, assemble, read, score |
| `service.py`, `static/site/` | the public service: capped anonymous jobs, case files, pages (see `docs/self-hosting.md`) |
| `ai.py` | AI marking: the public mark, C2PA manifests, checks and signed results (see `docs/spec.md`) |
| `keys.py`, `cli.py`, `web.py` | key and ledger, command line, local app |

The shape of the mark was tuned by comparing variants at equal visible distortion, by hand, during development; that
comparison is not reproduced here. What helped most: giving each tile an equal-magnitude spectrum with random phases
(a Gaussian spectrum leaves some keys with a poor matched filter; this lifted typical sync scores by 10-50% and made
small pictures workable), putting the sync tile in the low bands, and weighting the mask toward texture. Things tried
that did not help: a lower-band id tile for flat content (a little
better, not enough), more sampled frames (the noise is not independent across frames), and a scale search on the
power spectrum's lattice lines (it needs roughly 3.4x more mark than coherent matching).
