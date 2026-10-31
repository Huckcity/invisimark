# Self-hosting the service

`invisimark host` serves a free, anonymous, capped version of the tool: a page where a creator uploads a photo or
video and a list of recipients, gets back one marked copy per recipient plus a **case file**, and later uploads a leak
and the case file to find out whose copy it was. The code is `src/invisimark/service.py`; the pages are
`src/invisimark/static/site/`.
It also serves the AI marking checker at `/check`.

```
invisimark host --host 0.0.0.0 --port 8080        # or the Dockerfile in the repo root
```

## What it keeps, and what it forgets

| thing | where | for how long |
|---|---|---|
| the upload | `$IM_WORK/<job>/in` | deleted when the job finishes or fails (unless test mode is on, below) |
| a checker result: the file's hash and what was found | `$IM_WORK/<job>/` and in memory | `IM_TTL_SECONDS` |
| the copies (a zip) and the case file | `$IM_WORK/<job>/` | `IM_TTL_SECONDS` (30 minutes), or until the user presses delete |
| the secret key | inside the case file; in memory while the job runs | same as the copies. It is **never** stored on its own, and the service cannot recover it |
| the client's address | the job record, in memory | until the job is deleted. Used only to allow one active job per client |
| anything after a restart | nothing | job folders are wiped at start-up (only folders named like job ids) |

There are no accounts, no database and no cookies of its own. Run uvicorn without an access log (the `host` command
already does), or the client address will be written to your logs. What the `host` command does write to stdout is one
line per job event — `started`, `mark accepted`, `mark done`, `trace done`, `mark rejected`, `… results expired and removed` —
with the media kind, size, recipient count, duration and the first six characters of the job id. Those lines never hold a
client address, a file name, a recipient's name or a whole job id (`service.record`; pinned by a test), so they are safe
to ship to a log collector. Point `IM_WORK` at a dedicated scratch directory or a
tmpfs: it is for jobs only. Uploads are spooled by the web server into the system temp directory while they arrive, so
`/tmp` needs room for a couple of maximum-size uploads at once.

**Run exactly one process.** The queue and the job table live in memory. Several workers or replicas behind one address
would each know different jobs and the page would see "no such job".

## Test mode: keeping uploads

While testing, run `invisimark host --keep-uploads` (or set `IM_KEEP_UPLOADS=1`). Every upload, including ones the service
refuses and ones whose job fails, is copied to `IM_KEEP_DIR` (default `~/.local/share/invisimark/kept`) in a folder named
`<time>-<mark|trace>-<job id>`, mode 700, with a `meta.json` saying what happened to it. Kept uploads outlive the job: pressing
delete, the expiry sweep and a restart never touch them, so clear that folder yourself when you are done. Finished jobs
(the copies and case file) are kept 24 hours instead of 30 minutes unless `IM_TTL_SECONDS` says otherwise.

The privacy page changes with the mode so it never claims something untrue. Files sent to `/check` are never kept, even in this mode. **This mode is for
you testing your own files. Do not leave it on for strangers**: it holds their uploads, including anything sensitive, in
plain files on disk.

## Settings (environment variables)

| variable | default | meaning |
|---|---|---|
| `IM_WORK` | a new temp dir | where jobs live |
| `IM_MAX_MB` | 300 | largest single upload |
| `IM_MAX_WORK` | 20000 | cap on a video's size as megapixels x frames. 1080p30 is 62 per second, so 20000 is about 5.4 minutes; 720p30 gets about 12 minutes |
| `IM_MAX_IMAGE_PIXELS` | 50,000,000 | largest photo |
| `IM_MAX_RECIPIENTS` | 100 | per run |
| `IM_MAX_QUEUE` | 6 | jobs waiting or running; beyond that the service answers 503 |
| `IM_TTL_SECONDS` | 1800 | how long finished jobs are kept |
| `IM_RENDER_JOBS` | half the cores | parallel segment renders for one video |
| `IM_COPIES_MODE_MAX` | 8 | up to this many recipients, a short video is marked per recipient |
| `IM_KEEP_UPLOADS` | off | `1` keeps every upload (see above) |
| `IM_KEEP_DIR` | `~/.local/share/invisimark/kept` | where kept uploads go |
| `IM_NOTIFY_URL` | none | where to send one-line "an upload arrived" alerts: an ntfy topic URL, a Discord or Slack webhook, or any URL that takes a POST body. Treat it as a secret |
| `IM_NOTIFY_DETAIL` | off | `1` adds the uploaded file's name to alerts. It goes to whoever runs that service |
| `IM_ALLOW_MANUAL_STRENGTH` | off | the public page always calibrates the mark per file and per key. A fixed strength skips that, and a weak random key can then fail even on an untouched copy (seen live) |
| `IM_PUBLIC_URL` | none | canonical origin, e.g. `https://invisimark.app` (canonical tags, sitemap, noindex elsewhere) |
| `IM_CONTACT` | none | the address shown for abuse reports |
| `IM_REPORT_KEY` | none | private key from `invisimark report-key`, for signing detection results. A secret |
| `IM_TRUST_ANCHORS_FILE` | none | PEM roots, such as the C2PA trust list. Without it, no signer is reported as trusted |
| `IM_SOURCE_URL` | none | link to the public source code, shown on the AI marking page |
| `IM_AD_TOP`, `IM_AD_BOTTOM` | none | raw HTML for an ad slot, or `IM_AD_TOP_FILE` / `IM_AD_BOTTOM_FILE` for a file holding it |

Also: the client address is read from the `fly-client-ip` header when present (the Fly proxy sets it), otherwise from the
socket. Behind any other proxy, make sure it sets a header you trust, or every user will look like one client and the
one-job-per-client rule will lock them out of each other. This is not configurable yet.

## Sizing

Marking a video costs about **0.4 CPU-seconds per megapixel-frame, for both variants together**, measured on a 4-minute
720x1558 screen recording (3 min 35 s of wall time on 16 cores). So a job at the default cap (20,000) is about 8,000
CPU-seconds: roughly 9 minutes on 16 cores, 35 on 4. It does not grow with the number of recipients (that is the point
of the A/B method); copies come from a lossless join and take well under a second each. Tracing is much cheaper:
about 40 seconds for that same lesson.

Because one heavy job runs at a time, the queue wait is the product's real cost. Lower `IM_MAX_WORK` before you add
cores: halving it halves the longest job. Images take a few seconds each.

## Signing detection results

`invisimark report-key --out report-key` makes the key, and `IM_REPORT_KEY` gives it to the service. Without it the checker works but offers no signed result. The public half is served at `/api/report-key`, which is how people verify results, so changing the key strands old ones. Back it up.
