# Public AI-marking watermark: `invisimark-public-v1`

This is enough to write a detector without using this code. `src/invisimark/tile.py`, `embed.py` and `detect.py` are the reference implementation, and `invisimark check` is the reference detector.

## What the mark means

A file carrying this watermark was marked by someone as AI-generated or AI-manipulated. It does not say who marked it. The key is public, so anyone can detect the mark, copy it onto another file or subtract it almost perfectly (`bench/results/ai-adversarial.md`). Its absence says nothing about how a file was made.

## The key

```
key   = SHA-256("invisimark.ai-marking.v1")
label = "ai-generated"
```

## The tile

The pattern repeats every 128 x 128 pixels. Each component is built like this:

1. `seed` = the first 8 bytes of `HMAC-SHA256(key, name)`, read as a little-endian integer.
2. Draw 128 x 128 standard normals (32-bit floats) from NumPy's `Generator(PCG64(seed))`.
3. Take the real 2-D FFT, and divide each component by its magnitude.
4. Multiply by a raised-cosine band on radial frequency `r` in cycles per tile: `up = clip(r - (lo - 1), 0, 1)`, `down = clip((hi + 1) - r, 0, 1)`, `band = (0.5 - 0.5 cos(pi up)) (0.5 - 0.5 cos(pi down))`, with the DC term set to 0.
5. Invert the FFT, subtract the mean and divide by the standard deviation.

| component | `name` | `lo`, `hi` | weight |
|---|---|---|---|
| sync | `sync` | 1.5, 8 | 2.0 |
| id | `id:ai-generated` | 3, 18 | 1.8 |

The public tile is `2.0 sync + 1.8 id`.

A provider may add a private tile, `2.0 x` the id tile built from its own secret key with the name `id:provider`. Only the holder of that key can test for it, and it does not affect a public detector beyond lowering the public scores. It survives subtraction of the public tile, but it can be copied with the rest of the pattern, so it does not prove origin.

## Embedding

Tile the pattern from the top-left pixel and multiply it by the activity mask in `embed.py`, which puts more energy in textured areas. Scale it to a chosen RMS in 8-bit code values, then add it to all three colour channels of an image or to the luma plane of a video.

## Detecting

1. Band-pass and soft-clip each sampled frame, then average the frames.
2. For each candidate scale from 0.3 to 3.2, resample so the period is 128 pixels and fold the picture onto one 128 x 128 cell.
3. Whiten the cell and correlate it circularly with the public tile, in all eight orientations (flips and transposes). Take the peak's z-score as the alignment score. If it is below 8.0, repeat over tilts up to 15 degrees and one-axis stretches (`detect.geometry_grid`), then undo the best fit and score again.
4. At the best alignment, remove the sync component and correlate with the id tile. Convert that to a z-score against 1,500 decoy tiles.5. Report the mark as present only if the alignment score reaches 8.0. The id score is informational: the search already used the id tile to choose the alignment, so it is not independent evidence, and unmarked images have reached 6.8 on it. A private tile is searched for on its own and judged by the same rule.

`bench/results/ai-null.txt` has the measured false-positive rate at this threshold.

## Signed metadata

`invisimark mark` attaches a C2PA manifest, signed last so that its hash covers the final bytes. It declares the file with the IPTC digital source type `trainedAlgorithmicMedia` (generated) or `compositeWithTrainedAlgorithmicMedia` (manipulated). An `app.invisimark.marking` assertion records the watermark scheme and any provider, system and model names supplied.

Those names are the signer's own statement. A valid signature shows the manifest is unchanged, and the signer's identity is only verified when the certificate chains to a trust list the reader has loaded.

## Signed detection results

A result is JSON with four fields: `result`, `alg` (`Ed25519`), `public_key` and `signature`. The signature covers `result` serialised with sorted keys, separators `,` and `:`, in UTF-8 with non-ASCII characters left unescaped. `result` holds the file's SHA-256, the detector's identifier, a UTC timestamp and the findings.

The embedded `public_key` only shows the result is intact. To show who signed it, compare it with the key the detector publishes at `/api/report-key`.

## Known limits

The detector does not recover the mark after a tilt beyond about 15 degrees, a tilt combined with a large resize, or both axes scaled differently. Deliberate removal and copying succeed. Benchmarks are in `bench/results/ai.md` and `bench/results/ai-adversarial.md`.
