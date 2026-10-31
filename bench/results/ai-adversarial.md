Public key. Strength RMS 1.75. Longest side 1024 px. 2026-10-31. No generative model was run: "regenerate" rows are denoising and resampling stand-ins.

| attack | still detected | mean PSNR against the original or target | private tile still verifies |
|---|---|---|---|
| subtract x0.5 | 11 / 12 | 48.1 dB | - |
| subtract x1 | 0 / 12 | 53.9 dB | - |
| subtract x1.5 | 1 / 12 | 47.9 dB | - |
| subtract x2 | 8 / 12 | 42.9 dB | - |
| regenerate nlmeans | 11 / 12 | 34.9 dB | - |
| regenerate bilateral | 12 / 12 | 37.4 dB | - |
| regenerate median5 | 12 / 12 | 34.2 dB | - |
| regenerate quarter_and_back | 12 / 12 | 31.7 dB | - |
| regenerate blur3_sharpen | 11 / 12 | 28.9 dB | - |
| regenerate noise12_median3 | 12 / 12 | 32.3 dB | - |
| subtract public tile from a file with a private tile | 0 / 12 | 46.5 dB | 12 / 12 |
| lift the whole pattern from one file and subtract it | 7 / 12 | 42.8 dB | 7 / 12 |
| copy public mark onto another image | 11 / 12 | 42.8 dB | - |
| copy mark and private tile onto another image | 11 / 12 | 40.6 dB | 12 / 12 |
