# PERFORMANCE_FINAL.md

Final measured state after the performance changes.

- Date: 2026-10-26
- Branch: `feature/mvp-0-3`, base commit `b4fc6b6`
- Corpus: 517 real JPEGs, 688 MiB, mostly 22 MP
- Environment: Xeon E5-2670 v3 (24 threads), 15.8 GiB RAM, GTX 1060 6 GB,
  Windows 10 19045, Python 3.13.14
- Tests: **392 passed, 5 skipped**

## 1. Headline

48-file 22 MP-class subset, `web_1024`:

| Backend | Before | After | Speedup |
|---|---:|---:|---:|
| CPU | 2.45 files/s | 10.03 files/s | 4.1x |
| GPU | 3.64 files/s | 10.26 files/s | 2.8x |

`large_1600`:

| Backend | Before | After | Speedup |
|---|---:|---:|---:|
| CPU | 2.59 files/s | 10.39 files/s | 4.0x |
| GPU | 3.51 files/s | 9.98 files/s | 2.8x |

"Before" uses the original code at 1 GiB. "After" uses the adaptive default,
2 GiB on this host. At the same 1 GiB budget the code-only CPU result is
8.42 files/s (3.4x), so the budget change contributes the remaining ~18%.

Full 517-file pass:

| Backend | Elapsed | Wall | Throughput | Failures |
|---|---:|---:|---:|---:|
| CPU | 48.2 s | 49.1 s | 10.72 files/s | 0 |
| GPU | 51.3 s | 57.2 s | 10.08 files/s | 0 |

CPU and GPU are effectively tied at `web_1024` on this Pascal card. The GPU
still offloads the CPU and is much faster on decode-heavy and complex inputs;
its 5.9 s CUDA startup hurts small batches.

## 2. Stage breakdown

Per-file averages, 48 files. The stages run concurrently, so the numbers are
not additive.

| Backend / profile / budget | Read | Decode | Resize | Encode | Write | Elapsed | files/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU 1024, legacy 1 GiB | 575.8 | 492.7 | 811.6 | 21.7 | 536.3 | 19.598 | 2.45 |
| CPU 1024, new 1 GiB | 16.4 | 393.5 | 20.9 | 4.0 | 9.8 | 5.704 | 8.42 |
| CPU 1024, new 2 GiB | 23.1 | 589.5 | 42.7 | 7.6 | 20.2 | 4.785 | 10.03 |
| CPU 1600, legacy 1 GiB | 642.0 | 406.2 | 828.9 | 29.2 | 503.4 | 18.530 | 2.59 |
| CPU 1600, new 2 GiB | 18.9 | 546.1 | 57.2 | 15.7 | 22.6 | 4.619 | 10.39 |
| GPU 1024, legacy | 466.0 | 110.1 | 119.4 | 3.4 | 42.7 | 13.188 | 3.64 |
| GPU 1024, new | 13.8 | 81.6 | 1.03 | 3.3 | 19.2 | 4.679 | 10.26 |
| GPU 1600, new | 13.7 | 79.1 | 1.10 | 8.8 | 5.2 | 4.808 | 9.98 |

The legacy read and write figures are inflated by the Python truncation scan
holding the GIL; the new reader removes that serialization. The genuine CPU
resize improvement is 811.6 ms to 20.9-43.3 ms per file.

After the fixes, CPU decode is the largest default cost (394-580 ms at 1024,
546 ms at 1600). At 2 GiB, more concurrent decodes contend for memory
bandwidth, which raises per-file decode time while still raising throughput.

## 3. Scaling

48 files, CPU:

| Budget / workers | files/s | Peak RSS |
|---|---:|---:|
| 1 GiB / 8 | 8.42 | 816 MiB (24-file run) |
| 2 GiB / 8 | 10.03 | 1,522 MiB (24-file run) |
| 2 GiB / 16 processors | 10.76 | not measured |
| 3 GiB / 16 processors | 10.51 | not measured |

96 files, CPU:

| Configuration | files/s |
|---|---:|
| Default 8 workers | 9.96 |
| 16 processors, 4 readers, 4 writers | 9.95 |
| 8 workers, `--fast-jpeg-decode` | 15.62 |

96 files, GPU: 10.37 files/s at default settings.

GPU tuning on 48 files:

| Configuration | files/s |
|---|---:|
| Default batch 8, 8 workers | 10.26 |
| Batch 16 | 10.37 |
| Batch 16, 3 GiB VRAM reserve | 10.22 |
| 12 workers, 3 repeats | 10.40 |

There is no consistent gain from more workers or a larger batch. The default
configuration is retained.

## 4. Complex inputs

Four files per category, four workers; throughput in files/s.

| Category | CPU before | CPU after | GPU after |
|---|---:|---:|---:|
| Gray JPEG | 4.10 | 5.88 | 13.69 |
| CMYK JPEG | 2.65 | 4.28 | 6.06 |
| Alpha PNG | 3.86 | 6.86 | 9.68 |
| Alpha WebP | 3.83 | 6.61 | 9.25 |
| EXIF orientation | not measured | 6.00 | 10.01 |

Quality:

- Gray and alpha outputs: pixel-identical to the old path.
- CMYK: 46.8-46.9 dB, about one 8-bit level.
- Orientation: stored pixels remain 1024x683 with EXIF 274=1 and display as
  683x1024 portrait.

## 5. Optional paths

48 files unless noted.

| Option | files/s | Dominant cost |
|---|---:|---|
| default | 10.03 | decode |
| `--mode cover` | 10.23 | decode |
| `--strip-personal-data` | 10.07 | no measurable overhead |
| `--sharpen standard` | 8.82 | resize 137-147 ms/file |
| `--web-srgb` | 10.16 | decode 563-582 ms/file |
| `--linear-light` (6 files) | 3.28 | resize 617-736 ms/file |
| `--fast-jpeg-decode` (96 files) | 15.62 | quality 38.0-42.4 dB |
| 1024 + `thumb:512x512` | 10.31 jobs/s | 20.6 outputs/s |

The last row shows that an extra small variant is almost free once the source
is decoded.

## 6. Quality

PSNR against the previous output, not against a lossless source:

| Case | PSNR |
|---|---:|
| contain 1024 | 39.02 dB |
| contain 1600 | 42.53 dB |
| cover 1000 | 42.03 dB |
| cover 1234 | 44.24 dB |
| cover 1600 | 47.49 dB |
| area 6 MP crop | 87.8 dB |
| mitchell via cubic | 48.4 dB |
| nearest | 38.8 dB |
| fast JPEG decode | 38.0-42.4 dB |
| CMYK after CMS shortcut | 46.8-46.9 dB |
| linear-light vs reference | 41.0-45.7 dB |
| linear-light alpha vs reference | 53.9 dB |
| web-srgb vs old untagged path | bit-identical |
| gray / alpha | exact |

All 517 CPU outputs and all 517 GPU outputs pass `Image.verify()` and have the
expected dimensions.

## 7. Remaining problems

### P2: `--linear-light` residual cost

The new float path is 15x faster (0.22 to 3.28 files/s) and matches the
reference at 41.0-45.7 dB on six files. It is still CPU-only and slower than
the default because linearization is a full-frame float operation. The GPU
path falls back for linear-light.

### P2: `--web-srgb` with ICC profiles

Untagged RGB/RGBA is now 10.16 files/s and bit-identical to the old output.
ICC-tagged inputs still take the Pillow/littleCMS path. That path can be
optimized by caching transforms per profile, but skipping by profile name is
unsafe.

### P2: CPU decode

JPEG decode is now the default bottleneck: 560-670 ms per 22 MP file at
`web_1024`, 546 ms at `large_1600`. `--fast-jpeg-decode` reaches 15.62
files/s but loses 2-5 dB PSNR; it should stay opt-in unless the user accepts
that trade-off.

### P2: CPU EXIF orientation and sharpening

Orientation normalization costs 77-126 ms per file. `--sharpen standard` now
adds 137-147 ms of resize after removing the accidental CUDA initialization;
the first sharpened file fell from 1,076 ms to 47 ms. Rotating after resize
would remove the orientation pass but needs careful geometry work for
orientations 5-8, so it is deferred.

### P3: GPU startup

A 517-file GPU run spends 5.9 s more in wall time than pipeline time. For
small batches (under ~50 files) CPU is usually the better default on this
machine.

### P3: Reporting and documentation

The old performance text in `REPORT.md` (46-62 files/s, 1.8-2.2x) was measured
on small synthetic images and does not describe this corpus. `README.md`'s
test count now matches at 392 / 5, but throughput claims should always name
corpus, resolution, backend and cache state.

The original correctness findings AUD-001, AUD-002 and AUD-003 are outside
this performance work and remain open.

## 8. Recommendations

1. Keep the current defaults: OpenCV resampling, adaptive 2 GiB/25% budget,
   eight workers, Pillow alpha compositing, and the narrowed CMS condition.
2. Keep `--fast-jpeg-decode` opt-in and document its quality cost.
3. Keep the new float linear-light path and the untagged web-srgb shortcut;
   both have measured quality checks. Document the linear-light `INTER_AREA`
   choice for large reductions.
4. Use `--working-mb` when deterministic memory consumption matters. The
   adaptive default remains.
4. Add `benchmark/run_bench.py` to the release checklist and record the
   corpus and environment with every result.
5. Do not spend more effort on GPU batch/worker tuning for this card; the
   plateau is real.
6. When Tkinter is available, measure GUI startup and preview latency; the
   static review found no event-loop bottleneck.
7. Consider exposing the working budget as a documented option for users who
   need deterministic memory use, while keeping the adaptive default.

## 9. Reproduction

```powershell
# Generate the deterministic synthetic corpus (optional)
.\.venv\Scripts\python.exe benchmark\make_corpus.py

# Current code, real corpus, CPU
.\.venv\Scripts\python.exe benchmark\run_bench.py --label final-cpu `
  --input K:\Test\recoder\input --repeats 3 --warmup 1 `
  -- --backend cpu --workers 8 --size web_1024

# GPU
.\.venv\Scripts\python.exe benchmark\run_bench.py --label final-gpu `
  --input K:\Test\recoder\input --repeats 3 --warmup 1 `
  -- --backend gpu --workers 8 --size web_1024

# Full suite
.\.venv\Scripts\python.exe -m pytest tests -q -ra
```

Legacy baselines require the temporary `legacy_runner.py` and
`budget_runner.py` described in `PERFORMANCE_BASELINE.md`; they are not part of
the repository because they patch old behavior.

## 10. Conclusion

The default pipeline is now about 4.1x faster on CPU and 2.8x faster on GPU
for 22 MP camera files, and it processes the full 517-file corpus at roughly
10.7 files/s with zero failures. The previously blocking optional paths are
now usable: linear-light is 15x faster and web-srgb is 2.1x faster, with
quality checked against the reference. Remaining costs are CPU JPEG decode,
full-frame EXIF rotation, and ICC-tagged web-srgb; the default path is
covered by the full test suite and output-quality checks.
