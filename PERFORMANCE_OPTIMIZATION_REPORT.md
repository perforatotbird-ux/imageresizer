# PERFORMANCE_OPTIMIZATION_REPORT.md

Changes made during the performance audit, with rationale, evidence and
correctness checks.

- Date: 2026-10-26
- Branch: `feature/mvp-0-3`, base commit `b4fc6b6`
- Full suite after the final change: **392 passed, 5 skipped**
- Reference corpus: 517 JPEGs, 688 MiB, mostly 22 MP

## 1. Summary

| # | Change | Location | Main effect |
|---|---|---|---|
| 1 | C-level scan for entropy bytes in JPEG truncation check | `batchresize/imageio_meta.py:58` | 142.39 ms to 1.27 ms per MiB; removes the reader GIL bottleneck |
| 2 | OpenCV resampling kernels and rounded whole-pixel regions | `batchresize/resampling.py:173`, `:203`, `:241` | 22 MP resize 599 ms to 16-20 ms; sub-pixel cover crops stay at 11-29 ms instead of a 5-6 s reference fallback |
| 3 | Adaptive working budget, default 2 GiB capped at 25% RAM | `batchresize/pipeline.py:38`, `:95` | 4 to 8 concurrent 22 MP jobs on this host |
| 4 | Pillow `alpha_composite` fast path | `batchresize/cpu_backend.py:62` | 3.2 MP composite 263-275 ms to 62-101 ms, exact match |
| 5 | Skip CMS decode for L/CMYK/RGBA without a relevant ICC | `batchresize/pipeline.py:1057` | Gray/CMYK/alpha decode 1.4-4.2x faster |
| 6 | Float linear-light resampler through OpenCV + `cv2.LUT` | `batchresize/resampling.py:238`, `batchresize/color_management.py` | 6-file throughput 0.22 to 3.28 files/s; resize 13.2 s to 0.62-0.74 s |
| 7 | Fast untagged RGB/RGBA path for `--web-srgb` and `--linear-light` | `batchresize/pipeline.py:1063` | web_srgb 4.83 to 10.16 files/s; decode 1.38 s to 0.57 s |
| 8 | No CuPy import on the CPU sharpen/pad path | `batchresize/sharpen.py:113`, `batchresize/geometry.py:243` | First sharpen call 1.08 s to 47 ms; 8.33 to 8.82 files/s |
| 9 | CLI `--working-mb`; GUI undefined `raw` fix | `batchresize/cli.py`, `batchresize/gui.py:884` | Explicit memory control; GUI hint no longer raises |

## 2. Reader: `is_truncated_jpeg`

### Change

The entropy-coded segment is no longer walked one Python byte at a time. The
scanner calls `data.find(b"\xff", i + 1)` to jump directly to the next marker
candidate and only inspects the marker structure at that point. The segment
length variable was renamed from `length` to `seg_len` to avoid shadowing.

The function still checks only marker structure. It is not proof of bitstream
or pixel integrity; the pre-existing caveat is unchanged.

### Evidence

| Case | Before | After |
|---|---:|---:|
| 1 MiB JPEG | 142.39 ms | 1.273 ms |
| All 517 corpus files | - | 1.297 s total, 2.5 ms/file |
| Mismatches vs old scanner | - | 0 / 517 |

The old scanner also held the GIL for the whole scan. With eight reader
threads, that turned into a serial section that starved the GPU. The reader
stage dropped from 575.8 ms to 16-22 ms on CPU and from 466.0 ms to about
14 ms on GPU.

## 3. CPU resampling

### Change

`resample_cpu` now tries OpenCV first for full-frame and whole-pixel crops:

| Filter | OpenCV mapping |
|---|---|
| `lanczos3` | `INTER_LANCZOS4` |
| `mitchell` | `INTER_CUBIC` |
| `area` | `INTER_AREA` |
| `bilinear` | `INTER_LINEAR` |
| `box` | `INTER_AREA` |
| `nearest` | exact NumPy gather on the pixel region |

`BATCHRESIZE_RESAMPLE_ENGINE=reference` still selects the old separable NumPy
engine; `pillow` selects the previous Pillow path for A/B comparison.

Sub-pixel crops are rounded to a whole-pixel source region in
`_pixel_region`. A guard permits at most half a pixel of float geometry error
at each edge, clamps that error to the source bounds, and returns `None` for a
genuinely out-of-bounds plan so the reference engine handles it.

`area` and `box` fall back to the reference engine when upscaling, preserving
the previous semantics instead of silently changing the filter.

### Evidence

| Operation | Before | After |
|---|---:|---:|
| 22 MP to 1600, single image | 599 ms (Pillow) | 16-20 ms (OpenCV) |
| 22 MP to 1600, NumPy reference | 5,233-6,192 ms | retained as fallback |
| Batch resize, 48 files, 1 GiB | 811.6 ms/file | 20.9 ms/file |
| Batch resize, 48 files, 2 GiB | 811.6 ms/file | 43.3 ms/file |
| Cover crop, sub-pixel plan (Pillow before) | 599 ms/image | 11-29 ms/image |
| Cover crop if fast path rejected the plan | 5.2-6.2 s/image (reference) | 11-29 ms/image |

The 2 GiB resize number is higher than 1 GiB because twice as many jobs
compete for memory bandwidth; total throughput still improves.

### Quality

PSNR against the previous Pillow output:

| Case | PSNR |
|---|---:|
| contain 1024 | 39.02 dB |
| contain 1600 | 42.53 dB |
| cover 1000 | 42.03 dB |
| cover 1234 | 44.24 dB |
| cover 1600 | 47.49 dB |
| area, 6 MP crop | 87.8 dB |
| mitchell via cubic | 48.4 dB |

Nearest-neighbour ties can differ by one source pixel, giving about 38.8 dB.
That is inherent to resolving exact half-way ties, not a defect.

## 4. Working budget

### Change

`_default_working_bytes()` uses `GlobalMemoryStatusEx` and returns:

```
max(512 MiB, min(2048 MiB, physical_ram // 4))
```

On this 15.8 GiB host that is 2 GiB. The value is a soft limiter for CPU job
buffers, not a hard native allocation cap. A 22 MP job is charged about 238 MB
(`px * 10 + out_px * 16` for RGB), so 2 GiB allows eight concurrent jobs.

`RunOptions.max_working_bytes` uses `default_factory` so the host value is
computed at run time. No existing test asserted the old constant.

### Evidence

24 files, CPU, 8 processors:

| Budget | Peak RSS | files/s |
|---:|---:|---:|
| 1 GiB | 816.1 MiB | 7.5 |
| 2 GiB | 1,521.6 MiB | 8.9 |

48 files, CPU:

| Budget / workers | files/s |
|---|---:|
| 1 GiB / 8 | 8.42 |
| 2 GiB / 8 | 10.03 |
| 2 GiB / 16 processors | 10.76 |
| 3 GiB / 16 processors | 10.51 |

The plateau at 2 GiB is memory bandwidth, not the limiter. The default stays
at eight workers.

The trade-off is real: on a 4-8 GiB host the formula yields 1-2 GiB or the
512 MiB floor, so concurrency and throughput scale down with the machine.

## 5. Color management fast path

### Change

`needs_cms` was narrowed to require CMS only when a transform is actually
needed:

```
policy == web_srgb
or linear_light
or (has_alpha and mode != "RGBA")
or (icc and mode not in ("RGB", "RGBA", "P"))
or mode not in ("RGB", "RGBA", "L", "CMYK")
```

Previously L, CMYK and RGBA without ICC were sent through the Pillow
color-management decoder. Under `preserve_rgb`, an untagged L/CMYK image can
be decoded directly and resized.

The `has_alpha and mode != "RGBA"` clause is required for palette modes with
transparency: L+tRNS initially produced black output when decoded without
CMS, and `tests/test_audit_regressions.py` caught it. Those modes still take
the CMS path.

### Evidence

| Input | Decode before | Decode after | Throughput before | Throughput after |
|---|---:|---:|---:|---:|
| Gray | 593 ms | 141 ms | 4.1 files/s | 5.88 files/s |
| CMYK | 886 ms | 338 ms | 2.65 files/s | 4.28 files/s |
| RGBA PNG | 251 ms | 91 ms | 3.86 files/s | 6.86 files/s |
| RGBA WebP | 309 ms | 79 ms | 3.83 files/s | 6.61 files/s |

### Quality

- Gray: pixel-identical.
- Alpha PNG/WebP: pixel-identical.
- CMYK: 46.8-46.9 dB, about one 8-bit level, from using the OpenCV decoder
  instead of the Pillow color path.

## 6. Additional optimizations and fixes

### 6.1 Alpha compositing

#### Change

`CPUEngine.composite_bg` now uses Pillow:

```
fg = Image.fromarray(img, "RGBA")
canvas = Image.new("RGBA", fg.size, bg)
Image.alpha_composite(canvas, fg).convert("RGB")
```

The NumPy float path remains as a fallback. The background tuple is passed in
the same BGR-ordered space as the array; it is not reversed.

#### Evidence

3.2 MP alpha image:

| Path | Time |
|---|---:|
| NumPy float | 263-275 ms |
| Pillow | 62-101 ms |
| MAE | 0 |

### 6.2 Linear-light fast path

`--linear-light` previously converted the full 22 MP frame to float, ran the
NumPy separable reference and converted back. The new path keeps the same
linear-light semantics but uses OpenCV float kernels on a whole-pixel region:

- `resample_cpu_float` in `batchresize/resampling.py` handles float32 frames,
  with the reference engine as the fallback for out-of-bounds plans.
- For `lanczos3`/`mitchell` reductions of 2x or more, it uses `INTER_AREA`:
  Lanczos4 rings visibly in linear light, while area averaging is closer to
  the reference result.
- `srgb_to_linear` uses `cv2.LUT` instead of NumPy fancy indexing
  (318 ms to 64 ms on 22 MP, bit-exact).
- Untagged RGB/RGBA inputs use the OpenCV decode path; the two full-frame
  channel reversals are skipped for the non-alpha case.

| Case | Before | After |
|---|---:|---:|
| 6 files, 6 workers, `web_1024` | 0.22 files/s | 3.28 files/s |
| Resize per 22 MP file | 13,224 ms | 617-736 ms |
| Total elapsed for 6 files | 27.297 s | 1.832 s |
| PSNR vs reference, 6 files | reference | 41.0-45.7 dB |
| Alpha linear-light, 512 to 256 | reference | 53.9 dB |

This path is still slower than the default because linearization is a
full-frame float operation, but the 13-15x improvement removes it as a P1
blocker. `--linear-light` remains CPU-only; the GPU path still falls back.

### 6.3 Fast `--web-srgb` for untagged RGB/RGBA

An untagged RGB/RGBA image needs no ICC transform: the sRGB policy only needs
the standard output profile, which `_finish` attaches. `needs_cms` now lets
those inputs use the OpenCV decoder; ICC-tagged and non-RGB inputs keep the
Pillow/littleCMS path.

| Case | Before | After |
|---|---:|---:|
| 48 files, 8 workers, `--web-srgb` | 4.83 files/s | 10.16 files/s |
| Decode per 22 MP file | 1,380 ms | 563-582 ms |
| Output vs old path | reference | pixel-identical |
| ICC profile attached | 48/48 | 48/48 |

PSNR is not meaningful when the outputs are identical: every one of the 48
outputs was byte-identical to the previous path for this untagged corpus.

### 6.4 No CUDA initialization on CPU sharpen/pad

`apply_sharpen` imported CuPy even for NumPy images in order to test
`isinstance(img, cp.ndarray)`, which initialized CUDA on the first sharpened
CPU file. The check now uses `__cuda_array_interface__` and imports CuPy only
for actual GPU arrays. The same change was applied to `apply_pad`.

| Case | Before | After |
|---|---:|---:|
| First 1-file `--sharpen standard` | 1,075.8 ms resize | 47.3 ms resize |
| 48 files, `--sharpen standard` | 8.33 files/s | 8.82 files/s |
| Resize per file in that batch | 232 ms | 137-147 ms |

The 48-file gain is modest because the 1 s CUDA initialization was amortized;
for a single image or a small batch it is the dominant avoidable cost.

### 6.5 Budget flag and GUI fix

- `--working-mb N` sets `RunOptions.max_working_bytes` explicitly; the
  adaptive 2 GiB/25% default remains when the flag is omitted. A CLI test
  covers the override.
- `gui.py::_sync_size_limit_mode` now reads the size-limit string into `raw`
  before using it; the previous code referenced an undefined name inside a
  caught exception. The GUI could not be launched because `.venv` lacks
  Tkinter, so this is covered by static review and the existing skipped GUI
  tests only.

## 7. Tests and validation

Final full suite:

```
392 passed, 5 skipped in 59.04s
```

Skips: three GUI modules because Tkinter is absent from `.venv`, one symlink
privilege test, one missing CMYK ICC fixture.

Additional validation:

- 517-file CPU `web_1024`: 517 processed, 0 failed, all outputs decode.
- 517-file GPU `web_1024`: 517 processed, 0 failed, all outputs decode.
- Truncation scanner compared with the old byte-loop reference on all 517
  files and 50 truncation offsets: 0 mismatches.
- Cover-crop outputs compared at three target sizes: 42.0-47.5 dB.
- Complex-image categories (gray, CMYK, alpha PNG, alpha WebP, orientation)
  compared before/after.

## 8. Rejected or deferred options

- **Enable `--fast-jpeg-decode` by default.** It reaches 15.62 files/s on 96
  files, but PSNR is 38.0-42.4 dB. That is below the normal path and not
  acceptable as a silent default.
- **Skip CMS when the ICC description contains "sRGB".** Unsafe. A profile
  can be named "sRGB" with a gamma of 1.0; the description is not a
  correctness guarantee.
- **Map every `lanczos3` to `INTER_AREA`.** Faster for some downscales but
  changes the filter's behavior and upscaling semantics.
- **Use 16 processors as the default.** Only 2-6% more than eight workers at
  48 files, at higher memory pressure.
- **GPU batch 16 or larger VRAM reservation.** 10.2-10.4 files/s, no
  consistent gain over the default.
- **Enable reduced DCT JPEG decode by default.** It reaches 15.62 files/s on
  96 files but loses 2-5 dB PSNR; it stays opt-in.
- **Rotate after resize to remove full-frame orientation cost.** The geometry
  transform for orientations 5-8 is easy to get subtly wrong; the ~77-126 ms
  per 22 MP file does not justify the risk in this pass.

## 9. Risks

- The rounded pixel region shifts a sub-pixel crop by at most half a source
  pixel. That is perceptually negligible and far cheaper than the 5-6 s
  reference fallback, but it is an approximation.
- `mitchell` is approximated by cubic (48.4 dB). The filter is not bit-exact
  to the reference kernel.
- `area` via `INTER_AREA` is exact for the tested downscales; `area` and
  `box` upscaling use the reference engine rather than an approximation.
- The CMS shortcut assumes an RGBA ICC profile describes RGB space. The
  tested fixtures satisfy that; unusual device-link or CMYK-in-RGBA profiles
  would need explicit handling.
- The budget is heuristic. Observed peak RSS is below the nominal budget, but
  a future job type with larger transient buffers should re-check the charge.
- OpenCV resampling is an external behavior dependency; the reference engine
  remains available for numeric comparison.
- Linear-light uses `INTER_AREA` for reductions of 2x or more. This is a
  documented deviation from Lanczos3/4, chosen because Lanczos4 rings in
  linear light; the 6-file comparison is 41.0-45.7 dB against the reference.
- `cv2.LUT` was verified bit-exact against NumPy LUT indexing on a random
  uint8 array; the web-srgb untagged corpus output was bit-identical.

## 10. Files

Changed for performance:

- `batchresize/imageio_meta.py`: truncation scan.
- `batchresize/resampling.py`: OpenCV mapping, `_pixel_region`,
  `_nearest_resample`, `_cv2_resample`, `_pillow_resample`,
  `resample_cpu_float`.
- `batchresize/pipeline.py`: `_default_working_bytes`, `max_working_bytes`
  default, `needs_cms`, linear-light BGR fast path.
- `batchresize/cpu_backend.py`: `composite_bg`.
- `batchresize/color_management.py`: `cv2.LUT` fast linearization.
- `batchresize/sharpen.py`, `batchresize/geometry.py`: lazy CUDA detection.
- `batchresize/cli.py`: `--working-mb`.
- `batchresize/gui.py`: undefined `raw` fix.

Tests added:

- `tests/test_cli.py`: `--working-mb` override.
- `tests/test_resampling.py`: float fast path vs reference PSNR.

Added for reproducibility:

- `benchmark/make_corpus.py`
- `benchmark/run_bench.py`
- `benchmark/README.md`
- `benchmark/.gitignore`

Not changed: safety/atomic-write logic, resume state, locking, and `input/`.
