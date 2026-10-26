# PERFORMANCE_AUDIT.md

Independent performance audit of BatchResize.

- Date: 2026-10-26
- Branch: `feature/mvp-0-3`
- Base commit: `b4fc6b6`; the working tree also contains the optimizations
  described in `PERFORMANCE_OPTIMIZATION_REPORT.md`
- Reference corpus: 517 real 22 MP-class JPEGs (688 MiB) in `input/`
- Companion documents: `PERFORMANCE_BASELINE.md`,
  `PERFORMANCE_OPTIMIZATION_REPORT.md`, `PERFORMANCE_FINAL.md`

## 1. Executive summary

The pipeline is now **decode-bound on large JPEGs**. The old CPU path was
resize-bound on top of a reader that serialized on a Python byte-by-byte
truncation scan. Both costs were removed or reduced, and the working-memory
budget was raised so more 22 MP jobs can run concurrently.

Headline results on the 48-file 22 MP-class subset:

| Workload | Before | After | Speedup |
|---|---:|---:|---:|
| CPU, `web_1024` | 2.45 files/s | 10.03 files/s | 4.1x |
| CPU, `large_1600` | 2.59 files/s | 10.39 files/s | 4.0x |
| GPU, `web_1024` | 3.64 files/s | 10.26 files/s | 2.8x |
| GPU, `large_1600` | 3.51 files/s | 9.98 files/s | 2.8x |

The "before" column is the original code at its 1 GiB default. The "after"
column uses the new adaptive default, which resolves to 2 GiB on this 16 GiB
host. At the **same 1 GiB budget**, the code-only speedup is 3.4x on CPU
(2.45 to 8.42 files/s), so both the code changes and the budget change matter.

On the full 517-file corpus, one pass at `web_1024`:

- CPU: 48.2 s elapsed, 10.72 files/s, 0 failures, 517 valid JPEGs
- GPU: 51.3 s elapsed, 10.08 files/s, 0 failures, 517 valid JPEGs
- Wall time: 49.1 s CPU, 57.2 s GPU (GPU startup is ~5.9 s)

All 389 tests pass with 5 environment skips. Output quality remains within
39-48 dB PSNR of the previous engine for normal contain/cover resizes; gray
and alpha cases are pixel-identical.

The follow-up pass also fixed the two blocking optional paths: `--linear-light`
went from 0.22 to 3.28 files/s at 41.0-45.7 dB against the reference, and
`--web-srgb` went from 4.83 to 10.16 files/s with bit-identical output on the
untagged corpus. CPU JPEG decode remains the dominant default cost
(560-670 ms per 22 MP file at 8 workers).

## 2. Environment

| Item | Value |
|---|---|
| OS | Windows 10 22H2, build 19045 |
| CPU | Intel Xeon E5-2670 v3, 12 cores / 24 logical processors |
| RAM | 15.8 GiB |
| GPU | NVIDIA GeForce GTX 1060 6 GB, SM 6.1, driver 581.57 |
| CUDA | 12.9 runtime, nvImageCodec 0.9.0 |
| Python | 3.13.14 in `.venv` |
| numpy / OpenCV / Pillow | 2.5.4 / 5.0.0.93 / 13.0.0 |
| CuPy | 14.2.0 |

This is a Pascal GPU without a hardware JPEG decoder. GPU results should not
be extrapolated to Turing or newer cards with nvJPEG.

## 3. Workload and method

The corpus is 517 JPEGs: 303 portrait 3840x5760, 209 landscape
5760x3840, and five images between 6 and 13.5 MP. All are RGB without ICC or
alpha. The main batch measurements use the first 48 sorted files (43 large,
5 small) and the first 96; the full-corpus run uses all 517.

Method:

- Warm cache, one warmup, repeated runs, median reported.
- `elapsed` is the pipeline timer; `wall` includes process and CUDA startup.
- Stage timings are per-file averages from the pipeline report.
- Quality is measured against the old output with PSNR/MAE, not against a
  lossless source.
- 517-file runs validated every output with Pillow `verify()` and checked
  dimensions.
- Full suite run after the changes: `389 passed, 5 skipped`.

## 4. Where the time went

### 4.1 Reader: Python byte scan under the GIL

`is_truncated_jpeg` walked the entropy stream one byte at a time. On a 1 MiB
file it took 142.39 ms; on a 22 MP photo the reader stage averaged 575.8 ms
(CPU) and 466.0 ms (GPU). Because the scan is pure Python, all reader threads
serialize on the GIL. The GPU was idle while the CPU scanned bytes.

This was the single largest reason the GPU path underperformed: GPU decode was
110 ms, but the reader could not feed it.

### 4.2 CPU resize: Pillow float path

The old CPU resize used Pillow's general resampler on a full BGR frame. On
22 MP to 1600 it took 599 ms in isolation, and 811-829 ms per file in the
batch (including GIL contention). OpenCV's specialized kernels perform the
same class of operation in 16-20 ms.

Pillow handled sub-pixel crops through its general resampler (599 ms). An
OpenCV fast path that accepted only whole-pixel regions would have sent every
sub-pixel cover crop to the NumPy separable reference engine, which costs
5.2-6.2 s per 22 MP image. Rounding the crop to whole pixels keeps those
crops on the fast kernel.

### 4.3 Budget: four concurrent 22 MP jobs

The 1 GiB default charged roughly 238 MB per 22 MP job, allowing only four
processors to hold a buffer at once. The host has 15.8 GiB and eight
processors were configured, so half the CPU capacity waited on the limiter.
At 1 GiB, 48 files took 5.704 s (8.42 files/s); at 2 GiB, 4.785 s
(10.03 files/s). Going to 3 GiB, or to 16 processors, added no throughput
(10.5-10.8 files/s): the workload becomes memory-bandwidth-bound.

### 4.4 Color management on non-RGB inputs

For grayscale, CMYK, and RGBA inputs without an ICC profile, the old
`needs_cms` condition sent the image through the Pillow color-management
decoder before resize. That cost 593 ms for gray decode, 886 ms for CMYK, and
251-309 ms for alpha inputs, even when no profile existed. With the widened fast-decode condition,
gray decode fell to 141 ms, CMYK to 338 ms, alpha to 79-91 ms.

### 4.5 Alpha compositing

Compositing alpha over a background used a NumPy float path. At 3.2 MP it
took 263-275 ms per image. Pillow's `alpha_composite` does the same job in
62-101 ms, with an exact pixel match in the tested cases.

### 4.6 Optional paths

These are not regressions from the optimization work; they are pre-existing
costs that the audit measured so they are not mistaken for general pipeline
performance:

- `--linear-light`: initially 13,224 ms resize per image and 0.22 files/s.
  Follow-up: 617-736 ms and 3.28 files/s (15x), 41.0-45.7 dB vs reference.
- `--web-srgb`: initially 1,380 ms decode per image and 4.83 files/s.
  Follow-up for untagged RGB/RGBA: 563-582 ms and 10.16 files/s, bit-identical
  output. ICC-tagged inputs still take the Pillow/littleCMS path; skipping by
  profile name remains unsafe.
- `--sharpen standard`: 232 ms resize and 8.33 files/s initially. Follow-up:
  137-147 ms and 8.82 files/s after removing an accidental CUDA import on the
  CPU path.
- `--fast-jpeg-decode`: 15.62 files/s on 96 files, but 38.0-42.4 dB PSNR.
  Correctly opt-in only.

## 5. Findings

| ID | Severity | Finding | Status |
|---|---|---|---|
| P-01 | High | Python truncation scan serializes readers and starves the GPU | Fixed |
| P-02 | High | Pillow CPU resize is ~40x slower than OpenCV kernels | Fixed |
| P-03 | Medium | Sub-pixel cover crops cannot use a whole-pixel OpenCV path; a naive fast path would fall back to the 5-6 s reference engine | Fixed by rounded pixel regions |
| P-04 | Medium | 1 GiB budget allows only four 22 MP jobs | Fixed with adaptive 2 GiB default |
| P-05 | Medium | Gray/CMYK/RGBA are forced through the Pillow color path even without an ICC profile | Fixed |
| P-06 | Medium | Float alpha compositing is 3-4x slower than Pillow | Fixed |
| P-07 | High (optional path) | `--linear-light` resize is 13.2 s/image | Fixed: 0.22 to 3.28 files/s, 41.0-45.7 dB |
| P-08 | High (optional path) | `--web-srgb` forces the Pillow color path at 1.38 s/image decode | Fixed for untagged RGB/RGBA: 10.16 files/s, bit-identical; ICC path unchanged |
| P-09 | Medium | CPU JPEG decode is the remaining default bottleneck | Accepted; opt-in fast decode documented |
| P-10 | Low | GPU startup costs ~5.9 s on a 517-file process | Accepted; matters for small batches |
| P-11 | Informational | GPU no longer beats CPU at `web_1024` on this Pascal card | Expected; GPU still offloads CPU |
| P-12 | Low | CPU sharpen imported CuPy on first use, initializing CUDA | Fixed: first file 1,076 to 47 ms |
| P-13 | Low (correctness) | GUI size-limit hint referenced undefined `raw` | Fixed; GUI still not measurable without Tkinter |

## 6. Correctness and safety

- Full suite: 389 passed, 5 skipped. The skips are 3 missing Tkinter, 1
  symlink privilege, 1 missing CMYK ICC fixture.
- All 517 CPU outputs and all 517 GPU outputs decode and have the expected
  683x1024 or 1024x683 dimensions.
- Alpha and grayscale outputs are pixel-identical to the old path in the
  tested fixtures.
- CMYK output differs by about one 8-bit level (46.8-46.9 dB PSNR), caused by
  skipping the unnecessary CMS round trip.
- Contain/cover resizes are 39.0-47.5 dB PSNR against the old Pillow output.
- `area` is 87.8 dB; `mitchell` mapped to cubic is 48.4 dB.
- Fast JPEG decode is 38.0-42.4 dB and remains opt-in.
- The clamp guard in `_pixel_region` tolerates at most half a pixel of float
  geometry error and falls back to the reference engine for genuine
  out-of-bounds plans.
- Atomic writes, resume state, locking and `input/` were not modified by this
  work.

## 7. GUI review

The GUI event loop was reviewed statically; it could not be launched because
`.venv` has no Tkinter. Findings:

- Events are drained every 100 ms; preview work is debounced and runs in a
  subprocess, so the Tk loop is not doing image work.
- `ui_view._schedule_summary` is coalesced with `after_idle`, so summary
  redraws do not stack.
- `gui.py` had a correctness bug around line 903: `_sync_size_limit_mode`
  referenced an undefined `raw` inside a caught exception. It is now fixed;
  the GUI still cannot be measured without Tkinter.

No GUI startup or preview timing claim is made.

## 8. Limitations

- Warm cache only; no cold-disk measurement.
- No sampling profiler or Nsight; conclusions are from stage timers and
  controlled A/B runs.
- The 48/96-file subsets include five smaller images, so their absolute
  throughput is slightly optimistic. The 517-file run is the reliable
  end-to-end number.
- Parent-process RSS sampling in the temporary driver was unreliable
  (~3.4 MB); memory figures use the in-child `PeakWorkingSetSize`.
- The GPU has no hardware JPEG decoder.
- Original audit defects AUD-001/AUD-002/AUD-003 are outside this
  performance audit and remain open.

## 9. Recommendation

Keep the default changes. Do not broaden fast JPEG decode to default. Treat
`--linear-light` and `--web-srgb` as specialized paths with documented costs,
or optimize them separately. Update user-facing throughput text to state the
corpus, resolution, backend and warm-cache assumption.
