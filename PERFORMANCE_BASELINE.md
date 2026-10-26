# PERFORMANCE_BASELINE.md

Pre-optimization baseline for BatchResize on the reference machine.

- Date: 2026-10-26
- Branch: `feature/mvp-0-3`, base commit `b4fc6b6`
- Baseline definition: original reader/truncation scan, Pillow resampling
  engine, original 1 GiB working budget, original alpha compositing and
  color-management decision logic
- Harness: temporary `legacy_runner.py`, which patches
  `batchresize.imageio_meta.is_truncated_jpeg` to the byte-by-byte scanner and
  sets `batchresize.resampling.RESAMPLE_ENGINE = "pillow"`

## 1. Environment

| Item | Value |
|---|---|
| OS | Windows 10 22H2, build 19045 |
| CPU | Intel Xeon E5-2670 v3, 12 cores / 24 logical processors, 2.30 GHz |
| RAM | 15.8 GiB (17,013,354,496 bytes) |
| GPU | NVIDIA GeForce GTX 1060 6 GB, SM 6.1, max graphics clock 1911 MHz |
| GPU driver / CUDA | 581.57 / 12.9 |
| Python | 3.13.14 in `.venv` |
| numpy | 2.5.4 |
| OpenCV | opencv-python-headless 5.0.0.93 |
| Pillow | 13.0.0 |
| CuPy | cupy-cuda12x 14.2.0 |
| nvImageCodec | nvidia-nvimgcodec-cu12 0.9.0.20 |

`--doctor` reports both engines ready. The GPU has no hardware JPEG decoder;
decode runs through nvImageCodec on the Pascal GPU.

## 2. Workload

The reference corpus is `input/`: 517 JPEGs, 688.0 MiB.

| Dimensions | Count | Notes |
|---|---:|---|
| 3840 x 5760 | 303 | portrait, 22.1 MP |
| 5760 x 3840 | 209 | landscape, 22.1 MP |
| 2004 x 3006 | 1 | 6.0 MP |
| 2336 x 3504 | 1 | 8.2 MP |
| 2408 x 3612 | 1 | 8.7 MP |
| 2472 x 3708 | 1 | 9.2 MP |
| 3000 x 4500 | 1 | 13.5 MP |

All are RGB, JPEG, no ICC profile, no alpha. The 48-file and 96-file
subsets use the first N sorted paths, so they contain 43/91 22 MP files plus
the five smaller files. This makes their absolute throughput slightly
optimistic relative to the full corpus; all before/after comparisons use the
same subset.

Output profile for the main measurements: `web_1024` (1024 px long edge,
contain, JPEG q=80, 4:2:0, no upscale). The secondary profile is `large_1600`.

## 3. Method

- Warm OS cache: each workload was read at least once before timing. Cold-cache
  measurements were not performed (see limitations).
- One warmup run, then repeated runs; medians reported. Most 48-file cells are
  single stable runs; the headline cells were repeated 3 times.
- `elapsed` is the pipeline's own timer. `wall` includes interpreter startup
  and, on GPU, CUDA/nvImageCodec context creation.
- Stage timings are averages per processed file as reported by the pipeline.
  Reader threads, processor threads and writer threads run concurrently, so
  stage times are not additive across a batch.
- Quality is PSNR against the previous output, not against a lossless source.

## 4. Baseline: 48-image batch

### 4.1 CPU

| Profile | Read ms | Decode ms | Resize ms | Encode ms | Write ms | Elapsed s | files/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| web_1024 | 575.8 | 492.7 | 811.6 | 21.7 | 536.3 | 19.598 | 2.45 |
| large_1600 | 642.0 | 406.2 | 828.9 | 29.2 | 503.4 | 18.530 | 2.59 |

Workers: 8 readers, 8 processors, 4 writers. Budget: 1 GiB.

The resize stage is the largest genuine CPU cost. Read and write stage times
are also inflated because the Python byte-by-byte truncation scan holds the
GIL; all reader threads serialize behind it, and the delay is attributed to
whichever stage is timing at that moment.

### 4.2 GPU

| Profile | Read ms | Decode ms | Resize ms | Encode ms | Write ms | Elapsed s | files/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| web_1024 | 466.0 | 110.1 | 119.4 | 3.44 | 42.7 | 13.188 | 3.64 |
| large_1600 | 453.4 | 122.5 | 112.8 | 9.09 | 34.9 | 13.664 | 3.51 |

GPU decode is fast, but the same reader bottleneck leaves the GPU idle. This
is why the legacy GPU path was only 1.5x faster than the legacy CPU path on
22 MP files, despite decoding roughly 4x faster.

### 4.3 Older 24-image synthetic result

`README.md` and `REPORT.md` contain older figures measured on 24 synthetic
1800x1200 images (CPU 12.0 files/s, GPU 14.3 files/s, and historical 46-62
files/s on a different run). Those figures are not comparable to this corpus:
the images are 7x smaller, the code predates the current pipeline, and the
measurement mixed startup with steady state. They should be read as
workload-specific history, not as a claim for 22 MP camera files.

## 5. Microbenchmarks (22 MP JPEG, ~1.05 MiB)

| Operation | Baseline |
|---|---:|
| `sha256` of source bytes | 3.16 ms |
| `read_source_meta` | 0.096 ms |
| `is_truncated_jpeg` | 142.39 ms per 1 MiB |
| OpenCV decode | 289-305 ms |
| Pillow decode | 314.7 ms |
| Pillow Lanczos resize to 1600 | 599 ms |
| NumPy separable reference resize to 1600 | 5,233-6,192 ms |
| OpenCV resize | 16-20 ms |
| OpenCV JPEG encode | 7.0 ms |
| Pillow JPEG encode | 22.6 ms |
| EXIF orientation transform | 79-126 ms |
| float BGR-to-RGB (`astype` + dot) | 163.5 ms |
| `cv2.cvtColor` BGR-to-RGB | 20.7 ms |

The 142 ms truncation scan affects every discovered JPEG and runs on reader
threads. At 8 readers the GIL turns it into a serial section of roughly
0.14 s per file.

## 6. Memory baseline

The old default working budget was 1 GiB. A 22 MP job is charged roughly
238 MB, so only four processors could run concurrently. Measured peak process
working set for a 24-file CPU batch at the 1 GiB budget:

| Budget | Concurrent 22 MP jobs | Peak RSS |
|---:|---:|---:|
| 1 GiB | 4 | 816.1 MiB |

The measured peak is lower than the nominal budget because not every job
allocates its full charge simultaneously and because completed jobs release
buffers before the next ones reach peak.

## 7. Quality reference

Baseline output was produced by the Pillow resampling path. Differences after
optimization are reported as PSNR against that output in
`PERFORMANCE_OPTIMIZATION_REPORT.md`; the baseline itself is the reference,
not ground truth.

## 8. Reproduction

From the repository root:

```powershell
# Baseline, 48 files, CPU, 1024
.\.venv\Scripts\python.exe C:\Users\inval\AppData\Local\Temp\opencode\br\legacy_runner.py `
  -i input -o "$env:TEMP\br-baseline-cpu" --limit 48 --quiet `
  --backend cpu --workers 8 --size web_1024

# Same with the old 1 GiB budget on the new code
$env:BATCHRESIZE_WORKING_MB = "1024"
.\.venv\Scripts\python.exe C:\Users\inval\AppData\Local\Temp\opencode\br\budget_runner.py `
  -i input -o "$env:TEMP\br-new-cpu-1g" --limit 48 --quiet `
  --backend cpu --workers 8 --size web_1024
Remove-Item Env:BATCHRESIZE_WORKING_MB
```

The in-repo harness (`benchmark/run_bench.py`) reproduces the current code,
not the legacy code. The legacy runner was a temporary measuring tool and is
not part of the repository.

## 9. Limitations

- Warm cache only. Cold-cache disk behavior was not measured; flushing the
  Windows cache or rebooting was considered unsafe for this session.
- The GTX 1060 has no hardware JPEG decoder, so GPU decode results do not
  generalize to newer GPUs.
- No sampling profiler (py-spy, Scalene, Nsight) was available. Statements
  about CPU vs GPU time are based on stage timers, microbenchmarks and
  controlled A/B runs, not on call-stack samples.
- The 48/96-file subsets contain five smaller images; see section 2.
- The original correctness audit's known defects AUD-001 (report path
  labels), AUD-002 (stale lock file) and AUD-003 (Mitchell constant) are
  outside this performance baseline and remain unchanged.
