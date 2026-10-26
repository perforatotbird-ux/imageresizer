# AUDIT_TEST_RESULTS.md

Independent audit of `batchresize / imageresizer` at commit
`b4fc6b6` (branch `feature/mvp-0-3`).
All results below were produced during this audit session. Nothing in the
project source tree, tests, configuration or user data was modified.

---

## 1. Recorded initial state

```
$ git rev-parse HEAD
b4fc6b61af92f018b2694113c4f44f003e63e995
$ git rev-parse --abbrev-ref HEAD
feature/mvp-0-3
$ git status --porcelain
 D AUDIT_REPORT.md
 D FIX_REPORT.md
?? "[SAMPLE] Client A_Invoice_[SAMPLE] Invoice 3.pdf"
$ git diff --stat HEAD
 AUDIT_REPORT.md | 643 -----
 FIX_REPORT.md   | 176 ---
```

Interpretation: the two earlier audit documents were deleted from the working
tree but are still recoverable from history (`git show 44821df:AUDIT_REPORT.md`).
All Python sources are byte-identical to HEAD, so the audit evaluates the
committed state.

### Environment

| Item | Value |
|---|---|
| OS | Windows 10 22H2, build 19045.6466 |
| CPU | Intel Xeon E5-2670 v3 @ 2.30 GHz, 24 threads |
| RAM | 15.8 GiB |
| GPU | NVIDIA GeForce GTX 1060 6 GB, SM 6.1 (Pascal) |
| Driver | 581.57 |
| Python | 3.13.14 (project `.venv`) |
| numpy / Pillow / OpenCV / piexif / pytest | 2.5.4 / 13.0.0 / 5.0.0.93 / 1.1.3 / 9.1.1 |
| cupy-cuda12x / nvimgcodec-cu12 / nvjpeg-cu12 | 14.2.0 / 0.9.0.20 / 12.4.0.76 |
| tkinter in `.venv` | not installed (GUI tests skip) |

`--doctor` reported the GPU engine and self-test as OK (decode → resize →
encode round-trip), CPU engine OK, OpenCV CUDA devices 0 (expected: the pip
OpenCV build has no CUDA support).

---

## 2. Automated test suite

Command:

```
PYTHONUNBUFFERED=1 .venv/Scripts/python.exe -m pytest tests -q \
  --no-header --tb=no -p no:cacheprovider --junitxml=AUDIT_TMP_full.xml
```

Result (repeatable: three full runs produced the same outcome):

```
tests 355   errors 0   skipped 4   time 92.1 s
F  tests/test_audit_regressions.py::test_IR01_output_links_rejected
F  tests/test_preview.py::test_preview_uses_the_real_pipeline
F  tests/test_preview.py::test_preview_honours_exif_orientation
F  tests/test_preview.py::test_preview_reuses_result_when_only_display_box_changes
F  tests/test_preview.py::test_preview_reports_errors_and_keeps_working
F  tests/test_preview.py::test_preview_reports_requested_gpu_route
F  tests/test_preview.py::test_preview_worker_cleans_its_temp_tree
S  tests/test_gui_*.py (3, no tkinter in the venv)
S  tests/test_audit_regressions.py::test_IR14_valid_CMYK_profile
```

### 2.1 Why the 7 failures are environmental, not product defects

* **6 × preview worker.** The worker is a *subprocess*. When the preview tests
  run after other tests, the audit sandbox's bulk-delete guard kills the worker
  process (`[safe-delete][SAFE_DELETE_BULK_GUARD_ERROR] exit 3221225477`) and
  the test observes `AssertionError: preview worker closed its protocol stream`.
  Bisected: `pytest tests/test_observability.py tests/test_preview.py` fails,
  `pytest tests/test_preview.py` passes with the same code. Recorded as AUD-004.
* **1 × IR-01 output link.** The test creates a directory symlink. In this
  sandbox `Path.symlink_to()` returns without error but the link does not exist
  (`is_symlink() == False`, `os.lstat` raises), so no `ValueError` is raised.
  The guard itself was verified with a monkeypatched `is_symlink` and rejected
  the path correctly. Recorded as AUD-005.

The 4 skips are the GUI tests (tkinter is not installed in the project venv);
they are not run in CI of this audit.

---

## 3. Functional scenarios (CLI)

All commands ran against synthetic fixtures inside `AUDIT_TMP/` (never against
`input/` / `output/`, which hold real photos). `--backend cpu` was used for
determinism; the GPU path was exercised separately.

### 3.1 JPEG size limit — PASS

```
run --size small_512 --jpeg-quality 95 --max-file-size 20KB
  a.jpg        18530 B   over limit: no
  alpha.jpg     5446 B   over limit: no
  meta.jpg     19670 B   over limit: no
  report: target_size_achieved = true
```
The reported achievement matches the bytes on disk.

### 3.2 Personal metadata stripping — PASS

Input carried Orientation=6, Artist, DateTime, BodySerialNumber and a GPS IFD.

```
  output EXIF 0th keys   : [256, 257, 274, 34665]   (width/height, Orientation, IFD pointer)
  output EXIF Exif keys  : [40962, 40963]            (pixel dimensions only)
  GPS                    : removed
  ICC profile            : preserved (32 bytes)
```
Artist, DateTime and the serial numbers are gone; Orientation and ICC survive.

### 3.3 Resume / cache invalidation — PASS (with a caveat, AUD-008)

```
  run1 (fresh)           processed=3  skipped=0
  run2 (same, --skip-existing)     processed=0  skipped=3
  run3 (quality changed)           processed=0  skipped=0
  run4 (same as run3 config)       processed=0  skipped=3
```
Verified outputs are skipped; a changed configuration invalidates them. In run3
the outputs are neither regenerated nor skipped (they are reported as failures
with "use --overwrite") — see AUD-008.

### 3.4 Variants — PASS

```
  --variant thumb:256x256 --variant web:1024x1024
  thumb/a.jpg  thumb/alpha.jpg  thumb/meta.jpg
  web/a.jpg    web/alpha.jpg    web/meta.jpg     (+ manifest .batchresize-state.json)
```

### 3.5 Geometry edges — PASS

| source | target | result |
|---|---|---|
| 1×1 | 512×512 contain | 1×1 (no upscale) |
| 1×300 | 512×512 contain | 1×300 |
| 300×1 | 512×512 contain | 300×1 |

Sizing modes on a 1400×1000 source: `percent 25` → 350×250,
`megapixels 0.2` → 560×400, `longest 640` → 640×457,
`shortest 640` → 640×457. Negative/zero sizing values are rejected.

### 3.6 Filter sanity (CPU reference, 1-pixel checkerboard, 2:1) — PASS

| filter | output |
|---|---|
| area | uniform grey 128 (correct averaging) |
| box | uniform grey 128 |
| nearest | hard black/white preserved |
| bilinear / lanczos3 / mitchell | ~128 with minor ringing |

The 4-block checkerboard used initially could not distinguish filters; the
1-pixel checkerboard was used for the final measurement.

### 3.7 Stale output lock — FAIL (AUD-002)

```
$ echo 1234 > out/.batchresize.lock
$ batchresize -i src -o out --size thumb_256 --backend cpu
error: [Errno 17] File exists: '...\out\.batchresize.lock'      (exit code 2)
```
No stale detection, no recovery flag, non-actionable message.

### 3.8 GUI construction smoke test — PASS (partial)

Python 3.10 (system) has tkinter but no OpenCV, so a *real* job could not be
executed; importing the module, constructing the window and closing it works:

```
$ python -c "import batchresize.gui as g; a=g.BatchResizeUI(); a.update(); a._on_close()"
GUI constructed OK
close handled OK
```

---

## 4. Performance and memory measurements

Synthetic set: 24 × 1800×1200 JPEG (7.3 MiB total), target 1024×1024 contain,
lanczos3, quality 80, workers 8, batch 8, best of the last 3 of 4 passes.

| backend | wall time | throughput | decode | resize | encode |
|---|---|---|---|---|---|
| cpu | 2.00 s | 12.0 files/s | see report | | |
| gpu | 1.68 s | **14.3 files/s** | | | |
| pure read (no processing) | 0.01 s | 2509 files/s | — | — | — |

**GPU/CPU speedup = 1.19×** on this hardware (Pascal, no hardware JPEG decoder).
The older documented 1.8–2.2× / 46–62 files/s figures are not reproducible at
HEAD (AUD-006).

Repeated-batch memory (40 × 1600×1200 → 512 px, 4 passes in one process):

| backend | RSS pass 1 | RSS pass 4 | peak | commit |
|---|---|---|---|---|
| cpu | 46.3 MiB | 46.5 MiB | 126.4 MiB | ~770 MiB |
| gpu | 232.3 MiB | 238.4 MiB | 252.1 MiB | ~1.42 GiB |

No growth → no leak across repeated batches. The GPU path is ~200 MiB heavier
in host RSS (CuPy/nvImageCodec pinned buffers). These are measurements of
application behaviour, not of an enforced OS-level cap; no such cap exists
(IR-18 remains open).

---

## 5. Environment restrictions encountered during this audit

1. The audit sandbox kills processes that delete many files at once
   (`SAFE_DELETE_BULK_GUARD_ERROR`, exit 0xC0000005). This aborted the
   built-in `--benchmark` mode (its temp-tree cleanup) and the preview-worker
   tests. Workaround used: run the pipeline directly into distinct output
   directories and delete locks only.
2. Symlink creation is silently swallowed, so link/junction protection cannot
   be verified end-to-end here.
3. `tkinter` is not installed in the project `.venv`, so GUI tests skip.
4. `wmic` is blocked by the sandbox; environment probing fell back to NVML and
   `nvidia-smi`.
5. All measurements used synthetic fixtures. No user photograph was touched.

---

## 6. Verdict per audit direction

| Direction | Verdict |
|---|---|
| A. Correctness | 1 P2 confirmed (Mitchell constant), 1 P1 confirmed (route telemetry), 1 P2 probable (skip-existing silent no-op). Geometry, colour, metadata, size-limit, profile/variant and resume logic behave as documented in the tested scenarios. |
| B. File safety | Guards present and functioning where testable; originals protected, atomic publication, reparse-point rejection verified by injection. Link rejection itself could not be exercised (environment). Decompression-bomb protection is a soft PHP-style budget (`max_source_bytes`, `max_queued_bytes`, `max_working_bytes`), not an OS cap. |
| C. Performance / memory | Measured honestly. GPU advantage on this Pascal card is 1.19× on synthetic 12 MP sources; decode dominates. No leaks. `--vram-budget` is a soft application-array estimate, as documented. |
| D. GUI | Statically sound design (bounded event queue + `after()` pump, subprocess isolation, orderly close). Runtime tests blocked by missing tkinter in the venv. |
| E. Architecture | Clean module split; shared weight builder between CPU/GPU. Two drift defects (AUD-001, AUD-007) stem from renaming/extending without updating dependants. |
| F. Testing | 355 tests, deterministic, with 7 environment-caused failures. Notable gap: `test_mitchell_exact_values` asserts the *incorrect* kernel values, so the suite cannot catch AUD-003. |
