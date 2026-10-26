# AUDIT_REPORT_CURRENT.md

Independent technical audit of `batchresize / imageresizer`.
Commit `b4fc6b6`, branch `feature/mvp-0-3`, working tree otherwise clean.
No source file, test, configuration or user photograph was modified during
this audit.

Machine-readable findings: `AUDIT_FINDINGS.json`.
Test commands and raw results: `AUDIT_TEST_RESULTS.md`.

---

## 1. Executive summary

The project is in **working but imperfect** condition. The core pipeline
(discover → read → decode → resize → encode → publish) produces correct
geometry, honours EXIF orientation, strips/keeps metadata as documented, keeps
originals safe, publishes atomically, resumes verified results and skips
unverified ones. Memory is stable across repeated batches.

Three things need attention before this is trustworthy as a daily tool:

1. **`report.json` mislabels the compute path.** Every run — GPU included —
   reports `decode: OpenCV/Pillow CPU`, `resize: Pillow/CPU float`,
   `encode: OpenCV CPU JPEG`, because the route derivation still looks for the
   old labels `gpu`/`hybrid` while the engine now writes `nvimgcodec-auto`
   (AUD-001). Nobody can tell from the report what actually ran.
2. **A leftover `.batchresize.lock` permanently blocks the output folder**
   with an unhelpful `[Errno 17] File exists` (AUD-002). Any abnormal
   termination leaves one behind.
3. **The `mitchell` filter is not the Mitchell kernel** — a wrong constant
   term (AUD-003), and the existing unit test *asserts the wrong numbers*, so
   the suite is blind to it.

Also note that the throughput figures documented in `README.md` and
`REPORT.md` (1.8–2.2×, 46–62 files/s) do not reproduce at HEAD: measured
**1.19×** (CPU 12.0 files/s, GPU 14.3 files/s) on 24 synthetic 1800×1200
images. The GTX 1060 has no hardware JPEG decoder; decode dominates the GPU
path.

The three known limitations flagged in the development plan are still open and
are honestly documented in code and reports (entropy-level integrity, hard
native RSS/VRAM caps, GPU-only attestation).

---

## 2. Recorded state

| Item | Value |
|---|---|
| Repository root | `K:/Test/recoder` |
| Branch / commit | `feature/mvp-0-3` / `b4fc6b61af92f018b2694113c4f44f003e63e995` |
| Working tree | only `AUDIT_REPORT.md`, `FIX_REPORT.md` deleted (recoverable), 1 untracked PDF |
| OS / CPU / RAM | Windows 10 22H2 / Xeon E5-2670 v3, 24 threads / 15.8 GiB |
| GPU | GeForce GTX 1060 6 GB, SM 6.1 (Pascal), driver 581.57 |
| Python | 3.13.14 in `.venv`; tkinter **not** installed in the venv |

`--doctor` passes: GPU engine self-test (decode → resize → encode) OK, CPU
engine OK.

---

## 3. Architecture and data flow (as implemented)

```
cli.py (argparse → RunOptions)
  ├── profiles.py / variants.py   (task profiles, extra outputs)
  └── pipeline.Pipeline
        discover() ──► reader threads ──► bounded jobs queue
                                         ├── cpu: N processor threads (OpenCV/Pillow)
                                         ├── hybrid: 1 thread (host decode → GPU resize)
                                         └── gpu: 1 thread (nvImageCodec decode → CUDA
                                                    resize/encode; CPU hybrid fallback)
        bounded results queue ──► writer threads ──► atomic_write (temp + os.replace)
                                         └── safety.ResumeState (.batchresize-state.json)
```

Supporting modules: `geometry.py` (contain/cover/pad, anchors, focus),
`resampling.py` (filters + weight tables shared by both backends),
`color_management.py`, `imageio_meta.py` (EXIF/ICC, truncation, alpha hint),
`safety.py` (roots, links, lock, resume), `size_limit.py` (quality search),
`sharpen.py`, `observability.py`, `preview.py`/`preview_worker.py` (subprocess
preview), `gui.py` + `ui_view.py`/`ui_generated.py` (Tk UI), `launcher.py`.

Documentation drift worth noting: CLI defaults are pinned to this machine
(`-i K:\Test\recoder\input`, `-o K:\Test\recoder\output`).

---

## 4. Findings

Priority legend: P0 = data loss risk, P1 = serious defect, P2 = moderate,
P3 = minor.

### AUD-001 — `report.json` always reports CPU routes (P1, CONFIRMED)

* **Where:** `pipeline.py:1126` (label `"nvimgcodec-auto"`) and
  `pipeline.py:1477-1479` (`actual_routes` derivation testing for
  `"gpu"`/`"hybrid"`).
* **Expected:** `actual_routes` reflects the engine that produced the results.
* **Actual:** a run where 24/24 files were processed by the GPU engine reports
  `decode: "OpenCV/Pillow CPU"`, `resize: "Pillow/CPU float"`,
  `encode: "OpenCV CPU JPEG"`.
* **Evidence:** measured `backend_counts={'nvimgcodec-auto': 24}` together with
  those CPU route strings.
* **Cause:** label rename in Stage 8 without updating the derivation.
* **Impact:** every report, benchmark and the GUI's "движок" readout mislabel
  the path; a real CPU fallback is indistinguishable from a GPU run.
* **Fix:** one canonical name set used by both writers and readers + a unit
  test that asserts routes for each engine.

### AUD-002 — Stale `.batchresize.lock` blocks the output folder (P1, CONFIRMED)

* **Where:** `safety.py:96-107`.
* **Reproduction:** `mkdir out; echo 1234 > out/.batchresize.lock; run …` →
  `error: [Errno 17] File exists: …\.batchresize.lock`, exit code 2.
* **Impact:** any crash/kill leaves the lock; the folder cannot be used again
  until the user finds and deletes a hidden dot-file. Observed repeatedly
  during this audit alone.
* **Fix:** PID liveness check + age-based takeover, an explicit `--unlock`,
  and an actionable message naming the file.

### AUD-003 — Mitchell kernel constant 16/3: **RETRACTED (false positive)**

The original audit claimed the constant should be `5` instead of `16/3`.
Verification during the remediation pass showed the **implementation was
correct** and the proposed change would have been a regression:

| check | code (16/3) | proposed "fix" (5) |
|---|---|---|
| DC gain `k(0)+2k(1)+2k(2)` | **1.000000** | 0.944444 |
| brightness effect of a 1:1 resize | preserved | **−5.6 % darkening** |
| cubic B-spline anchor (B=1,C=0) | exact match | does not match |
| continuity at \|x\|=1 | continuous (0.0556 both sides) | discontinuous |

`16/3` is `6 − 2B` for B=1/3, which satisfies the Mitchell–Netravali anchors;
the textbook transcription `(7x³ − 12x² + 5)/6` used by the audit breaks DC gain.
The existing assertions in `tests/test_resampling.py` (`k(0)=8/9`, `k(1)=1/18`)
are correct and were left untouched. A DC-gain guard was added to the test file
so this cannot regress.

*(Retraction recorded in `AUDIT_FINDINGS.json` as AUD-003/RETRACTED.)*

### AUD-004 — Preview tests fail only in the full suite (P2, environment)

`tests/test_preview.py` fails all six tests after other tests have run,
because the audit sandbox kills the preview worker subprocess during bulk
deletion (`SAFE_DELETE_BULK_GUARD_ERROR`, 0xC0000005). Passing in isolation.
Not a product defect on Windows, but the worker's per-request temp churn is a
robustness smell worth reducing.

### AUD-005 — Symlink guard untestable in this sandbox (P2, environment)

Symlink creation is silently swallowed; the guard in `safety.py` was verified
by monkeypatching `is_symlink` and works. The test should skip with a reason
when the sandbox swallows the link.

### AUD-006 — Documented performance no longer reproduces (P3, CONFIRMED)

`README.md`/`REPORT.md` claim 1.8–2.2× at 46–62 files/s. At HEAD: CPU
12.0 files/s, GPU 14.3 files/s (**1.19×**) on 24 × 1800×1200 → 1024 px. The
documents describe the earlier prototype pipeline.

### AUD-007 — Quality preset aliases duplicated (P3, CONFIRMED)

`high=light=90`, `balanced=medium=80`, `compact=strong=65`; `--help` omits
`maximum=95` and `small=60`.

### AUD-008 — `--skip-existing` with a changed config is a silent no-op (P2, PROBABLE)

Measured `processed=0, skipped=0` when only quality changed; the code path
records `FileExistsError("… use --overwrite")` per output. Either regenerate
or say so in the flag's help.

---

## 5. What was verified as correct

* Geometry: contain/cover/pad, aspect preservation, no upscale by default,
  cover crops the exact box, anchors/focus honoured in the CLI contract.
* EXIF orientation: nvImageCodec applies it (verified against
  `PIL.ImageOps.exif_transpose` for all 8 values) and the output tag is reset
  to 1.
* Metadata: personal-data stripping removes GPS/artist/serial numbers while
  preserving Orientation and ICC (verified by re-reading the output).
* Size limit: output bytes stay under the limit and `target_size_achieved` is
  truthful; unreachable limits fall back to `min_quality` and are reported.
* Resume: outputs are verified by config digest + source hash + output hash +
  JPEG decode; changing any parameter invalidates them.
* Variants: one run can emit multiple sizes into separate subdirectories.
* Filters: `area`/`box` average, `nearest` preserves, `bilinear`/`lanczos3`
  smooth — checked with a 1-pixel checkerboard.
* Memory: no RSS growth over 4 repeated batches on either backend.
* Launch safety: roots, aliasing, hard links, reparse points and exclusive
  publication are implemented; the guard logic was exercised by injection
  where the environment allowed.
* GUI: bounded event queue + `after()` pump, subprocess isolation, orderly
  close; the module imports, constructs a window and closes cleanly under
  Python 3.10 with tkinter.

---

## 6. Unverified areas and why

| Area | Status | Reason |
|---|---|---|
| GUI interactive behaviour | NOT TESTED | tkinter absent from the project venv; only a construct/close smoke test on Python 3.10 |
| Hardware nvJPEG decode/encode | NOT TESTED | the library may route to CPU plugins; `codec_backend_attested` is False by design |
| JPEG entropy integrity | NOT TESTED | not implemented (IR-17 open) |
| Hard native RSS/VRAM cap | NOT TESTED | not implemented; budgets are soft estimates (IR-18 open) |
| Symlink/junction rejection end-to-end | BLOCKED | sandbox swallows symlink creation |
| User's real photos | NOT TESTED | audit used synthetic fixtures only |

---

## 7. Risk assessment

| Risk | Likelihood | Impact | Notes |
|---|---|---|---|
| A failed/killed run blocks the output folder | high | medium | AUD-002; workaround is deleting a hidden file |
| Reports show the wrong engine | certain (every run) | medium | AUD-001; masks genuine fallbacks |
| Mitchel filter differs from documentation | certain | low | AUD-003; visual impact subtle |
| Silent no-op with `--skip-existing` + changed config | medium | medium | AUD-008; may surprise users |
| Originals overwritten or lost | low | high | guards present; link rejection unverified here |
| Unbounded memory on huge inputs | low | high | soft budgets only; 100 MP cap via `validation.dimensions` |

Overall: **P1 issues are telemetry and recoverability, not data corruption.**
No evidence of original-file corruption was found in any tested scenario.

---

## 8. Recommended remediation order

1. **AUD-001** — one source of truth for route labels; assert routes per
   engine in a unit test. (30 lines of code + test)
2. **AUD-002** — stale-lock detection (PID liveness / age), `--unlock`, clear
   error text.
3. **AUD-003** — Mitchell constant + corrected reference test.
4. **AUD-008** — regenerate (or clearly document) on config change with
   `--skip-existing`.
5. **AUD-004/005** — make the two environment-sensitive tests skip with a
   reason; reduce preview worker temp churn.
6. **AUD-006/007** — refresh README/REPORT numbers and preset help text.
7. Optional follow-ups (not started): real entropy-level JPEG validation
   (IR-17), a hard memory cap via a Windows job object (IR-18), and hardware
   codec attestation (IR-26).

---

## 9. Regression-test matrix for the fixes

| Finding | Acceptance test |
|---|---|
| AUD-001 | run each engine on a fixture; assert `actual_routes` matches and `backend_counts` label |
| AUD-002 | dead-PID lock → run succeeds; live-PID lock → refusal with actionable message |
| AUD-003 | `mitchell_kernel(0) == 5/6` plus a 9-point reference table; CPU/GPU agreement within 40 dB on a photo fixture |
| AUD-008 | config change with `--skip-existing` either reprocesses all or fails with the documented message |
| AUD-004/005 | preview suite passes when run twice standalone; link test skips when the sandbox swallows symlinks |
| AUD-006 | any published throughput number must be reproducible by the documented command |

---

# Remediation pass — 2026-10-21 (lowest-risk items, one at a time)

Only the cheapest findings were fixed in this pass; each change was followed by
a targeted test run of the affected function. Findings and their new status are
in `AUDIT_FINDINGS.json`.

| Finding | Status | What changed | Verification |
|---|---|---|---|
| AUD-001 (P1, routes mislabelled) | **FIXED** | `pipeline.actual_route_flags()` is now the single place that maps recorded backend labels to route strings; `report.json` derives from it | unit table for all labels + end-to-end runs: `cpu` → CPU routes, `gpu` → GPU routes, `hybrid` → CPU decode + GPU resize/encode; `tests/test_pipeline.py` gained `test_actual_route_flags_maps_backend_labels` and `test_report_actual_routes_follow_backend` |
| AUD-006 (P3, stale perf docs) | **FIXED** | `REPORT.md` gained a re-measured section (CPU 12.0, GPU 14.3 files/s, 1.19×) and the old tables are explicitly marked historical | docs only |
| AUD-007 (P3, preset aliases) | **FIXED** | `--quality` help lists all ten presets with duplicates marked as aliases | `--help` output + `tests/test_cli.py` (14 passed) |
| AUD-008 (P2, silent no-op) | **FIXED (explicit)** | `--skip-existing` help now states that unverified/changed outputs are reported as errors; the per-output error message says the same | measured run: unchanged → 4 skipped; quality changed → 4 failed with the explicit message; `tests/test_cli.py` + `tests/test_audit_regressions.py` (70 passed, 1 skipped) |
| AUD-003 (P2, Mitchell) | **RETRACTED — false positive** | no code change; a DC-gain guard test was added instead | `tests/test_resampling.py` (29 passed), see the retraction note above |

### AUD-002 (P1, stale output lock) — **FIXED** (second remediation step)

* `safety.ResumeState.acquire()` now records a payload of
  `{"pid": <pid>, "token": <process start time>}` instead of a bare PID.
* `lock_is_live()` decides liveness from the token plus the process **exit
  code** (`GetExitCodeProcess`) — `OpenProcess` alone succeeds even for exited
  Windows processes, which is exactly why a naive PID check would never clear
  anything.
* Behaviour: a lock whose owner is dead (unknown PID, or PID reused by an
  unrelated process) is taken over automatically and the run continues. A lock
  genuinely held by a live instance is still refused, now with a message that
  names the PID and tells you about `--unlock`.
* New CLI flag `--unlock` removes a stale lock before running and **never**
  removes a live one (`clear_stale_lock` raises in that case).
* Legacy plain-PID lock files are still understood; a dead-PID legacy lock is
  cleared, a live one is refused.

Verification: 10 new regression tests in `tests/test_audit_regressions.py`
(dead-PID takeover, reused-PID takeover, live-lock refusal from another
process, `--unlock` semantics, and a PID-liveness test that fails if
`OpenProcess` is used without the exit-code check). Full suite after the
change: **365 tests, 7 failures — all seven are the pre-existing sandbox
artefacts (6 preview + 1 symlink)**, 4 skips. A two-process test confirmed the
concurrent run is refused (exit 2) and writes nothing, while the run after the
holder's death succeeds and releases the lock.

**Left open:** AUD-004/005 are sandbox artefacts of this audit environment, not
product defects; they need a normal Windows machine to settle.

---

## Appendix: why a resize to 80 % with quality 95 produced ~3x larger files

Measured on the real delivery set (`input/02`, `input/03`: 7 files, 30 MP and
6 MP sources, 8250 KB in total) with the CPU backend.

### What the sources are

| Property | Value |
|---|---|
| Format | baseline/progressive JPEG, 4:4:4 (no chroma subsampling) |
| Quantisation tables | match the libjpeg tables of **q=59** |
| Bit rate | 0.21 - 0.65 bit/pixel (extremely efficiently compressed) |
| Encoder | libjpeg family with **optimized Huffman tables** (re-encoding the decoded pixels at q59/4:4:4 without optimized Huffman is already 7-52 % larger; with optimized Huffman it matches the original) |

### Where the growth comes from

| Step (80 % scale) | Total | vs. original |
|---|---|---|
| originals | 8250 KB | 1.00x |
| re-encode only, no resize, q95 4:4:4 | 19366 KB | **2.35x** |
| + resize to 80 %, q95 4:4:4 (the reported case) | 24330 KB | **2.95x** |
| resize to 50 %, q95 4:4:4 | 11068 KB | 1.34x |

1. **q=95 is the largest possible JPEG encoding.** `auto_subsampling(95)`
   returns 4:4:4 (`presets.py:73`), so the GUI preset "Maximum - q=95" with
   subsampling "auto" encodes full chroma resolution at the highest quality.
   Re-encoding the sources at that setting already costs **2.35x** before a
   single pixel is resized.
2. **A downscale does not shrink a JPEG proportionally.** JPEG codes fixed
   8x8 blocks; at 80 % each output block covers 1.56x more scene area, so the
   high-frequency energy per pixel grows (Laplacian variance 87 -> 145, bytes
   per pixel 0.176 -> 0.349). The result is *larger* than the un-resized
   re-encode (+21 %). This is inherent to JPEG, not a defect of the
   resampler: PIL Lanczos, the support-scaled separable kernel, cv2 INTER_AREA
   and box averaging all land within 0.5 % of each other.
3. Filters, sharpening, EXIF/ICC and personal-data stripping are irrelevant for
   the size (all within +-1 %).

### Measured size matrix (80 % scale, 7 files, 8250 KB total)

| Settings | Total | vs. original |
|---|---|---|
| q95, auto (4:4:4) - the reported case | 24330 KB | 2.95x |
| q95, 4:2:0 | 19342 KB | 2.34x |
| q90, auto (4:2:2) | 14152 KB | 1.72x |
| q90, 4:2:0 | 12905 KB | 1.56x |
| q85, auto (4:2:0) | 10002 KB | 1.21x |
| q80, auto (4:2:0) | 8363 KB | 1.01x |
| q75, auto (4:2:0) | 7216 KB | 0.87x |
| q80, 4:2:0, optimized Huffman (after AUD-009 fix) | 8021 KB | 0.97x |
| q85, 4:2:0, optimized Huffman (after AUD-009 fix) | 9718 KB | 1.18x |
| `--max-file-size 1MB --min-quality 40` (any q) | 7108 KB | 0.86x (max file 1455 KB, warning logged) |

### Defect found and fixed: AUD-009

`--optimize-huffman` (GUI checkbox "Optimize Huffman") was plumbed only into
the GPU engine; the CPU encoder ignored it completely and the outputs were
byte-identical with and without the flag. The CPU path now sets
`IMWRITE_JPEG_OPTIMIZE` (baseline JPEG is preserved, verified via SOF0). The
saving depends on the coefficient statistics: -22..-25 % when no resize
happens, -1.8 % on downscaled content.

### Recommendation

For downscaled delivery files use 4:2:0 (either explicitly or via the
"auto" rule up to q=85) and q=80-q=85; that is the only combination that
lands at or below the size of the sources. If a hard size ceiling is needed,
`--max-file-size` with `--min-quality` is the built-in mechanism - but note
that the GUI does not expose it yet.

### Feature: quality preset "same as source" (--quality source)

Added after the file-size analysis above: the user asked for a mode in which
nothing but the geometry changes.

* New module `batchresize/source_quality.py` (pure Python, no numpy/PIL):
  parses the DQT segments and the SOF header of a JPEG, returns the
  quantisation tables in natural order, the chroma layout and the nearest
  standard quality. The estimator is exact for every libjpeg quality
  (1..100) and works for custom tables as well (the delivery set uses
  per-image optimised tables, which no standard quality number reproduces).
* The pipeline resolves the profile per file (`Pipeline._resolve_source_profile`),
  so a batch may mix sources with different qualities and layouts.
* The CPU encoder writes the tables verbatim through Pillow (OpenCV cannot);
  the GPU encoder accepts the parameter but falls back to the quality number
  because nvJPEG cannot write custom tables.
* Measured on the delivery set (7 files, 30 MP, resize to 80 %):
  q95/4:4:4 -> 2.95x the original size, "source" -> **0.79x**, with the
  quantisation tables of every source byte-identical in the output and a
  PSNR of 39-45 dB against an independent Lanczos reference.
* The GUI exposes the preset as "Как у исходника · только resize"; the
  quality slider is disabled in that mode and keeps its value as the
  fallback for sources without JPEG tables.
* Tests: `tests/test_source_quality.py` (19 cases) plus GUI regressions in
  `tests/test_gui_modern.py`.
