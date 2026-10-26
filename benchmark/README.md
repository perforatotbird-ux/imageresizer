# Benchmark harness

Reproducible throughput and latency checks for the BatchResize pipeline.

## Corpus

Generate the deterministic synthetic corpus (24 large JPEGs by default):

```powershell
.\.venv\Scripts\python.exe benchmark\make_corpus.py
```

The generator writes 5760x3840 and 3840x5760 images plus `manifest.json`.
`benchmark/corpus/` is ignored by git. The measured reference run is stored in
`benchmark/reference-results.json`.

For product-representative numbers, point `--input` at the real photo folder
instead. The performance reports in the repository root use the real 517-file
corpus in `input/`.

## Run

```powershell
.\.venv\Scripts\python.exe benchmark\run_bench.py --label cpu-web1024 `
  --input K:\Test\recoder\input --repeats 3 -- --backend cpu --workers 8 --size web_1024
```

Run a GPU comparison:

```powershell
.\.venv\Scripts\python.exe benchmark\run_bench.py --label gpu-web1024 `
  --input K:\Test\recoder\input -- --backend gpu --workers 8 --size web_1024
```

Arguments after `--` are passed through to `python -m batchresize`. Do not pass
`--input`/`--output` there. Each repeat uses a fresh output folder under
`benchmark/results/_runs/<label>`; median wall time, pipeline elapsed time,
throughput and stage timings are written to `benchmark/results/<label>.json`
and `.md`.

## Interpretation

- `elapsed_s` is the pipeline's own measured time; `wall_s` includes Python
  startup and, for the GPU backend, CUDA/nvImageCodec context creation.
- Use at least 48 files for steady-state CPU/GPU comparisons; smaller sets are
  dominated by startup.
- Runs are warm-cache. Cold-cache numbers require a reboot or a cache flush and
  are not comparable across runs.
- Synthetic gradients decode differently from camera JPEGs. Use a real corpus
  for release decisions; use the synthetic corpus for regression comparisons.
