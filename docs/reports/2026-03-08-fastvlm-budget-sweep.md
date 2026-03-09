# FastVLM Fixed-Budget Sweep Report

Date: 2026-03-08

## Command

```bash
env -u PYTHONHOME -u PYTHONPATH python3 scripts/fastvlm_cases_benchmark.py \
  --model artifacts/models/FastVLM-0.5B.qualcomm.sm8750.auxmaskrope_runtime.litertlm \
  --images IMAGES/person.jpeg IMAGES/download.jpeg IMAGES/images.jpeg \
  --budgets 96,128,160,192,224,256 \
  --warmup-runs 0 \
  --measured-runs 1 \
  --skip-build 1 \
  --max-num-tokens 384 \
  --answer-mode short \
  --output-dir artifacts/cases_benchmark/budget_sweep_20260308
```

## Output Artifacts

- raw benchmark directory:
  - `artifacts/cases_benchmark/budget_sweep_20260308/`
- HTML report:
  - `artifacts/cases_benchmark/budget_sweep_20260308/report.html`
- summary CSV:
  - `artifacts/cases_benchmark/budget_sweep_20260308/summary.csv`
- measured runs CSV:
  - `artifacts/cases_benchmark/budget_sweep_20260308/measured_runs.csv`

## Median Results By Budget

| budget | TTFT ms | prefill latency us | prefill tok/s | decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 96  | 101.029 | 90,218  | 1418.79 | 93.6998 |
| 128 | 190.511 | 179,875 | 1423.21 | 95.0221 |
| 160 | 189.917 | 179,515 | 1426.06 | 92.8665 |
| 192 | 190.882 | 180,050 | 1421.83 | 93.0103 |
| 224 | 190.756 | 180,041 | 1421.90 | 93.7583 |
| 256 | 280.211 | 269,592 | 1424.37 | 94.0174 |

## Headline Findings

1. Budget `96` cuts median TTFT from `280.211 ms` to `101.029 ms` relative to budget `256`.
   - TTFT reduction: `63.9%`
   - prefill latency reduction: `66.5%`
2. Prefill tokens/sec stays almost flat across budgets.
   - `1418.79` at budget `96`
   - `1424.37` at budget `256`
3. Decode tokens/sec stays almost flat across budgets.
   - the observed benefit is from reducing prefill work, not from changing decode speed
4. Budgets `128` through `224` all land on the same prefill token count bucket.
   - observed prefill tokens: `256`
   - their TTFT values are therefore nearly identical
5. Budget `96` lands on a smaller prefill token count bucket.
   - observed prefill tokens: `128`
   - this is why it produces the large TTFT drop

## Observed Prefill Token Buckets

From `artifacts/cases_benchmark/budget_sweep_20260308/measured_runs.csv`:

- budget `96` -> prefill tokens `128`
- budgets `128,160,192,224` -> prefill tokens `256`
- budget `256` -> prefill tokens `384`

The controller should treat these as discrete performance buckets unless quality measurements justify a different choice.

## Manual Sanity Check On Repo Images

The captions remained semantically correct across the sweep:
- `IMAGES/person.jpeg`: seated man on a bench, reflective posture, tree/background scene
- `IMAGES/download.jpeg`: forest path through dense trees/greenery
- `IMAGES/images.jpeg`: bowl with bananas, oranges, and peppers/fruits

## Long-Mode Spot Check

Command output directory:
- `artifacts/cases_benchmark/long_compare_20260308/`

Person image results:
- budget `96`: TTFT `100.438 ms`, decode tokens `25`
- budget `256`: TTFT `279.887 ms`, decode tokens `25`

Interpretation:
- current short/long mode scaffolding changes output length pressure, not decode backend identity
- true decode-mode switching still needs separate compatible decode exports

## What This Means For The Paper

This branch already supports one strong systems claim:
- post-projection visual token budgeting materially reduces TTFT on device, and the gain is explained by smaller prefill work rather than by a measurement artifact

The next paper-critical step is to connect these discrete prefill buckets to a real overlap scheduler once the runtime can support honest CPU-decode / NPU-prefill concurrency.
