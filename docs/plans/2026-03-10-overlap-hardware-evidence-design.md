# Overlap Hardware Evidence Design

## Scope
Build the remaining paper evidence around the existing FastVLM heterogeneous runtime:
- exact TTFT on the overlap path
- queue-aware per-request visual-token budgeting
- four benchmark variants for stream comparisons
- timeline and partition figures
- CPU/NPU utilization-style counters and wait/sync accounting
- energy/power collection when the device exposes usable battery rails

## Constraints
- Reuse the current FastVLM-only runtime and artifacts.
- No hidden fallbacks: if a hardware counter or power rail is unavailable, fail explicitly or mark it unavailable in the final report.
- Keep the overlap runtime authoritative for NPU-prefill/CPU-decode evidence.

## Recommended approach
1. Instrument the overlap runtime itself so TTFT and stage boundaries are emitted directly from the real heterogeneous path.
2. Add a queue-aware budget controller inside the overlap runner, but keep the controller policy simple and explicit: choose among allowed buckets using queue depth and rolling prefill/decode windows.
3. Build paper-facing analysis scripts around the emitted events instead of scraping ad-hoc logs.
4. Collect hardware counters outside the runtime only when the device surface is stable enough (e.g. `/proc`, `top`, battery/current sysfs). Treat unavailable counters as explicit missing evidence, not inferred values.

## Baseline set
The four stream variants will be:
1. `sequential_npu_npu_uniform`
2. `overlap_cpu_decode_npu_prefill_uniform`
3. `overlap_cpu_decode_npu_prefill_prompt_conditioned_v2`
4. `overlap_cpu_decode_npu_prefill_adaptive_prompt_conditioned_v2`

## Runtime evidence to emit
Per request:
- prepare start/done
- queue wait
- prefill start/done
- decode start
- first token timestamp / TTFT
- decode done
- overlap analysis
- controller decision (budget, reason, queue/decode inputs)

Global:
- overlap config
- process id for host-side sampling
- final stream done marker

## Hardware story
Primary metrics:
- TTFT
- makespan
- throughput
- overlap gain
- CPU decode busy time
- NPU prefill busy time
- sync/wait time
- copy/transfer counts from runtime-visible host/device handoff points

Secondary metrics if exposed by device:
- process CPU utilization sampled during the run
- battery current/voltage sampled during the run -> estimated power/energy

## Figures / outputs
- overlap timeline figure
- partition map figure
- baseline comparison table
- hardware counters table
- JSON/CSV machine-readable bundle for paper reuse
