# FastVLM GQA Short-Answer Results

Date: 2026-03-09

## Setup

- Model: `artifacts/models/FastVLM-0.5B.qualcomm.sm8750.auxmaskrope_runtime.litertlm`
- Runtime path: real LiteRT adb/NPU FastVLM path via
  `scripts/run_fastvlm_litert_gqa_eval.py`
- Pruning strategy: `prompt_conditioned_v1`
- Decode contract:
  - prompt suffix: `Respond exactly as: Answer: <one or two words>.`
  - regex: ` ?Answer: [A-Za-z0-9]+(?: [A-Za-z0-9]+)?`

## Completed 500-sample result

- Budget `96`
  - accuracy: `0.31` (`155 / 500`)
  - mean TTFT: `0.1761 s`
  - median TTFT: `0.1913 s`
  - mean prefill: `0.1644 s`
  - mean decode: `0.0579 s`
  - artifact: `artifacts/gqa_eval/gqa_quick500_20260308_204442/summary.json`

This replaces the earlier unconstrained short-answer run:

- Budget `96`, earlier contract
  - accuracy: `0.062` (`31 / 500`)
  - artifact: `artifacts/gqa_eval/gqa_quick500_20260308_184527/summary.json`

## Budget pilots

- Budget `160`
  - status: partial pilot stopped after `104` samples
  - accuracy: `0.2885` (`30 / 104`)
  - mean TTFT: `0.1921 s`
  - reason stopped: trailing the completed `96`-token result while already
    increasing TTFT
  - artifact:
    `artifacts/gqa_eval/gqa_budget160_pilot104_20260308_213550/summary.json`

- Budget `256`
  - status: `50`-sample pilot
  - accuracy: `0.32` (`16 / 50`)
  - mean TTFT: `0.2858 s`
  - artifact:
    `artifacts/gqa_eval/gqa_budget256_pilot50_20260308_214927/summary.json`

## Current takeaway

- The structured `Answer:` decode contract materially improves exact-match GQA
  accuracy on the real FastVLM LiteRT/NPU path.
- Budget `96` remains the best validated operating point so far because it has
  the only completed `500`-sample run and it keeps TTFT much lower than higher
  token budgets.
- Budget `160` does not justify a full exact sweep in its current form.
- Budget `256` may recover some quality, but its TTFT cost is substantially
  higher, so it is not currently the best paper point for edge deployment.
