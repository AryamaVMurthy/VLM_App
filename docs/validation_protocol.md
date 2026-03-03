# Validation Protocol (Phase 1)

## Functional Smoke

1. Device compatibility gate:
- Launch on `SM8750P`/SDK35: must pass.
- Launch on unsupported device: must fail fast with explicit message.

2. Model provisioning:
- Fresh install: must download pinned artifact and verify SHA256.
- Existing valid artifact: must skip download and activate.

3. Live VQA:
- Camera preview remains active during inference.
- Typed question returns streamed answer.

4. Single-turn:
- Consecutive asks do not include prior conversation context.

## Failure and Fallback

1. Preferred backend failure:
- Must show fallback notice.
- Must include `fallback_reason`.
- Must retry on CPU and emit completion/error explicitly.

2. Checksum mismatch:
- Must fail fast with remediation text.

3. Missing artifact file:
- Must re-download deterministically.

## Performance

1. Run 10x benchmark from app UI (`Run 10x`).
2. Validate benchmark JSON and text summary are created.
3. Validate each request includes `ttft_ms` and timing fields.

## Stability

1. Run repeated asks and benchmark in one app session.
2. Confirm no crashes/ANRs.
3. Confirm cancel action stops active request.

## Current Verified Blocker

- NPU warmup currently aborts process in native LiteRT-LM library on this tested device/runtime combination.
- CPU fallback for this pinned Qualcomm artifact currently fails with `Input tensor not found`.
- Therefore complete successful answer generation is blocked pending runtime patch/update, despite app lifecycle, model provisioning, camera flow, and observability being implemented.
