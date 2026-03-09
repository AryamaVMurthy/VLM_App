# FastVLM Intelligent Pruning Design

## Goal

Replace the current uniform post-projection visual token pruning in FastVLM with an intelligent CPU-side policy that:
- ranks vision tokens by prompt-conditioned importance,
- keeps slight diversity across selected tokens,
- chooses among discrete budget buckets using runtime-visible queue/hardware signals,
- preserves the current working FastVLM model/export path.

## Locked Scope

This design is intentionally limited to the current working FastVLM branch.

Included:
- scoring at the existing post-projection seam after vision adapter output
- CPU-side token scoring and selection
- prompt-conditioned token ranking
- discrete budget selection from runtime-visible signals
- strong logging and test coverage

Excluded from this iteration:
- encoder-internal pruning
- NPU-fused scoring
- learned scorer training
- hidden fallback to the old uniform policy
- claims of real CPU-decode / NPU-prefill overlap in the current shared runtime

## Why This Design

The existing pruning seam is already correct and working. It has on-device evidence, preserves FastVLM contracts, and avoids destabilizing the shipped model bundle.

The right next step is therefore not to move pruning deeper into the model. It is to make token choice smarter while keeping the same seam and the same runtime bundle.

This follows the user-approved adaptation of EvoPrune:
- use prompt-conditioned token relevance,
- keep some diversity so redundant tokens do not dominate,
- use hardware/pipeline/queue signals to choose the budget,
- start with a CPU implementation before attempting NPU fusion.

## System Design

### 1. Token Selection Policy

Current behavior:
- select tokens uniformly by position

New behavior:
- compute a per-token score after the vision adapter output
- keep the top tokens under the selected budget using a slightly diversity-aware policy

The v1 score is:
- prompt similarity term
- token salience term
- redundancy penalty term

Form:
- `score_i = a * prompt_similarity_i + b * salience_i - c * redundancy_penalty_i`

#### 1.1 Prompt Similarity

Prompt similarity is computed on CPU from the prompt token ids already present in `ProcessAndCombineContents`.

The practical v1 implementation is:
- derive a prompt feature vector from the text token ids using a deterministic hashed random projection into the same dimensionality as the vision embeddings
- L2-normalize the prompt vector
- L2-normalize each vision token vector
- compute cosine-style similarity between each vision token and the prompt vector

Why this approach:
- no model export change
- no new compiled model
- no dependency on running the embedder model just for scoring
- deterministic and cheap enough for CPU-side scoring

This is not the final semantic scorer. It is the first prompt-conditioned systems implementation.

If this is not accurate enough, the next scorer upgrade path is an attention-proxy scorer at the same seam.

#### 1.2 Salience

Salience is computed from the projected vision token itself.

v1 salience features:
- L2 norm of the token embedding
- optional centered norm relative to the image-token mean embedding

The default v1 salience term is the normalized token L2 norm. This is cheap, deterministic, and available from the current tensor.

#### 1.3 Redundancy Penalty / Slight Diversity

Selection should not keep many near-duplicate tokens from the same region.

v1 diversity behavior:
- sort tokens by base score (`prompt_similarity + salience`)
- greedily accept tokens in descending score order
- penalize candidates that are too similar to already selected tokens using cosine similarity
- keep this penalty slight, not dominant

This creates a diversity-aware top-K without introducing clustering or a heavy pairwise algorithm.

### 2. Budget Controller

The budget controller does not decide which tokens are important. It decides how many tokens to keep.

Budgets are discrete buckets, not arbitrary counts.

Initial buckets:
- `96`
- `160`
- `256`

Rationale:
- the current measured system already shows discrete latency buckets
- bucketed control is more stable and easier to analyze for CASES

#### 2.1 Controller Inputs

Use runtime-visible signals only in v1:
- queue depth
- queued-image count
- recent prefill latency
- recent decode latency
- recent prefill/decode throughput
- CPU/NPU idle gap if available from runtime timing

No OS thermal/frequency/energy telemetry is required in v1.

#### 2.2 Controller Policy

Hierarchical policy:
1. choose a base bucket from image/prompt demand
2. adjust for queue pressure and queued-image count
3. refine toward recent decode completion timing when the system is stable

In the current branch, real overlap is blocked. Therefore:
- the controller must not claim true overlap matching
- it may still expose the timing target logic and logs using measured prefill/decode timings from sequential runs or the benchmark harness

### 3. Where It Runs

#### Runtime scorer
- runs inside the existing CPU-side post-projection pruning seam
- code paths:
  - `third_party/litert-lm/runtime/core/session_basic.cc`
  - `third_party/litert-lm/runtime/framework/resource_management/execution_manager.cc`
  - `third_party/litert-lm/runtime/util/executor_data_util.cc`

#### Budget controller
- first integration can be at the benchmark/orchestration layer and then optionally exposed in runtime config
- must emit chosen bucket and reason codes

### 4. Logging / Observability

Every intelligent-pruning decision must be logged explicitly.

Per request log fields:
- selected budget bucket
- original token count
- kept token count
- scorer kind (`uniform`, `prompt_similarity_v1`, later `attention_proxy_v2`)
- prompt-similarity weight
- salience weight
- diversity penalty weight
- top selected token indices
- reason codes for the budget controller

No hidden fallback:
- if prompt-conditioned scoring cannot run, return an explicit error
- do not silently fall back to uniform pruning

### 5. Evaluation Gates

#### Functional
- selected token count matches chosen budget
- selection order is deterministic
- selected indices are valid and increasing when passed to slicing
- combined multimodal token count remains consistent

#### Quality
- exploratory tolerance: do not regress by more than roughly `5-10%` versus the unpruned or current safe baseline during early experiments
- if v1 scorer quality is not good enough, escalate the scorer only, not the seam

#### Performance
- verify TTFT improvement against bucket baselines
- verify scorer overhead is small relative to prefill savings

## Implementation Path

1. Add explicit pruning strategy/config types
2. Add failing unit tests for prompt-conditioned token selection and diversity-aware top-K
3. Implement CPU-side prompt-conditioned scorer in `executor_data_util`
4. Wire the scorer into both runtime paths
5. Add structured logging for scorer outputs and budget reasons
6. Add a discrete budget controller utility
7. Integrate the controller into the benchmark/orchestration path first
8. Run on-device sweeps and manual verification

## Failure Rules

- If scorer inputs are malformed, error out.
- If token slicing indices are invalid, error out.
- If controller selects a bucket outside the allowed set, error out.
- If evaluation shows unacceptable degradation, keep the old behavior available only as an explicit, user-chosen strategy flag, not a silent fallback.
