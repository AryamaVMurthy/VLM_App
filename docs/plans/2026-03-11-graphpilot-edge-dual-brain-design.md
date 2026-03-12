# GraphPilot-Edge Dual-Brain Design

## Goal

Implement GraphPilot-Edge as a two-brain heterogeneous runtime on SM8750:
- an offline brain that builds feasibility matrices, profiles stages and macro-regions, calibrates cost models, enumerates plans, and ranks them with explicit formulas
- an online brain that executes the chosen plan on device, reacts to thermal and memory pressure, and preserves support-safe backend placement without hidden fallback

## Architecture

GraphPilot-Edge operates over three workflow DAGs:
- workflow A: ASR -> Planner -> Responder -> TTS
- workflow B: ASR -> Planner -> VLM -> Responder -> TTS
- workflow C: ASR -> Planner -> (VLM || Retrieval) -> Responder -> TTS

Heavy stages are opened into macro-regions:
- FastVLM: vision encoder, projector/bridge, pruning seam, multimodal prefill, decode/head, aux/mask
- responder LLM: prefill, decode, optional coarse block groups for prefill-only exploration

The optimizer must reason jointly about:
- support-safe backend feasibility across CPU/GPU/NPU
- stage and macro-region placement
- transfer overheads and zero-copy opportunities
- responder prefill/decode split and KV migration cost
- workflow streaming and branch overlap
- interval-based memory reuse and KV admission/degradation
- thermal drift and plan-bank switching

## Objective and constraints

Candidate plan objective:

J(Pi) = alpha * P95(T_e2e)
      + beta * P95(T_TFS)
      + gamma * E_bar
      + delta * M_peak
      + eta * B_copy
      + zeta * Q_loss

Hard constraints:
- M_peak <= M_budget
- Q_loss <= epsilon
- every assigned stage or macro-region must be support-safe

## Cost model

Execution time:

T_i(v, b, x, theta, q) = T_hat_i(v, b, x)
                        * rho_b(theta)
                        * kappa_b(q)
                        + 1_cold * C_compile_i(v, b)

Transfer cost:

C_{i->j}(S, b_i, b_j) = 0 if same backend and compatible memory
                      = C_map(S) if zero-copy shared buffer is available and profitable
                      = tau0_{b_i,b_j} + S / BW_{b_i,b_j} + tau_layout otherwise

Contention:

kappa_b(q) = 1 + sum_{b'} lambda_{b,b'} * u_{b'}

Energy proxy:

E(Pi) = sum_i Pbar_i * T_i + P_idle * T_wall

Responder split:

T_resp = T_prefill + T_switch + N_out * t_decode

T_switch = 0 if b_prefill == b_decode
         = tau0 + M_KV / BW_{b_prefill,b_decode} otherwise

KV cache size:

M_KV = 2 * L * H_kv * D_head * T * B_dtype

## Search strategy

- exhaustive enumeration at stage level
- beam search when macro-regions are opened
- discrete-event simulation for top-K candidate validation
- HEFT-style upward rank plus runtime bonuses/penalties for the online scheduler
- plan-bank states: cool, warm, hot, lowmem

## Runtime policies

- preserve the existing FastVLM LiteRT NPU path as the primary VLM path
- keep decode sticky unless KV migration thresholds are clearly favorable
- make fallback explicit and user-visible; no silent degradation
- use interval-based buffer allocation and explicit KV admission/degradation rules
- allow VLM and retrieval branch overlap when the overlap gain exceeds contention and merge overhead

## Scope for the current implementation batch

This batch upgrades the offline brain and calibration stack to match the design:
- formula-driven cost model and objective scoring
- stage-level exhaustive ranking under J(Pi)
- calibrated orchestration and thermal factors from measured runs
- scheduler scoring utilities aligned with HEFT + first-output/copy/memory/thermal terms
- hyperparameter tuning scripts for objective and scheduler weights
- artifact-pack integration for calibration and tuning outputs

The online Android runtime remains the execution target and source of truth for measured backend behavior.
