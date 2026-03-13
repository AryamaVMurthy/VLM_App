# Introduction

Edge assistants are not single-model pipelines. A practical on-device assistant mixes ASR, planning, multimodal reasoning, retrieval, response generation, and speech synthesis on one thermally constrained mobile SoC. The scheduling problem is therefore not “pick the fastest accelerator.” It is a continuous mixed-criticality DAG scheduling problem with support-safe feasibility, transfer cost, queueing, memory pressure, and time-to-first-speech constraints.

```latex
\begin{figure*}[!t]
\centering
\includegraphics[width=0.97\textwidth]{figures/architecture_overview.png}
\caption{Checkpoint-pinned GraphPilot deployment surface. Workflows A/B/C are real device runs; FastVLM stays on the LiteRT NPU path; text and speech remain on verified CPU paths.}
\label{fig:architecture_overview}
\end{figure*}
```

The deployed workflows already running on the phone are **(A)** ASR -> Planner -> Responder -> TTS, **(B)** ASR -> Planner -> VLM -> Responder -> TTS, and **(C)** ASR -> Planner -> (VLM || Retrieval) -> Responder -> TTS. Hidden fallback is never counted as successful offload. Unsupported accelerator paths either remain simulator-only what-if points or fail with explicit evidence.

GraphPilot makes four concrete contributions:

1. a support-safe runtime that keeps FastVLM on the working NPU path and exposes streaming plus memory/KV control,
2. a calibrated multi-resource simulator that stays anchored to the same checkpointed device evidence,
3. a workload universe broad enough to test ranking, fallback sensitivity, and continuous-stream behavior,
4. a fair comparison surface against internal baselines and faithful method-class proxies.

The resulting claim is intentionally narrower than “everything runs everywhere.” GraphPilot is a measured mobile runtime plus a calibrated simulator for support-safe continuous assistant DAG scheduling on one SM8750-class device.
