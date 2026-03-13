# Related Work

```latex
\FloatBarrier

GraphPilot sits between several nearby systems threads. Classical heterogeneous schedulers such as HEFT and CPOP motivate the rank-based list-scheduling view, but they assume static execution costs and operation-complete resources. Mobile multi-DNN schedulers such as Band~\cite{band2022}, ADMS-style heterogeneous co-execution~\cite{adms2025}, and Puzzle~\cite{puzzle2025} target heterogeneous mobile processors but not continuous assistant DAGs with explicit support-safe fallback accounting. Compound-AI schedulers such as Twill~\cite{twill2025} and agentic SoC systems such as Agent.xpu~\cite{agentxpu2025} and HeRo~\cite{hero2026} move closer to the target setting, while HeteroInfer~\cite{heteroinfer2025} focuses on single-LLM heterogeneity and the prefill/decode split. Orca~\cite{orca2022} and PagedAttention~\cite{pagedattention2023} motivate the queueing and memory/KV perspective from serving.

GraphPilot's position is narrower and more explicit: support-safe fallback-aware scheduling of continuous multimodal assistant DAGs with checkpoint-pinned sim-to-real calibration on one heterogeneous mobile SoC.
```

```latex
{{COMPARISON_MATRIX_TABLE_TEX}}
```

The paper also makes the baseline surface explicit. The catalog below records the internal baselines, ablation policies, and faithful method-class proxies used in the shared simulator/runtime environment.

```latex
{{BASELINE_CATALOG_TABLE_TEX}}
```
