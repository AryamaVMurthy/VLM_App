# Sensitivity, Memory, and Knob Control

Memory pressure, KV growth, and stage-level knob tradeoffs are first-class systems concerns. The checkpointed artifact records both the calibrated curves that drive the simulator and the runtime-side admission/degrade/reject events that the Android path actually logs.

```latex
\begin{figure*}[!t]
\centering
\includegraphics[width=0.97\textwidth]{figures/sensitivity_overview.png}
\caption{Sensitivity overview combining fallback penalty, pipeline ablation, thermal plan-bank behavior, and objective sensitivity.}
\label{fig:sensitivity_overview}
\end{figure*}
```

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.96\columnwidth]{figures/memory_kv_overview.png}
\caption{Checkpoint-pinned KV-cache and context-memory growth curves used by the admission controller.}
\label{fig:memory_kv_overview}
\end{figure}
```

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.96\columnwidth]{figures/knob_frontier_overview.png}
\caption{Stage-knob frontiers showing latency-vs-quality proxy tradeoffs used for tuning.}
\label{fig:knob_frontier_overview}
\end{figure}
```

```latex
{{MEMORY_RUNTIME_TABLE_TEX}}
```

```latex
{{TUNING_TABLE_TEX}}
```

```latex
\FloatBarrier
```
